#!/usr/bin/env python
"""Install the official ESO ERIS/NIX science-filter curves for SED_Tools."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
import zipfile


ARCHIVE_URL = (
    "https://www.eso.org/sci/facilities/paranal/"
    "instruments/eris/img/nix-filters.zip"
)

# Output band name -> member name in ESO's official archive.
FILTERS = {
    "J": "nix-filters/filter_J.dat",
    "H": "nix-filters/filter_H.dat",
    "Ks": "nix-filters/filter_K.dat",
    "Short-Lp": "nix-filters/filter_Short-Lp.dat",
    "L-Broad": "nix-filters/filter_L-Broad.dat",
    "Lp": "nix-filters/filter_Lp.dat",
    "Mp": "nix-filters/filter_Mp.dat",
    "Pa-b": "nix-filters/Pa-b.dat",
    "Fe-II": "nix-filters/Fe-II.dat",
    "H2-cont": "nix-filters/H2-cont.dat",
    # ESO's archive uses the letter O in this filename. The instrument page
    # identifies the physical transition as H2 1-0 S, so use zero in output.
    "H2-1-0S": "nix-filters/H2-1-OS.dat",
    "Br-g": "nix-filters/Br-g.dat",
    "K-peak": "nix-filters/K-peak.dat",
    "IB-2.42": "nix-filters/IB-2.42.dat",
    "IB-2.48": "nix-filters/IB-2.48.dat",
    "Br-a-cont": "nix-filters/Br-a-cont.dat",
    "Br-a": "nix-filters/Br-a.dat",
}


def download_archive() -> bytes:
    request = Request(
        ARCHIVE_URL,
        headers={
            "User-Agent": (
                "SED_Tools filter preparation; scientific data retrieval"
            )
        },
    )

    with urlopen(request, timeout=120) as response:
        payload = response.read()

    if len(payload) < 1_000 or not payload.startswith(b"PK"):
        raise RuntimeError("ESO did not return the expected ZIP archive")

    return payload


def parse_curve(text: str, label: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.replace(",", " ").split()

        if len(fields) < 2:
            continue

        try:
            wavelength_m = float(fields[0])
            transmission = float(fields[1])
        except ValueError:
            continue

        if not (
            math.isfinite(wavelength_m)
            and math.isfinite(transmission)
        ):
            raise ValueError(
                f"{label}: non-finite value on line {line_number}"
            )

        if wavelength_m <= 0:
            raise ValueError(
                f"{label}: non-positive wavelength on line {line_number}"
            )

        if not 0.0 <= transmission <= 1.0:
            raise ValueError(
                f"{label}: transmission outside [0, 1] "
                f"on line {line_number}"
            )

        # ESO supplies wavelength in metres and transmission as a fraction.
        points.append((wavelength_m * 1.0e10, transmission))

    if len(points) < 2:
        raise ValueError(f"{label}: fewer than two usable samples")

    # Several official narrow-band files are stored in descending order.
    points.sort(key=lambda point: point[0])

    if any(
        second[0] <= first[0]
        for first, second in zip(points, points[1:])
    ):
        raise ValueError(
            f"{label}: wavelengths are duplicated or not strictly increasing"
        )

    if max(transmission for _, transmission in points) <= 0.0:
        raise ValueError(f"{label}: curve has no positive transmission")

    return points


def write_curve(path: Path, points: list[tuple[float, float]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("Wavelength,Transmission\n")

        for wavelength, transmission in points:
            handle.write(f"{wavelength:.8f},{transmission:.12g}\n")


def validate_written_curve(path: Path) -> tuple[int, float, float, float]:
    points: list[tuple[float, float]] = []

    with path.open("r", encoding="ascii") as handle:
        header = handle.readline().strip()

        if header != "Wavelength,Transmission":
            raise ValueError(f"{path.name}: invalid output header")

        for line_number, line in enumerate(handle, start=2):
            fields = line.strip().split(",")

            if len(fields) != 2:
                raise ValueError(
                    f"{path.name}: invalid output line {line_number}"
                )

            points.append((float(fields[0]), float(fields[1])))

    if len(points) < 2:
        raise ValueError(f"{path.name}: output has too few samples")

    if any(
        second[0] <= first[0]
        for first, second in zip(points, points[1:])
    ):
        raise ValueError(
            f"{path.name}: output wavelengths are not increasing"
        )

    if any(not 0.0 <= transmission <= 1.0 for _, transmission in points):
        raise ValueError(f"{path.name}: invalid output transmission")

    return (
        len(points),
        points[0][0],
        points[-1][0],
        max(transmission for _, transmission in points),
    )


def install(filters_root: Path) -> None:
    destination = filters_root / "ERIS" / "NIX"

    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists; inspect or move it before rerunning"
        )

    payload = download_archive()
    archive_sha256 = hashlib.sha256(payload).hexdigest()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        available = set(archive.namelist())
        missing = sorted(set(FILTERS.values()) - available)

        if missing:
            raise RuntimeError(
                "ESO archive is missing expected files: "
                + ", ".join(missing)
            )

        destination.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=".NIX-staging-",
            dir=destination.parent,
        ) as temporary_directory:
            staging = Path(temporary_directory) / "NIX"
            staging.mkdir()

            for output_name, archive_member in FILTERS.items():
                source_text = archive.read(archive_member).decode("ascii")
                points = parse_curve(source_text, output_name)
                write_curve(staging / f"{output_name}.dat", points)

            index_files = sorted(path.name for path in staging.glob("*.dat"))

            if len(index_files) != len(FILTERS):
                raise RuntimeError(
                    f"Expected {len(FILTERS)} output curves, "
                    f"found {len(index_files)}"
                )

            (staging / "NIX").write_text(
                "\n".join(index_files) + "\n",
                encoding="ascii",
            )

            for output_name in FILTERS:
                count, first, last, peak = validate_written_curve(
                    staging / f"{output_name}.dat"
                )
                print(
                    f"{output_name:10s} {count:5d} rows  "
                    f"{first:10.1f}-{last:10.1f} A  peak={peak:.6f}"
                )

            index_entries = (staging / "NIX").read_text(
                encoding="ascii"
            ).splitlines()

            if index_entries != index_files:
                raise RuntimeError("Generated NIX instrument index is invalid")

            shutil.move(str(staging), str(destination))

    print(
        f"\nInstalled {len(FILTERS)} ERIS/NIX curves in {destination}"
    )
    print(f"Instrument index: {destination / 'NIX'}")
    print(f"Source archive SHA-256: {archive_sha256}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Install ESO's official ERIS/NIX science-filter curves "
            "for SED_Tools."
        )
    )
    parser.add_argument(
        "filters_root",
        type=Path,
        help="SED_Tools filter root, normally data/filters",
    )
    args = parser.parse_args()

    install(args.filters_root.resolve())


if __name__ == "__main__":
    main()
