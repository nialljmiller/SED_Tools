#!/usr/bin/env python
"""Download and prepare all published ESO HAWK-I filter curves."""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = (
    "https://www.eso.org/sci/facilities/paranal/instruments/"
    "hawki/inst/filters"
)

# Explicit URLs avoid the inconsistent footnote placement in the ESO HTML.
# The live NB0984 link currently returns an HTML error page with HTTP 200, so
# use the archived copy of the same ESO file from when it was still available.
FILTERS = {
    "Y": f"{BASE_URL}/hawki_Y.dat",
    "J": f"{BASE_URL}/hawki_J.dat",
    "H": f"{BASE_URL}/hawki_H.dat",
    "Ks_2007": f"{BASE_URL}/hawki_Kold.dat",
    "Ks_2008": f"{BASE_URL}/hawki_Knew.dat",
    "CH4": f"{BASE_URL}/hawki_CH4.dat",
    "H2": f"{BASE_URL}/hawki_H2.dat",
    "BrG": f"{BASE_URL}/hawki_BrGamma.dat",
    "NB1060": f"{BASE_URL}/hawki_NB1060.dat",
    "NB1190": f"{BASE_URL}/hawki_NB1190.dat",
    "NB0984": (
        "https://web.archive.org/web/20130623151949id_/"
        "http://www.eso.org/sci/facilities/paranal/instruments/"
        "hawki/inst/filters/hawki_NB0984.dat"
    ),
    "NB2090": f"{BASE_URL}/hawki_NB2090.dat",
}


def download_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "SED_Tools/HAWK-I-filter-import "
                "(scientific data retrieval)"
            )
        },
    )

    try:
        with urlopen(request, timeout=120) as response:
            data = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"failed to download {url}: {exc}") from exc

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")

    leading_text = text.lstrip()[:256].lower()

    if "<!doctype html" in leading_text or "<html" in leading_text:
        raise RuntimeError(f"download returned HTML instead of data: {url}")

    return text


def parse_curve(text: str, name: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.replace(",", " ").split()

        if len(fields) < 2:
            continue

        try:
            wavelength_nm = float(fields[0])
            transmission_percent = float(fields[1])
        except ValueError:
            continue

        if not math.isfinite(wavelength_nm):
            raise ValueError(
                f"{name}: non-finite wavelength on line {line_number}"
            )

        if not math.isfinite(transmission_percent):
            raise ValueError(
                f"{name}: non-finite transmission on line {line_number}"
            )

        if wavelength_nm <= 0.0:
            raise ValueError(
                f"{name}: non-positive wavelength on line {line_number}"
            )

        # ESO supplies wavelength in nm and transmission in percent.
        wavelength_angstrom = wavelength_nm * 10.0

        # Negative laboratory measurements are background-subtraction noise.
        transmission = max(0.0, transmission_percent) / 100.0
        points.append((wavelength_angstrom, transmission))

    if len(points) < 2:
        raise ValueError(f"{name}: fewer than two usable data rows")

    for first, second in zip(points, points[1:]):
        if second[0] <= first[0]:
            raise ValueError(
                f"{name}: wavelengths are not strictly increasing"
            )

    peak = max(transmission for _, transmission in points)

    if not 0.0 < peak <= 1.0:
        raise ValueError(
            f"{name}: peak fractional transmission is invalid: {peak}"
        )

    return points


def write_curve(
    path: Path,
    points: list[tuple[float, float]],
) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("Wavelength,Transmission\n")

        for wavelength, transmission in points:
            handle.write(
                f"{wavelength:.8f},{transmission:.12g}\n"
            )


def validate_staged_collection(directory: Path) -> None:
    expected_files = {f"{name}.dat" for name in FILTERS}
    actual_files = {path.name for path in directory.glob("*.dat")}

    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise RuntimeError(
            "staged file-set mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )

    index_path = directory / "HAWK-I"
    index_entries = index_path.read_text(encoding="ascii").splitlines()

    if index_entries != sorted(expected_files):
        raise RuntimeError("staged HAWK-I index is incomplete or unsorted")

    for filename in index_entries:
        rows = (directory / filename).read_text(
            encoding="ascii"
        ).splitlines()

        if len(rows) < 3:
            raise RuntimeError(f"{filename}: output has too few rows")

        if rows[0] != "Wavelength,Transmission":
            raise RuntimeError(f"{filename}: invalid output header")


def install_collection(staged: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists; refusing to overwrite it"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staged, destination)


def main() -> int:
    if len(sys.argv) > 2:
        print(
            f"usage: {Path(sys.argv[0]).name} [FILTER_ROOT]",
            file=sys.stderr,
        )
        return 2

    filter_root = (
        Path(sys.argv[1])
        if len(sys.argv) == 2
        else Path("data/filters")
    )
    destination = filter_root / "HAWK-I" / "HAWK-I"

    with tempfile.TemporaryDirectory(
        prefix="sed_tools_hawki_"
    ) as temporary_directory:
        staged = Path(temporary_directory) / "HAWK-I"
        staged.mkdir()

        results: list[tuple[str, int, float, float, float]] = []

        with ThreadPoolExecutor(max_workers=6) as executor:
            downloaded = dict(
                zip(
                    FILTERS,
                    executor.map(download_text, FILTERS.values()),
                )
            )

        for output_name in FILTERS:
            points = parse_curve(downloaded[output_name], output_name)
            write_curve(staged / f"{output_name}.dat", points)

            results.append(
                (
                    output_name,
                    len(points),
                    points[0][0],
                    points[-1][0],
                    max(transmission for _, transmission in points),
                )
            )

        index_files = sorted(f"{name}.dat" for name in FILTERS)
        (staged / "HAWK-I").write_text(
            "\n".join(index_files) + "\n",
            encoding="ascii",
        )

        validate_staged_collection(staged)
        install_collection(staged, destination)

    for name, count, minimum, maximum, peak in results:
        print(
            f"{name:8s} {count:5d} rows  "
            f"{minimum:10.1f}-{maximum:10.1f} A  "
            f"peak={peak:.6f}"
        )

    print()
    print(f"Installed 12 HAWK-I curves in {destination}")
    print(f"Instrument index: {destination / 'HAWK-I'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
