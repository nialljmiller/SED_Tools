#!/usr/bin/env python3
"""Download and install the complete SVO HST passband collection.

The SVO Filter Profile Service groups HST response curves by detector/channel.
This importer preserves those groups as SED_Tools instruments beneath
``data/filters/HST``.  It stages and validates the complete collection before
replacing any existing HST directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
from pathlib import Path
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FPS_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php"
USER_AGENT = "SED_Tools HST filter preparation; scientific data retrieval"
EXPECTED_COUNTS = {
    "ACS_HRC": 26,
    "ACS_SBC": 8,
    "ACS_WFC": 31,
    "FGS": 6,
    "FOC_F48": 14,
    "FOC_F96": 44,
    "FOS_BLUE": 7,
    "FOS_RED": 9,
    "HSP_POL": 6,
    "HSP_UV1": 24,
    "HSP_UV2": 26,
    "HSP_VIS": 22,
    "NICMOS1": 19,
    "NICMOS2": 19,
    "NICMOS3": 19,
    "STIS_CCD": 27,
    "STIS_FUV": 30,
    "STIS_NUV": 52,
    "WFC3_IR": 17,
    "WFC3_UVIS1": 63,
    "WFC3_UVIS2": 63,
    "WFPC1-PC": 49,
    "WFPC1-WF": 49,
    "WFPC2-PC": 44,
    "WFPC2-WF": 52,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


@dataclass(frozen=True)
class Profile:
    filter_id: str
    instrument: str
    name: str


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def download_xml(params: dict[str, str], label: str) -> bytes:
    last_error: Exception | None = None
    url = f"{FPS_URL}?{urlencode(params)}"
    for attempt in range(1, 5):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=120) as response:
                payload = response.read()
            if not payload.lstrip().startswith(b"<?xml"):
                raise ValueError("response is not XML")
            return payload
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"{label}: download failed after four attempts") from last_error


def parse_table(payload: bytes, label: str) -> tuple[list[str], list[list[str]]]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise ValueError(f"{label}: invalid VOTable XML") from error

    for element in root.iter():
        if local_name(element.tag) == "INFO":
            if element.attrib.get("name") == "QUERY_STATUS":
                if element.attrib.get("value") not in {None, "OK"}:
                    detail = (element.text or "").strip()
                    raise RuntimeError(
                        f"{label}: SVO query failed: "
                        f"{element.attrib.get('value')} {detail}"
                    )

    tables = [element for element in root.iter() if local_name(element.tag) == "TABLE"]
    if len(tables) != 1:
        raise ValueError(f"{label}: expected one VOTable TABLE, found {len(tables)}")

    table = tables[0]
    fields = [
        element.attrib.get("name", "")
        for element in table
        if local_name(element.tag) == "FIELD"
    ]
    rows = []
    for row in table.iter():
        if local_name(row.tag) != "TR":
            continue
        values = [
            (cell.text or "").strip()
            for cell in row
            if local_name(cell.tag) == "TD"
        ]
        if len(values) != len(fields):
            raise ValueError(
                f"{label}: row has {len(values)} values for {len(fields)} fields"
            )
        rows.append(values)

    return fields, rows


def discover_profiles() -> list[Profile]:
    payload = download_xml({"Facility": "HST"}, "HST inventory")
    fields, rows = parse_table(payload, "HST inventory")
    try:
        filter_id_column = fields.index("filterID")
    except ValueError as error:
        raise ValueError("HST inventory: missing filterID field") from error

    profiles = []
    seen = set()
    for row in rows:
        filter_id = row[filter_id_column]
        match = re.fullmatch(r"HST/([^./]+)\.([^/]+)", filter_id)
        if match is None:
            raise ValueError(f"unexpected HST filter identifier: {filter_id!r}")
        instrument, name = match.groups()
        if not SAFE_NAME.fullmatch(instrument) or not SAFE_NAME.fullmatch(name):
            raise ValueError(f"unsafe path component in filter identifier: {filter_id!r}")
        if filter_id in seen:
            raise ValueError(f"duplicate filter identifier: {filter_id}")
        seen.add(filter_id)
        profiles.append(Profile(filter_id, instrument, name))

    actual_counts = Counter(profile.instrument for profile in profiles)
    if dict(actual_counts) != EXPECTED_COUNTS:
        expected = " ".join(f"{key}={value}" for key, value in EXPECTED_COUNTS.items())
        actual = " ".join(f"{key}={actual_counts[key]}" for key in sorted(actual_counts))
        raise RuntimeError(
            "SVO HST inventory has changed; refusing an unreviewed import.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )
    if len(profiles) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL} HST profiles, found {len(profiles)}"
        )

    return sorted(profiles, key=lambda item: (item.instrument, item.name))


def parse_curve(payload: bytes, profile: Profile) -> list[tuple[float, float]]:
    fields, rows = parse_table(payload, profile.filter_id)
    try:
        wavelength_column = fields.index("Wavelength")
        transmission_column = fields.index("Transmission")
    except ValueError as error:
        raise ValueError(f"{profile.filter_id}: missing curve fields") from error

    points = []
    for line_number, row in enumerate(rows, start=1):
        try:
            wavelength = float(row[wavelength_column])
            transmission = float(row[transmission_column])
        except ValueError as error:
            raise ValueError(
                f"{profile.filter_id}: non-numeric curve row {line_number}"
            ) from error
        if not math.isfinite(wavelength) or wavelength <= 0:
            raise ValueError(
                f"{profile.filter_id}: invalid wavelength {wavelength}"
            )
        if not math.isfinite(transmission):
            raise ValueError(
                f"{profile.filter_id}: invalid transmission {transmission}"
            )
        if -1e-12 <= transmission < 0:
            transmission = 0.0
        if not 0.0 <= transmission <= 1.0:
            raise ValueError(
                f"{profile.filter_id}: transmission outside [0, 1]: "
                f"{transmission}"
            )
        points.append((wavelength, transmission))

    if len(points) < 2:
        raise ValueError(f"{profile.filter_id}: fewer than two curve points")
    if any(second[0] <= first[0] for first, second in zip(points, points[1:])):
        raise ValueError(f"{profile.filter_id}: wavelengths are not strictly increasing")
    if max(transmission for _, transmission in points) <= 0:
        raise ValueError(f"{profile.filter_id}: curve has no positive transmission")
    return points


def write_curve(path: Path, points: list[tuple[float, float]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("Wavelength,Transmission\n")
        for wavelength, transmission in points:
            handle.write(f"{wavelength:.8f},{transmission:.12g}\n")


def fetch_and_write(profile: Profile, staging: Path) -> tuple[Profile, int, float, float, float]:
    payload = download_xml({"ID": profile.filter_id}, profile.filter_id)
    points = parse_curve(payload, profile)
    destination = staging / profile.instrument / f"{profile.name}.dat"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_curve(destination, points)
    transmissions = [point[1] for point in points]
    return profile, len(points), points[0][0], points[-1][0], max(transmissions)


def validate_written_collection(staging: Path, profiles: list[Profile]) -> None:
    expected_paths = {
        Path(profile.instrument) / f"{profile.name}.dat" for profile in profiles
    }
    actual_paths = {path.relative_to(staging) for path in staging.rglob("*.dat")}
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise RuntimeError(f"written collection mismatch; missing={missing}; extra={extra}")

    grouped: dict[str, list[str]] = defaultdict(list)
    for profile in profiles:
        grouped[profile.instrument].append(f"{profile.name}.dat")

    for instrument, filenames in grouped.items():
        index_path = staging / instrument / instrument
        index_path.write_text("\n".join(sorted(filenames)) + "\n", encoding="ascii")
        indexed = index_path.read_text(encoding="ascii").splitlines()
        if indexed != sorted(filenames):
            raise RuntimeError(f"{instrument}: instrument index validation failed")

    if len(actual_paths) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL} written curves, found {len(actual_paths)}"
        )


def install(staging: Path, destination: Path) -> None:
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        raise RuntimeError(f"refusing to overwrite existing backup: {backup}")

    if destination.exists():
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install every SVO HST response profile into SED_Tools."
    )
    parser.add_argument(
        "filters_root",
        type=Path,
        help="SED_Tools filter root, normally data/filters",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="simultaneous downloads (default: 8)",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")

    filters_root = args.filters_root.expanduser().resolve()
    filters_root.mkdir(parents=True, exist_ok=True)
    destination = filters_root / "HST"
    profiles = discover_profiles()
    print(
        f"SVO inventory verified: {len(EXPECTED_COUNTS)} instruments, "
        f"{len(profiles)} profiles"
    )

    with tempfile.TemporaryDirectory(prefix="hst_filters_", dir=filters_root) as temp:
        staging = Path(temp) / "HST"
        staging.mkdir()
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_and_write, profile, staging): profile
                for profile in profiles
            }
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % 25 == 0 or completed == len(profiles):
                    print(f"Downloaded and validated {completed:3d}/{len(profiles)}")

        validate_written_collection(staging, profiles)
        install(staging, destination)

    result_counts = Counter(result[0].instrument for result in results)
    print("\nInstalled HST collection:")
    for instrument in sorted(result_counts):
        print(f"  {instrument:15s} {result_counts[instrument]:3d} profiles")
    print(f"\nTotal: {sum(result_counts.values())} profiles")
    print(f"Destination: {destination}")


if __name__ == "__main__":
    main()
