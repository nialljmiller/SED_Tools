"""Audit published filter curves before using them for synthetic photometry.

Usage: python scripts/audit_filter_curves.py data/filters --output issues.csv
Only root-level published curves are inspected; component and product curves
have their own provenance and can be audited separately when needed.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def inspect_curve(path: Path) -> dict[str, object]:
    previous = None
    rows = duplicates = negative = nonfinite = malformed = 0
    minimum = math.inf
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("wavelength"):
            continue
        parts = line.replace(",", " ").split()
        try:
            wavelength, transmission = map(float, parts[:2])
        except (ValueError, TypeError):
            malformed += 1
            continue
        rows += 1
        if not math.isfinite(wavelength) or not math.isfinite(transmission):
            nonfinite += 1
            continue
        if wavelength <= 0 or (previous is not None and wavelength <= previous):
            duplicates += 1
        if transmission < 0:
            negative += 1
        minimum = min(minimum, transmission)
        previous = wavelength
    return {
        "rows": rows,
        "nonincreasing_wavelengths": duplicates,
        "negative_transmissions": negative,
        "nonfinite_rows": nonfinite,
        "malformed_rows": malformed,
        "minimum_transmission": minimum if rows and math.isfinite(minimum) else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="root containing Facility/Instrument/*.dat")
    parser.add_argument("--output", type=Path, required=True, help="CSV of curves requiring review")
    args = parser.parse_args()

    paths = sorted(args.root.glob("*/*/*.dat"))
    issues = []
    for path in paths:
        result = inspect_curve(path)
        if result["rows"] < 2 or any(result[key] for key in (
            "nonincreasing_wavelengths", "negative_transmissions", "nonfinite_rows", "malformed_rows"
        )):
            issues.append({"path": str(path.relative_to(args.root)), **result})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "rows", "nonincreasing_wavelengths",
                                                        "negative_transmissions", "nonfinite_rows",
                                                        "malformed_rows", "minimum_transmission"),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(issues)
    print(f"Audited {len(paths)} published curves; {len(issues)} require review: {args.output}")


if __name__ == "__main__":
    main()
