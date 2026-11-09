"""Editing the detector type of installed filter curves in place.

*Correct* changes only the label (``DetectorType``): the curve's numbers were
always in the other convention.  *Convert* changes the label and the numbers:
T_E(λ) = T_P(λ)·λ to energy counting, T_P(λ) = T_E(λ)/λ to photon counting,
rescaled to the same peak.  MESA divides an energy-counting curve by λ and
restores its peak, so a converted curve gives the same photometry as before.

Each edited curve gets ``DetectorTypeSource=user`` and keeps the type it had
before its first edit in ``DetectorTypeOriginal``; a converted curve records
the net factor in ``DetectorTypeConversion`` (none once converted back).  The
records beside the curves follow: a correction is a statement about the
detector, so the manifest takes it; a conversion is not, so the manifest
keeps the detector's type, and composition undoes a conversion before
multiplying a curve, so products built later are as they were.  NJM's
``info.json`` and ``properties.json`` are updated for the edited bands
(Vega- and Sun-based properties become unknown).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .curve_metadata import (COLUMN_ROW, ENERGY, PHOTON, PathLike, _read_curve, conversion_label, conversion_power,
                             format_metadata_field, normalize_detector_type, read_filter_header, write_filter_header)
from .manifest import MANIFEST_NAME, archived_bands, index_listing

#: The conversion to each convention, as the prompts show it.
FORMULAS = {ENERGY: "T_E(λ) = T_P(λ)·λ", PHOTON: "T_P(λ) = T_E(λ)/λ"}


def times_lambda(wavelength, transmission, power: int) -> np.ndarray:
    """``transmission`` times λ**power, rescaled to the same peak (so the wavelength unit does not matter)."""
    transmission = np.asarray(transmission, dtype=float)
    scaled = transmission * np.asarray(wavelength, dtype=float) ** power
    peak = scaled.max()
    return scaled * (transmission.max() / peak) if peak > 0 else scaled


def converted_transmission(wavelength, transmission, to: str) -> np.ndarray:
    """``transmission`` in the ``to`` convention (times λ to energy, over λ to photon), with the same peak."""
    return times_lambda(wavelength, transmission, 1 if to == ENERGY else -1)


def unconverted_transmission(wavelength, transmission, header: Mapping[str, str]) -> np.ndarray:
    """A curve's numbers as they were before any conversion recorded in its header."""
    power = conversion_power(header.get("DetectorTypeConversion"))
    return times_lambda(wavelength, transmission, -power) if power else np.asarray(transmission, dtype=float)


def curve_files(path: PathLike, bands: Optional[Sequence[str]] = None) -> List[Path]:
    """The curve at ``path``, or the curves of the set at ``path`` (its MESA index, less archived bands)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"No filter curve or set at {path}")
    listed = index_listing(path)
    names = [Path(name).stem for name in listed] if listed else [p.stem for p in sorted(path.glob("*.dat"))]
    names = [name for name in names if name not in archived_bands(path)]
    unknown = sorted(set(bands or ()) - set(names))
    if unknown:
        raise ValueError(f"{path} has no band {', '.join(unknown)}")
    return [path / f"{name}.dat" for name in names if not bands or name in bands]


def edit_detector_type(path: PathLike, detector_type: str, *, convert: bool,
                       bands: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """Set the detector type of the curve(s) at ``path``, converting their numbers if ``convert``.

    Curves already of ``detector_type`` are left alone.  Returns one row per
    curve: ``path``, ``band``, ``detector_type``, ``previous`` and ``changed``.
    """
    target = normalize_detector_type(detector_type)
    if target is None:
        raise ValueError(f"Unknown detector type: {detector_type!r}; choose energy or photon")
    headers = {curve: read_filter_header(curve) for curve in curve_files(path, bands)}
    if not headers:
        raise FileNotFoundError(f"No filter curves in {path}")
    untagged = [p.name for p, header in headers.items() if not normalize_detector_type(header.get("DetectorType"))]
    if convert and untagged:
        raise ValueError(f"No DetectorType to convert from in {', '.join(untagged)}; "
                         "tag first (sed-tools filters-detector)")
    rows, edited = [], {}
    for curve, header in headers.items():
        previous = normalize_detector_type(header.get("DetectorType"))
        if previous != target:
            fields = dict(header, DetectorType=target, DetectorTypeSource="user")
            fields.setdefault("DetectorTypeOriginal", previous)
            if convert:
                power = conversion_power(header.get("DetectorTypeConversion")) + (1 if target == ENERGY else -1)
                fields["DetectorTypeConversion"] = conversion_label(power)
                _write_converted(curve, fields, target)
            else:
                write_filter_header(curve, fields)
            edited.setdefault(curve.parent, {})[curve.stem] = {k: v for k, v in fields.items() if v}
        rows.append({"path": str(curve), "band": curve.stem, "detector_type": target, "previous": previous,
                     "changed": previous != target})
    for directory, bands_edited in edited.items():
        _update_set_records(directory, bands_edited, corrected=not convert)
    return rows


def _write_converted(path: Path, fields: Mapping[str, Optional[str]], target: str) -> None:
    wavelength, transmission = _read_curve(path)
    converted = converted_transmission(wavelength, transmission, target)
    meta = format_metadata_field(fields)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(COLUMN_ROW + ("," + meta if meta else "") + "\n")
        fh.writelines(f"{float(w)!r},{float(t)!r}\n" for w, t in zip(wavelength, converted))
    os.replace(tmp, path)


def _update_set_records(directory: Path, headers: Mapping[str, Mapping[str, str]], *, corrected: bool) -> None:
    """Bring the manifest (corrections only), ``info.json`` and ``properties.json`` beside the curves into step."""
    from .properties import curve_properties

    def edit(name, change):
        path = directory / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            change(data)
            path.write_text(json.dumps(data, indent=1 if name != MANIFEST_NAME else 2, sort_keys=True) + "\n",
                            encoding="utf-8")

    keys = ("DetectorType", "DetectorTypeSource", "DetectorTypeOriginal", "DetectorTypeConversion")

    def manifest(data):
        for band, header in headers.items():
            data.setdefault("detector_types", {})[band] = header["DetectorType"]
            data.setdefault("detector_type_sources", {})[band] = "user"

    def info(data):
        for band, header in headers.items():
            curve = ((data.get("bands") or {}).get(band) or {}).get("curve")
            if curve is not None:
                curve.update({key: header.get(key) for key in keys})
                for key in [key for key in keys if not curve[key]]:
                    del curve[key]

    def properties(data):
        for band, header in headers.items():
            record = (data.get("bands") or {}).get(band)
            if record is not None:
                record.update(curve_properties(directory / f"{band}.dat"))
                record.update({key: header[key] for key in keys[:2]})

    if corrected:
        edit(MANIFEST_NAME, manifest)
    edit("info.json", info)
    edit("properties.json", properties)
