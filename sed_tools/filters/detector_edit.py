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
``info.json`` and ``properties.json`` beside the curves take the new
detector fields.  A converted curve describes the same instrument, so its
other properties stay as they were; a corrected curve's properties are
recomputed as NJM computes them (:func:`.properties.properties_record`), with
the Vega and Sun spectra NJM serves (cached; their values are left unknown,
with a note, when NJM cannot be reached and nothing is cached).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .curve_metadata import (COLUMN_ROW, ENERGY, PHOTON, PathLike, _layout, _read_curve, _rewritten, conversion_label,
                             conversion_power, format_metadata_field, normalize_detector_type, read_filter_header,
                             write_filter_header)
from .manifest import MANIFEST_NAME, archived_bands, index_listing

#: Where NJM serves the reference spectra of ``properties.json``, when its ``definitions`` do not say.
SPECTRA = {"vega_spectrum": "vega/vega_flam.csv", "sun_spectrum": "sun/sun_flam.csv"}

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
                       spectrum: Callable[[str], PathLike],
                       bands: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """Set the detector type of the curve(s) at ``path``, converting their numbers if ``convert``.

    Curves already of ``detector_type`` are left alone.  ``spectrum`` gives
    the local copy of a reference spectrum NJM serves at a relative URL (see
    :meth:`.FilterCatalogue.spectrum`), for the properties of corrected
    curves.  Returns one row per curve: ``path``, ``band``,
    ``detector_type``, ``previous`` and ``changed``.
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
        _update_set_records(directory, bands_edited, corrected=not convert, spectrum=spectrum)
    return rows


def _write_converted(path: Path, fields: Mapping[str, Optional[str]], target: str) -> None:
    """Convert the samples of ``path`` to ``target``; the lines before them keep their text, ``fields`` on the
    column row."""
    wavelength, transmission = _read_curve(path)
    samples = [f"{float(w)!r},{float(t)!r}\n".encode()
               for w, t in zip(wavelength, converted_transmission(wavelength, transmission, target))]
    layout = _layout(path)
    if layout.column_row is None:  # metadata only on 0.3.0 "#" lines: the curve gets a column row
        head = [f"{COLUMN_ROW},{format_metadata_field(fields)}\n".encode()]
    else:
        head = _rewritten(layout, fields)[:layout.column_row + 1 - len(layout.legacy_rows)]
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        fh.writelines(head + samples)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


def _reference_spectra(definitions: Mapping, spectrum: Callable[[str], PathLike]) -> List:
    """The Vega and Sun spectra of ``properties.json`` (its ``definitions`` name them), each None when it cannot
    be had."""
    from .njm_catalogue import CatalogueError
    from .properties import load_vega

    spectra = []
    for key, default in SPECTRA.items():
        try:
            spectra.append(load_vega(spectrum((definitions.get(key) or {}).get("url") or default)))
        except CatalogueError as exc:
            print(f"[filters] {exc}: the {key.split('_')[0].capitalize()}-based properties of the corrected "
                  "curves are left unknown")
            spectra.append(None)
    return spectra


def _update_set_records(directory: Path, headers: Mapping[str, Mapping[str, str]], *, corrected: bool,
                        spectrum: Callable[[str], PathLike]) -> None:
    """Bring the manifest (corrections only), ``info.json`` and ``properties.json`` beside the curves into step."""
    from .properties import properties_record

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
        # A conversion keeps the instrument, so its properties; a correction recomputes them as NJM does.
        records = {band: record for band, record in (data.get("bands") or {}).items() if band in headers}
        spectra = _reference_spectra(data.get("definitions") or {}, spectrum) if corrected and records else ()
        for band, record in records.items():
            record.update(properties_record(directory / f"{band}.dat", *spectra) if corrected
                          else {key: headers[band][key] for key in keys[:2]})

    if corrected:
        edit(MANIFEST_NAME, manifest)
    edit("info.json", info)
    edit("properties.json", properties)
