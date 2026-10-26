"""Euclid NISP components from ESA's photometric passband reference data.

ESA's NISP-PHOTO-PASSBANDS V1.0 (2021-11-01; DOI 10.5270/esa-kx8w57c;
Schirmer et al. 2022, A&A 662, A92) gives the Y, J and H passbands at the
centre of the NISP focal plane as a total throughput and its four factors,
determined on ground before launch::

    # WAVE T_TOTAL T_FILTER T_TELESCOPE T_NI-OA T_QE

``T_TELESCOPE`` combines all mirrors and the dichroic, ``T_NI-OA`` is the
NISP optical assembly, and ``T_TOTAL`` is the product of the four (to 5e-7
in every band).  The files state no units.  ``WAVE`` is in nm: Y spans
905-1245, which is only a near-infrared band in nm, and x10 puts it on the
published NISP curves (in Angstrom).  The factors are fractions (peaks
0.88-1.00).

Components are written per band: ``filter`` = T_FILTER, ``optics`` =
T_TELESCOPE x T_NI-OA, ``qe`` = T_QE.  T_FILTER has small negative noise
samples in J and H; they are set to 0.  J's filter peaks at 1.003 and is
kept as published.

:func:`build` compares the product with the instrument's published Y/J/H
curves and records any release mismatch in the manifest notes; the
published curves are never altered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

from ..filter_components import FILTER, OPTICS, QE, InstrumentManifest, add_component
from ..filter_metadata import PHOTON
from ._original import Fetcher

logger = logging.getLogger(__name__)

FACILITY, INSTRUMENT = "Euclid", "NISP"
BANDS = ("Y", "J", "H")
PAGE = "https://euclid.esac.esa.int/msp/refdata/nisp/NISP-PHOTO-PASSBANDS-V1"
URL = "https://euclid.esac.esa.int/msp/refdata/data/NISP-PHOTO-PASSBANDS-V1-{band}_throughput.dat"
DOI = "https://doi.org/10.5270/esa-kx8w57c"
PAPER = "https://doi.org/10.1051/0004-6361/202142897"
COLUMNS = ("WAVE", "T_TOTAL", "T_FILTER", "T_TELESCOPE", "T_NI-OA", "T_QE")
#: Largest |product of the factors - T_TOTAL| accepted when parsing.
PRODUCT_TOLERANCE = 5e-6
#: Larger deviations from the published curve (relative to its peak) are a release mismatch.
MATCH_TOLERANCE = 1e-3


def parse_table(text: str) -> Dict[str, np.ndarray]:
    """Columns of one ESA passband file, by the names in its ``#`` header row."""
    names = None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            names = line.lstrip("#").split()
            continue
        rows.append([float(x) for x in line.split()])
    if names is None or tuple(names) != COLUMNS:
        raise ValueError(f"unexpected NISP passband columns {names}; expected {COLUMNS}")
    data = np.array(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(COLUMNS) or len(data) < 2:
        raise ValueError("NISP passband table has no data rows")
    cols = {name: data[:, i] for i, name in enumerate(COLUMNS)}
    if np.any(np.diff(cols["WAVE"]) <= 0):
        raise ValueError("NISP passband wavelengths are not increasing")
    product = cols["T_FILTER"] * cols["T_TELESCOPE"] * cols["T_NI-OA"] * cols["T_QE"]
    worst = float(np.max(np.abs(product - cols["T_TOTAL"])))
    if worst > PRODUCT_TOLERANCE:
        raise ValueError(f"NISP factors do not multiply to T_TOTAL (max difference {worst:.3g})")
    return cols


def factors(cols: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """(Angstrom, {role: curve}, number of negative filter samples set to 0)."""
    wl = cols["WAVE"] * 10.0  # nm -> Angstrom
    negative = int(np.sum(cols["T_FILTER"] < 0))
    return wl, {
        FILTER: np.maximum(cols["T_FILTER"], 0.0),
        OPTICS: cols["T_TELESCOPE"] * cols["T_NI-OA"],
        QE: cols["T_QE"],
    }, negative


def _published_deviation(inst_dir: Path, band: str, wl: np.ndarray, total: np.ndarray) -> Optional[float]:
    """max |composed - published| / published peak on the published grid (None if absent)."""
    path = inst_dir / f"{band}.dat"
    if not path.is_file():
        return None
    pw, pt = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1), unpack=True)
    mine = np.interp(pw, wl, total, left=0.0, right=0.0)
    return float(np.max(np.abs(mine - pt)) / np.max(pt))


def build(
    filter_root: Union[str, Path],
    *,
    bands: Optional[Iterable[str]] = None,
    session=None,
    **_ignored,
) -> Path:
    """Add ESA's NISP filter, optics and QE factors to ``Euclid/NISP``."""
    inst_dir = Path(filter_root) / FACILITY / INSTRUMENT
    present = {p.stem for p in inst_dir.glob("*.dat")}
    wanted = [b for b in (bands or BANDS) if b in BANDS and b in present]
    if not wanted:
        raise FileNotFoundError(f"{inst_dir}: no published Y/J/H curves; fetch them first")
    fetch = Fetcher(session)
    # Download and check every band before writing anything.
    tables = {b: parse_table(fetch.text(URL.format(band=b))) for b in wanted}

    manifest = InstrumentManifest.load(inst_dir)
    for role in (FILTER, OPTICS, QE):
        for b in wanted:
            manifest.components.get(role, {}).pop(b, None)
    clamped, deviations = {}, {}
    for band in wanted:
        wl, curves, negative = factors(tables[band])
        if negative:
            clamped[band] = negative
        url = URL.format(band=band)
        described = {FILTER: "T_FILTER", OPTICS: "T_TELESCOPE x T_NI-OA", QE: "T_QE"}
        for role, tr in curves.items():
            add_component(inst_dir, role, wl, tr, band=band, name=band,
                          source=f"{url} ({described[role]}; ESA NISP-PHOTO-PASSBANDS V1.0, {DOI})",
                          manifest=manifest)
        deviation = _published_deviation(inst_dir, band, wl, curves[FILTER] * curves[OPTICS] * curves[QE])
        if deviation is not None:
            deviations[band] = deviation

    manifest.facility, manifest.instrument = FACILITY, INSTRUMENT
    manifest.detector_type = PHOTON
    manifest.ground_based = False
    manifest.outside_range = "zero"  # the factors are tabulated only across each band
    manifest.sources["esa_nisp_passbands"] = f"{PAGE} ({DOI}; Schirmer et al. 2022, {PAPER})"
    notes = [
        "Y, J and H components from ESA NISP-PHOTO-PASSBANDS V1.0 (2021-11-01; on-ground factors at the "
        "centre of the NISP focal plane): filter = T_FILTER, optics = T_TELESCOPE (all mirrors and the "
        "dichroic) x T_NI-OA (NISP optical assembly), qe = T_QE. Their product equals ESA's T_TOTAL. "
        "WAVE is nm (the files state no unit; inferred from the values)."
    ]
    if clamped:
        notes.append("Negative T_FILTER noise samples set to 0: "
                     + ", ".join(f"{b} {n}" for b, n in clamped.items()) + ".")
    mismatched = {b: d for b, d in deviations.items() if d > MATCH_TOLERANCE}
    if mismatched:
        notes.append(
            "Release mismatch: filter x optics x qe is ESA's V1.0 T_TOTAL, not the published curve of the "
            "same band here; max |difference| / published peak = "
            + ", ".join(f"{b} {d:.3f}" for b, d in mismatched.items())
            + ". Compose these components for the V1.0 passbands; use the published curves as they are.")
    elif deviations:
        notes.append("filter x optics x qe reproduces the published curves to "
                     + ", ".join(f"{b} {d:.1e}" for b, d in deviations.items()) + " of their peak.")
    others = [b for b in manifest.bands if b not in BANDS]
    if others:
        notes.append("No separate factors are published for " + ", ".join(others) + ".")
    manifest.notes = " ".join(notes)
    manifest.save(inst_dir)
    if mismatched:
        logger.warning("Euclid/NISP: components do not reproduce the published curves: %s", mismatched)
    return inst_dir
