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

:func:`build` writes ESA's T_TOTAL as the published Y/J/H curves where none
exist (negative noise set to 0), compares the product of the components with
the published curves, and records any release mismatch in the manifest
notes.  Existing published curves and the MESA index are never altered.
On NJM, SVO's earlier copies are kept as ``Y_svo``, ``J_svo`` and ``H_svo``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

from ..filters.compose import FILTER, OPTICS, QE, InstrumentManifest, add_component
from ..filters.curve_metadata import PHOTON, write_filter_file
from ._original import ORIGINAL, Fetcher

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


def write_published(inst_dir: Path, band: str, cols: Dict[str, np.ndarray]) -> int:
    """Write ESA's T_TOTAL as ``<band>.dat`` unless it exists; return the negatives set to 0 (-1 if kept)."""
    path = Path(inst_dir) / f"{band}.dat"
    if path.exists():
        return -1
    total = cols["T_TOTAL"]
    write_filter_file(
        path, cols["WAVE"] * 10.0, np.maximum(total, 0.0),
        # T_TOTAL includes the detector QE (T_QE): a photon detection probability.
        detector_type=PHOTON, source=ORIGINAL, filter_id=f"{FACILITY}/{INSTRUMENT}.{band}",
        wavelength_unit="Angstrom",
        extra={"Components": "filter+optics+qe", "ComponentsSource": ORIGINAL,
               "Sources": f"{URL.format(band=band)} | {PAGE}"},
    )
    return int(np.sum(total < 0))


def _published_deviation(inst_dir: Path, band: str, wl: np.ndarray, total: np.ndarray) -> Optional[float]:
    """max |composed - published| / published peak on the published grid (None if absent)."""
    path = inst_dir / f"{band}.dat"
    if not path.is_file():
        return None
    pw, pt = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1), unpack=True)
    mine = np.interp(pw, wl, total, left=0.0, right=0.0)
    return float(np.max(np.abs(mine - pt)) / np.max(pt))


def _fresh_manifest(inst_dir: Path, bands) -> InstrumentManifest:
    """The manifest with its published curves described again and ``bands``' factors dropped."""
    manifest = InstrumentManifest.load(inst_dir)
    # The published curves may have changed since the manifest was written (e.g. SVO's copies
    # renamed to <band>_svo): describe them from their headers.
    known = InstrumentManifest.from_published(inst_dir)
    manifest.bands, manifest.published, manifest.published_bands = known.bands, known.published, known.published_bands
    manifest.detector_types, manifest.detector_type_sources = known.detector_types, known.detector_type_sources
    for role in (FILTER, OPTICS, QE):
        for b in bands:
            manifest.components.get(role, {}).pop(b, None)
    return manifest


def _add_factors(inst_dir: Path, manifest: InstrumentManifest, band: str,
                 cols: Dict[str, np.ndarray]) -> Tuple[int, Optional[float]]:
    """Write one band's filter, optics and QE; return (negative filter samples set to 0, deviation)."""
    wl, curves, negative = factors(cols)
    url = URL.format(band=band)
    described = {FILTER: "T_FILTER", OPTICS: "T_TELESCOPE x T_NI-OA", QE: "T_QE"}
    for role, tr in curves.items():
        add_component(inst_dir, role, wl, tr, band=band, name=band,
                      source=f"{url} ({described[role]}; ESA NISP-PHOTO-PASSBANDS V1.0, {DOI})",
                      manifest=manifest)
    deviation = _published_deviation(inst_dir, band, wl, curves[FILTER] * curves[OPTICS] * curves[QE])
    return negative, deviation


def _notes(bands, written: Dict[str, int], clamped: Dict[str, int], deviations: Dict[str, float],
           mismatched: Dict[str, float]) -> str:
    """The manifest notes: the source, what was written or clamped, and how the product compares."""
    notes = [
        "Y, J and H components from ESA NISP-PHOTO-PASSBANDS V1.0 (2021-11-01; on-ground factors at the "
        "centre of the NISP focal plane): filter = T_FILTER, optics = T_TELESCOPE (all mirrors and the "
        "dichroic) x T_NI-OA (NISP optical assembly), qe = T_QE. Their product equals ESA's T_TOTAL. "
        "WAVE is nm (the files state no unit; inferred from the values)."
    ]
    if written:
        notes.append("Published " + ", ".join(written) + " written from ESA's T_TOTAL (photon-counting: it "
                     "includes the detector QE)" + (": negative samples set to 0: "
                     + ", ".join(f"{b} {n}" for b, n in written.items() if n) if any(written.values()) else "")
                     + ".")
    if clamped:
        notes.append("Negative T_FILTER noise samples set to 0: "
                     + ", ".join(f"{b} {n}" for b, n in clamped.items()) + ".")
    if mismatched:
        notes.append(
            "Release mismatch: filter x optics x qe is ESA's V1.0 T_TOTAL, not the published curve of the "
            "same band here; max |difference| / published peak = "
            + ", ".join(f"{b} {d:.3f}" for b, d in mismatched.items())
            + ". Compose these components for the V1.0 passbands; use the published curves as they are.")
    elif deviations:
        notes.append("filter x optics x qe reproduces the published curves to "
                     + ", ".join(f"{b} {d:.1e}" for b, d in deviations.items()) + " of their peak.")
    copies = [b for b in bands if b.endswith("_svo") and b[:-4] in BANDS]
    if copies:
        notes.append(", ".join(copies) + " are SVO's copies of an earlier release, kept for reference; "
                     "these components do not compose to them.")
    others = [b for b in bands if b not in BANDS and b not in copies]
    if others:
        notes.append("No separate factors are published for " + ", ".join(others) + ".")
    return " ".join(notes)


def build(
    filter_root: Union[str, Path],
    *,
    bands: Optional[Iterable[str]] = None,
    session=None,
    **_ignored,
) -> Path:
    """Add ESA's NISP filter, optics and QE factors to ``Euclid/NISP``."""
    inst_dir = Path(filter_root) / FACILITY / INSTRUMENT
    requested = list(bands or BANDS)
    wanted = [b for b in requested if b in BANDS]
    if not wanted:
        raise ValueError(f"No NISP photometric band among {list(bands or [])}; ESA publishes {BANDS}")
    skipped = [b for b in requested if b not in BANDS]
    if skipped:
        logger.warning("Euclid/NISP: ESA publishes factors only for %s; %s skipped", BANDS, skipped)
    fetch = Fetcher(session)
    # Download and check every band before writing anything.
    tables = {b: parse_table(fetch.text(URL.format(band=b))) for b in wanted}

    inst_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for band in wanted:
        negatives = write_published(inst_dir, band, tables[band])
        if negatives >= 0:  # -1: the published curve existed and was kept
            written[band] = negatives
    index = inst_dir / INSTRUMENT
    if not index.exists():
        index.write_text("".join(f"{b}.dat\n" for b in BANDS if (inst_dir / f"{b}.dat").is_file()),
                         encoding="utf-8")
    manifest = _fresh_manifest(inst_dir, wanted)
    clamped, deviations = {}, {}
    for band in wanted:
        negative, deviation = _add_factors(inst_dir, manifest, band, tables[band])
        if negative:
            clamped[band] = negative
        if deviation is not None:
            deviations[band] = deviation

    manifest.facility, manifest.instrument = FACILITY, INSTRUMENT
    manifest.detector_type = PHOTON
    manifest.ground_based = False
    manifest.outside_range = "zero"  # the factors are tabulated only across each band
    manifest.sources["esa_nisp_passbands"] = f"{PAGE} ({DOI}; Schirmer et al. 2022, {PAPER})"
    mismatched = {b: d for b, d in deviations.items() if d > MATCH_TOLERANCE}
    # Archived copies (H_svo, ...) are no bands, but the notes still say what they are.
    manifest.notes = _notes([*manifest.bands, *manifest.archived_bands], written, clamped, deviations, mismatched)
    manifest.save(inst_dir)
    if mismatched:
        logger.warning("Euclid/NISP: components do not reproduce the published curves: %s", mismatched)
    return inst_dir
