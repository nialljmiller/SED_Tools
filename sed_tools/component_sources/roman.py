"""Roman WFI detector QE from the Roman technical information repository.

``FPSPerformance/WFI_Quantum_efficiency.ecsv`` gives the median quantum
efficiency of each of the 18 flight detectors (Sensor Chip Assemblies,
columns ``SCU 1`` ... ``SCU 18`` by focal-plane position), measured in the
NASA Goddard Detector Characterization Lab at 95 K, from 480 to 2700 nm in
10 nm steps (wavelength unit nm and dimensionless QE, as the ECSV header
declares).  Filter and optics transmissions are not published separately.

Representation: the shared ``qe`` component of ``Roman/WFI`` is the mean of
the 18 detectors at each wavelength.  The ``Roman/WFI`` curves describe the
whole focal plane, and the mean is the QE of the focal plane as a whole (a
mean of per-detector responses equals filter x optics x mean QE when filter
and optics are common); a median would not compose that way, and no single
detector is representative.  The team's own summary
(``WFI_Quantum_efficiency_mean_median.ecsv``: mean 0.694 below 900 nm and
0.887 over 900-2200 nm) is reproduced to 0.001 as a check of the parsing.
The detector-to-detector spread is recorded in the manifest notes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from ..filter_components import QE, SHARED, InstrumentManifest, add_component
from ._original import Fetcher

FACILITY, INSTRUMENT = "Roman", "WFI"
COMMIT = "14a9b0c98b0da723d0f2e4c6277956183c1fdd06"  # repository HEAD on 2026-10-26 (v1.6)
REPO = "https://github.com/RomanSpaceTelescope/roman-technical-information"
BASE = (f"https://raw.githubusercontent.com/RomanSpaceTelescope/roman-technical-information/{COMMIT}/"
        "roman_technical_information/data/WideFieldInstrument/FPSPerformance/")
QE_URL = BASE + "WFI_Quantum_efficiency.ecsv"
SUMMARY_URL = BASE + "WFI_Quantum_efficiency_mean_median.ecsv"
N_DETECTORS = 18
#: The team's mean QE below 900 nm and over 900-2200 nm (summary file).
SUMMARY_TOLERANCE = 0.001

#: Recorded on 2026-10-26 by comparing NJM's curves with the v8 files.
RELEASE_NOTE = (
    "Release: the QE is the DCL per-detector measurement from the repository at commit "
    f"{COMMIT[:7]} (2026-07-24). The effective areas of release v8 (2024-03-01) are combined "
    "products, and the QE they used is not stated: dividing each SCA's v8 effective area by the QE "
    "of the detector with the same position number leaves the SCA-to-SCA spread at about 3% (F087: "
    "3.1% before, 3.3% after; F129: 3.6% before, 3.3% after), so these may not be the same QE curves. SVO's Roman/WFI curves "
    "(here the <band>_old curves since 2026-10-26) are from an earlier release than v8: SVO's F129 "
    "peaks at 2.870 m^2, above every v8 SCA (2.416-2.814 m^2)."
)


def _declared_units(text: str) -> Dict[str, str]:
    """Column units from the ECSV ``datatype`` header (``''`` when dimensionless)."""
    units = {}
    for m in re.finditer(r"\{name: ([^,]+), unit: ([^,]*), datatype", text):
        units[m.group(1).strip()] = m.group(2).strip().strip("'\"")
    return units


def parse_qe(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """(wavelength in nm, QE array of shape (18, n)) from the per-detector ECSV."""
    units = _declared_units(text)
    if units.get("wavelength") != "nm":
        raise ValueError(f"Roman QE: wavelength unit {units.get('wavelength')!r}, expected nm")
    detectors = [f"SCU {i}" for i in range(1, N_DETECTORS + 1)]
    if any(units.get(d, None) != "" for d in detectors):
        raise ValueError("Roman QE: expected 18 dimensionless SCU columns")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    names = [c.strip() for c in lines[0].split(",")]
    if names != ["wavelength"] + detectors:
        raise ValueError(f"Roman QE: unexpected columns {names}")
    data = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]], dtype=float)
    wl, qe = data[:, 0], data[:, 1:].T
    if np.any(np.diff(wl) <= 0) or not np.all(np.isfinite(qe)) or qe.min() < 0 or qe.max() > 1.0:
        raise ValueError("Roman QE: wavelengths must increase and QE lie in [0, 1]")
    return wl, qe


def check_summary(text: str, wl: np.ndarray, qe: np.ndarray) -> None:
    """Compare the team's mean QE summary with the per-detector table."""
    rows = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")][1:]
    means = [float(r.split(",")[1]) for r in rows]
    mine = [qe[:, wl < 900].mean(), qe[:, (wl >= 900) & (wl <= 2200)].mean()]
    if len(means) != 2 or max(abs(a - b) for a, b in zip(means, mine)) > SUMMARY_TOLERANCE:
        raise ValueError(f"Roman QE: team summary means {means} vs table {mine}")


def spread_note(wl: np.ndarray, qe: np.ndarray) -> str:
    mean = qe.mean(0)
    rel = np.divide(qe.std(0), mean, out=np.zeros_like(mean), where=mean > 0)
    parts = []
    for lo, hi in ((480, 900), (900, 2200), (2200, 2700)):
        m = (wl >= lo) & (wl <= hi)
        if not m.any():
            continue
        parts.append(f"{lo}-{hi} nm: std/mean median {np.median(rel[m]):.3f}, max {rel[m].max():.3f}; "
                     f"largest max-min {np.ptp(qe[:, m], axis=0).max():.3f}")
    return ("Detector-to-detector spread of the QE around the mean (18 detectors): "
            + "; ".join(parts) + ". The spread beyond 2200 nm reflects each detector's cut-off.")


def build(filter_root: Union[str, Path], *, session=None, **_ignored) -> Path:
    """Add the focal-plane mean QE to ``Roman/WFI``."""
    inst_dir = Path(filter_root) / FACILITY / INSTRUMENT
    if not list(inst_dir.glob("*.dat")):
        raise FileNotFoundError(f"{inst_dir}: no published curves; fetch them first")
    fetch = Fetcher(session)
    wl, qe = parse_qe(fetch.text(QE_URL))
    check_summary(fetch.text(SUMMARY_URL), wl, qe)

    manifest = InstrumentManifest.load(inst_dir)
    manifest.components.pop(QE, None)
    add_component(inst_dir, QE, wl * 10.0, qe.mean(0), band=SHARED, name="qe",
                  source=f"{QE_URL} (mean of the 18 SCU columns; DCL median QE per detector, 95 K)",
                  manifest=manifest)
    starts = sorted(b for b in manifest.primary_bands()
                    if np.loadtxt(inst_dir / f"{b}.dat", delimiter=",", skiprows=1, usecols=(0,),
                                  max_rows=1) < wl[0] * 10.0)
    manifest.facility, manifest.instrument = FACILITY, INSTRUMENT
    manifest.ground_based = False
    manifest.sources["roman_technical_information"] = f"{REPO} (commit {COMMIT})"
    manifest.notes = " ".join(filter(None, [
        "QE only: the team publishes no separate filter or optics transmission, so no recipe can be "
        "composed and the published curves are used as they are. The qe component is the mean of the "
        "18 detectors' median QE (DCL, 95 K), chosen because the published curves describe the whole "
        "focal plane; per-detector curves are in the source table.",
        spread_note(wl, qe),
        (f"The QE starts at {wl[0] * 10:g} A; these bands' published curves start below that: "
         + ", ".join(starts) + ".") if starts else "",
        RELEASE_NOTE,
    ]))
    manifest.save(inst_dir)
    return inst_dir
