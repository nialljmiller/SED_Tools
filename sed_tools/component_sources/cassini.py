"""Cassini ISS NAC/WAC components from the PDS calibration volume COISS_0011.

The published curves (SVO copies on the njm mirror) equal the PDS system
transmission files ``calib/efficiency/systrans/iss<cam><f1><f2>_systrans.tab``
column 2, which is QE x optics x filter1 x filter2: every ISS exposure
passes through one filter of each of the two filter wheels.  The team
publishes the CCD quantum efficiency (``ccdqe.tab``) and the optics
transmission (``na_optics.tab``, ``wa_optics.tab``) separately; dividing the
system transmission by both yields the transmission of the filter *pair*
(e.g. BL1 x CL2), never a single filter.  That quotient is a derived
quantity, published here only within its validity range (to 1000 nm; for
the WAC from 440 nm, see :data:`VALID_NM`).

This builder only adds components: the published curves are kept as they
are (they must already be present, from the mirror or SVO).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..filter_components import InstrumentManifest, add_component
from ..filter_metadata import read_filter_header
from ._original import Fetcher, clean_curve

logger = logging.getLogger(__name__)

FACILITY = "Cassini"
VOLUME = "https://pds-rings.seti.org/holdings/volumes/COISS_0xxx/COISS_0011/"
EFFICIENCY = VOLUME + "calib/efficiency/"
QE_URL = EFFICIENCY + "ccdqe.tab"
INFO_URL = EFFICIENCY + "efficiency_info.txt"
THEORY_URL = VOLUME + "document/theoretical_basis.tex"

#: camera -> (systrans prefix, optics table, ccdqe.tab column, filter wheel 1, filter wheel 2)
CAMERAS: Dict[str, Tuple[str, str, int, frozenset, frozenset]] = {
    "ISS_NAC": ("issna", "na_optics.tab", 1,
                frozenset("cl1 red bl1 uv2 uv1 irp0 p120 p60 p0 hal ir4 ir2".split()),
                frozenset("cl2 cb1 cb1a cb1b cb2 cb3 mt1 mt2 mt3 uv3 ir1 ir3 bl2 grn".split())),
    "ISS_WAC": ("isswa", "wa_optics.tab", 2,
                frozenset("cl1 ir3 ir4 ir5 cb3 mt3 cb2 mt2 ir2".split()),
                frozenset("cl2 red grn bl1 vio hal irp90 irp0 ir1".split())),
}
#: Wavelength range (nm) where the derived filter pair is published.  Above
#: 1000 nm the component review gives no validity (C01); below 440 nm the WAC
#: quotient carries a sawtooth with the 2.5 nm period of wa_optics.tab (4 % at
#: 435-445 nm, 36 % at 395 nm), where the WAC optics rise from 0.007 at 390 nm.
VALID_NM = {"ISS_NAC": (200.0, 1000.0), "ISS_WAC": (440.0, 1000.0)}
#: A band gets a derived pair only if its system transmission above this
#: fraction of its peak lies inside VALID_NM (the threshold composition uses).
SUPPORT_FRACTION = 1e-3


def pds_table(text: str, ncol: int) -> np.ndarray:
    """Numeric rows after the ``\\begindata`` line of a PDS efficiency table."""
    rows, started = [], False
    for line in text.splitlines():
        if line.strip().startswith("\\begindata"):
            started = True
            continue
        if started and line.strip():
            rows.append([float(x) for x in line.split()[:ncol]])
    if not started or not rows:
        raise ValueError("PDS table without \\begindata rows")
    a = np.array(rows)
    if np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError("PDS table wavelengths not increasing")
    return a


def filter_pair(camera: str, band: str) -> Tuple[str, str]:
    """(wheel-1, wheel-2) filter names of an njm band name such as ``BL1``, ``RED_GRN`` or ``Clear``."""
    _prefix, _optics, _col, w1, w2 = CAMERAS[camera]
    if band == "Clear":
        return "cl1", "cl2"
    parts = band.lower().split("_")
    if len(parts) == 2 and parts[0] in w1 and parts[1] in w2:
        return parts[0], parts[1]
    if len(parts) == 1:
        (name,) = parts
        if name in w1 and name not in w2:
            return name, "cl2"
        if name in w2 and name not in w1:
            return "cl1", name
    raise ValueError(f"{camera}: cannot map band {band!r} to a filter pair")


def systrans_url(camera: str, band: str) -> str:
    f1, f2 = filter_pair(camera, band)
    return f"{EFFICIENCY}systrans/{CAMERAS[camera][0]}{f1}{f2}_systrans.tab"


def _support(wl: np.ndarray, tr: np.ndarray) -> Tuple[float, float]:
    above = np.nonzero(tr > SUPPORT_FRACTION * tr.max())[0]
    return float(wl[above[0]]), float(wl[above[-1]])


def _published_curve(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1), ndmin=2)
    return a[:, 0], a[:, 1]


def build(filter_root, *, instrument, session=None, facility=None, **_ignored) -> Path:
    if instrument not in CAMERAS:
        raise ValueError(f"No Cassini ISS camera {instrument!r}; available: {sorted(CAMERAS)}")
    if facility is not None and facility != FACILITY:
        raise ValueError(f"{instrument} belongs to {FACILITY}, not {facility}")
    inst_dir = Path(filter_root) / FACILITY / instrument
    bands = sorted(p.stem for p in inst_dir.glob("*.dat"))
    if not bands:
        raise ValueError(f"{inst_dir}: no published curves; fetch {FACILITY}/{instrument} first")
    prefix, optics_name, qe_col, _w1, _w2 = CAMERAS[instrument]
    optics_url = EFFICIENCY + optics_name
    lo_nm, hi_nm = VALID_NM[instrument]
    fetch = Fetcher(session)

    # Fetch and check everything before writing anything.
    qe = pds_table(fetch.text(QE_URL), 3)
    optics = pds_table(fetch.text(optics_url), 2)
    qe_wl, qe_tr = qe[:, 0], qe[:, qe_col]
    pairs, ranges, skipped, equal, unequal = {}, {}, [], [], []
    for band in bands:
        f1, f2 = filter_pair(instrument, band)
        url = systrans_url(instrument, band)
        st = pds_table(fetch.text(url), 2)
        wl, sys_tr = st[:, 0], st[:, 1]
        # Published (njm/SVO) curve vs systrans column 2, inside the band.
        pw, pt = _published_curve(inst_dir / f"{band}.dat")
        core = pt > 0.05 * pt.max()
        ratio = np.interp(pw[core] / 10.0, wl, sys_tr) / pt[core]
        (equal if np.all(np.abs(ratio - 1.0) < 1e-3) else unequal).append(
            band if np.all(np.abs(ratio - 1.0) < 1e-3) else f"{band} ({ratio.min():.4f}-{ratio.max():.4f})")
        s_lo, s_hi = _support(wl, sys_tr)
        if s_lo < lo_nm or s_hi > hi_nm:
            skipped.append(f"{band} ({f1.upper()}x{f2.upper()}, system transmission {s_lo:.0f}-{s_hi:.0f} nm)")
            continue
        keep = (wl >= max(lo_nm, qe_wl[0], optics[0, 0])) & (wl <= min(hi_nm, qe_wl[-1], optics[-1, 0]))
        den = np.interp(wl[keep], qe_wl, qe_tr) * np.interp(wl[keep], optics[:, 0], optics[:, 1])
        if np.any(den <= 0):
            raise ValueError(f"{instrument} {band}: QE x optics vanishes inside the validity range")
        pair = sys_tr[keep] / den
        if pair.max() > 1.0:
            raise ValueError(f"{instrument} {band}: derived filter pair exceeds 1 ({pair.max():.4f})")
        pw_a, pt_a, note = clean_curve(wl[keep] * 10.0, pair)
        pairs[band] = (pw_a, pt_a, url, f1, f2, note)
        ranges[band] = (pw[0], pw[-1], wl[0] * 10.0, wl[-1] * 10.0)

    manifest = InstrumentManifest.from_published(inst_dir)
    manifest.components = {}
    qw, qt, _ = clean_curve(qe_wl * 10.0, qe_tr)
    add_component(inst_dir, "qe", qw, qt, name="qe", manifest=manifest,
                  source=f"{QE_URL} column {qe_col + 1} (CCD {'220 (NAC)' if qe_col == 1 else '361 (WAC)'}, "
                         "e-/photon; nm converted to Angstrom)")
    ow, ot, _ = clean_curve(optics[:, 0] * 10.0, optics[:, 1])
    add_component(inst_dir, "optics", ow, ot, name="optics", manifest=manifest,
                  source=f"{optics_url} (optics transmission; nm converted to Angstrom)")
    for band, (w, t, url, f1, f2, note) in pairs.items():
        add_component(
            inst_dir, "filter", w, t, band=band, name=band, manifest=manifest,
            source=(f"DERIVED filter pair T({f1.upper()})*T({f2.upper()}) = column 2 of {url} / "
                    f"(ccdqe.tab x {optics_name}), valid {lo_nm:.0f}-{hi_nm:.0f} nm"
                    + (f"; {note}" if note else "")))

    camera = "NAC" if instrument == "ISS_NAC" else "WAC"
    manifest.ground_based = False
    manifest.outside_range = "error"
    manifest.sources.update({"qe": QE_URL, "optics": optics_url,
                             "filter": EFFICIENCY + "systrans/ (derived filter pairs)",
                             "documentation": f"{INFO_URL} | {THEORY_URL}"})
    pair_map = ", ".join(f"{b}={'x'.join(p.upper() for p in filter_pair(instrument, b))}" for b in bands)
    manifest.notes = (
        f"Cassini ISS {camera} components from the PDS Ring-Moon Systems Node calibration volume COISS_0011 "
        f"(calib/efficiency; requests need a User-Agent header). qe = {QE_URL} column {qe_col + 1} "
        f"(CCD quantum efficiency, e-/photon, 200-1100 nm, 2004-12-07). optics = {optics_url} (optics "
        "transmission, 2004-12-07). filter = DERIVED filter PAIR: every ISS image passes one filter of each "
        "wheel, and systrans/<cam><f1><f2>_systrans.tab column 2 is QE x optics x filter1 x filter2 "
        "(efficiency_info.txt), so column 2 / (QE x optics), with both tables interpolated linearly, is "
        "T(f1) x T(f2), not a single filter. It is published only from "
        f"{lo_nm:.0f} to {hi_nm:.0f} nm: the component review (C01) gives about 1000 nm as the upper validity "
        "limit (I did not find the basis in the PDS documents; the QE falls from 0.023 at 1000 nm to 0 at 1058 "
        "nm)"
        + ("; below 440 nm the WAC quotient has a sawtooth with the 2.5 nm sampling period of wa_optics.tab "
           "(about 36 % at 395 nm, 4 % at 435-445 nm), so the PDS system transmission was evidently not "
           "computed with linearly interpolated optics there" if camera == "WAC" else
           "; the NAC quotient shows small-scale ripple of order 1 % (Clear pair: up to 2.6 % within 5 nm "
           "windows at 200-260 nm)")
        + ". Bands whose system transmission (above 1e-3 of peak) extends outside that range get qe and "
        "optics only, no filter pair, because a truncated pair would silently cut the band when composed: "
        + ("; ".join(skipped) or "none") + ". Composing filter+optics+qe reproduces systrans column 2. "
        "Filter pairs (wheel 1 x wheel 2): " + pair_map + ". Published curves: equal to systrans column 2 to "
        f"better than 0.1 % wherever they exceed 5 % of their peak for {len(equal)} of {len(bands)} bands"
        + (f"; differ for {'; '.join(unequal)}" if unequal else "")
        + " (checked when these components were built). Their headers (SVO) say "
        "filter+optics+qe+instrument; the PDS definition is QE x optics x filter1 x filter2. "
        "DetectorType photon: the ISS theoretical basis (COISS_0011 document/theoretical_basis.tex eq. 1) "
        "counts electrons as the integral of photon radiance x optics x filters x QE (e-/photon). "
        "Polarizer combinations (P0, P60, P120, IRP0, IRP90) are not among the published bands.")
    manifest.save(inst_dir)

    from . import compare_to_published

    dmag = compare_to_published(inst_dir)
    worst = [f"{b} {d:.3f} mag (published {ranges[b][0]:.0f}-{ranges[b][1]:.0f} A, systrans "
             f"{ranges[b][2]:.0f}-{ranges[b][3]:.0f} A)" for b, d in sorted(dmag.items(), key=lambda x: -x[1])
             if d > 1e-3]
    manifest.notes += (
        " Composed filter+optics+qe vs published curve (max |dmag| for 3000 K and 30000 K black bodies): "
        + (f"below 0.001 mag except {'; '.join(worst)}" if worst else "below 0.001 mag for every band")
        + ". The composed curve is systrans column 2 (within the validity range); the published (SVO) curves are windows cut from "
        "it, so they drop the far tails (e.g. red leaks of the UV filters), which cool sources feel.")
    manifest.save(inst_dir)
    return inst_dir
