"""Cassini ISS NAC/WAC from the PDS calibration volume COISS_0011.

Published curves: the PDS system transmission files
``calib/efficiency/systrans/iss<cam><f1><f2>_systrans.tab`` column 2, which is
QE x optics x filter1 x filter2: every ISS exposure passes through one filter
of each of the two filter wheels.  SVO's copies are windows cut from these
files; on the njm mirror they are kept as ``<band>_svo``.

Components: the CCD quantum efficiency (``ccdqe.tab``) and the optics
transmission (``na_optics.tab``, ``wa_optics.tab``), published separately by
the team, and the transmission of each band's filter *pair* (e.g. BL1 x CL2),
derived as system transmission / (QE x optics).  That quotient is published
only within its validity range (see :data:`VALID_NM`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..filter_components import InstrumentManifest
from ._original import Component, Fetcher, Instrument, clean_curve, make_band, write_instrument

FACILITY = "Cassini"
VOLUME = "https://pds-rings.seti.org/holdings/volumes/COISS_0xxx/COISS_0011/"
EFFICIENCY = VOLUME + "calib/efficiency/"
QE_URL = EFFICIENCY + "ccdqe.tab"
INFO_URL = EFFICIENCY + "efficiency_info.txt"
THEORY_URL = VOLUME + "document/theoretical_basis.tex"
PHOTON = "photon"

#: camera -> (systrans prefix, optics table, ccdqe.tab column, filter wheel 1, filter wheel 2)
CAMERAS: Dict[str, Tuple[str, str, int, frozenset, frozenset]] = {
    "ISS_NAC": ("issna", "na_optics.tab", 1,
                frozenset("cl1 red bl1 uv2 uv1 irp0 p120 p60 p0 hal ir4 ir2".split()),
                frozenset("cl2 cb1 cb1a cb1b cb2 cb3 mt1 mt2 mt3 uv3 ir1 ir3 bl2 grn".split())),
    "ISS_WAC": ("isswa", "wa_optics.tab", 2,
                frozenset("cl1 ir3 ir4 ir5 cb3 mt3 cb2 mt2 ir2".split()),
                frozenset("cl2 red grn bl1 vio hal irp90 irp0 ir1".split())),
}
#: The njm band names (the SVO set): a filter name alone is that filter with the
#: other wheel's clear filter; ``A_B`` is wheel-1 filter A with wheel-2 filter B.
BANDS = {
    "ISS_NAC": ("BL1 BL1_BL2 BL1_GRN BL1_UV3 BL2 CB1 CB1a CB1b CB2 CB3 Clear GRN HAL HAL_CB1 HAL_GRN HAL_IR1 "
                "IR1 IR2 IR2_CB3 IR2_IR1 IR2_IR3 IR2_MT3 IR3 IR4 IR4_IR3 MT1 MT2 MT3 RED RED_CB1 RED_CB2 "
                "RED_GRN RED_IR1 RED_MT1 RED_MT2 UV1 UV1_UV3 UV2 UV2_UV3 UV3").split(),
    "ISS_WAC": ("BL1 CB2 CB2_IR1 CB2_RED CB3 Clear GRN HAL IR1 IR2 IR2_IR1 IR3 IR4 IR5 MT2 MT2_IR1 MT2_RED "
                "MT3 RED VIO").split(),
}
#: Suffix of the SVO copies kept beside the original curves on the njm mirror.
SVO_SUFFIX = "_svo"
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


def _camera(camera: str, fetch: Fetcher) -> Instrument:
    """Published curves, components and notes of one camera."""
    prefix, optics_name, qe_col, _w1, _w2 = CAMERAS[camera]
    optics_url = EFFICIENCY + optics_name
    lo_nm, hi_nm = VALID_NM[camera]
    qe = pds_table(fetch.text(QE_URL), 3)
    optics = pds_table(fetch.text(optics_url), 2)
    qe_wl, qe_tr = qe[:, 0], qe[:, qe_col]
    bands, comps, skipped = [], [], []
    for band in BANDS[camera]:
        f1, f2 = filter_pair(camera, band)
        url = systrans_url(camera, band)
        st = pds_table(fetch.text(url), 2)
        wl, sys_tr = st[:, 0], st[:, 1]
        bands.append(make_band(band, wl, sys_tr, unit="nm", detector_type=PHOTON, components="filter+optics+qe",
                               sources=f"{url} | {INFO_URL} | {THEORY_URL}"))
        s_lo, s_hi = _support(wl, sys_tr)
        if s_lo < lo_nm or s_hi > hi_nm:
            skipped.append(f"{band} ({f1.upper()}x{f2.upper()}, system transmission {s_lo:.0f}-{s_hi:.0f} nm)")
            continue
        keep = (wl >= max(lo_nm, qe_wl[0], optics[0, 0])) & (wl <= min(hi_nm, qe_wl[-1], optics[-1, 0]))
        den = np.interp(wl[keep], qe_wl, qe_tr) * np.interp(wl[keep], optics[:, 0], optics[:, 1])
        if np.any(den <= 0):
            raise ValueError(f"{camera} {band}: QE x optics vanishes inside the validity range")
        pair = sys_tr[keep] / den
        if pair.max() > 1.0:
            raise ValueError(f"{camera} {band}: derived filter pair exceeds 1 ({pair.max():.4f})")
        pw, pt, note = clean_curve(wl[keep] * 10.0, pair)
        comps.append(Component(
            "filter", pw, pt, band=band, name=band,
            source=(f"DERIVED filter pair T({f1.upper()})*T({f2.upper()}) = column 2 of {url} / "
                    f"(ccdqe.tab x {optics_name}), valid {lo_nm:.0f}-{hi_nm:.0f} nm" + (f"; {note}" if note else ""))))
    qw, qt, _ = clean_curve(qe_wl * 10.0, qe_tr)
    comps.append(Component("qe", qw, qt, name="qe",
                           source=f"{QE_URL} column {qe_col + 1} (CCD {'220 (NAC)' if qe_col == 1 else '361 (WAC)'}, "
                                  "e-/photon; nm converted to Angstrom)"))
    ow, ot, _ = clean_curve(optics[:, 0] * 10.0, optics[:, 1])
    comps.append(Component("optics", ow, ot, name="optics",
                           source=f"{optics_url} (optics transmission; nm converted to Angstrom)"))

    cam = "NAC" if camera == "ISS_NAC" else "WAC"
    pair_map = ", ".join(f"{b}={'x'.join(p.upper() for p in filter_pair(camera, b))}" for b in BANDS[camera])
    clean = "; ".join(f"{b.name}: {b.clean_note}" for b in bands if b.clean_note)
    note = (
        f"Cassini ISS {cam} from the PDS Ring-Moon Systems Node calibration volume COISS_0011 (calib/efficiency; "
        "requests need a User-Agent header). Published curves: column 2 of systrans/<cam><f1><f2>_systrans.tab, "
        "the system transmission QE x optics x filter1 x filter2 (efficiency_info.txt; e-/photon, so "
        "dimensionless) on its full 0.1 nm grid, nm converted to Angstrom"
        + (f" ({clean})" if clean else "") + ". Every ISS image passes one filter of each wheel; each band is "
        "a filter pair (wheel 1 x wheel 2): " + pair_map + ". SVO's copies, windows cut from these files, are "
        f"kept on the njm mirror as <band>{SVO_SUFFIX} and are not part of this manifest. Components: qe = "
        f"{QE_URL} column {qe_col + 1} (CCD quantum efficiency, e-/photon, 200-1100 nm, 2004-12-07); optics = "
        f"{optics_url} (optics transmission, 2004-12-07); filter = DERIVED filter PAIR T(f1) x T(f2) = "
        "column 2 / (QE x optics), both tables interpolated linearly, not a single filter. It is published only "
        f"from {VALID_NM[camera][0]:.0f} to {VALID_NM[camera][1]:.0f} nm: the component review (C01) gives about "
        "1000 nm as the upper validity limit (I did not find the basis in the PDS documents; the QE falls from "
        "0.023 at 1000 nm to 0 at 1058 nm)"
        + ("; below 440 nm the WAC quotient has a sawtooth with the 2.5 nm sampling period of wa_optics.tab "
           "(about 36 % at 395 nm, 4 % at 435-445 nm), so the PDS system transmission was evidently not "
           "computed with linearly interpolated optics there" if cam == "WAC" else
           "; the NAC quotient shows small-scale ripple of order 1 % (Clear pair: up to 2.6 % within 5 nm "
           "windows at 200-260 nm)")
        + ". Bands whose system transmission (above 1e-3 of peak) extends outside that range get qe and "
        "optics only, no filter pair, because a truncated pair would silently cut the band when composed: "
        + ("; ".join(skipped) or "none") + ". DetectorType photon: the ISS theoretical basis (COISS_0011 "
        "document/theoretical_basis.tex eq. 1) counts electrons as the integral of photon radiance x optics x "
        "filters x QE (e-/photon). Polarizer combinations (P0, P60, P120, IRP0, IRP90) are not among the bands.")
    return Instrument(FACILITY, camera, bands, note, ground_based=False, components=comps,
                      outside_range="error")


def iss_nac(fetch: Fetcher) -> Instrument:
    return _camera("ISS_NAC", fetch)


def iss_wac(fetch: Fetcher) -> Instrument:
    return _camera("ISS_WAC", fetch)


RECIPES = {"ISS_NAC": (FACILITY, iss_nac), "ISS_WAC": (FACILITY, iss_wac)}


def _curve(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1), ndmin=2)
    return a[:, 0], a[:, 1]


def build(filter_root, *, instrument, session=None, facility=None, **_ignored) -> Path:
    """Write missing published curves, the components and the manifest, then record the checks."""
    if instrument not in CAMERAS:
        raise ValueError(f"No Cassini ISS camera {instrument!r}; available: {sorted(CAMERAS)}")
    if facility is not None and facility != FACILITY:
        raise ValueError(f"{instrument} belongs to {FACILITY}, not {facility}")
    spec = _camera(instrument, Fetcher(session))
    inst_dir = write_instrument(filter_root, spec)

    # The manifest describes the original-source bands only; SVO copies stay outside it.
    manifest = InstrumentManifest.load(inst_dir)
    keep = set(BANDS[instrument])
    manifest.bands = [b for b in manifest.bands if b in keep]
    for field in (manifest.published_bands, manifest.detector_types, manifest.detector_type_sources):
        for b in list(field):
            if b not in keep:
                del field[b]
    recipes = set(manifest.published_bands.values())
    manifest.published = {"components": recipes.pop() if len(recipes) == 1 else "mixed", "source": "original"}
    manifest.save(inst_dir)

    from . import compare_to_published

    checks = []
    exact, other = [], []
    for band in BANDS[instrument]:
        pw, pt = _curve(inst_dir / f"{band}.dat")
        sys_wl = np.asarray(next(b for b in spec.bands if b.name == band).wavelength)
        sys_tr = np.asarray(next(b for b in spec.bands if b.name == band).transmission)
        (exact if np.array_equal(pw, sys_wl) and np.allclose(pt, sys_tr, rtol=1e-12, atol=0) else other).append(band)
        svo = inst_dir / f"{band}{SVO_SUFFIX}.dat"
        if svo.is_file():
            sw, st = _curve(svo)
            core = st > 0.05 * st.max()
            ratio = np.interp(sw[core], pw, pt) / st[core]
            if not np.all(np.abs(ratio - 1.0) < 1e-3):
                checks.append(f"{band}{SVO_SUFFIX} differs from the PDS curve by {ratio.min():.4f}-{ratio.max():.4f}")
    dmag = compare_to_published(inst_dir)
    worst = [f"{b} {d:.4f}" for b, d in sorted(dmag.items(), key=lambda x: -x[1]) if d > 1e-4]
    svo_bands = sorted(p.stem for p in inst_dir.glob(f"*{SVO_SUFFIX}.dat"))
    svo_note = ""
    if svo_bands:
        cuts = [f"{b} {_curve(inst_dir / f'{b}{SVO_SUFFIX}.dat')[0][[0, -1]].round().astype(int).tolist()} A"
                for b in BANDS[instrument] if (inst_dir / f"{b}{SVO_SUFFIX}.dat").is_file()]
        svo_note = (f" The {len(svo_bands)} SVO copies equal the PDS curve to better than 0.1 % wherever they exceed "
                    "5 % of their peak" + (f" except {'; '.join(checks)}" if checks else "") + "; they cover only "
                    "windows of it (" + ", ".join(cuts) + "), so they drop the far tails such as the UV filters' "
                    "red leaks.")
    manifest.notes = spec.notes + (
        f" Checks when built: the published curve is column 2 itself for {len(exact)} of {len(BANDS[instrument])} "
        "bands" + (f" (not for {', '.join(other)})" if other else "") + "; composed filter+optics+qe vs published "
        "(max |dmag| for 3000 K and 30000 K black bodies, bands with a filter pair): "
        + (f"above 1e-4 mag only for {'; '.join(worst)}" if worst else "below 1e-4 mag for every band")
        + " (the composed curve stops at the validity limits, where the system transmission is below 1e-3 of "
        "its peak)." + svo_note)
    manifest.save(inst_dir)
    return inst_dir
