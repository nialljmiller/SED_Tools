"""ESO Paranal: VLT FORS1, VLT FORS2 and VST OmegaCAM broad-band filters (gemini.csv, block C).

FORS (sources: FORS filter page https://www.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters.html,
its curve list .../fors/inst/Filters/curves.html, and the FORS User Manuals VLT-MAN-ESO-13100-1543
issue 82.1 (2008, Table B.1: FORS1/FORS2 broad-band filters, lambda0, FWHM) and P117.3 (Table 2.1:
FORS2 standard configuration)).  All FORS curves are filter-only transmission:
  * FORS1 = the original FORS1 set (U/B/V/R/I_BESS+33..37, u/v/g/r_GUNN+38..41) from the optical-bench
    lab scans M_BESS_*.txt / M_GUNN_*.txt (nm, fraction; M_GUNN_G is a coarse 20-point scan).
  * FORS2 = today's standard configuration (P117 manual Table 2.1): u/b/v/g_HIGH+112..115 (moved from
    FORS1 in 2009), R_SPECIAL+76, I_BESS+77.  u_HIGH.txt / g_HIGH.txt are lab scans (nm, %T, 300-1500 nm);
    b_HIGH+113COLL_SR, v_HIGH+114COLL_SR, R_SPECIAL+76COLL_SR, I_BESS+77COLL_SR are .ascii tables
    (Angstrom, fraction) measured with the SR collimator.
  * Every FORS curve is cut at 1100 nm, the long end of the FORS design range (330-1100 nm, manual P117
    section 2.1) where the CCDs stop responding: this drops the spurious final 1200 nm point of each M_
    scan (e.g. 0 -> 0.16 for BESS_V) and the >1.1 um red leak of the u_HIGH/g_HIGH lab scans (<=0.02 %
    at 1100 nm, rising to 12 % at 1500 nm), which would otherwise dominate a filter-only pivot.
  HELD: z_GUNN (+42/+78) is a long-pass filter whose red edge is set by the CCD QE, which ESO does not
  tabulate, so a filter-only curve is not a usable band; H_Alpha: gemini.csv's M_IF_691.txt is FILT_691_55
  (the "special 691" night-sky filter), and the in-instrument FILTER+83COLL_SR.txt linked for H_Alpha+83
  is internally labelled FILTER+81 / H_Alpha+81 and measured through GRIS_600R with uncalibrated lamp flux.

OmegaCAM (source: https://www.eso.org/sci/facilities/paranal/instruments/omegacam/tools/OmegaCAM_transmission.tar.gz,
linked from .../omegacam/tools.html; filter table on .../omegacam/inst.html).  Each member is a lab scan
(nm, fraction) at many positions on the filter; column 2 is the scan at the filter centre (x=y=0).  The
*_with_ccd members are the same scans multiplied by the CCD QE: these are the published curves used
(filter+qe); the filter-only scans are attached as per-band filter components.  u,g,r,i,z_SDSS are
monolithic; B_JOHN and V_JOHN are segmented (4 segments, same glass; inst.html), so their curve is the
mean of the four segment-centre scans (partA..partD, column 2).  Not built: v_STRM, H_ALPHA (4 quadrants
with different passbands), the u_g_r_i calibration filter.

Detectors: CCDs (FORS1 Tektronix then E2V, FORS2 MIT/E2V, OmegaCAM 32 E2V 44-82) -> photon counting.
"""
import tarfile

import numpy as np
from fxlib import ORIGINAL, Component, build_instrument, fetch, make_band

FORS = "https://www.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/"
FORS_PAGE = "https://www.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/curves.html"
MAN82 = "https://www.eso.org/sci/facilities/paranal/instruments/fors/doc/VLT-MAN-ESO-13100-1543_v82.pdf"
MAN117 = "https://www.eso.org/sci/facilities/paranal/instruments/fors/doc/VLT-MAN-ESO-13100-1543_P117_3.pdf"
CUT_NM = 1100.0

# band -> (file, unit, percent, lambda0 nm from manual 82.1 Table B.1)
FORS1 = {"U_BESS": ("M_BESS_U.txt", "nm", False, 366), "B_BESS": ("M_BESS_B.txt", "nm", False, 429),
         "V_BESS": ("M_BESS_V.txt", "nm", False, 554), "R_BESS": ("M_BESS_R.txt", "nm", False, 657),
         "I_BESS": ("M_BESS_I.txt", "nm", False, 768), "u_GUNN": ("M_GUNN_U.txt", "nm", False, 359),
         "v_GUNN": ("M_GUNN_V.txt", "nm", False, 398), "g_GUNN": ("M_GUNN_G.txt", "nm", False, 506),
         "r_GUNN": ("M_GUNN_R.txt", "nm", False, 653)}
FORS2 = {"u_HIGH": ("u_HIGH.txt", "nm", True, 365), "b_HIGH": ("b_HIGH+113COLL_SR.ascii", "Angstrom", False, 442),
         "v_HIGH": ("v_HIGH+114COLL_SR.ascii", "Angstrom", False, 561),
         "g_HIGH": ("g_HIGH.txt", "nm", True, 471),
         "R_SPECIAL": ("R_SPECIAL+76COLL_SR.ascii", "Angstrom", False, 655),
         "I_BESS": ("I_BESS+77COLL_SR.ascii", "Angstrom", False, 768)}
# lambda0 in the manual is a central wavelength, not a pivot: it is checked as ref_pivot only where the
# filter-only pivot agrees to <1.5 %.  Not checked (pivot vs lambda0): FORS1 U_BESS -2.2 %, I_BESS +3.4 %,
# u_GUNN -1.8 %; FORS2 I_BESS +3.6 % (both I_BESS scans have their half-power midpoint near 795 nm).
FORS_REF_OK = {"FORS1": {"B_BESS", "V_BESS", "R_BESS", "v_GUNN", "g_GUNN", "r_GUNN"},
               "FORS2": {"u_HIGH", "b_HIGH", "v_HIGH", "g_HIGH", "R_SPECIAL"}}

OCAM_URL = "https://www.eso.org/sci/facilities/paranal/instruments/omegacam/tools/OmegaCAM_transmission.tar.gz"
OCAM_PAGE = "https://www.eso.org/sci/facilities/paranal/instruments/omegacam/inst.html"
# inst.html "Lambda" (nm, rounded); checked where within 1.5 % (u +2.0 %, g -1.8 %, i -1.9 %, z -2.3 % not).
OCAM_REF = {"r_SDSS": 625, "B_JOHN": 440, "V_JOHN": 550}
OCAM = {"u_SDSS": ["u"], "g_SDSS": ["g"], "r_SDSS": ["r"], "i_SDSS": ["i"], "z_SDSS": ["z"],
        "B_JOHN": [f"B-part{p}" for p in "ABCD"], "V_JOHN": [f"V-part{p}" for p in "ABCD"]}


def _numeric(path, cols=(0, 1)):
    rows = []
    for ln in path.read_text(encoding="latin-1").splitlines():
        parts = ln.split()
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue                      # title / column-name lines
        if len(vals) > max(cols):
            rows.append([vals[c] for c in cols])
    return np.array(rows)


def _fors(instrument, table, detector_note):
    bands, raw = [], []
    for name, (fname, unit, percent, lam0) in table.items():
        path = fetch(FORS + fname, "Paranal")
        raw.append(path)
        if fname in ("u_HIGH.txt", "g_HIGH.txt"):
            head = path.read_text(encoding="latin-1").lower()
            if f"{name[0]}-fors" not in head or "%t" not in head:
                raise ValueError(f"{fname}: unexpected header")
        d = _numeric(path)
        wl_nm = d[:, 0] / (10.0 if unit == "Angstrom" else 1.0)
        if not (250 < wl_nm.min() and wl_nm.max() < 1600):
            raise ValueError(f"{fname}: wavelength range {wl_nm.min()}-{wl_nm.max()} nm unexpected")
        keep = wl_nm <= CUT_NM
        tr = d[keep, 1] / (100.0 if percent else 1.0)
        if not 0.4 < tr.max() <= 1.0:
            raise ValueError(f"{fname}: peak {tr.max()} not a fraction")
        ref = ((lam0 * 10.0, f"FORS User Manual 82.1 Table B.1 lambda0 ({MAN82})")
               if name in FORS_REF_OK[instrument] else None)
        bands.append(make_band(
            name, wl_nm[keep], tr, unit="nm", detector_type="photon", detector_source=ORIGINAL,
            components="filter", components_source=ORIGINAL, sources=f"{FORS + fname} | {FORS_PAGE}",
            svo_id=f"Paranal/{instrument}.{name}", ref_pivot=ref))
    note = (f"VLT {instrument} broad-band filter-only transmission curves from the ESO FORS filter page "
            f"({FORS_PAGE}); {detector_note} Curves cut at 1100 nm (end of the FORS 330-1100 nm range, "
            "manual P117 sec. 2.1) to drop the scans' spurious 1200 nm point and the >1.1 um leak of the "
            "u/g_HIGH lab scans. No detector QE, optics or atmosphere included (ESO publishes no QE table). "
            "z_GUNN and H_Alpha held (see recipe c_eso.py). "
            f"Detector: CCD, photon counting ({MAN117}).")
    build_instrument("Paranal", instrument, bands, notes=note, raw_files=raw, ground_based=True)


def _ocam_members(tar_path, wanted):
    out = {}
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            if m.name.startswith("/") or ".." in m.name or "/" in m.name or not m.isfile():
                raise ValueError(f"refusing tar member {m.name!r}")
            if m.name in wanted:
                out[m.name] = t.extractfile(m).read().decode("ascii")
    missing = set(wanted) - set(out)
    if missing:
        raise ValueError(f"tar lacks {sorted(missing)}")
    return out


def _ocam_centre(text):
    """Wavelength (nm) and the filter-centre scan (data column 2, pos_id 2 at x=y=0)."""
    lines = text.splitlines()
    xpos = next(l for l in lines if l.startswith("# x_pos")).split()[2:]
    ypos = next(l for l in lines if l.startswith("# y_pos")).split()[2:]
    if float(xpos[0]) != 0.0 or float(ypos[0]) != 0.0:
        raise ValueError("first scan is not at the filter centre")
    d = np.array([l.split() for l in lines if l.strip() and not l.startswith("#")], float)
    return d[:, 0], d[:, 1]


def _omegacam():
    tar = fetch(OCAM_URL, "Paranal")
    wanted = [f"{s}_modified_transmission_data{c}" for ss in OCAM.values() for s in ss for c in ("", "_with_ccd")]
    text = _ocam_members(tar, wanted)
    bands, comps = [], []
    for name, segs in OCAM.items():
        curves = {}
        for c in ("", "_with_ccd"):
            scans = [_ocam_centre(text[f"{s}_modified_transmission_data{c}"]) for s in segs]
            wl = scans[0][0]
            if any(not np.array_equal(wl, w) for w, _ in scans):
                raise ValueError(f"{name}: segment wavelength grids differ")
            if not (295 < wl.min() < 305 and 1195 < wl.max() < 1205):
                raise ValueError(f"{name}: wavelength range not 300-1200 nm")
            curves[c] = (wl, np.mean([t for _, t in scans], axis=0))
        member = " + ".join(f"{s}_modified_transmission_data_with_ccd" for s in segs)
        how = "centre scan" if len(segs) == 1 else "mean of the 4 segment-centre scans"
        bands.append(make_band(
            name, *curves["_with_ccd"], unit="nm", detector_type="photon", detector_source=ORIGINAL,
            components="filter+qe", components_source=ORIGINAL,
            sources=f"{OCAM_URL}#{member} ({how}) | {OCAM_PAGE}", svo_id=f"Paranal/OmegaCAM.{name}",
            ref_pivot=((OCAM_REF[name] * 10.0, f"OmegaCAM filter table Lambda ({OCAM_PAGE})")
                       if name in OCAM_REF else None)))
        fb = make_band(name, *curves[""], unit="nm", detector_type="photon", detector_source=ORIGINAL,
                       components="filter", components_source=ORIGINAL, sources=OCAM_URL)
        comps.append(Component("filter", fb.wavelength, fb.transmission, band=name,
                               source=f"{OCAM_URL}#{member.replace('_with_ccd', '')} ({how})"))
    note = ("VST OmegaCAM ugriz (SDSS, monolithic) and Johnson B, V (segmented) from ESO's "
            "OmegaCAM_transmission.tar.gz (2011): lab scans at the filter centre multiplied by the CCD QE "
            "(*_with_ccd members); B/V are the mean of their 4 segment-centre scans. Filter-only scans "
            "attached as filter components. No telescope mirrors, corrector or atmosphere included. "
            "Negative scan noise set to 0. Detector: 32 E2V 44-82 CCDs, photon counting (" + OCAM_PAGE + ").")
    build_instrument("Paranal", "OmegaCAM", bands, notes=note, components=comps, raw_files=[tar],
                     ground_based=True)


def run():
    _fors("FORS1", FORS1, "original FORS1 Bessell and Gunn set, optical-bench lab scans (M_*.txt); "
          "B/V/I_BESS existed in both FORS1 (+34/+35/+37) and FORS2 (+74/+75/+77), and the page does "
          "not say which unit was scanned. u/b/v/g_HIGH (FORS1 2007-2009) are in Paranal/FORS2.")
    _fors("FORS2", FORS2, "current FORS2 standard configuration (manual P117 Table 2.1); u/g_HIGH are lab "
          "scans in %T (converted to fractions), b/v_HIGH, R_SPECIAL, I_BESS are the *COLL_SR.ascii tables.")
    _omegacam()
