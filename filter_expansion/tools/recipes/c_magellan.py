"""Magellan (Las Campanas) FourStar and MegaCam.  Facility "Magellan" (not LasCumbres/LCO).

FourStar (Baade 6.5 m, 4 x Teledyne HAWAII-2RG arrays, photon counting;
https://instrumentation.obs.carnegiescience.edu/FourStar/quick_facts.html).
Curves: the team's own synthetic-photometry inputs in OPTICS/curves/HST_STD/<band>.txt
(Angstrom, fraction), written by OPTICS/curves/plot.sm from the files linked on
OPTICS/filters.html.  plot.sm states the <band>_cam_optics.txt files are "the filter
transmissions * 0.995^16, no detector, no atmos, no telescope" (camera AR-surface losses),
and applies the glass-absorption roll-off (1.714 - 0.380 lambda_um) to Ks and NB209; the
narrow bands FS118/FS209 (*_convolve.txt) are already convolved with their J2/Ks blockers.
The recipe re-derives every HST_STD curve from the linked *_cam_optics / *_convolve files to
check that.  Components "filter+optics" (product only; no separate curves published).
Published pivots: HST_STD/output.txt (computed by the team's fwhm.sm; that script may have
included an ATRAN sky, so agreement is only expected to ~1 %).

MegaCam (Magellan Clay 6.5 m, 36 e2v CCD42-90 back-illuminated CCDs, photon counting;
McLeod et al. 2015, PASP 127, 366; https://lweb.cfa.harvard.edu/~bmcleod/Megacam/).
Curves: Filters/filterthroughput.mag{u,g,r,i,z} (tab separated, Angstrom).  Per the Filters
page: T = filter, QE = CCD QE, EXTINCT = atmosphere at 1.2 airmasses (transmission),
THROUGHPUT = mirrors (R) x corrector/window surfaces (SOLGEL, WINDOW), TOTAL = T*QE*EXTINCT*
THROUGHPUT (verified to 5e-7).  Published curve = TOTAL ("filter+optics+qe+atmosphere");
components: filter (T, per band; identical to the filter-only mag{g,r,i,z}.tab), optics
(THROUGHPUT) and qe (QE), both identical in all five tables.  SED_Tools instrument components
cannot hold an atmosphere, so EXTINCT is kept in provenance (as for PAN-STARRS).
"""
import numpy as np
from fxlib import ORIGINAL, Component, build_instrument, clean_curve, fetch, make_band

FS = "https://instrumentation.obs.carnegiescience.edu/FourStar/OPTICS/curves/"
FS_BANDS = {"J": "J", "H": "H", "Ks": "Ks", "J1": "J1", "J2": "J2", "J3": "J3",
            "Hshort": "Hshort", "Hlong": "Hlong", "NB118": "FS118", "NB209": "FS209"}
# HST_STD/output.txt "pivot wavelength" (microns), order J H J1 J2 J3 Hshort Hlong Ks.
FS_PIVOT_UM = {"J": 1.24, "H": 1.617, "J1": 1.055, "J2": 1.145, "J3": 1.288,
               "Hshort": 1.552, "Hlong": 1.704, "Ks": 2.146}

MC = "https://lweb.cfa.harvard.edu/~bmcleod/Megacam/Filters/"
MC_COLS = ["WAVE", "FLUX", "T", "QE", "NU", "STARToverNU", "EXTINCT", "THROUGHPUT", "R",
           "SOLGEL", "WINDOW", "TOTAL"]


def _two_col(path):
    rows = []
    for ln in path.read_text(encoding="latin-1").splitlines():
        f = ln.split()
        if len(f) >= 2:
            try:
                rows.append((float(f[0]), float(f[1])))
            except ValueError:
                pass
    return np.array(rows)


def _rolloff(wl_a):
    return 1.714 - 0.380 * wl_a / 1e4


def fourstar():
    bands, raw, check = [], [], {}
    for name, stem in FS_BANDS.items():
        p = fetch(FS + f"HST_STD/{stem}.txt", "Magellan", f"FourStar_HST_STD_{stem}.txt")
        a = _two_col(p)
        raw.append(p)
        # Re-derive from the files linked on filters.html.
        if stem.startswith("FS"):
            src = fetch(FS + f"{stem}_convolve.txt", "Magellan", f"FourStar_{stem}_convolve.txt")
            s = _two_col(src)
            wl, tr = s[:, 0] * 10, s[:, 1]
            if stem == "FS209":
                tr = tr / 100 * _rolloff(wl)
        else:
            src = fetch(FS + f"{stem}_cam_optics.txt", "Magellan", f"FourStar_{stem}_cam_optics.txt")
            s = _two_col(src)
            wl, tr = s[:, 0], s[:, 1] * (_rolloff(s[:, 0]) if stem == "Ks" else 1.0)
        raw.append(src)
        dev = float(np.max(np.abs(np.interp(a[:, 0], wl, tr) - a[:, 1])))
        if dev > 1e-3:
            raise ValueError(f"FourStar {name}: HST_STD differs from re-derived curve by {dev}")
        check[name] = {"max_abs_diff_vs_rederived": round(dev, 6), "linked_file": src.name}
        ref = (FS_PIVOT_UM[name] * 1e4, "FourStar OPTICS/curves/HST_STD/output.txt (fwhm.sm)") \
            if name in FS_PIVOT_UM else None
        bands.append(make_band(
            name, a[:, 0], a[:, 1], unit="Angstrom", detector_type="photon",
            detector_source=ORIGINAL, components="filter+optics", components_source=ORIGINAL,
            sources=f"{FS}HST_STD/{stem}.txt | {FS}{src.name.split('_', 1)[1]} | {FS}plot.sm",
            svo_id=None if stem.startswith("FS") else f"LCO/FourStar.{stem}", ref_pivot=ref))
    note = ("Magellan Baade FourStar (4 x HAWAII-2RG, photon counting). Curves = Carnegie "
            "OPTICS/curves/HST_STD/<band>.txt, the team's synthetic-photometry inputs: filter "
            "transmission x 0.995^16 camera AR-surface losses (plot.sm: 'no detector, no atmos, no "
            "telescope'), with the glass-absorption roll-off (1.714-0.380 lambda_um) applied to Ks and "
            "NB209; NB118/NB209 (FS118/FS209) include their J2/Ks blocking filters. Each curve was "
            "re-derived from the *_cam_optics.txt / *_convolve.txt files linked on filters.html "
            "(max |diff| <= 1e-3, rounding). Components filter+optics (product only). No detector QE, "
            "telescope or atmosphere. Reference pivots from HST_STD/output.txt (team fwhm.sm; may "
            "include an ATRAN sky). SVO ids (LCO/FourStar.*) are unverified leads.")
    build_instrument("Magellan", "FourStar", bands, notes=note, ground_based=True, raw_files=raw,
                     extra_provenance={"rederivation_check": check})


def _megacam_table(path):
    lines = path.read_text(encoding="latin-1").splitlines()
    if lines[0].split("\t") != MC_COLS:
        raise ValueError(f"{path}: unexpected columns {lines[0]!r}")
    rows = []
    for ln in lines[2:]:
        f = ln.split("\t") + [""] * len(MC_COLS)
        rows.append([float(x) if x.strip() else np.nan for x in f[:len(MC_COLS)]])
    a = np.array(rows)
    return {c: a[:, i] for i, c in enumerate(MC_COLS)}


def megacam():
    bands, comps, raw, ref = [], [], [], None
    for b in "ugriz":
        url = MC + f"filterthroughput.mag{b}"
        p = fetch(url, "Magellan", f"MegaCam_filterthroughput.mag{b}")
        raw.append(p)
        t = _megacam_table(p)
        prod = t["T"] * t["QE"] * t["EXTINCT"] * t["THROUGHPUT"]
        if np.nanmax(np.abs(prod - t["TOTAL"])) > 1e-5:
            raise ValueError(f"MegaCam {b}: TOTAL != T*QE*EXTINCT*THROUGHPUT")
        if b != "u":  # filter-only table published separately for g,r,i,z
            fo_url = MC + f"mag{b}.tab"
            fo = fetch(fo_url, "Magellan", f"MegaCam_mag{b}.tab")
            raw.append(fo)
            f = _two_col(fo)
            m = (t["WAVE"] >= f[0, 0]) & (t["WAVE"] <= f[-1, 0])
            if np.max(np.abs(np.interp(t["WAVE"][m], f[:, 0], f[:, 1]) - t["T"][m])) > 1e-6:
                raise ValueError(f"MegaCam {b}: T column differs from mag{b}.tab")
        if ref is None:
            ref = t
        else:
            for c in ("QE", "EXTINCT", "THROUGHPUT"):
                if not np.allclose(t[c], ref[c]):
                    raise ValueError(f"MegaCam {b}: {c} differs between tables")
        bands.append(make_band(
            b, t["WAVE"], t["TOTAL"], unit="Angstrom", detector_type="photon",
            detector_source=ORIGINAL, components="filter+optics+qe+atmosphere",
            components_source=ORIGINAL, sources=f"{url} | {MC}"))
        fw, ft, rep = clean_curve(t["WAVE"], t["T"])
        if rep.note():
            bands[-1].clean_note = "; ".join(x for x in (bands[-1].clean_note, "filter component: " + rep.note()) if x)
        comps.append(Component("filter", fw, ft, band=b, source=url + " column T"))
    url_g = MC + "filterthroughput.magg"
    comps.append(Component("optics", ref["WAVE"], ref["THROUGHPUT"], source=url_g + " column THROUGHPUT"))
    comps.append(Component("qe", ref["WAVE"], ref["QE"], source=url_g + " column QE"))
    note = ("Magellan Clay MegaCam (36 e2v CCD42-90, photon counting; McLeod et al. 2015 PASP 127 "
            "366). Published curve = TOTAL column of Filters/filterthroughput.mag{u,g,r,i,z} = filter T "
            "x CCD QE x telescope/corrector throughput (2 Al mirrors, sol-gel and window surfaces) x "
            "atmosphere at 1.2 airmasses (product verified to 5e-7). Components: filter (T column; "
            "equals the filter-only mag{g,r,i,z}.tab), optics (THROUGHPUT) and qe (QE), the last two "
            "identical in all tables. The 1.2-airmass atmosphere (EXTINCT column) is recorded in "
            "provenance only (SED_Tools instrument components are filter/optics/qe). Filter T peaks "
            "slightly above 1 in z (1.0014, measurement noise, kept); small negative T values (noise) set to 0 in the filter components. No published pivot. MMT-era "
            "MegaCam curves on the same page are not included.")
    build_instrument("Magellan", "MegaCam", bands, notes=note, components=comps, ground_based=True,
                     raw_files=raw, extra_provenance={"atmosphere_1p2_airmass": {
                         "source": url_g + " column EXTINCT",
                         "wavelength_A": ref["WAVE"].tolist(),
                         "transmission": np.round(ref["EXTINCT"], 6).tolist()}})


def run():
    fourstar()
    megacam()
