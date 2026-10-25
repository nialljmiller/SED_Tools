"""Pan-STARRS1 grizy + w: Tonry et al. (2012, ApJ 750, 99) table3 via CDS J/ApJ/750/99 (codex.csv).

table3 columns (ReadMe): lambda (nm), Open, gp1..wp1 capture cross-sections (m^2 e-/photon,
including the nominal 1.2 airmasses of the PS1 standard atmosphere), then Aero/Ray/Mol
atmospheric transmissions at 1.2 airmasses.  table4 gives the "pivot" wavelength (nm, integer)
labelled int(lambda A dln nu)/int(A dln nu); numerically it matches the log-mean wavelength
exp(int A ln(lambda) dlambda/lambda / int A dlambda/lambda) to <0.1 %, not the pivot.
"""
import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

BASE = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/750/99/"
URL = BASE + "table3.dat"
PAPER = "https://ui.adsabs.harvard.edu/abs/2012ApJ...750...99T"
COLS = {"g": (11, 16), "r": (17, 22), "i": (23, 28), "z": (29, 34), "y": (35, 40), "w": (41, 46)}
ATM = {"Aero": (47, 52), "Ray": (53, 58), "Mol": (59, 64)}


def _ref_pivots(table4):
    out = {}
    for line in table4.read_text().splitlines():
        parts = line.split("|")
        if len(parts) > 2 and parts[0].endswith("_P1"):
            out[parts[0][0]] = float(parts[2]) * 10.0  # nm -> Å
    return out


def run():
    t3 = fetch(URL, "PAN-STARRS")
    t4 = fetch(BASE + "table4.dat", "PAN-STARRS")
    readme = fetch(BASE + "ReadMe", "PAN-STARRS")
    rows = [ln for ln in t3.read_text().splitlines() if ln.strip()]
    wl = np.array([float(ln[0:4]) for ln in rows])
    piv = _ref_pivots(t4)
    bands = []
    for name, (a, b) in COLS.items():
        tr = np.array([float(ln[a:b]) for ln in rows])
        bands.append(make_band(
            name, wl, tr, unit="nm", detector_type="photon", detector_source=ORIGINAL,
            components="filter+optics+qe+atmosphere", components_source=ORIGINAL,
            sources=f"{URL} | {PAPER}", svo_id=f"PAN-STARRS/PS1.{name}"))
    # table4 "pivot" is really the log-mean wavelength exp(int A ln(l) dl/l / int A dl/l), not
    # a pivot, so it is not passed as ref_pivot; it is checked here instead (integer nm).
    t = np.trapezoid
    logmean = {}
    for b in bands:
        l, a = b.wavelength, b.transmission
        lm = float(np.exp(t(a * np.log(l) / l, l) / t(a / l, l)))
        logmean[b.name] = {"computed_A": round(lm, 1), "table4_A": piv[b.name],
                           "diff_pct": round(100 * (lm / piv[b.name] - 1), 3)}
        if abs(lm / piv[b.name] - 1) > 0.005:
            raise ValueError(f"PS1 {b.name}: log-mean {lm:.1f} vs table4 {piv[b.name]}")
    atm = np.prod([[float(ln[a:b]) for ln in rows] for a, b in ATM.values()], axis=0)
    note = ("Tonry et al. (2012) PS1 system capture cross-sections, CDS J/ApJ/750/99 table3, 1 nm "
            "sampling 380-1100 nm. Values are m^2 e-/photon (effective area including telescope, "
            "corrector, filter, detector QE and the nominal 1.2 airmasses of the PS1 standard "
            "atmosphere); kept unscaled (synthetic photometry is scale-free). Photon-counting CCDs. "
            "The table also gives the 1.2-airmass atmosphere (Aero x Ray x Mol) on the same grid, but "
            "SED_Tools add_component accepts only filter/optics/qe roles, so it is recorded in "
            "provenance only and no separate component is published. No published pivot: table4's "
            "'pivot' column is the log-mean wavelength, reproduced here to <0.1 % (provenance).")
    build_instrument("PAN-STARRS", "PS1", bands, notes=note, raw_files=[t3, t4, readme],
                     extra_provenance={"table4_logmean_check": logmean, "atmosphere_1p2_airmass": {
                         "source": URL + " columns Aero*Ray*Mol",
                         "wavelength_A": (wl * 10).tolist(), "transmission": np.round(atm, 5).tolist()}})
