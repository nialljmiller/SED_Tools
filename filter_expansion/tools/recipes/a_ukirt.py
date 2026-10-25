"""UKIRT WFCAM ZYJHK (the UKIDSS system): Hewett et al. (2006, MNRAS 367, 454) tables 2-6 (codex.csv).

The tar members Table02..06_online.dat are the system throughput curves (microns, throughput)
for 1.0 mm water vapour and 1.3 airmasses.  The paper (astro-ph/0601592, section 2) says they
include atmospheric absorption, three aluminium reflections (Hass & Hadley 1963), flat other
optics, the WFCAM filter scans corrected to the 120 K operating temperature, and the Hawaii-II
array QE of its Table 1; normalised to the photon count rate of standard stars.
"""
import tarfile

import numpy as np
from fxlib import ORIGINAL, Component, build_instrument, fetch, make_band

URL = "http://www.ukidss.org/technical/photom/hewett-ukidss-tables.tar"
EPRINT = "https://arxiv.org/e-print/astro-ph/0601592"
PAPER = "https://ui.adsabs.harvard.edu/abs/2006MNRAS.367..454H"
TABLES = {"Z": "Table02_online.dat", "Y": "Table03_online.dat", "J": "Table04_online.dat",
          "H": "Table05_online.dat", "K": "Table06_online.dat"}
# Hewett et al. 2006 Table 1 (array q.e.), transcribed from photpap_aph.tex in the e-print.
QE_UM = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.2]
QE = [0.53, 0.55, 0.54, 0.54, 0.54, 0.57, 0.62, 0.61]


def _members(tar_path):
    out = {}
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            if m.name.startswith("/") or ".." in m.name or not m.isfile():
                raise ValueError(f"refusing tar member {m.name!r}")
            if m.name in TABLES.values():
                out[m.name] = t.extractfile(m).read().decode("ascii")
    return out


def run():
    tar = fetch(URL, "UKIRT")
    eprint = fetch(EPRINT, "UKIRT", "hewett_astro-ph_0601592.eprint")
    text = _members(tar)
    bands = []
    for name, member in TABLES.items():
        body = text[member]
        if "1.0mm water vapour, 1.3 airmass" not in body or f"WFCAM {name} filter" not in body:
            raise ValueError(f"{member}: unexpected header")
        data = np.array([ln.split() for ln in body.splitlines()
                         if ln.strip() and not ln.startswith("#")], float)
        bands.append(make_band(
            name, data[:, 0], data[:, 1], unit="micron", detector_type="photon",
            detector_source=ORIGINAL, components="filter+optics+qe+atmosphere",
            components_source=ORIGINAL, sources=f"{URL}#{member} | {PAPER}",
            svo_id=f"UKIRT/UKIDSS.{name}"))
    qe = Component("qe", np.array(QE_UM) * 1e4, np.array(QE), band="*",
                   source=f"Hewett et al. 2006 Table 1 (array q.e.), {EPRINT} photpap_aph.tex")
    note = ("Hewett et al. (2006) UKIDSS/WFCAM ZYJHK system throughput, tables 2-6 of "
            "hewett-ukidss-tables.tar, for the default 1.0 mm water vapour and 1.3 airmasses. "
            "Includes atmosphere, 3 Al mirror reflections, other optics (flat), filter scans "
            "corrected to 120 K, and Hawaii-II QE; normalised to the measured photon count rate "
            "(photon-counting HgCdTe arrays). These are the corrected total-system curves, not "
            "the room-temperature filter-only scans. Single filter per band (4 per band in "
            "WFCAM; <=0.01 mag differences per the paper). Component: array QE (paper Table 1, "
            "8 points, constant beyond 2.2 um per the paper). No published pivot wavelength "
            "(Hewett Table 7 lambda_eff is Vega-weighted), so none is checked.")
    build_instrument("UKIRT", "WFCAM", bands, notes=note, components=[qe],
                     raw_files=[tar, eprint], ground_based=True)
