"""Spitzer IRAC I1-I4, MIPS 24/70/160 and IRS peak-up blue/red (IRSA/SSC calibration files)."""
import re

import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

IRAC = "https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles/spectralresponse/201125ch{}trans_full.txt"
MIPS = "https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/MIPSfiltsumm.txt"
MIPS_HB = "https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/6/"
IRS = "https://irsa.ipac.caltech.edu/data/SPITZER/docs/files/spitzer/{}PUtrans.txt"


def two_col(path, comment):
    wl, tr = [], []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith(comment):
            continue
        a, b = s.split()[:2]
        wl.append(float(a))
        tr.append(float(b))
    return np.array(wl), np.array(tr)


def mips_columns(path):
    """Three (lambda, response) pairs side by side; shorter columns end early, so
    assign each token to a column by the character position where it ends."""
    lines = path.read_text().splitlines()
    num = re.compile(r"^-?\d+(\.\d*)?([eE][-+]?\d+)?$")
    data = [l for l in lines if l.strip() and all(num.match(t) for t in l.split())]
    full = next(l for l in data if len(l.split()) == 6)
    ends = []
    pos = 0
    for tok in full.split():
        pos = full.index(tok, pos) + len(tok)
        ends.append(pos)
    cols = [[] for _ in range(6)]
    for l in data:
        pos = 0
        for tok in l.split():
            pos = l.index(tok, pos) + len(tok)
            k = int(np.argmin([abs(pos - e) for e in ends]))
            cols[k].append(float(tok))
    out = {}
    for name, k in (("24", 0), ("70", 2), ("160", 4)):
        assert len(cols[k]) == len(cols[k + 1]), name
        out[name] = (np.array(cols[k]), np.array(cols[k + 1]))
    return out


def run():
    # IRAC: header "microns  electrons / photon" -> photon; whole-system response.
    paths, bands = [], []
    for ch in (1, 2, 3, 4):
        url = IRAC.format(ch)
        p = fetch(url, "Spitzer")
        paths.append(p)
        wl, tr = two_col(p, "#")
        bands.append(make_band(f"I{ch}", wl, tr, unit="micron", detector_type="photon",
                               detector_source=ORIGINAL, components="filter+optics+qe",
                               components_source=ORIGINAL, sources=url, svo_id=f"Spitzer/IRAC.I{ch}"))
    build_instrument("Spitzer", "IRAC", bands, raw_files=paths, notes=(
        "IRAC full-array average spectral response curves (IRSA/SSC, 201125ch*trans_full). File header "
        "gives microns and electrons/photon, so DetectorType=photon; the response includes telescope, "
        "optics, filters and detector QE. Native e-/photon scale kept (not normalised)."))

    # MIPS: handbook sect. 6 "R is the intrinsic spectral response of the detector, in units of e- per
    # unit energy"; the summary file header gives only 'lambda response'. -> energy.
    p = fetch(MIPS, "Spitzer")
    cols = mips_columns(p)
    bands = [make_band(n, wl, tr, unit="micron", detector_type="energy", detector_source=ORIGINAL,
                       components="filter+qe", components_source=ORIGINAL, sources=f"{MIPS} | {MIPS_HB}",
                       normalize=True, svo_id=f"Spitzer/MIPS.{n}mu") for n, (wl, tr) in cols.items()]
    build_instrument("Spitzer", "MIPS", bands, raw_files=[p], notes=(
        "MIPS imaging filter bandpasses (MIPSfiltsumm.txt, 21 March 2000), combined detector plus filter "
        "spectral response. The MIPS Instrument Handbook (sect. 6) defines R in electrons per unit "
        "energy, so DetectorType=energy (the file header itself says only 'response'). Relative "
        "response: peak-normalised (normalize=True). 70 um is the NB-labelled column of the file. "
        "Telescope optics not stated as included."))

    # IRS peak-up: header "RQExT ... multiply incident photons/second ... get electrons/second" -> photon.
    paths, bands = [], []
    for name, col in (("PU_blue", "blue"), ("PU_red", "red")):
        url = IRS.format(col)
        p = fetch(url, "Spitzer")
        paths.append(p)
        wl, tr = two_col(p, ";")
        bands.append(make_band(name, wl, tr, unit="micron", detector_type="photon",
                               detector_source=ORIGINAL, components="filter+qe",
                               components_source=ORIGINAL, sources=url, svo_id=f"Spitzer/IRS.{col}"))
    build_instrument("Spitzer", "IRS", bands, raw_files=paths, notes=(
        "IRS blue/red peak-up RQExT curves at 2.0 V (IRSA/SSC; same as SMART). Header: RQExT = detector "
        "responsive QE (QE x photoconductive gain) x filter transmission; multiplying photons/s gives "
        "electrons/s, so DetectorType=photon. Telescope optics not included. Native scale kept; "
        "negative noise samples set to 0."))
