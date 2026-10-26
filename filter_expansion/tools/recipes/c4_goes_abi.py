"""GOES-R series ABI (GOES-16/17/18/19): CWG-processed channel SRFs from NOAA NCC.

Source page: https://ncc.nesdis.noaa.gov/GOESR/ABI.php ("CWG processed PFM (version 3) (GOES 16)",
"FM2 (GOES 17, 60K)", "FM2 (GOES 17 IR, 81K)", "FM3 (GOES-18)", "FM4"), release note
2016_03_10_SRF_PFM_FM2_FM3_FM4_CWG_release.docx: vendor (Harris) SRFs, truncated by the GOES-R
Calibration Working Group to the innermost 0.1 % of the relative spectral response; three columns
wavelength [um], wavenumber [cm-1], "Relative spectral response".  Each curve is a channel
(detector-averaged) system relative spectral response, peak 1 (whole instrument: filter+optics+qe).

Naming follows SVO: GOES/ABI1 = "GOES16 PFM", ABI2 = "GOES17 FM2", ABI3 = "GOES18 FM3",
ABI4 = "GOES19 FM4" (Description PARAMs of the archived SVO VOTables in phase3/ia/raw/).
Ref pivots: nominal central wavelengths of https://www.goes-r.gov/mission/ABI-bands-quick-info.html.
"""
import re
import zipfile

import numpy as np
from fxlib import DOWNLOADS, Component, build_instrument, make_band  # noqa: F401

PAGE = "https://ncc.nesdis.noaa.gov/GOESR/ABI.php"
DOCS = "https://ncc.nesdis.noaa.gov/GOESR/docs/"
README = DOCS + "2016_03_10_SRF_PFM_FM2_FM3_FM4_CWG_release.docx"
CENTRAL_URL = "https://www.goes-r.gov/mission/ABI-bands-quick-info.html"
# goes-r.gov "ABI Bands Quick Guides": Central Wavelength (um) for bands 1-16.
CENTRAL_UM = [0.47, 0.64, 0.86, 1.37, 1.61, 2.24, 3.9, 6.2, 6.9, 7.3, 8.5, 9.6, 10.3, 11.2, 12.3, 13.3]

# (instrument, zip, member pattern, flight-model label, band suffix, channels)
SETS = [
    ("ABI1", "GOES-R_ABI_PFM_SRF_CWG_v3.zip", "GOES-R_ABI_PFM_SRF_CWG_ch{n}.txt", "PFM (GOES-16)", "", range(1, 17)),
    ("ABI2", "GOES-R_ABI_FM2_SRF_CWG.zip", "GOES-R_ABI_FM2_SRF_CWG_ch{n}.txt", "FM2 (GOES-17, 60K)", "", range(1, 17)),
    ("ABI2", "GOES-R_ABI_FM2_SRF_CWG_81K.zip", "GOES-R_ABI_FM02-81K_SRF_CWG_Ch{n:02d}.txt",
     "FM2 (GOES-17 IR, 81K)", "_81K", range(7, 17)),
    ("ABI3", "GOES-R_ABI_FM3_SRF_CWG.zip", "GOES-R_ABI_FM3_SRF_CWG_ch{n}.txt", "FM3 (GOES-18)", "", range(1, 17)),
    ("ABI4", "GOES-R_ABI_FM4_SRF_CWG.zip", "GOES-R_ABI_FM4_SRF_CWG_ch{n}.txt", "FM4 (GOES-19)", "", range(1, 17)),
]
SATELLITE = {"ABI1": "GOES-16 (ABI PFM)", "ABI2": "GOES-17 (ABI FM2)",
             "ABI3": "GOES-18 (ABI FM3)", "ABI4": "GOES-19 (ABI FM4)"}

DET_NOTE = (
    "DetectorType=energy is assumed: NOAA/CWG publish a 'Relative spectral response' "
    "(release note 2016-03-10; v2 readme 'relative response') without stating photon or energy "
    "weighting; weather-satellite SRFs are conventionally applied as the weighting of spectral "
    "radiance (energy convention), and SVO's archived GOES/ABI records also carry DetectorType 0 "
    "(energy counter).")


def read_member(zpath, member):
    """Return (header lines, array[N,3]) and check the units header."""
    with zipfile.ZipFile(zpath) as z:
        text = z.read(member).decode("ascii", errors="replace")
    head = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert len(head) == 2 and "micron" in head[1] and "cm-1" in head[1], (member, head)
    data = np.loadtxt(text.splitlines(), comments="#")
    assert data.ndim == 2 and data.shape[1] == 3, member
    return head, data


def run():
    raw = [DOWNLOADS / "GOES" / README.rsplit("/", 1)[1]]
    insts = {}
    for inst, zname, pattern, fm, suffix, chans in SETS:
        zpath = DOWNLOADS / "GOES" / zname
        raw.append(zpath)
        for n in chans:
            member = pattern.format(n=n)
            head, d = read_member(zpath, member)
            # Header code is the nominal wavelength in 0.01 um, e.g. "ch 047", "CH1035".
            code = int(re.search(r"(?i)ch\s*0*(\d+)\s+CWG", head[0]).group(1))
            central = CENTRAL_UM[n - 1]
            assert abs(code / 100 - central) / central < 0.03, (member, code, central)
            # Wavelength column vs wavenumber column must agree (1e4/um = cm-1).
            assert np.allclose(1e4 / d[:, 0], d[:, 1], rtol=2e-4), member
            assert d[:, 2].max() <= 1.0001 and d[:, 2].max() > 0.99, (member, d[:, 2].max())
            band = f"Ch{n:02d}{suffix}"
            src = f"{DOCS}{zname} ({member}) | {PAGE} | {README}"
            insts.setdefault(inst, []).append(make_band(
                band, d[:, 0], d[:, 2], unit="micron", detector_type="energy",
                detector_source="assumed", components="filter+optics+qe",
                components_source="original", sources=src, svo_id=f"GOES/{inst}.{band}",
                ref_pivot=(central * 1e4, f"{CENTRAL_URL} (ABI Band {n} central wavelength, nominal)")))
    for inst, bands in insts.items():
        extra = ""
        if inst == "ABI2":
            extra = (" Ch07_81K..Ch16_81K are the CWG 'FM2 (GOES 17 IR, 81K)' SRFs (GOES-R_ABI_FM2_SRF_CWG_81K.zip, "
                     "2019-09-04), i.e. the IR channels characterised for an 81 K focal-plane temperature; "
                     "Ch01..Ch16 are the 60 K set.")
        build_instrument("GOES", inst, bands, raw_files=[p for p in raw if p.exists()], ground_based=False, notes=(
            f"GOES-R series ABI on {SATELLITE[inst]}: CWG-processed channel spectral response functions from "
            f"NOAA NCC ({PAGE}), released 2016-03-10. Vendor (Harris) SRFs, truncated by the GOES-R CWG to the "
            "innermost 0.1 % of the relative spectral response to remove negative values (release note). "
            "Columns used: wavelength [um] (column 1; checked against wavenumber column 2) and relative "
            "spectral response (column 3, peak 1, not rescaled). Components: channel system response "
            "(filter+optics+qe as a whole; not split by the owner). " + DET_NOTE + extra +
            f" Ref pivots are the nominal goes-r.gov central wavelengths ({CENTRAL_URL})."))
