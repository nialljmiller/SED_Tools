"""NOT (Nordic Optical Telescope, La Palma) ALFOSC broad-band filters.

Filter identities verified against the NOT filter database
(https://www.not.iac.es/instruments/filters/filters.php) and the focus-offset page
(https://www.not.iac.es/instruments/filters/offsets.php, sections "ALFOSC" and
"FASU (ALFOSC/MOSCA)"):
  #7  U_Bes 362_60   "ALFOSC fixed set-up. Bessell U with rlb"
  #74 B_Bes 440_100  "ALFOSC fixed set-up"
  #75 V_Bes 530_80   "ALFOSC fixed set-up"
  #76 R_Bes 650_130  "ALFOSC fixed set-up"
  #12 i_int 797_157  "ALFOSC fixed set-up; interference i"
  #109 u', #120 g', #110 r', #111 i', #112 z' (SDSS) "FASU standard set-up" (FASU = filter
       unit of ALFOSC, listed under FASU (ALFOSC/MOSCA)).
The research rows' #8 (StanCam B) and #118-#122 (Stromgren H-beta / narrow bands, not SDSS)
were wrong and are not used.

Curves: https://www.not.iac.es/instruments/filters/curves-ascii/<id>.txt.
  #7, #74, #75, #76, #12: nm, transmission in percent (lab scans; #12 is in descending order).
  SDSS #109-#112, #120: Angstrom, fraction; 3-line header "filter #<id> <band> SDSS / grism ..
       / measured <date> in ALFOSC by Jyri Naranen" (in-situ scans with ALFOSC grisms).
Filter only: no telescope, CCD QE or atmosphere.  Detector: e2v CCD231-42 back-illuminated
CCD (photon counting; https://www.not.iac.es/instruments/detectors/CCD14/).
"""
import re

import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

BASE = "https://www.not.iac.es/instruments/filters/curves-ascii/"
DET = "https://www.not.iac.es/instruments/detectors/CCD14/"
# band: (NOT id, unit, percent?, header check)
FILTERS = {
    "U": (7, "nm", True, None), "B": (74, "nm", True, None), "V": (75, "nm", True, None),
    "R": (76, "nm", True, None), "i_int": (12, "nm", True, None),
    "up": (109, "Angstrom", False, "u' SDSS"), "gp": (120, "Angstrom", False, "g' SDSS"),
    "rp": (110, "Angstrom", False, "r' SDSS"), "ip": (111, "Angstrom", False, "i' SDSS"),
    # zp #112 held (manager): long-pass, scan ends at 10988 A with T=0.33; red edge needs CCD QE.
}
NAMES = {7: "U_Bes 362_60", 74: "B_Bes 440_100", 75: "V_Bes 530_80", 76: "R_Bes 650_130",
         12: "i_int 797_157", 109: "u'_SDSS 353_55", 120: "g'_SDSS 480_145",
         110: "r'_SDSS 618_148", 111: "i'_SDSS 771_171", 112: "z'_SDSS 832_LP"}


def _table(path):
    rows = []
    for ln in path.read_text(encoding="latin-1").splitlines():
        f = ln.split()
        if len(f) == 2 and all(re.fullmatch(r"[-+0-9.eE]+", x) for x in f):
            rows.append((float(f[0]), float(f[1])))
    return np.array(rows)


def run():
    bands, raw = [], []
    for name, (fid, unit, pct, head) in FILTERS.items():
        url = f"{BASE}{fid}.txt"
        p = fetch(url, "NOT", f"ALFOSC_{fid}.txt")
        raw.append(p)
        text = p.read_text(encoding="latin-1")
        if head:  # first line "filter #<id> [<band> SDSS]" (#111 omits the band label)
            first = text.splitlines()[0].strip()
            if not first.startswith(f"filter #{fid}") or first[len(f"filter #{fid}"):].strip() not in ("", head):
                raise ValueError(f"{p}: header {first!r} does not identify filter #{fid} {head}")
        a = _table(p)
        peak = a[:, 1].max()
        if pct != (peak > 1.5):
            raise ValueError(f"{p}: transmission scale unexpected (peak {peak})")
        bands.append(make_band(
            name, a[:, 0], a[:, 1] / (100.0 if pct else 1.0), unit=unit, detector_type="photon",
            detector_source=ORIGINAL, components="filter", components_source=ORIGINAL,
            sources=f"{url} | https://www.not.iac.es/instruments/filters/filters.php"))
    note = ("NOT ALFOSC imaging filters, filter-only transmission from the NOT filter database "
            "(curves-ascii/<id>.txt): " + "; ".join(f"{b}=#{v[0]} {NAMES[v[0]]}" for b, v in FILTERS.items())
            + ". Bessell/interference-i: lab scans in nm and percent (converted to Angstrom, /100). "
            "SDSS u'g'r'i'z' (FASU standard set-up, ALFOSC/MOSCA): in-situ scans in ALFOSC with "
            "grisms (J. Naranen 2004-05), Angstrom and fraction. No telescope, CCD QE (e2v "
            "CCD231-42, photon counting) or atmosphere. Caveats: the u' scan starts at 3176 A with "
            "T~0.1-0.19 (blue wing truncated by the measurement range). HELD: z' #112, a long-pass whose "
            "scan ends at 10988 A with T=0.33 (red edge set by the CCD QE, not included). Negative noise values set to 0. No published pivot (the "
            "database gives centre/FWHM only).")
    build_instrument("NOT", "ALFOSC", bands, notes=note, ground_based=True, raw_files=raw,
                     extra_provenance={"detector_evidence": DET,
                                       "filter_ids": {b: v[0] for b, v in FILTERS.items()}})
