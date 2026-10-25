"""J-PLUS (OAJ JAST80 / T80Cam) 12 bands: j-plus.es FullEfficiency .tab files (codex.csv).

STATUS: written against the gSDSS header recorded in round2/grok/sources.jsonl; www.j-plus.es timed
out on 2026-10-25, so this recipe has NOT been run.  The other 11 URLs follow the verified
gSDSS pattern and are unverified.  Each file must contain a header (authors Varela, Marin,
Iniguez, 2016-05-18): T = CCD QE x filter x optics x atmosphere, Lambda in Angstrom,
T in photons (0-1); atmosphere model from Apache Point measurements (not OAJ).
"""
import re

import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

BASE = "https://www.j-plus.es/doc/general_page/10/"
PAGE = "https://www.j-plus.es/survey/instrumentation"
BANDS = ["uJAVA", "J0378", "J0395", "J0410", "J0430", "gSDSS", "J0515", "rSDSS", "J0660",
         "iSDSS", "J0861", "zSDSS"]
NUM = re.compile(r"^\s*[-+]?\d+(\.\d*)?([eE][-+]?\d+)?\s+[-+]?\d+(\.\d*)?([eE][-+]?\d+)?\s*$")


def _parse(path):
    text = path.read_text(errors="replace")
    header = "\n".join(ln for ln in text.splitlines() if not NUM.match(ln))
    for need in ("Angstrom", "photon"):
        if need.lower() not in header.lower():
            raise ValueError(f"{path.name}: header lacks {need!r}; check units/convention by hand")
    data = np.array([ln.split() for ln in text.splitlines() if NUM.match(ln)], float)
    if len(data) < 100:
        raise ValueError(f"{path.name}: only {len(data)} numeric rows (truncated download?)")
    return data, "apache point" in header.lower()


def run():
    bands, raw, apo = [], [], {}
    for name in BANDS:
        url = f"{BASE}JPLUS_{name}_FullEfficiency.tab"
        path = fetch(url, "OAJ", timeout=60)
        data, apo[name] = _parse(path)
        raw.append(path)
        bands.append(make_band(name, data[:, 0], data[:, 1], unit="Angstrom", detector_type="photon",
                               detector_source=ORIGINAL, components="filter+optics+qe+atmosphere",
                               components_source=ORIGINAL, sources=f"{url} | {PAGE}",
                               svo_id=f"OAJ/JPLUS.{name}"))
    if not all(apo.values()):
        raise ValueError(f"atmosphere note missing for {[k for k, v in apo.items() if not v]}")
    note = ("J-PLUS full-efficiency curves (j-plus.es FullEfficiency .tab, Varela, Marin & Iniguez "
            "2016-05-18): CCD QE x filter (measured by CEFCA at the filter centre) x optics (mirror "
            "reflectivity, field corrector, T80Cam window) x atmosphere. The atmosphere model is from "
            "Apache Point measurements, not OAJ. Lambda in Angstrom; T in photons (photon-counting "
            "CCD). Filter-only scans (JPLUS_<band>.dat, no header, nm inferred) are not used. "
            "No published pivot checked.")
    build_instrument("OAJ", "JPLUS", bands, notes=note, raw_files=raw, ground_based=True)
