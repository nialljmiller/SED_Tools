"""IRAS 12/25/60/100 um: IPAC Explanatory Supplement Table II.C.5 (transmission, detector, system response)."""
import html
import re

import numpy as np
from fxlib import ORIGINAL, Component, build_instrument, make_band, DOWNLOADS

URL = "https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch2/tabC5.html"
EVID = ("https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch2/C4.html | "
        "https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch6/C3.html")


def parse(path):
    """Strip tags; after each 'NN um Band' marker read rows of 4 numbers (wl, T, det, system)."""
    text = html.unescape(re.sub(r"<[^>]*>", " ", path.read_text(errors="replace")))
    parts = re.split(r"(\d+)\s*\S{0,2}m\s+Band", text)
    out = {}
    for i in range(1, len(parts), 2):
        nums = [float(x) for x in re.findall(r"(?<![\w.])\d+\.\d+|(?<![\w.])\d+(?![\w.])", parts[i + 1])]
        nums = nums[: len(nums) // 4 * 4]
        out[parts[i]] = np.array(nums).reshape(-1, 4)
    return out


def run():
    # Local copy of round3/space/raw/IRAS/tabC5-ipac.html; SHA-256 equals a fresh fetch of URL.
    path = DOWNLOADS / "IRAS" / "tabC5-ipac.html"
    tables = parse(path)
    assert sorted(tables, key=int) == ["12", "25", "60", "100"], tables.keys()
    bands, comps = [], []
    for name, t in tables.items():
        assert np.all(np.diff(t[:, 0]) > 0), name
        bands.append(make_band(name, t[:, 0], t[:, 3], unit="micron", detector_type="energy",
                               detector_source=ORIGINAL, components="filter+optics+qe",
                               components_source=ORIGINAL, sources=f"{URL} | {EVID}",
                               normalize=True, svo_id=f"IRAS/IRAS.{name}mu"))
        comps.append(Component("filter+optics", t[:, 0] * 1e4, t[:, 1], band=name,
                               source=f"{URL} (Transmission: filters and field lenses)"))
        comps.append(Component("qe", t[:, 0] * 1e4, t[:, 2] / t[:, 2].max(), band=name,
                               source=f"{URL} (Relative Detector Response to constant energy input, peak-normalised)"))
    build_instrument("IRAS", "IRAS", bands, components=comps, raw_files=[path], notes=(
        "IRAS Explanatory Supplement Table II.C.5, 'Relative System Response' column as printed "
        "(including the 25 um 29.0 um row, whose detector value 0.19 is inconsistent with its "
        "system value 0.877; possible source typo, kept unchanged). DetectorType=energy: sect. II.C.4 "
        "Fig. II.C.9 caption gives the detector response 'to constant energy input', and sect. VI.C.3 "
        "eqs. VI.C.4-6 integrate f_nu*R over frequency with the nu*f_nu=const quoting convention. "
        "System response peak-normalised (normalize=True). Components: optical transmission (filters "
        "and field lenses, role filter+optics) and relative detector response (role qe, scaled to peak 1); "
        "pre-flight measurements of sample/spare parts."))
