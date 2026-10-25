"""ATLAS c and o: Tonry et al. (2018, PASP 130, 064505) Table 2 from the arXiv 1802.00879 e-print.

atlas_defn.tex \\label{tab:coband}: 5 nm sampling, columns (lambda nm, Tc) x3 then (lambda, To) x3.
Tablecomments: transmission includes 1.2 airmasses of atmosphere, mirror reflection and AR
coatings, filter transmission for the f/2 beam, and detector QE.  o is o(1) (Haleakala).
Table 3 gives a "pivot" (integer nm) defined as int(lambda T dln nu)/<T> = int T dl / int T/l dl.
That is not the pivot sqrt(int T l dl / int T/l dl), so it is checked here, not as ref_pivot.
"""
import re
import tarfile

import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

URL = "https://arxiv.org/e-print/1802.00879"
PAPER = "https://ui.adsabs.harvard.edu/abs/2018PASP..130f4505T"
TABLE3 = {"c": 533.0, "o": 679.0}  # Tonry 2018 Table 3 "pivot" [nm], c and o(1)


def _tex(path):
    with tarfile.open(path) as t:
        m = t.getmember("atlas_defn.tex")
        if m.name.startswith("/") or ".." in m.name or not m.isfile():
            raise ValueError("unsafe member")
        return t.extractfile(m).read().decode("utf-8", "replace")


def run():
    src = fetch(URL, "ATLAS", "tonry_1802.00879.eprint")
    tex = _tex(src)
    end = tex.index(r"\label{tab:coband}")
    start = tex.rindex(r"\begin{tabular}", 0, end)
    curves = {"c": [], "o": []}
    for line in tex[start:end].splitlines():
        cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
        if len(cells) == 12 and all(re.fullmatch(r"\d+(\.\d+)?", c) for c in cells):
            v = [float(c) for c in cells]
            for k in range(3):
                curves["c"].append((v[2 * k], v[2 * k + 1]))
                curves["o"].append((v[6 + 2 * k], v[7 + 2 * k]))
    t = np.trapezoid
    bands, table3_stat = [], {}
    for name, rows in curves.items():
        a = np.array(rows)
        if len(a) != 57:
            raise ValueError(f"ATLAS {name}: parsed {len(a)} rows, expected 57")
        b = make_band(name, a[:, 0], a[:, 1], unit="nm", detector_type="photon",
                      detector_source=ORIGINAL, components="filter+optics+qe+atmosphere",
                      components_source=ORIGINAL, sources=f"{URL} (atlas_defn.tex Table 2) | {PAPER}",
                      svo_id=f"Misc/Atlas.{ {'c': 'cyan', 'o': 'orange'}[name] }")
        l, tr = b.wavelength, b.transmission
        lm = float(t(tr, l) / t(tr / l, l))
        table3_stat[name] = {"computed_A": round(lm, 1), "table3_A": TABLE3[name] * 10,
                         "diff_pct": round(100 * (lm / (TABLE3[name] * 10) - 1), 3)}
        if abs(lm / (TABLE3[name] * 10) - 1) > 0.005:
            raise ValueError(f"ATLAS {name}: Table-3 statistic {lm:.1f} vs Table 3 {TABLE3[name]} nm")
        bands.append(b)
    note = ("Tonry et al. (2018) ATLAS c and o system bandpasses, Table 2 of the arXiv 1802.00879 "
            "source (atlas_defn.tex), 5 nm sampling. Includes 1.2 airmasses of atmosphere, mirror "
            "reflection (0.92 overcoated Al), AR coatings, filter transmission for the f/2 beam "
            "(from manufacturer curves, effective index 2; no in-situ measurement) and measured CCD "
            "QE. Photon convention (count-rate definition in the paper); SVO's DetectorType=energy "
            "label for Misc/Atlas is not inherited. o is o(1) (Haleakala); o(2) at Mauna Loa and the "
            "later southern units are not tabulated. Components are only plotted (raster), so none "
            "are published separately. No true pivot published: Table 3's 'pivot' is "
            "int T dl / int T/l dl, reproduced to <0.5 % (provenance).")
    build_instrument("ATLAS", "ATLAS", bands, notes=note, raw_files=[src], ground_based=True,
                     extra_provenance={"table3_stat_check": table3_stat})
