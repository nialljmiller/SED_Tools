"""Hipparcos Hp and Tycho BT/VT: Bessell & Murphy (2012, PASP 124, 140) table 2 (grok.csv)."""
import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

URL = "https://cdsarc.cds.unistra.fr/ftp/cats/J/PASP/124/140/table2.dat"
PAPER = "https://ui.adsabs.harvard.edu/abs/2012PASP..124..140B"


def run():
    path = fetch(URL, "Hipparcos")
    readme = fetch(URL.replace("table2.dat", "ReadMe"), "Hipparcos")
    cols = {"Hp": (0, 4, 5, 10), "BT": (11, 15, 16, 21), "VT": (22, 26, 27, 32)}
    curves = {k: ([], []) for k in cols}
    for line in path.read_text().splitlines():
        for band, (a, b, c, d) in cols.items():
            w, t = line[a:b].strip(), line[c:d].strip()
            if w and t:
                curves[band][0].append(float(w))
                curves[band][1].append(float(t))
    def band(name):
        wl, tr = curves[name]
        return make_band(name, np.array(wl), np.array(tr), unit="Angstrom", detector_type="photon",
                         detector_source=ORIGINAL, components="filter+optics+qe",
                         components_source=ORIGINAL, sources=f"{URL} | {PAPER}",
                         svo_id={"Hp": "Hipparcos/Hipparcos.Hp", "BT": "TYCHO/TYCHO.B", "VT": "TYCHO/TYCHO.V"}[name])
    note = ("Bessell & Murphy (2012) table 2 normalised photonic responses (CDS J/PASP/124/140), "
            "for photon-counting synthetic photometry. Wavelength column is 0.1 nm = Angstrom.")
    build_instrument("Hipparcos", "Hipparcos", [band("Hp")], notes=note, raw_files=[path, readme])
    build_instrument("Hipparcos", "Tycho", [band("BT"), band("VT")], notes=note, raw_files=[path, readme])
