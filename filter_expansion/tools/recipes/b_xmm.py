"""XMM-Newton Optical Monitor V B U UVW1 UVM2 UVW2 effective areas (HEASARC OM_response_081008.txt)."""
import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

URL = "https://heasarc.gsfc.nasa.gov/FTP/xmm/data/responses/OM/OM_response_081008.txt"
COLS = {"UVW2": 1, "UVM2": 2, "UVW1": 3, "U": 4, "B": 5, "V": 6}


def run():
    p = fetch(URL, "XMM")  # local copy of round3/space/raw/XMM (same file)
    a = np.loadtxt(p, comments="#")
    assert a.shape[1] == 7
    bands = [make_band(n, a[:, 0], a[:, COLS[n]], unit="nm", detector_type="photon",
                       detector_source=ORIGINAL, components="filter+optics+qe",
                       components_source=ORIGINAL, sources=URL, normalize=True, svo_id=f"XMM/OM.{n}")
             for n in ("V", "B", "U", "UVW1", "UVM2", "UVW2")]
    build_instrument("XMM", "OM", bands, raw_files=[p], notes=(
        "XMM-Newton OM effective area for all filters (HEASARC OM_response_081008.txt; header: nm, "
        "m^2 x 1000). Effective area includes telescope, filter and the photon-counting "
        "MCP-intensified detector, so DetectorType=photon. Peak-normalised (normalize=True), since "
        "the absolute area scale does not affect synthetic photometry. Zero-padded 100-700 nm grid "
        "kept as published (UV filters keep their red-leak tails)."))
