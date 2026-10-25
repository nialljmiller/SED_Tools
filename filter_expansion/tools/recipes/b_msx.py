"""MSX SPIRIT III bands A, C, D, E: IRSA relative spectral response tables (PSC v1.2, Egan+ 1999)."""
import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

URL = "https://irsa.ipac.caltech.edu/data/MSX/docs/rsr/msx_rsr_{}.tbl"
DESC = "https://irsa.ipac.caltech.edu/data/MSX/imageDescriptions.htm"


def run():
    paths, bands = [], []
    for b in "ACDE":
        url = URL.format(b.lower())
        p = fetch(url, "MSX")  # local copies of round3/space/raw/MSX
        paths.append(p)
        head = [l for l in p.read_text().splitlines() if l.startswith("|")]
        assert "microns" in head[2], head
        a = np.loadtxt(p, comments=("\\", "|"))
        bands.append(make_band(b, a[:, 0], a[:, 1], unit="micron", detector_type="energy",
                               detector_source=ORIGINAL, components="filter+optics+qe",
                               components_source=ORIGINAL, sources=f"{url} | {DESC}",
                               normalize=True, svo_id=f"MSX/MSX.{b}"))
    build_instrument("MSX", "MSX", bands, raw_files=paths, notes=(
        "MSX SPIRIT III relative spectral response (RSR) curves, IRSA IPAC tables from the MSX Point "
        "Source Catalog V1.2 (Egan, Price et al. 1999); whole-system relative response. "
        "DetectorType=energy although the Si:As BIB detectors are photoconductors: the RSR is applied "
        "as an energy weighting. IRSA imageDescriptions gives images in in-band radiance (W m-2 sr-1) "
        "and radiance-to-Jy/sr factors for S_nu ~ nu^-1 (A 7.133e12, C 2.863e13, D 3.216e13, E 2.476e13). "
        "lambda_iso/(c*Int RSR/lambda dlambda) from these tables reproduces C/D/E to 4 digits and A to "
        "1.1%; the photon-weighted alternative is off by 0.3-4.6%. Peak-normalised (normalize=True; "
        "already 1). B1/B2 held: no RSR table in the IRSA rsr directory."))
