"""Gaia DR2 passbands (as used for the DR2 catalogue) and DR2 revised passbands (Evans+ 2018), ESA zips."""
import zipfile
from pathlib import PurePosixPath

import numpy as np
from fxlib import ORIGINAL, build_instrument, fetch, make_band

ZIPS = {
    "DR2": ("https://www.cosmos.esa.int/documents/29201/1645651/GaiaDR2_Passbands_ZeroPoints.zip/"
            "49cdce41-8eee-655d-7ed2-4e7a83598c1d", "GaiaDR2_Passbands_ZeroPoints.zip", "GaiaDR2_Passbands.dat"),
    "DR2r": ("https://www.cosmos.esa.int/documents/29201/1645651/GaiaDR2_Revised_Passbands_ZeroPoints.zip/"
             "54db454f-69cb-ea0c-15be-f1b1f597f191", "GaiaDR2_Revised_Passbands_ZeroPoints.zip",
             "GaiaDR2_RevisedPassbands.dat"),
}
PAGE = "https://www.cosmos.esa.int/web/gaia/iow_20180316"


def safe_extract(zpath, members):
    """Extract only ``members`` into a fresh sibling folder; refuse unsafe member names."""
    out = zpath.parent / zpath.stem
    with zipfile.ZipFile(zpath) as zf:
        for m in members:
            pp = PurePosixPath(m)
            if pp.is_absolute() or ".." in pp.parts or "\\" in m:
                raise ValueError(f"unsafe zip member {m!r}")
        if not out.exists():
            out.mkdir()
            for m in members:
                (out / PurePosixPath(m).name).write_bytes(zf.read(m))
    return [out / PurePosixPath(m).name for m in members]


def run():
    for inst, (url, zname, dat) in ZIPS.items():
        z = fetch(url, "GAIA", zname)
        files = safe_extract(z, [dat, "GaiaDR2_ReadMe"])
        a = np.loadtxt(files[0])
        bands = []
        for name, col, svo in (("G", 1, "G"), ("BP", 3, "Gbp"), ("RP", 5, "Grp")):
            bands.append(make_band(name, a[:, 0], a[:, col], unit="nm", detector_type="photon",
                                   detector_source=ORIGINAL, components="filter+optics+qe",
                                   components_source=ORIGINAL, sources=f"{url} | {PAGE}",
                                   sentinel=99.99, svo_id=f"GAIA/GAIA{'2' if inst == 'DR2' else '2r'}.{svo}"))
        which = ("passbands used to generate the released DR2 magnitudes and astrophysical parameters"
                 if inst == "DR2" else "revised passbands based on the latest nominal instrument knowledge")
        build_instrument("GAIA", inst, bands, raw_files=[z, *files], notes=(
            f"Gaia DR2 {which} (Evans et al. 2018, A&A 616, A4; ESA zip {zname}). Wavelength nm; "
            "99.99 = undefined (ReadMe note 1), dropped as sentinel. Whole-instrument (telescope, "
            "optics, CCD QE) photon-counting transmissivities for CCD photometry -> DetectorType=photon. "
            "Uncertainty columns not used. Distinct from GAIA/GAIA (EDR3/DR3, Gaia2m BP bright/faint)."))
