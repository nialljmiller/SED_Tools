"""ERIS/NIX optics from ESO's ``nix-filters.zip``.

ESO's ERIS instrument page links ``nix-filters.zip`` as "the data used to
create" its filter plot.  Besides the filters (NJM's published, filter-only
NIX curves) it holds the other optical elements:

* ``cameras1-13masJHK.dat``, ``cameras2-27mas-JHK.dat``, ``cameras3-13mas-LM.dat``
  -- the three NIX camera barrels (metres, fraction);
* ``Dichroic.txt`` -- the AO dichroic "installed in June 2021" that passes
  the infrared to the instruments (nm, percent, as its header says);
* ``ND.txt`` -- the neutral-density filter in the pupil wheel (no header;
  nm and percent: as a percentage it reproduces the ND attenuations of the
  ERIS User Manual Table 14 -- J 6.00, H 5.50, Ks 5.06, Lp 4.57, Mp 4.50 mag
  -- to 0.01 mag, which is why its values exceed 1);
* ``filter_Blocking.dat`` -- the pupil-wheel blocking filter (not offered);
* ``ISSM.dat`` -- the calibration unit's integrating-sphere selector mirror
  (User Manual sec. 2.3), which is inserted only for internal flats and is
  not in the on-sky beam.

The registered ``optics`` of each band is the dichroic times the camera that
band is imaged with by default (13 mas JHK below 2.5 um, 13 mas LM above;
ESO's valid-configuration table).  The 27 mas JHK camera product is written
alongside, unregistered.  Optional pupil-wheel elements (ND, Blocking,
masks) and the ISSM are not included.  No detector QE, telescope mirror or
ERIS warm-optics curve is in the archive, so ``optics`` is partial.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, Union

import numpy as np

from ..filters.compose import OPTICS
from ._attach import (common_product, fresh_manifest, published_bands, register, with_earlier_note,
                      write_unregistered)
from ._original import Fetcher, clean_curve

FACILITY, INSTRUMENT = "ERIS", "NIX"
ZIP_URL = "https://www.eso.org/sci/facilities/paranal/instruments/eris/img/nix-filters.zip"
PAGE = "https://www.eso.org/sci/facilities/paranal/instruments/eris/inst.html"
MANUAL = "https://www.eso.org/sci/facilities/paranal/instruments/eris/doc/ERIS_User_Manual_v114.1.pdf"

CAMERAS = {
    "13mas-JHK": "nix-filters/cameras1-13masJHK.dat",
    "27mas-JHK": "nix-filters/cameras2-27mas-JHK.dat",
    "13mas-LM": "nix-filters/cameras3-13mas-LM.dat",
}
DICHROIC = "nix-filters/Dichroic.txt"

#: Published band -> default camera (ESO valid-configuration table: any filter
#: below 2.5 um with the 13 or 27 mas JHK camera, above 2.5 um with 13 mas LM).
SHORT = ("J", "H", "Ks", "Pa-b", "Fe-II", "H2-cont", "H2-1-0S", "Br-g", "K-peak", "IB-2p42", "IB-2p48")
LONG = ("Short-Lp", "L-Broad", "Lp", "Mp", "Br-a-cont", "Br-a")
DEFAULT_CAMERA: Dict[str, str] = {**{b: "13mas-JHK" for b in SHORT}, **{b: "13mas-LM" for b in LONG}}

NOTES = (
    "Published curves are ESO's NIX filter scans (filter only). Optics component: ERIS AO dichroic "
    "(Dichroic.txt, 'installed in June 2021', nm and percent) x the NIX camera barrel each band uses "
    "by default: 13 mas JHK camera (cameras1) for filters below 2.5 um, 13 mas LM camera (cameras3) "
    "above (ESO's valid-configuration table, {page}). Observations with the 27 mas JHK camera "
    "(cameras2) have ~4-5% lower camera throughput (band-averaged ratio 0.953 J, 0.957 H, 0.962 Ks); "
    "that product is kept unregistered as components/optics/dichroic_camera27mas_JHK.dat. Not "
    "included: ND.txt (pupil-wheel neutral-density filter, nm and percent -- its values exceed 1 "
    "because they are percentages; read so, they reproduce the ND attenuations of the ERIS User "
    "Manual Table 14 to 0.01 mag), filter_Blocking.dat (pupil-wheel blocking filter, not currently "
    "offered) and ISSM.dat (the calibration unit's integrating-sphere selector mirror, User Manual "
    "sec. 2.3, not in the on-sky beam). The archive has no detector QE, telescope mirror or ERIS "
    "warm-optics curve, so 'optics' covers only the dichroic and camera, and no filter+optics+qe "
    "product exists. The camera files start and end with explicit zeros and every factor is zero "
    "outside its tabulated range. Cite Davies et al. 2023, A&A 674, A207."
)


def member_curve(zf: zipfile.ZipFile, name: str, *, unit_scale: float, percent: bool):
    """(Angstrom, fraction) of a zip member, comment lines skipped."""
    rows = []
    for line in zf.read(name).decode("ascii").splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0].startswith("#"):
            continue
        rows.append((float(parts[0]), float(parts[1])))
    data = np.array(rows)
    tr = data[:, 1] / (100.0 if percent else 1.0)
    if tr.max() > 1.0:
        raise ValueError(f"{name}: transmission exceeds 1 after unit conversion")
    wl, tr, _ = clean_curve(data[:, 0] * unit_scale, tr)
    return wl, tr


def read_zip(raw: bytes):
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = set(zf.namelist())
        missing = sorted((set(CAMERAS.values()) | {DICHROIC}) - names)
        if missing:
            raise ValueError(f"nix-filters.zip lacks {missing}")
        cameras = {k: member_curve(zf, v, unit_scale=1e10, percent=False) for k, v in CAMERAS.items()}
        header = zf.read(DICHROIC).decode("ascii")
        if "Wavelength [nm]" not in header or "Transmission [%]" not in header:
            raise ValueError("Dichroic.txt header no longer states nm and percent")
        dichroic = member_curve(zf, DICHROIC, unit_scale=10.0, percent=True)
    return cameras, dichroic


def build(filter_root: Union[str, Path], *, session=None, **_ignored) -> Path:
    inst_dir = Path(filter_root) / FACILITY / INSTRUMENT
    present = published_bands(inst_dir)
    cameras, dichroic = read_zip(Fetcher(session).get(ZIP_URL))

    notes = with_earlier_note(inst_dir, NOTES.format(page=PAGE))
    manifest = fresh_manifest(inst_dir, FACILITY, INSTRUMENT)
    by_camera: Dict[str, list] = {}
    for band in sorted(present):
        camera = DEFAULT_CAMERA.get(band)
        if camera is None:
            raise ValueError(f"ERIS/NIX band {band!r} has no camera assignment")
        by_camera.setdefault(camera, []).append(band)
    for camera, bands in sorted(by_camera.items()):
        wl, tr = common_product([dichroic, cameras[camera]])
        register(manifest, inst_dir, OPTICS, wl, tr, bands=bands,
                 name=f"dichroic_camera{camera.replace('-', '_')}",
                 source=f"{ZIP_URL}: Dichroic.txt (nm, %) x {CAMERAS[camera].split('/')[-1]} (m, fraction)")
    wl, tr = common_product([dichroic, cameras["27mas-JHK"]])
    write_unregistered(inst_dir, OPTICS, wl, tr, name="dichroic_camera27mas_JHK",
                       source=f"{ZIP_URL}: Dichroic.txt (nm, %) x cameras2-27mas-JHK.dat (m, fraction); "
                              "alternative to the 13 mas JHK camera, not registered")
    manifest.notes = notes
    manifest.ground_based = True
    manifest.outside_range = "zero"
    manifest.sources.update({OPTICS: f"{ZIP_URL}: Dichroic.txt x the band's default NIX camera",
                             "archive": ZIP_URL, "page": PAGE, "manual": MANUAL})
    manifest.save(inst_dir)
    return inst_dir
