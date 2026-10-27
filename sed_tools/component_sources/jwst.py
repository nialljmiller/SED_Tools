"""JWST imager components from STScI's Pandeia reference data (in-flight).

Pandeia (the JWST ETC engine) keeps every throughput factor separately in
``jwst/<instrument>/{filters,optical,qe}`` with ``config.json`` naming the
files; the telescope (OTE) is in ``jwst/telescope``.  Imaging throughput is

* NIRCam:  OTE x internal optics x dichroic (SW/LW) x filter x QE (SW/LW)
* NIRISS:  OTE x internal optics x filter x QE
* MIRI:    OTE x filter x imager QE

``optics`` is the product of the non-filter, non-detector factors (MIRI's
imager has none besides the OTE: its ``mirror_eff`` is 1).  Wavelengths are
in microns (``TUNIT`` MICRONS in every throughput table).  The QE tables
also carry a ``CONVERSION`` column (electrons per photon, up to 1.84 for
NIRISS); it is not applied here, and whether Pandeia applies it to the
throughput has not been checked.

The reference data is the current release, 2026.7 (``-delivered-for-
version-`` in every ``config.json``): a 21.6 MB tarball without the PSF
library, which is a separate ~4 GB download and is not needed.  It is
downloaded once, checked against the SHA-1 that STScI's Box share lists,
and only the throughput files are extracted; pass ``raw`` to use an
unpacked copy instead.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ..filter_components import FILTER, OPTICS, QE, InstrumentManifest, add_component
from ..filter_metadata import PHOTON
from ._common import product
from ._original import safe_member

logger = logging.getLogger(__name__)

FACILITY = "JWST"
REFDATA_VERSION = "2026.7"
#: STScI's Box share of ``pandeia_data-2026.7-jwst.tar.gz`` (21,622,533 bytes).
REFDATA_PAGE = "https://stsci.box.com/v/pandeia-data-v2026p7-jwst"
REFDATA_URL = (
    "https://stsci.app.box.com/index.php?rm=box_download_shared_file"
    "&vanity_name=pandeia-data-v2026p7-jwst&file_id=f_2351491837746"
)
#: SHA-1 listed by the Box share (SHA-256 1666cb3e...7582692).
REFDATA_SHA1 = "e5c2507c42b10e8e1cf86310cb7a81c925bf9a70"
_MICRONS = {"micron", "microns", "um"}
INSTRUMENTS = {"NIRCam": "nircam", "NIRISS": "niriss", "MIRI": "miri"}
_KEEP = re.compile(r"jwst/(telescope|nircam|miri|niriss)/(config\.json|filters/|optical/|qe/|[^/]*\.fits$)")


def fetch_refdata(dest: Union[str, Path], url: str = REFDATA_URL, session=None,
                  sha1: Optional[str] = REFDATA_SHA1) -> Path:
    """Download the Pandeia tarball, check it, and extract only the throughput files."""
    if session is None:
        import requests

        session = requests.Session()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    resp = session.get(url, timeout=600)
    resp.raise_for_status()
    data = resp.content
    if sha1 and hashlib.sha1(data).hexdigest() != sha1:
        raise RuntimeError(f"Pandeia reference data from {url} does not match SHA-1 {sha1}")
    root = None
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for member in tf:
            if member.isfile() and _KEEP.search(member.name) and "psf" not in member.name.lower():
                safe_member(member.name)
                if hasattr(tarfile, "data_filter"):
                    tf.extract(member, dest, filter="data")
                else:  # Python without extraction filters; the name was checked above
                    tf.extract(member, dest)
                root = root or member.name.split("/jwst/")[0]
    if root is None:
        raise RuntimeError("No JWST throughput files found in the Pandeia tarball")
    return dest / root


def _read(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    from astropy.io import fits

    hdu = fits.open(path)[1]
    data = hdu.data
    names = {n.lower(): n for n in data.columns.names}
    unit = (hdu.columns[names["wavelength"]].unit or "").strip().lower()
    if unit and unit not in _MICRONS:
        raise ValueError(f"{path}: wavelength unit {unit!r}, expected microns")
    wl = np.asarray(data[names["wavelength"]], dtype=float) * 1e4  # micron -> Angstrom
    # Pandeia reference curves include small negative fit tails (notably the
    # MIRI imager QE); transmission outside the physical range is zero.
    tr = np.maximum(np.asarray(data[names["throughput"]], dtype=float), 0.0)
    order = np.argsort(wl)
    return wl[order], tr[order]


def _band_factors(inst: str, band: str, paths: Dict[str, str]) -> Optional[Dict[str, List[str]]]:
    """Reference-file keys per role for one imaging band (None if not imaging)."""
    key = band.lower()
    if not isinstance(paths.get(key), str) or not paths[key].startswith("filters/"):
        return None
    if inst == "nircam":
        channel = "sw" if int(re.sub(r"\D", "", key)[:3]) < 240 else "lw"
        return {FILTER: [key], OPTICS: ["internal", f"dbs_{channel}"], QE: [f"qe_{channel}"]}
    if inst == "niriss":
        return {FILTER: [key], OPTICS: ["internal"], QE: ["qe"]}
    if inst == "miri":
        return {FILTER: [key], OPTICS: [], QE: ["miri_imager_qe"]}
    return None


def build(
    filter_root: Union[str, Path],
    *,
    raw: Union[str, Path, None] = None,
    instruments: Optional[Iterable[str]] = None,
    session=None,
) -> List[Path]:
    """Populate components for JWST NIRCam, NIRISS and MIRI imaging bands."""
    filter_root = Path(filter_root)
    refdata = Path(raw) if raw else fetch_refdata(filter_root.parent / "cache" / "pandeia", session=session)
    if (refdata / "jwst").is_dir():
        refdata = refdata / "jwst"
    tel_cfg = json.loads((refdata / "telescope" / "config.json").read_text())
    ote_file = refdata / "telescope" / tel_cfg["paths"]["otefile"]
    ote = _read(ote_file)

    built = []
    for svo_inst in instruments or INSTRUMENTS:
        inst = INSTRUMENTS[svo_inst]
        inst_dir = filter_root / FACILITY / svo_inst
        bands = sorted(p.stem for p in inst_dir.glob("*.dat"))
        if not bands:
            logger.warning("JWST/%s: no published curves present; fetch them first", svo_inst)
            continue
        paths = json.loads((refdata / inst / "config.json").read_text())["paths"]
        manifest = InstrumentManifest.load(inst_dir)
        manifest.components = {}
        skipped = []
        for band in bands:
            factors = _band_factors(inst, band, paths)
            if factors is None:
                skipped.append(band)
                continue
            for role, keys in factors.items():
                curves = [_read(refdata / inst / paths[k]) for k in keys]
                sources = [f"Pandeia {REFDATA_VERSION} jwst/{inst}/{paths[k]}" for k in keys]
                if role == OPTICS:
                    curves.insert(0, ote)
                    sources.insert(0, f"Pandeia {REFDATA_VERSION} jwst/telescope/{ote_file.name}")
                wl, tr = product(curves)
                add_component(inst_dir, role, wl, tr, band=band, name=band,
                              source="; ".join(sources), manifest=manifest)
        manifest.detector_type = PHOTON
        manifest.ground_based = False
        manifest.outside_range = "zero"  # Pandeia's convention: factors are zero outside their tables
        manifest.sources["pandeia"] = f"Pandeia reference data {REFDATA_VERSION} (JWST, in-flight; {REFDATA_PAGE})"
        manifest.notes = (
            "Optics = OTE x instrument internal optics (x dichroic for NIRCam). "
            + (f"Not decomposed (not imaging filters in Pandeia): {', '.join(skipped)}." if skipped else "")
        )
        manifest.save(inst_dir)
        built.append(inst_dir)
    return built
