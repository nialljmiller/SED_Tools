"""Rubin/LSST components from the lsst/throughputs repository.

The Rubin system engineering team publishes every factor separately
(https://github.com/lsst/throughputs, ``baseline/``):

* ``filter_<b>.dat``  filter only                         -> ``filter``
* ``m1-3.dat``, ``lens1-3.dat``  mirrors and lenses       -> ``optics`` (their product)
* ``detector.dat``  detector QE                           -> ``qe``
* ``atmos_std.dat`` (X=1.2), ``atmos_10.dat`` (X=1.0)     -> atmosphere site ``lsst_std``

``hardware_<b>`` (= filter x optics x QE) and ``total_<b>`` (x atmos_std) are
used only to validate the composition.  Files are pinned to a release tag.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from ..filter_components import (ATMOSPHERE, FILTER, OPTICS, QE, SHARED, AtmosphereGrid,
                                 InstrumentManifest, add_component)
from ..filter_metadata import (ENERGY, PHOTON, SOURCE_SVO, SVOMetadataClient,
                               svo_components_header, write_filter_file)

logger = logging.getLogger(__name__)

FACILITY, INSTRUMENT = "LSST", "LSST"
BANDS = ("u", "g", "r", "i", "z", "y")
TAG = "v1.9.2"
REPO = "https://raw.githubusercontent.com/lsst/throughputs"
OPTICS_FILES = ("m1", "m2", "m3", "lens1", "lens2", "lens3")
ATMOSPHERE_SITE = "lsst_std"


def _fetch(session, name: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    resp = session.get(f"{REPO}/{tag}/baseline/{name}.dat", timeout=120)
    resp.raise_for_status()
    data = np.loadtxt(io.StringIO(resp.text), comments="#", usecols=(0, 1))
    return data[:, 0], data[:, 1]  # nm, fraction


def _source(name: str, tag: str) -> str:
    return f"lsst/throughputs {tag} baseline/{name}.dat"


def build(
    filter_root: Union[str, Path],
    *,
    atmosphere_root: Union[str, Path, None] = None,
    tag: str = TAG,
    session=None,
    svo_client: SVOMetadataClient = None,
) -> Path:
    if session is None:
        import requests

        session = requests.Session()
    filter_root = Path(filter_root)
    instrument_dir = filter_root / FACILITY / INSTRUMENT
    instrument_dir.mkdir(parents=True, exist_ok=True)

    # Published curves (SVO) are the MESA-facing default; add any that are missing.
    svo_client = svo_client or SVOMetadataClient()
    for band in BANDS:
        path = instrument_dir / f"{band}.dat"
        if path.exists():
            continue
        record = svo_client.get_filter(f"LSST/LSST.{band}")
        if record is None:
            raise RuntimeError(f"SVO has no LSST/LSST.{band}")
        write_filter_file(
            path, record.wavelength, record.transmission,
            detector_type=record.detector_type or PHOTON, source=SOURCE_SVO,
            filter_id=f"LSST/LSST.{band}", wavelength_unit=record.wavelength_unit,
            extra=svo_components_header(record.components),
        )
    if not (instrument_dir / INSTRUMENT).exists():
        from ..combine_filters import write_filter_index

        write_filter_index(instrument_dir, INSTRUMENT)

    manifest = InstrumentManifest.load(instrument_dir)
    manifest.components = {}

    for band in BANDS:
        wl, tr = _fetch(session, f"filter_{band}", tag)
        add_component(instrument_dir, FILTER, wl * 10, tr, band=band,
                      source=_source(f"filter_{band}", tag), manifest=manifest)

    grid_wl, optics = None, None
    for name in OPTICS_FILES:
        wl, tr = _fetch(session, name, tag)
        if grid_wl is None:
            grid_wl, optics = wl, tr.copy()
        else:
            optics = optics * np.interp(grid_wl, wl, tr, left=0.0, right=0.0)
    add_component(instrument_dir, OPTICS, grid_wl * 10, optics, band=SHARED, name="mirrors_lenses",
                  source="product of " + ", ".join(_source(n, tag) for n in OPTICS_FILES), manifest=manifest)

    wl, tr = _fetch(session, "detector", tag)
    add_component(instrument_dir, QE, wl * 10, tr, band=SHARED, name="detector",
                  source=_source("detector", tag), manifest=manifest)

    manifest.facility, manifest.instrument = FACILITY, INSTRUMENT
    manifest.bands = list(BANDS)
    manifest.detector_type = PHOTON
    manifest.ground_based = True
    manifest.default_site = ATMOSPHERE_SITE
    manifest.sources["repository"] = f"https://github.com/lsst/throughputs/tree/{tag}/baseline"
    manifest.notes = (
        "Components include Rubin's time-averaged ten-year losses. The published curves are SVO's "
        "LSST/LSST.* (filter+instrument+atmosphere). Rubin's own standard atmospheres are available "
        f"as atmosphere site '{ATMOSPHERE_SITE}' (airmass 1.0 and 1.2)."
    )
    manifest.save(instrument_dir)

    # Rubin's fiducial atmospheres as an atmosphere "site".
    atm_root = Path(atmosphere_root) if atmosphere_root else filter_root.parent / "atmospheres"
    site_dir = atm_root / ATMOSPHERE_SITE
    site_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, airmass in (("atmos_10", 1.0), ("atmos_std", 1.2)):
        wl, tr = _fetch(session, name, tag)
        fname = f"am{airmass:.2f}.dat"
        write_filter_file(
            site_dir / fname, wl * 10, tr, detector_type=ENERGY, source="component",
            wavelength_unit="Angstrom",
            extra={"Role": ATMOSPHERE, "Components": ATMOSPHERE,
                   "Atmosphere": f"site={ATMOSPHERE_SITE} airmass={airmass:g}",
                   "Sources": _source(name, tag)},
        )
        entries.append({"airmass": airmass, "pwv": None, "file": fname})
    AtmosphereGrid(
        site=ATMOSPHERE_SITE, entries=entries, altitude_m=2647.0,
        source=f"Rubin fiducial atmosphere (MODTRAN, with aerosols), lsst/throughputs {tag}",
        notes="Cerro Pachon; fixed water vapour as in Rubin's baseline.",
    ).save(site_dir)
    return instrument_dir


def validate(instrument_dir: Union[str, Path], tag: str = TAG, session=None,
             atmosphere_root: Union[str, Path, None] = None) -> Dict[str, Dict[str, float]]:
    """Compare composed products with Rubin's own hardware/total curves.

    Returns, per band, the magnitude difference (composed - Rubin) for a flat
    F_nu source and the maximum absolute curve difference.
    """
    from ..filter_components import compose_band

    if session is None:
        import requests

        session = requests.Session()
    instrument_dir = Path(instrument_dir)
    atm_root = Path(atmosphere_root) if atmosphere_root else instrument_dir.parent.parent.parent / "atmospheres"
    out: Dict[str, Dict[str, float]] = {}
    for band in BANDS:
        result = {}
        for kind, kwargs in (
            ("hardware", {}),
            ("total", {"atmosphere": {"site": ATMOSPHERE_SITE, "airmass": 1.2}, "atmosphere_root": atm_root}),
        ):
            mine = compose_band(instrument_dir, band, [FILTER, OPTICS, QE], **kwargs)
            wl, ref = _fetch(session, f"{kind}_{band}", tag)
            wl = wl * 10
            ours = np.interp(wl, mine.wavelength, mine.transmission, left=0.0, right=0.0)
            # flat F_nu source, photon counting: weight T*lambda * lambda^-2
            ratio = np.trapezoid(ours / wl, wl) / np.trapezoid(ref / wl, wl)
            result[f"{kind}_dmag"] = float(-2.5 * np.log10(ratio))
            result[f"{kind}_maxdiff"] = float(np.max(np.abs(ours - ref)))
        out[band] = result
    return out
