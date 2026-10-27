"""Rubin/LSST components from the lsst/throughputs repository.

The Rubin system engineering team publishes every factor separately
(https://github.com/lsst/throughputs, ``baseline/``), pinned by release tag:

* ``filter_<b>.dat``  filter only                         -> ``filter``
* ``m1-3.dat``, ``lens1-3.dat``  mirrors and lenses       -> ``optics`` (their product)
* ``detector.dat``  detector QE                           -> ``qe``
* ``atmos_std.dat`` (X=1.2), ``atmos_10.dat`` (X=1.0)     -> atmosphere site ``lsst_std``
* ``hardware_<b>`` (= filter x optics x QE) and ``total_<b>`` (x atmos_std)

Two instruments are kept apart, one per release:

``LSST/LSST``     SVO's ``LSST/LSST.*`` published curves, which equal the
                  release 1.5 ``total_<b>.dat`` (to 5e-7); its components
                  come from tag ``1.5``.
``LSST/LSSTv19``  release 1.9 (tag ``v1.9.2``; baseline data identical in
                  ``1.9`` and ``v1.9.1``): Ag-Ag-Ag mirrors and as-measured
                  filters.  Its published curves are the ``total_<b>.dat``
                  themselves.

Airmass of the atmospheres: ``baseline/README.md`` says "atmos_std.dat is a
fiducial atmosphere throughput at Rubin at 1.2 airmasses".  ``atmos_10.dat``
is not named there, but the syseng_throughputs README section included in
the same file says "To represent zenith, optimum throughputs, the X=1.0,
with aerosols ... atmos_10_aerosol.dat curve should be used", and
``atmos_10.dat`` equals syseng_throughputs ``siteProperties/atmos_10_aerosol.dat``
(the sha1 its header names) except its last sample (1150 nm: 0 instead of
0.93); ln T(atmos_std)/ln T(atmos_10) has median 1.199 (X 1.2 vs 1.0).  Both
atmosphere files are numerically identical in releases 1.5 and 1.9, so both
instruments share the ``lsst_std`` site.  An existing site is never rewritten.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..filter_components import (ATMOSPHERE, FILTER, OPTICS, QE, SHARED, AtmosphereGrid,
                                 InstrumentManifest, add_component)
from ..filter_metadata import (ENERGY, PHOTON, SOURCE_ASSUMED, SOURCE_SVO, SVOMetadataClient,
                               svo_components_header, write_filter_file)

FACILITY = "LSST"
BANDS = ("u", "g", "r", "i", "z", "y")
REPO = "https://raw.githubusercontent.com/lsst/throughputs"
OPTICS_FILES = ("m1", "m2", "m3", "lens1", "lens2", "lens3")
ATMOSPHERE_SITE = "lsst_std"
ATMOSPHERES = (("atmos_10", 1.0), ("atmos_std", 1.2))

#: instrument -> (release tag, component file suffix, published curves from)
RELEASES: Dict[str, Tuple[str, str, str]] = {
    "LSST": ("1.5", "_v1p5", "svo"),
    "LSSTv19": ("v1.9.2", "", "total"),
}
#: Kept for callers of the earlier single-release builder.
INSTRUMENT, TAG = "LSST", RELEASES["LSST"][0]

AIRMASS_EVIDENCE = (
    "baseline/README.md: 'atmos_std.dat is a fiducial atmosphere throughput at Rubin at 1.2 "
    "airmasses' (the same README also calls it 'the zenith atmos_std.dat atmosphere'). atmos_10.dat "
    "is X=1.0: the syseng_throughputs section of that README says 'To represent zenith, optimum "
    "throughputs, the X=1.0, with aerosols [atmosphere](./siteProperties/atmos_10_aerosol.dat) "
    "curve should be used', and atmos_10.dat equals that file at the sha1 in its header except the "
    "1150 nm sample; ln T(atmos_std)/ln T(atmos_10) has median 1.199."
)

NOTES = {
    "LSST": (
        "Published curves are SVO's LSST/LSST.*, which equal lsst/throughputs release 1.5 "
        "baseline/total_<b>.dat to 5e-7 (filter x m1-m3 x lens1-3 x detector x atmos_std, X=1.2). "
        "Components are that release's (tag 1.5), with Rubin's time-averaged ten-year losses, so "
        "filter x optics x qe x lsst_std(X=1.2) reproduces the published curves. Earlier builds put "
        "release 1.9.2 components here (components/*/<name>.dat without the _v1p5 suffix; kept on "
        "disk, no longer registered); release 1.9 is instrument LSST/LSSTv19. Atmosphere site "
        "'lsst_std': " + AIRMASS_EVIDENCE
    ),
    "LSSTv19": (
        "Rubin release 1.9 (lsst/throughputs tag v1.9.2; baseline data identical in tags 1.9 and "
        "v1.9.1): Ag-Ag-Ag mirror coatings and as-measured filters, lenses and mirrors. Published "
        "curves are baseline/total_<b>.dat (filter x m1-m3 x lens1-3 x detector x atmos_std, X=1.2; "
        "nm -> Angstrom). DetectorType photon is assumed: the totals multiply transmissions by the "
        "detector QE ('detector sensitivity'), but the files do not state a convention. Components "
        "are the same release's factors with Rubin's time-averaged ten-year losses. SVO's "
        "LSST/LSST is release 1.5. Atmosphere site 'lsst_std': " + AIRMASS_EVIDENCE
    ),
}


def _fetch(session, name: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    resp = session.get(f"{REPO}/{tag}/baseline/{name}.dat", timeout=120)
    resp.raise_for_status()
    data = np.loadtxt(io.StringIO(resp.text), comments="#", usecols=(0, 1))
    return data[:, 0], data[:, 1]  # nm, fraction


def _source(name: str, tag: str) -> str:
    return f"lsst/throughputs {tag} baseline/{name}.dat"


def _url(name: str, tag: str) -> str:
    return f"{REPO}/{tag}/baseline/{name}.dat"


def _publish_svo(instrument_dir: Path, svo_client: Optional[SVOMetadataClient]) -> None:
    missing = [b for b in BANDS if not (instrument_dir / f"{b}.dat").exists()]
    if not missing:
        return
    svo_client = svo_client or SVOMetadataClient()
    for band in missing:
        record = svo_client.get_filter(f"LSST/LSST.{band}")
        if record is None:
            raise RuntimeError(f"SVO has no LSST/LSST.{band}")
        write_filter_file(
            instrument_dir / f"{band}.dat", record.wavelength, record.transmission,
            detector_type=record.detector_type or PHOTON, source=SOURCE_SVO,
            filter_id=f"LSST/LSST.{band}", wavelength_unit=record.wavelength_unit,
            extra=svo_components_header(record.components),
        )


def _publish_totals(instrument_dir: Path, instrument: str, tag: str, totals) -> None:
    for band, (wl, tr) in totals.items():
        path = instrument_dir / f"{band}.dat"
        if path.exists():
            continue
        keep = np.nonzero(tr > 0)[0]
        lo, hi = max(keep[0] - 1, 0), min(keep[-1] + 1, len(wl) - 1)  # trim the zero padding
        write_filter_file(
            path, wl[lo:hi + 1] * 10, np.maximum(tr[lo:hi + 1], 0.0),
            detector_type=PHOTON, source=SOURCE_ASSUMED,
            filter_id=f"{FACILITY}/{instrument}.{band}", wavelength_unit="Angstrom",
            extra={"Components": "filter+optics+qe+atmosphere", "ComponentsSource": "original",
                   "Atmosphere": f"site={ATMOSPHERE_SITE} airmass=1.2",
                   "Sources": _url(f"total_{band}", tag)},
        )


def build(
    filter_root: Union[str, Path],
    *,
    instrument: str = "LSST",
    atmosphere_root: Union[str, Path, None] = None,
    tag: Optional[str] = None,
    session=None,
    svo_client: SVOMetadataClient = None,
) -> Path:
    if instrument not in RELEASES:
        raise ValueError(f"No LSST release for {instrument!r}; available: {sorted(RELEASES)}")
    release, suffix, published_from = RELEASES[instrument]
    tag = tag or release
    if session is None:
        import requests

        session = requests.Session()
    filter_root = Path(filter_root)
    instrument_dir = filter_root / FACILITY / instrument

    # Fetch every file before writing anything.
    filters = {b: _fetch(session, f"filter_{b}", tag) for b in BANDS}
    optics_parts = [_fetch(session, n, tag) for n in OPTICS_FILES]
    detector = _fetch(session, "detector", tag)
    totals = ({b: _fetch(session, f"total_{b}", tag) for b in BANDS}
              if published_from == "total" else {})
    atmospheres = {name: _fetch(session, name, tag) for name, _ in ATMOSPHERES}

    instrument_dir.mkdir(parents=True, exist_ok=True)
    if published_from == "svo":
        _publish_svo(instrument_dir, svo_client)
    else:
        _publish_totals(instrument_dir, instrument, tag, totals)
    if not (instrument_dir / instrument).exists():
        from ..combine_filters import write_filter_index

        write_filter_index(instrument_dir, instrument)

    _register_components(instrument_dir, instrument, tag, suffix, filters, optics_parts, detector)
    atm_root = Path(atmosphere_root) if atmosphere_root else filter_root.parent / "atmospheres"
    _write_site(atm_root / ATMOSPHERE_SITE, atmospheres, tag)
    return instrument_dir


def _register_components(instrument_dir: Path, instrument: str, tag: str, suffix: str, filters, optics_parts,
                         detector) -> None:
    """The release's filter, optics (mirrors x lenses) and detector QE, in a fresh manifest."""
    manifest = InstrumentManifest.from_published(instrument_dir)
    manifest.components = {}
    for band in BANDS:
        wl, tr = filters[band]
        add_component(instrument_dir, FILTER, wl * 10, tr, band=band, name=band + suffix,
                      source=_source(f"filter_{band}", tag), manifest=manifest)

    grid_wl, optics = optics_parts[0][0], optics_parts[0][1].copy()
    for wl, tr in optics_parts[1:]:
        optics = optics * np.interp(grid_wl, wl, tr, left=0.0, right=0.0)
    add_component(instrument_dir, OPTICS, grid_wl * 10, optics, band=SHARED,
                  name="mirrors_lenses" + suffix,
                  source="product of " + ", ".join(_source(n, tag) for n in OPTICS_FILES), manifest=manifest)
    add_component(instrument_dir, QE, detector[0] * 10, detector[1], band=SHARED, name="detector" + suffix,
                  source=_source("detector", tag), manifest=manifest)

    manifest.facility, manifest.instrument = FACILITY, instrument
    manifest.bands = list(BANDS)
    manifest.detector_type = PHOTON
    manifest.ground_based = True
    manifest.default_site = ATMOSPHERE_SITE
    manifest.sources["repository"] = f"https://github.com/lsst/throughputs/tree/{tag}/baseline"
    manifest.notes = NOTES[instrument]
    manifest.save(instrument_dir)


def _write_site(site_dir: Path, atmospheres, tag: str) -> None:
    """Rubin's fiducial atmospheres as the atmosphere site ``lsst_std``, written once and never rewritten."""
    if (site_dir / "manifest.json").exists():
        return
    site_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, airmass in ATMOSPHERES:
        wl, tr = atmospheres[name]
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
        notes="Cerro Pachon; fixed water vapour as in Rubin's baseline. " + AIRMASS_EVIDENCE,
    ).save(site_dir)


def validate(instrument_dir: Union[str, Path], tag: Optional[str] = None, session=None,
             atmosphere_root: Union[str, Path, None] = None) -> Dict[str, Dict[str, float]]:
    """Compare composed products with Rubin's own hardware/total curves.

    Returns, per band, the magnitude difference (composed - Rubin) for a flat
    F_nu source and the maximum absolute curve difference.  ``tag`` defaults
    to the instrument's own release.
    """
    from ..filter_components import compose_band

    if session is None:
        import requests

        session = requests.Session()
    instrument_dir = Path(instrument_dir)
    tag = tag or RELEASES.get(instrument_dir.name, (TAG,))[0]
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
