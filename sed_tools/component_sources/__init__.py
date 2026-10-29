"""Builders that populate ``components/`` for instruments whose teams publish them.

Each builder module exposes ``build(filter_root, **options) -> Path`` returning
the instrument directory it populated.  Builders only add ``components/`` and
``manifest.json`` (and published curves where none exist); they never alter
existing published ``.dat`` files or MESA index files.

Builders are run where the data is curated (the njm mirror), e.g.::

    sed-tools filters-build-components argus --raw /path/to/Argus/raw
    sed-tools filters-build-components argus --instrument DR280 --raw /path/to/Argus/raw
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Dict

#: name -> module (relative to this package)
BUILDERS: Dict[str, str] = {
    "argus": ".argus",
    "skycalc": ".skycalc",
    "team_atmospheres": ".team_atmospheres",
    "lsst": ".lsst",
    "hst": ".hst",
    "jwst": ".jwst",
    "hsc": ".hsc",
    "decam": ".decam",
    "vista": ".vista",
    "ztf": ".ztf",
    "tabular": ".tabular",
    "surveys": ".surveys",
    "space": ".space",
    "observatories": ".observatories",
    "standards": ".standards",
    "cfht": ".cfht",
    "eris": ".eris",
    "twomass": ".twomass",
    "legacy_surveys": ".legacy_surveys",
    "gmos": ".gmos",
    "sdss": ".sdss",
    "euclid": ".euclid",
    "roman": ".roman",
    "swift": ".swift",
    "csp": ".csp",
    "cassini": ".cassini",
}


#: Builders that ``filters-build-components all`` runs. Each downloads its own
#: inputs, but ``hst`` and ``jwst`` also need their instruments' published curves
#: installed. (``argus`` is not one of them: it needs its raw directory.)
AUTOMATIC_BUILDERS = ("lsst", "hst", "jwst", "hsc", "decam", "vista", "ztf", "skycalc", "team_atmospheres")


def get_builder(name: str) -> Callable[..., Path]:
    try:
        module = importlib.import_module(BUILDERS[name.lower()], __name__)
    except KeyError:
        raise ValueError(f"Unknown component source '{name}'. Available: {sorted(BUILDERS)}") from None
    return module.build


def compare_to_published(instrument_dir, components=("filter", "optics", "qe"), temperatures=(3000.0, 30000.0)):
    """Shape check of a composed recipe against the published curves.

    Returns ``{band: max |dmag|}`` over black bodies at ``temperatures``,
    using band-averaged F_lambda with each curve's detector weighting.  A
    constant scale cancels, so this tests shape only (what photometry sees).
    """
    import numpy as np

    from ..filter_components import compose_band, InstrumentManifest
    from ..photometry import FilterCurve, band_average_flux_lambda, load_filter_curve

    instrument_dir = Path(instrument_dir)
    manifest = InstrumentManifest.load(instrument_dir)
    out = {}
    for band in manifest.bands:
        if not manifest.cover(band, components):
            continue
        published = load_filter_curve(str(instrument_dir / f"{band}.dat"))
        mine = compose_band(instrument_dir, band, components, manifest=manifest)
        composed = FilterCurve(band, mine.wavelength, mine.transmission, {}, detector_type=published.detector_type)
        wl = np.geomspace(min(published.wavelength[0], composed.wavelength[0]),
                          max(published.wavelength[-1], composed.wavelength[-1]), 20000)
        worst = 0.0
        for t in temperatures:
            lam = wl * 1e-8
            flux = 1.0 / lam**5 / np.expm1(1.4388 / (lam * t))
            a = band_average_flux_lambda(wl, flux, composed)
            b = band_average_flux_lambda(wl, flux, published)
            worst = max(worst, abs(-2.5 * np.log10(a / b)))
        out[band] = worst
    return out
