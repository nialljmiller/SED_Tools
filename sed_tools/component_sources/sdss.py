"""SDSS 2.5 m imaging camera responses without the atmosphere (Gunn 2001; Doi et al. 2010).

The published ``u`` ... ``z`` are SVO's ``SLOAN/SDSS`` curves; they equal
column ``respt`` of the SDSS ``filter_curves.fits`` (J. Gunn, June 2001):
the camera's response "on the sky looking through 1.3 airmasses at APO for
a point source", averaged over the six camera columns.  The same file gives
``resnoa``, the response with no atmosphere (for large sources, i.e.
without the infrared scattering that affects r and i point sources), which
is the default ``filter+optics+qe`` composite (variant ``gunn2001``).  Its
``xatm`` column, the atmosphere at one airmass at APO, is not an instrument
component.

Doi et al. 2010, AJ 139, 1628 (2010AJ....139.1628D) measured the response
of each of the camera's 30 CCDs (six camera columns x u, g, r, i, z):
"CCD & filter & 1st corrector & mirrors but no atmosphere".  The tables are
on the first author's page (supplementary tables for Fig. 8, last updated
2010-02-24).  They are the ``filter+optics+qe`` variants ``camcol1`` ...
``camcol6``.  Doi et al. found the u response had changed since 2001.

SVO's ``uprime_filter`` ... ``zprime_filter`` are the filters of the USNO
40-in primed system: they are kept in air, while the 2.5 m camera's filters
sit in vacuum, which moves their red edges about 2.5 % blueward (SDSS DR7
flux calibration page).  So they are bands of their own, not the 2.5 m
camera's filter component.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from ..filters.compose import FILTER, OPTICS, QE, InstrumentManifest, add_component, recipe_name
from ._common import Table, register_variant
from ._original import CONNECT_TIMEOUT, download  # noqa: F401  CONNECT_TIMEOUT stays importable here

DOI = "http://www.ioa.s.u-tokyo.ac.jp/~doi/sdss"
GUNN = "https://www.sdss4.org/wp-content/uploads/2017/04/filter_curves.fits"
CAMERA = "https://www.sdss4.org/instruments/camera/"
PAPER = "https://ui.adsabs.harvard.edu/abs/2010AJ....139.1628D"
FLUXCAL = "https://classic.sdss.org/dr7/algorithms/fluxcal.html"
BANDS = ("u", "g", "r", "i", "z")
CAMCOLS = (1, 2, 3, 4, 5, 6)
KEY = recipe_name((FILTER, OPTICS, QE))


def read_gunn(session):
    """band -> (wavelength A, resnoa) from the SDSS ``filter_curves.fits``."""
    from astropy.io import fits

    raw = download(session, GUNN)
    out = {}
    with fits.open(io.BytesIO(raw)) as hdul:
        for b in BANDS:
            data = hdul[b.upper()].data
            wl = np.asarray(data["wavelength"], dtype=float)
            tr = np.asarray(data["resnoa"], dtype=float)
            if np.any(np.diff(wl) <= 0) or tr.min() < 0 or tr.max() > 1:
                raise ValueError(f"{GUNN} {b}: unexpected resnoa table")
            out[b] = (wl, tr)
    return out


def build(filter_root, *, session=None, **_ignored) -> Path:
    if session is None:
        import requests

        session = requests.Session()
    inst_dir = Path(filter_root) / "SLOAN" / "SDSS"
    missing = [b for b in BANDS if not (inst_dir / f"{b}.dat").is_file()]
    if missing:
        raise FileNotFoundError(f"{inst_dir} lacks published {missing}; fetch the instrument first")
    tables = {(b, c): Table(f"{DOI}/{b}{c}.dat") for b in BANDS for c in CAMCOLS}
    curves = {key: t.read(session) for key, t in tables.items()}
    gunn = read_gunn(session)

    manifest = InstrumentManifest.from_published(inst_dir)
    manifest.facility, manifest.instrument = "SLOAN", "SDSS"
    manifest.variants.pop(KEY, None)
    files = {}
    for b in BANDS:
        files[b] = add_component(inst_dir, KEY, *gunn[b], band=b, name=f"{b}_gunn2001",
                                 source=f"{GUNN} (extension {b.upper()}, column resnoa)", manifest=manifest)
    register_variant(manifest, KEY, "gunn2001", files, default=True, source=GUNN,
                     description=("J. Gunn (June 2001) camera response with no atmosphere (resnoa: the "
                                  "large-source response without the atmosphere), average of the six camera "
                                  "columns; the published curves are the same table's point-source response "
                                  "through 1.3 airmasses at APO (respt)"))
    for c in CAMCOLS:
        files = {}
        for b in BANDS:
            files[b] = add_component(inst_dir, KEY, *curves[(b, c)], band=b, name=f"{b}_camcol{c}",
                                     source=f"{tables[(b, c)].url} (Doi et al. 2010, camera column {c})")
        register_variant(manifest, KEY, f"camcol{c}", files, source=f"{DOI}/SDSSresponse.html",
                         description=(f"Doi et al. (2010) response of camera column {c}: CCD, filter, first "
                                      "corrector and mirrors, no atmosphere"),
                         sources={b: tables[(b, c)].url for b in BANDS})
    manifest.outside_range = "error"
    manifest.sources.update({"doi2010": PAPER, "doi2010_tables": f"{DOI}/SDSSresponse.html",
                             "gunn2001": GUNN, "camera": CAMERA, "primed_system": FLUXCAL})
    manifest.notes = (
        "Published u, g, r, i, z are SVO's SLOAN/SDSS curves, equal to column respt of the SDSS "
        "filter_curves.fits (J. Gunn, June 2001: point-source response through 1.3 airmasses at APO, averaged "
        "over the six camera columns). The default filter+optics+qe is that file's resnoa (no atmosphere; "
        "large-source response, so r and i lack the point-source infrared-scattering loss). "
        "uprime_filter ... zprime_filter are SVO's filter-only curves of the USNO 40-in primed system (filters "
        "in air); the 2.5 m camera's filters are in vacuum and their red edges sit about 2.5% bluer (SDSS DR7 "
        "flux calibration), so the primed curves are separate bands, not the camera's filter. The camera's "
        "measured responses per camera column (Doi et al. 2010: CCD, filter, first corrector and mirrors, no "
        "atmosphere) are the filter+optics+qe variants camcol1 ... camcol6. Only u, g, r, i, z have these "
        "responses: compose them with bands=['u', 'g', 'r', 'i', 'z']."
    )
    manifest.save(inst_dir)
    return inst_dir
