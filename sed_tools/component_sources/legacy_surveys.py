"""BASS (Bok 2.3 m, 90Prime) and MzLS (Mayall 4 m, Mosaic-3) from the Legacy Surveys files.

The DESI Legacy Imaging Surveys DR10 description links one file per band
(https://www.legacysurvey.org/dr10/description/):

``bass-g.txt``, ``bass-r.txt``
    Angstrom and four columns: "the pure filter transmission", "the filter
    response include CCD QE" and "the response including both CCD QE and
    atmospheric extinction at airmass of 1.0".  The BASS site adds that the
    extinction coefficients are KPNO's and that molecular absorption is not
    included.  Values are given to 0.001.

    * ``filter`` is column 1.  It equals SVO's ``BOK/BASS.<b>_filter``
      (published here as ``<b>_filter``), so that curve is the filter
      component and column 1 is not stored twice.
    * ``filter+qe`` is column 2, stored as published (a composite component).
    * ``qe`` is column 2 / column 1 where column 1 >= 0.05, so the 0.001
      rounding moves the ratio by at most 1 %; it is per band (each file
      covers its own wavelengths).
    * Column 3 / column 2 is the atmosphere at airmass 1.0; it is not an
      instrument component (atmospheres are registered as sites).

    The published ``g`` and ``r`` are SVO's ``BOK/BASS.g`` and ``.r``, which
    SVO describes as also including aluminium reflectivity and corrector
    throughput; those factors are not published separately, so no composed
    product reproduces them.

``kpzd.txt``, ``kpzdccdcorr3.txt``
    MzLS z: the area-weighted filter transmission (k1038, in per cent) and
    the curve "with corrections", which includes terms for the telescope,
    corrector, camera and atmosphere (airmass not stated).  These are not
    separable, so MzLS has a filter component only; ``z`` is the corrected
    curve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..filters.compose import FILTER, QE, InstrumentManifest, add_component, recipe_name
from ..filters.curve_metadata import PHOTON, SVOMetadataClient
from ._common import Table, ensure_published
from ._original import Band, Component, Fetcher, Instrument, clean_curve, write_instrument

FILES = "https://www.legacysurvey.org/files"
DR10 = "https://www.legacysurvey.org/dr10/description/"
BASS_SITE = "http://batc.bao.ac.cn/BASS/doku.php?id=datarelease:telescope_and_instrument:home"
BASS_BANDS = ("g", "r")
#: Column-1 floor for the QE ratio: values are rounded to 0.001.
QE_FLOOR = 0.05


def read_bass(raw: bytes):
    """(wavelength A, filter, filter*QE, filter*QE*atmosphere) from a bass-<b>.txt file."""
    rows = []
    for line in raw.decode("utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) != 4:
            raise ValueError(f"expected 4 columns, got {line!r}")
        rows.append([float(x) for x in parts])
    data = np.array(rows)
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("wavelengths are not increasing")
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def bass_qe(wl, filt, filt_qe, floor: float = QE_FLOOR):
    """Column 2 / column 1 where column 1 >= ``floor`` (wavelength, QE)."""
    ok = filt >= floor
    if ok.sum() < 2:
        raise ValueError("too few samples above the QE floor")
    return wl[ok], filt_qe[ok] / filt[ok]


def build_bass(filter_root, *, session=None, svo_client=None, **_ignored) -> Path:
    fetch = Fetcher(session)
    curves = {b: read_bass(fetch.get(f"{FILES}/bass-{b}.txt")) for b in BASS_BANDS}
    inst_dir = Path(filter_root) / "BOK" / "BASS"
    ids = {b: f"BOK/BASS.{b}" for b in BASS_BANDS}
    ids.update({f"{b}_filter": f"BOK/BASS.{b}_filter" for b in BASS_BANDS})
    ensure_published(inst_dir, ids, svo_client or SVOMetadataClient())

    manifest = InstrumentManifest.from_published(inst_dir)
    manifest.facility, manifest.instrument = "BOK", "BASS"
    manifest.ground_based = True
    for band, (wl, filt, filt_qe, _with_atm) in curves.items():
        url = f"{FILES}/bass-{band}.txt"
        if manifest.component_file(FILTER, band) is None:
            add_component(inst_dir, FILTER, wl, np.maximum(filt, 0.0), band=band, name=band,
                          source=f"{url} (column 1, pure filter transmission)", manifest=manifest)
        add_component(inst_dir, recipe_name((FILTER, QE)), wl, np.maximum(filt_qe, 0.0), band=band,
                      name=band, source=f"{url} (column 2, filter response including CCD QE)",
                      manifest=manifest)
        qwl, qtr = bass_qe(wl, filt, filt_qe)
        add_component(inst_dir, QE, qwl, qtr, band=band, name=f"qe_{band}",
                      source=f"{url} (column 2 / column 1 where column 1 >= {QE_FLOOR})",
                      manifest=manifest)
    manifest.outside_range = "error"
    manifest.sources.update({"legacysurvey": DR10, "bass": BASS_SITE})
    manifest.notes = (
        "BASS g and r. Published g and r are SVO's BOK/BASS.g and .r (filter, CCD QE, aluminium reflectivity, "
        "corrector and atmosphere at airmass 1.0, per SVO); g_filter and r_filter are SVO's filter-only curves "
        "and equal column 1 of the Legacy Surveys bass-<b>.txt files. Components from those files: filter+qe "
        f"(column 2, as published) and qe (column 2 / column 1 where column 1 >= {QE_FLOOR}; values are given "
        "to 0.001). The telescope and corrector throughput is not published separately, so no composed product "
        "reproduces the published g and r. Column 3 / column 2 is the KPNO atmosphere at airmass 1.0 without "
        "molecular absorption (BASS site); it belongs to an atmosphere site, not to the instrument."
    )
    manifest.save(inst_dir)
    return inst_dir


def _mzls(fetch: Fetcher) -> Instrument:
    filt_url, total_url = f"{FILES}/kpzd.txt", f"{FILES}/kpzdccdcorr3.txt"
    filt = Table(filt_url, tr_scale=0.01).parse(fetch.get(filt_url))
    total = Table(total_url).parse(fetch.get(total_url))
    if filt[1].max() > 1.0 or total[1].max() > 1.0:
        raise ValueError("MzLS curves are not fractions after scaling")
    wl, tr, note = clean_curve(*total)
    band = Band("z", wl, tr, PHOTON, "filter+optics+qe+atmosphere", f"{total_url} | {DR10}",
                detector_source="assumed", clean_note=note)
    return Instrument(
        facility="KPNO", instrument="MzLS", bands=[band], ground_based=True,
        components=[Component(FILTER, filt[0], filt[1], band="z",
                              source=f"{filt_url} (area-weighted zd filter k1038, per cent / 100)")],
        outside_range="error",
        notes=("MzLS z (Mosaic-3 on the Mayall 4 m). z is the Legacy Surveys DR10 curve 'with corrections', "
               "which includes terms for the telescope, corrector, camera and atmosphere (airmass not stated). "
               "The filter component is the area-weighted zd filter transmission (kpzd.txt, k1038). The other "
               "factors are not published separately, so only the filter can be composed."),
    )


def build_mzls(filter_root, *, session=None, **_ignored) -> Path:
    inst_dir = write_instrument(filter_root, _mzls(Fetcher(session)))
    manifest = InstrumentManifest.load(inst_dir)
    manifest.sources["legacysurvey"] = DR10
    manifest.save(inst_dir)
    return inst_dir


def build(filter_root, *, instrument="BASS", **kwargs) -> Path:
    if instrument == "BASS":
        return build_bass(filter_root, **kwargs)
    if instrument == "MzLS":
        return build_mzls(filter_root, **kwargs)
    raise ValueError(f"No Legacy Surveys builder for {instrument!r}; available: BASS, MzLS")
