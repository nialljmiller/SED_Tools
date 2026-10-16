"""Blanco/DECam components from the DES standard bandpasses (NOIRLab/CTIO).

The DES DR1/DR2 standard bandpass table gives each band's total throughput
(instrument + atmosphere at airmass 1.2) together with the atmosphere used
(``atm``).  As the DES documentation says, the instrument-only response is
the band column divided by ``atm``; that is DECam's own filter+optics+qe
curve.  The ``atm`` column is DECam's own-site atmosphere (``ctio_des``).
Filter-only curves come from SVO's ``CTIO/DECam.<b>_filter`` siblings.
"""

from __future__ import annotations

import io

import numpy as np

from ._common import SimpleSource, SiteAtmosphere, Table, build_simple

DES_STD = "https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc1884.txt"
BANDS = ("g", "r", "i", "z", "Y")


def _des_table(session):
    from astropy.io import fits

    resp = session.get(DES_STD, timeout=300)
    resp.raise_for_status()
    return fits.open(io.BytesIO(resp.content))[1].data


def _instrument_only(band):
    def make(session):
        data = _des_table(session)
        wl = np.asarray(data["LAMBDA"], dtype=float)
        atm = np.asarray(data["atm"], dtype=float)
        tr = np.where(atm > 0, np.asarray(data[band], dtype=float) / np.where(atm > 0, atm, 1.0), 0.0)
        return wl, tr
    return make


class _AtmTable(Table):
    def read(self, session):
        data = _des_table(session)
        return np.asarray(data["LAMBDA"], dtype=float), np.asarray(data["atm"], dtype=float)


SPEC = SimpleSource(
    facility="CTIO",
    instrument="DECam",
    filters={},
    svo_ids={**{b: f"CTIO/DECam.{b}" for b in BANDS}, **{f"{b}_filter": f"CTIO/DECam.{b}_filter" for b in BANDS}},
    composites={b: ("filter+optics+qe", _instrument_only(b)) for b in BANDS},
    site=SiteAtmosphere(
        site="ctio_des", altitude_m=2207.0,
        entries=[(1.2, None, _AtmTable(DES_STD))],
        source="DES standard atmosphere (airmass 1.2) from the DES standard bandpass table",
    ),
    references={"filter+optics+qe": f"DES standard bandpasses / atm ({DES_STD})"},
    notes="Instrument-only response = DES standard bandpass / DES atmosphere; filter-only curves are SVO's DECam *_filter.",
)


def build(filter_root, **kwargs):
    return build_simple(SPEC, filter_root, **kwargs)
