"""Blanco/DECam components from the DES standard bandpasses (NOIRLab/CTIO).

The DES DR1/DR2 standard bandpass table gives each band's total throughput
(instrument + atmosphere at airmass 1.2) together with the atmosphere used
(``atm``).  As the DES documentation says, the instrument-only response is
the band column divided by ``atm``; that is DECam's own filter+optics+qe
curve.  The ``atm`` column is DECam's own-site atmosphere (``ctio_des``).
Filter-only curves come from SVO's ``CTIO/DECam.<b>_filter`` siblings.

The quotient exceeds 1 in i (peak 1.0217 at 8500 A) and z (1.0370 at
9060 A) in the table as retrieved on 2026-10-26.  A physical instrument throughput
cannot, so the band columns are not on an absolute scale relative to
``atm`` (the table does not say why).  The quotients are kept as published, not
clipped (clipping would change the band shape; a constant scale does not
affect synthetic photometry), and :func:`build` records the samples above 1
in the manifest notes.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from ..filters.compose import InstrumentManifest
from ._common import SimpleSource, SiteAtmosphere, Table, build_simple
from ._original import CONNECT_TIMEOUT

DES_STD = "https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc1884.txt"
BANDS = ("g", "r", "i", "z", "Y")
RECIPE = "filter+optics+qe"


def _des_table(session):
    from astropy.io import fits

    resp = session.get(DES_STD, timeout=(CONNECT_TIMEOUT, 300))
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
    composites={b: (RECIPE, _instrument_only(b)) for b in BANDS},
    site=SiteAtmosphere(
        site="ctio_des", altitude_m=2207.0,
        entries=[(1.2, None, _AtmTable(DES_STD))],
        source="DES standard atmosphere (airmass 1.2) from the DES standard bandpass table",
    ),
    references={RECIPE: f"DES standard bandpasses / atm ({DES_STD})"},
    notes="Instrument-only response = DES standard bandpass / DES atmosphere; filter-only curves are SVO's DECam *_filter.",
)


def above_unity(inst_dir) -> str:
    """Describe the instrument-only samples above 1 (empty if none)."""
    manifest = InstrumentManifest.load(inst_dir)
    notes = []
    for band, rel in sorted(manifest.components.get(RECIPE, {}).items()):
        wl, tr = np.loadtxt(Path(inst_dir) / rel, delimiter=",", skiprows=1, usecols=(0, 1), unpack=True)
        above = tr > 1.0
        if above.any():
            i = int(np.argmax(tr))
            notes.append(f"{band} peaks at {tr[i]:.6f} at {wl[i]:g} A ({int(above.sum())} samples above 1 "
                         f"between {wl[above].min():g} and {wl[above].max():g} A)")
    if not notes:
        return ""
    return ("Flag: the instrument-only quotient exceeds 1 in " + "; ".join(notes) + ". A physical "
            "throughput cannot, so the DES band columns are not on an absolute scale relative to atm "
            "(the table does not say why). Kept unclipped: clipping would change the band shape, and a "
            "constant scale does not affect synthetic photometry.")


def build(filter_root, **kwargs):
    inst_dir = build_simple(SPEC, filter_root, **kwargs)
    flag = above_unity(inst_dir)
    if flag:
        manifest = InstrumentManifest.load(inst_dir)
        manifest.notes = f"{SPEC.notes} {flag}"
        manifest.save(inst_dir)
    return inst_dir
