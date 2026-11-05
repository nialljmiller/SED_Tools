"""Offline tests for the component builders' shared machinery."""

from __future__ import annotations

import gzip
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from sed_tools.component_sources import catalog
from sed_tools.component_sources._common import SimpleSource, Table, build_simple, product
from sed_tools.filter_components import (FILTER, OPTICS, QE, ComponentError, Curve,
                                         InstrumentManifest, add_component, multiply_curves)
from sed_tools.filter_metadata import PHOTON, write_filter_file


class _Resp:
    def __init__(self, content, status=200):
        self.content = content if isinstance(content, bytes) else content.encode()
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self, files):
        self.files = files

    def get(self, url, timeout=None, **kw):
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


# ---------------------------------------------------------------------------
# Tables and products
# ---------------------------------------------------------------------------
def test_table_floors_negative_measured_transmission():
    wl, tr = Table("u").parse(b"4000,-0.0001\n5000,0.8\n")
    np.testing.assert_allclose(wl, [4000, 5000])
    np.testing.assert_allclose(tr, [0.0, 0.8])


def test_percent_component_is_rejected(tmp_path):
    with pytest.raises(ComponentError, match="percent"):
        add_component(tmp_path, FILTER, [4000.0, 5000.0], [10.0, 95.0], band="g")


# ---------------------------------------------------------------------------
# build_simple
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# HST
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# JWST
# ---------------------------------------------------------------------------
def test_jwst_reference_negative_tail_is_zero(tmp_path):
    from astropy.io import fits
    from sed_tools.component_sources.jwst import _read

    columns = [
        fits.Column(name="wavelength", format="D", array=[2.0, 1.0]),
        fits.Column(name="throughput", format="D", array=[0.7, -0.02]),
    ]
    path = tmp_path / "throughput.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)

    wl, tr = _read(path)
    np.testing.assert_allclose(wl, [10000.0, 20000.0])
    np.testing.assert_allclose(tr, [0.0, 0.7])


def test_jwst_band_factors():
    from sed_tools.component_sources.jwst import _band_factors

    paths = {"f200w": "filters/a.fits", "f444w": "filters/b.fits", "grismr": "blaze/c.fits"}
    assert _band_factors("nircam", "F200W", paths) == {FILTER: ["f200w"], OPTICS: ["internal", "dbs_sw"], QE: ["qe_sw"]}
    assert _band_factors("nircam", "F444W", paths)[QE] == ["qe_lw"]
    assert _band_factors("nircam", "GRISMR", paths) is None
    assert _band_factors("miri", "F444W", paths) == {FILTER: ["f444w"], OPTICS: [], QE: ["miri_imager_qe"]}


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def test_ensure_components_runs_own_builder_and_skips_heavy(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(catalog.BUILDERS_FOR, ("Fac", "Cam"), catalog.BuilderRef("fake"))
    monkeypatch.setitem(catalog.BUILDERS_FOR, ("Fac", "Big"), catalog.BuilderRef("fake", heavy=True))
    import sed_tools.component_sources as cs
    from sed_tools import filter_library
    monkeypatch.setattr(filter_library.NJMFilterGrabber, "_check_availability", lambda self: False)

    def build(root, **kw):
        calls.append(("fake", root))
        inst = root / "Fac" / "Cam"
        manifest = InstrumentManifest.load(inst)
        add_component(inst, FILTER, [4000.0, 5000.0], [1.0, 1.0], band="g", manifest=manifest)
        manifest.save(inst)

    monkeypatch.setattr(filter_library, "get_builder", lambda name: build)

    for name in ("Cam", "Big", "None"):
        d = tmp_path / "Fac" / name
        d.mkdir(parents=True)
        (d / "g.dat").write_text("Wavelength,Transmission\n4000,1\n5000,1\n")

    assert catalog.ensure_components(tmp_path / "Fac" / "Cam") is True
    assert catalog.ensure_components(tmp_path / "Fac" / "Big") is False
    assert catalog.ensure_components(tmp_path / "Fac" / "None") is False
    assert calls == [("fake", tmp_path)]


# ---------------------------------------------------------------------------
# Instrument selection (regressions from interactive testing)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,key,others,expected", [
    ("GAIA", "GAIA0", ["GAIA2", "GAIA3", "Gaia2m"], True),   # every Gaia release is "GAIA" at SVO
    ("GAIA", "GAIA3", ["GAIA0"], True),
    ("Johnson", "Johnson", ["Bessell"], False),             # name == key: path unchanged
    ("ACS_WFC", "ACS_WFC", ["ACS_HRC"], False),
    ("Cousins", "Cousins_old", ["Bessell"], False),          # differs but nothing to clash with
])
def test_svo_instruments_that_share_a_name_get_their_own_directory(name, key, others, expected):
    from sed_tools.svo_filter_grabber import _would_collide

    assert _would_collide(name, key, others) is expected


def test_njm_instrument_match_is_exact(monkeypatch):
    from sed_tools.njm_filter_grabber import NJMFilterGrabber

    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    njm = NJMFilterGrabber(base_dir="/tmp/unused")
    monkeypatch.setattr(njm, "discover_instruments", lambda fac: ["GAIA", "ACS_WFC"])
    assert njm.find_instrument("GAIA", "GAIA0") is None        # was: prefix-matched to GAIA
    assert njm.find_instrument("GAIA", "gaia") == "GAIA"
    assert njm.find_instrument("HST", "ACS-WFC") == "ACS_WFC"


def test_facility_wide_status_applies_to_every_instrument():
    assert catalog.status_for("GAIA", "GAIA0")[0] == catalog.TOTAL_ONLY
    assert catalog.status_for("GAIA", "GAIA3")[0] == catalog.TOTAL_ONLY
    assert catalog.status_for("LSST", "LSST") is None


# ----------------------------------------------------------------------
# original-source downloads fail fast when a host is down
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# An unknown tabular instrument is a clear ValueError
# ----------------------------------------------------------------------
def test_tabular_names_its_instruments_for_an_unknown_one(tmp_path):
    from sed_tools.component_sources import tabular

    with pytest.raises(ValueError, match=r"tabular has no instrument 'NOPE'; choose one of: HAWKI"):
        tabular.build(tmp_path, instrument="NOPE")


# ----------------------------------------------------------------------
# Spelling
# ----------------------------------------------------------------------
