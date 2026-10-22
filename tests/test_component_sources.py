"""Offline tests for the component builders' shared machinery."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

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
def test_table_reads_nm_percent_and_gzip():
    text = "# wavelength(nm) T(%)\n500 10\n400 50\n600 0\n"
    session = _Session({"u": gzip.compress(text.encode())})
    wl, tr = Table("u", wl_scale=10.0, tr_scale=0.01).read(session)
    np.testing.assert_allclose(wl, [4000, 5000, 6000])
    np.testing.assert_allclose(tr, [0.5, 0.1, 0.0])


def test_table_floors_negative_measured_transmission():
    wl, tr = Table("u").parse(b"4000,-0.0001\n5000,0.8\n")
    np.testing.assert_allclose(wl, [4000, 5000])
    np.testing.assert_allclose(tr, [0.0, 0.8])


def test_product_is_on_common_range():
    a = (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0, 1.0]))
    b = (np.array([2.0, 5.0]), np.array([0.5, 0.5]))
    wl, tr = product([a, b])
    assert wl[0] == 2.0 and wl[-1] == 4.0
    np.testing.assert_allclose(tr, 0.5)


def test_percent_component_is_rejected(tmp_path):
    with pytest.raises(ComponentError, match="percent"):
        add_component(tmp_path, FILTER, [4000.0, 5000.0], [10.0, 95.0], band="g")


def test_edge_policy_holds_component_edge_values():
    band = Curve(np.linspace(4000, 6000, 21), np.ones(21), {})
    short = Curve(np.array([4500.0, 5500.0]), np.array([0.5, 0.8]), {})
    grid, prod = multiply_curves([band, short], outside="edge")
    assert prod[grid == 4000.0][0] == pytest.approx(0.5)
    assert prod[grid == 6000.0][0] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# build_simple
# ---------------------------------------------------------------------------
def test_build_simple_writes_components_and_keeps_svo_siblings(tmp_path):
    inst = tmp_path / "filters" / "Fac" / "Cam"
    inst.mkdir(parents=True)
    wl = np.linspace(4000, 6000, 21)
    for band, comps in (("g", "filter+instrument"), ("g_filter", "filter")):
        write_filter_file(inst / f"{band}.dat", wl, np.ones_like(wl), detector_type=PHOTON, source="svo",
                          wavelength_unit="Angstrom", extra={"Components": comps})
    files = {"f": "400 0.9\n700 0.9\n", "o": "300 0.8\n800 0.8\n", "q": "300 0.5\n800 0.5\n"}
    spec = SimpleSource("Fac", "Cam", filters={"g": Table("f", wl_scale=10)},
                        optics=(Table("o", wl_scale=10),), qe=Table("q", wl_scale=10))

    build_simple(spec, tmp_path / "filters", session=_Session(files), svo_client=object())

    m = InstrumentManifest.load(inst)
    assert m.component_file(FILTER, "g") == "components/filter/g.dat"  # builder's own filter wins
    assert m.component_file(FILTER, "g_filter") == "g_filter.dat"      # SVO sibling kept
    assert m.buildable() == ["filter", "filter+optics", "filter+qe", "filter+optics+qe"]
    assert m.primary_bands() == ["g"]


def test_original_filter_source_builds_without_svo_and_preserves_all_bands(tmp_path):
    files = {"a": b"400 50\n500 80\n", "b": b"400 20\n500 90\n"}
    spec = SimpleSource("Fac", "Cam", filters={
        "a": Table("a", wl_scale=10, tr_scale=0.01),
        "b": Table("b", wl_scale=10, tr_scale=0.01),
    }, published_from_filters=True)
    directory = build_simple(spec, tmp_path, session=_Session(files), svo_client=object())
    manifest = InstrumentManifest.load(directory)
    assert manifest.bands == ["a", "b"]
    assert manifest.published_bands == {"a": "filter", "b": "filter"}
    assert (directory / "Cam").read_text().splitlines() == ["a.dat", "b.dat"]
    assert manifest.component_file(FILTER, "a") == "a.dat"


def test_original_filter_source_does_not_leave_partial_download(tmp_path):
    spec = SimpleSource("Fac", "Cam", filters={"a": Table("a"), "b": Table("missing")},
                        published_from_filters=True)
    with pytest.raises(RuntimeError):
        build_simple(spec, tmp_path, session=_Session({"a": b"4000 0.5\n5000 0.8\n"}))
    assert not (tmp_path / "Fac" / "Cam").exists()


def test_eso_dual_band_channels_are_distinct():
    from sed_tools.component_sources.eso_tables import SOURCES

    spec = SOURCES["SPHERE"]
    left = spec.filters["D_H23_H2"]
    right = spec.filters["D_H23_H3"]
    assert left.url == right.url
    np.testing.assert_allclose(left.parse(b"400 0.2 0.8\n500 0.3 0.7\n")[1], [0.2, 0.3])
    np.testing.assert_allclose(right.parse(b"400 0.2 0.8\n500 0.3 0.7\n")[1], [0.8, 0.7])


# ---------------------------------------------------------------------------
# HST
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("component,band,role", [
    ("acs_f606w_wfc", "F606W", FILTER),
    ("wfc3uvis1_f606w", "F606W", FILTER),
    ("nic2_f160w", "F160W", FILTER),
    ("acs_wfc_ccd1", "F606W", QE),
    ("wfc3_ir_qe", "F160W", QE),
    ("wfpc2_dqewfc4", "F606W", QE),
    ("acs_sbc_mama", "F150LP", QE),
    ("acs_f150lp_tds_def", "F150LP", QE),   # correction named after the filter
    ("wfc3uvis1_cor", "F606W", QE),
    ("hst_ota", "F606W", OPTICS),
    ("acs_wfc_ebe_win12f", "F606W", OPTICS),
])
def test_hst_classify(component, band, role):
    from sed_tools.component_sources.hst import classify

    assert classify(component, band) == role


def test_hst_graph_traversal():
    from sed_tools.component_sources.hst import SynphotTables

    t = SynphotTables.__new__(SynphotTables)
    # node 1 -> (acs) 2 -> (wfc1) 3 -> (f606w|default) 4 -> end
    t.innode = np.array([1, 2, 3, 3])
    t.outnode = np.array([2, 3, 4, 4])
    t.keyword = np.array(["acs", "wfc1", "f606w", "default"])
    t.compname = np.array(["hst_ota", "acs_wfc_ccd1", "acs_f606w_wfc", "clear"])
    assert t.chain("acs,wfc1,f606w") == ["hst_ota", "acs_wfc_ccd1", "acs_f606w_wfc"]
    assert t.chain("acs,wfc1") == ["hst_ota", "acs_wfc_ccd1"]
    with pytest.raises(ValueError, match="not in graph"):
        t.chain("acs,wfc1,f999w")


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


def test_total_only_instruments_explain_why(tmp_path):
    inst = tmp_path / "GAIA" / "GAIA"
    inst.mkdir(parents=True)
    write_filter_file(inst / "G.dat", [4000.0, 9000.0], [1.0, 1.0], detector_type=PHOTON, source="svo",
                      wavelength_unit="Angstrom", extra={"Components": "filter+qe+instrument"})
    assert "only total passbands" in InstrumentManifest.load(inst).describe()["component_status"]


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
