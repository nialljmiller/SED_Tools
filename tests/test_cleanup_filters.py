"""Filter handlers that now raise or warn instead of carrying on silently (cleanup of the filters area).

Every network call is mocked; a real connection fails the test.
"""

import logging

import pytest

from sed_tools import filter_metadata as fm


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.delenv(fm.SVO_FPS_URL_ENV, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("test attempted a real network connection")

    monkeypatch.setattr("socket.socket.connect", refuse)


def _curve(path, header="Wavelength,Transmission,WavelengthUnit=Angstrom"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n4000.0,0.1\n5000.0,1.0\n6000.0,0.1\n")
    return path


# ----------------------------------------------------------------------
# filter_colors
# ----------------------------------------------------------------------
def test_instrument_colors_refuses_an_unreadable_curve(tmp_path):
    from sed_tools.filter_colors import instrument_colors

    good = _curve(tmp_path / "g.dat")
    empty = tmp_path / "r.dat"
    empty.write_text("Wavelength,Transmission\n")
    with pytest.raises(ValueError, match="r.dat"):
        instrument_colors([good, empty])


# ----------------------------------------------------------------------
# filter_catalogue
# ----------------------------------------------------------------------
def _cached(catalogue, rel, text, checked):
    import json

    path = catalogue.cache_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.with_name(path.name + ".http.json").write_text(json.dumps({"checked": checked}))
    return path


def _down_mirror():
    import requests
    from types import SimpleNamespace

    def get(url, **kwargs):
        raise requests.ConnectionError("mirror down")

    return SimpleNamespace(base_url="https://njm.invalid/sed_tools", is_available=lambda: False, _get=get)


def test_unreadable_installed_catalogue_file_is_refused(tmp_path):
    from sed_tools.filter_catalogue import CatalogueError, FilterCatalogue

    root = tmp_path / "filters"
    (root / "Fac" / "Cam").mkdir(parents=True)
    (root / "Fac" / "Cam" / "properties.json").write_text("{not json")
    catalogue = FilterCatalogue(root, mirror=_down_mirror())
    with pytest.raises(CatalogueError, match="properties.json is unreadable"):
        catalogue.properties("Fac", "Cam")


def test_stale_cached_index_used_without_the_mirror_is_reported(tmp_path, caplog):
    from sed_tools.filter_catalogue import FilterCatalogue

    catalogue = FilterCatalogue(tmp_path / "filters", mirror=_down_mirror(), index_max_age=60)
    _cached(catalogue, "index.json", '{"filters": {}}', checked=0)
    with caplog.at_level(logging.WARNING):
        assert catalogue.index() == {"filters": {}}
    assert "Cannot ask NJM (unreachable); using the cached index.json" in caplog.text


def test_offline_catalogue_uses_its_cache_without_a_warning(tmp_path, caplog):
    from sed_tools.filter_catalogue import FilterCatalogue

    catalogue = FilterCatalogue(tmp_path / "filters", offline=True, index_max_age=60)
    _cached(catalogue, "index.json", '{"filters": {}}', checked=0)
    with caplog.at_level(logging.WARNING):
        assert catalogue.index() == {"filters": {}}
    assert caplog.text == ""


def test_vega_path_without_an_index_is_reported(tmp_path, caplog):
    import time

    from sed_tools.filter_catalogue import VEGA_URL, FilterCatalogue

    catalogue = FilterCatalogue(tmp_path / "filters", mirror=_down_mirror())
    catalogue._mirror.is_available = lambda: True
    vega = _cached(catalogue, VEGA_URL, "WAVELENGTH,FLUX\n5000,1e-9\n", checked=time.time())
    with caplog.at_level(logging.WARNING):
        assert catalogue.vega() == vega
    assert "Cannot read NJM's index.json" in caplog.text and VEGA_URL in caplog.text


# ----------------------------------------------------------------------
# filter_components
# ----------------------------------------------------------------------
def test_unknown_components_header_is_reported(caplog):
    from sed_tools.filter_components import InstrumentManifest

    headers = {"g": {"Components": "filter+warp"}, "r": {"Components": "filter"}}
    with caplog.at_level(logging.WARNING):
        manifest = InstrumentManifest.from_headers("Fac", "Cam", headers)
    assert manifest.published_bands == {"r": "filter"}
    assert "Fac/Cam band g" in caplog.text and "'warp'" in caplog.text


# ----------------------------------------------------------------------
# svo_filter_grabber
# ----------------------------------------------------------------------
def _svo_record(filter_id):
    import numpy as np

    return fm.SVOFilterRecord(filter_id, fm.PHOTON, np.array([4000.0, 6000.0]), np.ones(2), "Angstrom", "Filter")


def test_svo_download_saves_what_it_can_then_lists_the_failures(tmp_path):
    import requests

    from sed_tools.filter_components import InstrumentManifest
    from sed_tools.svo_filter_grabber import FilterRow, SVOFilterBrowser

    browser = SVOFilterBrowser(base_dir=str(tmp_path))

    def get_filter(filter_id):
        if filter_id.endswith(".r"):
            return None
        if filter_id.endswith(".i"):
            raise requests.HTTPError("500 Server Error")
        return _svo_record(filter_id)

    browser.metadata_client.get_filter = get_filter
    rows = [FilterRow(f"Fac/Cam.{b}", b, "Fac", "Cam", "") for b in "gri"]
    with pytest.raises(RuntimeError) as info:
        browser.download_filters(rows)
    message = str(info.value)
    assert "2 of 3 SVO filters could not be saved" in message
    assert "Fac/Cam.r: SVO returned no transmission data" in message and "Fac/Cam.i: 500" in message
    directory = tmp_path / "Fac" / "Cam"
    assert (directory / "Cam").read_text() == "g.dat\n"
    assert InstrumentManifest.load(directory).missing_files(directory) == ["i.dat", "r.dat"]


def test_svo_bulk_download_finishes_then_reports_failed_instruments():
    from types import SimpleNamespace

    from sed_tools.svo_filter_grabber import Facility, Instrument, _download_facilities

    downloaded = []

    def download(filters):
        downloaded.append(filters[0])
        if filters[0] == "A/one.g":
            raise RuntimeError("1 of 1 SVO filters could not be saved")

    browser = SimpleNamespace(
        list_instruments=lambda key: [Instrument(key, "one", "One"), Instrument(key, "two", "Two")],
        list_filters=lambda fac, inst: [f"{fac}/{inst}.g"],
        download_filters=download,
    )
    with pytest.raises(RuntimeError, match="Bulk download incomplete:\nA/One: 1 of 1"):
        _download_facilities(browser, [Facility("A", "A"), Facility("B", "B")])
    assert downloaded == ["A/one.g", "A/two.g", "B/one.g", "B/two.g"]


# ----------------------------------------------------------------------
# filter_library
# ----------------------------------------------------------------------
def test_published_reports_each_fallback_and_every_source_failure(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace

    from requests import ConnectionError

    from sed_tools import filter_library
    from sed_tools.component_sources import catalog
    from sed_tools.filter_components import ComponentError
    from sed_tools.filter_library import FilterLibrary

    library = FilterLibrary(tmp_path)
    library.mirror = SimpleNamespace(is_available=lambda: False)
    library.svo = SimpleNamespace(list_facilities=lambda: (_ for _ in ()).throw(RuntimeError("SVO index down")))
    monkeypatch.setitem(catalog.BUILDERS_FOR, ("Fac", "Cam"), catalog.BuilderRef("test", publishes=True))

    def builder(root, **kwargs):
        raise ConnectionError("team site offline")

    monkeypatch.setattr(filter_library, "get_builder", lambda name: builder)
    with caplog.at_level(logging.WARNING), pytest.raises(ComponentError) as info:
        library.published("Fac", "Cam")
    assert str(info.value) == ("No source can provide Fac/Cam: NJM: the mirror is unavailable; "
                               "original source: team site offline; SVO: SVO index down")
    assert "NJM cannot provide Fac/Cam (the mirror is unavailable); trying its original-source builder" in caplog.text
    assert "The original source cannot provide Fac/Cam (team site offline); trying SVO" in caplog.text
