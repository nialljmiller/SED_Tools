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
