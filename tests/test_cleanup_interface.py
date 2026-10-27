"""Failures in the public API and CLI are raised, warned about or reported with a non-zero exit."""

import logging

import pytest
import requests

from sed_tools.api import SED


class NetworkUsed(BaseException):
    """Raised past every ``except Exception`` so a network call cannot be swallowed."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise NetworkUsed("tests must not use the network")

    monkeypatch.setattr(requests.Session, "send", refuse)


# ----------------------------------------------------------------------
# Spectral grid sources
# ----------------------------------------------------------------------
GRABBERS = {
    "njm": "sed_tools.njm_spectra_grabber.NJMSpectraGrabber",
    "svo": "sed_tools.svo_spectra_grabber.SVOSpectraGrabber",
    "msg": "sed_tools.msg_spectra_grabber.MSGSpectraGrabber",
    "mast": "sed_tools.mast_spectra_grabber.MASTSpectraGrabber",
}


def fake_grabber(models=(), error=None, available=True, metadata=None, downloaded=0):
    class Grabber:
        base_url = "https://mirror.invalid"

        def __init__(self, base_dir=None, max_workers=None):
            self.base_dir = base_dir

        def is_available(self):
            return available

        def discover_models(self):
            if error is not None:
                raise error
            return list(models)

        def get_model_metadata(self, name):
            return metadata

        def download_model_spectra(self, name, meta, **cuts):
            return downloaded

    return Grabber


@pytest.fixture
def sources(monkeypatch):
    """Replace the four spectral grid grabbers; unspecified ones list nothing."""
    def install(**grabbers):
        for name, target in GRABBERS.items():
            monkeypatch.setattr(target, grabbers.get(name, fake_grabber()))
    return install


def test_query_warns_about_a_failing_source_and_lists_the_others(sources, caplog):
    sources(svo=fake_grabber(error=requests.ConnectionError("SVO down")), msg=fake_grabber(["MSGgrid"]))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"):
        grids = SED.query(include_local=False)
    assert [g.name for g in grids] == ["MSGgrid"]
    assert "svo spectral grid discovery failed: SVO down" in caplog.text


def test_query_warns_when_the_njm_mirror_is_unreachable(sources, caplog):
    sources(njm=fake_grabber(["NJMgrid"], available=False), mast=fake_grabber(["BOSZ"]))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"):
        grids = SED.query(include_local=False)
    assert [g.name for g in grids] == ["BOSZ"]
    assert "njm spectral grid discovery failed: NJM mirror https://mirror.invalid is unreachable" in caplog.text


def test_query_raises_when_the_requested_source_fails(sources):
    sources(svo=fake_grabber(error=requests.ConnectionError("SVO down")))
    with pytest.raises(RuntimeError, match="svo: SVO down"):
        SED.query(source="svo", include_local=False)


def test_query_raises_with_every_error_when_all_sources_fail(sources):
    down = {name: fake_grabber(error=RuntimeError(f"{name} down")) for name in GRABBERS}
    sources(**down)
    with pytest.raises(RuntimeError) as excinfo:
        SED.query(include_local=False)
    for name in GRABBERS:
        assert f"{name}: {name} down" in str(excinfo.value)


def test_query_rejects_an_unknown_source(sources):
    sources()
    with pytest.raises(ValueError, match="Unknown source: sov"):
        SED.query(source="sov")


def test_fetch_error_names_every_source_that_failed(sources, tmp_path, caplog):
    sources(njm=fake_grabber(available=False), msg=fake_grabber(error=ValueError("bad index")))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"), pytest.raises(ValueError) as excinfo:
        SED.fetch("Missing", model_root=tmp_path)
    message = str(excinfo.value)
    assert "Spectral grid 'Missing' not found" in message
    assert "njm: discovery failed: NJM mirror https://mirror.invalid is unreachable" in message
    assert "msg: discovery failed: bad index" in message
    assert "svo: no metadata for Missing" in message
    assert "msg discovery failed: bad index; trying the other sources" in caplog.text
    assert "svo: no metadata returned" in caplog.text


def test_fetch_falls_back_to_the_next_source_with_a_warning(sources, tmp_path, caplog):
    sources(njm=fake_grabber(["Grid"]), svo=fake_grabber(["Grid"], metadata=[{"fid": 1}]))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"), pytest.raises(RuntimeError, match="from svo"):
        SED.fetch("Grid", model_root=tmp_path)
    assert "njm: no metadata returned, trying next" in caplog.text


def test_fetch_rejects_an_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="Unknown source: sov"):
        SED.fetch("Grid", source="sov", model_root=tmp_path)


def test_fetch_raises_when_nothing_was_downloaded(sources, tmp_path):
    sources(svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], downloaded=0))
    with pytest.raises(RuntimeError, match="No spectra of 'Grid' were downloaded from svo"):
        SED.fetch("Grid", source="svo", model_root=tmp_path)


def test_fetch_raises_when_cleaning_leaves_unusable_spectra(sources, tmp_path, monkeypatch):
    sources(svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], downloaded=3))
    summary = {"total": 3, "converted": [], "skipped_invalid": ["a.txt"], "skipped_index": [], "error": ["b.txt"]}
    monkeypatch.setattr("sed_tools.spectra_cleaner.clean_model_dir", lambda *a, **k: summary)
    with pytest.raises(RuntimeError, match="2 of 3 spectra of 'Grid' .* are unusable after cleaning"):
        SED.fetch("Grid", source="svo", model_root=tmp_path)
