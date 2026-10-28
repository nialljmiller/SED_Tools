"""SVO FPS access fails fast with one clear error when the host is unreachable (offline)."""

from types import SimpleNamespace

import pytest
import requests

from sed_tools import filter_metadata as fm
from sed_tools.filter_library import FilterLibrary
from sed_tools.svo_filter_grabber import FilterRow, SVOFilterBrowser


class DownSession:
    """Session whose host never answers the TCP handshake."""

    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        raise requests.ConnectTimeout("connect timed out")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.delenv(fm.SVO_FPS_URL_ENV, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("test attempted a real network connection")

    monkeypatch.setattr("socket.socket.connect", refuse)


def test_fps_url_default_and_environment_override(monkeypatch):
    assert fm.svo_fps_url() == fm.SVO_FPS_URL == "https://svo2.cab.inta-csic.es/theory/fps/fps.php"
    monkeypatch.setenv("SED_SVO_FPS_URL", "https://mirror.example/fps/fps.php")
    assert fm.svo_fps_url() == "https://mirror.example/fps/fps.php"


def test_connect_timeout_raises_one_clear_error_and_is_not_retried():
    session = DownSession()
    client = fm.SVOMetadataClient(session=session)
    with pytest.raises(fm.SVOUnreachableError) as info:
        client.get_filter("Generic/Johnson.V")
    message = str(info.value)
    assert "SVO Filter Profile Service unreachable" in message
    assert fm.SVO_FPS_URL in message
    assert "NJM" in message and "original-source builders" in message and "SED_SVO_FPS_URL" in message
    assert session.calls[0][1]["timeout"] == (fm.SVO_CONNECT_TIMEOUT, 120.0)
    assert fm.SVO_CONNECT_TIMEOUT <= 10
    # Existing handlers for either exception family still catch it.
    assert isinstance(info.value, requests.RequestException) and isinstance(info.value, RuntimeError)

    with pytest.raises(fm.SVOUnreachableError):
        client.get_filter("Generic/Johnson.B")
    assert len(session.calls) == 1


def test_url_override_is_honoured_by_metadata_and_index(monkeypatch, tmp_path):
    monkeypatch.setenv("SED_SVO_FPS_URL", "https://mirror.example/fps/fps.php")
    session = DownSession()
    browser = SVOFilterBrowser(base_dir=str(tmp_path), session=session)
    with pytest.raises(fm.SVOUnreachableError, match="https://mirror.example/fps/index.php"):
        browser.list_facilities()
    with pytest.raises(fm.SVOUnreachableError, match="https://mirror.example/fps/fps.php"):
        browser.metadata_client.get_filter("Generic/Johnson.V")
    assert [url for url, _ in session.calls] == ["https://mirror.example/fps/index.php",
                                                 "https://mirror.example/fps/fps.php"]
    assert session.calls[0][1]["timeout"] == (fm.SVO_CONNECT_TIMEOUT, 60)


def test_download_stops_at_first_unreachable_filter(tmp_path):
    session = DownSession()
    browser = SVOFilterBrowser(base_dir=str(tmp_path), session=session)
    rows = [FilterRow(f"Generic/Johnson.{b}", b, "Generic", "Johnson", "") for b in "UBV"]
    with pytest.raises(fm.SVOUnreachableError):
        browser.download_filters(rows)
    assert len(session.calls) == 1


def test_filter_library_reports_unreachable_svo(tmp_path):
    library = FilterLibrary(tmp_path)
    library.mirror = SimpleNamespace(fetch_manifest=lambda facility, instrument: None,
                                     discover_filters=lambda facility, instrument: [])
    library.svo = SVOFilterBrowser(base_dir=str(tmp_path), session=DownSession())
    info = library.available("NoSuchFacility", "NoSuchCamera")
    assert info["availability_source"] == "unknown"
    assert any("SVO Filter Profile Service unreachable" in e for e in info["errors"])
