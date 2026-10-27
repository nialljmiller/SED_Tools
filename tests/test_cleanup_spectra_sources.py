"""Failure handling in the spectral-grid sources and spectrum readers.

Every test here runs without the network: the autouse fixture makes any real
HTTP request fail the test, and each grabber gets a fake session.
"""

import json
import logging

import pytest
import requests

from sed_tools import njm_spectra_grabber as njm
from sed_tools.header_parser import parse_header
from sed_tools.spectrum_io import read_text_spectrum


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail the test on any real HTTP request.

    AssertionError, not a requests exception, so that no handler under test
    can mistake it for a network failure and carry on.
    """
    def refuse(*args, **kwargs):
        raise AssertionError("test tried to use the network")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", refuse)


# ---------------------------------------------------------------- readers

def test_parse_header_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_header(str(tmp_path / "missing.txt"))


def test_read_text_spectrum_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_text_spectrum(tmp_path / "missing.txt")


# ---------------------------------------------------------------- fakes

class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self.content = body if isinstance(body, bytes) else body.encode()
        self.text = self.content.decode("utf-8", "ignore")
        self.headers = headers or {}

    def json(self):
        return json.loads(self.text)  # ValueError for a body that is not JSON

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Answers from a {url: response} table; unknown URLs get a 404.

    A value may be a FakeResponse, an exception to raise, or a callable
    taking the request's keyword arguments and returning either.
    """

    def __init__(self, routes=None):
        self.routes = dict(routes or {})
        self.headers = {}
        self.verify = True
        self.calls = []

    def _answer(self, method, url, params=None, **kwargs):
        if params:
            url = url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        self.calls.append((method, url))
        answer = self.routes.get(url, FakeResponse(404))
        if callable(answer) and not isinstance(answer, FakeResponse):
            answer = answer(**kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def get(self, url, params=None, **kwargs):
        return self._answer("GET", url, params, **kwargs)

    def head(self, url, params=None, **kwargs):
        return self._answer("HEAD", url, params, **kwargs)

    def mount(self, prefix, adapter):
        pass


def warnings_from(caplog, module):
    return [r.getMessage() for r in caplog.records
            if r.name == f"sed_tools.{module}" and r.levelno >= logging.WARNING]


# ---------------------------------------------------------------- NJM mirror

NJM = "https://nillmill.ddns.net/sed_tools"
GRID = f"{NJM}/stellar_models/Grid/"
LOOKUP = "#file_name,teff,logg,meta\na.txt,5000,4.0,0.0\nb.txt,6000,4.5,0.0\n"
SPECTRUM = "# teff = 5000\n" + "".join(f"{wl} 1.0\n" for wl in range(1000, 1400))


def make_njm(tmp_path, monkeypatch, routes):
    session = FakeSession(routes)
    monkeypatch.setattr(njm.requests, "Session", lambda: session)
    return njm.NJMSpectraGrabber(base_dir=str(tmp_path)), session


def njm_online(extra=None):
    routes = {f"{NJM}/index.json": FakeResponse(200, json.dumps({"models": ["Grid", ".hidden"]}))}
    routes.update(extra or {})
    return routes


def test_njm_unreachable_warns_with_every_attempt_and_named_grid_requests_raise(tmp_path, monkeypatch, caplog):
    down = requests.ConnectionError("connection refused")
    with caplog.at_level(logging.WARNING):
        grabber, _ = make_njm(tmp_path, monkeypatch, {
            f"{NJM}/index.json": down, NJM: down, f"{NJM.replace('https', 'http')}/index.json": down})

    assert not grabber.is_available()
    [message] = warnings_from(caplog, "njm_spectra_grabber")
    assert "unreachable" in message and message.count("connection refused") == 3
    assert grabber.discover_models() == []
    with pytest.raises(RuntimeError, match="unreachable"):
        grabber.get_model_metadata("Grid")
    with pytest.raises(RuntimeError, match="unreachable"):
        grabber.download_model_spectra("Grid", {"txt_files": ["a.txt"]})


def test_njm_dropping_certificate_checks_is_warned(tmp_path, monkeypatch, caplog):
    def index(verify=True, **kwargs):
        if verify:
            return requests.exceptions.SSLError("certificate expired")
        return FakeResponse(200, json.dumps({"models": ["Grid"]}))

    with caplog.at_level(logging.WARNING):
        grabber, session = make_njm(tmp_path, monkeypatch, {f"{NJM}/index.json": index})

    assert grabber.is_available() and session.verify is False
    assert any("certificate expired" in m for m in warnings_from(caplog, "njm_spectra_grabber"))


def test_njm_discovery_failure_warns_instead_of_vanishing(tmp_path, monkeypatch, caplog):
    grabber, session = make_njm(tmp_path, monkeypatch, {f"{NJM}/index.json": FakeResponse(200, "not json")})
    session.routes[f"{NJM}/stellar_models"] = FakeResponse(403)

    with caplog.at_level(logging.WARNING):
        assert grabber.discover_models() == []

    messages = warnings_from(caplog, "njm_spectra_grabber")
    assert any("directory listing instead" in m for m in messages)
    assert any("403" in m and "not offered" in m for m in messages)


def test_njm_discovery_reads_cached_index(tmp_path, monkeypatch):
    grabber, _ = make_njm(tmp_path, monkeypatch, njm_online())
    assert grabber.discover_models() == ["Grid"]


def test_njm_metadata_raises_when_spectra_cannot_be_listed(tmp_path, monkeypatch, caplog):
    grabber, _ = make_njm(tmp_path, monkeypatch, njm_online({GRID: FakeResponse(403)}))

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="lookup_table.csv"):
        grabber.get_model_metadata("Grid")
    assert any("no directory listing" in m for m in warnings_from(caplog, "njm_spectra_grabber"))


def test_njm_aux_probe_without_an_answer_raises(tmp_path, monkeypatch):
    grabber, _ = make_njm(tmp_path, monkeypatch, njm_online({
        GRID: FakeResponse(403),
        GRID + "lookup_table.csv": FakeResponse(200, LOOKUP),
        GRID + "flux_cube.bin": lambda **kw: requests.ConnectionError("reset"),
    }))

    with pytest.raises(RuntimeError, match="could not check .*flux_cube.bin"):
        grabber.get_model_metadata("Grid")


def test_njm_metadata_falls_back_to_lookup_table(tmp_path, monkeypatch):
    grabber, _ = make_njm(tmp_path, monkeypatch, njm_online({
        GRID: FakeResponse(403),
        GRID + "lookup_table.csv": FakeResponse(200, LOOKUP),
    }))
    meta = grabber.get_model_metadata("Grid")
    assert meta["txt_files"] == ["a.txt", "b.txt"]
    assert meta["csv_files"] == ["lookup_table.csv"] and meta["bin_files"] == []


def test_njm_grid_info_warns_and_returns_none_without_lookup(tmp_path, monkeypatch, caplog):
    grabber, _ = make_njm(tmp_path, monkeypatch, njm_online())
    with caplog.at_level(logging.WARNING):
        assert grabber.get_grid_info("Grid") is None
    assert any("404" in m for m in warnings_from(caplog, "njm_spectra_grabber"))


def test_njm_parameter_cut_without_lookup_raises_instead_of_downloading_everything(tmp_path, monkeypatch):
    grabber, session = make_njm(tmp_path, monkeypatch, njm_online())
    metadata = {"model_url": GRID, "txt_files": ["a.txt", "b.txt"]}

    with pytest.raises(RuntimeError, match="lookup_table.csv"):
        grabber.download_model_spectra("Grid", metadata, teff_range=(4000, 5500))
    assert not any(url.endswith(".txt") for _, url in session.calls)


def test_njm_failed_downloads_raise_after_the_rest(tmp_path, monkeypatch):
    def broken_stream(**kwargs):
        response = FakeResponse(200, SPECTRUM)
        def iter_content(chunk_size=1):
            yield response.content[:2000]
            raise requests.exceptions.ChunkedEncodingError("connection dropped")
        response.iter_content = iter_content
        return response

    grabber, _ = make_njm(tmp_path, monkeypatch, njm_online({
        GRID + "a.txt": FakeResponse(200, SPECTRUM),
        GRID + "b.txt": broken_stream,
        GRID + "lookup_table.csv": FakeResponse(200, LOOKUP),
    }))
    metadata = {"model_url": GRID, "txt_files": ["a.txt", "b.txt", "c.txt"], "csv_files": ["lookup_table.csv"]}

    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("Grid", metadata)

    message = str(err.value)
    assert "2 file(s)" in message and "b.txt: connection dropped" in message and "c.txt: 404" in message
    model_dir = tmp_path / "Grid"
    assert (model_dir / "a.txt").exists() and (model_dir / "lookup_table.csv").exists()
    assert not (model_dir / "b.txt").exists()  # partial file removed


def test_njm_complete_download_returns_count(tmp_path, monkeypatch):
    grabber, _ = make_njm(tmp_path, monkeypatch, njm_online({
        GRID + "lookup_table.csv": FakeResponse(200, LOOKUP),
        GRID + "a.txt": FakeResponse(200, SPECTRUM),
        GRID + "b.txt": FakeResponse(200, SPECTRUM),
    }))
    metadata = {"model_url": GRID, "txt_files": ["a.txt", "b.txt"]}
    assert grabber.download_model_spectra("Grid", metadata, teff_range=(4000, 5500)) == 1
    assert sorted(p.name for p in (tmp_path / "Grid").iterdir()) == ["a.txt"]


def test_njm_wavelength_cut_failures_are_raised(tmp_path, monkeypatch, caplog):
    grabber, _ = make_njm(tmp_path, monkeypatch, njm_online({
        GRID + "a.txt": FakeResponse(200, SPECTRUM),
        GRID + "b.txt": FakeResponse(200, SPECTRUM),
        f"{NJM}/stellar_models/Grid/a.txt?wl_min=1100&wl_max=1200": requests.ConnectionError("probe refused"),
    }))
    real_trim = njm._trim_spectrum_file

    def trim(path, wl_range):
        if path.endswith("b.txt"):
            raise PermissionError("read-only")
        return real_trim(path, wl_range)

    monkeypatch.setattr(njm, "_trim_spectrum_file", trim)
    metadata = {"model_url": GRID, "txt_files": ["a.txt", "b.txt"]}

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="b.txt: wavelength cut failed: read-only"):
        grabber.download_model_spectra("Grid", metadata, wl_range=(1100, 1200))

    assert any("probe refused" in m for m in warnings_from(caplog, "njm_spectra_grabber"))
    wl, _ = read_text_spectrum(tmp_path / "Grid" / "a.txt")
    assert wl.min() == 1100 and wl.max() == 1200
