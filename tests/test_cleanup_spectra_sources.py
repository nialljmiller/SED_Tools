"""Failure handling in the spectral-grid sources and spectrum readers.

Every test here runs without the network: the autouse fixture makes any real
HTTP request fail the test, and each grabber gets a fake session.
"""

import gzip
import json
import logging

import h5py
import numpy as np
import pytest
import requests

from sed_tools import mast_spectra_grabber as mast
from sed_tools import msg_spectra_grabber as msg
from sed_tools import njm_spectra_grabber as njm
from sed_tools import svo_spectra_grabber as svo
from sed_tools.header_parser import parse_header
from sed_tools.spectrum_io import read_text_spectrum
from sed_tools.svo_regen_spectra_lookup import regenerate_lookup_table


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
    with pytest.raises(FileNotFoundError, match="missing.txt"):
        read_text_spectrum(tmp_path / "missing.txt")


# ---------------------------------------------------------------- lookup table

def write_spectra(directory, headers):
    directory.mkdir(exist_ok=True)
    for name, header in headers.items():
        (directory / name).write_text(header + "1000 1.0\n2000 2.0\n")


def test_lookup_table_leaves_out_parameters_no_spectrum_states(tmp_path, capsys):
    grid = tmp_path / "bbody"
    write_spectra(grid, {"bb_3000.txt": "# teff = 3000\n", "bb_4000.txt": "# teff = 4000\n"})

    regenerate_lookup_table(str(grid))

    assert (grid / "lookup_table.csv").read_text().splitlines() == [
        "#file_name, teff", "bb_3000.txt,3000.0", "bb_4000.txt,4000.0"]
    assert "no logg column" in capsys.readouterr().out


def test_lookup_table_raises_when_only_some_spectra_state_a_parameter(tmp_path):
    grid = tmp_path / "mixed"
    write_spectra(grid, {"a.txt": "# teff = 3000\n# logg = 4.0\n", "b.txt": "# teff = 4000\n",
                         "c.txt": "# teff = 5000\n# logg = n/a\n"})

    with pytest.raises(ValueError, match="2 of 3 spectra .* do not state logg in their headers: b.txt, c.txt"):
        regenerate_lookup_table(str(grid))
    assert not (grid / "lookup_table.csv").exists()


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
        grabber, session = make_njm(tmp_path, monkeypatch, {
            f"{NJM}/index.json": down, NJM: down, f"{NJM.replace('https', 'http')}/index.json": down})

    assert not grabber.is_available()
    [message] = warnings_from(caplog, "njm_spectra_grabber")
    # No answer means the mirror is down: one attempt, not three (a silent outage cost 26 s).
    assert "unreachable" in message and message.count("connection refused") == 1 and len(session.calls) == 1
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


# ---------------------------------------------------------------- SVO

SVO_INDEX = "http://svo2.cab.inta-csic.es/theory/newov2/index.php"
SSAP = "http://svo2.cab.inta-csic.es/theory/newov2/ssap.php"
SSAP_QUERY = f"{SSAP}?model=Grid&REQUEST=queryData&FORMAT=votable"


def svo_spectrum_url(fid):
    return f"{SSAP}?model=Grid&fid={fid}&format=ascii"


def make_svo(tmp_path, monkeypatch, routes):
    session = FakeSession(routes)
    monkeypatch.setattr(svo.requests, "Session", lambda: session)
    monkeypatch.setattr(svo.time, "sleep", lambda seconds: None)
    return svo.SVOSpectraGrabber(base_dir=str(tmp_path)), session


def test_svo_discovery_failure_warns_instead_of_vanishing(tmp_path, monkeypatch, caplog):
    grabber, _ = make_svo(tmp_path, monkeypatch, {SVO_INDEX: requests.ConnectionError("no route")})
    with caplog.at_level(logging.WARNING):
        assert grabber.discover_models() == []
    # One plain line (the raw urllib3 text was several lines' worth for a proxy or timeout).
    assert any("could not be reached" in m and "not offered" in m for m in warnings_from(caplog, "svo_spectra_grabber"))


def test_svo_metadata_warns_at_each_fallback_and_raises_when_a_probe_fails(tmp_path, monkeypatch, caplog):
    grabber, _ = make_svo(tmp_path, monkeypatch, {
        SSAP_QUERY: FakeResponse(503),
        f"{SVO_INDEX}?models=Grid": requests.ConnectionError("index down"),
        svo_spectrum_url(7): requests.ConnectionError("probe refused"),
    })

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="FID 7 of Grid: probe refused"):
        grabber.get_model_metadata("Grid")

    messages = warnings_from(caplog, "svo_spectra_grabber")
    assert any("HTTP 503" in m and "index page" in m for m in messages)
    assert any("index down" in m and "probing" in m for m in messages)
    assert not (tmp_path / ".cache" / "svo_fids" / "Grid.json").exists()


def test_svo_metadata_raises_when_no_method_finds_spectra(tmp_path, monkeypatch):
    grabber, _ = make_svo(tmp_path, monkeypatch, {SSAP_QUERY: FakeResponse(200, "<VOTABLE/>")})
    with pytest.raises(RuntimeError, match="no spectra for Grid: SSAP query: the response lists none"):
        grabber.get_model_metadata("Grid")


def test_svo_truncated_ssap_response_raises_and_is_not_cached(tmp_path, monkeypatch):
    body = "".join(f"<TD>ssap.php?fid={i}</TD>" for i in range(1, 200))
    grabber, _ = make_svo(tmp_path, monkeypatch, {SSAP_QUERY: FakeResponse(200, body)})
    grabber.ssap_max_bytes = 1000

    with pytest.raises(RuntimeError, match="cap"):
        grabber.get_model_metadata("Grid")
    assert not (tmp_path / ".cache" / "svo_fids" / "Grid.json").exists()


def test_svo_metadata_is_cached_and_a_corrupt_cache_is_warned_about(tmp_path, monkeypatch, caplog):
    body = "<TD>ssap.php?fid=2</TD><TD>ssap.php?fid=1</TD>"
    grabber, session = make_svo(tmp_path, monkeypatch, {SSAP_QUERY: FakeResponse(200, body)})
    cache = tmp_path / ".cache" / "svo_fids" / "Grid.json"

    assert grabber.get_model_metadata("Grid") == [{"fid": 1}, {"fid": 2}]
    assert json.loads(cache.read_text()) == [1, 2]
    session.calls.clear()
    assert grabber.get_model_metadata("Grid") == [{"fid": 1}, {"fid": 2}]
    assert session.calls == []

    cache.write_text("[1, 2")
    with caplog.at_level(logging.WARNING):
        assert grabber.get_model_metadata("Grid") == [{"fid": 1}, {"fid": 2}]
    assert any("unreadable FID cache" in m for m in warnings_from(caplog, "svo_spectra_grabber"))


def test_svo_download_refuses_ranges_it_cannot_apply(tmp_path, monkeypatch):
    grabber, session = make_svo(tmp_path, monkeypatch, {})
    with pytest.raises(ValueError, match="cannot be cut"):
        grabber.download_model_spectra("Grid", [{"fid": 1}], teff_range=(4000, 5000))
    assert session.calls == []


SVO_SPECTRUM = "# teff = 5000\n# logg = 4.5\n" + "".join(f"{wl} 1.0e-10\n" for wl in range(1000, 1100))


def test_svo_failed_downloads_raise_and_leave_no_lookup_table(tmp_path, monkeypatch):
    grabber, _ = make_svo(tmp_path, monkeypatch, {
        svo_spectrum_url(1): FakeResponse(200, SVO_SPECTRUM),
        svo_spectrum_url(2): FakeResponse(200, "<html>Error</html>" + " " * 600),
        svo_spectrum_url(3): requests.ConnectionError("reset by peer"),
    })

    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("Grid", [{"fid": 1}, {"fid": 2}, {"fid": 3}])

    message = str(err.value)
    assert "2 of 3 spectra" in message
    assert "FID 2: content is not spectral data" in message and "FID 3: connection error: reset by peer" in message
    model_dir = tmp_path / "Grid"
    assert (model_dir / "Grid_fid1.txt").exists()
    assert not (model_dir / "lookup_table.csv").exists()


def test_svo_complete_download_writes_lookup_table(tmp_path, monkeypatch):
    grabber, _ = make_svo(tmp_path, monkeypatch, {
        svo_spectrum_url(1): FakeResponse(200, SVO_SPECTRUM),
        svo_spectrum_url(2): FakeResponse(200, SVO_SPECTRUM),
    })
    assert grabber.download_model_spectra("Grid", [{"fid": 1}, {"fid": 2}]) == 2
    lookup = (tmp_path / "Grid" / "lookup_table.csv").read_text().splitlines()
    assert lookup[0] == "#file_name,logg,metallicity,teff" and len(lookup) == 3


# ---------------------------------------------------------------- MSG

MSG_INDEX = "http://user.astro.wisc.edu/~townsend/static.php?ref=msg-grids"
MSG_FILE = "http://user.astro.wisc.edu/msg/grids/sg-Test.h5"


def make_msg(tmp_path, monkeypatch, routes):
    session = FakeSession(routes)
    monkeypatch.setattr(msg.requests, "Session", lambda: session)
    return msg.MSGSpectraGrabber(base_dir=str(tmp_path), max_workers=1), session


def write_msg_grid(path):
    """Three spectra: one good, one with no flux dataset, one with an undecodable wavelength segment."""
    wave = np.linspace(3000.0, 9000.0, 50)
    with h5py.File(path, "w") as f:
        for i, values in enumerate(([5000.0, 6000.0], [-1.0, 0.0], [4.0, 4.5])):
            f[f"vgrid/axes[{i}]/x"] = values
        f["vgrid/v_lin_seq"] = [1, 2, 3]
        good = f.create_group("specsource/specints[1]")
        good["flux"] = np.ones(50)
        good["range/x"] = wave
        no_flux = f.create_group("specsource/specints[2]")
        no_flux["a"] = np.ones(50)
        no_flux["b"] = np.ones(50)
        gapped = f.create_group("specsource/specints[3]")
        gapped["flux"] = np.ones(60)
        gapped["range/ranges[1]/x"] = wave[:30]
        gapped.create_group("range/ranges[2]").attrs["label"] = "no numbers here"


def test_msg_discovery_failure_warns_instead_of_vanishing(tmp_path, monkeypatch, caplog):
    grabber, _ = make_msg(tmp_path, monkeypatch, {MSG_INDEX: FakeResponse(500)})
    with caplog.at_level(logging.WARNING):
        assert grabber.discover_models() == []
    assert any("500" in m and "not offered" in m for m in warnings_from(caplog, "msg_spectra_grabber"))


def test_msg_metadata_raises_for_unknown_grid_and_failed_download(tmp_path, monkeypatch):
    index = f'<a href="{MSG_FILE}">sg-Test</a>'
    grabber, _ = make_msg(tmp_path, monkeypatch, {
        MSG_INDEX: FakeResponse(200, index), MSG_FILE: requests.ConnectionError("dropped")})

    with pytest.raises(RuntimeError, match="unknown to this grabber"):
        grabber.get_model_metadata("sg-Test")
    assert grabber.discover_models() == ["sg-Test"]
    with pytest.raises(RuntimeError, match="could not download .*sg-Test.h5: dropped"):
        grabber.get_model_metadata("sg-Test")


def test_msg_metadata_raises_for_unreadable_or_empty_file(tmp_path, monkeypatch):
    grabber, _ = make_msg(tmp_path, monkeypatch, {})
    grabber.model_urls["sg-Test"] = MSG_FILE
    h5_path = tmp_path / "sg-Test" / "sg-Test.h5"
    h5_path.parent.mkdir()

    h5_path.write_bytes(b"\x89HDF partial download")
    with pytest.raises(RuntimeError, match="could not read .*partial"):
        grabber.get_model_metadata("sg-Test")

    with h5py.File(h5_path, "w") as f:
        f.create_group("other")
    with pytest.raises(RuntimeError, match="no 'specsource' group"):
        grabber.get_model_metadata("sg-Test")


def test_msg_failed_extractions_raise_after_the_rest(tmp_path, monkeypatch):
    grabber, _ = make_msg(tmp_path, monkeypatch, {})
    grabber.model_urls["sg-Test"] = MSG_FILE
    model_dir = tmp_path / "sg-Test"
    model_dir.mkdir()
    write_msg_grid(model_dir / "sg-Test.h5")

    spectra = grabber.get_model_metadata("sg-Test")
    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("sg-Test", spectra)

    message = str(err.value)
    assert "2 of 3 spectra" in message
    assert "sg-Test_fid2.txt: No suitable flux dataset" in message
    assert "sg-Test_fid3.txt: Unable to recover wavelength grid" in message  # was written truncated to 30 points
    wl, _ = read_text_spectrum(model_dir / "sg-Test_fid1.txt")
    assert wl.size == 50
    assert not (model_dir / "lookup_table.csv").exists()


def test_msg_cut_exclusions_are_not_failures(tmp_path, monkeypatch):
    grabber, _ = make_msg(tmp_path, monkeypatch, {})
    grabber.model_urls["sg-Test"] = MSG_FILE
    model_dir = tmp_path / "sg-Test"
    model_dir.mkdir()
    write_msg_grid(model_dir / "sg-Test.h5")

    spectra = grabber.get_model_metadata("sg-Test")[:1]
    assert grabber.download_model_spectra("sg-Test", spectra, teff_range=(5500, 7000)) == 0
    assert grabber.download_model_spectra("sg-Test", spectra, teff_range=(4500, 5500)) == 1
    assert (model_dir / "lookup_table.csv").exists()


def test_msg_axis_role_beyond_the_grid_raises(tmp_path, monkeypatch):
    grabber, _ = make_msg(tmp_path, monkeypatch, {})
    grabber.model_axes["sg-C3K-test"] = {
        "axes": [np.array([5000.0, 6000.0]), np.array([-1.0, 0.0]), np.array([4.0, 4.5])],
        "shape": (2, 2, 2), "vlin": np.array([0]), "roles": {"teff": 0, "meta": 1, "logg": 3}}
    with pytest.raises(RuntimeError, match="logg is mapped to axis 3, but the grid has 3 axes"):
        grabber._params_from_index("sg-C3K-test", 0)


# ---------------------------------------------------------------- MAST (BOSZ)

BOSZ_SCRIPT = f"{mast.SCRIPTS_DIR}/hlsp_bosz_bosz2024_sim_r1000_m%2B0.00_v1_bulkdl.sh"
BOSZ_WAVE = f"{mast.DATA_BASE}/wavelength_grids/bosz2024_wave_r1000.txt"


def bosz_url(teff):
    return (f"{mast.DATA_BASE}/m+0.00/r1000/"
            f"bosz2024_mp_t{teff:05d}_g+4.5_m+0.00_a+0.00_c+0.00_v2_r1000_resam.txt.gz")


def bosz_resam(n):
    """n rows of (flux, continuum); the flux is not monotonic, so it is not read as wavelengths."""
    return gzip.compress("".join(f"{1.0 + (i % 3)} 2.0\n" for i in range(n)).encode())


def test_mast_download_text_distinguishes_missing_from_failed():
    session = FakeSession({"https://x/missing": FakeResponse(404), "https://x/broken": FakeResponse(503)})
    assert mast._download_text("https://x/missing", session=session) == ""
    with pytest.raises(requests.HTTPError):
        mast._download_text("https://x/broken", session=session)


def test_mast_gz_download_raises_for_corrupt_gzip_instead_of_decoding_it():
    good = gzip.compress(b"1 2\n")
    session = FakeSession({
        "https://x/plain": FakeResponse(200, b"1 2\n"),
        "https://x/good": FakeResponse(200, good),
        "https://x/corrupt": FakeResponse(200, good[:-6] + b"\x00" * 6),
        "https://x/gone": FakeResponse(404),
    })
    assert mast._download_text_gz("https://x/plain", session=session) == "1 2\n"
    assert mast._download_text_gz("https://x/good", session=session) == "1 2\n"
    with pytest.raises(ValueError, match="corrupt gzip"):
        mast._download_text_gz("https://x/corrupt", session=session)
    with pytest.raises(requests.HTTPError):
        mast._download_text_gz("https://x/gone", session=session)


def test_mast_missing_wavelength_grid_raises_listing_each_location():
    with pytest.raises(RuntimeError, match="No BOSZ wavelength grid found for r1000") as err:
        mast._try_wave_urls("r1000", FakeSession({BOSZ_WAVE: FakeResponse(200, "1\n2\n")}))
    assert "not an increasing wavelength grid (2 values)" in str(err.value)
    assert str(err.value).count("not found") == 4


def test_mast_unreadable_download_script_raises():
    session = FakeSession({BOSZ_SCRIPT: requests.ConnectionError("timed out")})
    with pytest.raises(RuntimeError, match="metallicities would be missing:\n  .*timed out"):
        mast._gather_urls_from_scripts("r1000", ["m+0.00"], session)
    with pytest.raises(RuntimeError, match="not found"):
        mast._gather_urls_from_scripts("r1000", ["m+0.00"], FakeSession())


def test_mast_metallicity_prompt_rejects_values_it_would_have_dropped(monkeypatch):
    grabber = mast.MASTSpectraGrabber(base_dir="unused", session=FakeSession())
    for typed, error in (("-1.0,abc", "'abc' is not a number"), ("-3.0", "no \\[M/H\\] = -3.0")):
        monkeypatch.setattr("builtins.input", lambda prompt, typed=typed: typed)
        with pytest.raises(ValueError, match=error):
            grabber.get_model_metadata("BOSZ-2024-r1000")
    monkeypatch.setattr("builtins.input", lambda prompt: "-1.0, +0.25,")
    assert grabber.get_model_metadata("BOSZ-2024-r1000")["metals"] == ["m-1.00", "m+0.25"]


def test_mast_failed_spectra_raise_after_the_rest(tmp_path):
    urls = [bosz_url(5750), bosz_url(6000), bosz_url(6250)]
    session = FakeSession({
        BOSZ_SCRIPT: FakeResponse(200, "\n".join(f"curl -O {u}" for u in urls)),
        BOSZ_WAVE: FakeResponse(200, "\n".join(str(3000 + 100 * i) for i in range(12))),
        urls[0]: FakeResponse(200, bosz_resam(12)),
        urls[1]: FakeResponse(404),
        urls[2]: FakeResponse(200, bosz_resam(11)),
    })
    grabber = mast.MASTSpectraGrabber(base_dir=str(tmp_path), max_workers=1, session=session)

    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("BOSZ-2024-r1000", {"reskey": "r1000", "metals": ["m+0.00"]})

    message = str(err.value)
    assert "2 of 3 BOSZ spectra" in message
    assert "t06000" in message and "404" in message and "length mismatch" in message
    model_dir = tmp_path / "BOSZ-2024-r1000"
    assert [p.name for p in model_dir.glob("*.txt")] == [
        "bosz2024_mp_t05750_g_4.5_m_0.00_a_0.00_c_0.00_v2_r1000_resam.txt"]  # '+' sanitised to '_'
    assert not (model_dir / "lookup_table.csv").exists()


def test_mast_wavelength_cut_that_leaves_nothing_is_not_a_failure(tmp_path):
    session = FakeSession({
        BOSZ_SCRIPT: FakeResponse(200, f"curl -O {bosz_url(5750)}"),
        BOSZ_WAVE: FakeResponse(200, "\n".join(str(3000 + 100 * i) for i in range(12))),
        bosz_url(5750): FakeResponse(200, bosz_resam(12)),
    })
    grabber = mast.MASTSpectraGrabber(base_dir=str(tmp_path), session=session)
    info = {"reskey": "r1000", "metals": ["m+0.00"]}
    assert grabber.download_model_spectra("BOSZ-2024-r1000", info, wl_range=(9000, 9500)) == 0


def test_svo_cache_that_cannot_be_read_or_written_is_warned_about(tmp_path, monkeypatch, caplog):
    grabber, _ = make_svo(tmp_path, monkeypatch, {SSAP_QUERY: FakeResponse(200, "<TD>ssap.php?fid=3</TD>")})
    (tmp_path / ".cache" / "svo_fids" / "Grid.json").mkdir()  # a directory where the cache file goes

    with caplog.at_level(logging.WARNING):
        assert grabber.get_model_metadata("Grid") == [{"fid": 3}]

    messages = warnings_from(caplog, "svo_spectra_grabber")
    assert any("unreadable FID cache" in m for m in messages)
    assert any("could not cache the FID list" in m for m in messages)


def test_mast_filter_keeps_names_it_cannot_parse_so_the_download_reports_them():
    odd = "https://x/m+0.00/r1000/bosz2024_xx_t05750_r1000_resam.txt.gz"
    assert mast._filter_urls_by_params([odd], teff_range=(5000, 6000)) == [odd]


def test_mast_failed_first_spectrum_does_not_stop_the_rest(tmp_path):
    urls = [bosz_url(5750), bosz_url(6000)]
    session = FakeSession({
        BOSZ_SCRIPT: FakeResponse(200, "\n".join(f"curl -O {u}" for u in urls)),
        BOSZ_WAVE: FakeResponse(200, "\n".join(str(3000 + 100 * i) for i in range(12))),
        urls[0]: requests.ConnectionError("first one dropped"),
        urls[1]: FakeResponse(200, bosz_resam(12)),
    })
    grabber = mast.MASTSpectraGrabber(base_dir=str(tmp_path), max_workers=1, session=session)

    with pytest.raises(RuntimeError, match="1 of 2 BOSZ spectra.*\n.*t05750.*first one dropped"):
        grabber.download_model_spectra("BOSZ-2024-r1000", {"reskey": "r1000", "metals": ["m+0.00"]})
    assert len(list((tmp_path / "BOSZ-2024-r1000").glob("*t06000*.txt"))) == 1


def test_msg_discover_and_run_finishes_other_grids_then_raises(tmp_path, monkeypatch):
    index = (f'<a href="{MSG_FILE}">a</a>'
             '<a href="http://user.astro.wisc.edu/msg/grids/sg-Other.h5">b</a>')
    grabber, _ = make_msg(tmp_path, monkeypatch, {MSG_INDEX: FakeResponse(200, index)})
    done = []

    def metadata(name):
        if name == "sg-Test":
            raise RuntimeError("download failed")
        return [{"fid": 1}]

    def download(name, meta):
        done.append(name)
        return 1

    monkeypatch.setattr(grabber, "get_model_metadata", metadata)
    monkeypatch.setattr(grabber, "download_model_spectra", download)

    with pytest.raises(RuntimeError, match="1 of 2 grid\\(s\\) failed:\n  sg-Test: download failed"):
        grabber.discover_and_run()
    assert done == ["sg-Other"]
