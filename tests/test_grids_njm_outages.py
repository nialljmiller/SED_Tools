"""Spectral grids with NJM down: one connect timeout for a silent outage, HEAD to confirm
an error status, plain HTTP only when TLS fails both ways, grid indexes asked once, and no spectra asked
while NJM is paused.
"""

from __future__ import annotations

from types import SimpleNamespace

import requests

from sed_tools import njm_filter_grabber

from helpers_filters import NJM


# ----------------------------------------------------------------------
# The NJM spectra grabber fails fast
# ----------------------------------------------------------------------
def _spectra_grabber(tmp_path, monkeypatch, answers):
    """An NJMSpectraGrabber whose session answers each (method, url, verify) from ``answers``."""
    from sed_tools import njm_spectra_grabber

    calls = []

    def answer(method, url, verify=True, timeout=None, **kwargs):
        calls.append((method, url, verify, timeout))
        result = answers.get((method, url, verify), SimpleNamespace(status_code=404))
        if isinstance(result, Exception):
            raise result
        return result

    session = SimpleNamespace(headers={}, verify=True, get=lambda url, **kw: answer("GET", url, **kw),
                              head=lambda url, **kw: answer("HEAD", url, **kw))
    monkeypatch.setattr(njm_spectra_grabber.requests, "Session", lambda: session)
    return njm_spectra_grabber.NJMSpectraGrabber(base_dir=str(tmp_path)), calls


def test_a_silent_spectra_outage_costs_one_connect_timeout(tmp_path, monkeypatch):
    index = f"{NJM}/index.json"
    grabber, calls = _spectra_grabber(tmp_path, monkeypatch, {("GET", index, True): requests.ConnectTimeout("t")})
    assert not grabber.is_available()
    assert calls == [("GET", index, True, (njm_filter_grabber.NJM_CONNECT_TIMEOUT, 10))]


def test_a_spectra_mirror_that_answers_with_an_error_status_is_confirmed_by_head(tmp_path, monkeypatch):
    index = f"{NJM}/index.json"
    grabber, calls = _spectra_grabber(tmp_path, monkeypatch, {
        ("GET", index, True): SimpleNamespace(status_code=503), ("HEAD", NJM, True): SimpleNamespace(status_code=403)})
    assert grabber.is_available() and [c[:2] for c in calls] == [("GET", index), ("HEAD", NJM)]


def test_plain_http_only_when_tls_fails_both_ways(tmp_path, monkeypatch):
    index, plain = f"{NJM}/index.json", f"{NJM.replace('https', 'http')}/index.json"
    ok = SimpleNamespace(status_code=200, json=lambda: {"models": ["Grid"]})
    grabber, calls = _spectra_grabber(tmp_path, monkeypatch, {
        ("GET", index, True): requests.exceptions.SSLError("bad certificate"),
        ("GET", index, False): requests.exceptions.SSLError("handshake failed"),
        ("GET", plain, False): ok})
    assert grabber.is_available() and grabber.base_url.startswith("http://")
    assert [c[1:3] for c in calls] == [(index, True), (index, False), (plain, False)]
