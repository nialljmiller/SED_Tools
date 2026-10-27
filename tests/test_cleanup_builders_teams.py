"""Failures in the team builders are either raised or warned about, never silent.

Offline: every remote file is a fake response.
"""

from __future__ import annotations

import io
import json
import logging

import numpy as np
import pytest
import requests

from sed_tools.component_sources import skycalc, swift
from sed_tools.component_sources._original import Fetcher


class _Resp:
    def __init__(self, content=b"", status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(str(self.status_code))


class _Session:
    """Serves ``files``; a URL mapped to an exception raises it; anything else is a 404."""

    def __init__(self, files):
        self.files, self.headers = files, {}

    def get(self, url, timeout=None, **kw):
        found = self.files.get(url)
        if isinstance(found, Exception):
            raise found
        return _Resp(found) if found is not None else _Resp(status=404)


# --------------------------------------------------------------------------- Swift CALDB mirrors
FITS = b"SIMPLE  =                    T" + b" " * 50


def test_caldb_warns_when_a_later_mirror_serves_the_file(caplog):
    primary, mirror = (root + swift.FTRANS for root in swift.CALDB_ROOTS)
    fetch = Fetcher(_Session({primary: requests.exceptions.SSLError("certificate for another host"),
                              mirror: FITS}))
    with caplog.at_level(logging.WARNING, logger=swift.__name__):
        data, url = swift.caldb_get(fetch, swift.FTRANS)
    assert (data, url) == (FITS, mirror)
    assert "certificate for another host" in caplog.text and primary in caplog.text and mirror in caplog.text


def test_caldb_names_every_mirror_when_all_fail():
    primary, mirror = (root + swift.FTRANS for root in swift.CALDB_ROOTS)
    fetch = Fetcher(_Session({primary: b"not fits"}))  # the mirror answers 404
    with pytest.raises(RuntimeError) as err:
        swift.caldb_get(fetch, swift.FTRANS)
    assert f"{primary}: not a FITS file" in str(err.value)
    assert f"{mirror}: 404" in str(err.value)


def test_caldb_does_not_hide_a_programming_error():
    class Broken:
        def get(self, url):
            raise TypeError("bug")

    with pytest.raises(TypeError, match="bug"):
        swift.caldb_get(Broken(), swift.FTRANS)


# --------------------------------------------------------------------------- SkyCalc API and form
def _skytable(echo):
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    primary.header.add_comment("SkyCalc version: 2.0.9")
    for key, value in echo.items():
        primary.header.add_comment(f"{key}={value}")
    lam = np.geomspace(float(echo.get("SKYMODEL.WAVELENGTH.MIN", 1000)),
                       float(echo.get("SKYMODEL.WAVELENGTH.MAX", 1100)), 50)
    table = fits.BinTableHDU.from_columns([fits.Column("lam", "D", array=lam),
                                           fits.Column("trans", "D", array=np.full_like(lam, 0.8))])
    buf = io.BytesIO()
    fits.HDUList([primary, table]).writeto(buf)
    return buf.getvalue()


class _SkyCalcResp:
    def __init__(self, content=b"", text=""):
        self.content, self.text = content, text

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.content)


class _SkyCalcSession:
    """The API host is down or answers; the form answers with ``form_html`` and echoes its inputs."""

    LINK = "<a href=/observing/etc/tmp/abc/skytable.fits>skytable.fits</a>"

    def __init__(self, api_up=False, form_html=LINK, echo_override=None, rmtmp_error=None):
        self.api_up, self.form_html, self.echo_override = api_up, form_html, echo_override or {}
        self.rmtmp_error, self.form_posts = rmtmp_error, []

    def post(self, url, data=None, json=None, timeout=None, **kw):
        if json is not None:
            if not self.api_up:
                raise requests.exceptions.ConnectTimeout("api host down")
            return _SkyCalcResp(b'{"status": "success", "tmpdir": "t1"}')
        self.form_posts.append(dict(data))
        return _SkyCalcResp(text=self.form_html)

    def get(self, url, timeout=None, **kw):
        if url.endswith("/api/rmtmp"):
            if self.rmtmp_error is not None:
                raise self.rmtmp_error
            return _SkyCalcResp(b"{}")
        if "/tmp/t1/" in url:
            return _SkyCalcResp(_skytable({}))
        return _SkyCalcResp(_skytable({**self.form_posts[-1], **self.echo_override}))


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(skycalc.time, "sleep", lambda seconds: None)


def test_skycalc_fallback_to_the_form_warns_with_the_api_error(caplog, no_wait):
    client = skycalc.SkyCalcClient(session=_SkyCalcSession(), interface="auto", pause=0)
    with caplog.at_level(logging.WARNING, logger=skycalc.__name__):
        result = client.fetch("paranal", 1.0, 2.5, 1000.0, 1100.0, 2000)
    assert result.interface == "form"
    assert "api host down" in caplog.text and "using the web form" in caplog.text


def test_skycalc_form_echo_is_checked_after_a_fallback(no_wait):
    session = _SkyCalcSession(echo_override={"SKYMODEL.TARGET.AIRMASS": "1.50"})
    client = skycalc.SkyCalcClient(session=session, interface="auto", pause=0)
    with pytest.raises(RuntimeError, match="echoed SKYMODEL.TARGET.AIRMASS"):
        client.fetch("paranal", 1.0, 2.5, 1000.0, 1100.0, 2000)


def test_skycalc_failure_after_a_fallback_names_both_errors(no_wait):
    session = _SkyCalcSession(form_html="<p>Service unavailable</p>")
    client = skycalc.SkyCalcClient(session=session, interface="auto", pause=0, retries=3)
    with pytest.raises(RuntimeError) as err:
        client.fetch("paranal", 1.0, 2.5, 1000.0, 1100.0, 2000)
    message = str(err.value)
    assert "returned no skytable.fits" in message and "Service unavailable" in message  # the form's error
    assert "api host down" in message and skycalc.BASE_URL in message   # and why the form was used
    assert len(session.form_posts) == 2  # one attempt went to the switch


def test_skycalc_api_temporary_directory_left_behind_is_warned(caplog, no_wait):
    session = _SkyCalcSession(api_up=True, rmtmp_error=requests.exceptions.ReadTimeout("rmtmp timed out"))
    client = skycalc.SkyCalcClient(session=session, interface="api", pause=0)
    with caplog.at_level(logging.WARNING, logger=skycalc.__name__):
        result = client.fetch("paranal", 1.0, 2.5, 1000.0, 1100.0, 2000)
    assert result.interface == "api" and result.version == "2.0.9"
    assert "Could not delete SkyCalc tmpdir t1" in caplog.text and "rmtmp timed out" in caplog.text
