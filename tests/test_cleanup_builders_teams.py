"""Failures in the team builders are either raised or warned about, never silent.

Offline: every remote file is a fake response.
"""

from __future__ import annotations

import logging

import pytest
import requests

from sed_tools.component_sources import swift
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
