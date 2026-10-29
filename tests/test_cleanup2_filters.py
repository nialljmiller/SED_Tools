"""Second cleanup of the filters area: handlers whose wording or path changed.

Every source is mocked; nothing here touches the network.
"""

import logging
from types import SimpleNamespace

import pytest
import requests

from sed_tools.njm_filter_grabber import NJMFilterGrabber

NJM = "https://nillmill.ddns.net/sed_tools"


def _refusing_grabber(monkeypatch, status):
    """An NJM grabber whose HEAD and GET both answer ``status``."""
    monkeypatch.setattr("sed_tools.njm_filter_grabber.time.sleep", lambda seconds: None)

    def answer(url, **kwargs):
        def raise_for_status():
            raise requests.HTTPError(f"{status} Client Error: Forbidden for url: {url}",
                                     response=SimpleNamespace(status_code=status, headers={}))

        return SimpleNamespace(status_code=status, raise_for_status=raise_for_status)

    grabber = NJMFilterGrabber.__new__(NJMFilterGrabber)
    grabber.base_url, grabber._index_data = NJM, None
    grabber.session = SimpleNamespace(head=answer, get=answer, headers={}, verify=True)
    return grabber


def test_a_block_that_does_not_lift_is_reported_as_the_http_status(monkeypatch, caplog):
    grabber = _refusing_grabber(monkeypatch, 403)
    with caplog.at_level(logging.WARNING):
        assert grabber._check_availability() is False
    assert caplog.text.count(f"NJM mirror {NJM} is unavailable (HTTP 403)") == 1
    assert "could not connect" not in caplog.text
    with pytest.raises(requests.ConnectionError, match=r"NJM is unreachable \(HTTP 403\)"):
        grabber._get(f"{NJM}/index.json")
