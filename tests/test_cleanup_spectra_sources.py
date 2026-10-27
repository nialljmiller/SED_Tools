"""Failure handling in the spectral-grid sources and spectrum readers.

Every test here runs without the network: the autouse fixture makes any real
HTTP request fail the test, and each grabber gets a fake session.
"""

import pytest
import requests

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
