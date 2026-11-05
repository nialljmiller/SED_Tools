"""Fakes shared by the API and CLI failure tests: fake grabbers, a filters stand-in and tagging
results, and the guard that fails any real request.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import requests


class NetworkUsed(BaseException):
    """Raised past every ``except Exception`` so a network call cannot be swallowed."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise NetworkUsed("tests must not use the network")

    monkeypatch.setattr(requests.Session, "send", refuse)


GRABBERS = {
    "njm": "sed_tools.njm_spectra_grabber.NJMSpectraGrabber",
    "svo": "sed_tools.svo_spectra_grabber.SVOSpectraGrabber",
    "msg": "sed_tools.msg_spectra_grabber.MSGSpectraGrabber",
    "mast": "sed_tools.mast_spectra_grabber.MASTSpectraGrabber",
}


def fake_grabber(models=(), available=True, metadata=None, downloaded=0, download_error=None):
    """A grabber class; ``metadata`` may be an exception to raise, and ``download_error`` is
    raised after ``downloaded`` spectra have been written."""
    class Grabber:
        base_url = "https://mirror.invalid"

        def __init__(self, base_dir=None, max_workers=None):
            self.base_dir = base_dir

        def is_available(self):
            return available

        def discover_models(self):
            return list(models)

        def get_model_metadata(self, name):
            if metadata is None:
                raise RuntimeError(f"no metadata for {name}")  # as the real grabbers do
            if isinstance(metadata, Exception):
                raise metadata
            return metadata

        def download_model_spectra(self, name, meta, **cuts):
            for i in range(downloaded):
                path = Path(self.base_dir) / name / f"spectrum{i}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("1000 1\n2000 1\n")
            if download_error is not None:
                raise download_error
            return downloaded

    return Grabber


@pytest.fixture
def filters(tmp_path, monkeypatch):
    from sed_tools.api import Filters

    monkeypatch.setattr(Filters, "_filter_root", tmp_path / "filters")
    monkeypatch.setattr(Filters, "_catalogues", {})
    return Filters


CLEAN = {"total": 1, "converted": [], "skipped_already": [], "skipped_invalid": [], "skipped_index": [],
         "error": [], "recovered": [], "detection_stats": {}, "catalog_units": None}


def tagged(path):
    return SimpleNamespace(path=path, detector_type="photon", source="svo", filter_id="F", changed=True,
                           migrated=False, note="", kept=False, resolved_type="photon", resolved_source="svo")
