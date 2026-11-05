"""Saved upstream responses for the builder tests, and the checks every published instrument meets.

The files are in tests/fixtures/builders; its README says where each comes from and how it was cut.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from sed_tools.filter_components import InstrumentManifest
from sed_tools.filter_metadata import read_filter_header, released_mesa_reads

FIX = Path(__file__).parent / "fixtures" / "builders"


def saved() -> dict:
    """URL -> the bytes it served, for every saved response."""
    out = {}
    for url, spec in json.loads((FIX / "sources.json").read_text()).items():
        out[url] = (FIX / spec).read_bytes() if isinstance(spec, str) else gzip.decompress(
            (FIX / spec["gunzip"]).read_bytes())
    return out


class Resp:
    def __init__(self, content, status=200):
        self.content, self.status_code, self.headers = content, status, {}
        self.text = content.decode("utf-8", "replace")

    def raise_for_status(self):
        if self.status_code != 200:
            import requests

            raise requests.HTTPError(str(self.status_code))


class Session:
    """Serves the saved responses (or ``files``); any other URL is a 404."""

    def __init__(self, files=None):
        self.files, self.headers, self.calls = saved() if files is None else files, {}, []

    def get(self, url, timeout=None, params=None, **kw):
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        self.calls.append(url)
        return Resp(self.files[url]) if url in self.files else Resp(b"", 404)


def curve(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1), ndmin=2).T


def check_published(inst_dir, facility, instrument, bands, components, detector):
    """Curves, headers, MESA index and manifest of an instrument built from its team's files."""
    assert sorted(p.stem for p in inst_dir.glob("*.dat")) == sorted(bands)
    assert (inst_dir / instrument).read_text().split() == sorted(f"{b}.dat" for b in bands)
    manifest = InstrumentManifest.load(inst_dir)
    assert (manifest.facility, manifest.instrument) == (facility, instrument)
    for band in bands:
        path = inst_dir / f"{band}.dat"
        header = read_filter_header(path)
        assert header["FilterID"] == f"{facility}/{instrument}.{band}"
        assert (header["DetectorType"], header["DetectorTypeSource"]) == (detector, "original")
        assert (header["Components"], header["ComponentsSource"]) == (components, "original")
        assert header["Sources"].startswith("http")
        wl, tr = curve(path)
        assert released_mesa_reads(path) and np.all(np.diff(wl) > 0) and tr.min() >= 0 and tr.max() > 0
        assert manifest.published_bands[band] == components
    assert manifest.detector_type_sources == dict.fromkeys(bands, "original")
    return manifest
