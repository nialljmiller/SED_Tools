"""Builders that used to skip a band or file and still report success now stop.

Offline: every download goes through a fake session or client.
"""

from __future__ import annotations

import numpy as np
import pytest

from sed_tools.component_sources._common import SimpleSource, Table, build_simple, ensure_published
from sed_tools.filter_metadata import PHOTON, SVOFilterRecord, write_filter_file


class _Resp:
    def __init__(self, content, status=200):
        self.content = content if isinstance(content, bytes) else content.encode()
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            import requests

            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(self, files):
        self.files, self.headers, self.calls = files, {}, []

    def get(self, url, timeout=None, **kw):
        self.calls.append(url)
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


class _SVO:
    """SVO metadata client that knows only ``records``."""

    def __init__(self, records):
        self.records = records

    def get_filter(self, fid):
        return self.records.get(fid)


def _record(fid):
    return SVOFilterRecord(fid, PHOTON, np.array([4000.0, 5000.0]), np.array([0.5, 0.6]),
                           wavelength_unit="Angstrom", components="filter")


def _published(inst, band):
    inst.mkdir(parents=True, exist_ok=True)
    write_filter_file(inst / f"{band}.dat", [4000.0, 5000.0], [0.5, 0.6], detector_type=PHOTON,
                      source="svo", wavelength_unit="Angstrom", extra={"Components": "filter"})


# ---------------------------------------------------------------------------
# _common.ensure_published / build_simple
# ---------------------------------------------------------------------------
def test_ensure_published_raises_for_ids_svo_lacks_and_writes_no_index(tmp_path):
    inst = tmp_path / "Fac" / "Cam"
    client = _SVO({"Fac/Cam.g": _record("Fac/Cam.g")})
    with pytest.raises(RuntimeError, match="SVO has no Fac/Cam.r"):
        ensure_published(inst, {"g": "Fac/Cam.g", "r": "Fac/Cam.r"}, client)
    assert (inst / "g.dat").is_file() and not (inst / "r.dat").exists()
    assert not (inst / "Cam").exists()  # no MESA index for an incomplete set


def test_ensure_published_writes_index_when_complete(tmp_path):
    inst = tmp_path / "Fac" / "Cam"
    ensure_published(inst, {"g": "Fac/Cam.g"}, _SVO({"Fac/Cam.g": _record("Fac/Cam.g")}))
    assert (inst / "Cam").read_text() == "g.dat\n"


def test_build_simple_refuses_a_filter_band_without_a_published_curve(tmp_path):
    inst = tmp_path / "Fac" / "Cam"
    _published(inst, "g")
    spec = SimpleSource("Fac", "Cam", filters={"g": Table("f"), "r": Table("f")})
    with pytest.raises(FileNotFoundError, match="no published curve for r"):
        build_simple(spec, tmp_path, session=_Session({"f": "4000 0.5\n5000 0.8\n"}), svo_client=_SVO({}))
    assert not (inst / "manifest.json").exists() and not (inst / "components").exists()


def test_build_simple_refuses_a_composite_band_without_a_published_curve(tmp_path):
    inst = tmp_path / "Fac" / "Cam"
    _published(inst, "g")

    def make(session):
        return np.array([4000.0, 5000.0]), np.array([0.5, 0.5])

    spec = SimpleSource("Fac", "Cam", filters={}, composites={"z": ("filter+qe", make)})
    with pytest.raises(FileNotFoundError, match="no published curve for z"):
        build_simple(spec, tmp_path, session=_Session({}), svo_client=_SVO({}))
    assert not (inst / "manifest.json").exists()
