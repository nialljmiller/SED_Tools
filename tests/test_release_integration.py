"""Tests for the 0.5.0 integration items (RELEASE_050_INTEGRATION_TODO.md).

No test uses the network: NJM, SVO and the original sources are stand-ins.
"""

from __future__ import annotations

import importlib
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from sed_tools import njm_filter_grabber
from sed_tools.njm_filter_grabber import NJMFilterGrabber

pcf = importlib.import_module("sed_tools.precompute_flux_cube")


# ----------------------------------------------------------------------
# Item 1: provenance columns of combined and ML-extended grids are not axes
# ----------------------------------------------------------------------
def _spectrum(path, wavelengths):
    path.write_text("".join(f"{w:.1f} {1e-10:.6e}\n" for w in wavelengths))


def _extended_grid(model_dir):
    """An ML-extended grid: each spectrum was extended to a different longest wavelength."""
    model_dir.mkdir(parents=True)
    rows = []
    for i, (teff, logg) in enumerate([(5000.0, 4.0), (6000.0, 4.0), (5000.0, 4.5), (6000.0, 4.5)]):
        wl_max = 20000.0 + 5000.0 * i
        _spectrum(model_dir / f"s{i}.txt", np.linspace(1000.0, wl_max, 30))
        rows.append(f"s{i}.txt,{teff},{logg},0.0,True,1000.0,{wl_max}")
    (model_dir / "lookup_table.csv").write_text(
        "#file_name,teff,logg,metallicity,extended_by_ml,wl_min,wl_max\n" + "\n".join(rows) + "\n")
    return model_dir


def test_an_ml_extended_grid_with_different_wavelength_ends_is_one_cube(tmp_path, capsys):
    # Each row's wl_max differs, so these columns used to split the grid into
    # one "variant" per spectrum, with no flux_cube.bin in the grid itself.
    grid = _extended_grid(tmp_path / "Kurucz_extended")
    pcf.precompute_flux_cube(str(grid), str(grid / "flux_cube.bin"))
    assert (grid / "flux_cube.bin").is_file()
    assert not (grid / "variants_index.csv").exists()
    assert not [p for p in grid.iterdir() if p.is_dir()]
    assert "No extra axes detected" in capsys.readouterr().out


def test_a_combined_grid_is_one_cube(tmp_path):
    grid = tmp_path / "combined_models"
    grid.mkdir()
    rows = []
    for i, (teff, logg) in enumerate([(5000.0, 4.0), (6000.0, 4.0), (5000.0, 4.5), (6000.0, 4.5)]):
        _spectrum(grid / f"s{i}.txt", np.linspace(1000.0, 10000.0, 20))
        rows.append(f"s{i}.txt,{teff},{logg},0.0,{'Kurucz' if i % 2 else 'Phoenix'},orig{i}.txt")
    (grid / "lookup_table.csv").write_text(
        "#file_name,teff,logg,metallicity,source_model,original_file\n" + "\n".join(rows) + "\n")
    pcf.precompute_flux_cube(str(grid), str(grid / "flux_cube.bin"))
    assert (grid / "flux_cube.bin").is_file() and not (grid / "variants_index.csv").exists()


# ----------------------------------------------------------------------
# Item 24: NJM's availability and pause are shared by the whole process
# ----------------------------------------------------------------------
class _CountingSession:
    """A stand-in requests session: HEAD answers ``head`` (a status or an exception); counts requests."""

    def __init__(self, head):
        self.head_answer, self.calls, self.headers, self.verify = head, [], {}, True

    def head(self, url, **kwargs):
        self.calls.append(("HEAD", url, kwargs.get("verify", self.verify)))
        if isinstance(self.head_answer, Exception):
            raise self.head_answer
        return SimpleNamespace(status_code=self.head_answer)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs.get("verify", self.verify)))
        raise requests.ConnectionError("not served in this test")


def _grabbers(monkeypatch, tmp_path, head, n=3):
    """``n`` grabbers made one after another, all on one session; returns (grabbers, session)."""
    session = _CountingSession(head)
    monkeypatch.setattr(njm_filter_grabber.requests, "Session", lambda: session)
    return [NJMFilterGrabber(base_dir=str(tmp_path)) for _ in range(n)], session


def test_a_silent_outage_is_checked_and_reported_once_per_process(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="sed_tools.njm_filter_grabber"):
        grabbers, session = _grabbers(monkeypatch, tmp_path, requests.ConnectTimeout("timed out"))
    assert [g.is_available() for g in grabbers] == [False, False, False]
    assert session.calls == [("HEAD", "https://nillmill.ddns.net/sed_tools", True)]  # one 5 s wait, not three
    assert caplog.text.count("is unavailable (timed out)") == 1
    with pytest.raises(requests.ConnectionError, match="not asked again for 5 minutes"):
        grabbers[2]._get("https://nillmill.ddns.net/sed_tools/index.json")
    assert len(session.calls) == 1


def test_the_outage_is_checked_again_once_the_pause_expires(tmp_path, monkeypatch, caplog):
    clock = [1000.0]
    monkeypatch.setattr(njm_filter_grabber, "monotonic", lambda: clock[0])
    with caplog.at_level(logging.WARNING, logger="sed_tools.njm_filter_grabber"):
        _, session = _grabbers(monkeypatch, tmp_path, requests.ConnectionError("refused"), n=1)
        clock[0] += njm_filter_grabber.NJM_PAUSE + 1
        again = NJMFilterGrabber(base_dir=str(tmp_path))
    assert not again.is_available() and len(session.calls) == 2
    assert caplog.text.count("is unavailable (refused)") == 1  # still the same outage


def test_a_reachable_mirror_is_checked_once_and_its_ssl_fallback_is_kept(tmp_path, monkeypatch, caplog):
    calls, sessions = [], []

    def head(url, verify=True, **kwargs):
        calls.append(("HEAD", verify))
        if verify:
            raise requests.exceptions.SSLError("certificate expired")
        return SimpleNamespace(status_code=200)

    def new_session():
        sessions.append(SimpleNamespace(headers={}, verify=True, head=head))
        return sessions[-1]

    monkeypatch.setattr(njm_filter_grabber.requests, "Session", new_session)
    with caplog.at_level(logging.WARNING, logger="sed_tools.njm_filter_grabber"):
        first = NJMFilterGrabber(base_dir=str(tmp_path))
        second = NJMFilterGrabber(base_dir=str(tmp_path))
    assert first.is_available() and second.is_available()
    assert calls == [("HEAD", True), ("HEAD", False)]  # the second grabber asked nothing
    assert second.session.verify is False  # and skips the certificate check as the first does
    assert caplog.text.count("could not be verified") == 1


def test_a_catalogue_failure_pauses_njm_for_later_catalogues(tmp_path):
    from sed_tools.filter_catalogue import CatalogueError, FilterCatalogue

    asked = []

    def get(url, **kwargs):
        asked.append(url)
        raise requests.HTTPError("503", response=SimpleNamespace(status_code=503))

    mirror = SimpleNamespace(base_url="https://njm.invalid/sed_tools", is_available=lambda: True, _get=get)
    with pytest.raises(CatalogueError, match="Cannot fetch index.json"):
        FilterCatalogue(tmp_path / "filters", mirror=mirror).index()
    with pytest.raises(CatalogueError, match="index.json is not cached and NJM is unreachable"):
        FilterCatalogue(tmp_path / "filters", mirror=mirror).index()
    assert len(asked) == 1


# ----------------------------------------------------------------------
# Item 19: installed curves are checked against NJM only when there is a reason
# ----------------------------------------------------------------------
NJM = "https://nillmill.ddns.net/sed_tools"
CURVE = "Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=original\n4000,0\n5000,1\n6000,0\n"


class Served:
    """A stand-in NJM session: ``files`` maps paths under the mirror to text or an HTTP status."""

    def __init__(self, files):
        self.files, self.requested, self.headers, self.verify = dict(files), [], {}, True

    def get(self, url, timeout=None, headers=None, **kwargs):
        rel = url[len(NJM) + 1:]
        self.requested.append(rel)
        body = self.files.get(rel, 404)
        status, text = (body, "") if isinstance(body, int) else (200, body)

        def raise_for_status():
            if status >= 400:
                raise requests.HTTPError(f"{status} for {rel}", response=SimpleNamespace(status_code=status, headers={}))

        return SimpleNamespace(status_code=status, text=text, content=text.encode(), headers={},
                               raise_for_status=raise_for_status, json=lambda: json.loads(text))

    def curves_requested(self):
        return [rel.rsplit("/", 1)[1] for rel in self.requested if rel.endswith(".dat")]


def _njm_set(size_bytes=150):
    index = {"generated": "2026-10-28", "filters": {"facilities": ["Fac"], "facilities_detailed": {
        "Fac": {"instruments": {"Cam": {"filters": ["g", "r"], "size_bytes": size_bytes}}}}}}
    return {"index.json": json.dumps(index), "filters/Fac/Cam/Cam": "g.dat\nr.dat\n",
            "filters/Fac/Cam/g.dat": CURVE, "filters/Fac/Cam/r.dat": CURVE}


@pytest.fixture
def grabber(tmp_path, monkeypatch):
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.time.sleep", lambda seconds: None)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.tag_filter_file", lambda *a, **k: None)

    def make(files):
        made = NJMFilterGrabber(base_dir=str(tmp_path / "filters"))
        made.session = Served(files)
        return made

    return make


def test_a_repeated_download_of_an_unchanged_set_fetches_no_curve(tmp_path, grabber):
    first = grabber(_njm_set())
    assert first._download_set("Fac", "Cam").downloaded == 2
    again = grabber(_njm_set())  # a later run: a new grabber
    result = again._download_set("Fac", "Cam")
    assert (result.downloaded, result.verified, result.curve_failures) == (0, 2, [])
    assert again.session.curves_requested() == []
    assert (tmp_path / "cache" / "njm" / "filters" / "Fac" / "Cam" / "verified_curves.json").is_file()


def test_a_changed_curve_or_a_changed_njm_record_is_checked_again(tmp_path, grabber, caplog):
    grabber(_njm_set())._download_set("Fac", "Cam")
    r = tmp_path / "filters" / "Fac" / "Cam" / "r.dat"
    r.write_text(CURVE + "7000,0\n")  # changed since NJM's copy was checked
    with caplog.at_level(logging.WARNING):
        local_change = grabber(_njm_set())
        local_change._download_set("Fac", "Cam")
    assert local_change.session.curves_requested() == ["r.dat"] and r.read_text() == CURVE
    assert f"{r} differs from NJM's copy" in caplog.text

    njm_change = grabber(_njm_set(size_bytes=151))  # NJM's set changed since
    njm_change._download_set("Fac", "Cam")
    assert njm_change.session.curves_requested() == ["g.dat", "r.dat"]


def test_a_retry_after_a_partial_download_fetches_only_what_is_missing(grabber):
    partial = _njm_set()
    partial["filters/Fac/Cam/r.dat"] = 404
    assert grabber(partial)._download_set("Fac", "Cam").curve_failures == ["r.dat: 404 for filters/Fac/Cam/r.dat"]
    retry = grabber(_njm_set())
    assert retry._download_set("Fac", "Cam").downloaded == 1
    assert retry.session.curves_requested() == ["r.dat"]
