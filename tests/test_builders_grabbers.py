"""SVO's filter browser and VOTables (saved pages, tests/fixtures/builders), and the spectra grabbers' discovery."""

from __future__ import annotations

import numpy as np
import pytest

from builder_fixtures import Session, saved
from sed_tools.filters import svo
from sed_tools.filters.curve_metadata import SVOMetadataClient, _param_text, svo_fps_url
from sed_tools.grids import mast, msg
from sed_tools.grids import svo as svo_grid
from test_release_cli import answer

GAIA3 = svo_fps_url().replace("fps.php", "index.php?mode=browse&gname=GAIA&gname2=GAIA3&asttype=")


# ------------------------------------------------------------------ SVO filter browser
def test_browser_lists_an_instruments_filters_from_svos_page(tmp_path):
    session = Session()
    browser = svo.SVOFilterBrowser(base_dir=str(tmp_path), session=session)
    rows = browser.list_filters("GAIA", "GAIA3")
    # SVO's instrument column says "GAIA" for every Gaia release; the rows are filed under GAIA3.
    assert [(r.filter_id, r.band, r.facility, r.instrument, r.description) for r in rows] == [
        ("GAIA/GAIA3.Gbp", "Gbp", "GAIA", "GAIA3", "GAIA Gbp filter, DR3"),
        ("GAIA/GAIA3.G", "G", "GAIA", "GAIA3", "GAIA G filter, DR3"),
        ("GAIA/GAIA3.Grp", "Grp", "GAIA", "GAIA3", "GAIA Grp filter, DR3"),
        ("GAIA/GAIA3.Grvs", "Grvs", "GAIA", "GAIA3", "GAIA Grvs filter, DR3"),
    ]
    assert [i.key for i in browser.list_instruments("GAIA")] == ["GAIA0", "GAIA2", "Gaia2m", "GAIA2r", "GAIA3"]
    assert browser.list_filters("GAIA", "GAIA3") is rows and len(session.calls) == 2  # both pages cached


def test_browser_finds_no_filters_on_a_page_without_the_table(tmp_path):
    files = saved()
    files[GAIA3] = b"<html><body><table><tr><td>No filters</td></tr></table></body></html>"
    assert svo.SVOFilterBrowser(base_dir=str(tmp_path), session=Session(files)).list_filters("GAIA", "GAIA3") == []


@pytest.mark.parametrize("filter_id, band", [("GAIA/GAIA3.Gbp", "Gbp"), ("Generic/Bessell", "Bessell"), ("J", "J")])
def test_band_from_filter_id(filter_id, band):
    assert svo._band_from_filter_id(filter_id) == band


def test_metadata_client_reads_svos_votable():
    record = SVOMetadataClient(session=Session()).get_filter("GAIA/GAIA3.Grvs")
    assert (record.filter_id, record.detector_type, record.wavelength_unit, record.components) == (
        "GAIA/GAIA3.Grvs", "photon", "Angstrom", "Filter + CCD + Instrument")
    assert len(record.wavelength) == 32 and record.wavelength[0] == 8459.75 and np.all(np.diff(record.wavelength) > 0)
    assert 0 < record.transmission.max() <= 1


@pytest.mark.parametrize("value, text", [(b"Angstrom", "Angstrom"), ("1", "1"), (1.5, "1.5"), (None, "")])
def test_param_text(value, text):
    assert _param_text(value) == text  # astropy gives char PARAMs as str, or bytes in older versions


# ---------------------------------------------------------------- SVO spectra: probing
def _probing(tmp_path, monkeypatch, present, window, gap):
    grabber = svo_grid.SVOSpectraGrabber(base_dir=str(tmp_path))
    asked = []
    monkeypatch.setattr(grabber, "_head_exists", lambda model, fid: asked.append(fid) or fid in present)
    grabber.expand_window, grabber.expand_max_gap = window, gap
    return grabber, asked


def test_expansion_walks_out_from_a_seed_until_a_gap(tmp_path, monkeypatch):
    grabber, asked = _probing(tmp_path, monkeypatch, {4, 7, 12, 13, 17}, window=2, gap=3)
    assert grabber._expand_around("Kurucz", 10) == {4, 7, 10, 12, 13}  # 17 lies beyond 3 misses
    assert asked == [8, 7, 6, 5, 4, 3, 2, 1, 12, 13, 14, 15, 16]


def test_expansion_never_asks_for_fids_below_1(tmp_path, monkeypatch):
    grabber, asked = _probing(tmp_path, monkeypatch, {1}, window=1, gap=2)
    assert grabber._expand_around("Kurucz", 2) == {1, 2}
    assert asked == [1, 3, 4]


def test_probe_finds_a_contiguous_grid(tmp_path, monkeypatch):
    grabber, _ = _probing(tmp_path, monkeypatch, set(range(1, 61)), window=12, gap=20)
    assert grabber._bounded_probe("Kurucz") == set(range(1, 61))


# -------------------------------------------------------------------- MAST and MSG
def test_mast_lists_every_bosz_resolution_and_accepts_each(tmp_path, monkeypatch, capsys):
    grabber = mast.MASTSpectraGrabber(base_dir=str(tmp_path))
    names = grabber.discover_models()
    assert names == ["BOSZ-2024-r500", "BOSZ-2024-r1000", "BOSZ-2024-r2000", "BOSZ-2024-r5000",
                     "BOSZ-2024-r10000", "BOSZ-2024-r20000", "BOSZ-2024-r50000", "BOSZ-2024-rorig"]
    assert "Discovering spectral grids from MAST" in capsys.readouterr().out
    answer(monkeypatch, *[""] * len(names))
    assert [grabber.get_model_metadata(n)["reskey"] for n in names] == [n.rsplit("-", 1)[1] for n in names]


@pytest.mark.parametrize("meta, expected", [
    ([-1.0, 0.0], (100.0, 4.0, 0.0)),         # neither order is plausible: the C order wins a tie
    ([-1.0, np.nan], (200.0, 4.0, -1.0)),     # ... and loses to the order with more finite values
])
def test_msg_takes_the_most_complete_order_when_neither_is_plausible(tmp_path, meta, expected):
    grabber = msg.MSGSpectraGrabber(base_dir=str(tmp_path), max_workers=1)
    grabber.model_axes["sg-Cool"] = {
        "axes": [np.array([100.0, 200.0]), np.array([4.0, 4.5]), np.array(meta)], "shape": (2, 2, 2),
        "vlin": np.array([1]), "roles": {"teff": 0, "logg": 1, "meta": 2}}
    teff, logg, metal = grabber._params_from_index("sg-Cool", 0)
    np.testing.assert_array_equal([teff, logg, metal], expected)
