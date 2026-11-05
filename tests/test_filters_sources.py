"""Which source supplies a filter set: picker entries, spelling, completeness, no SVO
overwrite, NJM optional.

Every source is mocked: NJM through a fake session, SVO through stand-ins or
saved-page fixtures.  Nothing here touches the network.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from bs4 import BeautifulSoup

from sed_tools import filter_library
from sed_tools.component_sources import catalog
from sed_tools.filter_components import ComponentError, InstrumentManifest
from sed_tools.filter_library import FilterLibrary, instrument_catalogue, load_instrument_catalogue
from sed_tools.filter_metadata import read_filter_header
from sed_tools.njm_filter_grabber import NJMFilterGrabber
from sed_tools.svo_filter_grabber import Facility, FilterRow, Instrument, SVOFilterBrowser

from helpers_filters import CURVE, NJM, Served, njm_index, svo_stub, write_set


SVO_CURVE = "Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=svo\n4100,0\n5000,1\n5900,0\n"

GAIA_SVO = {"GAIA0": "Gaia pre-launch filters",
            "GAIA2": "Original passbands used for the magnitudes published in Gaia DR2.",
            "Gaia2m": "Revised passbands for Gaia DR2.", "GAIA2r": "Revised passbands for Gaia DR2.",
            "GAIA3": "GAIA eDR3 release."}
GAIA_NJM = {"DR2": {"description": "Gaia DR2 passbands (Evans et al. 2018)", "svo_instruments": ["GAIA/GAIA2"]},
            "DR2r": {"description": "Gaia DR2 revised passbands", "svo_instruments": ["GAIA/GAIA2r"]},
            "GAIA": {"description": "Gaia EDR3 passbands (Riello et al. 2021)", "svo_instruments": ["GAIA/GAIA3"]},
            "GAIA0": {"svo_instruments": ["GAIA/GAIA0"]}}


@pytest.fixture
def njm(monkeypatch):
    """Build an available NJMFilterGrabber that talks to a :class:`Served` session."""
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.time.sleep", lambda seconds: None)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.tag_filter_file", lambda *a, **k: None)

    def make(root, files):
        grabber = NJMFilterGrabber(base_dir=str(root))
        grabber.session = Served(files)
        return grabber

    return make


@pytest.fixture
def bundled(monkeypatch):
    """Replace the shipped instrument catalogue with ``catalogue`` (facility -> instrument -> metadata)."""
    def use(catalogue):
        facilities = {fac: {"instruments": insts} for fac, insts in catalogue.items()}
        monkeypatch.setattr(filter_library, "load_instrument_catalogue",
                            lambda: instrument_catalogue(facilities))
    use({})
    return use


def labels(result):
    return [(e["label"], e["sources"], e["description"]) for e in result["entries"]]


# ----------------------------------------------------------------------
# Picker entries: GAIA, merging and duplicates
# ----------------------------------------------------------------------
def test_gaia_picker_shows_each_product_once_with_its_source_and_description(tmp_path, njm, bundled):
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({"GAIA": "Gaia mission."}, {"GAIA": GAIA_SVO})
    library.mirror = njm(tmp_path / "filters", {"index.json": njm_index({"GAIA": GAIA_NJM})})
    result = library.instruments("GAIA")
    assert labels(result) == [
        ("DR2", ["original", "njm"], "Gaia DR2 passbands (Evans et al. 2018)"),
        ("DR2r", ["original", "njm"], "Gaia DR2 revised passbands"),
        ("GAIA", ["njm"], "Gaia EDR3 passbands (Riello et al. 2021)"),
        ("GAIA0", ["njm"], "Gaia pre-launch filters"),  # NJM gives none: SVO's text for the same product
        ("Gaia2m", ["svo"], "Revised passbands for Gaia DR2."),
    ]
    assert all(e["fetch_key"] == e["key"] == e["label"] for e in result["entries"])
    assert result["errors"] == []
    assert library.mirror.session.requested == ["index.json"]  # instruments come from index.json, not listings


def test_svo_and_njm_down_leave_installed_and_original_sources(tmp_path, bundled):
    bundled({"GAIA": GAIA_NJM, "CTIO": {"S_PLUS": {}}})
    root = tmp_path / "filters"
    (root / "CTIO" / "S_PLUS").mkdir(parents=True)
    (root / "CTIO" / "S_PLUS" / "g.dat").write_text(CURVE)
    library = FilterLibrary(root)
    library.svo = svo_stub({}, {}, down=True)
    library.mirror = SimpleNamespace(is_available=lambda: False)
    result = library.facilities()
    entries = {e["label"]: e for e in result["entries"]}
    # CTIO's only builder (DECam) makes components, not curves, so it is no source of CTIO's sets.
    assert entries["CTIO"]["sources"] == ["local"] and entries["GAIA"]["sources"] == ["original"]
    assert result["errors"] == ["SVO: Failed to fetch SVO index: connection refused", "NJM unavailable"]
    assert labels(library.instruments("CTIO")) == [("S_PLUS", ["local"], "")]
    assert labels(library.instruments("GAIA"))[0] == ("DR2", ["original"], "Gaia DR2 passbands (Evans et al. 2018)")


# ----------------------------------------------------------------------
# Spelling
# ----------------------------------------------------------------------
def test_a_same_named_svo_product_njm_does_not_claim_keeps_its_own_name_and_text(tmp_path, bundled):
    bundled({"Gemini": {"NIRI": {"description": "NJM's NIRI filter curves", "svo_instruments": []},
                        "GMOS_N": {"description": "NJM's GMOS-N", "svo_instruments": ["Gemini/GMOS-N"]}}})
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({"Gemini": ""}, {"Gemini": {"Niri": "SVO's NIRI", "GMOS-N": "SVO's GMOS-N"}})
    library.mirror = SimpleNamespace(is_available=lambda: False)
    shown = {label: rest for label, *rest in labels(library.instruments("Gemini"))}
    assert shown["GMOS_N"] == [["svo"], "NJM's GMOS-N"]  # NJM's product: NJM's name and text
    assert shown["Niri"] == [["svo"], "SVO's NIRI"] and "NIRI" not in shown  # another product: SVO's own


# ----------------------------------------------------------------------
# Completeness
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# NJM fails fast
# ----------------------------------------------------------------------
def test_svo_may_complete_a_set_njm_wrote_nothing_into(tmp_path, njm, bundled):
    root = tmp_path / "filters"
    directory = write_set(root / "Fac" / "Cam", ["g"], manifest_bands=["g", "r"], text=SVO_CURVE)  # SVO, partial
    library = FilterLibrary(root)
    down = requests.ConnectTimeout("connect timed out")
    library.mirror = njm(root, {"filters/Fac/Cam/Cam": "g.dat\nr.dat\n", "filters/Fac/Cam/g.dat": down,
                                "filters/Fac/Cam/r.dat": down})

    def download(rows):
        (directory / "r.dat").write_text(SVO_CURVE)

    library.svo = SimpleNamespace(list_facilities=lambda: [Facility("Fac", "Fac")],
                                  list_instruments=lambda key: [Instrument("Fac", "Cam", "Cam")],
                                  list_filters=lambda f, i: [FilterRow(f"Fac/Cam.{b}", b, "Fac", "Cam", "")
                                                             for b in "gr"],
                                  download_filters=download)
    assert library.published("Fac", "Cam") == directory
    assert (directory / "g.dat").read_text() == (directory / "r.dat").read_text() == SVO_CURVE


# ----------------------------------------------------------------------
# Compose and availability
# ----------------------------------------------------------------------
def test_compose_refuses_an_unknown_band_before_asking_any_source(tmp_path, bundled, monkeypatch):
    root = tmp_path / "filters"
    write_set(root / "Argus" / "DR280", ["g"])
    library = FilterLibrary(root)
    library.mirror = SimpleNamespace(is_available=lambda: pytest.fail("NJM was asked"))
    monkeypatch.setattr(filter_library, "get_builder", lambda name: pytest.fail("a builder was run"))
    with pytest.raises(ComponentError, match="^Argus/DR280 has no band 'nosuch'$"):
        library.compose(root / "Argus" / "DR280", "filter", bands=["nosuch"])


def test_njm_instrument_without_manifest_is_described_not_an_error(tmp_path, njm, bundled):
    info = {"bands": {"G": {"curve": {"Components": "filter+qe+instrument", "ComponentsSource": "svo",
                                      "DetectorType": "photon", "DetectorTypeSource": "svo-curve-match"}},
                      "Grvs": {"curve": {"Components": "filter+qe+instrument", "DetectorType": "photon"}}}}
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = njm(root, {"filters/GAIA/GAIA/GAIA": "G.dat\nGrvs.dat\n",
                                "filters/GAIA/GAIA/info.json": json.dumps(info)})
    library.svo = SimpleNamespace(list_facilities=lambda: pytest.fail("SVO was asked"))
    result = library.available("GAIA", "GAIA")
    assert result["availability_source"] == "njm" and result["errors"] == []
    assert result["bands"] == ["G", "Grvs"] and result["detector_type"] == "photon"
    assert result["notes"][0].startswith("no component manifest")


# ----------------------------------------------------------------------
# The bundled catalogue
# ----------------------------------------------------------------------
