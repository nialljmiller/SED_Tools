"""0.5.0 filter library: picker entries, spelling, completeness, no SVO overwrite, NJM optional.

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

NJM = "https://nillmill.ddns.net/sed_tools"
CURVE = "Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=original\n4000,0\n5000,1\n6000,0\n"
SVO_CURVE = "Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=svo\n4100,0\n5000,1\n5900,0\n"

GAIA_SVO = {"GAIA0": "Gaia pre-launch filters",
            "GAIA2": "Original passbands used for the magnitudes published in Gaia DR2.",
            "Gaia2m": "Revised passbands for Gaia DR2.", "GAIA2r": "Revised passbands for Gaia DR2.",
            "GAIA3": "GAIA eDR3 release."}
GAIA_NJM = {"DR2": {"description": "Gaia DR2 passbands (Evans et al. 2018)", "svo_instruments": ["GAIA/GAIA2"]},
            "DR2r": {"description": "Gaia DR2 revised passbands", "svo_instruments": ["GAIA/GAIA2r"]},
            "GAIA": {"description": "Gaia EDR3 passbands (Riello et al. 2021)", "svo_instruments": ["GAIA/GAIA3"]},
            "GAIA0": {"svo_instruments": ["GAIA/GAIA0"]}}


class Served:
    """A fake NJM session: ``files`` maps paths under the mirror to text, an HTTP status, or an exception."""

    def __init__(self, files):
        self.files = dict(files)
        self.headers, self.verify, self.requested = {}, True, []

    def get(self, url, timeout=None, headers=None, **kwargs):
        rel = url[len(NJM) + 1:]
        self.requested.append(rel)
        body = self.files.get(rel, 404)
        if isinstance(body, Exception):
            raise body
        status, text = (body, "") if isinstance(body, int) else (200, body)

        def raise_for_status():
            if status >= 400:
                raise requests.HTTPError(f"{status} for {rel}", response=SimpleNamespace(status_code=status, headers={}))

        return SimpleNamespace(status_code=status, text=text, content=text.encode(), headers={},
                               raise_for_status=raise_for_status, json=lambda: json.loads(text))


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


def svo_stub(facilities, instruments, *, down=False):
    """An SVO stand-in: ``facilities`` {key: text}; ``instruments`` {facility: {key: text}}."""
    def check():
        if down:
            raise RuntimeError("Failed to fetch SVO index: connection refused")

    def list_facilities():
        check()
        return [Facility(key, key, text) for key, text in facilities.items()]

    def list_instruments(key):
        check()
        return [Instrument(key, name, name, text) for name, text in instruments.get(key, {}).items()]

    return SimpleNamespace(list_facilities=list_facilities, list_instruments=list_instruments)


def njm_index(detailed):
    return json.dumps({"generated": "2026-10-27T00:00:00",
                       "filters": {"facilities": sorted(detailed), "facilities_detailed": {
                           fac: {"instruments": insts} for fac, insts in detailed.items()}}})


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


def test_gaia_picker_without_njm_lists_every_other_source(tmp_path, bundled):
    bundled({"GAIA": GAIA_NJM})
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({"GAIA": "Gaia mission."}, {"GAIA": GAIA_SVO})
    library.mirror = SimpleNamespace(is_available=lambda: False)
    result = library.instruments("GAIA")
    # The same products, names and descriptions as with NJM up: SVO's copies of NJM's products take NJM's
    # names (GAIA2 is DR2, GAIA3 is GAIA), and are what can be fetched now. Gaia2m is SVO's own.
    assert labels(result) == [
        ("DR2", ["original", "svo"], "Gaia DR2 passbands (Evans et al. 2018)"),
        ("DR2r", ["original", "svo"], "Gaia DR2 revised passbands"),
        ("GAIA", ["svo"], "Gaia EDR3 passbands (Riello et al. 2021)"),
        ("GAIA0", ["svo"], "Gaia pre-launch filters"),
        ("Gaia2m", ["svo"], "Revised passbands for Gaia DR2."),
    ]
    assert {e["label"]: e["names"]["svo"] for e in result["entries"]} == {
        "DR2": "GAIA2", "DR2r": "GAIA2r", "GAIA": "GAIA3", "GAIA0": "GAIA0", "Gaia2m": "Gaia2m"}
    assert result["errors"] == ["NJM unavailable"]


def test_svo_down_keeps_njm_and_builders(tmp_path, njm, bundled):
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({}, {}, down=True)
    library.mirror = njm(tmp_path / "filters", {"index.json": njm_index({"GAIA": GAIA_NJM})})
    result = library.instruments("GAIA")
    assert [e["label"] for e in result["entries"]] == ["DR2", "DR2r", "GAIA", "GAIA0"]
    assert result["errors"] == ["SVO GAIA: Failed to fetch SVO index: connection refused"]
    facilities = library.facilities()
    assert "GAIA" in [e["label"] for e in facilities["entries"]]
    assert facilities["errors"] == ["SVO: Failed to fetch SVO index: connection refused"]


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


def test_njm_junk_index_falls_back_to_its_listing_and_other_sources(tmp_path, njm, bundled, caplog):
    bundled({"GAIA": GAIA_NJM})
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({"GAIA": ""}, {"GAIA": GAIA_SVO})
    listing = '<a href="?C=N">Name</a><a href="DR2/">DR2/</a><a href="GAIA/">GAIA/</a><a href="README.txt">x</a>'
    library.mirror = njm(tmp_path / "filters", {"index.json": "<html>not json</html>",
                                                 "filters": '<a href="GAIA/">GAIA/</a><a href="notes.txt">n</a>',
                                                 "filters/GAIA/": listing})
    with caplog.at_level(logging.WARNING):
        result = library.instruments("GAIA")
        facilities = library.facilities()
    assert "Cannot read NJM's index.json" in caplog.text and "reading the directory listing of GAIA" in caplog.text
    by_label = {e["label"]: e for e in result["entries"]}
    assert "README.txt" not in by_label and "README" not in by_label  # files are never instruments
    assert by_label["DR2"]["sources"] == ["original", "njm"]
    assert by_label["DR2"]["description"] == "Gaia DR2 passbands (Evans et al. 2018)"  # from the bundled catalogue
    # NJM still lists GAIA, and the bundled catalogue says it is SVO's GAIA3: one entry.
    assert by_label["GAIA"]["sources"] == ["njm"] and "GAIA3" not in by_label
    assert by_label["Gaia2m"]["sources"] == ["svo"]
    assert [e["label"] for e in facilities["entries"] if "njm" in e["sources"]] == ["GAIA"]


def test_njm_junk_records_are_ignored_not_guessed(tmp_path, njm, bundled):
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({}, {})
    index = njm_index({"GAIA": {"DR2": "junk", "GAIA0": {"description": 7, "svo_instruments": "GAIA/GAIA0"}}})
    library.mirror = njm(tmp_path / "filters", {"index.json": index})
    assert labels(library.instruments("GAIA")) == [("DR2", ["original", "njm"], ""), ("DR2r", ["original"], ""),
                                                  ("GAIA0", ["njm"], "")]


def test_one_sources_names_never_merge(tmp_path, njm, bundled):
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({"Cameras": ""}, {"Cameras": {"Sony_NEX5N": "Sony"}})
    library.mirror = njm(tmp_path / "filters", {"index.json": njm_index(
        {"Cameras": {"Sony_NEX_5N": {}, "Sony_NEX5N": {}}})})
    assert labels(library.instruments("Cameras")) == [("Sony_NEX5N", ["svo", "njm"], "Sony"),
                                                     ("Sony_NEX_5N", ["njm"], "")]


# ----------------------------------------------------------------------
# Spelling
# ----------------------------------------------------------------------
def test_builder_and_status_lookups_ignore_case_and_punctuation():
    assert catalog.builder_for("Gemini", "GMOS-N") is catalog.BUILDERS_FOR[("Gemini", "GMOS_N")]
    assert catalog.registered_name("herschel", "Pacs") == ("Herschel", "PACS")
    assert catalog.registered_name("PLANCK", "hfi") == ("Planck", "HFI")
    assert catalog.builder_for("Gemini", "GMOS-X") is None
    assert catalog.status_for("Pan-STARRS", "ps1") == catalog.STATUS[("PAN-STARRS", "PS1")]
    assert catalog.status_for("gaia", "GAIA3") == catalog.STATUS[("GAIA", "*")]
    assert catalog.component_availability("Gemini", "GMOS-N")["status"] == "supported"
    assert catalog.match_name("Sony_NEX5N", ["Sony_NEX_5N", "Sony_NEX5N"]) == "Sony_NEX5N"
    assert catalog.match_name("sony nex5n", ["Sony_NEX_5N", "Sony_NEX5N"]) is None  # ambiguous: no guess


@pytest.mark.parametrize("njm_up", [True, False])
def test_picker_names_use_njm_or_builder_spelling_whether_or_not_njm_answers(tmp_path, njm, bundled, njm_up):
    bundled({"CTIO": {"S_PLUS": {"svo_instruments": ["CTIO/S-PLUS"]}},
             "Herschel": {"PACS": {"svo_instruments": ["Herschel/Pacs"]}},
             "Planck": {"HFI": {"svo_instruments": ["PLANCK/HFI"]}}, "Bepi_Colombo": {"HRIC": {}},
             "JWST": {"NIRSPEC": {"svo_instruments": ["JWST/NIRSpec"]}}})
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({"CTIO": "", "Herschel": "", "PLANCK": "Planck mission.", "Bepi-Colombo": "", "JWST": ""},
                           {"CTIO": {"S-PLUS": "Southern Photometric Local Universe Survey"},
                            "Herschel": {"Pacs": ""}, "PLANCK": {"HFI": ""}, "JWST": {"NIRSpec": ""}})
    if njm_up:
        library.mirror = njm(tmp_path / "filters", {"index.json": njm_index(
            {"CTIO": {"S_PLUS": {}}, "Herschel": {"PACS": {}}, "Planck": {"HFI": {}}, "Bepi_Colombo": {"HRIC": {}},
             "JWST": {"NIRSPEC": {}}})})
    else:
        library.mirror = SimpleNamespace(is_available=lambda: False)
    facilities = {e["label"]: e for e in library.facilities()["entries"]}
    assert {"Bepi_Colombo", "Planck"} <= set(facilities) and not {"Bepi-Colombo", "PLANCK"} & set(facilities)
    assert facilities["Planck"]["description"] == "Planck mission."
    fetch = lambda fac, inst: next(e["fetch_key"] for e in library.instruments(fac)["entries"]
                                   if catalog.name_key(e["label"]) == catalog.name_key(inst))
    assert fetch("CTIO", "S-PLUS") == "S_PLUS" and fetch("Herschel", "Pacs") == "PACS"
    assert fetch("PLANCK", "HFI") == "HFI" and fetch("JWST", "NIRSpec") == "NIRSPEC"


def test_a_same_named_svo_product_njm_does_not_claim_keeps_its_own_name_and_text(tmp_path, bundled):
    bundled({"Gemini": {"NIRI": {"description": "NJM's NIRI filter curves", "svo_instruments": []},
                        "GMOS_N": {"description": "NJM's GMOS-N", "svo_instruments": ["Gemini/GMOS-N"]}}})
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({"Gemini": ""}, {"Gemini": {"Niri": "SVO's NIRI", "GMOS-N": "SVO's GMOS-N"}})
    library.mirror = SimpleNamespace(is_available=lambda: False)
    shown = {label: rest for label, *rest in labels(library.instruments("Gemini"))}
    assert shown["GMOS_N"] == [["svo"], "NJM's GMOS-N"]  # NJM's product: NJM's name and text
    assert shown["Niri"] == [["svo"], "SVO's NIRI"] and "NIRI" not in shown  # another product: SVO's own


def test_svo_labels_are_instrument_names_with_svo_descriptions(tmp_path):
    two = ('<a href="index.php?mode=browse&gname=GAIA&gname2=GAIA2&asttype=" onMouseOver="Tip(\'&lt;b&gt;GAIA/GAIA2'
           '&lt;/b&gt;&lt;br&gt;&lt;table&gt;&lt;td class=helptd&gt;Original &amp;#956; passbands &lt;sup&gt;'
           'DR2&lt;/sup&gt; \\\'quoted\\\'&lt;/td&gt;&lt;/table&gt;\')">GAIA2</a>'
           '<a href="index.php?id=GAIA/GAIA2.G&&mode=browse&gname=GAIA&gname2=GAIA2#filter">GAIA/GAIA2.G</a>')
    one = '<a href="index.php?id=2MASS/2MASS.J&&mode=browse&gname=2MASS&gname2=2MASS#filter">2MASS/2MASS.J</a>'
    browser = SVOFilterBrowser(base_dir=str(tmp_path))
    pages = {"GAIA": two, "2MASS": one}
    browser._fetch_index = lambda params: BeautifulSoup(pages[params["gname"]], "html.parser")
    assert browser.list_instruments("GAIA") == [
        Instrument("GAIA", "GAIA2", "GAIA2", "Original μ passbands DR2 'quoted'")]
    assert browser.list_instruments("2MASS") == [Instrument("2MASS", "2MASS", "2MASS", "")]  # not 2MASS/2MASS.J


# ----------------------------------------------------------------------
# Completeness
# ----------------------------------------------------------------------
def write_set(directory, bands, *, index=None, manifest_bands=None, archived=None, text=CURVE):
    directory.mkdir(parents=True, exist_ok=True)
    for band in bands:
        (directory / f"{band}.dat").write_text(text)
    (directory / directory.name).write_text("".join(f"{b}.dat\n" for b in (index if index is not None else bands)))
    data = asdict(InstrumentManifest(directory.parent.name, directory.name, list(manifest_bands or bands)))
    if archived is not None:
        data["archived_bands"] = archived
    (directory / "manifest.json").write_text(json.dumps(data))
    return directory


@pytest.mark.parametrize("archived", [{"H_svo": {"detector_type": "photon"}}, ["H_svo"]])
def test_archived_bands_are_not_expected(tmp_path, archived):
    directory = write_set(tmp_path / "Euclid" / "NISP", ["H", "J"], manifest_bands=["H", "H_svo", "J"],
                          archived=archived)
    assert FilterLibrary.published_complete(directory)


def test_completeness_needs_every_index_curve_and_every_unarchived_manifest_band(tmp_path):
    listed = write_set(tmp_path / "A" / "Listed", ["g"], index=["g", "r"])
    assert not FilterLibrary.published_complete(listed)
    # A download that failed part-way: its index names only what it saved, its manifest what it expected.
    partial = write_set(tmp_path / "A" / "Partial", ["g"], manifest_bands=["g", "r"])
    assert not FilterLibrary.published_complete(partial)
    assert FilterLibrary.published_complete(write_set(tmp_path / "A" / "Whole", ["g", "r"]))


def test_njm_set_complete_by_its_index_stops_there(tmp_path, njm, bundled, caplog):
    manifest = asdict(InstrumentManifest("Euclid", "NISP", ["H", "H_svo"]))  # archive band left in "bands"
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = njm(root, {"filters/Euclid/NISP/NISP": "H.dat\n", "filters/Euclid/NISP/H.dat": CURVE,
                                "filters/Euclid/NISP/manifest.json": json.dumps(manifest)})
    library.svo = SimpleNamespace(list_facilities=lambda: pytest.fail("SVO was asked"))
    with caplog.at_level(logging.WARNING):
        assert library.published("Euclid", "NISP") == root / "Euclid" / "NISP"
    assert "declared bands that NJM's index does not list and that are not installed (H_svo)" in caplog.text
    assert (root / "Euclid" / "NISP" / "H.dat").read_text() == CURVE
    assert FilterLibrary.published_complete(root / "Euclid" / "NISP")  # the next fetch asks no source


def test_svo_never_writes_into_a_set_njm_partly_filled(tmp_path, njm, bundled):
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = njm(root, {"filters/Fac/Cam/Cam": "g.dat\nr.dat\n", "filters/Fac/Cam/g.dat": CURVE,
                                "filters/Fac/Cam/r.dat": 404})
    library.svo = SimpleNamespace(list_facilities=lambda: pytest.fail("SVO was asked"))
    with pytest.raises(ComponentError) as info:
        library.published("Fac", "Cam")
    message = str(info.value)
    assert message.startswith("NJM provided only part of Fac/Cam; run again to fetch the rest. SVO is not used")
    assert "1 of 2 curves could not be downloaded or checked: r.dat: 404 for filters/Fac/Cam/r.dat" in message
    assert (root / "Fac" / "Cam" / "g.dat").read_text() == CURVE


def test_svo_is_still_the_last_resort_when_njm_wrote_nothing(tmp_path, bundled):
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = SimpleNamespace(is_available=lambda: True, _download_set=lambda f, i: SimpleNamespace(listed=[]))

    def download(rows):
        directory = root / "Fac" / "Cam"
        directory.mkdir(parents=True)
        (directory / "g.dat").write_text(SVO_CURVE)
        (directory / "Cam").write_text("g.dat\n")
        InstrumentManifest("Fac", "Cam", ["g"]).save(directory)

    library.svo = SimpleNamespace(list_facilities=lambda: [Facility("Fac", "Fac")],
                                  list_instruments=lambda key: [Instrument("Fac", "Cam", "Cam")],
                                  list_filters=lambda f, i: [FilterRow("Fac/Cam.g", "g", "Fac", "Cam", "")],
                                  download_filters=download)
    assert library.published("Fac", "Cam") == root / "Fac" / "Cam"


def test_retry_replaces_a_curve_that_is_not_njms(tmp_path, njm, bundled, caplog):
    root = tmp_path / "filters"
    directory = write_set(root / "Euclid" / "NISP", ["H", "Y"], manifest_bands=["H", "Y", "J"], text=SVO_CURVE)
    (directory / "H.dat").write_text(CURVE.replace("original", "original;FilterID=Euclid/NISP.H"))  # NJM's, tagged
    library = FilterLibrary(root)
    library.mirror = njm(root, {"filters/Euclid/NISP/NISP": "H.dat\nJ.dat\nY.dat\n",
                                **{f"filters/Euclid/NISP/{b}.dat": CURVE for b in "HJY"}})
    with caplog.at_level(logging.WARNING):
        assert library.published("Euclid", "NISP") == directory
    assert (directory / "Y.dat").read_text() == CURVE  # SVO's copy under NJM's name, replaced
    assert read_filter_header(directory / "H.dat")["FilterID"] == "Euclid/NISP.H"  # NJM's samples: kept as is
    assert f"{directory / 'Y.dat'} differs from NJM's copy" in caplog.text
    assert f"{directory / 'H.dat'} differs" not in caplog.text


def test_published_downloads_njms_spelling_of_the_name(tmp_path, njm, bundled):
    bundled({"CTIO": {"S_PLUS": {}}})
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = njm(root, {"filters/CTIO/S_PLUS/S_PLUS": "g.dat\n", "filters/CTIO/S_PLUS/g.dat": CURVE})
    assert library.published("CTIO", "S-PLUS") == root / "CTIO" / "S_PLUS"
    assert library.mirror.session.requested[0] == "filters/CTIO/S_PLUS/S_PLUS"  # no request under SVO's spelling


def test_published_builds_under_the_builders_spelling(tmp_path, bundled, monkeypatch):
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = SimpleNamespace(is_available=lambda: False)
    built = []

    def gmos(target, **kwargs):
        built.append(kwargs)
        write_set(target / "Gemini" / "GMOS_N", ["g"])

    monkeypatch.setitem(catalog.BUILDERS_FOR, ("Gemini", "GMOS_N"), catalog.BuilderRef(
        "gmos", (("instrument", "GMOS_N"),), publishes=True))
    monkeypatch.setattr(filter_library, "get_builder", lambda name: gmos)
    assert library.published("Gemini", "GMOS-N") == root / "Gemini" / "GMOS_N"
    assert built == [{"instrument": "GMOS_N"}]


# ----------------------------------------------------------------------
# NJM fails fast
# ----------------------------------------------------------------------
def test_njm_that_stops_answering_is_not_asked_again(tmp_path, njm, caplog):
    root = tmp_path / "filters"
    down = requests.ConnectTimeout("connect timed out")
    grabber = njm(root, {"filters/Fac/Cam/Cam": "a.dat\nb.dat\nc.dat\n", "filters/Fac/Cam/a.dat": down,
                         "filters/Fac/Cam/b.dat": down, "filters/Fac/Cam/c.dat": down})
    with caplog.at_level(logging.WARNING):
        result = grabber._download_set("Fac", "Cam")
    assert grabber.session.requested.count("filters/Fac/Cam/a.dat") == 3  # retried, then given up
    assert "filters/Fac/Cam/b.dat" not in grabber.session.requested
    assert result.curve_failures == ["3 curves from a.dat on: connect timed out"]
    assert caplog.text.count("stopped answering") == 1 and "original-source builders and SVO" in caplog.text
    assert not grabber.is_available()
    with pytest.raises(requests.ConnectionError, match="not asked again"):
        grabber._get(f"{NJM}/index.json")


def test_njm_is_asked_again_once_its_pause_is_over(tmp_path, njm, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("sed_tools.njm_filter_grabber.monotonic", lambda: clock[0])
    grabber = njm(tmp_path, {"index.json": requests.ConnectionError("refused")})
    with pytest.raises(requests.ConnectionError, match="refused"):
        grabber._get(f"{NJM}/index.json")
    assert not grabber.is_available()
    clock[0] += 299
    assert not grabber.is_available()
    clock[0] += 2
    grabber.session.files["index.json"] = "{}"
    assert grabber.is_available() and grabber._get(f"{NJM}/index.json").text == "{}"


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


def test_unavailable_njm_says_which_sources_are_used(tmp_path, monkeypatch, caplog):
    def head(url, **kwargs):
        raise requests.ConnectionError("no route to host")

    grabber = NJMFilterGrabber.__new__(NJMFilterGrabber)
    grabber.base_url, grabber.session = NJM, SimpleNamespace(head=head)
    with caplog.at_level(logging.WARNING):
        assert grabber._check_availability() is False
    assert ("it will not be used for 5 minutes, and filters come from installed copies, the original-source "
            "builders and SVO") in caplog.text


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
def test_shipped_catalogue_loads_with_njm_spellings():
    catalogue = load_instrument_catalogue()
    assert "S_PLUS" in catalogue["CTIO"]["instruments"] and "PACS" in catalogue["Herschel"]["instruments"]
    assert set(catalogue["GAIA"]["instruments"]) >= {"DR2", "DR2r", "GAIA", "GAIA0"}
    for record in catalogue.values():
        for meta in record["instruments"].values():
            assert set(meta) == {"description", "svo_instruments", "bands", "svo_bands"} and meta["bands"]
            assert set(meta["svo_bands"].values()) <= set(meta["bands"])
    gaia = catalogue["GAIA"]["instruments"]
    # NJM may add bands to a served set; the shipped snapshot holds at least the DR3 set
    assert {"G", "Gbp", "Gbp_bright", "Gbp_faint", "Grp", "Grvs"} <= set(gaia["GAIA"]["bands"])
    assert gaia["GAIA"]["svo_bands"] == {"GAIA/GAIA3.G": "G", "GAIA/GAIA3.Gbp": "Gbp", "GAIA/GAIA3.Grp": "Grp",
                                         "GAIA/GAIA3.Grvs": "Grvs"}
    assert gaia["DR2"]["svo_bands"] == {"GAIA/GAIA2.G": "G", "GAIA/GAIA2.Gbp": "BP", "GAIA/GAIA2.Grp": "RP"}
    assert "BP_svo" not in gaia["DR2"]["bands"]  # archived copies are not part of the set


def test_catalogue_generator_writes_what_the_loader_reads(tmp_path, monkeypatch):
    import runpy
    import sys

    index = tmp_path / "index.json"
    index.write_text(njm_index({"GAIA": GAIA_NJM, "Junk": {"X": None}}))
    evidence = tmp_path / "svo_instruments.json"
    same = lambda *bands: {"njm": {f"GAIA/DR2/{b}": {"same": True} for b in bands}}
    evidence.write_text(json.dumps({"svo_groups": {"GAIA/GAIA2": {"bands": {
        "GAIA/GAIA2.G": same("G", "G_svo"), "GAIA/GAIA2.Gbp": same("BP", "BP_svo"),
        "GAIA/GAIA2.Grp": {"njm": {"GAIA/DR2/RP": {"same": False}}}}}}}))
    served = {"GAIA/DR2/DR2": "BP.dat\nG.dat\nRP.dat\n", "GAIA/DR2r/DR2r": "G.dat\n", "GAIA/GAIA/GAIA": "G.dat\n",
              "GAIA/GAIA0/GAIA0": "G.dat\n", "Junk/X/X": "a.dat\n"}
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        return SimpleNamespace(text=served[url.split("/filters/", 1)[1]], raise_for_status=lambda: None)

    monkeypatch.setattr("requests.get", get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    out = tmp_path / "catalogue.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_instrument_catalogue.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--index", str(index), "--svo-evidence", str(evidence),
                                      "--output", str(out)])
    runpy.run_path(str(script), run_name="__main__")
    catalogue = load_instrument_catalogue(out)
    assert catalogue["GAIA"]["instruments"]["DR2"] == dict(
        GAIA_NJM["DR2"], bands=["BP", "G", "RP"], svo_bands={"GAIA/GAIA2.G": "G", "GAIA/GAIA2.Gbp": "BP"})
    assert catalogue["GAIA"]["instruments"]["GAIA0"] == {"description": "", "svo_instruments": ["GAIA/GAIA0"],
                                                         "bands": ["G"], "svo_bands": {}}
    assert catalogue["Junk"]["instruments"]["X"] == {"description": "", "svo_instruments": [], "bands": ["a"],
                                                     "svo_bands": {}}
    stored = json.loads(out.read_text())["facilities"]["GAIA"]["instruments"]
    assert stored["GAIA0"] == {"svo_instruments": ["GAIA/GAIA0"], "bands": ["G"]}  # empty fields are not stored
    assert stored["DR2"]["svo_bands"] == {"GAIA/GAIA2": ["G", "Gbp=BP"]}  # Grp is not the same curve as RP
    assert sum('"bands": [' in line for line in out.read_text().splitlines()) == 5  # one line per instrument
    assert all(url.startswith("https://nillmill.ddns.net/sed_tools/filters/") for url in requested)


def test_catalogue_search_reads_an_index_with_the_new_instrument_fields(tmp_path):
    from sed_tools.filter_catalogue import FilterCatalogue

    detailed = {"GAIA": {"DR2": dict(GAIA_NJM["DR2"], filters=["BP", "G"], detector_types={"photon": 2},
                                     properties=None, info=None)}}
    server = Served({"index.json": njm_index(detailed)})
    mirror = SimpleNamespace(base_url=NJM, is_available=lambda: True,
                             _get=lambda url, **kwargs: server.get(url, **kwargs))
    catalogue = FilterCatalogue(tmp_path / "filters", mirror=mirror)
    found = catalogue.search("DR2")
    assert list(found["band"]) == ["BP", "G"] and found.attrs["errors"] == []
    assert found.attrs["gaps"] == ["GAIA/DR2: NJM has no properties file"]
    assert catalogue.resolve("gaia", "dr2")["instrument"] == "DR2"
