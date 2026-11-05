"""Filters with NJM or SVO down: NJM's names, SVO under NJM's names, one folder per product,
a partial set installed with one warning, each outage announced once, and failing fast.

Every source is mocked; nothing here touches the network.
"""

import json
import logging
from types import SimpleNamespace

import pytest
import requests

from sed_tools import cli, filter_library, svo_filter_grabber
from sed_tools.api import Filters
from sed_tools.filter_catalogue import CatalogueError, FilterCatalogue, format_citations
from sed_tools.filter_components import ComponentError
from sed_tools.filter_library import FilterLibrary
from sed_tools.filter_metadata import SVOUnreachableError, read_filter_header
from sed_tools.njm_filter_grabber import NJMFilterGrabber
from sed_tools.svo_filter_grabber import SVOFilterBrowser

from helpers_filters import (CURVE, GAIA, NJM_DOWN, STALE, Served, atmosphere_site, njm_index, svo_browser, svo_stub,
                             write_set)
from helpers_filters import bundled, njm, njm as njm_filters  # noqa: F401 (fixtures)
from helpers_grids import run_cli


EUCLID = {"NISP": {"description": "Euclid NISP", "svo_instruments": ["Euclid/NISP", "Euclid/NISP0"],
                   "bands": ["Blue", "H", "J", "Red", "Y"],
                   "svo_bands": {"Euclid/NISP": ["H", "J", "Y"], "Euclid/NISP0": ["Blue", "Red"]}}}


def labels(result):
    return [(e["label"], e["sources"]) for e in result["entries"]]


# ----------------------------------------------------------------------
# The pickers name SVO's copies after NJM while NJM is down
# ----------------------------------------------------------------------
def test_an_installed_copy_under_svos_name_stays_its_own_entry(tmp_path, bundled):
    bundled({"GAIA": GAIA})
    root = tmp_path / "filters"
    write_set(root / "GAIA" / "GAIA3", ["G"])  # downloaded from SVO by 0.4.1
    library = FilterLibrary(root)
    library.svo = svo_stub({"GAIA": ""}, {"GAIA": {"GAIA3": "eDR3"}})
    library.mirror = NJM_DOWN
    assert labels(library.instruments("GAIA")) == [("DR2", ["original"]), ("DR2r", ["original"]),
                                                  ("GAIA", ["svo"]), ("GAIA3", ["local"])]


# ----------------------------------------------------------------------
# NJM's products from SVO, under NJM's names
# ----------------------------------------------------------------------
def test_an_svo_curve_that_differs_from_njms_is_used_with_a_warning(tmp_path, bundled, caplog):
    bundled({"Roman": {"WFI": {"svo_instruments": ["Roman/WFI"], "bands": ["F062", "F087"],
                               "svo_bands": {"Roman/WFI": ["F087"]}}}})
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    library.svo = svo_browser(root, {("Roman", "WFI"): ["F062", "F087"]})
    with caplog.at_level(logging.WARNING):
        directory = library.published("Roman", "WFI")
    assert sorted(p.name for p in directory.glob("*.dat")) == ["F062.dat", "F087.dat"]
    assert "SVO's F062 of Roman/WFI differ from NJM's current curves" in caplog.text


def test_fetch_from_njm_later_replaces_svo_curves_that_differ_and_leaves_no_duplicates(tmp_path, bundled, njm,
                                                                                     caplog):
    bundled({"GAIA": GAIA})
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    library.svo = svo_browser(root, {("GAIA", "GAIA3"): ["G", "Gbp", "Grp", "Grvs"]})
    library.published("GAIA", "GAIA")  # 4 of 6 bands, with a warning
    assert not FilterLibrary.published_complete(root / "GAIA" / "GAIA")

    bands = GAIA["GAIA"]["bands"]
    files = {"index.json": njm_index({"GAIA": {"GAIA": {"filters": bands, "size_bytes": 1}}}),
             "filters/GAIA/GAIA/GAIA": "".join(f"{b}.dat\n" for b in bands)}
    files.update({f"filters/GAIA/GAIA/{b}.dat": CURVE for b in bands})
    library = FilterLibrary(root)
    library.mirror = njm(root, files)
    with caplog.at_level(logging.WARNING):
        assert library.published("GAIA", "GAIA") == root / "GAIA" / "GAIA"
    directory = root / "GAIA" / "GAIA"
    assert sorted(p.stem for p in directory.glob("*.dat")) == bands
    assert caplog.text.count("with NJM's copies, which differ") == 1  # said once, not silent
    assert "replaced G.dat, Gbp.dat, Grp.dat, Grvs.dat with NJM's copies" in caplog.text
    assert FilterLibrary.published_complete(directory)
    verified = json.loads((root.parent / "cache" / "njm" / "filters" / "GAIA" / "GAIA" /
                           "verified_curves.json").read_text())
    assert sorted(verified["files"]) == [f"{b}.dat" for b in bands]


# ----------------------------------------------------------------------
# One installed folder per product, whatever its spelling
# ----------------------------------------------------------------------
def test_njm_download_completes_a_folder_installed_under_another_spelling(tmp_path, bundled, njm):
    bundled({"Gemini": {"GMOS_N": {"svo_instruments": ["Gemini/GMOS-N"]}}})
    root = tmp_path / "filters"
    legacy = write_set(root / "Gemini" / "GMOS-N", ["g"], manifest_bands=["g", "r"])  # incomplete
    library = FilterLibrary(root)
    library.mirror = njm(root, {"index.json": njm_index({"Gemini": {"GMOS_N": {}}}),
                                "filters/Gemini/GMOS_N/GMOS_N": "g.dat\nr.dat\n",
                                "filters/Gemini/GMOS_N/g.dat": CURVE, "filters/Gemini/GMOS_N/r.dat": CURVE})
    assert library.installed_directory("Gemini", "GMOS_N") == legacy
    assert library.published("Gemini", "GMOS_N") == legacy
    assert sorted(p.name for p in (root / "Gemini").iterdir()) == ["GMOS-N"]  # no second copy beside it
    assert (legacy / "GMOS-N").read_text() == "g.dat\nr.dat\n"  # the index MESA looks for in that folder
    assert library.available("Gemini", "GMOS_N")["availability_source"] == "local"


def test_a_complete_folder_under_another_spelling_is_used_as_it_is(tmp_path, bundled):
    root = tmp_path / "filters"
    legacy = write_set(root / "PLANCK" / "HFI", ["100GHz"])
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    assert library.published("Planck", "HFI") == legacy


def test_a_builder_that_would_write_beside_an_installed_spelling_is_not_run(tmp_path, bundled, monkeypatch,
                                                                            caplog):
    root = tmp_path / "filters"
    legacy = write_set(root / "Herschel" / "Pacs_", ["blue"], manifest_bands=["blue", "red"])  # incomplete
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    library.svo = svo_stub({}, {}, down=True)
    monkeypatch.setattr(filter_library, "get_builder", lambda name: pytest.fail("the builder was run"))
    with caplog.at_level(logging.WARNING), pytest.raises(ComponentError) as raised:
        library.published("Herschel", "PACS")
    assert f"its builder writes Herschel/PACS, which would be a second copy beside {legacy}" in str(raised.value)
    assert "trying SVO (the original source is not used: its builder writes Herschel/PACS" in caplog.text
    assert sorted(p.name for p in (root / "Herschel").iterdir()) == ["Pacs_"]


def test_two_installed_spellings_stop_unless_one_is_exact(tmp_path):
    root = tmp_path / "filters"
    write_set(root / "Gemini" / "GMOS-N", ["g"])
    write_set(root / "Gemini" / "GMOS N", ["g"])
    library = FilterLibrary(root)
    with pytest.raises(ComponentError, match="installed more than once.*GMOS N.*GMOS-N"):
        library.published("Gemini", "GMOS_N")
    assert library.installed_directory("Gemini", "GMOS-N") == root / "Gemini" / "GMOS-N"  # exact wins


# ----------------------------------------------------------------------
# Builders that make only components are no download source
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# SVO's availability is known process-wide
# ----------------------------------------------------------------------
def test_svo_that_does_not_answer_is_waited_for_once_per_process(tmp_path, monkeypatch):
    calls = []

    def get(session, url, **kwargs):
        calls.append(url)
        raise requests.ReadTimeout("read timed out")

    monkeypatch.setattr(svo_filter_grabber, "svo_get", get)
    first = FilterLibrary(tmp_path / "a")
    errors = first.facilities()["errors"] + first.instruments("GAIA")["errors"]
    second = FilterLibrary(tmp_path / "b")  # another library, as Filters.fetch makes
    errors += second.facilities()["errors"] + second.instruments("GAIA")["errors"]
    assert len(calls) == 1
    svo_errors = {e for e in errors if e.startswith("SVO")}
    assert len(svo_errors) == 1  # the same words each time, so the CLI prints them once
    [error] = svo_errors
    assert error.startswith("SVO: SVO Filter Profile Service did not answer") and "(ReadTimeout)" in error
    assert error.endswith("SVO is not asked again for 5 minutes.")
    with pytest.raises(SVOUnreachableError):
        SVOFilterBrowser(base_dir=str(tmp_path / "c")).list_facilities()
    assert len(calls) == 1


def test_svo_is_asked_again_after_its_pause(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(svo_filter_grabber, "monotonic", lambda: clock[0])
    monkeypatch.setattr(svo_filter_grabber, "svo_get", lambda *a, **k: (_ for _ in ()).throw(
        SVOUnreachableError("SVO Filter Profile Service unreachable.")))
    browser = SVOFilterBrowser(base_dir=str(tmp_path))
    with pytest.raises(SVOUnreachableError):
        browser.list_facilities()
    assert svo_filter_grabber.svo_unavailable()
    clock[0] += svo_filter_grabber.SVO_PAUSE
    assert svo_filter_grabber.svo_unavailable() is None


# ----------------------------------------------------------------------
# filters-info and filters-cite
# ----------------------------------------------------------------------
def catalogue_with(tmp_path, files):
    server = Served(dict({"index.json": njm_index({"GAIA": {"DR2": {
        "filters": ["BP", "BP_svo", "G"], "properties": "properties.json", "info": "info.json"}}})}, **files))
    mirror = SimpleNamespace(base_url="https://data.nialljmiller.com/sed_tools", is_available=lambda: True,
                             _get=lambda url, **kwargs: server.get(url, **kwargs))
    return FilterCatalogue(tmp_path / "filters", mirror=mirror)


def test_citation_titles_are_one_line_and_url_keys_are_plain(tmp_path):
    text = format_citations({"instruments": [], "acknowledgements": [], "errors": [], "references": [
        {"key": "bibcode:2016A&A...595A...1G", "short": "Prusti et al. 2016", "title": "The\n          Gaia\n  mission",
         "doi": "10.1051/0004-6361/201629272", "used_by": ["GAIA/DR2"]},
        {"key": "url:https://www.cosmos.esa.int/web/gaia/iow_20180316/",
         "url": "https://www.cosmos.esa.int/web/gaia/iow_20180316/", "used_by": ["GAIA/DR2"]}]})
    assert '[1] Prusti et al. 2016. "The Gaia mission". doi:10.1051/0004-6361/201629272  <- GAIA/DR2' in text
    assert "[2] https://www.cosmos.esa.int/web/gaia/iow_20180316/  <- GAIA/DR2" in text


# ----------------------------------------------------------------------
# What available() says without NJM
# ----------------------------------------------------------------------
def test_available_from_svo_describes_njms_product_and_its_missing_bands(tmp_path, bundled):
    bundled({"GAIA": GAIA})
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    library.svo = svo_browser(root, {("GAIA", "GAIA3"): ["G", "Gbp", "Grp", "Grvs"]})
    info = library.available("GAIA", "GAIA")
    assert info["availability_source"] == "svo" and info["bands"] == ["G", "Gbp", "Grp", "Grvs"]
    assert info["notes"] == ["SVO has no curve that is NJM's Gbp_bright, Gbp_faint"]


# ----------------------------------------------------------------------
# Installed atmosphere sites follow NJM's manifest
# ----------------------------------------------------------------------
SERVED = dict(STALE, citation="lsst/throughputs v1.9.2", terms="No licence.",
              entries=[{"airmass": 1.0, "pwv": None, "file": "am1.00.dat"},
                       {"airmass": 1.2, "pwv": None, "file": "am1.20.dat"}])


def test_a_stale_installed_atmosphere_takes_njms_manifest_and_keeps_its_grids(tmp_path, njm, caplog):
    root = tmp_path / "data"
    site = atmosphere_site(root, "lsst_std", STALE, ["am1.20.dat"])
    grid = (site / "am1.20.dat").read_bytes()
    served = json.dumps(SERVED, indent=2, sort_keys=True) + "\n"
    library = FilterLibrary(root / "filters")
    library.mirror = njm(root / "filters", {"atmospheres/": '<a href="lsst_std/">lsst_std/</a>',
                                            "atmospheres/lsst_std/manifest.json": served})
    with caplog.at_level(logging.WARNING):
        listed = library.atmospheres()["lsst_std"]
    assert listed["citation"] == "lsst/throughputs v1.9.2" and listed["airmass"] == [1.0, 1.2]
    assert (site / "manifest.json").read_text() == served and (site / "am1.20.dat").read_bytes() == grid
    assert caplog.text == ""  # a routine update: NJM's citation and terms, a new grid point
    requested = len(library.mirror.session.requested)
    library.atmospheres()  # checked today: only the site list is asked for
    assert library.mirror.session.requested[requested:] == ["atmospheres/"]


def test_points_built_here_and_installed_files_njm_describes_otherwise_are_kept(tmp_path, njm, caplog):
    root = tmp_path / "data"
    local = dict(STALE, entries=[{"airmass": 1.3, "pwv": None, "file": "am1.20.dat"},
                                 {"airmass": 1.5, "pwv": 2.5, "file": "skycalc_am1.50.dat"},
                                 {"airmass": 2.0, "pwv": None, "file": "dropped_by_njm.dat"}])  # never fetched
    site = atmosphere_site(root, "lsst_std", local, ["am1.20.dat", "skycalc_am1.50.dat"])
    library = FilterLibrary(root / "filters")
    library.mirror = njm(root / "filters", {"atmospheres/": '<a href="lsst_std/">lsst_std/</a>',
                                            "atmospheres/lsst_std/manifest.json": json.dumps(SERVED)})
    with caplog.at_level(logging.WARNING):
        library.atmospheres()
    entries = json.loads((site / "manifest.json").read_text())["entries"]
    assert {e["file"]: e["airmass"] for e in entries} == {"am1.00.dat": 1.0, "am1.20.dat": 1.3,
                                                          "skycalc_am1.50.dat": 1.5}
    assert "kept the installed description of am1.20.dat" in caplog.text


def test_installed_atmospheres_are_listed_while_njm_is_down(tmp_path, caplog):
    root = tmp_path / "data"
    atmosphere_site(root, "lsst_std", STALE)
    library = FilterLibrary(root / "filters")
    library.mirror = SimpleNamespace(list_atmosphere_sites=lambda: (_ for _ in ()).throw(
        requests.ConnectionError("NJM is unreachable")))
    with caplog.at_level(logging.WARNING):
        assert list(library.atmospheres()) == ["lsst_std"]
    assert "the installed ones are used" in caplog.text
    library.mirror = NJM_DOWN_ATMOSPHERES
    assert list(library.atmospheres()) == ["lsst_std"]


NJM_DOWN_ATMOSPHERES = SimpleNamespace(list_atmosphere_sites=lambda: [])  # what an unavailable grabber lists


# ----------------------------------------------------------------------
# A partial set from SVO, NJM down: installed, warned about, a success
# ----------------------------------------------------------------------
@pytest.fixture
def gaia_from_svo(tmp_path, bundled, monkeypatch):
    """Every FilterLibrary sees NJM down and an SVO holding 4 of GAIA/GAIA's 6 bands; returns the filter root."""
    bundled({"GAIA": GAIA})
    root = tmp_path / "filters"
    browser = svo_browser(root, {("GAIA", "GAIA3"): ["G", "Gbp", "Grp", "Grvs"]})
    monkeypatch.setattr(FilterLibrary, "mirror", NJM_DOWN)
    monkeypatch.setattr(FilterLibrary, "svo", property(lambda self: browser))
    monkeypatch.setenv("COLUMNS", "80")
    return root


def incomplete_warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and "is incomplete" in r.getMessage()]


def test_filters_info_and_the_picker_summary_say_the_set_is_incomplete(gaia_from_svo, monkeypatch, capsys):
    Filters.fetch("GAIA", "GAIA", filter_root=gaia_from_svo)
    capsys.readouterr()
    assert run_cli(monkeypatch, "filters-info", "GAIA/GAIA", "--base", str(gaia_from_svo), "--offline") == 0
    out = " ".join(capsys.readouterr().out.split())  # wrapped at 80 columns
    assert ("Incomplete : 4 of 6 bands installed; missing Gbp_bright, Gbp_faint; fetch it again when NJM is "
            "reachable to complete it") in out

    info = Filters.available("GAIA", "GAIA", filter_root=gaia_from_svo, include_remote=False)
    lines = cli._pick_summary("GAIA/GAIA", {}, info, {"info": None, "properties": None, "errors": []}, "SVO")
    assert "  Incomplete : 4 of 6 bands installed; missing Gbp_bright, Gbp_faint" in lines
    assert all(len(line) <= 80 for line in lines)


def test_a_complete_set_has_no_incomplete_line(tmp_path, monkeypatch, capsys):
    from helpers_filters import write_set

    root = tmp_path / "filters"
    write_set(root / "Fac" / "Cam", ["B", "V"])
    # An older manifest listing NJM's archived copy among its bands, without the copy installed
    write_set(root / "Fac" / "Old", ["H"], manifest_bands=["H", "H_svo"], archived={"H_svo": {}})
    for target in ("Fac/Cam", "Fac/Old"):
        assert run_cli(monkeypatch, "filters-info", target, "--base", str(root), "--offline") == 0
        assert "Incomplete" not in capsys.readouterr().out
    assert FilterLibrary.published_complete(root / "Fac" / "Old")  # complete for the library too


def test_an_svo_curve_that_could_not_be_saved_still_fails(gaia_from_svo, monkeypatch):
    def refuse(self, row, out_path):
        if row.band == "Grvs":
            raise ValueError("SVO returned no transmission data")
        original(self, row, out_path)

    from sed_tools.svo_filter_grabber import SVOFilterBrowser

    original = SVOFilterBrowser._save_filter
    monkeypatch.setattr(SVOFilterBrowser, "_save_filter", refuse)
    with pytest.raises(ComponentError) as raised:
        Filters.fetch("GAIA", "GAIA", filter_root=gaia_from_svo)
    message = str(raised.value)
    assert "GAIA/GAIA is only partly installed" in message and "3 of NJM's 6 bands came from SVO's GAIA/GAIA3" in message
    assert "GAIA/GAIA3.Grvs: SVO returned no transmission data" in message


# ----------------------------------------------------------------------
# Messages and folders
# ----------------------------------------------------------------------
def test_filters_info_describes_curves_installed_under_other_band_names(tmp_path):
    """0.4.1's PLANCK/HFI from SVO (100GHz ...) is what NJM keeps as 100_svo ...: its records describe it."""
    from types import SimpleNamespace

    from sed_tools.filter_catalogue import FilterCatalogue

    installed = tmp_path / "filters" / "PLANCK" / "HFI"
    installed.mkdir(parents=True)
    for band in ("100GHz", "857GHz"):
        (installed / f"{band}.dat").write_text(f"Wavelength,Transmission,FilterID=PLANCK/HFI.{band}\n1,1\n2,1\n")
    (installed / "HFI").write_text("100GHz.dat\n857GHz.dat\n")
    info = {"bands": {"100": {"Description": "HFI 100", "curve": {"FilterID": "Planck/HFI.100"}},
                      "100_svo": {"Description": "PLANCK HFI 100GH", "curve": {"FilterID": "PLANCK/HFI.100GHz"}},
                      "857": {"Description": "HFI 857", "curve": {"FilterID": "Planck/HFI.857"}},
                      "857_svo": {"Description": "PLANCK HFI 857GHz", "curve": {"FilterID": "PLANCK/HFI.857GHz"}}}}
    properties = {"bands": {"100": {"WavelengthPivot": 30013450.0}, "100_svo": {"WavelengthPivot": 30035700.0},
                            "857": {"WavelengthPivot": 3525232.0}, "857_svo": {"WavelengthPivot": 3527564.0}}}
    server = Served({"index.json": njm_index({"Planck": {"HFI": {
                         "filters": ["100", "100_svo", "857", "857_svo"], "properties": "properties.json",
                         "info": "info.json"}}}),
                     "filters/Planck/HFI/HFI": "100.dat\n857.dat\n",
                     "filters/Planck/HFI/info.json": json.dumps(info),
                     "filters/Planck/HFI/properties.json": json.dumps(properties)})
    mirror = SimpleNamespace(base_url="https://data.nialljmiller.com/sed_tools", is_available=lambda: True,
                             _get=lambda url, **kwargs: server.get(url, **kwargs))
    catalogue = FilterCatalogue(tmp_path / "filters", mirror=mirror)

    described = catalogue.info("PLANCK", "HFI")
    assert {band: record["Description"] for band, record in described["bands"].items()} == {
        "100GHz": "PLANCK HFI 100GH", "857GHz": "PLANCK HFI 857GHz"}
    assert sorted(described["archived_bands"]) == ["100", "857"]  # NJM's own curves, not installed
    pivots = catalogue.properties("PLANCK", "HFI")["bands"]
    assert {band: record["WavelengthPivot"] for band, record in pivots.items()} == {
        "100GHz": 30035700.0, "857GHz": 3527564.0}


def test_a_set_njm_completes_gets_the_manifest_of_a_fresh_download(tmp_path, bundled, njm_filters):
    bundled({"GAIA": GAIA})
    bands = GAIA["GAIA"]["bands"]
    curve = CURVE.replace("original\n", "original;Components=filter+instrument;ComponentsSource=original\n")
    files = {"index.json": njm_index({"GAIA": {"GAIA": {"filters": bands, "size_bytes": 1}}}),
             "filters/GAIA/GAIA/GAIA": "".join(f"{b}.dat\n" for b in bands)}
    files.update({f"filters/GAIA/GAIA/{b}.dat": curve for b in bands})

    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    library.svo = svo_browser(root, {("GAIA", "GAIA3"): ["G", "Gbp", "Grp", "Grvs"]})
    library.published("GAIA", "GAIA")  # 4 of 6 bands, with a warning
    library = FilterLibrary(root)
    library.mirror = njm_filters(root, files)
    library.published("GAIA", "GAIA")

    fresh = FilterLibrary(tmp_path / "fresh")
    fresh.mirror = njm_filters(tmp_path / "fresh", files)
    fresh.published("GAIA", "GAIA")
    completed = json.loads((root / "GAIA" / "GAIA" / "manifest.json").read_text())
    assert sorted(completed["detector_types"]) == sorted(completed["published_bands"]) == bands
    assert completed == json.loads((tmp_path / "fresh" / "GAIA" / "GAIA" / "manifest.json").read_text())


NJM = "https://data.nialljmiller.com/sed_tools"


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


def test_a_curve_without_a_column_row_is_reported_once_by_its_caller(tmp_path, caplog):
    from sed_tools.filter_metadata import migrate_filter_file, write_filter_header

    bare = tmp_path / "bare.dat"
    bare.write_text("4000,0.1\n5000,0.9\n")
    with caplog.at_level(logging.WARNING):
        assert write_filter_header(bare, {"DetectorType": "photon"}) is False  # the caller reports it
    assert caplog.text == ""
    legacy = tmp_path / "legacy.dat"
    legacy.write_text("# DetectorType: photon\n4000,0.1\n5000,0.9\n")  # tagged by SED_Tools 0.3.0
    with caplog.at_level(logging.WARNING):
        assert migrate_filter_file(legacy) is True
    assert caplog.text.count("has no column-name row to hold SED_Tools metadata") == 1
    assert legacy.read_text() == "4000,0.1\n5000,0.9\n"


@pytest.mark.parametrize("njm_answer, warned", [(None, False), (RuntimeError("2 component files failed"), True)])
def test_components_njm_says_it_does_not_have_are_not_announced_without_a_builder(tmp_path, caplog, njm_answer, warned):
    from sed_tools import filter_metadata as fm
    from sed_tools.filter_library import FilterLibrary

    directory = tmp_path / "NoBuilderFac" / "Cam"
    directory.mkdir(parents=True)
    fm.write_filter_file(directory / "g.dat", [4000.0, 6000.0], [1.0, 1.0], detector_type=fm.PHOTON,
                         source="svo", wavelength_unit="Angstrom", extra={"Components": "filter+optics+qe"})

    def download_components(facility, instrument):
        if njm_answer is not None:
            raise njm_answer
        return 0  # NJM serves no components for it

    library = FilterLibrary(tmp_path)
    library.mirror = SimpleNamespace(is_available=lambda: True, download_components=download_components)
    with caplog.at_level(logging.WARNING):
        assert library.prepare_components(directory, components="filter+qe") is False
    assert ("NJM cannot provide the components of NoBuilderFac/Cam (2 component files failed)" in caplog.text) == warned
    assert bool(caplog.text) == warned


def test_an_unreadable_verification_cache_means_every_curve_is_checked(tmp_path):
    grabber = NJMFilterGrabber.__new__(NJMFilterGrabber)
    damaged = tmp_path / "verified_curves.json"
    damaged.write_text("{not json")
    assert grabber._saved_verified(damaged, "record") == {}
    locked = tmp_path / "locked.json"
    locked.write_text('{"njm_record": "record", "files": {"g.dat": 10}}')
    locked.chmod(0)
    try:
        assert grabber._saved_verified(locked, "record") == {}  # not NJM's fault: never a fallback to SVO
    finally:
        locked.chmod(0o600)
    damaged.write_text('{"njm_record": "old", "files": {"g.dat": 10}}')
    assert grabber._saved_verified(damaged, "record") == {}  # NJM's set changed since
    assert grabber._saved_verified(damaged, "old") == {"g.dat": 10}


def test_a_manifest_answer_without_a_body_is_unreadable_not_absent(tmp_path):
    import json

    def get(url, **kwargs):
        return SimpleNamespace(status_code=204, text="", raise_for_status=lambda: None, json=lambda: json.loads(""))

    grabber = NJMFilterGrabber.__new__(NJMFilterGrabber)
    grabber.base_url, grabber.filters_url, grabber._available = NJM, f"{NJM}/filters", True
    grabber.session = SimpleNamespace(get=get, headers={}, verify=True)
    with pytest.raises(ValueError, match="unreadable manifest.json for Fac/Cam"):
        grabber.fetch_manifest("Fac", "Cam")


@pytest.mark.parametrize("njm_up", [True])
def test_an_njm_outage_is_not_announced_again_for_each_fetch(tmp_path, caplog, njm_up):
    from sed_tools.filter_library import FilterLibrary
    from sed_tools.njm_filter_grabber import SetDownload
    from helpers_filters import svo_browser

    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = SimpleNamespace(is_available=lambda: njm_up, _download_set=lambda f, i: SetDownload(listed=[]))
    library.svo = svo_browser(root, {("Fac", "Cam"): ["g", "r"]})
    with caplog.at_level(logging.WARNING, logger="sed_tools.filter_library"):
        assert library.published("Fac", "Cam") == root / "Fac" / "Cam"
    # Down: the NJM client said so once for the process.  Up but without the set: this fetch says so.
    assert ("NJM cannot provide Fac/Cam (it does not serve this instrument); trying SVO" in caplog.text) == njm_up
    assert ("NJM cannot provide" in caplog.text) == njm_up


def test_a_set_whose_njm_index_cannot_be_saved_is_never_returned(tmp_path, monkeypatch):
    from sed_tools.filter_components import ComponentError
    from sed_tools.filter_library import FilterLibrary
    from helpers_filters import Served

    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.time.sleep", lambda seconds: None)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.tag_filter_file", lambda *a, **k: None)
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJMFilterGrabber(base_dir=str(root))
    library.mirror.session = Served({"filters/Fac/Cam/Cam": 500, "filters/Fac/Cam/": '<a href="g.dat">g.dat</a>',
                                     "filters/Fac/Cam/g.dat": CURVE})
    with pytest.raises(ComponentError, match=r"NJM's index of Fac/Cam could not be saved in .*Fac/Cam") as raised:
        library.published("Fac", "Cam")
    assert "run again" in str(raised.value)
    assert (root / "Fac" / "Cam" / "g.dat").is_file() and not (root / "Fac" / "Cam" / "Cam").exists()


def test_the_catalogue_finds_an_installed_copy_as_fetch_does(tmp_path):
    import json

    from sed_tools.filter_catalogue import FilterCatalogue

    installed = tmp_path / "filters" / "Gemini" / "GMOS-N"  # NJM spells it GMOS_N
    installed.mkdir(parents=True)
    (installed / "g.dat").write_text(CURVE)
    (installed / "info.json").write_text(json.dumps({"description": "as installed", "bands": {"g": {}}}))
    catalogue = FilterCatalogue(tmp_path / "filters", offline=True)
    assert catalogue.info("Gemini", "GMOS_N")["description"] == "as installed"
    assert catalogue.origin("Gemini", "GMOS_N", "info")["source"] == "installed"
