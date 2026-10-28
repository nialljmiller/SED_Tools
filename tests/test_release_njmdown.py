"""0.5.0 with NJM or SVO down: NJM's names, SVO under NJM's names, one folder per product, fail fast.

Every source is mocked; nothing here touches the network.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from sed_tools import filter_library, svo_filter_grabber
from sed_tools.filter_catalogue import CatalogueError, FilterCatalogue, format_citations
from sed_tools.filter_components import ComponentError
from sed_tools.filter_library import FilterLibrary, instrument_catalogue
from sed_tools.filter_metadata import SVOUnreachableError, read_filter_header
from sed_tools.njm_filter_grabber import NJMFilterGrabber
from sed_tools.svo_filter_grabber import Facility, FilterRow, Instrument, SVOFilterBrowser
from test_release_library import CURVE, Served, njm_index, svo_stub, write_set

GAIA = {"DR2": {"description": "Gaia DR2 passbands (Evans et al. 2018)", "svo_instruments": ["GAIA/GAIA2"],
                "bands": ["BP", "G", "RP"], "svo_bands": {"GAIA/GAIA2": ["G", "Gbp=BP", "Grp=RP"]}},
        "GAIA": {"description": "Gaia DR3 passbands", "svo_instruments": ["GAIA/GAIA3"],
                 "bands": ["G", "Gbp", "Gbp_bright", "Gbp_faint", "Grp", "Grvs"],
                 "svo_bands": {"GAIA/GAIA3": ["G", "Gbp", "Grp", "Grvs"]}},
        "GAIA0": {"svo_instruments": ["GAIA/GAIA0"], "bands": ["G"], "svo_bands": {"GAIA/GAIA0": ["G"]}}}
EUCLID = {"NISP": {"description": "Euclid NISP", "svo_instruments": ["Euclid/NISP", "Euclid/NISP0"],
                   "bands": ["Blue", "H", "J", "Red", "Y"],
                   "svo_bands": {"Euclid/NISP": ["H", "J", "Y"], "Euclid/NISP0": ["Blue", "Red"]}}}


@pytest.fixture
def bundled(monkeypatch):
    def use(catalogue):
        facilities = {fac: {"instruments": insts} for fac, insts in catalogue.items()}
        monkeypatch.setattr(filter_library, "load_instrument_catalogue", lambda: instrument_catalogue(facilities))
    use({})
    return use


@pytest.fixture
def njm(monkeypatch):
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.time.sleep", lambda seconds: None)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.tag_filter_file", lambda *a, **k: None)

    def make(root, files):
        grabber = NJMFilterGrabber(base_dir=str(root))
        grabber.session = Served(files)
        return grabber

    return make


#: An NJMFilterGrabber that found NJM down: it asks nothing and lists nothing.
NJM_DOWN = SimpleNamespace(is_available=lambda: False, fetch_manifest=lambda f, i: None,
                           discover_filters=lambda f, i: [])


def svo_browser(root, filters, *, detector="photon"):
    """A real SVOFilterBrowser whose listings and curves come from ``filters`` {(facility, instrument): [band]}."""
    browser = SVOFilterBrowser(base_dir=str(root))
    browser.list_facilities = lambda: [Facility(f, f) for f in sorted({f for f, _ in filters})]
    browser.list_instruments = lambda fac: [Instrument(fac, i, i) for f, i in filters if f == fac]
    browser.list_filters = lambda fac, inst: [FilterRow(f"{fac}/{inst}.{b}", b, fac, inst, "")
                                              for b in filters[(fac, inst)]]
    curve = SimpleNamespace(wavelength=np.array([4000.0, 5000.0, 6000.0]), transmission=np.array([0.0, 1.0, 0.0]),
                            detector_type=detector, wavelength_unit="Angstrom", components="Filter")
    browser.saved = []
    browser.metadata_client = SimpleNamespace(get_filter=lambda fid: browser.saved.append(fid) or curve)
    return browser


def labels(result):
    return [(e["label"], e["sources"]) for e in result["entries"]]


# ----------------------------------------------------------------------
# 1. The pickers name SVO's copies after NJM while NJM is down
# ----------------------------------------------------------------------
def test_two_svo_instruments_of_one_njm_product_are_one_entry(tmp_path, bundled):
    bundled({"Euclid": EUCLID})
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({"Euclid": ""}, {"Euclid": {"NISP": "SVO NISP", "NISP0": "pre-launch", "VIS": "v"}})
    library.mirror = NJM_DOWN
    assert labels(library.instruments("Euclid")) == [("NISP", ["svo"]), ("VIS", ["svo"])]


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
# 2. NJM's products from SVO, under NJM's names
# ----------------------------------------------------------------------
def test_njm_product_from_svo_is_installed_under_njm_names_and_says_what_is_missing(tmp_path, bundled, caplog):
    bundled({"GAIA": GAIA})
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    library.svo = svo_browser(root, {("GAIA", "GAIA3"): ["G", "Gbp", "Grp", "Grvs"], ("GAIA", "GAIA2"): ["G"]})
    with caplog.at_level(logging.WARNING), pytest.raises(ComponentError) as raised:
        library.published("GAIA", "GAIA")
    directory = root / "GAIA" / "GAIA"
    message = str(raised.value)
    assert message.startswith(f"GAIA/GAIA is only partly installed, in {directory}: 4 of NJM's 6 bands came from "
                              "SVO's GAIA/GAIA3: G, Gbp, Grp, Grvs; SVO has no curve that is NJM's Gbp_bright, "
                              "Gbp_faint")
    assert "Fetch it again when NJM is reachable" in message and "NJM: the mirror is unavailable" in message
    assert "trying SVO" in caplog.text
    assert sorted(p.name for p in directory.glob("*.dat")) == ["G.dat", "Gbp.dat", "Grp.dat", "Grvs.dat"]
    assert read_filter_header(directory / "Gbp.dat")["FilterID"] == "GAIA/GAIA3.Gbp"  # where the curve came from
    assert json.loads((directory / "manifest.json").read_text())["bands"] == GAIA["GAIA"]["bands"]
    assert not FilterLibrary.published_complete(directory)  # a later fetch asks NJM again
    assert not (root / "GAIA" / "GAIA3").exists()


def test_svo_band_names_become_njms(tmp_path, bundled, monkeypatch):
    bundled({"GAIA": GAIA})
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    monkeypatch.setattr(library, "_publishing_builder", lambda f, i, d: (None, None, "builder offline"))
    library.svo = svo_browser(root, {("GAIA", "GAIA2"): ["G", "Gbp", "Grp"]}, detector="energy")
    directory = library.published("GAIA", "DR2")
    assert directory == root / "GAIA" / "DR2"
    assert (directory / "DR2").read_text().split() == ["BP.dat", "G.dat", "RP.dat"]
    assert read_filter_header(directory / "BP.dat")["DetectorType"] == "energy"  # SVO's form, as SVO labels it


def test_several_svo_instruments_fill_one_njm_product_without_their_archived_copies(tmp_path, bundled):
    bundled({"Euclid": EUCLID})
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    library.svo = svo_browser(root, {("Euclid", "NISP"): ["H", "J", "Y"],
                                     ("Euclid", "NISP0"): ["Blue", "H", "J", "Red", "Y"]})
    directory = library.published("Euclid", "NISP")
    assert directory == root / "Euclid" / "NISP" and FilterLibrary.published_complete(directory)
    taken = {read_filter_header(p)["FilterID"] for p in directory.glob("*.dat")}
    # NISP0's H, J and Y are NJM's archived H_svo, J_svo, Y_svo: never NJM's H, J and Y.
    assert taken == {"Euclid/NISP.H", "Euclid/NISP.J", "Euclid/NISP.Y", "Euclid/NISP0.Blue", "Euclid/NISP0.Red"}


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
    with pytest.raises(ComponentError, match="only partly installed"):
        library.published("GAIA", "GAIA")

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
    assert caplog.text.count("differs from NJM's copy; replacing it with NJM's") == 4  # said, not silent
    assert FilterLibrary.published_complete(directory)
    verified = json.loads((root.parent / "cache" / "njm" / "filters" / "GAIA" / "GAIA" /
                           "verified_curves.json").read_text())
    assert sorted(verified["files"]) == [f"{b}.dat" for b in bands]


# ----------------------------------------------------------------------
# 3. One installed folder per product, whatever its spelling
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
    assert "The original source of Herschel/PACS is not used" in caplog.text
    assert sorted(p.name for p in (root / "Herschel").iterdir()) == ["Pacs_"]


def test_two_installed_spellings_stop_unless_one_is_exact(tmp_path):
    root = tmp_path / "filters"
    write_set(root / "Gemini" / "GMOS-N", ["g"])
    write_set(root / "Gemini" / "GMOS N", ["g"])
    library = FilterLibrary(root)
    with pytest.raises(ComponentError, match="installed more than once.*GMOS N.*GMOS-N"):
        library.published("Gemini", "GMOS_N")
    assert library.installed_directory("Gemini", "GMOS-N") == root / "Gemini" / "GMOS-N"  # exact wins


def test_another_products_exact_name_is_never_taken(tmp_path, bundled):
    bundled({"Cameras": {"Sony_NEX5N": {}, "Sony_NEX_5N": {}}})
    root = tmp_path / "filters"
    write_set(root / "Cameras" / "Sony_NEX5N", ["g"])
    library = FilterLibrary(root)
    assert library.installed_directory("Cameras", "Sony_NEX_5N") is None
    assert library.installed_directory("Cameras", "sony-nex5n") is not None  # a spelling of the installed one


def test_folders_without_curves_do_not_count(tmp_path):
    root = tmp_path / "filters"
    (root / "Gemini" / "GMOS_N").mkdir(parents=True)  # left empty by a failed attempt
    legacy = write_set(root / "Gemini" / "GMOS-N", ["g"])
    assert FilterLibrary(root).installed_directory("Gemini", "GMOS_N") == legacy


# ----------------------------------------------------------------------
# 4. Builders that make only components are no download source
# ----------------------------------------------------------------------
def test_component_only_builders_are_not_listed_or_tried(tmp_path, bundled, caplog):
    library = FilterLibrary(tmp_path / "filters")
    library.svo = svo_stub({}, {}, down=True)
    library.mirror = NJM_DOWN
    assert "2MASS" not in [e["label"] for e in library.facilities()["entries"]]
    with caplog.at_level(logging.WARNING), pytest.raises(ComponentError) as raised:
        library.published("2MASS", "2MASS")
    assert "original source: its builder makes only the separate components" in str(raised.value)
    assert "trying its original-source builder" not in caplog.text


def test_with_svo_known_down_no_fallback_to_it_is_announced(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(svo_filter_grabber._STATE, "error", "SVO Filter Profile Service unreachable: x.")
    monkeypatch.setattr(svo_filter_grabber._STATE, "until", float("inf"))
    library = FilterLibrary(tmp_path / "filters")
    library.mirror = NJM_DOWN
    with caplog.at_level(logging.WARNING), pytest.raises(ComponentError) as raised:
        library.published("2MASS", "2MASS")
    assert "trying SVO" not in caplog.text and caplog.text.count("\n") == 0  # one message: the error
    assert str(raised.value).endswith("SVO: SVO Filter Profile Service unreachable: x.")


# ----------------------------------------------------------------------
# 5. SVO's availability is known process-wide
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
# 6. Grid discovery fails fast; NJM's spectra share NJM's outage
# ----------------------------------------------------------------------
def test_grid_indexes_are_asked_once_and_downloads_keep_their_retries(tmp_path):
    from sed_tools.msg_spectra_grabber import MSGSpectraGrabber
    from sed_tools.svo_spectra_grabber import DISCOVERY_TIMEOUT, SVOSpectraGrabber

    svo = SVOSpectraGrabber(base_dir=str(tmp_path / "svo"))
    msg = MSGSpectraGrabber(base_dir=str(tmp_path / "msg"))
    assert svo.session.get_adapter(svo.model_index_url).max_retries.total == 0
    assert msg.session.get_adapter(msg.index_url).max_retries.total == 0
    assert svo.session.get_adapter(svo.spectra_base_url).max_retries.total == 5
    assert sum(DISCOVERY_TIMEOUT) <= 15


def test_njm_spectra_are_not_asked_while_njm_is_paused(tmp_path, monkeypatch):
    from sed_tools.njm_spectra_grabber import NJMSpectraGrabber

    NJMFilterGrabber._pause("refused", "NJM mirror is unavailable (refused)")
    monkeypatch.setattr(NJMSpectraGrabber, "_check_availability", lambda self: pytest.fail("NJM was asked"))
    assert not NJMSpectraGrabber(base_dir=str(tmp_path)).is_available()


# ----------------------------------------------------------------------
# filters-info and filters-cite
# ----------------------------------------------------------------------
def catalogue_with(tmp_path, files):
    server = Served(dict({"index.json": njm_index({"GAIA": {"DR2": {
        "filters": ["BP", "BP_svo", "G"], "properties": "properties.json", "info": "info.json"}}})}, **files))
    mirror = SimpleNamespace(base_url="https://nillmill.ddns.net/sed_tools", is_available=lambda: True,
                             _get=lambda url, **kwargs: server.get(url, **kwargs))
    return FilterCatalogue(tmp_path / "filters", mirror=mirror)


def test_info_puts_archived_copies_apart_from_the_bands(tmp_path):
    bands = {"BP": {"Description": "Gbp"}, "BP_svo": {"Description": "Gbp (SVO)"}, "G": {"Description": "G"}}
    catalogue = catalogue_with(tmp_path, {"filters/GAIA/DR2/info.json": json.dumps({"bands": bands}),
                                          "filters/GAIA/DR2/DR2": "BP.dat\nG.dat\n"})
    info = catalogue.info("GAIA", "DR2")
    assert list(info["bands"]) == ["BP", "G"] and list(info["archived_bands"]) == ["BP_svo"]


def test_installed_index_decides_which_bands_are_archived(tmp_path):
    directory = tmp_path / "filters" / "GAIA" / "DR2"
    directory.mkdir(parents=True)
    (directory / "DR2").write_text("BP.dat\nG.dat\n")
    (directory / "properties.json").write_text(json.dumps({"bands": {"BP": {}, "G": {}, "G_svo": {}}}))
    catalogue = FilterCatalogue(tmp_path / "filters", offline=True)
    assert list(catalogue.properties("GAIA", "DR2")["archived_bands"]) == ["G_svo"]


def test_unreachable_njm_is_said_plainly_once(tmp_path):
    def down(url, **kwargs):
        raise requests.ConnectionError("('Connection aborted.', ConnectionResetError(54, 'reset'))")

    mirror = SimpleNamespace(base_url="https://nillmill.ddns.net/sed_tools", is_available=lambda: True, _get=down)
    catalogue = FilterCatalogue(tmp_path / "filters", mirror=mirror)
    with pytest.raises(CatalogueError) as raised:
        catalogue.info("GAIA", "DR2")
    assert str(raised.value) == "index.json is not cached and NJM is unreachable"


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
def test_available_without_any_source_gives_the_bundled_bands(tmp_path, bundled):
    bundled({"GAIA": GAIA})
    library = FilterLibrary(tmp_path / "filters")
    library.mirror = NJM_DOWN
    library.svo = svo_stub({}, {}, down=True)
    info = library.available("GAIA", "GAIA")
    assert info["availability_source"] == "unknown" and info["bands"] == GAIA["GAIA"]["bands"]
    assert "released" in info["notes"][0]


def test_available_from_svo_describes_njms_product_and_its_missing_bands(tmp_path, bundled):
    bundled({"GAIA": GAIA})
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    library.svo = svo_browser(root, {("GAIA", "GAIA3"): ["G", "Gbp", "Grp", "Grvs"]})
    info = library.available("GAIA", "GAIA")
    assert info["availability_source"] == "svo" and info["bands"] == ["G", "Gbp", "Grp", "Grvs"]
    assert info["notes"] == ["SVO has no curve that is NJM's Gbp_bright, Gbp_faint"]


def test_available_without_njm_follows_published_to_the_builder_before_svo(tmp_path, bundled):
    bundled({"Planck": {"HFI": {"svo_instruments": ["PLANCK/HFI"], "bands": ["100", "545"],
                                "svo_bands": {"PLANCK/HFI": ["545GHz=545"]}}}})
    library = FilterLibrary(tmp_path / "filters")
    library.mirror = NJM_DOWN
    library.svo = svo_stub({}, {}, down=True)  # never asked: the builder comes first
    info = library.available("Planck", "HFI")
    assert info["availability_source"] == "original" and info["bands"] == ["100", "545"]
    assert info["notes"][0] == "its original-source builder makes the curves when it is fetched"
