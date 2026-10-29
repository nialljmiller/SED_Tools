"""Second cleanup of the filters area: handlers whose wording or path changed.

Every source is mocked; nothing here touches the network.
"""

import logging
from types import SimpleNamespace

import pytest
import requests

from sed_tools.njm_filter_grabber import NJMFilterGrabber

NJM = "https://nillmill.ddns.net/sed_tools"


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


def test_a_block_that_does_not_lift_is_reported_as_the_http_status(monkeypatch, caplog):
    grabber = _refusing_grabber(monkeypatch, 403)
    with caplog.at_level(logging.WARNING):
        assert grabber._check_availability() is False
    assert caplog.text.count(f"NJM mirror {NJM} is unavailable (HTTP 403)") == 1
    assert "could not connect" not in caplog.text
    with pytest.raises(requests.ConnectionError, match=r"NJM is unreachable \(HTTP 403\)"):
        grabber._get(f"{NJM}/index.json")


def test_a_partial_svo_fill_with_curves_that_differ_from_njms_gives_one_warning(tmp_path, monkeypatch, caplog):
    from sed_tools import filter_library
    from sed_tools.filter_library import FilterLibrary, instrument_catalogue
    from test_release_njmdown import NJM_DOWN, svo_browser

    record = {"svo_instruments": ["Roman/WFI"], "bands": ["F062", "F087", "F106"],
              "svo_bands": {"Roman/WFI": ["F087"]}}
    monkeypatch.setattr(filter_library, "load_instrument_catalogue",
                        lambda: instrument_catalogue({"Roman": {"instruments": {"WFI": record}}}))
    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    library.svo = svo_browser(root, {("Roman", "WFI"): ["F062", "F087"]})
    with caplog.at_level(logging.WARNING, logger="sed_tools.filter_library"):
        library.published("Roman", "WFI")
    after_svo = [r.getMessage() for r in caplog.records if "trying SVO" not in r.getMessage()]
    assert len(after_svo) == 1
    [warning] = after_svo
    assert "Roman/WFI is incomplete" in warning and "missing F106" in " ".join(warning.split())
    assert "SVO's F062 of Roman/WFI differ from NJM's current curves" in " ".join(warning.split())


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


CURVE = "Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=original\n4000,0\n5000,1\n6000,0\n"


@pytest.mark.parametrize("njm_up", [False, True])
def test_an_njm_outage_is_not_announced_again_for_each_fetch(tmp_path, caplog, njm_up):
    from sed_tools.filter_library import FilterLibrary
    from sed_tools.njm_filter_grabber import SetDownload
    from test_release_njmdown import svo_browser

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
    from test_release_library import Served

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
