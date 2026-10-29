"""Filter handlers that now raise or warn instead of carrying on silently (cleanup of the filters area).

Every network call is mocked; a real connection fails the test.
"""

import logging

import pytest

from sed_tools import filter_metadata as fm


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.delenv(fm.SVO_FPS_URL_ENV, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("test attempted a real network connection")

    monkeypatch.setattr("socket.socket.connect", refuse)


def _curve(path, header="Wavelength,Transmission,WavelengthUnit=Angstrom"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n4000.0,0.1\n5000.0,1.0\n6000.0,0.1\n")
    return path


# ----------------------------------------------------------------------
# filter_colors
# ----------------------------------------------------------------------
def test_instrument_colors_refuses_an_unreadable_curve(tmp_path):
    from sed_tools.filter_colors import instrument_colors

    good = _curve(tmp_path / "g.dat")
    empty = tmp_path / "r.dat"
    empty.write_text("Wavelength,Transmission\n")
    with pytest.raises(ValueError, match="r.dat"):
        instrument_colors([good, empty])


# ----------------------------------------------------------------------
# filter_catalogue
# ----------------------------------------------------------------------
def _cached(catalogue, rel, text, checked):
    import json

    path = catalogue.cache_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.with_name(path.name + ".http.json").write_text(json.dumps({"checked": checked}))
    return path


def _down_mirror():
    import requests
    from types import SimpleNamespace

    def get(url, **kwargs):
        raise requests.ConnectionError("mirror down")

    return SimpleNamespace(base_url="https://njm.invalid/sed_tools", is_available=lambda: False, _get=get)


def test_unreadable_installed_catalogue_file_is_refused(tmp_path):
    from sed_tools.filter_catalogue import CatalogueError, FilterCatalogue

    root = tmp_path / "filters"
    (root / "Fac" / "Cam").mkdir(parents=True)
    (root / "Fac" / "Cam" / "g.dat").write_text("Wavelength,Transmission\n4000,1\n6000,1\n")  # an installed copy holds curves
    (root / "Fac" / "Cam" / "properties.json").write_text("{not json")
    catalogue = FilterCatalogue(root, mirror=_down_mirror())
    with pytest.raises(CatalogueError, match="properties.json is unreadable"):
        catalogue.properties("Fac", "Cam")


def test_stale_cached_index_used_without_the_mirror_is_reported(tmp_path, caplog):
    from sed_tools.filter_catalogue import FilterCatalogue

    catalogue = FilterCatalogue(tmp_path / "filters", mirror=_down_mirror(), index_max_age=60)
    _cached(catalogue, "index.json", '{"filters": {}}', checked=0)
    with caplog.at_level(logging.WARNING):
        assert catalogue.index() == {"filters": {}}
    assert "Cannot ask NJM (unreachable); using the cached index.json" in caplog.text


def test_offline_catalogue_uses_its_cache_without_a_warning(tmp_path, caplog):
    from sed_tools.filter_catalogue import FilterCatalogue

    catalogue = FilterCatalogue(tmp_path / "filters", offline=True, index_max_age=60)
    _cached(catalogue, "index.json", '{"filters": {}}', checked=0)
    with caplog.at_level(logging.WARNING):
        assert catalogue.index() == {"filters": {}}
    assert caplog.text == ""


def test_vega_path_without_an_index_stops(tmp_path):
    import time

    from sed_tools.filter_catalogue import VEGA_URL, CatalogueError, FilterCatalogue

    catalogue = FilterCatalogue(tmp_path / "filters", mirror=_down_mirror())
    catalogue._mirror.is_available = lambda: True
    _cached(catalogue, VEGA_URL, "WAVELENGTH,FLUX\n5000,1e-9\n", checked=time.time())
    with pytest.raises(CatalogueError, match="index.json"):
        catalogue.vega()  # the Vega spectrum's path is NJM's index.json's to say


# ----------------------------------------------------------------------
# filter_components
# ----------------------------------------------------------------------
def test_unknown_components_header_is_reported(caplog):
    from sed_tools.filter_components import InstrumentManifest

    headers = {"g": {"Components": "filter+warp"}, "r": {"Components": "filter"}}
    with caplog.at_level(logging.WARNING):
        manifest = InstrumentManifest.from_headers("Fac", "Cam", headers)
    assert manifest.published_bands == {"r": "filter"}
    assert "Fac/Cam band g" in caplog.text and "'warp'" in caplog.text


# ----------------------------------------------------------------------
# svo_filter_grabber
# ----------------------------------------------------------------------
def _svo_record(filter_id):
    import numpy as np

    return fm.SVOFilterRecord(filter_id, fm.PHOTON, np.array([4000.0, 6000.0]), np.ones(2), "Angstrom", "Filter")


def test_svo_download_saves_what_it_can_then_lists_the_failures(tmp_path):
    import requests

    from sed_tools.filter_components import InstrumentManifest
    from sed_tools.svo_filter_grabber import FilterRow, SVOFilterBrowser

    browser = SVOFilterBrowser(base_dir=str(tmp_path))

    def get_filter(filter_id):
        if filter_id.endswith(".r"):
            return None
        if filter_id.endswith(".i"):
            raise requests.HTTPError("500 Server Error")
        return _svo_record(filter_id)

    browser.metadata_client.get_filter = get_filter
    rows = [FilterRow(f"Fac/Cam.{b}", b, "Fac", "Cam", "") for b in "gri"]
    with pytest.raises(RuntimeError) as info:
        browser.download_filters(rows)
    message = str(info.value)
    assert "2 of 3 SVO filters could not be saved" in message
    assert "Fac/Cam.r: SVO returned no transmission data" in message and "Fac/Cam.i: 500" in message
    directory = tmp_path / "Fac" / "Cam"
    assert (directory / "Cam").read_text() == "g.dat\n"
    assert InstrumentManifest.load(directory).missing_files(directory) == ["i.dat", "r.dat"]


def test_svo_bulk_download_finishes_then_reports_failed_instruments():
    from types import SimpleNamespace

    from sed_tools.svo_filter_grabber import Facility, Instrument, _download_facilities

    downloaded = []

    def download(filters):
        downloaded.append(filters[0])
        if filters[0] == "A/one.g":
            raise RuntimeError("1 of 1 SVO filters could not be saved")

    browser = SimpleNamespace(
        list_instruments=lambda key: [Instrument(key, "one", "One"), Instrument(key, "two", "Two")],
        list_filters=lambda fac, inst: [f"{fac}/{inst}.g"],
        download_filters=download,
    )
    with pytest.raises(RuntimeError, match="Bulk download incomplete:\nA/One: 1 of 1"):
        _download_facilities(browser, [Facility("A", "A"), Facility("B", "B")])
    assert downloaded == ["A/one.g", "A/two.g", "B/one.g", "B/two.g"]


# ----------------------------------------------------------------------
# filter_library
# ----------------------------------------------------------------------
def test_published_reports_each_fallback_and_every_source_failure(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace

    from requests import ConnectionError

    from sed_tools import filter_library
    from sed_tools.component_sources import catalog
    from sed_tools.filter_components import ComponentError
    from sed_tools.filter_library import FilterLibrary

    library = FilterLibrary(tmp_path)
    library.mirror = SimpleNamespace(is_available=lambda: False)
    library.svo = SimpleNamespace(list_facilities=lambda: (_ for _ in ()).throw(RuntimeError("SVO index down")))
    monkeypatch.setitem(catalog.BUILDERS_FOR, ("Fac", "Cam"), catalog.BuilderRef("test", publishes=True))

    def builder(root, **kwargs):
        raise ConnectionError("team site offline")

    monkeypatch.setattr(filter_library, "get_builder", lambda name: builder)
    with caplog.at_level(logging.WARNING), pytest.raises(ComponentError) as info:
        library.published("Fac", "Cam")
    assert str(info.value) == ("No source can provide Fac/Cam: NJM: the mirror is unavailable; "
                               "original source: team site offline; SVO: SVO index down")
    assert "NJM cannot provide" not in caplog.text  # an NJM outage is the NJM client's one notice per process
    assert "The original source cannot provide Fac/Cam (team site offline); trying SVO" in caplog.text


# ----------------------------------------------------------------------
# njm_filter_grabber (a fake session; never the network)
# ----------------------------------------------------------------------
NJM = "https://nillmill.ddns.net/sed_tools"
CURVE = "Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=svo\n4000,0.1\n5000,1\n6000,0.1\n"


class Served:
    """A fake NJM session: ``files`` maps paths under the mirror to text, or to an HTTP status."""

    def __init__(self, files):
        self.files = dict(files)
        self.headers, self.verify, self.requested = {}, True, []

    def get(self, url, timeout=None, headers=None, **kwargs):
        import json
        from types import SimpleNamespace

        import requests

        rel = url[len(NJM) + 1:]
        self.requested.append(rel)
        body = self.files.get(rel, 404)
        status, text = (body, "") if isinstance(body, int) else (200, body)

        def raise_for_status():
            if status >= 400:
                raise requests.HTTPError(f"{status} for {rel}", response=SimpleNamespace(status_code=status, headers={}))

        return SimpleNamespace(status_code=status, text=text, content=text.encode(), headers={},
                               raise_for_status=raise_for_status, json=lambda: json.loads(text))


@pytest.fixture
def njm(monkeypatch):
    """Build an available NJMFilterGrabber that talks to a :class:`Served` session."""
    from sed_tools.njm_filter_grabber import NJMFilterGrabber

    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.time.sleep", lambda seconds: None)

    def make(root, files):
        grabber = NJMFilterGrabber(base_dir=str(root))
        grabber.session = Served(files)
        return grabber

    return make


def test_unreachable_mirror_is_reported_with_its_reason(tmp_path, caplog):
    import requests
    from types import SimpleNamespace

    from sed_tools.njm_filter_grabber import NJMFilterGrabber

    def head(url, **kwargs):
        raise requests.ConnectionError("no route to host")

    grabber = NJMFilterGrabber.__new__(NJMFilterGrabber)
    grabber.base_url, grabber.session = NJM, SimpleNamespace(head=head)
    with caplog.at_level(logging.WARNING):
        assert grabber._check_availability() is False
    assert f"NJM mirror {NJM} is unavailable (no route to host)" in caplog.text


def test_directory_listing_is_empty_only_when_nothing_is_there(tmp_path, njm):
    import requests

    grabber = njm(tmp_path, {"filters/Fac/": '<a href="Cam/">Cam/</a>', "filters/Down/": 500})
    assert grabber.discover_instruments("Fac") == ["Cam"]
    assert grabber.discover_instruments("Nope") == []
    with pytest.raises(requests.HTTPError, match="500"):
        grabber.discover_instruments("Down")


def test_unreadable_index_falls_back_to_the_listing_with_a_warning(tmp_path, njm, caplog):
    grabber = njm(tmp_path, {"index.json": "{broken", "filters": '<a href="Fac/">Fac/</a>',
                             "filters/Fac/Cam/Cam": 500, "filters/Fac/Cam/": '<a href="g.dat">g.dat</a>'})
    with caplog.at_level(logging.WARNING):
        assert grabber.discover_facilities() == ["Fac"]
        assert grabber.discover_filters("Fac", "Cam") == ["g"]
    assert "Cannot read NJM's index.json" in caplog.text
    assert "Cannot read NJM's index of Fac/Cam (500 for filters/Fac/Cam/Cam); listing its directory instead" in caplog.text


def test_catalogue_files_skip_unserved_and_report_unreadable(tmp_path, njm, caplog):
    grabber = njm(tmp_path, {"filters/Fac/Cam/info.json": "{not json"})
    with caplog.at_level(logging.WARNING):
        assert grabber.download_catalogue_files("Fac", "Cam") == []
    assert "info.json for Fac/Cam was not saved" in caplog.text
    assert "properties.json" not in caplog.text  # not served (404): skipped quietly


def test_manifest_is_none_only_when_not_served(tmp_path, njm):
    import json

    import requests

    manifest = {"facility": "Fac", "instrument": "Cam", "bands": ["g"]}
    grabber = njm(tmp_path, {"filters/Fac/Cam/manifest.json": json.dumps(manifest),
                             "filters/Fac/Other/manifest.json": json.dumps(manifest),
                             "filters/Fac/Bad/manifest.json": '{"facility": "Fac"}',
                             "filters/Fac/Down/manifest.json": 503})
    assert grabber.fetch_manifest("Fac", "Cam") == manifest
    assert grabber.fetch_manifest("Fac", "Nope") is None
    with pytest.raises(ValueError, match="describes Fac/Cam"):
        grabber.fetch_manifest("Fac", "Other")
    with pytest.raises(ValueError, match="unreadable manifest.json for Fac/Bad"):
        grabber.fetch_manifest("Fac", "Bad")
    with pytest.raises(requests.HTTPError, match="503"):
        grabber.fetch_manifest("Fac", "Down")


def test_component_download_saves_what_it_can_then_lists_the_failures(tmp_path, njm):
    import json

    manifest = {"facility": "Fac", "instrument": "Cam", "bands": ["g"],
                "components": {"filter": {"g": "components/filter/g.dat"}, "qe": {"*": "components/qe/ccd.dat"}}}
    grabber = njm(tmp_path, {"filters/Fac/Cam/manifest.json": json.dumps(manifest),
                             "filters/Fac/Cam/components/filter/g.dat": CURVE,
                             "filters/Fac/Cam/components/qe/ccd.dat": 500})
    with pytest.raises(RuntimeError, match="1 component files of Fac/Cam were not downloaded:\n  components/qe/ccd.dat"):
        grabber.download_components("Fac", "Cam")
    assert (tmp_path / "Fac/Cam/components/filter/g.dat").is_file()


def test_atmosphere_is_false_only_when_not_served(tmp_path, njm):
    import requests

    grabber = njm(tmp_path, {"atmospheres/down/manifest.json": 500})
    assert grabber.download_atmosphere(str(tmp_path), "nowhere") is False
    with pytest.raises(requests.HTTPError, match="500"):
        grabber.download_atmosphere(str(tmp_path), "down")


def test_download_all_filters_finishes_then_lists_incomplete_instruments(tmp_path, njm, monkeypatch):
    import requests

    grabber = njm(tmp_path, {})
    monkeypatch.setattr(grabber, "discover_facilities", lambda: ["Down", "Fac"])
    attempted = []

    def instruments(facility):
        if facility == "Down":
            raise requests.HTTPError("500 for filters/Down/")
        return ["A", "B", "C"]

    def download(facility, instrument):
        attempted.append(instrument)
        if instrument == "A":
            raise RuntimeError("NJM download of Fac/A is incomplete")
        if instrument == "B":
            raise requests.HTTPError("502 for filters/Fac/B/")
        return 2

    monkeypatch.setattr(grabber, "discover_instruments", instruments)
    monkeypatch.setattr(grabber, "download_filters", download)
    with pytest.raises(RuntimeError, match="3 NJM downloads are incomplete:\nDown: 500 .*\n"
                                           "Fac/A: NJM download of Fac/A is incomplete\nFac/B: 502"):
        grabber.download_all_filters()
    assert attempted == ["A", "B", "C"]


def test_svo_failure_while_checking_a_mirror_curve_keeps_the_curve(tmp_path, njm, monkeypatch, caplog):
    def tag(path, **kwargs):
        raise fm.SVOUnreachableError("SVO Filter Profile Service unreachable")

    monkeypatch.setattr("sed_tools.njm_filter_grabber.tag_filter_file", tag)
    grabber = njm(tmp_path, {"filters/Fac/Cam/Cam": "g.dat\n", "filters/Fac/Cam/g.dat": CURVE})
    with caplog.at_level(logging.WARNING):
        assert grabber.download_filters("Fac", "Cam") == 1
    assert (tmp_path / "Fac/Cam/g.dat").read_text() == CURVE
    assert ("kept the mirror's headers of g.dat, which could not be checked against SVO "
            "(SVO Filter Profile Service unreachable)") in caplog.text


# ----------------------------------------------------------------------
# filter_library with NJM failures
# ----------------------------------------------------------------------
def test_njm_components_failure_does_not_discard_complete_curves(tmp_path, njm, caplog):
    import json

    from sed_tools.filter_library import FilterLibrary

    manifest = {"facility": "Fac", "instrument": "Cam", "bands": ["g"],
                "components": {"qe": {"*": "components/qe/ccd.dat"}}}
    library = FilterLibrary(tmp_path)
    library.mirror = njm(tmp_path, {"filters/Fac/Cam/Cam": "g.dat\n", "filters/Fac/Cam/g.dat": CURVE,
                                    "filters/Fac/Cam/manifest.json": json.dumps(manifest),
                                    "filters/Fac/Cam/components/qe/ccd.dat": 500})
    with caplog.at_level(logging.WARNING):
        assert library.published("Fac", "Cam") == tmp_path / "Fac" / "Cam"
    assert "NJM provided every curve of Fac/Cam, but" in caplog.text and "components/qe/ccd.dat" in caplog.text


def test_availability_and_catalogues_record_njm_failures(tmp_path):
    import requests
    from types import SimpleNamespace

    from sed_tools.filter_library import FilterLibrary
    from sed_tools.svo_filter_grabber import Facility

    def down(*args):
        raise requests.ConnectionError("NJM timed out")

    library = FilterLibrary(tmp_path)
    library.svo = SimpleNamespace(list_facilities=lambda: [Facility("SvoOnly", "SvoOnly")],
                                  list_instruments=lambda key: [])
    library.mirror = SimpleNamespace(is_available=lambda: True, fetch_manifest=down, discover_facilities=down)
    info = library.available("Fac", "Cam")
    assert info["availability_source"] == "unknown" and "NJM: NJM timed out" in info["errors"]
    assert "NJM: NJM timed out" in library.facilities()["errors"]
    assert "NJM SvoOnly: NJM timed out" in library.instruments("SvoOnly")["errors"]

    library.mirror = SimpleNamespace(is_available=lambda: False)
    assert "NJM unavailable" in library.instruments("SvoOnly")["errors"]


def test_njm_component_failure_is_reported_before_the_original_source(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace

    from sed_tools import filter_library
    from sed_tools.component_sources import catalog
    from sed_tools.filter_library import FilterLibrary

    directory = tmp_path / "Fac" / "Cam"
    directory.mkdir(parents=True)
    fm.write_filter_file(directory / "g.dat", [4000.0, 6000.0], [1.0, 1.0], detector_type=fm.PHOTON,
                         source="svo", wavelength_unit="Angstrom", extra={"Components": "filter+optics+qe"})

    def fail(facility, instrument):
        raise RuntimeError("1 component files of Fac/Cam were not downloaded")

    library = FilterLibrary(tmp_path)
    library.mirror = SimpleNamespace(is_available=lambda: True, download_components=fail)
    monkeypatch.setitem(catalog.BUILDERS_FOR, ("Fac", "Cam"), catalog.BuilderRef("test"))
    monkeypatch.setattr(filter_library, "get_builder", lambda name: lambda root, **kwargs: None)
    with caplog.at_level(logging.WARNING):
        assert not library.prepare_components(directory, components="filter+qe")
    assert ("NJM cannot provide the components of Fac/Cam (1 component files of Fac/Cam were not downloaded); "
            "trying their original source") in caplog.text


def test_njm_atmosphere_failure_is_reported_before_its_source(tmp_path, monkeypatch, caplog):
    import requests
    from types import SimpleNamespace

    from sed_tools import filter_library
    from sed_tools.filter_components import AtmosphereGrid, AtmosphereSpec, InstrumentManifest
    from sed_tools.filter_library import FilterLibrary

    def down(*args):
        raise requests.HTTPError("502 for atmospheres/paranal/manifest.json")

    library = FilterLibrary(tmp_path / "filters")
    library.mirror = SimpleNamespace(download_atmosphere=down)

    def skycalc(root, **kwargs):
        site = library.atmosphere_root / "paranal"
        site.mkdir(parents=True)
        fm.write_filter_file(site / "a.dat", [3000.0, 7000.0], [0.8, 0.8], detector_type="energy",
                             source="component", wavelength_unit="Angstrom", extra={"Components": "atmosphere"})
        AtmosphereGrid("paranal", [{"airmass": 1.2, "pwv": 2.5, "file": "a.dat"}]).save(site)

    monkeypatch.setattr(filter_library, "get_builder", lambda name: skycalc)
    manifest = InstrumentManifest("Paranal", "Cam", ["g"], ground_based=True, default_site="paranal")
    with caplog.at_level(logging.WARNING):
        path = library.prepare_atmosphere(manifest, AtmosphereSpec(None, 1.2, 2.5))
    assert path.name == "a.dat"
    assert "NJM cannot provide the paranal_am1.2_pwv2.5 atmosphere (502" in caplog.text


# ----------------------------------------------------------------------
# filter_metadata: SVO failures are no longer taken for "SVO has no answer"
# ----------------------------------------------------------------------
class SVOClient:
    """Offline stand-in for SVOMetadataClient: ``fail`` maps filter IDs to the exception they raise."""

    def __init__(self, records=None, fail=None, listing=()):
        self.records, self.fail, self.listing = records or {}, fail or {}, list(listing)

    def get_filter(self, filter_id):
        if filter_id in self.fail:
            raise self.fail[filter_id]
        return self.records.get(filter_id)

    def facility_filter_ids(self, facility):
        return self.listing


class SVOSession:
    """A requests session for SVOMetadataClient: every answer is ``content`` with ``status``."""

    def __init__(self, content=b"", status=200, error=None):
        self.content, self.status, self.error, self.headers = content, status, error, {}

    def get(self, url, **kwargs):
        import requests
        from types import SimpleNamespace

        if self.error is not None:
            raise self.error

        def raise_for_status():
            if self.status >= 400:
                raise requests.HTTPError(f"{self.status} Server Error")

        return SimpleNamespace(content=self.content, status_code=self.status, raise_for_status=raise_for_status)


def test_svo_error_is_raised_and_nothing_is_written(tmp_path):
    import requests

    path = _curve(tmp_path / "GAIA" / "GAIA" / "G.dat", header="Wavelength,Transmission,FilterID=GAIA/GAIA3.G")
    before = path.read_bytes()
    client = SVOClient(fail={"GAIA/GAIA3.G": requests.HTTPError("500 Server Error")})
    with pytest.raises(requests.HTTPError, match="500"):
        fm.tag_filter_file(path, client=client)
    assert path.read_bytes() == before  # no "energy, assumed" placeholder


def test_svo_timeout_during_curve_matching_is_raised(tmp_path):
    import requests

    path = _curve(tmp_path / "GAIA" / "GAIA" / "G.dat", header="Wavelength,Transmission")
    before = path.read_bytes()
    client = fm.SVOMetadataClient(session=SVOSession(error=requests.ReadTimeout("read timed out")))
    with pytest.raises(requests.ReadTimeout):
        fm.tag_filter_file(path, client=client)
    assert path.read_bytes() == before


def test_unreadable_svo_answer_is_raised(tmp_path):
    client = fm.SVOMetadataClient(session=SVOSession(content=b"<html><body>Service down</body></html>"))
    with pytest.raises(ValueError, match="VOTABLE"):
        client.facility_filter_ids("GAIA")
    with pytest.raises(ValueError, match="VOTABLE"):
        client.get_filter("GAIA/GAIA3.G")


def test_failed_facility_listing_is_raised_and_not_cached_as_empty():
    import requests

    session = SVOSession(status=500)
    client = fm.SVOMetadataClient(session=session)
    with pytest.raises(requests.HTTPError):
        client.facility_filter_ids("GAIA")
    session.status, session.content = 200, (
        b'<?xml version="1.0"?><VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
        b'<RESOURCE><INFO name="QUERY_STATUS" value="ERROR">none</INFO></RESOURCE></VOTABLE>')
    assert client.facility_filter_ids("GAIA") == []


def test_tag_tree_tags_what_it_can_then_lists_the_failures(tmp_path):
    import numpy as np
    import requests

    instrument = tmp_path / "Fac" / "Cam"
    for band in "ab":
        _curve(instrument / f"{band}.dat", header=f"Wavelength,Transmission,FilterID=Fac/Cam.{band}")
    before = (instrument / "a.dat").read_bytes()
    record = fm.SVOFilterRecord("Fac/Cam.b", fm.PHOTON, np.array([4000.0, 5000.0, 6000.0]),
                                np.array([0.1, 1.0, 0.1]), "Angstrom", "Filter")
    client = SVOClient({"Fac/Cam.b": record}, fail={"Fac/Cam.a": requests.HTTPError("502 Bad Gateway")})
    with pytest.raises(RuntimeError, match=r"(?s)1 of 2 filter files could not be tagged.*a\.dat: 502 Bad Gateway"):
        fm.tag_filter_tree(tmp_path, client=client)
    assert (instrument / "a.dat").read_bytes() == before
    assert fm.read_filter_header(instrument / "b.dat")["DetectorType"] == fm.PHOTON


def test_tag_tree_stops_at_once_when_svo_is_unreachable(tmp_path):
    for band in "ab":
        _curve(tmp_path / "Fac" / "Cam" / f"{band}.dat", header=f"Wavelength,Transmission,FilterID=Fac/Cam.{band}")
    down = fm.SVOUnreachableError("SVO Filter Profile Service unreachable")
    client = SVOClient(fail={"Fac/Cam.a": down, "Fac/Cam.b": down})
    with pytest.raises(fm.SVOUnreachableError):
        fm.tag_filter_tree(tmp_path, client=client)


def test_unverified_ssl_fallback_is_a_warning(caplog):
    import requests
    from types import SimpleNamespace

    from sed_tools.njm_filter_grabber import NJMFilterGrabber

    def head(url, timeout=None, verify=True):
        if verify:
            raise requests.exceptions.SSLError("certificate expired")
        return SimpleNamespace(status_code=200)

    grabber = NJMFilterGrabber.__new__(NJMFilterGrabber)
    grabber.base_url, grabber.session = NJM, SimpleNamespace(head=head, verify=True)
    with caplog.at_level(logging.WARNING):
        assert grabber._check_availability() is True
    assert grabber.session.verify is False
    assert "NJM's SSL certificate could not be verified" in caplog.text


def test_unreadable_served_catalogue_file_is_refused(tmp_path):
    from types import SimpleNamespace

    from sed_tools.filter_catalogue import CatalogueError, FilterCatalogue

    mirror = SimpleNamespace(base_url=NJM, is_available=lambda: True,
                             _get=lambda url, **kwargs: SimpleNamespace(status_code=200, content=b"\xff{",
                                                                        headers={}))
    catalogue = FilterCatalogue(tmp_path / "filters", mirror=mirror)
    with pytest.raises(CatalogueError, match="NJM served an unreadable index.json"):
        catalogue.index()


def test_published_carries_every_source_reason(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace

    from sed_tools import filter_library
    from sed_tools.component_sources import catalog
    from sed_tools.filter_components import ComponentError
    from sed_tools.filter_library import FilterLibrary
    from sed_tools.svo_filter_grabber import Facility, FilterRow, Instrument

    def njm_download(facility, instrument):
        raise RuntimeError("NJM download of Fac/Cam is incomplete")

    def svo_download(rows):
        raise fm.SVOUnreachableError("SVO Filter Profile Service unreachable")

    library = FilterLibrary(tmp_path)
    library.mirror = SimpleNamespace(is_available=lambda: True, _download_set=njm_download)
    library.svo = SimpleNamespace(list_facilities=lambda: [Facility("Fac", "Fac")],
                                  list_instruments=lambda key: [Instrument("Fac", "Cam", "Cam")],
                                  list_filters=lambda fac, inst: [FilterRow("Fac/Cam.g", "g", "Fac", "Cam", "")],
                                  download_filters=svo_download)
    monkeypatch.setitem(catalog.BUILDERS_FOR, ("Fac", "Cam"), catalog.BuilderRef("test", publishes=True))
    monkeypatch.setattr(filter_library, "get_builder", lambda name: lambda root, **kwargs: None)
    with caplog.at_level(logging.WARNING), pytest.raises(ComponentError) as info:
        library.published("Fac", "Cam")
    assert str(info.value) == ("No source can provide Fac/Cam: NJM: NJM download of Fac/Cam is incomplete; "
                               "original source: the builder left an incomplete set; "
                               "SVO: SVO Filter Profile Service unreachable")
    assert "The original source cannot provide Fac/Cam (the builder left an incomplete set); trying SVO" in caplog.text


def test_svo_error_on_a_listed_candidate_is_raised(tmp_path):
    import requests

    path = _curve(tmp_path / "GAIA" / "GAIA" / "G.dat", header="Wavelength,Transmission")
    client = SVOClient(listing=["GAIA/GAIA3.G"], fail={"GAIA/GAIA3.G": requests.HTTPError("503 Service Unavailable")})
    with pytest.raises(requests.HTTPError, match="503"):
        fm.resolve_detector_type(path, client=client)
