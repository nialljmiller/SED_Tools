"""The filter library and NJM's mirror: outages checked once per process,
downloads that fetch only what changed, old instrument spellings, NJM's archived copies, compose, and
where the catalogue comes from.

No test uses the network: NJM, SVO and the original sources are stand-ins.
"""

from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace

import pytest
import requests

from sed_tools import cli, filter_library, njm_filter_grabber
from sed_tools.api import Filters  # noqa: E402
from sed_tools.combine_filters import write_filter_index  # noqa: E402
from sed_tools.component_sources.catalog import name_key  # noqa: E402
from sed_tools.filter_components import ComponentError, InstrumentManifest
from sed_tools.filter_library import FilterLibrary, instrument_catalogue, load_instrument_catalogue  # noqa: E402
from sed_tools.njm_filter_grabber import NJMFilterGrabber

from helpers_filters import NJM


# ----------------------------------------------------------------------
# NJM's availability and pause are shared by the whole process
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
    assert session.calls == [("HEAD", "https://data.nialljmiller.com/sed_tools", True)]  # one 5 s wait, not three
    assert caplog.text.count("is unavailable (timed out)") == 1
    with pytest.raises(requests.ConnectionError, match="not asked again for 5 minutes"):
        grabbers[2]._get("https://data.nialljmiller.com/sed_tools/index.json")
    assert len(session.calls) == 1


def test_the_outage_is_checked_again_once_the_pause_expires(tmp_path, monkeypatch, caplog):
    clock = [1000.0]
    monkeypatch.setattr("sed_tools.njm_mirror.monotonic", lambda: clock[0])
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


def test_an_error_answer_to_one_catalogue_is_not_an_outage_for_the_next(tmp_path):
    # Only the NJM client pauses NJM (no answer, or a block that does not lift); an error answer does not.
    from sed_tools.filter_catalogue import CatalogueError, FilterCatalogue

    asked = []

    def get(url, **kwargs):
        asked.append(url)
        raise requests.HTTPError("503", response=SimpleNamespace(status_code=503))

    mirror = SimpleNamespace(base_url="https://njm.invalid/sed_tools", is_available=lambda: True, _get=get)
    for _ in range(2):
        with pytest.raises(CatalogueError, match="Cannot fetch index.json"):
            FilterCatalogue(tmp_path / "filters", mirror=mirror).index()
    assert len(asked) == 2


# ----------------------------------------------------------------------
# Installed curves are checked against NJM only when there is a reason
# ----------------------------------------------------------------------
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


def test_a_retry_after_a_partial_download_fetches_only_what_is_missing(grabber):
    partial = _njm_set()
    partial["filters/Fac/Cam/r.dat"] = 404
    partial["filters/Fac/Cam/Cam"] = "g.dat\n#g_old.dat\nr.dat\n"  # a '#' line of NJM's index is not a band
    first = grabber(partial)
    assert first._download_set("Fac", "Cam").curve_failures == ["r.dat: 404 for filters/Fac/Cam/r.dat"]
    assert InstrumentManifest.load(f"{first.base_dir}/Fac/Cam").bands == ["g", "r"]
    retry = grabber(_njm_set())
    assert retry._download_set("Fac", "Cam").downloaded == 1
    assert retry.session.curves_requested() == ["r.dat"]


# ----------------------------------------------------------------------
# Every old spelling still works as input to the API and the CLI
# ----------------------------------------------------------------------
#: (spelling given, NJM's spelling): SVO's, 0.4.1's and NJM's own.
LEGACY = [
    (("PLANCK", "HFI"), ("Planck", "HFI")),
    (("Planck", "HFI"), ("Planck", "HFI")),
    (("Gemini", "GMOS-N"), ("Gemini", "GMOS_N")),
    (("Gemini", "GMOS_N"), ("Gemini", "GMOS_N")),
    (("Herschel", "Pacs"), ("Herschel", "PACS")),
    (("CTIO", "S-PLUS"), ("CTIO", "S_PLUS")),
    (("Bepi-Colombo", "HRIC"), ("Bepi_Colombo", "HRIC")),
]
FILTER_CURVE = ("Wavelength,Transmission,FilterID=X.b;DetectorType=photon;DetectorTypeSource=original;"
                "WavelengthUnit=Angstrom;Components=filter;ComponentsSource=original;Sources=https://example.invalid\n"
                "4000,0\n5000,1\n6000,0\n")


def _legacy_njm_files():
    detailed, files = {}, {}
    for _, (fac, inst) in LEGACY:
        detailed.setdefault(fac, {"instruments": {}})["instruments"][inst] = {
            "filters": ["b"], "size_bytes": 200, "info": "info.json", "properties": "properties.json",
            "description": f"{fac} {inst} passbands", "svo_instruments": []}
        files[f"filters/{fac}/{inst}/{inst}"] = "b.dat\n"
        files[f"filters/{fac}/{inst}/b.dat"] = FILTER_CURVE
        files[f"filters/{fac}/{inst}/info.json"] = json.dumps(
            {"description": f"{fac} {inst} passbands", "bands": {"b": {"Description": "band b"}}})
        files[f"filters/{fac}/{inst}/properties.json"] = json.dumps({"bands": {"b": {"WavelengthPivot": 5000.0}}})
    files["index.json"] = json.dumps({"generated": "2026-10-28", "filters": {
        "facilities": sorted(detailed), "facilities_detailed": detailed}})
    return files, detailed


@pytest.fixture
def legacy_sources(tmp_path, monkeypatch):
    """NJM serves the instruments under its own spelling; SVO is down; nothing is built from source."""
    files, detailed = _legacy_njm_files()
    served = Served(files)
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    monkeypatch.setattr(njm_filter_grabber.requests, "Session", lambda: served)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.time.sleep", lambda seconds: None)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.tag_filter_file", lambda *a, **k: None)

    def svo_down(*args, **kwargs):
        raise RuntimeError("SVO is down (test)")

    monkeypatch.setattr(filter_library, "SVOFilterBrowser", lambda **kwargs: SimpleNamespace(
        list_facilities=svo_down, list_instruments=svo_down, list_filters=svo_down, download_filters=svo_down))
    monkeypatch.setattr(filter_library, "get_builder", lambda name: pytest.fail(f"builder {name} was run"))
    monkeypatch.setattr(filter_library, "load_instrument_catalogue", lambda: instrument_catalogue(detailed))
    monkeypatch.setattr(Filters, "_catalogues", {})
    monkeypatch.setenv("COLUMNS", "100")
    return SimpleNamespace(root=tmp_path / "filters", served=served)


def _njm_curve_requests(served, njm):
    return [rel for rel in served.requested if rel.endswith("/b.dat")], f"filters/{njm[0]}/{njm[1]}/b.dat"


def test_the_shipped_catalogue_names_each_instrument_as_njm_does():
    shipped = load_instrument_catalogue()
    for _, (fac, inst) in LEGACY:
        assert inst in shipped[fac]["instruments"], f"{fac}/{inst}"


@pytest.mark.parametrize("given,njm", LEGACY, ids=["/".join(g) for g, _ in LEGACY])
def test_fetch_info_and_compose_take_the_old_spelling(legacy_sources, given, njm):
    directory = Filters.fetch(*given, filter_root=legacy_sources.root)
    # macOS folds case, so PLANCK/HFI and Planck/HFI may be one directory: compare names, not paths.
    assert (name_key(directory.parent.name), name_key(directory.name)) == (name_key(njm[0]), name_key(njm[1]))
    assert FilterLibrary.published_complete(directory)
    curves, njm_curve = _njm_curve_requests(legacy_sources.served, njm)
    assert curves == [njm_curve]  # downloaded once, under NJM's spelling

    assert Filters.info(*given, filter_root=legacy_sources.root)["bands"]["b"]["Description"] == "band b"
    product = Filters.compose("/".join(given), "filter", filter_root=legacy_sources.root)
    assert (product / "b.dat").is_file()
    assert _njm_curve_requests(legacy_sources.served, njm)[0] == [njm_curve]  # nothing downloaded again


@pytest.mark.parametrize("given,njm", LEGACY, ids=["/".join(g) for g, _ in LEGACY])
def test_the_cli_takes_the_old_spelling(legacy_sources, monkeypatch, capsys, given, njm):
    target, base = "/".join(given), str(legacy_sources.root)
    for argv in (["filters-info", target], ["filters", target], ["filters-info", target],
                 ["filters-compose", target]):
        monkeypatch.setattr(sys, "argv", ["sed-tools", *argv, "--base", base])
        assert cli.main() == 0, (argv, capsys.readouterr())
    out = capsys.readouterr().out
    assert "products/filter" in out
    assert _njm_curve_requests(legacy_sources.served, njm)[0] == [f"filters/{njm[0]}/{njm[1]}/b.dat"]


def test_a_set_installed_under_an_old_spelling_is_used_as_it_is(legacy_sources):
    # 0.4.1 saved SVO's Gemini/GMOS-N; NJM's spelling is GMOS_N. The installed set is used, offline.
    old = legacy_sources.root / "Gemini" / "GMOS-N"
    old.mkdir(parents=True)
    (old / "b.dat").write_text(FILTER_CURVE)
    (old / "GMOS-N").write_text("b.dat\n")
    filter_library.InstrumentManifest.from_published(old).save(old)
    assert Filters.fetch("Gemini", "GMOS-N", filter_root=legacy_sources.root) == old
    assert (Filters.compose("Gemini/GMOS-N", "filter", filter_root=legacy_sources.root) / "b.dat").is_file()
    assert legacy_sources.served.requested == []


def test_query_keeps_svo_spelling_beside_the_download_key(tmp_path, monkeypatch, legacy_sources):
    from sed_tools.svo_filter_grabber import Facility

    monkeypatch.setattr(filter_library, "SVOFilterBrowser", lambda **kwargs: SimpleNamespace(
        list_facilities=lambda: [Facility("PLANCK", "PLANCK", "Planck mission")]))
    monkeypatch.setattr(Filters, "_filter_root", legacy_sources.root)
    for asked in ("PLANCK", "planck", "Planck"):
        rows = [row for row in Filters.query(asked, include_local=False) if row["source"] == "svo"]
        assert [(r["facility_key"], r["svo_facility_key"]) for r in rows] == [("Planck", "PLANCK")], asked


# ----------------------------------------------------------------------
# Archive copies stay in archived_bands
# ----------------------------------------------------------------------
ARCHIVED = {"H_svo": {"detector_type": "photon", "detector_type_source": "svo", "published": "filter"}}


def _nisp_like(tmp_path, *, index=True):
    """A set as NJM serves Euclid/NISP: H and Y, plus SVO's H_svo kept under archived_bands."""
    inst = tmp_path / "Euclid" / "NISP"
    inst.mkdir(parents=True)
    for band in ("H", "Y", "H_svo"):
        (inst / f"{band}.dat").write_text(FILTER_CURVE)
    if index:
        (inst / "NISP").write_text("H.dat\nY.dat\n")
    (inst / "manifest.json").write_text(json.dumps(
        {"facility": "Euclid", "instrument": "NISP", "bands": ["H", "Y"], "archived_bands": ARCHIVED}))
    return inst


def test_a_manifest_keeps_archived_bands_when_saved(tmp_path):
    inst = _nisp_like(tmp_path)
    manifest = InstrumentManifest.load(inst)  # what jwst.py and roman.py do: load, change, save
    assert manifest.archived_bands == ARCHIVED and manifest.bands == ["H", "Y"]
    manifest.notes = "rebuilt"
    manifest.save(inst)
    assert json.loads((inst / "manifest.json").read_text())["archived_bands"] == ARCHIVED
    InstrumentManifest("Fac", "Cam", ["g"]).save(tmp_path)
    assert "archived_bands" not in json.loads((tmp_path / "manifest.json").read_text())  # unchanged format


def test_described_from_its_curves_a_set_leaves_archive_copies_out(tmp_path):
    inst = _nisp_like(tmp_path)
    described = InstrumentManifest.from_published(inst)  # what euclid, hsc, gmos and _common do
    assert described.bands == ["H", "Y"] and described.archived_bands == ARCHIVED
    (inst / "NISP").unlink()
    assert InstrumentManifest.from_published(inst).bands == ["H", "Y"]  # no index file: the manifest decides


def test_only_index_listed_curves_are_bands_and_the_index_never_lists_an_archive_copy(tmp_path):
    inst = _nisp_like(tmp_path)
    (inst / "stray.dat").write_text(FILTER_CURVE)  # not in the MESA index: MESA never loads it
    assert InstrumentManifest.from_published(inst).bands == ["H", "Y"]
    write_filter_index(inst, "NISP")  # a builder rewriting the index (hsc, lsst, argus, _common)
    assert (inst / "NISP").read_text().split() == ["H.dat", "Y.dat", "stray.dat"]


def test_dropping_bands_njm_does_not_list_keeps_archived_bands(tmp_path):
    inst = _nisp_like(tmp_path)
    data = json.loads((inst / "manifest.json").read_text())
    data["bands"] = ["H", "J", "Y"]  # J: declared, but NJM's index does not list it and it is not installed
    (inst / "manifest.json").write_text(json.dumps(data))
    filter_library._drop_unlisted_bands(inst, ["H.dat", "Y.dat"])
    saved = json.loads((inst / "manifest.json").read_text())
    assert saved["bands"] == ["H", "Y"] and saved["archived_bands"] == ARCHIVED
    assert FilterLibrary.published_complete(inst)


# ----------------------------------------------------------------------
# Compose refuses bands without the components before any download
# ----------------------------------------------------------------------
def _nisp_with_declared_components(tmp_path):
    """Euclid/NISP as NJM's manifest declares it: filter, optics and qe for H and Y; grisms Blue and Red bare."""
    inst = tmp_path / "filters" / "Euclid" / "NISP"
    inst.mkdir(parents=True)
    for band in ("Blue", "H", "Red", "Y"):
        (inst / f"{band}.dat").write_text(FILTER_CURVE.replace("Components=filter", "Components=filter+optics+qe"))
    (inst / "NISP").write_text("Blue.dat\nH.dat\nRed.dat\nY.dat\n")
    components = {role: {band: f"components/{role}/{band}.dat" for band in ("H", "Y")}
                  for role in ("filter", "optics", "qe")}  # declared; not downloaded yet
    InstrumentManifest("Euclid", "NISP", ["Blue", "H", "Red", "Y"], components=components).save(inst)
    return inst


def test_compose_names_the_bands_without_components_before_asking_any_source(tmp_path):
    _nisp_with_declared_components(tmp_path)
    library = FilterLibrary(tmp_path / "filters")
    library.mirror = SimpleNamespace(is_available=lambda: pytest.fail("NJM was asked"))
    with pytest.raises(ComponentError, match=r"Euclid/NISP has no filter\+optics\+qe components for Blue, Red; "
                                             r"they exist for H, Y\. Compose only those bands"):
        library.compose("Euclid/NISP", "filter,optics,qe")


def test_compose_of_the_covered_bands_goes_on_to_the_sources(tmp_path):
    _nisp_with_declared_components(tmp_path)
    library = FilterLibrary(tmp_path / "filters")
    asked = []
    library.prepare_components = lambda directory, **kwargs: asked.append(kwargs["bands"]) or False
    with pytest.raises(ComponentError):  # nothing downloaded in this test, so the build itself fails
        library.compose("Euclid/NISP", "filter,optics,qe", bands=["H", "Y"])
    assert asked == [["H", "Y"]]


# ----------------------------------------------------------------------
# available() says which source failures were only fallbacks
# ----------------------------------------------------------------------
def test_available_reports_failures_by_source_and_which_were_fallbacks(tmp_path, monkeypatch):
    from sed_tools.svo_filter_grabber import FilterRow

    library = FilterLibrary(tmp_path / "filters")
    library.mirror = SimpleNamespace(is_available=lambda: True, fetch_manifest=lambda f, i: (_ for _ in ()).throw(
        requests.ConnectionError("Connection refused")))
    monkeypatch.setattr(library, "_njm_spelling", lambda f, i: None)
    monkeypatch.setattr(library, "svo_filters", lambda f, i: [FilterRow("Fac/Cam.g", "g", "Fac", "Cam", ""),
                                                             FilterRow("Fac/Cam.r", "r", "Fac", "Cam", "")])
    library.svo = SimpleNamespace(metadata_client=SimpleNamespace(
        get_filter=lambda fid: None if fid.endswith(".r") else SimpleNamespace(components=None, detector_type="photon")))
    info = library.available("Fac", "Cam")
    assert info["availability_source"] == "svo"
    assert info["errors"] == ["NJM: Connection refused", "SVO: no metadata for Fac/Cam.r"]  # as before
    assert info["sources_failed"] == {"njm": ["Connection refused"], "svo": ["no metadata for Fac/Cam.r"]}
    assert info["fallback_from"] == ["njm"]
    assert cli._availability_errors(info) == ["no metadata for Fac/Cam.r"]  # NJM's failure left nothing out


def test_compose_leaves_variants_to_the_sources_that_register_them(tmp_path):
    # A set whose manifest has no variants yet (from SVO, or installed by 0.4.1): the builder or
    # NJM's manifest adds the variant while the components are prepared, so it is not refused first.
    inst = _nisp_with_declared_components(tmp_path)
    library = FilterLibrary(tmp_path / "filters")

    def prepare(directory, **kwargs):
        manifest = InstrumentManifest.load(inst)
        manifest.variants = {"qe": {"cold": {"files": {"*": "components/qe/cold.dat"}}}}
        manifest.save(inst)
        raise ComponentError("stop here: the variant was registered")

    library.prepare_components = prepare
    with pytest.raises(ComponentError, match="the variant was registered"):
        library.compose("Euclid/NISP", "filter,optics,qe", bands=["H", "Y"], variants={"qe": "cold"})


def test_no_index_json_is_fetched_for_the_record_while_njm_is_paused(tmp_path, grabber, caplog):
    made = grabber(_njm_set())
    njm_filter_grabber.NJMFilterGrabber._pause("test outage", "NJM is down (test)")
    with caplog.at_level(logging.WARNING):
        assert made._njm_record("Fac", "Cam") is None
    assert made.session.requested == [] and "Cannot read NJM's index.json" not in caplog.text


# ----------------------------------------------------------------------
# filters-info before anything is installed and with no source answering:
# the detector is not known, never the energy default
# ----------------------------------------------------------------------
@pytest.fixture
def bundled_dr2(monkeypatch):
    from sed_tools import filter_library
    from sed_tools.filter_library import instrument_catalogue

    gaia = {"DR2": {"svo_instruments": ["GAIA/GAIA2"], "bands": ["BP", "G", "RP"],
                    "svo_bands": {"GAIA/GAIA2": ["G", "Gbp=BP", "Grp=RP"]}}}
    monkeypatch.setattr(filter_library, "load_instrument_catalogue",
                        lambda: instrument_catalogue({"GAIA": {"instruments": gaia}}))


@pytest.mark.parametrize("builder", [True, False])
def test_bundled_availability_leaves_the_detector_unknown(builder, tmp_path, bundled_dr2, monkeypatch, capsys):
    from types import SimpleNamespace

    from sed_tools import filter_library
    from sed_tools.cli import print_filter_availability
    from sed_tools.filter_library import FilterLibrary
    from helpers_filters import svo_stub

    if not builder:
        monkeypatch.delitem(filter_library.BUILDERS_FOR, ("GAIA", "DR2"))
    library = FilterLibrary(tmp_path / "filters")
    library.mirror = SimpleNamespace(is_available=lambda: False, fetch_manifest=lambda f, i: None,
                                     discover_filters=lambda f, i: [])
    library.svo = svo_stub({}, {}, down=True)
    info = library.available("GAIA", "DR2")
    assert info["availability_source"] == ("original" if builder else "unknown")
    # GAIA/DR2's builder writes photon-counting curves; this used to say energy, "unverified"
    assert info["detector_type"] is None
    assert info["detector_type_sources"] == {"BP": "unknown", "G": "unknown", "RP": "unknown"}
    print_filter_availability(info)
    out = capsys.readouterr().out
    assert "  Detector                 : not known until the curves are installed\n" in out
    assert "energy" not in out and "unverified" not in out


# ----------------------------------------------------------------------
# filters-info: NJM's archived copies (*_svo, *_old) stay apart from the set,
# also beside an installed copy whose band names differ from NJM's
# ----------------------------------------------------------------------
HFI_INFO = {"bands": {"100": {"Description": "HFI 100"}, "100_svo": {"Description": "PLANCK HFI 100GH"},
                      "857": {"Description": "HFI 857"}, "857_svo": {"Description": "PLANCK HFI 857GHz"}}}


def _hfi_catalogue(tmp_path, *, njm_index_file=True, offline=False):
    """A catalogue client for NJM's Planck/HFI beside 0.4.1's PLANCK/HFI from SVO (bands 100GHz, 857GHz)."""
    import json
    from types import SimpleNamespace

    from sed_tools.filter_catalogue import FilterCatalogue
    from helpers_filters import Served, njm_index

    installed = tmp_path / "filters" / "PLANCK" / "HFI"
    installed.mkdir(parents=True)
    for band in ("100GHz", "857GHz"):
        (installed / f"{band}.dat").write_text("Wavelength,Transmission\n1,1\n2,1\n")
    (installed / "HFI").write_text("100GHz.dat\n857GHz.dat\n")
    files = {"index.json": njm_index({"Planck": {"HFI": {
                 "filters": ["100", "100_svo", "857", "857_svo"], "properties": "properties.json",
                 "info": "info.json"}}}),
             "filters/Planck/HFI/info.json": json.dumps(HFI_INFO),
             "filters/Planck/HFI/properties.json": json.dumps(HFI_INFO)}
    if njm_index_file:
        files["filters/Planck/HFI/HFI"] = "100.dat\n857.dat\n"
    server = Served(files)
    mirror = SimpleNamespace(base_url="https://data.nialljmiller.com/sed_tools", is_available=lambda: True,
                             _get=lambda url, **kwargs: server.get(url, **kwargs))
    return FilterCatalogue(tmp_path / "filters", mirror=mirror, offline=offline)


@pytest.mark.parametrize("spelling", [("Planck", "HFI"), ("PLANCK", "HFI")])
def test_archived_copies_stay_apart_beside_an_installed_copy_under_other_band_names(spelling, tmp_path):
    catalogue = _hfi_catalogue(tmp_path)
    for kind in ("info", "properties"):
        data = getattr(catalogue, kind)(*spelling)
        assert list(data["bands"]) == ["100", "857"]  # the *_svo rows used to be listed with them
        assert list(data["archived_bands"]) == ["100_svo", "857_svo"]
