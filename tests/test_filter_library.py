"""Offline integration checks for the shared CLI/API optical-path workflow."""

from types import SimpleNamespace

import numpy as np
import pytest

from sed_tools import cli, filter_library
from sed_tools.api import Filters
from sed_tools.component_sources import catalog
from sed_tools.filter_components import (
    AtmosphereGrid, AtmosphereSpec, ComponentError, InstrumentManifest,
    add_component, build_product,
)
from sed_tools.filter_library import FilterLibrary
from sed_tools.filter_metadata import PHOTON, SVOFilterRecord, read_filter_header, write_filter_file
from sed_tools.svo_filter_grabber import Facility, Instrument, FilterRow


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(filter_library.NJMFilterGrabber, "_check_availability", lambda self: False)


@pytest.fixture
def instrument(tmp_path):
    root = tmp_path / "filters"
    directory = root / "Test" / "Camera"
    directory.mkdir(parents=True)
    for band in ("g", "r"):
        write_filter_file(directory / f"{band}.dat", [4000., 5000., 6000.], [0.2, 0.8, 0.2],
                          detector_type=PHOTON, source="svo", wavelength_unit="Angstrom",
                          extra={"Components": "filter+optics+qe"})
    manifest = InstrumentManifest.load(directory)
    manifest.ground_based = True
    for band in ("g", "r"):
        add_component(directory, "filter", [4000., 5000., 6000.], [0.1, 1., 0.1],
                      band=band, manifest=manifest)
    add_component(directory, "qe", [3000., 7000.], [0.8, 0.6], manifest=manifest)
    manifest.save(directory)
    return directory


def test_missing_component_is_not_advertised_as_installed(instrument):
    manifest = InstrumentManifest.load(instrument)
    qe = instrument / manifest.component_file("qe", "g")
    qe.unlink()
    info = Filters.available("Test", "Camera", filter_root=instrument.parent.parent, include_remote=False)
    assert info["components"] == ["filter"]
    assert info["buildable"] == ["filter"]
    assert str(qe.relative_to(instrument)) in info["missing_files"]
    assert not FilterLibrary.components_ready(instrument, ["filter", "qe"])


def test_partial_band_selection_has_cli_api_parity(instrument, monkeypatch):
    manifest = InstrumentManifest.load(instrument)
    (instrument / manifest.component_file("filter", "r")).unlink()
    root = instrument.parent.parent
    info = Filters.available("Test", "Camera", filter_root=root, include_remote=False)
    assert info["buildable"] == []
    assert info["recipes_by_band"]["filter+qe"] == ["g"]
    product = Filters.compose(instrument, "filter+qe", bands=["g"], detector_type="energy", filter_root=root)
    before = (product / "g.dat").read_bytes()
    monkeypatch.setattr("sys.argv", ["sed-tools", "filters-compose", "Test/Camera", "--base", str(root),
                                    "--components", "filter,qe", "--bands", "g", "--detector-type", "energy"])
    assert cli.main() in (0, None)
    assert (product / "g.dat").read_bytes() == before
    assert not (product / "r.dat").exists()
    header = read_filter_header(product / "g.dat")
    assert header["DetectorType"] == "energy"
    assert header["DetectorTypeOriginal"] == "photon"
    assert read_filter_header(instrument / "g.dat")["DetectorType"] == "photon"
    with pytest.raises(ComponentError, match="Missing throughput file"):
        Filters.compose(instrument, "filter+qe", filter_root=root)


def test_failed_rebuild_preserves_existing_product(instrument):
    product = build_product(instrument, "filter+qe")
    original = {p.name: p.read_bytes() for p in product.iterdir()}
    manifest = InstrumentManifest.load(instrument)
    (instrument / manifest.component_file("filter", "r")).unlink()
    with pytest.raises(ComponentError, match="Missing throughput file"):
        build_product(instrument, "filter+qe")
    assert original == {p.name: p.read_bytes() for p in product.iterdir()}


def test_weighting_override_accepts_unknown_published_contents(tmp_path):
    directory = tmp_path / "Mystery" / "Camera"
    directory.mkdir(parents=True)
    write_filter_file(directory / "g.dat", [4000., 6000.], [1., 1.],
                      detector_type=PHOTON, source="svo", wavelength_unit="Angstrom")
    out = Filters.compose(directory, "published", detector_type="energy", filter_root=tmp_path)
    assert read_filter_header(out / "g.dat")["DetectorType"] == "energy"


def test_unresolved_instrument_rejects_extra_detector_or_optics(instrument):
    manifest = InstrumentManifest.load(instrument)
    manifest.published_bands["g"] = "filter+instrument"
    manifest.save(instrument)
    with pytest.raises(ComponentError, match="unresolved"):
        build_product(instrument, "published+qe", bands=["g"])


def test_manifest_role_must_match_component_header(instrument):
    manifest = InstrumentManifest.load(instrument)
    path = instrument / manifest.component_file("filter", "g")
    path.write_text(path.read_text().replace("# Components: filter", "# Components: filter+qe"))
    with pytest.raises(ComponentError, match="disagree"):
        build_product(instrument, "filter", bands=["g"])


def test_mixed_detector_conventions_are_preserved(instrument):
    path = instrument / "r.dat"
    path.write_text(path.read_text().replace("# DetectorType: photon", "# DetectorType: energy"))
    manifest = InstrumentManifest.from_published(instrument)
    manifest.save(instrument)
    assert manifest.describe()["detector_type"] == "mixed"
    product = build_product(instrument, "published")
    assert read_filter_header(product / "r.dat")["DetectorType"] == "energy"
    assert read_filter_header(product / "g.dat")["DetectorType"] == "photon"


def test_missing_manifest_reference_runs_upstream_builder(instrument, monkeypatch):
    manifest = InstrumentManifest.load(instrument)
    path = instrument / manifest.component_file("qe", "g")
    path.unlink()
    monkeypatch.setitem(catalog.BUILDERS_FOR, ("Test", "Camera"), catalog.BuilderRef("test"))
    calls = []

    def build(root):
        calls.append(root)
        add_component(instrument, "qe", [3000., 7000.], [0.8, 0.6], manifest=manifest)
        manifest.save(instrument)

    monkeypatch.setattr(filter_library, "get_builder", lambda name: build)
    Filters.compose(instrument, "filter+qe", filter_root=instrument.parent.parent)
    assert calls == [instrument.parent.parent]
    assert path.exists()


def test_heavy_source_requires_explicit_opt_in(instrument, monkeypatch):
    manifest = InstrumentManifest.load(instrument)
    path = instrument / manifest.component_file("qe", "g")
    path.unlink()
    monkeypatch.setitem(catalog.BUILDERS_FOR, ("Test", "Camera"), catalog.BuilderRef("test", heavy=True))
    calls = []

    def build(root):
        calls.append(root)
        add_component(instrument, "qe", [3000., 7000.], [0.8, 0.6], manifest=manifest)
        manifest.save(instrument)

    monkeypatch.setattr(filter_library, "get_builder", lambda name: build)
    library = FilterLibrary(instrument.parent.parent)
    assert not library.prepare_components(instrument, components="filter+qe")
    assert not calls
    assert library.prepare_components(instrument, components="filter+qe", allow_heavy=True)
    assert len(calls) == 1


def test_builder_programming_errors_reach_the_caller(instrument, monkeypatch):
    manifest = InstrumentManifest.load(instrument)
    (instrument / manifest.component_file("qe", "g")).unlink()
    monkeypatch.setitem(catalog.BUILDERS_FOR, ("Test", "Camera"), catalog.BuilderRef("test"))

    def broken_builder(root):
        raise TypeError("incorrect builder arguments")

    monkeypatch.setattr(filter_library, "get_builder", lambda name: broken_builder)
    with pytest.raises(TypeError, match="incorrect builder arguments"):
        Filters.compose(instrument, "filter+qe", filter_root=instrument.parent.parent)


def test_catalogue_programming_errors_are_not_reported_as_an_outage(tmp_path):
    def broken_catalogue():
        raise AttributeError("incorrect catalogue field")

    library = FilterLibrary(tmp_path)
    library.svo = SimpleNamespace(list_facilities=broken_catalogue)
    with pytest.raises(AttributeError, match="incorrect catalogue field"):
        library.facilities()


@pytest.mark.parametrize("airmass,pwv", [(0.5, 1), (float("nan"), 1), (1.2, -1), (1.2, float("inf"))])
def test_invalid_atmosphere_conditions_are_rejected(airmass, pwv):
    with pytest.raises(ComponentError):
        AtmosphereSpec("paranal", airmass, pwv)


def test_atmosphere_requires_pwv_when_grid_is_ambiguous():
    grid = AtmosphereGrid("paranal", [
        {"airmass": 1.2, "pwv": 1., "file": "a.dat"},
        {"airmass": 1.2, "pwv": 3., "file": "b.dat"},
    ])
    with pytest.raises(ComponentError, match="choose pwv"):
        grid.select(AtmosphereSpec("paranal", 1.2))
    assert grid.select(AtmosphereSpec("paranal", 1.2, 3)) == "b.dat"


def test_skycalc_fallback_requests_only_the_instruments_site(instrument, monkeypatch):
    manifest = InstrumentManifest.load(instrument)
    manifest.default_site = "paranal"
    manifest.save(instrument)
    library = FilterLibrary(instrument.parent.parent)
    calls = []

    def build(root, **kw):
        calls.append(kw)
        directory = library.atmosphere_root / "paranal"
        directory.mkdir(parents=True)
        write_filter_file(directory / "a.dat", [3000., 7000.], [0.8, 0.8], detector_type="energy",
                          source="component", wavelength_unit="Angstrom", extra={"Components": "atmosphere"})
        AtmosphereGrid("paranal", [{"airmass": 1.2, "pwv": 2.5, "file": "a.dat"}]).save(directory)

    monkeypatch.setattr(filter_library, "get_builder", lambda name: build)
    library.compose(instrument, "filter", atmosphere={"airmass": 1.2, "pwv": 2.5})
    assert calls[0]["sites"] == ["paranal"]
    assert calls[0]["airmasses"] == [1.2]
    assert calls[0]["pwvs"] == [2.5]
    library.compose(instrument, "filter", atmosphere={"airmass": 1.2, "pwv": 2.5})
    assert len(calls) == 1  # installed grid needs no source requests


class Catalogue:
    def list_facilities(self):
        return [Facility("NewFacility", "New facility")]

    def list_instruments(self, facility):
        return [Instrument(facility, "NewCamera", "New camera")]

    def list_filters(self, facility, instrument):
        return [FilterRow("NewFacility/NewCamera.g", "g", facility, instrument, "")]


def test_inventory_retains_unreviewed_svo_instruments(tmp_path):
    library = FilterLibrary(tmp_path)
    library.svo = Catalogue()
    result = library.inventory()
    row = next(r for r in result["instruments"] if r["facility"] == "NewFacility")
    assert row["component_availability"]["status"] == "unreviewed"
    assert row["sources"] == ["svo"]
    assert "NJM unavailable" in result["errors"]


def test_svo_availability_works_without_njm_or_downloads(tmp_path):
    library = FilterLibrary(tmp_path)
    library.svo = Catalogue()
    library.svo.metadata_client = SimpleNamespace(get_filter=lambda fid:
        SVOFilterRecord(fid, PHOTON, np.array([4000., 6000.]), np.ones(2), "Angstrom", "Filter"))
    info = library.available("NewFacility", "NewCamera")
    assert info["availability_source"] == "svo"
    assert info["buildable"] == ["filter"]
    assert info["atmosphere"] is False
    assert not list(tmp_path.rglob("*.dat"))


def test_declared_partial_download_is_not_accepted(tmp_path):
    directory = tmp_path / "Test" / "Camera"
    directory.mkdir(parents=True)
    (directory / "g.dat").write_text("4000 1\n6000 1\n")
    InstrumentManifest("Test", "Camera", ["g", "r"]).save(directory)
    assert not FilterLibrary.published_complete(directory)


def test_svo_fallback_tracks_and_recovers_partial_downloads(tmp_path, monkeypatch):
    from sed_tools.svo_filter_grabber import SVOFilterBrowser

    browser = SVOFilterBrowser(base_dir=str(tmp_path))
    catalogue = Catalogue()
    monkeypatch.setattr(browser, "list_facilities", catalogue.list_facilities)
    monkeypatch.setattr(browser, "list_instruments", catalogue.list_instruments)
    rows = [FilterRow(f"NewFacility/NewCamera.{b}", b, "NewFacility", "NewCamera", "") for b in ("g", "r")]
    monkeypatch.setattr(browser, "list_filters", lambda f, i: rows)
    record = lambda fid: SVOFilterRecord(fid, PHOTON, np.array([4000., 6000.]), np.ones(2), "Angstrom", "Filter")
    monkeypatch.setattr(browser.metadata_client, "get_filter", lambda fid: None if fid.endswith(".r") else record(fid))
    library = FilterLibrary(tmp_path)
    library.svo = browser
    with pytest.raises(ComponentError, match="Incomplete passband"):
        library.published("NewFacility", "NewCamera")
    directory = tmp_path / "NewFacility" / "NewCamera"
    assert InstrumentManifest.load(directory).missing_files(directory) == ["r.dat"]
    monkeypatch.setattr(browser.metadata_client, "get_filter", record)
    assert library.published("NewFacility", "NewCamera") == directory
    assert FilterLibrary.published_complete(directory)


def test_svo_outage_keeps_njm_and_original_catalogues(tmp_path):
    library = FilterLibrary(tmp_path)
    library.svo = SimpleNamespace(list_facilities=lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    library.mirror = SimpleNamespace(is_available=lambda: True,
                                     discover_facilities=lambda: ["MirrorOnly"],
                                     discover_instruments=lambda f: ["Camera"])
    inventory = library.inventory()
    assert any(r["facility"] == "MirrorOnly" and r["instrument"] == "Camera" for r in inventory["instruments"])
    assert any("SVO" in error for error in inventory["errors"])


def test_inventory_offline_cli_emits_clean_json(tmp_path, monkeypatch, capsys):
    import json
    monkeypatch.setattr("sys.argv", ["sed-tools", "filters-inventory", "--offline", "--json", "--base", str(tmp_path)])
    assert cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["instruments"]
    assert report["errors"] == []
