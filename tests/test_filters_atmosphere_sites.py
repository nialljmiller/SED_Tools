"""Atmosphere sites in filter manifests: legacy and team sites, provenance, unstated
airmass, which site is offered, compose with a team atmosphere, the coverage tolerance, and errors that
name the band, the sites or the conditions.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sed_tools import filter_library
from sed_tools.component_sources._common import DerivedCurve, SiteAtmosphere, write_site
from sed_tools.facility_sites import instrument_sites
from sed_tools.filter_components import (FILTER, AtmosphereGrid, AtmosphereSpec, ComponentCoverageError, ComponentError,
                                         InstrumentManifest, add_component, build_product, compose_band,
                                         resolve_atmosphere)
from sed_tools.filter_library import FilterLibrary
from sed_tools.filter_metadata import PHOTON, read_filter_header, released_mesa_reads, write_filter_file

from helpers_builders_sites import _curve, _site


WL = np.linspace(4000.0, 6000.0, 201)


def _instrument(root, facility="Test", instrument="Camera", *, default_site=None, sites=(), outside="error"):
    inst = root / facility / instrument
    inst.mkdir(parents=True)
    write_filter_file(inst / "g.dat", WL, np.exp(-0.5 * ((WL - 5000) / 200) ** 2), detector_type=PHOTON,
                      source="svo", wavelength_unit="Angstrom", extra={"Components": "filter"})
    manifest = InstrumentManifest.load(inst)
    manifest.facility, manifest.instrument = facility, instrument
    manifest.ground_based = True
    manifest.default_site = default_site
    manifest.atmosphere_sites = list(sites)
    manifest.outside_range = outside
    manifest.save(inst)
    return inst


# ---------------------------------------------------------------------------
# Manifests stay byte-compatible
# ---------------------------------------------------------------------------
LEGACY_SITE = {"altitude_m": 2207.0, "entries": [{"airmass": 1.2, "file": "am1.20.dat", "pwv": None}],
               "notes": "", "schema": 1, "site": "ctio_des", "source": "DES standard atmosphere"}


def test_legacy_site_manifest_round_trips_unchanged(tmp_path):
    (tmp_path / "ctio_des").mkdir()
    text = json.dumps(LEGACY_SITE, indent=2, sort_keys=True) + "\n"
    (tmp_path / "ctio_des" / "manifest.json").write_text(text)
    AtmosphereGrid.load(tmp_path / "ctio_des").save(tmp_path / "ctio_des")
    assert (tmp_path / "ctio_des" / "manifest.json").read_text() == text


def test_instrument_manifest_without_extra_sites_has_no_new_key(tmp_path):
    inst = _instrument(tmp_path, default_site="lsst_std")
    assert "atmosphere_sites" not in json.loads((inst / "manifest.json").read_text())


def test_grid_entries_need_a_numeric_airmass():
    with pytest.raises(ComponentError, match="unstated"):
        AtmosphereGrid("s", [{"airmass": None, "pwv": None, "file": "a.dat"}])


# ---------------------------------------------------------------------------
# Registering a team atmosphere as a site
# ---------------------------------------------------------------------------
def test_write_site_records_provenance_and_derivation(tmp_path):
    verbatim = DerivedCurve("https://team/atm10.dat", _curve([0.4, 0.7, 0.8]), derivation="", note="as published")
    site = _site(entries=[(1.0, 1.5, verbatim),
                          (1.2, None, DerivedCurve("https://team/a | https://team/b", _curve([0.5, 0.8, 0.9]),
                                                   "a / b"))],
                 terms="Cite us.", outside_range="error")
    site_dir = write_site(tmp_path, site, None)

    raw = json.loads((site_dir / "manifest.json").read_text())
    assert raw["kind"] == "team" and raw["model"] == "MODTRAN" and raw["terms"] == "Cite us."
    assert raw["entries"][0] == {"airmass": 1.0, "pwv": 1.5, "file": "am1.00_pwv1.5.dat",
                                 "source": "https://team/atm10.dat", "note": "as published"}
    assert raw["entries"][1]["derivation"] == "a / b"
    assert "unstated" not in raw
    original, derived = (read_filter_header(site_dir / e["file"]) for e in raw["entries"])
    assert original["ComponentsSource"] == "original" and derived["ComponentsSource"] == "derived"
    assert derived["Atmosphere"] == "site=here_team airmass=1.2"
    assert all(released_mesa_reads(site_dir / e["file"]) for e in raw["entries"])


def test_write_site_without_provenance_keeps_the_legacy_layout(tmp_path):
    site = SiteAtmosphere("ctio_des", 2207.0, [(1.2, None, DerivedCurve("u", _curve([0.5, 0.8, 0.9]), ""))],
                          "DES standard atmosphere")
    write_site(tmp_path, site, None)
    assert json.loads((tmp_path / "ctio_des" / "manifest.json").read_text()) == LEGACY_SITE
    assert "ComponentsSource" not in read_filter_header(tmp_path / "ctio_des" / "am1.20.dat")


def test_unstated_airmass_is_listed_apart_and_selected_explicitly(tmp_path):
    site = _site("plexus_team", entries=[(None, None, DerivedCurve("u", _curve([0.5, 0.8, 0.9]), ""))])
    write_site(tmp_path, site, None)
    raw = json.loads((tmp_path / "plexus_team" / "manifest.json").read_text())
    assert raw["entries"] == [] and raw["unstated"][0]["file"] == "am_unstated.dat"
    # Released SED_Tools lists every served site with float(e["airmass"]).
    assert sorted({float(e["airmass"]) for e in raw["entries"]}) == []

    grid = AtmosphereGrid.load(tmp_path / "plexus_team")
    assert grid.select(AtmosphereSpec("plexus_team", None)) == "am_unstated.dat"
    with pytest.raises(ComponentError, match="unstated airmass: airmass=None"):
        grid.select(AtmosphereSpec("plexus_team", 1.0))
    with pytest.raises(ComponentError, match="atmosphere needs at least 'airmass'"):
        AtmosphereSpec.coerce({"site": "plexus_team"})
    assert AtmosphereSpec.coerce({"site": "plexus_team", "airmass": None}).label == "plexus_team_amunstated"
    with pytest.raises(ComponentError, match="no pwv"):
        AtmosphereSpec("plexus_team", None, 2.0)


# ---------------------------------------------------------------------------
# Which sites an instrument may use
# ---------------------------------------------------------------------------
def test_team_site_is_offered_next_to_the_default_site(tmp_path):
    inst = _instrument(tmp_path / "filters", default_site="paranal", sites=["here_team"])
    manifest = InstrumentManifest.load(inst)
    assert manifest.sites() == ["paranal", "here_team"]
    assert resolve_atmosphere(manifest, AtmosphereSpec(None, 1.2, 2.5)).site == "paranal"
    assert resolve_atmosphere(manifest, AtmosphereSpec("here_team", 1.2)).site == "here_team"
    with pytest.raises(ComponentError, match=r"observes from paranal, here_team; atmospheres from other sites \(lsst_std\) are not mixed in"):
        resolve_atmosphere(manifest, AtmosphereSpec("lsst_std", 1.2))
    info = manifest.describe()
    assert info["atmosphere"] is True and info["atmosphere_sites"] == ["paranal", "here_team"]


def test_registered_team_sites_need_no_manifest_change():
    ps1 = InstrumentManifest(facility="PAN-STARRS", instrument="PS1", bands=["g"], ground_based=True)
    assert resolve_atmosphere(ps1, AtmosphereSpec(None, 1.2)).site == "haleakala_ps1"
    twomass = InstrumentManifest(facility="2MASS", instrument="2MASS", bands=["J"], ground_based=True)
    with pytest.raises(ComponentError, match="several atmosphere sites"):
        resolve_atmosphere(twomass, AtmosphereSpec(None, None))
    assert resolve_atmosphere(twomass, AtmosphereSpec("ctio_2mass", None)).site == "ctio_2mass"
    vista = InstrumentManifest(facility="Paranal", instrument="VISTA", bands=["J"], ground_based=True,
                               default_site="paranal")
    assert vista.sites() == ["paranal", "paranal_vircam"]
    assert instrument_sites("Gemini", "GPI") == ()  # moved between sites: deliberately absent


def test_existing_single_site_instruments_are_unchanged(tmp_path):
    inst = _instrument(tmp_path / "filters", "LSST", "LSST", default_site="lsst_std")
    manifest = InstrumentManifest.load(inst)
    assert manifest.sites() == ["lsst_std"]
    with pytest.raises(ComponentError, match=r"observes from lsst_std; atmospheres from other sites \(.*\) are not mixed in"):
        resolve_atmosphere(manifest, AtmosphereSpec("paranal", 1.2, 2.5))


# ---------------------------------------------------------------------------
# Composition with team atmospheres
# ---------------------------------------------------------------------------
def test_compose_with_unstated_team_atmosphere(tmp_path):
    inst = _instrument(tmp_path / "filters", sites=["plexus_team"])
    write_site(tmp_path / "atm", _site("plexus_team", entries=[
        (None, None, DerivedCurve("u", _curve([0.5, 0.5, 0.5]), ""))]), None)
    out = build_product(inst, ["filter"], atmosphere={"site": "plexus_team", "airmass": None},
                        atmosphere_root=tmp_path / "atm")
    assert out.name == "filter+atm_plexus_team_amunstated"
    header = read_filter_header(out / "g.dat")
    assert header["Atmosphere"] == "site=plexus_team airmass=unstated"
    assert header["Components"] == "filter+atmosphere"


def test_site_outside_range_overrides_the_instruments_policy(tmp_path):
    short = DerivedCurve("u", lambda s: (np.array([4500.0, 5500.0]), np.array([0.9, 0.9])), "")
    inst = _instrument(tmp_path / "filters", default_site="short_team", outside="edge")
    write_site(tmp_path / "atm", _site("short_team", entries=[(1.2, None, short)], outside_range="error"), None)
    with pytest.raises(ComponentCoverageError):
        compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=tmp_path / "atm")
    # Unset on the site: the instrument's own convention (here edge) applies, as before.
    write_site(tmp_path / "atm2", _site("short_team", entries=[(1.2, None, short)]), None)
    curve = compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=tmp_path / "atm2")
    assert curve.transmission.max() == pytest.approx(0.9 * 1.0, rel=1e-3)


def test_add_component_points_atmospheres_to_sites(tmp_path):
    with pytest.raises(ComponentError, match="site of its own"):
        add_component(tmp_path, "atmosphere", [4000.0, 5000.0], [0.9, 0.9])


def test_library_rebuilds_a_missing_team_site(tmp_path, monkeypatch):
    monkeypatch.setattr(filter_library.NJMFilterGrabber, "_check_availability", lambda self: False)
    inst = _instrument(tmp_path / "filters", "PAN-STARRS", "PS1")
    library = FilterLibrary(tmp_path / "filters")
    calls = []

    def build(root, **kw):
        calls.append(kw)
        write_site(kw["atmosphere_root"], _site("haleakala_ps1"), None)

    monkeypatch.setattr(filter_library, "get_builder", lambda name: calls.append(name) or build)
    path = library.prepare_atmosphere(InstrumentManifest.load(inst), AtmosphereSpec(None, 1.2))
    assert path.name == "am1.20.dat"
    assert calls[0] == "team_atmospheres" and calls[1]["sites"] == ["haleakala_ps1"]
    listing = library.atmospheres(remote=False)["haleakala_ps1"]
    assert listing["kind"] == "team" and listing["airmass"] == [1.2]


# ---------------------------------------------------------------------------
# SkyCalc: derived grids, the web form, provenance
# ---------------------------------------------------------------------------
def test_coverage_tolerance_allows_faint_wings_only(tmp_path):
    # Band: a Gaussian at 5000 A (sigma 200); atmosphere table from 4500 A.
    atm = DerivedCurve("u", lambda s: (np.array([4500.0, 7000.0]), np.array([0.8, 0.8])), "")
    inst = _instrument(tmp_path / "filters", default_site="short_team")
    write_site(tmp_path / "loose", _site("short_team", entries=[(1.2, None, atm)], outside_range="error",
                                         coverage_tolerance=1e-2), None)
    curve = compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=tmp_path / "loose")
    assert curve.transmission[curve.wavelength < 4500].max() > 0  # the wing holds the edge value
    write_site(tmp_path / "tight", _site("short_team", entries=[(1.2, None, atm)], outside_range="error",
                                         coverage_tolerance=1e-4), None)
    with pytest.raises(ComponentCoverageError, match="lies beyond it"):
        compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=tmp_path / "tight")


# ---------------------------------------------------------------------
# Atmosphere and coverage errors say what to choose, not Python syntax
# ---------------------------------------------------------------------
def _camera(root, *, default_site=None, sites=()):
    """Test/Camera with one band g, a Gaussian at 5000 Å, and the given atmosphere sites."""
    from sed_tools.filter_components import InstrumentManifest
    from sed_tools.filter_metadata import PHOTON, write_filter_file

    inst = root / "Test" / "Camera"
    inst.mkdir(parents=True)
    wl = np.linspace(4000.0, 6000.0, 201)
    write_filter_file(inst / "g.dat", wl, np.exp(-0.5 * ((wl - 5000) / 200) ** 2), detector_type=PHOTON,
                      source="svo", wavelength_unit="Angstrom", extra={"Components": "filter"})
    manifest = InstrumentManifest.load(inst)
    manifest.facility, manifest.instrument = "Test", "Camera"
    manifest.ground_based = True
    manifest.default_site = default_site
    manifest.atmosphere_sites = list(sites)
    manifest.outside_range = "error"
    manifest.save(inst)
    return inst


def _write_team_site(root, name, entries, **kw):
    """A team atmosphere site; entries are (airmass, pwv, (wavelengths, transmission))."""
    from sed_tools.component_sources._common import DerivedCurve, SiteAtmosphere, write_site
    from sed_tools.filter_components import TEAM

    def table(wavelength, transmission):
        return lambda session: (np.array(wavelength), np.array(transmission))

    curves = [(am, pwv, DerivedCurve("u", table(*curve), "")) for am, pwv, curve in entries]
    write_site(root, SiteAtmosphere(site=name, altitude_m=1000.0, entries=curves, source="test team",
                                    kind=TEAM, model="MODTRAN", model_version="not stated",
                                    citation="Someone 2001", **kw), None)


@pytest.mark.parametrize("site_kw, match", [
    ({}, r"^Test/Camera band g: .* covers 4500-7000 A but the band spans"),
    ({"outside_range": "error", "coverage_tolerance": 1e-4}, r"^Test/Camera band g: .* lies beyond it"),
])
def test_coverage_error_names_the_band(tmp_path, site_kw, match):
    # Used to be "am1.00_pwv1.dat covers 9000-56000 A but 70.25% of the band lies beyond it".
    from sed_tools.filter_components import FILTER, ComponentCoverageError, compose_band

    inst = _camera(tmp_path / "filters", default_site="short_team")
    _write_team_site(tmp_path / "atm", "short_team", [(1.2, None, ([4500.0, 7000.0], [0.8, 0.8]))], **site_kw)
    with pytest.raises(ComponentCoverageError, match=match):
        compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=tmp_path / "atm")


def test_several_sites_error_lists_the_sites_to_choose_from():
    # Used to be "... sites ['hopkins_2mass', 'ctio_2mass']; choose one with atmosphere={'site': ..., ...}".
    from sed_tools.filter_components import AtmosphereSpec, ComponentError, InstrumentManifest, resolve_atmosphere

    twomass = InstrumentManifest(facility="2MASS", instrument="2MASS", bands=["J"], ground_based=True)
    with pytest.raises(ComponentError) as excinfo:
        resolve_atmosphere(twomass, AtmosphereSpec(None, None))
    assert str(excinfo.value) == "2MASS/2MASS has several atmosphere sites; choose a site: hopkins_2mass, ctio_2mass"


def test_missing_conditions_error_says_what_the_site_has(tmp_path):
    # Used to be "No cached atmosphere for hopkins_2mass_am1.2; fetch or generate the requested conditions".
    from sed_tools.filter_components import AtmosphereSpec, ComponentError, atmosphere_file

    flat = ([3000.0, 7000.0], [0.9, 0.9])
    _write_team_site(tmp_path, "plexus_team", [(None, None, flat)])
    _write_team_site(tmp_path, "grid_team", [(1.0, 2.5, flat), (1.2, 2.5, flat), (1.2, 5.0, flat)])
    cases = [
        (AtmosphereSpec("plexus_team", 1.2),
         "Site plexus_team has no atmosphere for airmass 1.2; it has unstated airmass (the atmosphere as published)"),
        (AtmosphereSpec("grid_team", 1.5, 2.5),
         "Site grid_team has no atmosphere for airmass 1.5 and PWV 2.5 mm; it has airmass 1, 1.2 with PWV 2.5, 5 mm"),
        (AtmosphereSpec("nowhere", 1.2), "No atmosphere of site nowhere is cached; fetch or generate the site first"),
    ]
    for spec, message in cases:
        with pytest.raises(ComponentError) as excinfo:
            atmosphere_file(tmp_path, spec)
        assert str(excinfo.value) == message


def test_atmosphere_without_conditions_error_has_no_python_syntax(tmp_path):
    from sed_tools.filter_components import ATMOSPHERE, FILTER, ComponentError, compose_band

    inst = _camera(tmp_path / "filters", default_site="short_team")
    with pytest.raises(ComponentError) as excinfo:
        compose_band(inst, "g", [FILTER, ATMOSPHERE], atmosphere_root=tmp_path / "atm")
    assert str(excinfo.value) == ("Including the atmosphere needs the observing conditions: an airmass, "
                                  "and a PWV where the site models it")
