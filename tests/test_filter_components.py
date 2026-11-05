"""Separable filter components: filter, optics, QE, atmosphere."""

from __future__ import annotations


import numpy as np
import pytest

from sed_tools.combine_filters import write_filter_index
from sed_tools.filter_components import (ATMOSPHERE, FILTER, OPTICS, QE,
                                         AtmosphereGrid, ComponentCoverageError, ComponentError,
                                         Curve, InstrumentManifest, add_component,
                                         build_all_products, build_product, compose_band,
                                         multiply_curves, parse_components)
from sed_tools.filter_metadata import ENERGY, PHOTON, write_filter_file
from sed_tools.flux_cube_tool import load_filter_curve

WL = np.linspace(4000.0, 6000.0, 201)


def _gauss(centre, width, wl=WL):
    return np.exp(-0.5 * ((wl - centre) / width) ** 2)


def _published(directory, band, wl, tr, components, detector=PHOTON, source="svo"):
    directory.mkdir(parents=True, exist_ok=True)
    write_filter_file(directory / f"{band}.dat", wl, tr, detector_type=detector, source=source,
                      wavelength_unit="Angstrom",
                      extra={"Components": components, "ComponentsSource": source})


def _component_instrument(root, facility="Argus", instrument="System"):
    """Instrument with filter (per band), shared optics and QE components."""
    inst = root / facility / instrument
    _published(inst, "b1", WL, _gauss(4800, 150) * 0.8 * 0.6, "filter+optics+qe")
    _published(inst, "b2", WL, _gauss(5300, 150) * 0.8 * 0.6, "filter+optics+qe")
    write_filter_index(inst, instrument)
    manifest = InstrumentManifest.load(inst)
    manifest.components = {}
    add_component(inst, FILTER, WL, _gauss(4800, 150), band="b1", manifest=manifest)
    add_component(inst, FILTER, WL, _gauss(5300, 150), band="b2", manifest=manifest)
    add_component(inst, OPTICS, [3000.0, 7000.0], [0.8, 0.8], name="optics", manifest=manifest)
    add_component(inst, QE, [3000.0, 7000.0], [0.6, 0.6], name="ccd", manifest=manifest)
    manifest.detector_type = ENERGY
    manifest.detector_types = {}  # legacy manifest with one instrument-wide convention
    manifest.save(inst)
    return inst


def _with_site(inst, site="paranal"):
    m = InstrumentManifest.load(inst)
    m.default_site = site
    m.save(inst)
    return inst


ATM = {"airmass": 1.2, "pwv": 2.5}


def _atmosphere(root, site="paranal", airmass=1.2, pwv=2.5, value=0.9):
    site_dir = root / site
    site_dir.mkdir(parents=True, exist_ok=True)
    name = f"am{airmass:.2f}_pwv{pwv:g}.dat"
    write_filter_file(site_dir / name, [1000.0, 3000.0, 3001.0, 300000.0], [0.0, 0.0, value, value],
                      detector_type=ENERGY, source="component", wavelength_unit="Angstrom",
                      extra={"Role": ATMOSPHERE, "Components": ATMOSPHERE})
    AtmosphereGrid(site=site, entries=[{"airmass": airmass, "pwv": pwv, "file": name}]).save(site_dir)
    return root


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
def test_parse_components():
    assert parse_components("") == ()


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------
def test_multiply_curves_refuses_to_extrapolate():
    band = Curve(WL, _gauss(5000, 300), {})
    short = Curve(np.array([4500.0, 7000.0]), np.array([1.0, 1.0]), {})
    with pytest.raises(ComponentCoverageError):
        multiply_curves([band, short])
    grid, product = multiply_curves([band, short], allow_partial=True)
    assert np.all(product[grid < 4500.0] == 0.0)


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
def test_compose_requires_filter_or_published(tmp_path):
    inst = _component_instrument(tmp_path)
    with pytest.raises(ComponentError, match="needs the 'filter'"):
        compose_band(inst, "b1", ["optics", "qe"])


def test_published_with_atmosphere_is_not_double_counted(tmp_path):
    inst = tmp_path / "2MASS" / "2MASS"
    _published(inst, "J", WL, _gauss(5000, 300), "filter+instrument+atmosphere")
    _with_site(inst)
    atm = _atmosphere(tmp_path / "atm")
    with pytest.raises(ComponentError, match="double count"):
        build_product(inst, ["published"], atmosphere=ATM, atmosphere_root=atm)


def test_space_instrument_refuses_atmosphere(tmp_path):
    inst = tmp_path / "TESS" / "TESS"
    _published(inst, "Red", WL, _gauss(5000, 300), "filter+instrument")
    atm = _atmosphere(tmp_path / "atm")
    with pytest.raises(ComponentError, match="space-based"):
        build_product(inst, ["published"], atmosphere=ATM, atmosphere_root=atm)


def test_published_requires_known_contents(tmp_path):
    inst = tmp_path / "Generic" / "Mystery"
    inst.mkdir(parents=True)
    (inst / "X.dat").write_text("Wavelength,Transmission\n4000,1\n5000,1\n")
    _with_site(inst)
    atm = _atmosphere(tmp_path / "atm")
    with pytest.raises(ComponentError, match="does not say what it contains"):
        build_product(inst, ["published"], atmosphere=ATM, atmosphere_root=atm)


# ---------------------------------------------------------------------------
# No mixing between instruments or sites
# ---------------------------------------------------------------------------
def test_instrument_without_own_site_gets_no_atmosphere(tmp_path):
    inst = _component_instrument(tmp_path / "filters")  # ground-based, no site assigned
    atm = _atmosphere(tmp_path / "atm")
    with pytest.raises(ComponentError, match="no atmosphere site of its own"):
        build_product(inst, ["filter"], atmosphere=ATM, atmosphere_root=atm)
    assert InstrumentManifest.load(inst).describe()["atmosphere"] is False


def test_build_all_products_makes_every_with_and_without_variant(tmp_path):
    inst = _with_site(_component_instrument(tmp_path / "filters"))
    atm = _atmosphere(tmp_path / "atm")

    built = build_all_products(inst, conditions=[ATM], atmosphere_root=atm)

    assert sorted(p.name for p in built) == sorted([
        "filter", "filter+optics", "filter+qe", "filter+optics+qe",
        "filter+atm_paranal_am1.2_pwv2.5", "filter+optics+atm_paranal_am1.2_pwv2.5",
        "filter+qe+atm_paranal_am1.2_pwv2.5", "filter+optics+qe+atm_paranal_am1.2_pwv2.5",
    ])


# ---------------------------------------------------------------------------
# SkyCalc builder (offline)
# ---------------------------------------------------------------------------
class _FakeSkyCalc:
    def __init__(self):
        self.calls = 0

    def transmission(self, site, airmass, pwv, lo, hi, resolution):
        self.calls += 1
        lam = lo * np.exp(np.arange(0, np.log(hi / lo), 1 / 2000.0))
        return lam, np.full_like(lam, 0.8 ** airmass)


def test_skycalc_build_writes_grid_and_resumes(tmp_path):
    from sed_tools.component_sources.skycalc import build

    fake = _FakeSkyCalc()
    root = build(tmp_path / "filters", sites=["paranal"], airmasses=[1.0, 2.0], pwvs=[2.5],
                 fetch_resolution=2000, store_resolution=500, chunks=2, client=fake)

    grid = AtmosphereGrid.load(root / "paranal")
    assert [e["file"] for e in grid.entries] == ["am1.00_pwv2.5.dat", "am2.00_pwv2.5.dat"]
    curve = load_filter_curve(str(root / "paranal" / "am2.00_pwv2.5.dat"))
    assert curve.transmission[curve.wavelength > 4000].mean() == pytest.approx(0.64)
    assert curve.transmission[curve.wavelength < 2999].max() == 0.0  # opaque below 300 nm
    calls = fake.calls
    build(tmp_path / "filters", sites=["paranal"], airmasses=[1.0, 2.0], pwvs=[2.5], client=fake)
    assert fake.calls == calls  # nothing refetched


# ---------------------------------------------------------------------------
# NJM mirror download
# ---------------------------------------------------------------------------
