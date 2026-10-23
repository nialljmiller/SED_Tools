"""Separable filter components: filter, optics, QE, atmosphere."""

from __future__ import annotations

import json

import numpy as np
import pytest
import requests

from sed_tools.combine_filters import write_filter_index
from sed_tools.filter_components import (ATMOSPHERE, FILTER, OPTICS, PUBLISHED, QE,
                                         AtmosphereGrid, AtmosphereSpec,
                                         ComponentCoverageError, ComponentError,
                                         Curve, InstrumentManifest, add_component,
                                         build_all_products, build_product, compose_band,
                                         multiply_curves, parse_components,
                                         recipe_name)
from sed_tools.filter_metadata import ENERGY, PHOTON, read_filter_header, write_filter_file
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
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Filter + CCD + Instrument", (FILTER, QE, "instrument")),
        ("Filter + Instrument + Atmosphere", (FILTER, "instrument", ATMOSPHERE)),
        ("Filter + CCD + Instrument + Other optical components + Atmosphere",
         (FILTER, OPTICS, QE, "instrument", ATMOSPHERE)),
        ("Theoretically defined", (FILTER,)),
        ("qe+filter", (FILTER, QE)),
        (["optics", "filter"], (FILTER, OPTICS)),
        ("published+atmosphere", (PUBLISHED, ATMOSPHERE)),
        ("", ()),
    ],
)
def test_parse_components(text, expected):
    assert parse_components(text) == expected


def test_parse_components_rejects_unknown():
    with pytest.raises(ComponentError):
        parse_components("filter+magic")


def test_recipe_name_is_canonical_order():
    assert recipe_name(["qe", "filter", "optics"]) == "filter+optics+qe"


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------
def test_multiply_curves_merges_grids_and_multiplies():
    a = Curve(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), {})
    b = Curve(np.array([0.0, 1.5, 4.0]), np.array([0.5, 0.5, 0.5]), {})
    grid, product = multiply_curves([a, b])
    np.testing.assert_allclose(grid, [1.0, 1.5, 2.0, 3.0])
    np.testing.assert_allclose(product, 0.5)


def test_multiply_curves_refuses_to_extrapolate():
    band = Curve(WL, _gauss(5000, 300), {})
    short = Curve(np.array([4500.0, 7000.0]), np.array([1.0, 1.0]), {})
    with pytest.raises(ComponentCoverageError):
        multiply_curves([band, short])
    grid, product = multiply_curves([band, short], allow_partial=True)
    assert np.all(product[grid < 4500.0] == 0.0)


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------
def test_manifest_from_headers_registers_filter_only_bands(tmp_path):
    inst = tmp_path / "Generic" / "Johnson"
    _published(inst, "V", WL, _gauss(5500, 300), "filter", detector=ENERGY)
    _published(inst, "B", WL, _gauss(4400, 300), "filter", detector=ENERGY)

    m = InstrumentManifest.load(inst)

    assert m.published["components"] == "filter"
    assert m.available_roles() == (FILTER,)
    assert m.component_file(FILTER, "V") == "V.dat"
    assert m.ground_based is True
    assert m.detector_type == ENERGY


def test_manifest_reports_mixed_published_contents(tmp_path):
    inst = tmp_path / "GAIA" / "GAIA"
    _published(inst, "G", WL, _gauss(5500, 300), "filter+qe+instrument")
    _published(inst, "Gbp_faint", WL, _gauss(5000, 300), "filter+instrument")

    info = InstrumentManifest.load(inst).describe()

    assert info["published"] == "mixed"
    assert info["published_bands"] == {"G": "filter+qe+instrument", "Gbp_faint": "filter+instrument"}
    assert info["platform"] == "space"


def test_manifest_round_trip(tmp_path):
    inst = _component_instrument(tmp_path)
    m = InstrumentManifest.load(inst)
    assert set(m.available_roles()) == {FILTER, OPTICS, QE}
    assert json.loads((inst / "manifest.json").read_text())["schema"] == 1


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
def test_build_product_multiplies_components_and_is_a_mesa_instrument(tmp_path):
    inst = _component_instrument(tmp_path)
    before = {p.name: p.read_bytes() for p in inst.iterdir() if p.is_file()}

    out = build_product(inst, ["filter", "optics", "qe"])

    assert out == inst / "products" / "filter+optics+qe"
    assert (out / "filter+optics+qe").read_text().split() == ["b1.dat", "b2.dat"]
    curve = load_filter_curve(str(out / "b1.dat"))
    np.testing.assert_allclose(curve.transmission, _gauss(4800, 150, curve.wavelength) * 0.48, rtol=1e-6)
    header = read_filter_header(out / "b1.dat")
    assert header["Components"] == "filter+optics+qe"
    assert header["DetectorType"] == PHOTON  # QE included
    # Published curves and the MESA index are untouched.
    assert {p.name: p.read_bytes() for p in inst.iterdir() if p.is_file()} == before


def test_filter_only_product_inherits_detector_type(tmp_path):
    inst = _component_instrument(tmp_path)  # manifest detector_type = energy
    out = build_product(inst, ["filter"])
    assert read_filter_header(out / "b2.dat")["DetectorType"] == ENERGY


def test_compose_requires_filter_or_published(tmp_path):
    inst = _component_instrument(tmp_path)
    with pytest.raises(ComponentError, match="needs the 'filter'"):
        compose_band(inst, "b1", ["optics", "qe"])


def test_missing_component_explains_what_exists(tmp_path):
    inst = tmp_path / "GAIA" / "GAIA"
    _published(inst, "G", WL, _gauss(5500, 300), "filter+qe+instrument")
    with pytest.raises(ComponentError, match="cannot provide 'filter'.*buildable: none; published curve: filter\\+qe\\+instrument"):
        build_product(inst, ["filter"])


def test_composite_component_covers_its_roles(tmp_path):
    """Sources such as SDSS publish only the system response without atmosphere."""
    inst = tmp_path / "SLOAN" / "SDSS"
    _published(inst, "g", WL, _gauss(4800, 150) * 0.5, "filter+instrument+atmosphere")
    m = InstrumentManifest.load(inst)
    add_component(inst, "qe+filter+optics", WL, _gauss(4800, 150) * 0.6, band="g", manifest=m)
    m.save(inst)
    m = InstrumentManifest.load(inst)

    assert m.buildable() == ["filter+optics+qe"]
    assert m.available_roles() == ()
    out = build_product(inst, ["filter", "optics", "qe"])
    np.testing.assert_allclose(load_filter_curve(str(out / "g.dat")).transmission,
                               _gauss(4800, 150) * 0.6, rtol=1e-6)
    with pytest.raises(ComponentError, match="buildable: filter\\+optics\\+qe"):
        build_product(inst, ["filter"])


def test_cover_combines_composite_and_single_pieces(tmp_path):
    inst = tmp_path / "X" / "Y"
    _published(inst, "b", WL, _gauss(5000, 150), "filter+instrument")
    m = InstrumentManifest.load(inst)
    add_component(inst, "filter+optics", WL, _gauss(5000, 150) * 0.9, band="b", manifest=m)
    add_component(inst, QE, [3000.0, 7000.0], [0.5, 0.5], name="ccd", manifest=m)

    pieces = m.cover("b", [FILTER, OPTICS, QE])

    assert [p[0] for p in pieces] == [(FILTER, OPTICS), (QE,)]
    assert m.buildable() == ["filter+optics", "filter+optics+qe"]


def test_atmosphere_is_composed_from_grid(tmp_path):
    inst = _with_site(_component_instrument(tmp_path / "filters"))
    atm = _atmosphere(tmp_path / "atmospheres")

    out = build_product(inst, ["filter"], atmosphere=ATM, atmosphere_root=atm)

    assert out.name == "filter+atm_paranal_am1.2_pwv2.5"
    curve = load_filter_curve(str(out / "b1.dat"))
    np.testing.assert_allclose(curve.transmission, _gauss(4800, 150, curve.wavelength) * 0.9, rtol=1e-6)
    assert read_filter_header(out / "b1.dat")["Atmosphere"] == "site=paranal airmass=1.2 pwv=2.5"


def test_atmosphere_grid_reports_available_points(tmp_path):
    atm = _atmosphere(tmp_path)
    grid = AtmosphereGrid.load(atm / "paranal")
    with pytest.raises(ComponentError, match=r"Grid: airmass \[1.2\], pwv \[2.5\]"):
        grid.select(AtmosphereSpec("paranal", 1.5, 2.5))


def test_published_with_atmosphere_is_not_double_counted(tmp_path):
    inst = tmp_path / "2MASS" / "2MASS"
    _published(inst, "J", WL, _gauss(5000, 300), "filter+instrument+atmosphere")
    _with_site(inst)
    atm = _atmosphere(tmp_path / "atm")
    with pytest.raises(ComponentError, match="double count"):
        build_product(inst, ["published"], atmosphere=ATM, atmosphere_root=atm)


def test_published_plus_atmosphere_keeps_published_detector_type(tmp_path):
    inst = tmp_path / "Generic" / "Johnson"
    _published(inst, "V", WL, _gauss(5000, 300), "filter", detector=ENERGY)
    _with_site(inst)
    atm = _atmosphere(tmp_path / "atm")
    out = build_product(inst, ["published"], atmosphere=ATM, atmosphere_root=atm)
    header = read_filter_header(out / "V.dat")
    assert header["DetectorType"] == ENERGY
    assert header["Components"] == "filter+atmosphere"
    assert header["Recipe"].startswith("published+atmosphere")


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
def test_atmosphere_from_another_site_is_refused(tmp_path):
    inst = _with_site(_component_instrument(tmp_path / "filters"), "lsst_std")
    atm = _atmosphere(tmp_path / "atm")
    with pytest.raises(ComponentError, match="observes from 'lsst_std'.*not mixed in"):
        build_product(inst, ["filter"], atmosphere={"site": "paranal", **ATM}, atmosphere_root=atm)


def test_instrument_without_own_site_gets_no_atmosphere(tmp_path):
    inst = _component_instrument(tmp_path / "filters")  # ground-based, no site assigned
    atm = _atmosphere(tmp_path / "atm")
    with pytest.raises(ComponentError, match="no atmosphere site of its own"):
        build_product(inst, ["filter"], atmosphere=ATM, atmosphere_root=atm)
    assert InstrumentManifest.load(inst).describe()["atmosphere"] is False


def test_manifest_cannot_point_at_another_instrument(tmp_path):
    other = tmp_path / "SLOAN" / "SDSS"
    other.mkdir(parents=True)
    write_filter_file(other / "qe.dat", [3000.0, 7000.0], [0.5, 0.5], detector_type=PHOTON,
                      source="component", wavelength_unit="Angstrom", extra={"Role": "qe", "Components": "qe"})
    inst = _component_instrument(tmp_path)
    m = InstrumentManifest.load(inst)
    m.components[QE] = {"*": "../../SLOAN/SDSS/qe.dat"}
    m.save(inst)
    with pytest.raises(ComponentError, match="outside the instrument directory"):
        build_product(inst, ["filter", "qe"])


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
def test_bin_average_preserves_the_mean():
    from sed_tools.component_sources.skycalc import bin_average

    lam = 1000.0 * np.exp(np.arange(0, np.log(2.0), 1 / 200000))
    trans = 0.5 + 0.5 * np.sign(np.sin(lam * 50.0))  # unresolved "lines"
    wl, tr = bin_average(lam, trans, 1000, 1000.0, 2000.0)
    assert tr.mean() == pytest.approx(trans.mean(), abs=2e-3)
    assert np.all(np.diff(wl) > 0)


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
class _Resp:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(str(self.status_code), response=self)


def test_njm_downloads_manifest_and_components(tmp_path, monkeypatch):
    from sed_tools.njm_filter_grabber import NJMFilterGrabber

    manifest = {"facility": "Argus", "instrument": "System", "bands": ["b1"],
                "components": {"filter": {"b1": "components/filter/b1.dat"}, "qe": {"*": "components/qe/ccd.dat"}}}
    files = {
        "/Argus/System/manifest.json": json.dumps(manifest),
        "/Argus/System/components/filter/b1.dat": "# Role: filter\nWavelength,Transmission\n4000,1\n",
        "/Argus/System/components/qe/ccd.dat": "# Role: qe\nWavelength,Transmission\n4000,0.5\n",
    }
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    njm = NJMFilterGrabber(base_dir=str(tmp_path))

    def fake_get(url, timeout=None, **kw):
        for suffix, text in files.items():
            if url.endswith(suffix):
                return _Resp(text)
        return _Resp("", 404)

    monkeypatch.setattr(njm.session, "get", fake_get)

    assert njm.download_components("Argus", "System") == 2
    assert (tmp_path / "Argus/System/components/qe/ccd.dat").exists()
    assert InstrumentManifest.load(tmp_path / "Argus/System").component_file(QE, "b1") == "components/qe/ccd.dat"
    assert njm.download_components("Nope", "Nope") == 0


def test_filters_fetch_without_components_is_unchanged(tmp_path, monkeypatch):
    from sed_tools.api import Filters

    inst = tmp_path / "GAIA" / "GAIA"
    monkeypatch.setattr(Filters, "_fetch_published", classmethod(lambda cls, f, i, filter_root=None: inst))
    assert Filters.fetch("GAIA", "GAIA", filter_root=tmp_path) == inst


def test_products_do_not_shadow_published_filters(tmp_path):
    """photometry('b1') must still resolve the published curve after composing."""
    from sed_tools.combine_filters import find_filter_sets
    from sed_tools.models import _build_filter_index

    inst = _component_instrument(tmp_path)
    build_product(inst, ["filter"])

    index = _build_filter_index(tmp_path)
    assert index["b1"] == [inst / "b1.dat"]
    sets = find_filter_sets(tmp_path)
    assert inst in sets and inst / "products" / "filter" in sets
    assert not any("components" in p.parts for p in sets)


@pytest.mark.parametrize("facility,site", [
    ("Paranal", "paranal"), ("HAWK-I", "paranal"), ("La Silla", "lasilla"),
    ("KPNO", None), ("Subaru", None), ("JWST", None),
])
def test_only_eso_sites_get_a_skycalc_atmosphere(tmp_path, facility, site):
    inst = tmp_path / facility / "Inst"
    _published(inst, "b", WL, _gauss(5000, 300), "filter")
    info = InstrumentManifest.load(inst).describe()
    assert info["atmosphere_site"] == site
    assert info["atmosphere"] is (site is not None)


def test_svo_filter_only_sibling_is_the_bands_filter_component(tmp_path):
    inst = tmp_path / "SLOAN" / "SDSS"
    _published(inst, "g", WL, _gauss(4800, 150) * 0.5, "filter+instrument+atmosphere")
    _published(inst, "gprime_filter", WL, _gauss(4800, 150), "filter")
    _published(inst, "r", WL, _gauss(6200, 150) * 0.5, "filter+instrument+atmosphere")

    m = InstrumentManifest.load(inst)

    assert m.component_file(FILTER, "g") == "gprime_filter.dat"
    assert m.component_file(FILTER, "gprime_filter") == "gprime_filter.dat"
    assert m.component_file(FILTER, "r") is None
    assert m.primary_bands() == ["g", "r"]  # the sibling is g's component, not a band to compose
    assert m.bands_with(FILTER) == ["g"]
