"""Offline tests for the Argus builders (``Argus/DR280`` and the legacy ``Argus/System``).

``tests/fixtures/argus/`` holds excerpts of the real inputs pinned in
``sed_tools.component_sources.argus``: every 5th data row of each 1 nm
table (headers and bytes as they are), the whole 52-point IMX455 QE, and
every 100th row of the legacy System curves.  The pins are pointed at the
excerpts' SHA-256; public files are served by a fake session.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from sed_tools.api import Filters
from sed_tools.component_sources import argus, catalog
from sed_tools.filter_components import ComponentError, InstrumentManifest, compose_band, read_curve
from sed_tools.filter_metadata import read_filter_header, released_mesa_reads

FIXTURES = Path(__file__).parent / "fixtures" / "argus"
PUBLIC = ("argus_beta_ct446.csv", "argus_rho_ct699.csv", "imx455_qe.csv")
PRIVATE = ("addem_optics.csv", "synthref_response_rho.csv")  # DR280's non-public inputs
SYSTEM_RAW = ("argus_g.csv", "bandpass_560_825nm.csv", "optics_throughput.csv", "imx455_qe.csv")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Resp:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status_code, self.headers = content, status, {}

    def raise_for_status(self):
        if self.status_code != 200:
            import requests

            raise requests.HTTPError(str(self.status_code))


class _Session:
    """Serves the public excerpts at their pinned URLs."""

    def __init__(self):
        self.headers, self.calls = {}, []
        self.files = {argus.ARGUS_SIM_RAW.format(commit=argus.ARGUS_SIM_COMMIT, file=name): (FIXTURES / name).read_bytes()
                      for name in PUBLIC}

    def get(self, url, timeout=None, **kw):
        self.calls.append(url)
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


@pytest.fixture
def pins(monkeypatch):
    """Point every pin at the fixture excerpt of the same file."""
    def repinned(inputs):
        return {key: dataclasses.replace(spec, sha256=_sha(FIXTURES / spec.file)) for key, spec in inputs.items()}

    monkeypatch.setattr(argus, "DR280_INPUTS", repinned(argus.DR280_INPUTS))
    monkeypatch.setattr(argus, "SYSTEM_INPUTS", repinned(argus.SYSTEM_INPUTS))


@pytest.fixture
def raw(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    for name in PRIVATE + SYSTEM_RAW:
        shutil.copyfile(FIXTURES / name, directory / name)
    return directory


@pytest.fixture
def dr280(tmp_path, pins, raw):
    return argus.build(tmp_path / "filters", raw, instrument="DR280", session=_Session())


def _nm_table(name, column=1, skip=1):
    table = np.loadtxt(FIXTURES / name, delimiter=",", skiprows=skip, usecols=(0, column))
    return table[:, 0] * 10.0, table[:, 1]


# --------------------------------------------------------------------------- DR280 build
def test_dr280_writes_mesa_readable_totals_and_index(dr280):
    assert (dr280 / "DR280").read_text() == "beta.dat\nrho.dat\n"
    for band in ("beta", "rho"):
        path = dr280 / f"{band}.dat"
        assert released_mesa_reads(path)
        lines = path.read_text().splitlines()
        assert lines[0].startswith("Wavelength,Transmission,")
        assert not any(line.startswith("#") or not line.strip() for line in lines)
        header = read_filter_header(path)
        assert header["FilterID"] == f"Argus/DR280.{band}"
        assert header["DetectorType"] == "energy" and header["DetectorTypeSource"] == "user"
        assert header["Components"] == "filter+optics+qe"
        assert header["ComponentsSource"] == "composed"
        assert header["WavelengthUnit"] == "Angstrom"
        assert argus.ARGUS_SIM_COMMIT in header["Sources"] and "addem_optics.csv" in header["Sources"]
        curve = read_curve(path)
        assert curve.wavelength[0] == 3000.0 and curve.wavelength[-1] == 11000.0  # nm -> Angstrom
        assert np.all(np.diff(curve.wavelength) > 0) and curve.transmission.min() >= 0


def test_dr280_manifest_components_and_filter_variants(dr280):
    manifest = InstrumentManifest.load(dr280)
    assert (manifest.facility, manifest.instrument, manifest.bands) == ("Argus", "DR280", ["beta", "rho"])
    assert manifest.published == {"components": "filter+optics+qe", "source": "composed"}
    assert manifest.published_bands == {"beta": "filter+optics+qe", "rho": "filter+optics+qe"}
    assert manifest.detector_type == "energy"
    assert manifest.detector_types == {"beta": "energy", "rho": "energy"}
    assert manifest.detector_type_sources == {"beta": "user", "rho": "user"}
    assert "energy counting (N. Miller, 2026-10-27)" in manifest.sources["detector_type"]
    assert "photon-rate response" in manifest.sources["detector_type"]
    assert manifest.components == {
        "filter": {"beta": "components/filter/beta_ct446.dat", "rho": "components/filter/rho_ct699.dat"},
        "optics": {"*": "components/optics/dr280.dat"},
        "qe": {"*": "components/qe/imx455.dat"},
    }
    assert manifest.variants["filter"]["production"]["files"] == manifest.components["filter"]
    assert manifest.variants["filter"]["production"]["default"] is True
    assert manifest.variants["filter"]["addem"]["files"] == {
        "beta": "components/filter/beta_ct446.dat", "rho": "components/filter/rho0_ct692.dat"}
    assert "default" not in manifest.variants["filter"]["addem"]
    assert list(manifest.variants) == ["filter"]  # the DR280 optics are the only optics
    assert manifest.sources["optics"] == argus.DR280_INPUTS["dr280"].source
    assert manifest.missing_files(dr280) == []
    assert sorted(str(p.relative_to(dr280)) for p in (dr280 / "components").rglob("*.dat")) == [
        "components/filter/beta_ct446.dat", "components/filter/rho0_ct692.dat", "components/filter/rho_ct699.dat",
        "components/optics/dr280.dat", "components/qe/imx455.dat"]
    assert manifest.buildable() == ["filter", "filter+optics", "filter+qe", "filter+optics+qe"]
    described = manifest.describe()
    assert described["atmosphere"] is False and described["atmosphere_sites"] == []
    assert "IMX461" in manifest.notes and "No atmosphere is published" in manifest.notes
    assert "DetectorType energy (user)" in manifest.notes and "'cdk'" not in manifest.notes
    assert "available on request" in manifest.notes


def test_dr280_components_are_the_inputs_in_angstrom(dr280):
    expected = {
        "filter/beta_ct446.dat": _nm_table("argus_beta_ct446.csv"),
        "filter/rho_ct699.dat": _nm_table("argus_rho_ct699.csv"),
        "filter/rho0_ct692.dat": _nm_table("synthref_response_rho.csv", column=2),
        "optics/dr280.dat": _nm_table("addem_optics.csv"),
        "qe/imx455.dat": _nm_table("imx455_qe.csv", skip=0),
    }
    for rel, (wavelength, transmission) in expected.items():
        curve = read_curve(dr280 / "components" / rel)
        np.testing.assert_array_equal(curve.wavelength, wavelength)
        np.testing.assert_array_equal(curve.transmission, transmission)
        assert released_mesa_reads(dr280 / "components" / rel)


def test_dr280_totals_equal_the_product_of_the_default_components(dr280):
    optics = read_curve(dr280 / "components/optics/dr280.dat")
    qe = read_curve(dr280 / "components/qe/imx455.dat")
    for band, stem in (("beta", "beta_ct446"), ("rho", "rho_ct699")):
        total = read_curve(dr280 / f"{band}.dat")
        filt = read_curve(dr280 / f"components/filter/{stem}.dat")
        grid = total.wavelength
        # Zero outside a component's tabulated range, as the published curve states.
        product = (np.interp(grid, filt.wavelength, filt.transmission)
                   * np.interp(grid, optics.wavelength, optics.transmission, left=0.0, right=0.0)
                   * np.interp(grid, qe.wavelength, qe.transmission, left=0.0, right=0.0))
        np.testing.assert_allclose(total.transmission, product, rtol=1e-12, atol=1e-15)
        # Every sample of every factor inside the band lies on the published grid.
        for factor in (filt, optics, qe):
            inside = factor.wavelength[(factor.wavelength > grid[0]) & (factor.wavelength < grid[-1])]
            assert np.isin(inside, grid).all()


@pytest.mark.parametrize("band", ["beta", "rho"])
def test_compose_every_filter_variant(dr280, band):
    published = read_curve(dr280 / f"{band}.dat")
    implicit = compose_band(dr280, band, "filter+optics+qe")
    np.testing.assert_array_equal(implicit.wavelength, published.wavelength)
    np.testing.assert_array_equal(implicit.transmission, published.transmission)
    composed = {}
    for filt in ("production", "addem"):
        curve = compose_band(dr280, band, "filter+optics+qe", variants={"filter": filt})
        assert curve.header["Components"] == "filter+optics+qe"
        assert curve.transmission.max() > 0.5
        composed[filt] = curve
    np.testing.assert_array_equal(composed["production"].wavelength, published.wavelength)
    np.testing.assert_array_equal(composed["production"].transmission, published.transmission)
    addem = composed["addem"]
    if band == "beta":  # ADDem flew the production beta filter
        np.testing.assert_array_equal(addem.transmission, published.transmission)
    else:
        assert addem.header["Recipe"] == "filter+optics+qe [filter=addem]"
        assert not np.array_equal(addem.wavelength, published.wavelength)
    with pytest.raises(ComponentError, match="no optics variant 'cdk'"):
        compose_band(dr280, band, "filter+optics+qe", variants={"optics": "cdk"})


def test_filters_compose_builds_an_energy_labelled_mesa_product_for_each_filter_variant(dr280):
    for filt in ("production", "addem"):
        out = Filters.compose(dr280, "filter+optics+qe", variants={"filter": filt})
        assert out.name == f"filter+optics+qe+filter_{filt}"
        assert (out / out.name).read_text() == "beta.dat\nrho.dat\n"
        for band in ("beta", "rho"):
            assert released_mesa_reads(out / f"{band}.dat")
            header = read_filter_header(out / f"{band}.dat")
            assert header["DetectorType"] == "energy" and header["DetectorTypeSource"] == "composed"
    # Every recipe keeps the team's convention, with or without the QE.
    for recipe in ("filter", "filter+optics", "filter+qe"):
        out = Filters.compose(dr280, recipe)
        assert {read_filter_header(out / f"{b}.dat")["DetectorType"] for b in ("beta", "rho")} == {"energy"}
    production = Filters.compose(dr280, "filter+optics+qe", variants={"filter": "production"})
    for band in ("beta", "rho"):  # the product's samples are the published curve's
        assert ((production / f"{band}.dat").read_text().splitlines()[1:]
                == (dr280 / f"{band}.dat").read_text().splitlines()[1:])


def test_the_addem_rho_variant_matches_synthrefs_own_response(dr280):
    """synthref's R_noatm column is its rho0 filter_beam x DR280 optics x IMX455 QE."""
    reference = np.loadtxt(FIXTURES / "synthref_response_rho.csv", delimiter=",", skiprows=1)
    curve = compose_band(dr280, "rho", "filter+optics+qe", variants={"filter": "addem"})
    mine = np.interp(reference[:, 0] * 10.0, curve.wavelength, curve.transmission)
    np.testing.assert_allclose(mine, reference[:, 1], atol=2e-6)  # its CSV has 6 significant digits


def test_dr280_rebuild_is_identical(tmp_path, pins, raw):
    first = argus.build(tmp_path / "filters", raw, instrument="DR280", session=_Session())
    before = {p: p.read_bytes() for p in first.rglob("*") if p.is_file()}
    argus.build(tmp_path / "filters", raw, instrument="DR280", session=_Session())
    assert {p: p.read_bytes() for p in first.rglob("*") if p.is_file()} == before


def test_dr280_archives_every_download(tmp_path, pins, raw):
    archive = tmp_path / "archive"
    argus.build(tmp_path / "filters", raw, instrument="DR280", session=_Session(), archive_dir=archive)
    rows = json.loads((archive / "manifest.json").read_text())["files"]
    by_url = {row["url"]: row for row in rows}
    for name in PUBLIC:
        url = argus.ARGUS_SIM_RAW.format(commit=argus.ARGUS_SIM_COMMIT, file=name)
        assert by_url[url]["sha256"] == _sha(FIXTURES / name)
        assert by_url[url]["retrieved_utc"].endswith("Z")
        assert (archive / by_url[url]["file"]).read_bytes() == (FIXTURES / name).read_bytes()


# --------------------------------------------------------------------------- DR280 refusals
def test_dr280_without_the_private_inputs_points_to_njm(tmp_path, pins):
    session = _Session()
    with pytest.raises(FileNotFoundError) as excinfo:
        argus.build(tmp_path / "filters", None, instrument="DR280", session=session)
    message = str(excinfo.value)
    for name in PRIVATE:
        assert name in message
    assert "not public" in message and "NJM" in message and "Filters.fetch('Argus', 'DR280')" in message
    assert session.calls == []  # nothing downloaded before refusing
    assert not (tmp_path / "filters").exists()


def test_dr280_with_one_private_input_missing_names_it(tmp_path, pins, raw):
    (raw / "addem_optics.csv").rename(tmp_path / "elsewhere.csv")
    with pytest.raises(FileNotFoundError, match="addem_optics.csv .*hdps") as excinfo:
        argus.build(tmp_path / "filters", raw, instrument="DR280", session=_Session())
    assert "synthref_response_rho.csv" not in str(excinfo.value)


@pytest.mark.parametrize("name", ["addem_optics.csv", "argus_beta_ct446.csv"])
def test_dr280_refuses_an_input_that_is_not_the_pinned_file(tmp_path, pins, raw, name):
    session = _Session()
    if name in PUBLIC:
        url = argus.ARGUS_SIM_RAW.format(commit=argus.ARGUS_SIM_COMMIT, file=name)
        session.files[url] = session.files[url] + b"1101.000000,0.000000\n"
    else:
        (raw / name).write_bytes((raw / name).read_bytes() + b"1101,0.1\n")
    with pytest.raises(ValueError, match="SHA-256 .* is not the pinned"):
        argus.build(tmp_path / "filters", raw, instrument="DR280", session=session)
    assert not (tmp_path / "filters").exists()


@pytest.mark.parametrize("name, text", [("beta.dat", "Wavelength,Transmission\n4000.0,0.5\n5000.0,0.5\n"),
                                        ("DR280", "beta.dat\n")])
def test_dr280_never_replaces_a_different_published_file(tmp_path, pins, raw, name, text):
    inst = tmp_path / "filters" / "Argus" / "DR280"
    inst.mkdir(parents=True)
    (inst / name).write_text(text)
    with pytest.raises(RuntimeError, match="never replaced; nothing was written"):
        argus.build(tmp_path / "filters", raw, instrument="DR280", session=_Session())
    assert (inst / name).read_text() == text
    assert [p.name for p in inst.iterdir()] == [name]


def test_build_refuses_unknown_instruments_and_options(tmp_path):
    with pytest.raises(ValueError, match="available: DR280, System"):
        argus.build(tmp_path, instrument="ADDem")
    with pytest.raises(ValueError, match="local files only"):
        argus.build(tmp_path, instrument="System", archive_dir=tmp_path / "a")


# --------------------------------------------------------------------------- System (legacy)
@pytest.fixture
def system_dir(tmp_path):
    inst = tmp_path / "filters" / "Argus" / "System"
    inst.mkdir(parents=True)
    for band in ("beta", "rho"):
        shutil.copyfile(FIXTURES / f"system_{band}.dat", inst / f"{band}.dat")
    (inst / "System").write_text("beta.dat\nrho.dat\n")
    return inst


def test_system_builder_labels_line_1_and_keeps_every_sample(tmp_path, pins, raw, system_dir):
    before = {p.name: p.read_bytes() for p in system_dir.iterdir()}
    argus.build(tmp_path / "filters", raw, instrument="DR280", session=_Session())
    assert {p.name: p.read_bytes() for p in system_dir.iterdir()} == before
    assert read_filter_header(system_dir / "beta.dat") == {}  # as Nill wrote them
    argus.build(tmp_path / "filters", raw)  # System is the default instrument
    assert (system_dir / "System").read_bytes() == before["System"]
    for band in ("beta", "rho"):
        path = system_dir / f"{band}.dat"
        old, new = before[f"{band}.dat"].splitlines(keepends=True), path.read_bytes().splitlines(keepends=True)
        assert new[1:] == old[1:]  # MESA discards line 1, so it reads the same numbers
        assert old[0] == b"Wavelength,Transmission\n" and new[0].startswith(b"Wavelength,Transmission,")
        assert released_mesa_reads(path)
        header = read_filter_header(path)
        assert header == argus.system_header(band)
        assert header["FilterID"] == f"Argus/System.{band}"
        assert (header["DetectorType"], header["DetectorTypeSource"]) == ("energy", "user")
        assert "optics_throughput.csv" in header["Sources"] and "785c92e8" in header["Sources"]
    labelled = {p.name: p.read_bytes() for p in system_dir.rglob("*") if p.is_file()}
    argus.build(tmp_path / "filters", raw)
    assert {p.name: p.read_bytes() for p in system_dir.rglob("*") if p.is_file()} == labelled


def test_system_curve_without_a_column_row_is_refused(tmp_path, pins, raw, system_dir):
    samples = (system_dir / "beta.dat").read_text().splitlines(keepends=True)[1:]
    (system_dir / "beta.dat").write_text("".join(samples))
    with pytest.raises(ValueError, match="beta.dat has no column row"):
        argus.build(tmp_path / "filters", raw)
    assert (system_dir / "beta.dat").read_text() == "".join(samples)


def test_system_components_and_composition(tmp_path, pins, raw, system_dir):
    argus.build(tmp_path / "filters", raw, instrument="System")
    manifest = InstrumentManifest.load(system_dir)
    assert manifest.components == {
        "filter": {"beta": "components/filter/beta.dat", "rho": "components/filter/rho.dat"},
        "optics": {"*": "components/optics/optics.dat"},
        "qe": {"*": "components/qe/imx455.dat"},
    }
    assert manifest.published["components"] == "filter+optics+qe"
    assert manifest.detector_type == "energy"
    assert manifest.detector_types == {"beta": "energy", "rho": "energy"}
    assert manifest.detector_type_sources == {"beta": "user", "rho": "user"}
    assert "Argus/DR280" in manifest.notes and "DetectorType energy (user)" in manifest.notes
    beta = read_curve(system_dir / "components/filter/beta.dat")
    wavelength, percent = _nm_table("argus_g.csv")
    np.testing.assert_array_equal(beta.wavelength, wavelength)
    np.testing.assert_array_equal(beta.transmission, percent / 100.0)
    for band in ("beta", "rho"):
        curve = compose_band(system_dir, band, "filter+optics+qe")
        assert curve.transmission.max() > 0.5
    out = Filters.compose(system_dir, "filter+optics+qe")
    assert {read_filter_header(out / f"{b}.dat")["DetectorType"] for b in ("beta", "rho")} == {"energy"}


def test_system_without_raw_inputs_points_to_njm(tmp_path, pins, system_dir):
    with pytest.raises(FileNotFoundError, match="optics_throughput.csv .*ArgusSim.*NJM"):
        argus.build(tmp_path / "filters")
    assert not (system_dir / "components").exists()


# --------------------------------------------------------------------------- registry
def test_both_instruments_are_registered():
    dr280 = catalog.builder_for("Argus", "DR280")
    system = catalog.builder_for("Argus", "System")
    assert (dr280.name, dict(dr280.kwargs), dr280.publishes) == ("argus", {"instrument": "DR280"}, True)
    assert (system.name, dict(system.kwargs), system.publishes) == ("argus", {"instrument": "System"}, False)
    assert catalog.component_availability("Argus", "DR280")["status"] == "supported"
