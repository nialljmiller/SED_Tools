import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sed_tools
from sed_tools.models import SED as _SEDCore
from sed_tools._flux import AB_ZERO_FLUX


def _write_flux_cube(path, teff, logg, meta, wavelengths, flux=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    teff = np.asarray(teff, dtype=np.float64)
    logg = np.asarray(logg, dtype=np.float64)
    meta = np.asarray(meta, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if flux is None:
        flux = np.ones((teff.size, logg.size, meta.size, wavelengths.size))
    with path.open("wb") as fh:
        fh.write(struct.pack("4i", teff.size, logg.size, meta.size, wavelengths.size))
        for grid in (teff, logg, meta, wavelengths):
            grid.tofile(fh)
        flux.astype(np.float64).transpose(3, 2, 1, 0).ravel().tofile(fh)


def _write_filter(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# wavelength_unit = angstrom\n4000 1.0\n5000 1.0\n")


# --- import / version ---

def test_import():
    assert hasattr(sed_tools, "__version__")


def test_version_string():
    parts = sed_tools.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts)


# --- find_atm / find_atmospheres ---

def test_find_atm_alias(tmp_path):
    cube = tmp_path / "demo" / "flux_cube.bin"
    _write_flux_cube(cube, [4000., 6000.], [3.0, 5.0], [-1.0, 0.0], [4000., 5000.])
    matches = sed_tools.find_atm(
        teff_range=(4500., 5500.),
        logg_range=(3.5, 4.5),
        Z_range=(-0.5, 0.0),  # within cube meta range [-1, 0]
        model_root=tmp_path,
    )
    assert len(matches) >= 1


def test_find_atmospheres_returns_model_match(tmp_path):
    cube = tmp_path / "m1" / "flux_cube.bin"
    _write_flux_cube(cube, [4000., 6000.], [3.0, 5.0], [-1.0, 0.0], [4000., 5000.])
    sed = _SEDCore(model_root=tmp_path)
    matches = sed.find_atmospheres(
        teff_range=(4500., 5500.),
        logg_range=(3.5, 4.5),
        metallicity_range=(-0.5, 0.0),
    )
    assert len(matches) >= 1
    m = matches[0]
    assert hasattr(m, "name")
    assert hasattr(m, "contains_point")
    assert hasattr(m, "teff_range")
    assert hasattr(m, "logg_range")
    assert hasattr(m, "metallicity_range")


def test_find_atmospheres_no_match(tmp_path):
    cube = tmp_path / "m1" / "flux_cube.bin"
    _write_flux_cube(cube, [4000., 5000.], [3.0, 4.0], [-1.0, 0.0], [4000., 5000.])
    sed = _SEDCore(model_root=tmp_path)
    matches = sed.find_atmospheres(
        teff_range=(50000., 60000.),
        logg_range=(3.5, 4.5),
        metallicity_range=(-0.5, 0.0),
    )
    assert matches == []


# --- interpolation ---

def test_interpolation_returns_correct_shape(tmp_path):
    flux_value = 3.0
    teff = np.array([1000., 2000.])
    logg = np.array([1.0, 2.0])
    meta = np.array([-1.0, 0.0])
    wl   = np.array([4000., 5000.])
    flux = np.full((teff.size, logg.size, meta.size, wl.size), flux_value)

    cube = tmp_path / "demo" / "flux_cube.bin"
    _write_flux_cube(cube, teff, logg, meta, wl, flux)

    sed = _SEDCore(model_root=tmp_path)
    atm = sed.find_atmospheres(
        teff_range=(1000., 2000.),
        logg_range=(1.0, 2.0),
        metallicity_range=(-1.0, 0.0),
    )
    model = sed.model(atm[0])
    result = model(teff=1500., logg=1.5, metallicity=-0.5)
    assert result.wavelength.shape == wl.shape
    assert result.flux.shape == wl.shape


def test_interpolation_flat_field(tmp_path):
    flux_value = 2.0
    teff = np.array([1000., 2000.])
    logg = np.array([1.0, 2.0])
    meta = np.array([-1.0, 0.0])
    wl   = np.array([4000., 5000.])
    flux = np.full((teff.size, logg.size, meta.size, wl.size), flux_value)

    cube = tmp_path / "demo" / "flux_cube.bin"
    _write_flux_cube(cube, teff, logg, meta, wl, flux)

    sed = _SEDCore(model_root=tmp_path)
    atm = sed.find_atmospheres(
        teff_range=(1000., 2000.),
        logg_range=(1.0, 2.0),
        metallicity_range=(-1.0, 0.0),
    )
    model = sed.model(atm[0])
    result = model(teff=1500., logg=1.5, metallicity=-0.5)
    assert np.allclose(result.flux, flux_value)


# --- photometry ---

def test_photometry_ab_magnitude_finite(tmp_path):
    flux_value = 2.0
    teff = np.array([1000., 2000.])
    logg = np.array([1.0, 2.0])
    meta = np.array([-1.0, 0.0])
    wl   = np.array([4000., 5000.])
    flux = np.full((teff.size, logg.size, meta.size, wl.size), flux_value)

    cube = tmp_path / "demo" / "flux_cube.bin"
    _write_flux_cube(cube, teff, logg, meta, wl, flux)

    filter_file = tmp_path / "filters" / "GAIA" / "GAIA.dat"
    _write_filter(filter_file)

    sed = _SEDCore(model_root=tmp_path, filter_root=tmp_path / "filters")
    atm = sed.find_atmospheres(
        teff_range=(1000., 2000.),
        logg_range=(1.0, 2.0),
        metallicity_range=(-1.0, 0.0),
    )
    model = sed.model(atm[0])
    result = model(teff=1500., logg=1.5, metallicity=-0.5)
    phot = result.photometry("Gaia")
    assert "GAIA" in phot
    assert math.isfinite(phot["GAIA"].magnitude)
    assert phot["GAIA"].system == "AB"


def test_photometry_magnitude_formula(tmp_path):
    flux_value = 2.0
    teff = np.array([1000., 2000.])
    logg = np.array([1.0, 2.0])
    meta = np.array([-1.0, 0.0])
    wl   = np.array([4000., 5000.])
    flux = np.full((teff.size, logg.size, meta.size, wl.size), flux_value)

    cube = tmp_path / "demo" / "flux_cube.bin"
    _write_flux_cube(cube, teff, logg, meta, wl, flux)

    filter_file = tmp_path / "filters" / "GAIA" / "GAIA.dat"
    _write_filter(filter_file)

    sed = _SEDCore(model_root=tmp_path, filter_root=tmp_path / "filters")
    atm = sed.find_atmospheres(
        teff_range=(1000., 2000.),
        logg_range=(1.0, 2.0),
        metallicity_range=(-1.0, 0.0),
    )
    model = sed.model(atm[0])
    result = model(teff=1500., logg=1.5, metallicity=-0.5)
    phot = result.photometry("Gaia")
    g = phot["GAIA"]
    expected = -2.5 * math.log10(g.flux_density / AB_ZERO_FLUX)
    assert math.isclose(g.magnitude, expected)


# --- EvaluatedSED properties ---

def test_evaluated_sed_property_aliases(tmp_path):
    teff = np.array([1000., 2000.])
    logg = np.array([1.0, 2.0])
    meta = np.array([-1.0, 0.0])
    wl   = np.array([4000., 5000.])
    flux = np.ones((teff.size, logg.size, meta.size, wl.size))

    cube = tmp_path / "demo" / "flux_cube.bin"
    _write_flux_cube(cube, teff, logg, meta, wl, flux)

    sed = _SEDCore(model_root=tmp_path)
    atm = sed.find_atmospheres(
        teff_range=(1000., 2000.),
        logg_range=(1.0, 2.0),
        metallicity_range=(-1.0, 0.0),
    )
    model = sed.model(atm[0])
    result = model(teff=1500., logg=1.5, metallicity=-0.5)

    assert np.array_equal(result.wl, result.wavelength)
    assert np.array_equal(result.fl, result.flux)
    assert math.isclose(result.teff, 1500.)
    assert math.isclose(result.logg, 1.5)
    assert math.isclose(result.metallicity, -0.5)
