"""Tests for two fixes that change results.

grid_densifier read and wrote flux_cube.bin with its axes reversed, so every
densified cube was scrambled. spectra_cleaner's bolometric renormalisation
(the fix for SVO NextGen fluxes about 100x too large) never ran.

The densifier tests run on the golden Kurucz2003all slice (Teff 5750-6250 K,
log g 4.0-4.5, [M/H] -0.5-0.0), whose spectra integrate to sigma*T^4.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from sed_tools._constants import SIGMA, planck_flam
from sed_tools.flux_cube_tool import FluxCube
from sed_tools.grid_densifier import densify_grid
from sed_tools.precompute_flux_cube import precompute_flux_cube

KURUCZ = Path(__file__).parent / "golden" / "inputs" / "stellar_models" / "Kurucz2003all"


# ---------------------------------------------------------------------
# grid_densifier: on-disk layout
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def kurucz_cubes(tmp_path_factory):
    """(source, densified) FluxCubes for the Kurucz slice at 125 K spacing."""
    work = tmp_path_factory.mktemp("densify")
    model = work / "Kurucz2003all"
    shutil.copytree(KURUCZ, model)
    precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    densify_grid(str(model / "flux_cube.bin"), str(work / "dense.bin"), teff_spacing=125.0)
    return FluxCube.from_file(str(model / "flux_cube.bin")), FluxCube.from_file(str(work / "dense.bin"))


def _integral_over_sigma_t4(cube, i_t, i_l, i_m):
    return np.trapezoid(cube.flux[i_t, i_l, i_m], cube.wavelengths) / (SIGMA * cube.teff_grid[i_t] ** 4)


def test_densified_grid_keeps_source_axes_and_adds_teff_nodes(kurucz_cubes):
    source, dense = kurucz_cubes
    np.testing.assert_array_equal(source.teff_grid, [5750.0, 6000.0, 6250.0])
    np.testing.assert_array_equal(dense.teff_grid, [5750.0, 5875.0, 6000.0, 6125.0, 6250.0])
    np.testing.assert_array_equal(dense.logg_grid, source.logg_grid)
    np.testing.assert_array_equal(dense.meta_grid, source.meta_grid)
    np.testing.assert_array_equal(dense.wavelengths, source.wavelengths)


def test_densified_cube_copies_source_spectra_exactly(kurucz_cubes):
    # Before the fix the copied rows differed from the source by up to 2.6e9.
    source, dense = kurucz_cubes
    for i_src, teff in enumerate(source.teff_grid):
        i_dense = int(np.flatnonzero(dense.teff_grid == teff)[0])
        np.testing.assert_array_equal(dense.flux[i_dense], source.flux[i_src])


def test_every_densified_node_integrates_to_sigma_t4(kurucz_cubes):
    # Before the fix every node integrated to 64-89 times sigma*T^4.
    source, dense = kurucz_cubes
    for cube in (source, dense):
        for i_t in range(cube.teff_grid.size):
            for i_l in range(cube.logg_grid.size):
                for i_m in range(cube.meta_grid.size):
                    ratio = _integral_over_sigma_t4(cube, i_t, i_l, i_m)
                    assert ratio == pytest.approx(1.0, rel=0.02), (cube.teff_grid[i_t], i_l, i_m)


def test_densified_cube_interpolates(kurucz_cubes):
    _, dense = kurucz_cubes
    wavelengths, flux = dense.interpolate_spectrum(5940.0, 4.2, -0.3)
    np.testing.assert_array_equal(wavelengths, dense.wavelengths)
    assert np.all(np.isfinite(flux))
    assert np.trapezoid(flux, wavelengths) / (SIGMA * 5940.0 ** 4) == pytest.approx(1.0, rel=0.02)
    # At a node the interpolation returns that node's spectrum.
    _, at_node = dense.interpolate_spectrum(5875.0, 4.5, 0.0)
    np.testing.assert_allclose(at_node, dense.flux[1, 1, 1], rtol=1e-12)


def test_inserted_teff_node_is_the_scaled_blackbody_between_its_neighbours(kurucz_cubes):
    """The default (non-ML) node is pi*B_lambda(T_new) scaled by the nearest node's integral.

    It is not between its neighbours at every wavelength: a smooth Planck curve
    sits far above line-blanketed Kurucz spectra in the UV and in strong lines.
    Its integral is between theirs.
    """
    source, dense = kurucz_cubes
    wl = dense.wavelengths
    for t_new in (5875.0, 6125.0):
        i_new = int(np.flatnonzero(dense.teff_grid == t_new)[0])
        i_near = int(np.argmin(np.abs(source.teff_grid - t_new)))
        i_lo = int(np.searchsorted(source.teff_grid, t_new)) - 1
        for i_l in range(dense.logg_grid.size):
            for i_m in range(dense.meta_grid.size):
                scale = (np.trapezoid(source.flux[i_near, i_l, i_m], wl)
                         / np.trapezoid(planck_flam(wl, source.teff_grid[i_near]), wl))
                np.testing.assert_allclose(dense.flux[i_new, i_l, i_m], planck_flam(wl, t_new) * scale,
                                           rtol=1e-12)
                lower = np.trapezoid(source.flux[i_lo, i_l, i_m], wl)
                upper = np.trapezoid(source.flux[i_lo + 1, i_l, i_m], wl)
                assert lower < np.trapezoid(dense.flux[i_new, i_l, i_m], wl) < upper


def test_densifier_keeps_every_node_in_place_on_an_uneven_grid(tmp_path, cube_writer):
    # Distinct axis lengths and a distinct value at every node, so any axis mix-up shows.
    teff, logg, meta = [4000.0, 6000.0], [3.0, 4.0, 5.0], [-1.0, 0.0]
    wl = np.linspace(3000.0, 9000.0, 7)
    nt, nl, nm, nw = len(teff), len(logg), len(meta), wl.size
    flux = 1.0 + np.arange(nt * nl * nm * nw, dtype=np.float64).reshape(nt, nl, nm, nw)
    cube_writer(tmp_path / "src.bin", teff, logg, meta, wl, flux)

    densify_grid(str(tmp_path / "src.bin"), str(tmp_path / "dense.bin"), teff_spacing=1000.0)

    dense = FluxCube.from_file(str(tmp_path / "dense.bin"))
    np.testing.assert_array_equal(dense.teff_grid, [4000.0, 5000.0, 6000.0])
    np.testing.assert_array_equal(dense.flux[0], flux[0])
    np.testing.assert_array_equal(dense.flux[2], flux[1])
