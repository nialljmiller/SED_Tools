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


# ---------------------------------------------------------------------
# spectra_cleaner: bolometric renormalisation
# ---------------------------------------------------------------------

def _write_spectrum(path, teff, wl, flux):
    with path.open("w") as fh:
        fh.write(f"# teff = {teff}\n# logg = 4.0\n# metallicity = 0.0\n"
                 "# wavelength_unit = Angstrom\n# flux_unit = erg/cm2/s/A\n")
        for w, f in zip(wl, flux):
            fh.write(f"{w:.6f} {f:.8e}\n")


def _read(path):
    from sed_tools.spectrum_io import read_text_spectrum

    return read_text_spectrum(str(path))


def test_spectrum_100x_too_bright_is_renormalised_to_sigma_t4(tmp_path):
    # SVO's NextGen T >= 5000 K entries are about 100 times too bright.
    from sed_tools.spectra_cleaner import clean_spectrum_file

    wl = np.geomspace(100.0, 1e6, 3000)
    path = tmp_path / "nextgen_like.txt"
    _write_spectrum(path, 6000.0, wl, 100.0 * planck_flam(wl, 6000.0))

    status, detail = clean_spectrum_file(str(path))

    assert status == "converted_renormed"
    assert "flux divided by 100 " in detail
    wl_after, flux_after = _read(path)
    assert np.trapezoid(flux_after, wl_after) / (SIGMA * 6000.0 ** 4) == pytest.approx(1.0, rel=0.01)


def test_cool_giant_far_below_the_blackbody_in_the_optical_is_left_alone(tmp_path):
    # Granada HRES PHOENIX giants at 3000-3500 K, log g <= 1, have 2-10 % of a blackbody's
    # flux over 3000-7000 A (TiO), which holds 8-15 % of sigma*T^4. They are correct; the
    # first version, which also rescaled ratios below 0.1, multiplied them by 10-50.
    from sed_tools.spectra_cleaner import clean_spectrum_file

    wl = np.linspace(3000.0, 7000.0, 4000)
    path = tmp_path / "m_giant.txt"
    _write_spectrum(path, 3000.0, wl, 0.03 * planck_flam(wl, 3000.0))
    _, written_before = _read(path)

    status, _ = clean_spectrum_file(str(path))

    assert status == "converted"
    np.testing.assert_array_equal(_read(path)[1], written_before)


@pytest.mark.parametrize("teff", [20000.0, 30000.0, 37000.0])
def test_correct_hot_blackbody_on_a_wide_grid_is_left_alone(tmp_path, teff):
    # The first version integrated the blackbody on a 2000-point linear grid. Over
    # 10 A-1e7 A that misses a hot star's peak, so these correctly scaled spectra
    # came out 4-11 times too bright and would have been divided by that.
    from sed_tools.spectra_cleaner import clean_spectrum_file

    wl = np.geomspace(10.0, 1e7, 5000)
    flux = planck_flam(wl, teff)
    path = tmp_path / "hot.txt"
    _write_spectrum(path, teff, wl, flux)
    _, written_before = _read(path)

    status, _ = clean_spectrum_file(str(path))

    assert status == "converted"
    np.testing.assert_array_equal(_read(path)[1], written_before)


def test_pi_less_blackbody_is_within_the_thresholds(tmp_path):
    # B_lambda instead of pi*B_lambda (MESA's bbody grid) is 1/pi of sigma*T^4: not rescaled.
    from sed_tools.spectra_cleaner import clean_spectrum_file

    wl = np.geomspace(100.0, 1e6, 3000)
    path = tmp_path / "bbody.txt"
    _write_spectrum(path, 5000.0, wl, planck_flam(wl, 5000.0) / np.pi)
    assert clean_spectrum_file(str(path))[0] == "converted"


def test_spectrum_covering_little_of_sigma_t4_is_not_checked(tmp_path):
    # 3000-9000 A holds about 3 % of a 50000 K blackbody's flux: too little to judge the scale.
    from sed_tools.spectra_cleaner import clean_spectrum_file

    wl = np.linspace(3000.0, 9000.0, 500)
    path = tmp_path / "optical_only.txt"
    _write_spectrum(path, 50000.0, wl, 100.0 * planck_flam(wl, 50000.0))
    assert clean_spectrum_file(str(path))[0] == "converted"


def test_clean_model_dir_counts_renormed_spectra_as_converted_and_warns(tmp_path, caplog):
    import logging

    from sed_tools.spectra_cleaner import clean_model_dir

    wl = np.geomspace(100.0, 1e6, 2000)
    grid = tmp_path / "grid"
    grid.mkdir()
    _write_spectrum(grid / "a.txt", 5000.0, wl, 100.0 * planck_flam(wl, 5000.0))
    _write_spectrum(grid / "b.txt", 6000.0, wl, 120.0 * planck_flam(wl, 6000.0))
    _write_spectrum(grid / "c.txt", 7000.0, wl, planck_flam(wl, 7000.0))

    with caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = clean_model_dir(str(grid), try_h5_recovery=False, sample_fraction=1.0)

    assert summary["converted"] == ["a.txt", "b.txt", "c.txt"]
    assert summary["error"] == []
    assert sorted(summary["renormed"]) == ["a.txt", "b.txt"]
    assert summary["renormed"]["a.txt"] == pytest.approx(100.0)
    assert summary["renormed"]["b.txt"] == pytest.approx(120.0)
    warning = [r.getMessage() for r in caplog.records if "bolometric renormalisation" in r.getMessage()]
    assert len(warning) == 1 and "2 spectra integrated to 100-120 times" in warning[0]
    for name, teff in (("a.txt", 5000.0), ("b.txt", 6000.0), ("c.txt", 7000.0)):
        wl_after, flux_after = _read(grid / name)
        assert np.trapezoid(flux_after, wl_after) / (SIGMA * teff ** 4) == pytest.approx(1.0, rel=0.01)


def test_unstandardized_kurucz_slice_is_converted_without_renormalisation(tmp_path, caplog):
    # The golden clean_model_dir case only skips these files (they are marked standardized),
    # so it does not show the check leaves correctly scaled Kurucz spectra alone. This does.
    import logging

    from sed_tools.spectra_cleaner import clean_model_dir

    grid = shutil.copytree(KURUCZ, tmp_path / "Kurucz2003all")
    before = {}
    for path in sorted(grid.glob("*.txt")):
        path.write_text("".join(line for line in path.read_text().splitlines(keepends=True)
                                if not line.startswith("# units_standardized")))
        before[path.name] = _read(path)

    with caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = clean_model_dir(str(grid), try_h5_recovery=False, sample_fraction=1.0)

    assert len(summary["converted"]) == 12
    assert "renormed" not in summary
    assert not any("renormalisation" in r.getMessage() for r in caplog.records)
    for name, (wl, flux) in before.items():
        wl_after, flux_after = _read(grid / name)
        np.testing.assert_array_equal(wl_after, wl)
        np.testing.assert_array_equal(flux_after, flux)
