"""The grid densifier: its cubes keep the source axes and spectra, integrate to sigma*T^4,
interpolate, and do not depend on the wavelength block size; it stops when ML generation fails, and its
menu refuses a choice out of range.

The Kurucz tests run on the golden Kurucz2003all slice (Teff 5750-6250 K,
log g 4.0-4.5, [M/H] -0.5-0.0), whose spectra integrate to sigma*T^4.
"""

import shutil
import sys

import numpy as np
import pytest

from sed_tools._constants import SIGMA, planck_flam
from sed_tools._resample import resample_to_grid
from sed_tools.flux_cube_tool import FluxCube
from sed_tools.grid_densifier import densify_grid
from sed_tools.precompute_flux_cube import precompute_flux_cube

from helpers_cube import KURUCZ


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
# grid_densifier
# ---------------------------------------------------------------------
def _write_small_cube(path, teff=(5000.0, 7000.0), nw=40):
    import struct

    import numpy as np

    teff = np.asarray(teff, dtype=np.float64)
    logg, meta = np.array([4.0]), np.array([0.0])
    wl = np.linspace(3000.0, 10000.0, nw)
    flux = np.ones((len(teff), 1, 1, nw))
    with open(path, "wb") as fh:
        fh.write(struct.pack("4i", len(teff), 1, 1, nw))
        for arr in (teff, logg, meta, wl, flux):
            fh.write(arr.astype(np.float64).tobytes())


def test_densify_stops_when_ml_generation_fails(tmp_path, monkeypatch):
    # Used to log a warning and write a blackbody in place of the ML SED.
    import sys
    import types

    from sed_tools.grid_densifier import densify_grid

    class FailingGenerator:
        def __init__(self, path):
            pass

        def generate(self, **kwargs):
            raise ValueError("network produced NaN")

    fake = types.ModuleType("sed_tools.ml_sed_generator")
    fake.SEDGenerator = FailingGenerator
    monkeypatch.setitem(sys.modules, "sed_tools.ml.generator", fake)

    src = tmp_path / "src.bin"
    _write_small_cube(src)
    with pytest.raises(RuntimeError, match=r"Teff=6000 logg=4\.00 meta=0\.00.*network produced NaN"):
        densify_grid(str(src), str(tmp_path / "dense.bin"), teff_spacing=1000.0, ml_model="fake")


@pytest.mark.parametrize("answer", ["0", "-1", "3", "x"])
def test_densifier_menu_rejects_out_of_range_choice(tmp_path, monkeypatch, capsys, answer):
    # "0" and negative numbers used to pick cubes from the end of the list.
    import sed_tools.grid_densifier as gd

    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        _write_small_cube(tmp_path / name / "flux_cube.bin")
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    monkeypatch.setattr(gd, "densify_grid", lambda **kw: pytest.fail("must not run"))
    gd.run_interactive_workflow(str(tmp_path))
    assert "Invalid selection." in capsys.readouterr().out


# ---------------------------------------------------------------------
# Densifier: written one block of wavelengths at a time, same bytes
# ---------------------------------------------------------------------
def _uneven_source_cube(path, cube_writer):
    """3 Teff x 3 logg x 2 [M/H] x 30 wavelengths, a different spectrum at every node."""
    teff, logg, meta = [4000.0, 6000.0, 9000.0], [3.0, 4.0, 5.0], [-1.0, 0.0]
    wl = np.geomspace(1000.0, 50000.0, 30)
    rng = np.random.default_rng(3)
    flux = np.empty((3, 3, 2, 30))
    for i_t, t in enumerate(teff):
        flux[i_t] = planck_flam(wl, t) * rng.uniform(0.5, 1.5, size=(3, 2, 30))
    cube_writer(path, teff, logg, meta, wl, flux)


def _densify(monkeypatch, tmp_path, name, ram, **kwargs):
    """Densify src.bin into <name>.bin as if `ram` bytes of RAM were available; return its bytes."""
    import sed_tools.grid_densifier as gd

    monkeypatch.setattr(gd, "_available_ram_bytes", lambda: ram)
    gd.densify_grid(str(tmp_path / "src.bin"), str(tmp_path / f"{name}.bin"), teff_spacing=1000.0, **kwargs)
    return (tmp_path / f"{name}.bin").read_bytes()


# 6 dense Teffs: each wavelength holds (3 + 6) x 3 x 2 doubles = 432 bytes, of which a
# block may use half the RAM, so 6048 bytes gives blocks of 7 wavelengths: 7, 7, 7, 7, 2.
SEVEN_WAVELENGTHS = 2 * 7 * 432


def test_densified_cube_does_not_depend_on_the_block_size(tmp_path, cube_writer, monkeypatch, capsys):
    _uneven_source_cube(tmp_path / "src.bin", cube_writer)

    one_block = _densify(monkeypatch, tmp_path, "one", 2**40)
    assert "in 1 wavelength block(s)" in capsys.readouterr().out
    assert _densify(monkeypatch, tmp_path, "five", SEVEN_WAVELENGTHS) == one_block
    assert "in 5 wavelength block(s)" in capsys.readouterr().out
    assert _densify(monkeypatch, tmp_path, "thirty", 1) == one_block
    assert "in 30 wavelength block(s)" in capsys.readouterr().out

    source = FluxCube.from_file(str(tmp_path / "src.bin"))
    dense = FluxCube.from_file(str(tmp_path / "five.bin"))
    np.testing.assert_array_equal(dense.teff_grid, [4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0])
    np.testing.assert_array_equal(dense.flux[[0, 2, 5]], source.flux)


def test_densified_ml_nodes_do_not_depend_on_the_block_size(tmp_path, cube_writer, monkeypatch):
    import types

    model_wl = np.geomspace(900.0, 60000.0, 200)

    class PowerLawGenerator:
        def __init__(self, path):
            pass

        def generate(self, teff, logg, meta, check_bounds=True):
            return model_wl.copy(), 1e-3 * teff * (model_wl / 5000.0) ** (-2.0 - 0.1 * logg + meta)

    fake = types.ModuleType("sed_tools.ml_sed_generator")
    fake.SEDGenerator = PowerLawGenerator
    monkeypatch.setitem(sys.modules, "sed_tools.ml.generator", fake)
    _uneven_source_cube(tmp_path / "src.bin", cube_writer)

    one_block = _densify(monkeypatch, tmp_path, "one", 2**40, ml_model="fake")
    assert _densify(monkeypatch, tmp_path, "five", SEVEN_WAVELENGTHS, ml_model="fake") == one_block

    dense = FluxCube.from_file(str(tmp_path / "five.bin"))
    for i_t, t in ((1, 5000.0), (3, 7000.0), (4, 8000.0)):
        for i_l, logg in enumerate(dense.logg_grid):
            for i_m, meta in enumerate(dense.meta_grid):
                expected = resample_to_grid(*PowerLawGenerator(None).generate(t, logg, meta), t, dense.wavelengths)
                np.testing.assert_array_equal(dense.flux[i_t, i_l, i_m], expected)
