"""Tests for the torch-free surface of ``sed_tools.ml_sed_generator.SEDGenerator``.

Mirrors ``test_ml_completer_data_prep.py``: ``ml_sed_generator.py`` also
lazily imports torch only inside individual methods, so everything
here — wavelength grid, flux-cube loading, ``_prepare_training_data``,
and ``list_models`` — is importable and runnable without torch.
"""

import json

import numpy as np
import pytest

from sed_tools.ml_sed_generator import SEDGenerator


def test_module_imports_without_torch():
    import sed_tools.ml_sed_generator  # noqa: F401


def test_init_without_model_path_is_torch_free_and_untrained():
    g = SEDGenerator()
    assert g.model is None
    assert g.config == {}
    assert g.wavelength_grid is None
    assert g.scaler_params is None


def test_default_wavelength_grid_is_log_spaced_100_to_100000():
    g = SEDGenerator()
    grid = g._create_wavelength_grid()
    assert grid.shape == (1000,)
    assert grid[0] == pytest.approx(100.0)
    assert grid[-1] == pytest.approx(100000.0)


# ---------------------------------------------------------------------
# _load_flux_cube
# ---------------------------------------------------------------------

def test_load_flux_cube_reads_grids_and_indexes_wmlt_order(tmp_path, cube_writer):
    teff = np.array([4000.0, 5000.0, 6000.0])
    logg = np.array([4.0, 4.5])
    meta = np.array([-0.5, 0.0])
    wl = np.array([4000.0, 5000.0, 6000.0, 7000.0])
    flux = np.zeros((teff.size, logg.size, meta.size, wl.size))
    for ti in range(teff.size):
        for li in range(logg.size):
            for mi in range(meta.size):
                flux[ti, li, mi, :] = ti * 100 + li * 10 + mi + np.arange(wl.size)

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    cube_writer(lib_dir / "flux_cube.bin", teff, logg, meta, wl, flux)

    g = SEDGenerator()
    teff_r, logg_r, meta_r, wl_r, flux_cube = g._load_flux_cube(str(lib_dir))

    np.testing.assert_array_equal(teff_r, teff)
    assert flux_cube.shape == (wl.size, meta.size, logg.size, teff.size)
    for ti in range(teff.size):
        for li in range(logg.size):
            for mi in range(meta.size):
                np.testing.assert_array_equal(flux_cube[:, mi, li, ti], flux[ti, li, mi, :])


def test_load_flux_cube_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="flux_cube.bin not found"):
        SEDGenerator()._load_flux_cube(str(tmp_path / "nonexistent"))


# ---------------------------------------------------------------------
# _prepare_training_data
# ---------------------------------------------------------------------

def _make_training_library(tmp_path, cube_writer, teff=(4000.0, 5000.0, 6000.0), logg=(4.0, 4.5), meta=(-0.5, 0.0)):
    teff = np.asarray(teff)
    logg = np.asarray(logg)
    meta = np.asarray(meta)
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
    flux = np.zeros((teff.size, logg.size, meta.size, wl.size))
    for ti in range(teff.size):
        for li in range(logg.size):
            for mi in range(meta.size):
                flux[ti, li, mi, :] = 1e-10 * (1 + 0.1 * ti + 0.01 * li)
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    cube_writer(lib_dir / "flux_cube.bin", teff, logg, meta, wl, flux)
    return lib_dir


def test_prepare_training_data_shapes(tmp_path, cube_writer):
    lib_dir = _make_training_library(tmp_path, cube_writer)
    np.random.seed(7)
    params, flux, masks = SEDGenerator()._prepare_training_data(str(lib_dir), max_samples=5)

    assert params.shape == (5, 3)
    assert flux.shape == (5, 1000)
    assert masks.shape == (5, 1000)
    assert np.all(np.isfinite(flux))


def test_prepare_training_data_params_come_from_actual_grid(tmp_path, cube_writer):
    teff, logg, meta = (4000.0, 5000.0, 6000.0), (4.0, 4.5), (-0.5, 0.0)
    lib_dir = _make_training_library(tmp_path, cube_writer, teff, logg, meta)
    np.random.seed(1)
    params, _, _ = SEDGenerator()._prepare_training_data(str(lib_dir), max_samples=12)

    assert set(params[:, 0]).issubset(set(teff))
    assert set(params[:, 1]).issubset(set(logg))
    assert set(params[:, 2]).issubset(set(meta))


def test_prepare_training_data_is_reproducible_with_fixed_seed(tmp_path, cube_writer):
    lib_dir = _make_training_library(tmp_path, cube_writer)

    np.random.seed(7)
    p1, f1, m1 = SEDGenerator()._prepare_training_data(str(lib_dir), max_samples=5)
    np.random.seed(7)
    p2, f2, m2 = SEDGenerator()._prepare_training_data(str(lib_dir), max_samples=5)

    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(m1, m2)


def test_prepare_training_data_raises_when_all_flux_invalid(tmp_path, cube_writer):
    teff = np.array([4000.0, 5000.0])
    logg = np.array([4.0])
    meta = np.array([0.0])
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
    flux = np.zeros((teff.size, logg.size, meta.size, wl.size))  # invalid: not > 0

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    cube_writer(lib_dir / "flux_cube.bin", teff, logg, meta, wl, flux)

    with pytest.raises(ValueError, match="No valid training samples"):
        SEDGenerator()._prepare_training_data(str(lib_dir))


def test_prepare_training_data_skips_a_spectrum_with_any_nonpositive_flux(tmp_path, cube_writer):
    # Unlike SEDCompleter (which extracts a partial "known region"),
    # SEDGenerator requires the entire resampled spectrum to be usable
    # and drops any sample containing a non-finite/non-positive point
    # anywhere (see: `if np.any(flux <= 0) or np.any(~np.isfinite(flux))`).
    teff = np.array([4000.0, 5000.0])
    logg = np.array([4.0])
    meta = np.array([0.0])
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
    flux = np.full((teff.size, logg.size, meta.size, wl.size), 1e-10)
    flux[0, 0, 0, 5] = -1.0  # poison the first node only

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    cube_writer(lib_dir / "flux_cube.bin", teff, logg, meta, wl, flux)

    params, _, _ = SEDGenerator()._prepare_training_data(str(lib_dir), max_samples=2)
    assert params.shape[0] == 1
    assert params[0, 0] == 5000.0  # only the clean (5000K) node survives


# ---------------------------------------------------------------------
# list_models — filters by model_type, unlike SEDCompleter.list_models
# ---------------------------------------------------------------------

def test_list_models_empty_when_directory_missing(tmp_path):
    assert SEDGenerator.list_models(str(tmp_path / "nope")) == []


def test_list_models_includes_sed_generator_configs(tmp_path):
    model_dir = tmp_path / "gen1"
    model_dir.mkdir()
    (model_dir / SEDGenerator.CONFIG_FILE).write_text(
        json.dumps({"model_type": "sed_generator", "parameter_ranges": {"teff": [4000, 6000]}, "architecture": {}})
    )
    models = SEDGenerator.list_models(str(tmp_path))
    assert [m["name"] for m in models] == ["gen1"]
    assert models[0]["parameter_ranges"] == {"teff": [4000, 6000]}


def test_list_models_excludes_configs_without_matching_model_type(tmp_path):
    # e.g. a SEDCompleter model directory sitting in the same models/ dir.
    model_dir = tmp_path / "completer1"
    model_dir.mkdir()
    (model_dir / SEDGenerator.CONFIG_FILE).write_text(json.dumps({"framework": "pytorch", "architecture": {}}))
    assert SEDGenerator.list_models(str(tmp_path)) == []


def test_list_models_skips_unreadable_config_without_raising(tmp_path):
    model_dir = tmp_path / "broken"
    model_dir.mkdir()
    (model_dir / SEDGenerator.CONFIG_FILE).write_text("{not valid json")
    assert SEDGenerator.list_models(str(tmp_path)) == []
