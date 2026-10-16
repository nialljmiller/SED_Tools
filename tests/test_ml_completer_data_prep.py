"""Tests for the torch-free surface of ``sed_tools.ml_sed_completer.SEDCompleter``.

``ml_sed_completer.py`` lazily imports ``torch`` inside individual
methods (network construction, ``train``, ``save``/``load``,
``complete_sed``) rather than at module level, specifically so the
module — and everything that doesn't need a trained network — stays
usable without torch installed. This file covers exactly that surface:
wavelength-grid construction, parameter normalization, flux-cube
loading (shared binary format with the rest of the package),
``_prepare_training_data``'s pure-numpy sampling/masking logic, and
model bookkeeping (``list_models``).

Network construction, training, inference, and save/load are out of
scope here — they require torch, which is intentionally not a hard
dependency of this package (see module docstring above and
CONTRIBUTING).
"""

import json

import numpy as np
import pytest

from sed_tools.ml_sed_completer import SEDCompleter


# ---------------------------------------------------------------------
# Import safety — must succeed with or without torch installed
# ---------------------------------------------------------------------

def test_module_imports_without_torch():
    import sed_tools.ml_sed_completer  # noqa: F401 — import success is the test


# ---------------------------------------------------------------------
# __init__ (untrained state)
# ---------------------------------------------------------------------

def test_init_without_model_path_is_torch_free_and_untrained():
    c = SEDCompleter()
    assert c.model is None
    assert c.config == {}
    assert c.wavelength_grid is None
    assert c.scaler_params is None


# ---------------------------------------------------------------------
# _create_wavelength_grid
# ---------------------------------------------------------------------

def test_default_wavelength_grid_is_log_spaced_100_to_100000():
    c = SEDCompleter()
    grid = c._create_wavelength_grid()
    assert grid.shape == (1000,)
    assert grid[0] == pytest.approx(100.0)
    assert grid[-1] == pytest.approx(100000.0)
    log_steps = np.diff(np.log10(grid))
    np.testing.assert_allclose(log_steps, log_steps[0], rtol=1e-10)


def test_custom_wavelength_grid_bounds_and_count():
    c = SEDCompleter()
    grid = c._create_wavelength_grid(min_wl=500.0, max_wl=5000.0, n_points=10)
    assert grid.shape == (10,)
    assert grid[0] == pytest.approx(500.0)
    assert grid[-1] == pytest.approx(5000.0)


# ---------------------------------------------------------------------
# _normalize_params
# ---------------------------------------------------------------------

def test_normalize_params_fallback_defaults_when_no_scaler():
    c = SEDCompleter()
    result = c._normalize_params(5000.0, 4.0, 0.0)
    expected = np.array([(5000.0 - 5000.0) / 10000.0, (4.0 - 3.0) / 3.0, 0.0 / 2.0])
    np.testing.assert_allclose(result, expected)


def test_normalize_params_uses_stored_scaler():
    c = SEDCompleter()
    c.scaler_params = {
        "teff_mean": 5500.0, "teff_std": 1000.0,
        "logg_mean": 4.0, "logg_std": 0.5,
        "meta_mean": 0.0, "meta_std": 0.3,
    }
    result = c._normalize_params(6500.0, 4.5, 0.3)
    np.testing.assert_allclose(result, [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------
# _load_flux_cube — shares the on-disk format used everywhere else
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

    c = SEDCompleter()
    teff_r, logg_r, meta_r, wl_r, flux_cube = c._load_flux_cube(str(lib_dir))

    np.testing.assert_array_equal(teff_r, teff)
    np.testing.assert_array_equal(logg_r, logg)
    np.testing.assert_array_equal(meta_r, meta)
    np.testing.assert_array_equal(wl_r, wl)
    # This reader intentionally does NOT transpose to (t,l,m,w) like
    # models.FluxCube — it keeps the raw on-disk (w,m,l,t) order, and
    # every caller in this module indexes it as flux_cube[:, m, l, t].
    assert flux_cube.shape == (wl.size, meta.size, logg.size, teff.size)
    for ti in range(teff.size):
        for li in range(logg.size):
            for mi in range(meta.size):
                np.testing.assert_array_equal(flux_cube[:, mi, li, ti], flux[ti, li, mi, :])


def test_load_flux_cube_missing_file_raises(tmp_path):
    c = SEDCompleter()
    with pytest.raises(FileNotFoundError, match="flux_cube.bin not found"):
        c._load_flux_cube(str(tmp_path / "nonexistent"))


# ---------------------------------------------------------------------
# _check_combined_grid / _build_coverage_map
#
# Both are currently dead code (never called from train() or anywhere
# else in this module — grep confirms no call sites), so this bug has
# no live effect today. Documented here so it's caught if/when someone
# wires these into the training pipeline.
# ---------------------------------------------------------------------

def test_check_combined_grid_false_when_no_lookup_table(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    assert SEDCompleter()._check_combined_grid(str(lib)) is False


@pytest.mark.xfail(
    reason=(
        "_check_combined_grid reads lookup_table.csv with "
        "pd.read_csv(..., comment='#'), but every lookup_table.csv this "
        "project writes (_lookup_io.write_lookup_csv) has a '#'-prefixed "
        "header line. comment='#' drops that entire line, so pandas uses "
        "the first DATA row as column names instead — 'source_model' can "
        "then never match. Currently dead code (no call sites), so no "
        "live impact, but the check itself never works as written."
    ),
    strict=True,
)
def test_check_combined_grid_detects_source_model_column(tmp_path):
    from sed_tools._lookup_io import write_lookup_csv

    lib = tmp_path / "lib"
    lib.mkdir()
    write_lookup_csv(
        {
            "file_name": ["a.txt", "b.txt"],
            "teff": [5000, 6000],
            "logg": [4.0, 4.5],
            "metallicity": [0.0, 0.2],
            "source_model": ["Kurucz", "BTSettl"],
        },
        str(lib / "lookup_table.csv"),
    )
    assert SEDCompleter()._check_combined_grid(str(lib)) is True


@pytest.mark.xfail(
    reason="Same pd.read_csv(comment='#') vs '#'-prefixed-header mismatch as "
    "_check_combined_grid above; _build_coverage_map returns {} for any "
    "real lookup_table.csv this project produces.",
    strict=True,
)
def test_build_coverage_map_reads_wl_bounds_per_node(tmp_path):
    from sed_tools._lookup_io import write_lookup_csv

    lib = tmp_path / "lib"
    lib.mkdir()
    write_lookup_csv(
        {
            "file_name": ["a.txt", "b.txt"],
            "teff": [5000, 6000],
            "logg": [4.0, 4.5],
            "metallicity": [0.0, 0.2],
            "wl_min": [3000, 4000],
            "wl_max": [9000, 8000],
        },
        str(lib / "lookup_table.csv"),
    )
    coverage = SEDCompleter()._build_coverage_map(
        str(lib), np.array([5000.0, 6000.0]), np.array([4.0, 4.5]), np.array([0.0, 0.2])
    )
    assert coverage == {(5000.0, 4.0, 0.0): (3000.0, 9000.0), (6000.0, 4.5, 0.2): (4000.0, 8000.0)}


def test_build_coverage_map_empty_when_no_lookup_table(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    coverage = SEDCompleter()._build_coverage_map(str(lib), np.array([5000.0]), np.array([4.0]), np.array([0.0]))
    assert coverage == {}


# ---------------------------------------------------------------------
# _prepare_training_data — pure numpy sampling/masking, no torch
# ---------------------------------------------------------------------

def _make_training_library(tmp_path, cube_writer, teff=(4000.0, 5000.0, 6000.0), logg=(4.0, 4.5), meta=(-0.5, 0.0)):
    teff = np.asarray(teff)
    logg = np.asarray(logg)
    meta = np.asarray(meta)
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)  # wide coverage
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
    np.random.seed(42)
    X, y, params, masks = SEDCompleter()._prepare_training_data(str(lib_dir), max_samples=5)

    assert X.shape == (5, 400)          # known_width = 1000 * 2 // 5
    assert y.shape == (5, 1000)         # full default wavelength grid
    assert params.shape == (5, 3)       # (teff, logg, meta)
    assert masks.shape == (5, 1000)
    assert np.all(np.isfinite(X))
    assert np.all(X > 0)
    assert np.all(np.isfinite(y))


def test_prepare_training_data_params_come_from_actual_grid(tmp_path, cube_writer):
    teff, logg, meta = (4000.0, 5000.0, 6000.0), (4.0, 4.5), (-0.5, 0.0)
    lib_dir = _make_training_library(tmp_path, cube_writer, teff, logg, meta)
    np.random.seed(1)
    _, _, params, _ = SEDCompleter()._prepare_training_data(str(lib_dir), max_samples=12)

    assert set(params[:, 0]).issubset(set(teff))
    assert set(params[:, 1]).issubset(set(logg))
    assert set(params[:, 2]).issubset(set(meta))


def test_prepare_training_data_is_reproducible_with_fixed_seed(tmp_path, cube_writer):
    lib_dir = _make_training_library(tmp_path, cube_writer)

    np.random.seed(42)
    X1, y1, params1, masks1 = SEDCompleter()._prepare_training_data(str(lib_dir), max_samples=5)
    np.random.seed(42)
    X2, y2, params2, masks2 = SEDCompleter()._prepare_training_data(str(lib_dir), max_samples=5)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(params1, params2)
    np.testing.assert_array_equal(masks1, masks2)


def test_prepare_training_data_raises_when_all_flux_invalid(tmp_path, cube_writer):
    teff = np.array([4000.0, 5000.0])
    logg = np.array([4.0])
    meta = np.array([0.0])
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
    flux = np.zeros((teff.size, logg.size, meta.size, wl.size))  # all zero -> invalid (not > 0)

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    cube_writer(lib_dir / "flux_cube.bin", teff, logg, meta, wl, flux)

    with pytest.raises(ValueError, match="No valid training samples"):
        SEDCompleter()._prepare_training_data(str(lib_dir))


def test_prepare_training_data_sets_known_region_attributes(tmp_path, cube_writer):
    lib_dir = _make_training_library(tmp_path, cube_writer)
    np.random.seed(0)
    c = SEDCompleter()
    c._prepare_training_data(str(lib_dir), max_samples=3)

    assert c._known_width == 400
    assert c._known_start == (1000 - 400) // 2
    assert c._known_end == c._known_start + 400
    assert c.wavelength_grid.shape == (1000,)


# ---------------------------------------------------------------------
# list_models — pure filesystem/JSON scan
# ---------------------------------------------------------------------

def test_list_models_empty_when_directory_missing(tmp_path):
    assert SEDCompleter.list_models(str(tmp_path / "nope")) == []


def test_list_models_skips_dirs_without_config(tmp_path):
    (tmp_path / "not_a_model").mkdir()
    assert SEDCompleter.list_models(str(tmp_path)) == []


def test_list_models_reads_config_fields(tmp_path):
    model_dir = tmp_path / "m1"
    model_dir.mkdir()
    (model_dir / SEDCompleter.CONFIG_FILE).write_text(
        json.dumps({"version": "2.0.0", "framework": "pytorch", "architecture": {"input_dim": 10}})
    )

    models = SEDCompleter.list_models(str(tmp_path))
    assert len(models) == 1
    assert models[0]["name"] == "m1"
    assert models[0]["version"] == "2.0.0"
    assert models[0]["framework"] == "pytorch"
    assert models[0]["architecture"] == {"input_dim": 10}


def test_list_models_does_not_filter_by_model_type(tmp_path):
    # Unlike SEDGenerator.list_models, SEDCompleter.list_models includes
    # any directory with a config.json, regardless of its content.
    model_dir = tmp_path / "anything"
    model_dir.mkdir()
    (model_dir / SEDCompleter.CONFIG_FILE).write_text(json.dumps({"model_type": "sed_generator"}))
    assert [m["name"] for m in SEDCompleter.list_models(str(tmp_path))] == ["anything"]


def test_list_models_skips_unreadable_config_without_raising(tmp_path):
    model_dir = tmp_path / "broken"
    model_dir.mkdir()
    (model_dir / SEDCompleter.CONFIG_FILE).write_text("{not valid json")
    assert SEDCompleter.list_models(str(tmp_path)) == []
