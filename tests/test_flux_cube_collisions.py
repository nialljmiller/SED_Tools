"""Integration tests for extra-axis collision handling in ``precompute_flux_cube``.

Covers the three on-collision strategies against small synthetic grids
where multiple spectra share the same (Teff, logg, [M/H]) node but
differ along an extra physical axis (e.g. alpha-enhancement):

* ``split`` (default) — one MESA-ready subgrid directory per variant
* ``mean``             — a single cube, arithmetic mean over variants
* ``filter``           — one cube built from a single selected variant

Also covers the pure filesystem-safe naming helpers used to build
subgrid/variant names, and a single-group case that isn't a real
collision at all.
"""

import csv

import numpy as np
import pytest

from sed_tools.models import SEDModel
from sed_tools.precompute_flux_cube import (
    _extra_key_to_filename,
    _subdir_name,
    _value_to_safe_str,
    precompute_flux_cube,
)
from sed_tools.spectra_cleaner import clean_model_dir

C_ANGSTROM_PER_S = 2.99792458e18


def _write_node(model_dir, fname, teff, logg, meta, alpha, flux_jy_value, wl_nm):
    lines = [
        "# source = synthetic\n",
        f"# teff = {teff}\n",
        f"# logg = {logg}\n",
        f"# metallicity = {meta}\n",
        f"# alpha = {alpha}\n",
        "# wavelength_unit = nm\n",
        "# flux in Jansky\n",
    ]
    with (model_dir / fname).open("w") as fh:
        fh.writelines(lines)
        for w in wl_nm:
            fh.write(f"{w:.6f} {flux_jy_value:.8e}\n")


def _make_collision_grid(model_dir):
    """2 Teff nodes x 2 alpha variants = 4 files, 2 real collisions."""
    model_dir.mkdir()
    wl_nm = np.linspace(400.0, 700.0, 8)
    # (teff, alpha) -> flux_jy, distinct per file so mislabeling is caught.
    _write_node(model_dir, "a1.txt", 5000.0, 4.0, 0.0, 0.0, 100.0, wl_nm)
    _write_node(model_dir, "a2.txt", 5000.0, 4.0, 0.0, 0.4, 300.0, wl_nm)
    _write_node(model_dir, "b1.txt", 6000.0, 4.0, 0.0, 0.0, 400.0, wl_nm)
    _write_node(model_dir, "b2.txt", 6000.0, 4.0, 0.0, 0.4, 600.0, wl_nm)
    clean_model_dir(str(model_dir))
    return wl_nm


# ---------------------------------------------------------------------
# split strategy (default) — per-variant MESA-ready subgrids
# ---------------------------------------------------------------------

def test_split_strategy_creates_one_subgrid_per_variant(tmp_path):
    model_dir = tmp_path / "CollisionGrid"
    _make_collision_grid(model_dir)

    precompute_flux_cube(str(model_dir), str(model_dir / "flux_cube.bin"))

    assert (model_dir / "CollisionGrid_alpha_0p00").is_dir()
    assert (model_dir / "CollisionGrid_alpha_0p40").is_dir()
    assert (model_dir / "CollisionGrid_alpha_0p00" / "flux_cube.bin").is_file()
    assert (model_dir / "CollisionGrid_alpha_0p40" / "flux_cube.bin").is_file()


def test_split_strategy_does_not_write_parent_flux_cube(tmp_path):
    # The parent directory is "not MESA-ready" by design when variants exist.
    model_dir = tmp_path / "CollisionGrid"
    _make_collision_grid(model_dir)

    precompute_flux_cube(str(model_dir), str(model_dir / "flux_cube.bin"))

    assert not (model_dir / "flux_cube.bin").exists()


def test_split_strategy_preserves_full_grid_lookup(tmp_path):
    model_dir = tmp_path / "CollisionGrid"
    _make_collision_grid(model_dir)
    precompute_flux_cube(str(model_dir), str(model_dir / "flux_cube.bin"))

    full = (model_dir / "lookup_table_full.csv").read_text()
    original = (model_dir / "lookup_table.csv").read_text()
    assert full == original
    assert "alpha" in original  # extra-axis column still present at parent level


def test_split_strategy_writes_deterministic_variants_index(tmp_path):
    model_dir = tmp_path / "CollisionGrid"
    _make_collision_grid(model_dir)
    precompute_flux_cube(str(model_dir), str(model_dir / "flux_cube.bin"))

    with (model_dir / "variants_index.csv").open() as fh:
        header = fh.readline().lstrip("#").strip().split(",")
        rows = list(csv.reader(fh))

    assert header == ["variant_name", "alpha", "n_spectra", "path"]
    assert [r[header.index("variant_name")] for r in rows] == [
        "CollisionGrid_alpha_0p00",
        "CollisionGrid_alpha_0p40",
    ]  # sorted / deterministic ordering
    assert all(r[header.index("n_spectra")] == "2" for r in rows)


def test_split_strategy_exact_variant_retrieval(tmp_path):
    model_dir = tmp_path / "CollisionGrid"
    wl_nm = _make_collision_grid(model_dir)
    precompute_flux_cube(str(model_dir), str(model_dir / "flux_cube.bin"))

    wl_ang = wl_nm * 10.0

    model_a0 = SEDModel(
        name="a0", flux_cube_path=model_dir / "CollisionGrid_alpha_0p00" / "flux_cube.bin"
    )
    model_a4 = SEDModel(
        name="a4", flux_cube_path=model_dir / "CollisionGrid_alpha_0p40" / "flux_cube.bin"
    )

    r_a0 = model_a0(teff=5000.0, logg=4.0, metallicity=0.0)
    r_a4 = model_a4(teff=5000.0, logg=4.0, metallicity=0.0)

    expected_a0 = (100.0 * 1e-23) * C_ANGSTROM_PER_S / (wl_ang ** 2)
    expected_a4 = (300.0 * 1e-23) * C_ANGSTROM_PER_S / (wl_ang ** 2)
    np.testing.assert_allclose(r_a0.flux, expected_a0, rtol=1e-6)
    np.testing.assert_allclose(r_a4.flux, expected_a4, rtol=1e-6)
    # The two variants must not be interchangeable. (atol=0: these fluxes
    # are ~1e-10, well under np.allclose's default atol=1e-8, which would
    # otherwise call any two such small arrays "close" regardless of rtol.)
    assert not np.allclose(r_a0.flux, r_a4.flux, atol=0)


def test_single_extra_axis_value_is_not_treated_as_collision(tmp_path):
    # All files share the same alpha value -> discover_extra_axes still
    # flags "alpha" as an extra column, but there's only one group, so
    # precompute_flux_cube must build directly with no subgrid split.
    model_dir = tmp_path / "NoRealCollision"
    model_dir.mkdir()
    wl_nm = np.linspace(400.0, 700.0, 8)
    _write_node(model_dir, "a.txt", 5000.0, 4.0, 0.0, 0.4, 100.0, wl_nm)
    _write_node(model_dir, "b.txt", 6000.0, 4.0, 0.0, 0.4, 200.0, wl_nm)
    clean_model_dir(str(model_dir))

    cube_path = model_dir / "flux_cube.bin"
    precompute_flux_cube(str(model_dir), str(cube_path))

    assert cube_path.exists()
    assert not any(p.is_dir() and p.name.startswith("NoRealCollision_") for p in model_dir.iterdir())


# ---------------------------------------------------------------------
# mean strategy — arithmetic mean over colliding variants
# ---------------------------------------------------------------------

def test_mean_strategy_equals_arithmetic_mean_of_variants(tmp_path):
    model_dir = tmp_path / "MeanGrid"
    wl_nm = _make_collision_grid(model_dir)  # flux_jy: 100/300 at 5000K, 400/600 at 6000K
    cube_path = model_dir / "flux_cube.bin"

    precompute_flux_cube(str(model_dir), str(cube_path), override_dict={"on_collision": {"strategy": "mean"}})

    assert cube_path.exists()
    model = SEDModel(name="mean", flux_cube_path=cube_path)
    wl_ang = wl_nm * 10.0

    r5000 = model(teff=5000.0, logg=4.0, metallicity=0.0)
    r6000 = model(teff=6000.0, logg=4.0, metallicity=0.0)

    expected_5000 = (200.0 * 1e-23) * C_ANGSTROM_PER_S / (wl_ang ** 2)  # mean(100, 300)
    expected_6000 = (500.0 * 1e-23) * C_ANGSTROM_PER_S / (wl_ang ** 2)  # mean(400, 600)
    np.testing.assert_allclose(r5000.flux, expected_5000, rtol=1e-6)
    np.testing.assert_allclose(r6000.flux, expected_6000, rtol=1e-6)


def test_mean_strategy_does_not_create_subgrids(tmp_path):
    model_dir = tmp_path / "MeanGrid"
    _make_collision_grid(model_dir)
    precompute_flux_cube(
        str(model_dir), str(model_dir / "flux_cube.bin"), override_dict={"on_collision": {"strategy": "mean"}}
    )
    assert not any(p.is_dir() and p.name.startswith("MeanGrid_") for p in model_dir.iterdir())


# ---------------------------------------------------------------------
# filter strategy — build from one selected variant
# ---------------------------------------------------------------------

def test_filter_strategy_exact_value_match(tmp_path):
    model_dir = tmp_path / "FilterGrid"
    wl_nm = _make_collision_grid(model_dir)
    cube_path = model_dir / "flux_cube.bin"

    precompute_flux_cube(
        str(model_dir), str(cube_path),
        override_dict={"on_collision": {"strategy": "filter", "filter": {"alpha": "0.4"}}},
    )

    model = SEDModel(name="filt", flux_cube_path=cube_path)
    wl_ang = wl_nm * 10.0
    result = model(teff=5000.0, logg=4.0, metallicity=0.0)
    expected = (300.0 * 1e-23) * C_ANGSTROM_PER_S / (wl_ang ** 2)  # alpha=0.4 variant, not alpha=0.0
    np.testing.assert_allclose(result.flux, expected, rtol=1e-6)


def test_filter_strategy_min_and_max_shortcuts(tmp_path):
    model_dir = tmp_path / "FilterGridMinMax"
    wl_nm = _make_collision_grid(model_dir)
    wl_ang = wl_nm * 10.0

    cube_min = model_dir / "cube_min.bin"
    precompute_flux_cube(
        str(model_dir), str(cube_min),
        override_dict={"on_collision": {"strategy": "filter", "filter": {"alpha": "min"}}},
    )
    model_min = SEDModel(name="min", flux_cube_path=cube_min)
    expected_min = (100.0 * 1e-23) * C_ANGSTROM_PER_S / (wl_ang ** 2)  # alpha=0.0 is min
    np.testing.assert_allclose(model_min(teff=5000.0, logg=4.0, metallicity=0.0).flux, expected_min, rtol=1e-6)

    cube_max = model_dir / "cube_max.bin"
    precompute_flux_cube(
        str(model_dir), str(cube_max),
        override_dict={"on_collision": {"strategy": "filter", "filter": {"alpha": "max"}}},
    )
    model_max = SEDModel(name="max", flux_cube_path=cube_max)
    expected_max = (300.0 * 1e-23) * C_ANGSTROM_PER_S / (wl_ang ** 2)  # alpha=0.4 is max
    np.testing.assert_allclose(model_max(teff=5000.0, logg=4.0, metallicity=0.0).flux, expected_max, rtol=1e-6)


def test_filter_strategy_no_match_raises_clear_error(tmp_path):
    model_dir = tmp_path / "FilterGridBad"
    _make_collision_grid(model_dir)

    with pytest.raises(ValueError, match="No exact match"):
        precompute_flux_cube(
            str(model_dir), str(model_dir / "flux_cube.bin"),
            override_dict={"on_collision": {"strategy": "filter", "filter": {"alpha": "0.9999"}}},
        )


def test_filter_strategy_unmatched_key_raises_clear_error(tmp_path):
    model_dir = tmp_path / "FilterGridBadKey"
    _make_collision_grid(model_dir)

    with pytest.raises(ValueError, match="does not match any extra axis"):
        precompute_flux_cube(
            str(model_dir), str(model_dir / "flux_cube.bin"),
            override_dict={"on_collision": {"strategy": "filter", "filter": {"bogus_axis": "0.4"}}},
        )


# ---------------------------------------------------------------------
# Filesystem-safe naming helpers (pure functions)
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", "0p00"),
        ("0.01", "0p01"),
        ("0.1", "0p10"),
        ("0.25", "0p25"),
        ("-0.5", "m0p50"),
        ("+0.5", "0p50"),
        ("h-rich", "h_rich"),
        ("h rich!", "h_rich"),
    ],
)
def test_value_to_safe_str(raw, expected):
    assert _value_to_safe_str(raw) == expected


def test_subdir_name_single_and_multi_axis():
    assert _subdir_name("Husfeld", (("yhe", "0.10"),)) == "Husfeld_yhe_0p10"
    assert _subdir_name("Husfeld", (("alpha", "0.4"), ("yhe", "0.10"))) == "Husfeld_alpha_0p40_yhe_0p10"
    assert _subdir_name("Model", ()) == "Model"


def test_extra_key_to_filename_sanitizes_unsafe_characters():
    name = _extra_key_to_filename((("[alpha/Fe]", "0.4"), ("f sed", "2.0")))
    assert "/" not in name
    assert " " not in name
    assert name == "alpha_Fe_0.4__f_sed_2.0"


def test_extra_key_to_filename_empty_key_returns_default():
    assert _extra_key_to_filename(()) == "default"
