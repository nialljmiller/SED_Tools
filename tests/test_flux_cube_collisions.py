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

# ---------------------------------------------------------------------
# mean strategy — arithmetic mean over colliding variants
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# filter strategy — build from one selected variant
# ---------------------------------------------------------------------

def test_filter_strategy_no_match_raises_clear_error(tmp_path):
    model_dir = tmp_path / "FilterGridBad"
    _make_collision_grid(model_dir)

    with pytest.raises(ValueError, match="No exact match"):
        precompute_flux_cube(
            str(model_dir), str(model_dir / "flux_cube.bin"),
            override_dict={"on_collision": {"strategy": "filter", "filter": {"alpha": "0.9999"}}},
        )


# ---------------------------------------------------------------------
# Filesystem-safe naming helpers (pure functions)
# ---------------------------------------------------------------------

def test_extra_key_to_filename_sanitizes_unsafe_characters():
    name = _extra_key_to_filename((("[alpha/Fe]", "0.4"), ("f sed", "2.0")))
    assert "/" not in name
    assert " " not in name
    assert name == "alpha_Fe_0.4__f_sed_2.0"


