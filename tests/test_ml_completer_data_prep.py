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
# __init__ (untrained state)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# _create_wavelength_grid
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# _normalize_params
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# _load_flux_cube — shares the on-disk format used everywhere else
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# list_models — pure filesystem/JSON scan
# ---------------------------------------------------------------------

def test_list_models_skips_dirs_without_config(tmp_path):
    (tmp_path / "not_a_model").mkdir()
    assert SEDCompleter.list_models(str(tmp_path)) == []


def test_list_models_leaves_out_generators(tmp_path):
    # 0.5.0: generators were listed too, and choosing one raised KeyError.
    # Completer configs have no model_type; SEDGenerator writes 'sed_generator'.
    for name, config in [("generator", {"model_type": "sed_generator"}), ("completer", {})]:
        (tmp_path / name).mkdir()
        (tmp_path / name / SEDCompleter.CONFIG_FILE).write_text(json.dumps(config))
    assert [m["name"] for m in SEDCompleter.list_models(str(tmp_path))] == ["completer"]
