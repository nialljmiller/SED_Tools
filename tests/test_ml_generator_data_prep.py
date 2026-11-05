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


# ---------------------------------------------------------------------
# _load_flux_cube
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# list_models — filters by model_type, unlike SEDCompleter.list_models
# ---------------------------------------------------------------------

