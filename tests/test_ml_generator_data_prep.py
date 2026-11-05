"""SEDGenerator without torch: ``_prepare_training_data`` raises when no flux is usable.
"""


import numpy as np
import pytest

from sed_tools.ml_sed_generator import SEDGenerator


# ---------------------------------------------------------------------
# _prepare_training_data
# ---------------------------------------------------------------------

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
