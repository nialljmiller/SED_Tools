"""Integration tests for the full pipeline:

    raw (non-standard-unit) .txt spectra
    -> spectra_cleaner.clean_model_dir (unit standardization + lookup rebuild)
    -> precompute_flux_cube.precompute_flux_cube (flux_cube.bin construction)
    -> models.SEDModel (public API load + exact-node evaluation)

This is the "principal integration test" called for in the testing
review: expected flux values are computed independently in each test
(from the raw Jansky/nm inputs, via the textbook F_nu -> F_lambda
Jacobian) rather than by re-using any of the pipeline's own conversion
code, so a bug anywhere in the chain — wrong catalog units applied,
wrong node written to the cube, wavelength grid corruption on disk I/O,
axis-order mixups — will show up as a mismatch here even if each stage
passes its own unit tests in isolation.
"""

import itertools

import numpy as np
import pytest

from sed_tools.models import SEDModel
from sed_tools.precompute_flux_cube import precompute_flux_cube
from sed_tools.spectra_cleaner import clean_model_dir

# Speed of light in Angstrom/s, hardcoded independently (see
# test_spectra_cleaner_units.py for the same convention).
C_ANGSTROM_PER_S = 2.99792458e18


def _write_raw_node(path, teff, logg, meta, wl_nm, flux_jy):
    """Write one *uncleaned* spectrum: nm wavelengths, Jansky flux."""
    lines = [
        "# source = synthetic\n",
        f"# teff = {teff}\n",
        f"# logg = {logg}\n",
        f"# metallicity = {meta}\n",
        "# wavelength_unit = nm\n",
        "# flux in Jansky\n",
    ]
    with path.open("w") as fh:
        fh.writelines(lines)
        for w, f in zip(wl_nm, np.broadcast_to(flux_jy, wl_nm.shape)):
            fh.write(f"{w:.6f} {f:.8e}\n")


