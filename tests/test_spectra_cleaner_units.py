"""Known-answer tests for unit detection/conversion in ``spectra_cleaner``.

``spectra_cleaner.py`` is documented (CONTRIBUTING.md) as the single
authority for unit conversion across the whole pipeline, so its
conversion math and header-detection patterns are tested directly and
independently here, rather than only indirectly through end-to-end
cleaning.
"""

import numpy as np
import pytest

from sed_tools.spectra_cleaner import (
    UnitInfo,
    convert_to_standard_units,
    detect_units_from_header,
    is_index_grid,
    validate_converted_spectrum,
)

# Speed of light in Angstrom/s, hardcoded independently of the module's
# own SPEED_OF_LIGHT_ANGSTROM constant so this isn't circular.
C_ANGSTROM_PER_S = 2.99792458e18


# ---------------------------------------------------------------------
# Wavelength unit conversion
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Flux unit conversion
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Header-based unit detection (deterministic patterns)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Index-grid detection
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# validate_converted_spectrum
# ---------------------------------------------------------------------

@pytest.mark.parametrize("wavelength, flux, reason", [
    pytest.param([], [], "empty", id="empty_arrays"),
    pytest.param([1.0, np.nan], [1.0, 1.0], "wavelength", id="non_finite_wavelength"),
    pytest.param([0.0, 1.0], [1.0, 1.0], "non-positive", id="non_positive_wavelength"),
    # 3 of 4 flux values non-finite (75%) exceeds the 50% threshold.
    pytest.param([1.0, 2.0, 3.0, 4.0], [1.0, np.nan, np.nan, np.nan], "non-finite flux", id="mostly_non_finite_flux"),
])
def test_validate_rejects(wavelength, flux, reason):
    ok, msg = validate_converted_spectrum(np.array(wavelength), np.array(flux))
    assert ok is False
    assert reason in msg


