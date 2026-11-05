"""validate_converted_spectrum refuses empty arrays, non-finite or non-positive wavelengths and
mostly non-finite flux.
"""

import numpy as np
import pytest

from sed_tools.spectra_cleaner import (
    validate_converted_spectrum,
)

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
