"""AB and bolometric magnitudes are what the formula gives, for positive flux and for
non-positive flux or reference.

No test uses the network: the autouse fixture fails any real HTTP request.
"""

import math
import warnings

import numpy as np
import pytest

from sed_tools.flux_cube_tool import bolometric_magnitude

from helpers_grids import no_network  # noqa: F401 (an autouse fixture)


# ---------------------------------------------------------------------
# AB, Vega and bolometric magnitudes: the computed value, no error
# ---------------------------------------------------------------------
# Zero and negative band flux in AB and Vega are tested in test_photometry.py.

WL = np.linspace(2000.0, 8000.0, 300)


def _sloped_flux():
    return 3e-12 * (1.0 + WL / 7000.0)


@pytest.mark.parametrize("flux_scale, reference_flux, expected", [
    (0.0, 1.0, math.inf),     # used to be inf by a special case
    (-1.0, 1.0, math.nan),    # used to be inf by a special case
    (1.0, 0.0, -math.inf),    # used to raise ZeroDivisionError
    (1.0, -1.0, math.nan),    # used to raise ValueError from math.log10
    (0.0, 0.0, math.nan),     # used to be inf by a special case
])
def test_bolometric_magnitude_of_non_positive_flux_or_reference_is_computed(flux_scale, reference_flux, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, mbol = bolometric_magnitude(WL, flux_scale * _sloped_flux(), reference_flux=reference_flux)

    if math.isnan(expected):
        assert math.isnan(mbol)
    else:
        assert mbol == expected
