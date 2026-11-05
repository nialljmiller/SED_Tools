"""AB and bolometric magnitudes are what the formula gives, for positive flux and for
non-positive flux or reference.

No test uses the network: the autouse fixture fails any real HTTP request.
"""

import math
import warnings

import numpy as np
import pytest

from sed_tools._flux import AB_ZERO_FLUX
from sed_tools.flux_cube_tool import FilterCurve, _band_magnitude, bolometric_magnitude
from sed_tools.models import EvaluatedSED

from helpers_grids import no_network  # noqa: F401 (an autouse fixture)


# ---------------------------------------------------------------------
# AB, Vega and bolometric magnitudes: the computed value, no error
# ---------------------------------------------------------------------
# Zero and negative band flux in AB and Vega are tested in test_photometry.py.

VEGA_ZERO_POINT = 2.0e-10
WL = np.linspace(2000.0, 8000.0, 300)


class _OneFilter:
    """Stands in for the SEDModel behind an EvaluatedSED: every filter resolves to one curve."""

    def __init__(self, curve):
        self.curve = curve

    def _resolve_filters(self, filters):
        return {"Band": self.curve}


def _band():
    return FilterCurve("Band", np.array([3000.0, 4000.0, 5000.0, 6000.0]), np.array([0.0, 1.0, 1.0, 0.0]),
                       {"zp_vega": VEGA_ZERO_POINT})


def _sloped_flux():
    return 3e-12 * (1.0 + WL / 7000.0)


@pytest.mark.parametrize("system, zero_point", [("AB", AB_ZERO_FLUX), ("Vega", VEGA_ZERO_POINT)])
def test_magnitude_of_positive_flux_is_the_formula_exactly(system, zero_point):
    sed = EvaluatedSED(WL, _sloped_flux(), {"teff": 5000.0, "logg": 4.0, "meta": 0.0}, _OneFilter(_band()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = sed.photometry("Band", system=system)["Band"]
        _, flux_density, magnitude = _band_magnitude(WL, _sloped_flux(), _band(), system, None)

    assert flux_density == result.flux_density > 0
    for value in (result.magnitude, magnitude):
        assert type(value) is float
        assert value == -2.5 * math.log10(flux_density / zero_point)


def test_bolometric_magnitude_of_positive_flux_is_the_formula_exactly():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fbol, mbol = bolometric_magnitude(WL, _sloped_flux(), reference_flux=2.5e-8, reference_magnitude=-26.8)

    assert fbol > 0
    assert type(mbol) is float
    assert mbol == -26.8 - 2.5 * math.log10(fbol / 2.5e-8)


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
