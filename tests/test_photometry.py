"""Synthetic photometry: filter curves are read strictly, filters past a SED are named in one
warning, and magnitudes of zero or negative flux, or with a zero or negative zero point, are inf, NaN
or computed.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from helpers_api import no_network  # noqa: F401 (an autouse fixture)


# ---------------------------------------------------------------- flux_cube_tool
def test_a_non_numeric_line_inside_a_filter_curve_raises(tmp_path):
    # Bad rows used to be dropped silently, removing points from the curve.
    from sed_tools.flux_cube_tool import load_filter_curve

    path = tmp_path / "Band.dat"
    path.write_text("Wavelength,Transmission\n4000.0,0.5\n4500.0,O.9\n5000.0,0.5\n")
    with pytest.raises(ValueError, match=r"Band\.dat, line 3: expected two numbers, found '4500\.0,O\.9'"):
        load_filter_curve(str(path))


def test_header_lines_before_the_data_are_still_skipped(tmp_path):
    from sed_tools.flux_cube_tool import load_two_column_file

    path = tmp_path / "vega.csv"
    path.write_text("Vega reference\nwavelength,flux\n4000.0,1e-9\n5000.0,2e-9\n")
    assert load_two_column_file(str(path))[:2] == ([4000.0, 5000.0], [1e-9, 2e-9])


# ---------------------------------------------------------------------
# Magnitudes of zero or negative band flux: the computed value, no error
# ---------------------------------------------------------------------
def _box_band(zero_point=2.0e-10):
    from sed_tools.flux_cube_tool import FilterCurve

    return FilterCurve("Band", np.array([3000.0, 4000.0, 5000.0, 6000.0]), np.array([0.0, 1.0, 1.0, 0.0]),
                       {"zp_vega": zero_point})


class _OneFilter:
    """Stands in for the SEDModel behind an EvaluatedSED: every filter resolves to one curve."""

    def __init__(self, curve):
        self.curve = curve

    def _resolve_filters(self, filters):
        return {"Band": self.curve}


def _photometry(flux_level, system, zero_point=2.0e-10):
    from sed_tools.models import EvaluatedSED

    wl = np.linspace(2000.0, 8000.0, 300)
    sed = EvaluatedSED(wl, np.full(wl.size, flux_level), {"teff": 5000.0, "logg": 4.0, "meta": 0.0},
                       _OneFilter(_box_band(zero_point)))
    return sed.photometry("Band", system=system)["Band"].magnitude


@pytest.mark.parametrize("system", ["AB", "Vega"])
def test_magnitude_of_zero_flux_is_infinite_in_both_systems(system):
    # The Vega path used to raise ValueError from math.log10(0).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _photometry(0.0, system) == math.inf
        assert _band_magnitude(0.0, system) == math.inf


@pytest.mark.parametrize("system", ["AB", "Vega"])
def test_magnitude_of_negative_flux_is_nan_in_both_systems(system):
    # The AB path used to return inf for a negative flux.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(_photometry(-1e-12, system))
        assert math.isnan(_band_magnitude(-1e-12, system))


def _band_magnitude(flux_level, system, zero_point=2.0e-10):
    from sed_tools.flux_cube_tool import _band_magnitude as band_magnitude

    wl = np.linspace(2000.0, 8000.0, 300)
    return band_magnitude(wl, np.full(wl.size, flux_level), _box_band(zero_point), system, None)[2]


@pytest.mark.parametrize("flux_level, zero_point, expected", [
    (1e-12, 0.0, -math.inf), (0.0, 0.0, math.nan), (1e-12, -2.0e-10, math.nan), (0.0, -2.0e-10, math.inf),
])
def test_vega_magnitude_with_a_zero_or_negative_zero_point_is_computed(flux_level, zero_point, expected):
    # A non-positive Vega zero point used to raise ValueError in both modules.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for magnitude in (_photometry(flux_level, "Vega", zero_point),
                          _band_magnitude(flux_level, "Vega", zero_point)):
            if math.isnan(expected):
                assert math.isnan(magnitude)
            else:
                assert magnitude == expected
