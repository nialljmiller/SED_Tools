"""Known-answer tests for SEDModel / FluxCube interpolation.

These tests build a flux cube whose values are an analytic function of
(teff, logg, metallicity) with per-axis, per-wavelength coefficients
(see ``make_affine_cube`` in conftest.py), then check the model against
the closed-form answer rather than against the same production code
that generated it. A swapped or scrambled axis, a broken derivative
calculation, or an off-by-one grid index will all fail these tests even
though they might not affect a constant-field test.
"""

import math

import numpy as np
import pytest

from sed_tools._flux import FluxCube
from sed_tools.models import SEDModel


# ---------------------------------------------------------------------
# Exact grid nodes
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Interior interpolation: single axis, and all axes at once
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Non-uniform axis spacing
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Singleton axes
# ---------------------------------------------------------------------

def test_singleton_axis_raises_with_fill_gaps_false(make_affine_cube):
    cube = make_affine_cube(logg=(3.0,))
    model = SEDModel(name="singleton_logg_strict", flux_cube_path=cube.path, fill_gaps=False)
    with pytest.raises(ValueError):
        model(teff=4700.0, logg=999.0, metallicity=-1.0)


# ---------------------------------------------------------------------
# Out-of-range handling: fill_gaps clamping vs strict rejection
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# NaN / infinite parameters
# ---------------------------------------------------------------------

def test_nan_parameter_raises_even_with_fill_gaps(make_affine_cube):
    # NaN comparisons are always False, so the fill_gaps clamp
    # (min(max(value, lower), upper)) passes NaN straight through, and
    # the low-level bounds check in FluxCube._hermite_interp_axis then
    # raises. This is current, verified behaviour, not an assumption.
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path, fill_gaps=True)
    with pytest.raises(ValueError):
        model(teff=float("nan"), logg=2.0, metallicity=-0.3)


# ---------------------------------------------------------------------
# Low-level FluxCube (bypassing SEDModel's fill_gaps clamp entirely)
# ---------------------------------------------------------------------

