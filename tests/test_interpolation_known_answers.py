"""SEDModel on an analytic cube (``make_affine_cube`` in conftest.py): a request off a singleton
axis raises without fill_gaps, and a NaN parameter raises even with it.
"""


import pytest

from sed_tools.models import SEDModel


# ---------------------------------------------------------------------
# Singleton axes
# ---------------------------------------------------------------------

def test_singleton_axis_raises_with_fill_gaps_false(make_affine_cube):
    cube = make_affine_cube(logg=(3.0,))
    model = SEDModel(name="singleton_logg_strict", flux_cube_path=cube.path, fill_gaps=False)
    with pytest.raises(ValueError):
        model(teff=4700.0, logg=999.0, metallicity=-1.0)


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
