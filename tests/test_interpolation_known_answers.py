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

def test_exact_grid_nodes_all_corners(make_affine_cube):
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path)

    for T in (cube.teff[0], cube.teff[-1]):
        for L in (cube.logg[0], cube.logg[-1]):
            for M in (cube.meta[0], cube.meta[-1]):
                result = model(teff=T, logg=L, metallicity=M)
                np.testing.assert_allclose(result.flux, cube.expected(T, L, M), rtol=1e-10)


def test_exact_grid_nodes_interior_node(make_affine_cube):
    # A grid node that isn't a boundary corner still must land exactly
    # (idx = searchsorted(x) - 1 boundary handling in _hermite_interp_axis).
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path)
    T, L, M = cube.teff[1], cube.logg[1], cube.meta[1]
    result = model(teff=T, logg=L, metallicity=M)
    np.testing.assert_allclose(result.flux, cube.expected(T, L, M), rtol=1e-10)


# ---------------------------------------------------------------------
# Interior interpolation: single axis, and all axes at once
# ---------------------------------------------------------------------

def test_interior_interpolation_teff_only(make_affine_cube):
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path)
    T, L, M = 4700.0, cube.logg[1], cube.meta[1]  # T interior, others on-grid
    result = model(teff=T, logg=L, metallicity=M)
    np.testing.assert_allclose(result.flux, cube.expected(T, L, M), rtol=1e-10)


def test_interior_interpolation_logg_only(make_affine_cube):
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path)
    T, L, M = cube.teff[1], 1.8, cube.meta[1]
    result = model(teff=T, logg=L, metallicity=M)
    np.testing.assert_allclose(result.flux, cube.expected(T, L, M), rtol=1e-10)


def test_interior_interpolation_metallicity_only(make_affine_cube):
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path)
    T, L, M = cube.teff[1], cube.logg[1], -1.1
    result = model(teff=T, logg=L, metallicity=M)
    np.testing.assert_allclose(result.flux, cube.expected(T, L, M), rtol=1e-10)


@pytest.mark.parametrize(
    "T,L,M",
    [
        (4700.0, 1.8, -1.0),
        (6100.0, 4.2, -0.1),
        (4300.0, 3.9, -1.6),
        (5900.0, 1.2, -0.05),
    ],
)
def test_interior_interpolation_all_axes(make_affine_cube, T, L, M):
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path)
    result = model(teff=T, logg=L, metallicity=M)
    np.testing.assert_allclose(result.flux, cube.expected(T, L, M), rtol=1e-10)


def test_interior_interpolation_without_cross_term(make_affine_cube):
    # Purely additive/separable field (no T*g coupling) — the simplest
    # case the review calls out, kept as its own test independent of
    # the bilinear-term cases above.
    cube = make_affine_cube(include_cross_term=False)
    model = SEDModel(name="affine_no_cross", flux_cube_path=cube.path)
    result = model(teff=4700.0, logg=1.8, metallicity=-1.0)
    np.testing.assert_allclose(result.flux, cube.expected(4700.0, 1.8, -1.0), rtol=1e-10)


# ---------------------------------------------------------------------
# Non-uniform axis spacing
# ---------------------------------------------------------------------

def test_nonuniform_axis_spacing_multiple_intervals(make_affine_cube):
    # Default grids are already non-uniform; sample points that fall in
    # different sub-intervals of the teff axis in particular.
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path)
    for T in (4200.0, 4600.0, 5200.0, 6100.0):  # spans all 3 teff intervals
        result = model(teff=T, logg=2.0, metallicity=-0.3)
        np.testing.assert_allclose(result.flux, cube.expected(T, 2.0, -0.3), rtol=1e-10)


# ---------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------

@pytest.mark.parametrize("axis", ["teff", "logg", "meta"])
def test_boundary_exact_min_and_max(make_affine_cube, axis):
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path)
    grids = {"teff": cube.teff, "logg": cube.logg, "meta": cube.meta}
    grid = grids[axis]
    fixed = {"teff": cube.teff[1], "logg": cube.logg[1], "meta": cube.meta[1]}
    for boundary in (grid[0], grid[-1]):
        params = dict(fixed)
        params[axis] = boundary
        result = model(teff=params["teff"], logg=params["logg"], metallicity=params["meta"])
        np.testing.assert_allclose(
            result.flux, cube.expected(params["teff"], params["logg"], params["meta"]), rtol=1e-10
        )


# ---------------------------------------------------------------------
# Singleton axes
# ---------------------------------------------------------------------

def test_singleton_logg_axis(make_affine_cube):
    cube = make_affine_cube(logg=(3.0,))
    model = SEDModel(name="singleton_logg", flux_cube_path=cube.path)
    result = model(teff=4700.0, logg=3.0, metallicity=-1.0)
    np.testing.assert_allclose(result.flux, cube.expected(4700.0, 3.0, -1.0), rtol=1e-10)


def test_singleton_teff_axis(make_affine_cube):
    cube = make_affine_cube(teff=(5000.0,))
    model = SEDModel(name="singleton_teff", flux_cube_path=cube.path)
    result = model(teff=5000.0, logg=2.0, metallicity=-0.5)
    np.testing.assert_allclose(result.flux, cube.expected(5000.0, 2.0, -0.5), rtol=1e-10)


def test_singleton_metallicity_axis(make_affine_cube):
    cube = make_affine_cube(meta=(0.0,))
    model = SEDModel(name="singleton_meta", flux_cube_path=cube.path)
    result = model(teff=4700.0, logg=2.0, metallicity=0.0)
    np.testing.assert_allclose(result.flux, cube.expected(4700.0, 2.0, 0.0), rtol=1e-10)


def test_singleton_axis_silently_pins_value_when_fill_gaps(make_affine_cube):
    # Documents current (non-obvious) behaviour: with fill_gaps=True
    # (the default), a request that doesn't match a singleton axis's
    # only grid point is silently pinned to that grid point rather than
    # raising. If this pinning behaviour is ever changed intentionally,
    # this test should be updated alongside it.
    cube = make_affine_cube(logg=(3.0,))
    model = SEDModel(name="singleton_logg_pin", flux_cube_path=cube.path)
    result = model(teff=4700.0, logg=999.0, metallicity=-1.0)
    assert result.metadata["logg"] == 3.0
    np.testing.assert_allclose(result.flux, cube.expected(4700.0, 3.0, -1.0), rtol=1e-10)


def test_singleton_axis_raises_with_fill_gaps_false(make_affine_cube):
    cube = make_affine_cube(logg=(3.0,))
    model = SEDModel(name="singleton_logg_strict", flux_cube_path=cube.path, fill_gaps=False)
    with pytest.raises(ValueError):
        model(teff=4700.0, logg=999.0, metallicity=-1.0)


# ---------------------------------------------------------------------
# Out-of-range handling: fill_gaps clamping vs strict rejection
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs,expected_point",
    [
        ({"teff": 9000.0, "logg": 2.0, "metallicity": -0.3}, (6200.0, 2.0, -0.3)),
        ({"teff": 1000.0, "logg": 2.0, "metallicity": -0.3}, (4000.0, 2.0, -0.3)),
        ({"teff": 5000.0, "logg": 50.0, "metallicity": -0.3}, (5000.0, 5.0, -0.3)),
        ({"teff": 5000.0, "logg": -10.0, "metallicity": -0.3}, (5000.0, 1.0, -0.3)),
        ({"teff": 5000.0, "logg": 2.0, "metallicity": 5.0}, (5000.0, 2.0, 0.0)),
        ({"teff": 5000.0, "logg": 2.0, "metallicity": -10.0}, (5000.0, 2.0, -2.0)),
    ],
)
def test_fill_gaps_clamps_to_boundary(make_affine_cube, kwargs, expected_point):
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path, fill_gaps=True)
    result = model(**kwargs)
    np.testing.assert_allclose(result.flux, cube.expected(*expected_point), rtol=1e-10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"teff": 9000.0, "logg": 2.0, "metallicity": -0.3},
        {"teff": 1000.0, "logg": 2.0, "metallicity": -0.3},
        {"teff": 5000.0, "logg": 50.0, "metallicity": -0.3},
        {"teff": 5000.0, "logg": -10.0, "metallicity": -0.3},
        {"teff": 5000.0, "logg": 2.0, "metallicity": 5.0},
        {"teff": 5000.0, "logg": 2.0, "metallicity": -10.0},
    ],
)
def test_fill_gaps_false_raises_out_of_range(make_affine_cube, kwargs):
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path, fill_gaps=False)
    with pytest.raises(ValueError):
        model(**kwargs)


# ---------------------------------------------------------------------
# NaN / infinite parameters
# ---------------------------------------------------------------------

@pytest.mark.parametrize("param", ["teff", "logg", "metallicity"])
def test_nan_parameter_raises_even_with_fill_gaps(make_affine_cube, param):
    # NaN comparisons are always False, so the fill_gaps clamp
    # (min(max(value, lower), upper)) passes NaN straight through, and
    # the low-level bounds check in FluxCube._hermite_interp_axis then
    # raises. This is current, verified behaviour, not an assumption.
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path, fill_gaps=True)
    kwargs = {"teff": 5000.0, "logg": 2.0, "metallicity": -0.3}
    kwargs[param] = float("nan")
    with pytest.raises(ValueError):
        model(**kwargs)


@pytest.mark.parametrize("param,sign", [("teff", 1), ("teff", -1), ("logg", 1), ("logg", -1)])
def test_infinite_parameter_clamps_to_boundary(make_affine_cube, param, sign):
    cube = make_affine_cube()
    model = SEDModel(name="affine", flux_cube_path=cube.path, fill_gaps=True)
    kwargs = {"teff": 5000.0, "logg": 2.0, "metallicity": -0.3}
    kwargs[param] = math.inf * sign
    result = model(**kwargs)

    grids = {"teff": cube.teff, "logg": cube.logg, "metallicity": cube.meta}
    expected_point = [kwargs["teff"], kwargs["logg"], kwargs["metallicity"]]
    idx = ["teff", "logg", "metallicity"].index(param)
    expected_point[idx] = float(grids[param][-1] if sign > 0 else grids[param][0])
    np.testing.assert_allclose(result.flux, cube.expected(*expected_point), rtol=1e-10)


# ---------------------------------------------------------------------
# Low-level FluxCube (bypassing SEDModel's fill_gaps clamp entirely)
# ---------------------------------------------------------------------

def test_fluxcube_interpolate_spectrum_matches_analytic(make_affine_cube):
    cube = make_affine_cube()
    fc = FluxCube.from_file(str(cube.path))
    wl, flux = fc.interpolate_spectrum(4700.0, 1.8, -1.0)
    np.testing.assert_array_equal(wl, cube.wavelengths)
    np.testing.assert_allclose(flux, cube.expected(4700.0, 1.8, -1.0), rtol=1e-10)


@pytest.mark.parametrize(
    "teff,logg,meta",
    [
        (9000.0, 2.0, -0.3),
        (5000.0, 50.0, -0.3),
        (5000.0, 2.0, 5.0),
    ],
)
def test_fluxcube_interpolate_spectrum_raises_out_of_range(make_affine_cube, teff, logg, meta):
    # Unlike SEDModel, FluxCube.interpolate_spectrum has no fill_gaps
    # concept at all — any out-of-range axis raises unconditionally.
    cube = make_affine_cube()
    fc = FluxCube.from_file(str(cube.path))
    with pytest.raises(ValueError):
        fc.interpolate_spectrum(teff, logg, meta)


def test_fluxcube_singleton_axis_ignores_out_of_range_query(make_affine_cube):
    # At the FluxCube level (no SEDModel clamping in front of it), a
    # degenerate/singleton axis skips the bounds check entirely and
    # always returns the lone slice, regardless of the requested value.
    cube = make_affine_cube(logg=(3.0,))
    fc = FluxCube.from_file(str(cube.path))
    wl, flux = fc.interpolate_spectrum(4700.0, 999.0, -1.0)
    np.testing.assert_allclose(flux, cube.expected(4700.0, 3.0, -1.0), rtol=1e-10)
