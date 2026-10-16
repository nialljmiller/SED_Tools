"""Known-answer tests for ``sed_tools._resample.resample_to_grid``.

``resample_to_grid`` is the single authoritative resampling function used
by both ``precompute_flux_cube.py`` and ``combine_stellar_atm.py`` (per
its own docstring), so a bug here silently propagates into every flux
cube built from text spectra.
"""

import numpy as np
import pytest

from sed_tools._resample import resample_to_grid


def test_semi_log_interpolation_matches_analytic_within_coverage():
    # resample_to_grid interpolates log10(flux) linearly against
    # (linear) wavelength, so a spectrum with log10(flux) exactly linear
    # in wavelength must be reproduced exactly at any point in coverage.
    wl_src = np.linspace(4000.0, 8000.0, 50)
    m, b = -0.0003, -14.0
    fl_src = 10.0 ** (m * wl_src + b)

    wl_tgt = np.array([4000.0, 5000.0, 6123.4, 7999.9, 8000.0])
    fl_tgt = resample_to_grid(wl_src, fl_src, teff=5000.0, wl_tgt=wl_tgt)

    expected = 10.0 ** (m * wl_tgt + b)
    np.testing.assert_allclose(fl_tgt, expected, rtol=1e-10)


def test_out_of_coverage_is_zero_filled_not_extrapolated():
    wl_src = np.linspace(4000.0, 8000.0, 20)
    fl_src = np.full(wl_src.size, 5.0)
    wl_tgt = np.array([1000.0, 3999.999, 4000.0, 8000.0, 8000.001, 20000.0])

    fl_tgt = resample_to_grid(wl_src, fl_src, teff=5000.0, wl_tgt=wl_tgt)

    # In-coverage values round-trip through log10/10** so aren't bit-exact;
    # out-of-coverage zero-fill is exact.
    np.testing.assert_allclose(fl_tgt, [0.0, 0.0, 5.0, 5.0, 0.0, 0.0], rtol=1e-12, atol=1e-12)


def test_output_length_matches_target_grid_regardless_of_source_length():
    wl_src = np.linspace(4000.0, 8000.0, 7)
    fl_src = np.full(wl_src.size, 1.0)
    wl_tgt = np.linspace(3000.0, 9000.0, 123)

    fl_tgt = resample_to_grid(wl_src, fl_src, teff=5000.0, wl_tgt=wl_tgt)

    assert fl_tgt.shape == wl_tgt.shape


def test_output_is_never_negative_even_with_negative_source_flux():
    wl_src = np.linspace(4000.0, 8000.0, 10)
    fl_src = np.full(wl_src.size, 5.0)
    fl_src[3] = -1.0  # unphysical negative flux point in the source

    fl_tgt = resample_to_grid(wl_src, fl_src, teff=5000.0, wl_tgt=wl_src[3:4])

    assert fl_tgt[0] >= 0.0
    assert np.isfinite(fl_tgt[0])


def test_teff_argument_does_not_affect_output():
    # Documented in the docstring: teff is retained for interface
    # stability but not used to extrapolate. Confirm that holds.
    wl_src = np.linspace(4000.0, 8000.0, 20)
    fl_src = np.linspace(1.0, 2.0, 20)
    wl_tgt = np.linspace(4500.0, 7500.0, 15)

    low_t = resample_to_grid(wl_src, fl_src, teff=3000.0, wl_tgt=wl_tgt)
    high_t = resample_to_grid(wl_src, fl_src, teff=50000.0, wl_tgt=wl_tgt)

    np.testing.assert_array_equal(low_t, high_t)


def test_single_point_target_grid():
    wl_src = np.linspace(4000.0, 8000.0, 10)
    fl_src = np.full(wl_src.size, 3.0)
    fl_tgt = resample_to_grid(wl_src, fl_src, teff=5000.0, wl_tgt=np.array([6000.0]))
    np.testing.assert_allclose(fl_tgt, [3.0])
