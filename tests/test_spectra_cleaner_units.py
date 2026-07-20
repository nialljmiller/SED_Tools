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

@pytest.mark.parametrize(
    "unit,factor",
    [
        ("angstrom", 1.0),
        ("nm", 10.0),
        ("um", 1e4),
        ("cm", 1e8),
        ("m", 1e10),
    ],
)
def test_wavelength_conversion_to_angstrom(unit, factor):
    wl = np.array([100.0, 200.0, 300.0])
    flux = np.array([1.0, 1.0, 1.0])
    unit_info = UnitInfo(unit, factor, "flam", 1.0, "high", "header")

    wl_ang, _ = convert_to_standard_units(wl, flux, unit_info)

    np.testing.assert_allclose(wl_ang, wl * factor)


# ---------------------------------------------------------------------
# Flux unit conversion
# ---------------------------------------------------------------------

def test_flam_passthrough_unchanged():
    wl = np.array([4000.0, 5000.0, 6000.0])
    flux = np.array([1e-10, 2e-10, 3e-10])
    unit_info = UnitInfo("angstrom", 1.0, "flam", 1.0, "high", "header")

    wl_ang, flux_flam = convert_to_standard_units(wl, flux, unit_info)

    np.testing.assert_array_equal(wl_ang, wl)
    np.testing.assert_array_equal(flux_flam, flux)


def test_fnu_to_flam_jacobian():
    # F_lambda = F_nu * c / lambda^2
    wl = np.array([4000.0, 5000.0, 6000.0])
    flux_nu = np.array([1.0, 2.0, 3.0])
    unit_info = UnitInfo("angstrom", 1.0, "fnu", 1.0, "high", "header")

    _, flux_flam = convert_to_standard_units(wl, flux_nu, unit_info)

    expected = flux_nu * C_ANGSTROM_PER_S / (wl ** 2)
    np.testing.assert_allclose(flux_flam, expected, rtol=1e-10)


def test_fnu_jy_applies_jansky_scale_before_jacobian():
    # 1 Jy = 1e-23 erg/cm^2/s/Hz, applied as flux_factor before the
    # F_nu -> F_lambda Jacobian.
    wl = np.array([4000.0, 5000.0, 6000.0])
    flux_jy = np.array([1.0, 2.0, 3.0])
    unit_info = UnitInfo("angstrom", 1.0, "fnu_jy", 1e-23, "high", "header")

    _, flux_flam = convert_to_standard_units(wl, flux_jy, unit_info)

    expected = (flux_jy * 1e-23) * C_ANGSTROM_PER_S / (wl ** 2)
    np.testing.assert_allclose(flux_flam, expected, rtol=1e-10)


def test_normalized_flux_only_scaled_not_jacobian_converted():
    wl = np.array([4000.0, 5000.0])
    flux = np.array([1.0, 2.0])
    unit_info = UnitInfo("angstrom", 1.0, "normalized", 2.0, "low", "range")

    _, flux_out = convert_to_standard_units(wl, flux, unit_info)

    np.testing.assert_allclose(flux_out, flux * 2.0)


def test_combined_wavelength_and_flux_conversion():
    # nm wavelengths + Jansky flux together, as would occur for a real
    # non-standard catalog file.
    wl_nm = np.array([400.0, 500.0, 600.0])
    flux_jy = np.array([1.0, 1.0, 1.0])
    unit_info = UnitInfo("nm", 10.0, "fnu_jy", 1e-23, "medium", "header")

    wl_ang, flux_flam = convert_to_standard_units(wl_nm, flux_jy, unit_info)

    expected_wl = wl_nm * 10.0
    expected_flux = (flux_jy * 1e-23) * C_ANGSTROM_PER_S / (expected_wl ** 2)
    np.testing.assert_allclose(wl_ang, expected_wl)
    np.testing.assert_allclose(flux_flam, expected_flux, rtol=1e-10)


def test_fnu_conversion_handles_zero_wavelength_without_crashing():
    # convert_to_standard_units is defensive against wl=0 (division by
    # zero -> np.where(isfinite, ..., 0.0)), even though clean_spectrum_file
    # filters wl>0 upstream. Test the function's own guard directly.
    wl = np.array([0.0, 4000.0])
    flux = np.array([1.0, 1.0])
    unit_info = UnitInfo("angstrom", 1.0, "fnu", 1.0, "low", "range")

    _, flux_flam = convert_to_standard_units(wl, flux, unit_info)

    assert np.all(np.isfinite(flux_flam))
    assert flux_flam[0] == 0.0


# ---------------------------------------------------------------------
# Header-based unit detection (deterministic patterns)
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "header,expected_wl,expected_flux",
    [
        ("# wavelength_unit = nm\n# flux_unit = erg/s/cm2/A\n", "nm", "flam"),
        ("# Wavelength (nm)  Flux\n", "nm", None),
        ("# columns: wavelength(nm), flux\n", "nm", None),
        ("# units: F_nu\n", None, "fnu"),
        ("# flux in Jansky\n", None, "fnu_jy"),
        ("# flux W/m2/A\n", None, "flam_si"),
        ("# normalized flux\n", None, "normalized"),
        ("# no unit info here at all\n", None, None),
    ],
)
def test_detect_units_from_header_patterns(header, expected_wl, expected_flux):
    result = detect_units_from_header(header)
    assert result["wavelength_unit"] == expected_wl
    assert result["flux_unit"] == expected_flux


# ---------------------------------------------------------------------
# Index-grid detection
# ---------------------------------------------------------------------

def test_is_index_grid_zero_based():
    assert is_index_grid(np.arange(10, dtype=float)) is True


def test_is_index_grid_one_based():
    assert is_index_grid(np.arange(1, 11, dtype=float)) is True


def test_is_index_grid_false_for_physical_wavelengths():
    assert is_index_grid(np.array([4000.0, 5000.0, 6000.0, 7000.0])) is False


def test_is_index_grid_false_for_uniform_step_not_starting_near_zero():
    # Uniform step of 1 but starting well above the near-zero threshold
    # is not treated as an index grid.
    assert is_index_grid(np.array([3.0, 4.0, 5.0, 6.0])) is False


def test_is_index_grid_false_for_too_few_points():
    assert is_index_grid(np.array([0.0, 1.0, 2.0])) is False


# ---------------------------------------------------------------------
# validate_converted_spectrum
# ---------------------------------------------------------------------

def test_validate_rejects_empty_arrays():
    ok, msg = validate_converted_spectrum(np.array([]), np.array([]))
    assert ok is False
    assert "empty" in msg


def test_validate_rejects_non_finite_wavelength():
    ok, msg = validate_converted_spectrum(np.array([1.0, np.nan]), np.array([1.0, 1.0]))
    assert ok is False
    assert "wavelength" in msg


def test_validate_rejects_non_positive_wavelength():
    ok, msg = validate_converted_spectrum(np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    assert ok is False
    assert "non-positive" in msg


def test_validate_rejects_mostly_non_finite_flux():
    # 3 of 4 flux values non-finite (75%) exceeds the 50% threshold.
    ok, msg = validate_converted_spectrum(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, np.nan, np.nan, np.nan])
    )
    assert ok is False
    assert "non-finite flux" in msg


def test_validate_accepts_exactly_50_percent_finite_flux_boundary():
    # Documents the current boundary behaviour: exactly 50% valid passes
    # (the check is strictly "< 50%", not "<= 50%").
    ok, msg = validate_converted_spectrum(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, np.nan, np.nan, 1.0])
    )
    assert ok is True


def test_validate_accepts_clean_spectrum():
    ok, msg = validate_converted_spectrum(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert ok is True
    assert msg == "ok"
