"""Filter properties (sed_tools.filter_properties): definitions and agreement with SED_Tools photometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sed_tools.filter_metadata import write_filter_file
from sed_tools.filter_properties import (
    AB_JY,
    ST_FLAM,
    curve_properties,
    filter_properties,
    instrument_properties,
)
from sed_tools.flux_cube_tool import (
    SPEED_OF_LIGHT_ANG_PER_S,
    FilterCurve,
    band_average_flux_lambda_from_arrays,
    band_average_flux_nu,
)


def box(lo=4000.0, hi=6000.0, ramp=1e-3, n=4001):
    """A top-hat filter with near-vertical edges (zero just outside lo..hi)."""
    inner = np.linspace(lo, hi, n)
    lam = np.concatenate([[lo - ramp], inner, [hi + ramp]])
    t = np.concatenate([[0.0], np.ones(n), [0.0]])
    return lam, t


def gaussian(center=5500.0, sigma=300.0):
    lam = np.linspace(center - 6 * sigma, center + 6 * sigma, 6001)
    return lam, np.exp(-0.5 * ((lam - center) / sigma) ** 2)


def test_box_filter_definitions():
    lo, hi = 4000.0, 6000.0
    p = filter_properties(*box(lo, hi), detector_type="photon")
    assert p["WavelengthMean"] == pytest.approx((lo + hi) / 2, rel=1e-6)
    assert p["WidthEff"] == pytest.approx(hi - lo, rel=1e-5)
    assert p["FWHM"] == pytest.approx(hi - lo, rel=1e-5)
    assert p["WavelengthCen"] == pytest.approx((lo + hi) / 2, rel=1e-6)
    assert (p["WavelengthMin"], p["WavelengthMax"]) == (pytest.approx(lo), pytest.approx(hi))
    # photon-counting pivot: λp² = ∫ λ T dλ / ∫ T/λ dλ
    assert p["WavelengthPivot"] == pytest.approx(math.sqrt((hi**2 - lo**2) / 2 / math.log(hi / lo)), rel=1e-6)


def test_energy_counting_pivot_uses_the_energy_weight():
    lo, hi = 4000.0, 6000.0
    p = filter_properties(*box(lo, hi), detector_type="energy")
    # energy-counting pivot: λp² = ∫ T dλ / ∫ T/λ² dλ
    assert p["WavelengthPivot"] == pytest.approx(math.sqrt((hi - lo) / (1 / lo - 1 / hi)), rel=1e-6)


def test_gaussian_fwhm_and_peak():
    p = filter_properties(*gaussian(5500.0, 300.0), detector_type="photon")
    assert p["FWHM"] == pytest.approx(2 * math.sqrt(2 * math.log(2)) * 300.0, rel=1e-4)
    assert p["WavelengthPeak"] == pytest.approx(5500.0)
    assert p["WidthEff"] == pytest.approx(math.sqrt(2 * math.pi) * 300.0, rel=1e-4)


@pytest.mark.parametrize("detector", ["photon", "energy"])
def test_zero_points_agree_with_sed_tools_band_averages(detector):
    lam, t = gaussian(7000.0, 900.0)
    vega_lam = np.linspace(900.0, 30000.0, 20000)
    vega_flam = 3.6e-9 * (5556.0 / vega_lam) ** 4 * np.exp(-((vega_lam - 5000.0) / 9000.0) ** 2)
    p = filter_properties(lam, t, detector, vega=(vega_lam, vega_flam))
    curve = FilterCurve("x", lam, t, {}, detector_type=detector)

    vega_on_curve = np.interp(lam, vega_lam, vega_flam)
    assert p["ZeroPointVega_flam"] == pytest.approx(band_average_flux_lambda_from_arrays(lam, vega_on_curve, t, detector))
    assert p["ZeroPointVega_Jy"] == pytest.approx(band_average_flux_nu(vega_lam, vega_flam, curve) / 1e-23, rel=1e-9)

    ab_flam = AB_JY * 1e-23 * SPEED_OF_LIGHT_ANG_PER_S / lam**2
    assert p["ZeroPointAB_Jy"] == pytest.approx(band_average_flux_nu(lam, ab_flam, curve) / 1e-23, rel=1e-9)
    assert p["ZeroPointAB_flam"] == pytest.approx(band_average_flux_lambda_from_arrays(lam, ab_flam, t, detector), rel=1e-9)

    st = np.full_like(lam, ST_FLAM)
    assert p["ZeroPointST_Jy"] == pytest.approx(band_average_flux_nu(lam, st, curve) / 1e-23, rel=1e-9)


def test_vega_quantities_need_full_coverage():
    lam, t = gaussian(50000.0, 3000.0)
    p = filter_properties(lam, t, "photon", vega=(np.array([900.0, 30000.0]), np.array([1e-9, 1e-12])))
    assert p["ZeroPointVega_Jy"] is None and p["WavelengthEff"] is None
    assert p["ZeroPointAB_Jy"] == AB_JY


def test_curve_at_full_transmission_at_its_edge():
    lam = np.linspace(4000.0, 5000.0, 101)
    p = filter_properties(lam, np.linspace(1.0, 0.0, 101), "photon")
    assert p["WavelengthCen"] - p["FWHM"] / 2 == pytest.approx(4000.0)
    assert p["FWHM"] == pytest.approx(500.0)


def test_rejects_unusable_curves():
    with pytest.raises(ValueError):
        filter_properties([5000.0], [1.0])
    with pytest.raises(ValueError):
        filter_properties([4000.0, 5000.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        filter_properties([4000.0, 5000.0], [1.0, 1.0], detector_type="quantum")


def test_file_and_instrument_helpers_use_the_curve_metadata(tmp_path):
    inst = tmp_path / "Test" / "Camera"
    inst.mkdir(parents=True)
    lam, t = gaussian(5500.0, 300.0)
    write_filter_file(inst / "V.dat", lam / 1e4, t, detector_type="photon", source="svo",
                      wavelength_unit="micron")  # stored in µm: the loader converts to Å
    write_filter_file(inst / "B.dat", lam - 1100.0, t, detector_type="energy", source="original")
    expected = filter_properties(lam, t, "photon")
    for key, value in curve_properties(inst / "V.dat").items():
        assert value == pytest.approx(expected[key], rel=1e-9, abs=1e-30) if value is not None else expected[key] is None
    props = instrument_properties(inst)
    assert sorted(props) == ["B", "V"]
    assert (props["V"]["DetectorType"], props["B"]["DetectorType"]) == ("photon", "energy")
    assert props["B"]["DetectorTypeSource"] == "original"
