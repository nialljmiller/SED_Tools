"""Filter properties (sed_tools.filter_properties): definitions and agreement with SED_Tools photometry."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

import sed_tools.filter_properties as fp
from sed_tools.filter_metadata import write_filter_file
from sed_tools.filter_properties import (
    AB_JY,
    EXTINCTION_RV,
    ST_FLAM,
    UNITS,
    curve_properties,
    extinction_law,
    extinction_ratio,
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


@pytest.mark.parametrize("detector", ["photon", "energy"])
def test_magnitude_offsets_vanish_when_vega_is_the_reference_spectrum(detector):
    lam, t = gaussian(7000.0, 900.0)
    grid = np.linspace(900.0, 30000.0, 20000)
    ab = (grid, AB_JY * 1e-23 * SPEED_OF_LIGHT_ANG_PER_S / grid**2)
    assert filter_properties(lam, t, detector, vega=ab)["ABminusVega"] == pytest.approx(0.0, abs=1e-6)
    st = (grid, np.full_like(grid, ST_FLAM))
    assert filter_properties(lam, t, detector, vega=st)["STminusVega"] == pytest.approx(0.0, abs=1e-9)
    # A source 10x fainter than Vega in f_nu is 2.5 mag fainter in both systems.
    p = filter_properties(lam, t, detector, vega=(ab[0], ab[1] * 10))
    assert p["ABminusVega"] == pytest.approx(-2.5, abs=1e-6)


def test_vega_quantities_need_coverage_where_the_filter_transmits():
    lam, t = gaussian(50000.0, 3000.0)
    p = filter_properties(lam, t, "photon", vega=(np.array([900.0, 30000.0]), np.array([1e-9, 1e-12])))
    assert p["ZeroPointVega_Jy"] is None and p["WavelengthEff"] is None
    assert p["ABminusVega"] is None and p["STminusVega"] is None
    assert p["ZeroPointAB_Jy"] == AB_JY
    # A negligible tail beyond the spectrum counts as zero flux, as in SED_Tools photometry.
    vega = (np.linspace(900.0, 7400.0, 2000), np.full(2000, 1e-9))
    lam, t = gaussian(5500.0, 300.0)  # tabulated out to 7300 A; above 1 % of peak only to ~6400 A
    tail = np.concatenate([lam, [8000.0]]), np.concatenate([t, [0.0]])
    assert filter_properties(*tail, "photon", vega=vega)["ZeroPointVega_flam"] == pytest.approx(1e-9, rel=1e-6)


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


def test_every_property_has_a_unit():
    assert set(filter_properties(*gaussian()).keys()) == set(UNITS)


@pytest.mark.parametrize("detector", ["photon", "energy"])
def test_fsun_is_the_band_average_with_the_detector_weight(detector):
    lo, hi = 4000.0, 6000.0
    grid = np.linspace(1000.0, 30000.0, 290001)
    sun = (grid, 2.0e9 / grid**2)  # f_lambda = A / lambda^2
    p = filter_properties(*box(lo, hi), detector, sun=sun)
    if detector == "photon":  # W = T lambda: A ln(hi/lo) / ((hi^2 - lo^2) / 2)
        expected = 2.0e9 * math.log(hi / lo) / ((hi**2 - lo**2) / 2)
    else:  # W = T: A (1/lo - 1/hi) / (hi - lo)
        expected = 2.0e9 * (1 / lo - 1 / hi) / (hi - lo)
    assert p["Fsun"] == pytest.approx(expected, rel=1e-5)
    # A flat spectrum averages to itself whatever the weight.
    assert filter_properties(*box(lo, hi), detector, sun=(grid, np.full_like(grid, 1.9e2)))["Fsun"] \
        == pytest.approx(1.9e2, rel=1e-12)


@pytest.mark.parametrize("detector", ["photon", "energy"])
def test_fsun_uses_the_vega_zero_point_definition(detector):
    lam, t = gaussian(7000.0, 900.0)
    grid = np.linspace(900.0, 30000.0, 20000)
    spectrum = (grid, 3.6e-9 * (5556.0 / grid) ** 4)
    p = filter_properties(lam, t, detector, vega=spectrum, sun=spectrum)
    assert p["Fsun"] == p["ZeroPointVega_flam"]
    assert filter_properties(lam, t, detector, vega=spectrum)["Fsun"] is None


def test_fsun_needs_coverage_where_the_filter_transmits():
    lam, t = gaussian(50000.0, 3000.0)
    sun = (np.array([1195.0, 30000.0]), np.array([1.0, 1.0]))
    p = filter_properties(lam, t, "photon", sun=sun)
    assert p["Fsun"] is None and p["WavelengthPivot"] > 0
    assert filter_properties(*gaussian(5500.0, 300.0), "photon", sun=sun)["Fsun"] == pytest.approx(1.0)


def test_afav_is_dust_extinction_f99_at_the_pivot():
    pytest.importorskip("dust_extinction")
    import astropy.units as u
    from dust_extinction.parameter_averages import F99

    law = F99(Rv=3.1)
    for curve, detector in ((gaussian(5500.0, 300.0), "photon"), (box(12000.0, 13500.0), "energy"),
                            (gaussian(2200.0, 150.0), "energy")):
        p = filter_properties(*curve, detector)
        assert p["AfAv"] == pytest.approx(float(law(p["WavelengthPivot"] * u.AA)), rel=1e-12)
    assert EXTINCTION_RV == 3.1


def test_afav_follows_the_definition_of_r_v():
    pytest.importorskip("dust_extinction")
    # dust_extinction's normalisation notes put A(lambda)/A(V) = 1 at 0.5414 micron and
    # 1 + 1/R(V) (the B band, since E(B-V) = A(V)/R(V)) at 0.4354 micron for F99.
    assert extinction_ratio(5414.0) == pytest.approx(1.0, abs=1e-4)
    assert extinction_ratio(4354.0) == pytest.approx(1 + 1 / 3.1, abs=1e-3)
    # Extinction falls with wavelength across the optical and near-infrared.
    lam = np.geomspace(3000.0, 33000.0, 50)
    assert np.all(np.diff([extinction_ratio(x) for x in lam]) < 0)


def test_afav_is_none_outside_the_laws_range():
    pytest.importorskip("dust_extinction")
    from dust_extinction.parameter_averages import F99

    lo, hi = F99.x_range  # inverse microns: 1000 A to 33333 A
    assert extinction_ratio(1e4 / hi * 1.001) is not None
    assert extinction_ratio(1e4 / hi * 0.999) is None
    assert extinction_ratio(1e4 / lo * 0.999) is not None
    assert extinction_ratio(1e4 / lo * 1.001) is None
    assert filter_properties(*gaussian(50000.0, 3000.0), "photon")["AfAv"] is None  # WISE W2-like
    law = extinction_law()
    assert law["valid_range_angstrom"] == pytest.approx([1000.0, 1e5 / 3])
    assert law["dust_extinction_version"] and law["R_V"] == 3.1


def test_afav_is_none_without_dust_extinction(monkeypatch):
    monkeypatch.setitem(sys.modules, "dust_extinction", None)  # makes the import fail
    monkeypatch.setitem(sys.modules, "dust_extinction.parameter_averages", None)
    monkeypatch.setattr(fp, "_f99_model", None)  # forget a model built by an earlier test
    with pytest.warns(UserWarning, match="dust_extinction"):
        p = filter_properties(*gaussian(5500.0, 300.0), "photon")
    assert p["AfAv"] is None and p["WavelengthPivot"] > 0
    assert extinction_ratio(5500.0) is None
    assert "valid_range_angstrom" not in extinction_law()


def test_helpers_pass_the_sun_through(tmp_path):
    inst = tmp_path / "Test" / "Camera"
    inst.mkdir(parents=True)
    lam, t = gaussian(5500.0, 300.0)
    write_filter_file(inst / "V.dat", lam, t, detector_type="photon", source="svo")
    sun = (np.linspace(1000.0, 30000.0, 100), np.full(100, 185.0))
    assert curve_properties(inst / "V.dat", sun=sun)["Fsun"] == pytest.approx(185.0)
    assert instrument_properties(inst, sun=sun)["V"]["Fsun"] == pytest.approx(185.0)
