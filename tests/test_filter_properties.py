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
    curve_properties,
    extinction_law,
    extinction_ratio,
    filter_properties,
    instrument_properties,
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


def test_rejects_unusable_curves():
    with pytest.raises(ValueError):
        filter_properties([5000.0], [1.0])
    with pytest.raises(ValueError):
        filter_properties([4000.0, 5000.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        filter_properties([4000.0, 5000.0], [1.0, 1.0], detector_type="quantum")


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
