"""Wavelength-unit handling when loading filter curves and spectra."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from sed_tools.flux_cube_tool import (infer_unit_from_values, load_filter_curve,
                                      load_spectrum)


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (3200.0, 10800.0, "angstrom"),   # Gaia G in A (was misread as nm)
        (4600.0, 7400.0, "angstrom"),    # Johnson V in A
        (1340.0, 1806.0, "angstrom"),    # GALEX FUV in A
        (6000.0, 2.8e5, "angstrom"),     # JWST MIRI in A
        (900.0, 3.0e6, "angstrom"),      # full-range stellar spectrum in A
        (298.0, 413.0, "nanometer"),     # SDSS u in nm
        (460.0, 740.0, "nanometer"),     # Johnson V in nm
        (0.3, 28.0, "micrometer"),
        (5e-7, 1e-6, "meter"),
    ],
)
def test_infer_unit_from_values(lo, hi, expected):
    assert infer_unit_from_values(np.array([lo, hi])) == expected


def _write(path, rows, header=""):
    path.write_text(header + "Wavelength,Transmission\n" + "".join(f"{w},{t}\n" for w, t in rows))
    return str(path)


def test_optical_filter_without_header_loads_in_angstrom(tmp_path, caplog):
    path = _write(tmp_path / "V.dat", [(4600.0, 0.0), (5500.0, 1.0), (7400.0, 0.0)])

    with caplog.at_level(logging.WARNING):
        curve = load_filter_curve(path)

    np.testing.assert_allclose(curve.wavelength, [4600.0, 5500.0, 7400.0])
    assert curve.metadata["wavelength_unit_source"] == "inferred"
    assert "declares no wavelength unit" in caplog.text


@pytest.mark.parametrize("unit,factor", [("Angstrom", 1.0), ("nm", 10.0), ("um", 1e4), ("Å", 1.0), ("microns", 1e4)])
def test_header_unit_is_honoured(tmp_path, unit, factor):
    path = _write(tmp_path / "f.dat", [(2.0, 1.0), (3.0, 1.0)], header=f"# WavelengthUnit: {unit}\n")

    curve = load_filter_curve(path)

    np.testing.assert_allclose(curve.wavelength, [2.0 * factor, 3.0 * factor])
    assert curve.metadata["wavelength_unit_source"] == "header"


def test_argument_overrides_header_and_inference(tmp_path):
    # A K-band curve in nm is indistinguishable from a UV curve in A by value alone.
    path = _write(tmp_path / "K.dat", [(1950.0, 1.0), (2400.0, 1.0)])

    curve = load_filter_curve(path, wavelength_unit="nm")

    np.testing.assert_allclose(curve.wavelength, [19500.0, 24000.0])
    assert curve.metadata["wavelength_unit_source"] == "argument"


def test_unknown_header_unit_falls_back_to_inference(tmp_path, caplog):
    path = _write(tmp_path / "f.dat", [(4000.0, 1.0), (5000.0, 1.0)], header="# WavelengthUnit: furlongs\n")

    with caplog.at_level(logging.WARNING):
        curve = load_filter_curve(path)

    np.testing.assert_allclose(curve.wavelength, [4000.0, 5000.0])
    assert "Unrecognised wavelength unit" in caplog.text


def test_spectrum_accepts_explicit_unit(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("500 1.0\n600 2.0\n")

    spec = load_spectrum(str(path), wavelength_unit="angstrom")

    np.testing.assert_allclose(spec.wavelength, [500.0, 600.0])
