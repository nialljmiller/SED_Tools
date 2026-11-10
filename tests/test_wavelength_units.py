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


@pytest.mark.parametrize("header, argument, rows, factor, source, warning", [
    pytest.param("", None, [(4600.0, 0.0), (5500.0, 1.0), (7400.0, 0.0)], 1.0, "inferred",
                 "declares no wavelength unit", id="optical_filter_without_header_in_angstrom"),
    *[pytest.param(f"# WavelengthUnit: {unit}\n", None, [(2.0, 1.0), (3.0, 1.0)], factor, "header", "",
                   id=f"header-{unit}")
      for unit, factor in [("Angstrom", 1.0), ("nm", 10.0), ("um", 1e4), ("Å", 1.0), ("microns", 1e4)]],
    # A K-band curve in nm is indistinguishable from a UV curve in A by value alone.
    pytest.param("", "nm", [(1950.0, 1.0), (2400.0, 1.0)], 10.0, "argument", "", id="argument_overrides_inference"),
])
def test_filter_curve_wavelength_unit(tmp_path, caplog, header, argument, rows, factor, source, warning):
    path = _write(tmp_path / "f.dat", rows, header=header)

    with caplog.at_level(logging.WARNING):
        curve = load_filter_curve(path, wavelength_unit=argument)

    np.testing.assert_allclose(curve.wavelength, [w * factor for w, _ in rows])
    assert source is None or curve.metadata["wavelength_unit_source"] == source
    assert warning in caplog.text


def test_spectrum_accepts_explicit_unit(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("500 1.0\n600 2.0\n")

    spec = load_spectrum(str(path), wavelength_unit="angstrom")

    np.testing.assert_allclose(spec.wavelength, [500.0, 600.0])
    with pytest.raises(ValueError, match="Unknown wavelength unit 'nm \\(air\\)': declare WavelengthUnit as"):
        load_spectrum(str(path), wavelength_unit="nm (air)")
