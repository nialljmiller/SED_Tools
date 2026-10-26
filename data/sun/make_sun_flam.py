#!/usr/bin/env python3
"""Write sun_flam.csv, the solar spectrum NJM uses for Fsun, from the two CALSPEC files here.

sun_reference_stis_002 for wavelengths below 9600 Angstrom and sun_mod_001 from 9600 Angstrom
on (README.md says why).  Rows are copied unchanged: no interpolation, scaling or smoothing.
Values are written with the fewest digits that read back to the FITS value (float32 columns
at float32 precision).  Usage: python3 make_sun_flam.py [output.csv]
"""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
SPLICE = 9600.0  # Angstrom


def _wavelength(value):
    if isinstance(value, np.float32):
        return np.format_float_positional(value, unique=True, trim="-")
    return repr(float(value))


def _flux(value):
    return np.format_float_scientific(np.float32(value), unique=True, trim="-")


def composite():
    """Rows (wavelength, flux) of the composite, with the native FITS scalar types."""
    stis = fits.getdata(HERE / "sun_reference_stis_002.fits", 1)
    model = fits.getdata(HERE / "sun_mod_001.fits", 1)
    rows = [(w, f) for w, f in zip(stis["WAVELENGTH"], stis["FLUX"]) if w < SPLICE]
    rows += [(w, f) for w, f in zip(model["WAVELENGTH"], model["FLUX"]) if w >= SPLICE]
    wavelength = np.array([float(w) for w, _ in rows])
    assert np.all(np.diff(wavelength) > 0), "the composite must be strictly increasing"
    assert all(f > 0 for _, f in rows), "every flux must be positive"
    assert all(isinstance(f, np.float32) for _, f in rows), "both FLUX columns are float32"
    return rows


def main(out=HERE / "sun_flam.csv"):
    lines = ["WAVELENGTH,FLUX"] + [f"{_wavelength(w)},{_flux(f)}" for w, f in composite()]
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(lines) - 1} rows")


if __name__ == "__main__":
    main(*sys.argv[1:])
