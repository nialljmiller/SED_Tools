"""flux_cube.bin: four int32 axis lengths (Teff, log g, [M/H], wavelength), the four float64 axes, then the
float64 flux in (wavelength, [M/H], log g, Teff) order with Teff varying fastest, as MESA reads it.

Callers keep their own checks of a short or corrupt file, so each keeps the error it has always raised.
"""

import struct

import numpy as np


def flux_offset(nt, nl, nm, nw):
    """Byte offset where the flux starts."""
    return 16 + 8 * (nt + nl + nm + nw)


def read_axes(fh, nt, nl, nm, nw):
    """The Teff, log g, [M/H] and wavelength axes, read from ``fh`` just after the 16-byte header."""
    return [np.fromfile(fh, dtype=np.float64, count=n) for n in (nt, nl, nm, nw)]


def load(path):
    """Teff, log g, [M/H], wavelengths and the flux as stored, shape (nw, nm, nl, nt); nothing is checked."""
    with open(path, "rb") as fh:
        nt, nl, nm, nw = struct.unpack("4i", fh.read(16))
        teff, logg, meta, wavelengths = read_axes(fh, nt, nl, nm, nw)
        flux = np.fromfile(fh, dtype=np.float64, count=nt * nl * nm * nw).reshape((nw, nm, nl, nt))
    return teff, logg, meta, wavelengths, flux


def write_header(fh, teff, logg, meta, wavelengths):
    """The axis lengths and the four axes, as float64."""
    fh.write(struct.pack("4i", len(teff), len(logg), len(meta), len(wavelengths)))
    for axis in (teff, logg, meta, wavelengths):
        np.asarray(axis).astype(np.float64).tofile(fh)
