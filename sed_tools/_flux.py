"""Thin wrappers around the legacy flux cube utilities.

The command line tool :mod:`flux_cube_tool` already implements robust handling
of the binary ``flux_cube.bin`` format together with filter parsing helpers. To
avoid duplicating mature logic we import and re-export the relevant classes and
functions here.
"""
from __future__ import annotations

from .flux_cube_tool import FluxCube
from .photometry import (AB_ZERO_FLUX, FILTER_EXTENSIONS, VEGA_ZP_KEYS, FilterCurve, Spectrum,
                         band_average_flux_lambda, band_average_flux_lambda_from_arrays, band_average_flux_nu,
                         load_filter_curve, load_spectrum)

__all__ = [
    "AB_ZERO_FLUX",
    "FILTER_EXTENSIONS",
    "FilterCurve",
    "FluxCube",
    "Spectrum",
    "VEGA_ZP_KEYS",
    "band_average_flux_lambda",
    "band_average_flux_lambda_from_arrays",
    "band_average_flux_nu",
    "load_filter_curve",
    "load_spectrum",
]
