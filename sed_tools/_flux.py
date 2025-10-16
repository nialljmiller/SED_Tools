"""Thin wrappers around the legacy flux cube utilities.

The command line tool :mod:`flux_cube_tool` already implements robust handling
of the binary ``flux_cube.bin`` format together with filter parsing helpers. To
avoid duplicating mature logic we import and re-export the relevant classes and
functions here.  Import errors are turned into actionable messages so that the
package fails gracefully when optional dependencies (e.g. ``matplotlib``) are
missing.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import error path depends on user environment
    from flux_cube_tool import (  # type: ignore
        AB_ZERO_FLUX,
        FILTER_EXTENSIONS,
        FilterCurve,
        FluxCube,
        Spectrum,
        VEGA_ZP_KEYS,
        band_average_flux_lambda,
        band_average_flux_lambda_from_arrays,
        band_average_flux_nu,
        load_filter_curve,
        load_spectrum,
    )
except Exception as exc:  # pragma: no cover - surfaced to users
    raise ImportError(
        "The sed_tools package requires the bundled 'flux_cube_tool' module. "
        "Ensure the SED Tools repository is available on PYTHONPATH and that "
        "its optional dependencies are installed."
    ) from exc

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
