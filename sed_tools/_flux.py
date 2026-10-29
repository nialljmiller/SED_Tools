"""Old import path; the code is in sed_tools/cube/tool.py, which also re-exports the photometry names."""

from . import _old_path
from .cube import tool as _module

# The old module's own list: forward() keeps dunder names, so `from sed_tools._flux import *` still gives these.
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

_old_path.forward(__name__, _module)
