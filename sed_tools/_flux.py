"""Old import path; the code is in sed_tools/cube/tool.py, which also re-exports the photometry names."""

from . import _old_path
from .cube import tool as _module

# What `from sed_tools._flux import *` gave in 0.5.0.
__all__ = """
AB_ZERO_FLUX FILTER_EXTENSIONS FilterCurve FluxCube Spectrum VEGA_ZP_KEYS band_average_flux_lambda
band_average_flux_lambda_from_arrays band_average_flux_nu load_filter_curve load_spectrum
""".split()

_old_path.forward(__name__, _module)
