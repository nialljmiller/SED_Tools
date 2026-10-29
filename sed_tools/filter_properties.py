"""Old import path; the code is in sed_tools/filters/properties.py."""

from . import _old_path
from .filters import properties as _module

# What `from sed_tools.filter_properties import *` gave in 0.5.0.
__all__ = """
AB_JY Dict EXTINCTION_LAW EXTINCTION_RV JY Optional Path PathLike SPEED_OF_LIGHT_ANG_PER_S ST_FLAM
Spectrum Tuple UNITS Union annotations curve_properties extinction_law extinction_ratio
filter_properties filter_weight instrument_properties load_filter_curve load_spectrum load_sun load_vega
math metadata np read_filter_header warnings
""".split()

_old_path.forward(__name__, _module)
