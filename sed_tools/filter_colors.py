"""Old import path; the code is in sed_tools/filters/display.py."""

from . import _old_path
from .filters import display as _module

# What `from sed_tools.filter_colors import *` gave in 0.5.0.
__all__ = """
CIE_DIR Curve Dict INFRARED_RAINBOW Mapping Path Sequence TRUE_COLOR_FRACTION Tuple ULTRAVIOLET_RAINBOW
Union VISIBLE_ANGSTROM XYZ_TO_SRGB annotations colorsys display_colors instrument_colors lru_cache np
true_color visible_fraction
""".split()

_old_path.forward(__name__, _module)
