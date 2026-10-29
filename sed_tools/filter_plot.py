"""Old import path; the code is in sed_tools/filters/display.py."""

from . import _old_path
from .filters import display as _module

# What `from sed_tools.filter_plot import *` gave in 0.5.0.
__all__ = """
Curve Mapping NORMALIZATIONS Optional Sequence Tuple Union annotations display_colors np plot_curves
""".split()

_old_path.forward(__name__, _module)
