"""Old import path; the code is in sed_tools/grids/coverage.py."""

from . import _old_path
from .grids import coverage as _module

# What `from sed_tools.grid_coverage import *` gave in 0.5.0.
__all__ = """
Dict List Optional Path Union annotations coverage_summary grid_coverage math np parse_header pd
plot_coverage print_coverage print_terminal_coverage re read_grid_nodes read_text_spectrum
""".split()

_old_path.forward(__name__, _module)
