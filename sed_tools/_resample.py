"""Old import path; the code is in sed_tools/cube/build.py."""

from . import _old_path
from .cube import build as _module

# What `from sed_tools._resample import *` gave in 0.5.0.
__all__ = """
annotations np resample_to_grid
""".split()

_old_path.forward(__name__, _module, annotations="__future__:annotations")
