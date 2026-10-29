"""Old import path; the code is in sed_tools/grids/spectrum_io.py."""

from . import _old_path
from .grids import spectrum_io as _module

_old_path.forward(__name__, _module)
