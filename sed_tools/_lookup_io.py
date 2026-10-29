"""Old import path; the code is in sed_tools/grids/_lookup_io.py."""

from . import _old_path
from .grids import _lookup_io as _module

_old_path.forward(__name__, _module)
