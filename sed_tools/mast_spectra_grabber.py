"""Old import path; the code is in sed_tools/grids/mast.py."""

from . import _old_path
from .grids import mast as _module

_old_path.forward(__name__, _module)
