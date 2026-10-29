"""Old import path; the code is in sed_tools/grids/njm.py."""

from . import _old_path
from .grids import njm as _module

_old_path.forward(__name__, _module)
