"""Old import path; the code is in sed_tools/grids/lookup.py."""

from . import _old_path
from .grids import lookup as _module

_old_path.forward(__name__, _module)
