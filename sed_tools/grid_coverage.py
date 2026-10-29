"""Old import path; the code is in sed_tools/grids/coverage.py."""

from . import _old_path
from .grids import coverage as _module

_old_path.forward(__name__, _module)
