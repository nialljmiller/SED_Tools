"""Old import path; the code is in sed_tools/grids/clean.py."""

from . import _old_path
from .grids import clean as _module

_old_path.forward(__name__, _module)
