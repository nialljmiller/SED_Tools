"""Old import path; the code is in sed_tools/grids/header_parser.py."""

from . import _old_path
from .grids import header_parser as _module

_old_path.forward(__name__, _module)
