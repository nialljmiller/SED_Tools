"""Old import path; the code is in sed_tools/grids/header_parser.py."""

from . import _old_path
from .grids import header_parser as _module

# What `from sed_tools.header_parser import *` gave in 0.5.0.
__all__ = """
ALIASES Dict annotations math parse_header re
""".split()

_old_path.forward(__name__, _module)
