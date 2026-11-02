"""Old import path; the code is in sed_tools/ui_utils.py."""

from . import _old_path
from . import ui_utils as _module

# What `from sed_tools.parsing import *` gave in 0.5.0.
__all__ = """
Optional Tuple annotations parse_multi_selection parse_numeric_range re
""".split()

_old_path.forward(__name__, _module, annotations="__future__:annotations")
