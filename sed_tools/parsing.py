"""Old import path; the code is in sed_tools/ui_utils.py."""

from . import _old_path
from . import ui_utils as _module

_old_path.forward(__name__, _module)
