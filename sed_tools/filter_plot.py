"""Old import path; the code is in sed_tools/filters/display.py."""

from . import _old_path
from .filters import display as _module

_old_path.forward(__name__, _module)
