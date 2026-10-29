"""Old import path; the code is in sed_tools/filters/library.py."""

from . import _old_path
from .filters import library as _module

_old_path.forward(__name__, _module)
