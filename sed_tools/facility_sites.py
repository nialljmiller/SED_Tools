"""Old import path; the code is in sed_tools/filters/sites.py."""

from . import _old_path
from .filters import sites as _module

_old_path.forward(__name__, _module)
