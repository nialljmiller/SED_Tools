"""Old import path; the code is in sed_tools/filters/njm_catalogue.py."""

from . import _old_path
from .filters import njm_catalogue as _module

_old_path.forward(__name__, _module)
