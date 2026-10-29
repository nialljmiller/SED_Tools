"""Old import path; the code is in sed_tools/filters/curve_metadata.py."""

from . import _old_path
from .filters import curve_metadata as _module

_old_path.forward(__name__, _module)
