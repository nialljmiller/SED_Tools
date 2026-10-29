"""Old import path; the code is in sed_tools/cube/resample.py."""

from . import _old_path
from .cube import resample as _module

_old_path.forward(__name__, _module)
