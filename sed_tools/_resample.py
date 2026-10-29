"""Old import path; the code is in sed_tools/cube/build.py."""

from . import _old_path
from .cube import build as _module

_old_path.forward(__name__, _module)
