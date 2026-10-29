"""Old import path; the code is in sed_tools/cube/collision.py."""

from . import _old_path
from .cube import collision as _module

_old_path.forward(__name__, _module)
