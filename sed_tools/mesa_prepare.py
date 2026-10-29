"""Old import path; the code is in sed_tools/cube/mesa_export.py."""

from . import _old_path
from .cube import mesa_export as _module

_old_path.forward(__name__, _module)
