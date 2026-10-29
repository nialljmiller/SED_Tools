"""Old import path; the code is in sed_tools/cube/model.py."""

from . import _old_path
from .cube import model as _module

_old_path.forward(__name__, _module)
