"""Interim import path (cube/build.py and api import from it); the code is in sed_tools/grids/lookup.py."""

from .. import _old_path
from . import lookup as _module

_old_path.forward(__name__, _module)
