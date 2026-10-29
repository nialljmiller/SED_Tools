"""Interim import path (sed_tools.api uses it); the plotting code is in sed_tools/filters/display.py."""

from .. import _old_path
from . import display as _module

_old_path.forward(__name__, _module)
