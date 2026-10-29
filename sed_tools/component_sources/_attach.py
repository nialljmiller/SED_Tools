"""Old import path; the code is in sed_tools/component_sources/_common.py."""

from .. import _old_path
from . import _common as _module

_old_path.forward(__name__, _module)
