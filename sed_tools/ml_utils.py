"""Old import path; the code is in sed_tools/ml/common.py."""

from . import _old_path
from .ml import common as _module

_old_path.forward(__name__, _module)
