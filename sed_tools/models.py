"""Old import path; the code is in sed_tools/cube/model.py."""

from . import _old_path
from .cube import model as _module

# What `from sed_tools.models import *` gave in 0.5.0.
__all__ = """
EvaluatedSED FILTER_DIR_DEFAULT ModelMatch PhotometryResult SED SEDModel STELLAR_DIR_DEFAULT
""".split()

_old_path.forward(__name__, _module)
