"""Old import path; the code is in sed_tools/filters/sites.py."""

from . import _old_path
from .filters import sites as _module

# What `from sed_tools.facility_sites import *` gave in 0.5.0.
__all__ = """
AIRBORNE GROUND Optional SPACE Tuple annotations instrument_sites is_ground_based own_site platform re
""".split()

_old_path.forward(__name__, _module)
