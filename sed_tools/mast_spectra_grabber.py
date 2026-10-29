"""Old import path; the code is in sed_tools/grids/mast.py."""

from . import _old_path
from .grids import mast as _module

# What `from sed_tools.mast_spectra_grabber import *` gave in 0.5.0.
__all__ = """
Any DATA_BASE Dict HEAD_RE List MASTSpectraGrabber METALS Optional RES_KEYS SCRIPTS_DIR SCRIPT_URL_RE
Tuple annotations csv gzip io logger logging np os re requests string urllib zlib
""".split()

_old_path.forward(__name__, _module, csv="csv")
