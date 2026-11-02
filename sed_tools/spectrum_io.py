"""Old import path; the code is in sed_tools/grids/spectrum_io.py."""

from . import _old_path
from .grids import spectrum_io as _module

# What `from sed_tools.spectrum_io import *` gave in 0.5.0.
__all__ = """
Path PathLike Union annotations build_h5_bundle bundled_spectra h5py list_text_spectra np parse_header
read_text_spectrum
""".split()

_old_path.forward(__name__, _module)
