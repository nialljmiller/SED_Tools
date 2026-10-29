"""Old import path; the code is in sed_tools/grids/lookup.py."""

from . import _old_path
from .grids import lookup as _module

# What `from sed_tools._lookup_io import *` gave in 0.5.0.
__all__ = """
Dict Iterable List Optional Union annotations csv find_column find_file_column find_logg_column
find_metallicity_column find_teff_column logger logging write_lookup_csv
""".split()

_old_path.forward(__name__, _module, annotations="__future__:annotations")
