"""Old import path; the code is in sed_tools/grids/lookup.py."""

from . import _old_path
from .grids import lookup as _module

# What `from sed_tools.svo_regen_spectra_lookup import *` gave in 0.5.0.
__all__ = """
Iterable List Optional csv main math os parse_header regenerate_lookup_table scan_existing_files
""".split()

if __name__ == "__main__":
    _module.main()
else:
    _old_path.forward(__name__, _module)
