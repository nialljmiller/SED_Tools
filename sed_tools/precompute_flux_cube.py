"""Old import path; the code is in sed_tools/cube/build.py, which uses cube/variants.py."""

from . import _old_path
from .cube import build as _module

# What `from sed_tools.precompute_flux_cube import *` gave in 0.5.0.
__all__ = """
Any Dict List Optional Path SPLIT_THRESHOLD Tuple argparse build_grids compute_common_wavelengths
copy_global_config_to_model csv discover_extra_axes find_file_column find_logg_column
find_metallicity_column find_teff_column load_config load_lookup_table load_sed logger logging math np
os precompute_flux_cube re read_text_spectrum resample_to_grid shutil spectra_not_in_cube struct sys
time write_lookup_csv
""".split()

if __name__ == "__main__":
    _module._main()
else:
    _old_path.forward(__name__, _module, re="re", shutil="shutil")
