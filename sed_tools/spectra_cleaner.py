"""Old import path; the code is in sed_tools/grids/clean.py."""

from . import _old_path
from .grids import clean as _module

# What `from sed_tools.spectra_cleaner import *` gave in 0.5.0.
__all__ = """
C Counter Dict FLUX_GUESS_WHY FLUX_UNITS List Optional SIGMA SPEED_OF_LIGHT_ANGSTROM
STANDARD_HEADER_TEMPLATE SpectrumMeta Tuple UnitInfo WAVELENGTH_FACTORS WAVE_KEYS annotations
check_flux_unit clean_model_dir clean_spectrum_file convert_to_standard_units dataclass dataclasses
defaultdict detect_catalog_units detect_flux_unit_from_range detect_units_from_header
detect_units_single_file detect_wavelength_unit_from_range glob h5py is_index_grid logger logging np os
planck_flam random re read_spectrum_data read_text_spectrum rebuild_lookup_table
recover_wavelengths_from_hdf5 time undeclared_flux_unit validate_converted_spectrum write_lookup_csv
write_standardized_spectrum
""".split()

_old_path.forward(
    __name__, _module,
    Counter="collections:Counter", dataclass="dataclasses:dataclass", dataclasses="dataclasses", h5py="h5py", re="re")
