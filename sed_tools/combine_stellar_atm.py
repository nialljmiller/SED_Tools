"""Old import path; the code is in sed_tools/cube/combine.py."""

from . import _old_path
from .cube import combine as _module

# What `from sed_tools.combine_stellar_atm import *` gave in 0.5.0.
__all__ = """
Counter argparse build_combined_flux_cube compute_normalization_factors create_common_wavelength_grid
create_unified_grid find_stellar_models identify_reference_model load_model_data load_sed logger logging
main np os pd plot_sed_samples plt prepare_sed read_text_spectrum resample_to_grid save_combined_data
select_models_interactive shutil struct tqdm visualize_parameter_space
""".split()

if __name__ == "__main__":
    _module.main()
else:
    _old_path.forward(__name__, _module, struct="struct")
