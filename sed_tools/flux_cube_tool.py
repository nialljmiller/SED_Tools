"""Old import path and plain script; the code is in sed_tools/cube/tool.py."""

if __package__:
    from . import _old_path
    from .cube import tool as _module
else:  # run as a plain script: python sed_tools/flux_cube_tool.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sed_tools.cube import tool as _module

# What `from sed_tools.flux_cube_tool import *` gave in 0.5.0.
__all__ = """
AB_ZERO_FLUX Dict ENERGY FILTER_EXTENSIONS FilterCurve FluxCube Iterable List PC_TO_M PHOTON Path
RSUN_TO_M SPEED_OF_LIGHT_ANG_PER_S Sequence Spectrum Tuple UNIT_TO_ANGSTROM VEGA_ZP_KEYS annotations
argparse band_average_flux_lambda band_average_flux_lambda_from_arrays band_average_flux_nu
bolometric_magnitude collect_filter_files dataclass discover_flux_cube_files discover_vega_candidates
ensure_dir extract_filter_system filter_weight get_metadata_float infer_filter_directories
infer_unit_from_values integrate_flux load_filter_curve load_spectrum load_two_column_file logger
logging main math normalize_detector_type normalize_metadata_key np os parse_metadata_value plot_sed plt
prompt_distance_m prompt_for_filter_system prompt_for_flux_cube_path prompt_for_parameter
prompt_for_photometric_system prompt_radius_m prompt_scale_choice re read_filter_header
resolve_flux_cube_path resolve_vega_spectrum_path save_sed standardize_wavelength_array struct
""".split()

if __name__ == "__main__":
    raise SystemExit(_module.main())
else:
    _old_path.forward(__name__, _module, re="re")
