"""Public package API for SED Tools: the supported import surface and thin compatibility helpers.

The code is in the subpackages (api, cli, filters, grids, cube, ml) and the old module paths forward to them.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Union

from .api import (
    Catalog,
    CatalogInfo,
    Filters,
    MLCompleter,
    SED,
    Spectrum,
    SpectralGrid,
    SpectralGridInfo,
    fetch,
    local,
    query,
)
from .cli import (
    ensure_dir,
    menu,
    run_combine_flow,
    run_filters_flow,
    run_rebuild_flow,
    run_spectra_flow,
)
from .grids.spectrum_io import build_h5_bundle as build_h5_bundle_from_txt
from .grids.spectrum_io import list_text_spectra, read_text_spectrum as load_txt_spectrum
from .grids.mast import MASTSpectraGrabber
from .cube.model import (
    DATA_DIR_DEFAULT,
    FILTER_DIR_DEFAULT,
    STELLAR_DIR_DEFAULT,
    EvaluatedSED,
    ModelMatch,
    PhotometryResult,
    SED as _SEDCore,
    SEDModel,
)
from .grids.msg import MSGSpectraGrabber
from .filters.njm import NJMFilterGrabber
from .grids.njm import NJMSpectraGrabber
from .grids.clean import clean_model_dir
from .grids.lookup import regenerate_lookup_table
from .grids.svo import SVOSpectraGrabber

# Through the old path, as before the move, so the package attribute is the function, not the module.
from .precompute_flux_cube import precompute_flux_cube

# The old module paths that `import sed_tools` has always made attributes of the package.
from . import (_constants, _flux, _lookup_io, _resample, collision_config, config, facility_sites,  # noqa: F401
               filter_catalogue, filter_components, filter_metadata, flux_cube_tool, header_parser,
               mast_spectra_grabber, models, msg_spectra_grabber, njm_filter_grabber, njm_spectra_grabber,
               parsing, spectra_cleaner, spectrum_io, svo_regen_spectra_lookup, svo_spectra_grabber,
               terminal_plots, ui_utils)

__version__ = "0.5.0"


def list_txt_spectra(model_dir: str) -> list[str]:
    """Compatibility wrapper returning filenames rather than Path objects."""
    return [path.name for path in list_text_spectra(model_dir)]


def find_grids(
    *,
    teff_range: Optional[Sequence[float]] = None,
    logg_range: Optional[Sequence[float]] = None,
    metallicity_range: Optional[Sequence[float]] = None,
    limit: Optional[int] = None,
    allow_partial: bool = False,
    model_root: Optional[Union[str, os.PathLike[str]]] = None,
    filter_root: Optional[Union[str, os.PathLike[str]]] = None,
) -> list[ModelMatch]:
    """Discover local spectral grids matching the requested parameter ranges."""
    return _SEDCore(model_root=model_root, filter_root=filter_root).find_grids(
        teff_range=teff_range,
        logg_range=logg_range,
        metallicity_range=metallicity_range,
        limit=limit,
        allow_partial=allow_partial,
    )


find_atmospheres = find_grids


def find_atm(**kwargs) -> list[ModelMatch]:
    """Backward-compatible alias for :func:`find_grids`."""
    for legacy_key in ("Z_range", "z_range"):
        if legacy_key in kwargs and "metallicity_range" not in kwargs:
            kwargs["metallicity_range"] = kwargs.pop(legacy_key)
    if "logg_range" not in kwargs and "log_g_range" in kwargs:
        kwargs["logg_range"] = kwargs.pop("log_g_range")
    return find_grids(**kwargs)


__all__ = [
    "Catalog",
    "CatalogInfo",
    "DATA_DIR_DEFAULT",
    "EvaluatedSED",
    "FILTER_DIR_DEFAULT",
    "Filters",
    "MASTSpectraGrabber",
    "MLCompleter",
    "MSGSpectraGrabber",
    "ModelMatch",
    "NJMFilterGrabber",
    "NJMSpectraGrabber",
    "PhotometryResult",
    "SED",
    "SEDModel",
    "STELLAR_DIR_DEFAULT",
    "SVOSpectraGrabber",
    "Spectrum",
    "SpectralGrid",
    "SpectralGridInfo",
    "build_h5_bundle_from_txt",
    "clean_model_dir",
    "ensure_dir",
    "fetch",
    "find_atm",
    "find_atmospheres",
    "find_grids",
    "list_txt_spectra",
    "load_txt_spectrum",
    "local",
    "menu",
    "precompute_flux_cube",
    "query",
    "regenerate_lookup_table",
    "run_combine_flow",
    "run_filters_flow",
    "run_rebuild_flow",
    "run_spectra_flow",
]
