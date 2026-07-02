"""Public package API for SED Tools.

Implementation code lives in the package's focused modules.  This module only
defines the supported import surface and thin compatibility helpers.
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
from .spectrum_io import build_h5_bundle as build_h5_bundle_from_txt
from .spectrum_io import list_text_spectra, read_text_spectrum as load_txt_spectrum
from .mast_spectra_grabber import MASTSpectraGrabber
from .models import (
    DATA_DIR_DEFAULT,
    FILTER_DIR_DEFAULT,
    STELLAR_DIR_DEFAULT,
    EvaluatedSED,
    ModelMatch,
    PhotometryResult,
    SED as _SEDCore,
    SEDModel,
)
from .msg_spectra_grabber import MSGSpectraGrabber
from .njm_filter_grabber import NJMFilterGrabber
from .njm_spectra_grabber import NJMSpectraGrabber
from .precompute_flux_cube import precompute_flux_cube
from .spectra_cleaner import clean_model_dir
from .svo_regen_spectra_lookup import regenerate_lookup_table
from .svo_spectra_grabber import SVOSpectraGrabber

__version__ = "0.1.4"


def list_txt_spectra(model_dir: str) -> list[str]:
    """Compatibility wrapper returning filenames rather than Path objects."""
    return [path.name for path in list_text_spectra(model_dir)]


def find_atmospheres(
    *,
    teff_range: Optional[Sequence[float]] = None,
    logg_range: Optional[Sequence[float]] = None,
    metallicity_range: Optional[Sequence[float]] = None,
    limit: Optional[int] = None,
    allow_partial: bool = False,
    model_root: Optional[Union[str, os.PathLike[str]]] = None,
    filter_root: Optional[Union[str, os.PathLike[str]]] = None,
) -> list[ModelMatch]:
    """Discover local model grids matching the requested parameter ranges."""
    return _SEDCore(model_root=model_root, filter_root=filter_root).find_atmospheres(
        teff_range=teff_range,
        logg_range=logg_range,
        metallicity_range=metallicity_range,
        limit=limit,
        allow_partial=allow_partial,
    )


def find_atm(**kwargs) -> list[ModelMatch]:
    """Backward-compatible alias for :func:`find_atmospheres`."""
    for legacy_key in ("Z_range", "z_range"):
        if legacy_key in kwargs and "metallicity_range" not in kwargs:
            kwargs["metallicity_range"] = kwargs.pop(legacy_key)
    if "logg_range" not in kwargs and "log_g_range" in kwargs:
        kwargs["logg_range"] = kwargs.pop("log_g_range")
    return find_atmospheres(**kwargs)


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
    "build_h5_bundle_from_txt",
    "clean_model_dir",
    "ensure_dir",
    "fetch",
    "find_atm",
    "find_atmospheres",
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
