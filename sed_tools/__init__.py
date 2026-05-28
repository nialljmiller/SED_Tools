"""Public package interface for SED Tools."""

from __future__ import annotations

import os
from importlib import metadata
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
from .io_utils import (
    build_h5_bundle_from_txt,
    ensure_dir,
    list_txt_spectra,
    load_txt_spectrum,
)
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

try:
    __version__ = metadata.version("sed-tools")
except metadata.PackageNotFoundError:
    __version__ = "0.1.2"


def run_rebuild_flow(*args, **kwargs):
    """Run the CLI rebuild workflow without importing CLI machinery at startup."""
    from .cli import run_rebuild_flow as _run_rebuild_flow

    return _run_rebuild_flow(*args, **kwargs)


def run_spectra_flow(*args, **kwargs):
    """Run the CLI spectra download workflow."""
    from .cli import run_spectra_flow as _run_spectra_flow

    return _run_spectra_flow(*args, **kwargs)


def run_filters_flow(*args, **kwargs):
    """Run the CLI filter download workflow."""
    from .cli import run_filters_flow as _run_filters_flow

    return _run_filters_flow(*args, **kwargs)


def run_combine_flow(*args, **kwargs):
    """Run the CLI grid-combination workflow."""
    from .cli import run_combine_flow as _run_combine_flow

    return _run_combine_flow(*args, **kwargs)


def menu(*args, **kwargs):
    """Display the CLI menu."""
    from .cli import menu as _menu

    return _menu(*args, **kwargs)


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
    """Discover locally available model grids matching the requested ranges."""
    sed = _SEDCore(model_root=model_root, filter_root=filter_root)
    return sed.find_atmospheres(
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
    "SED",
    "Catalog",
    "CatalogInfo",
    "Spectrum",
    "Filters",
    "MLCompleter",
    "query",
    "fetch",
    "local",
    "SEDModel",
    "EvaluatedSED",
    "PhotometryResult",
    "ModelMatch",
    "DATA_DIR_DEFAULT",
    "STELLAR_DIR_DEFAULT",
    "FILTER_DIR_DEFAULT",
    "ensure_dir",
    "list_txt_spectra",
    "load_txt_spectrum",
    "build_h5_bundle_from_txt",
    "run_rebuild_flow",
    "run_spectra_flow",
    "run_filters_flow",
    "run_combine_flow",
    "menu",
    "find_atmospheres",
    "find_atm",
    "__version__",
]
