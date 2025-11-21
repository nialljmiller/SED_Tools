"""High level API for working with pre-computed stellar SED libraries.

This module exposes the :class:`SED` facade together with helper classes that
wrap the existing command line tooling in a programmatic friendly interface.
Users can discover locally available model grids, load a specific grid into
memory and evaluate spectra or synthetic photometry directly from Python code.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .cli import menu, run_filters_flow, run_rebuild_flow, run_spectra_flow
from .models import (
    DATA_DIR_DEFAULT,
    FILTER_DIR_DEFAULT,
    STELLAR_DIR_DEFAULT,
    EvaluatedSED,
    ModelMatch,
    PhotometryResult,
    SED,
    SEDModel,
)

__version__ = "0.1.0"


def find_atmospheres(
    *,
    teff_range: Optional[Sequence[float]] = None,
    logg_range: Optional[Sequence[float]] = None,
    metallicity_range: Optional[Sequence[float]] = None,
    limit: Optional[int] = None,
    allow_partial: bool = False,
) -> list[ModelMatch]:
    """Discover flux cubes compatible with the provided parameter ranges.

    This is a thin convenience wrapper around :meth:`SED.find_atmospheres`
    that constructs an :class:`SED` with default paths and delegates to it.
    """

    sed = SED()
    return sed.find_atmospheres(
        teff_range=teff_range,
        logg_range=logg_range,
        metallicity_range=metallicity_range,
        limit=limit,
        allow_partial=allow_partial,
    )


def find_atm(**kwargs) -> list[ModelMatch]:
    """Alias for :func:`find_atmospheres` matching the historical API sketch."""

    for legacy_key in ("Z_range", "z_range"):
        if legacy_key in kwargs and "metallicity_range" not in kwargs:
            kwargs["metallicity_range"] = kwargs.pop(legacy_key)
    if "logg_range" not in kwargs and "log_g_range" in kwargs:
        kwargs["logg_range"] = kwargs.pop("log_g_range")
    return find_atmospheres(**kwargs)


__all__ = [
    "SED",
    "SEDModel",
    "EvaluatedSED",
    "PhotometryResult",
    "ModelMatch",
    "DATA_DIR_DEFAULT",
    "STELLAR_DIR_DEFAULT",
    "FILTER_DIR_DEFAULT",
    "find_atmospheres",
    "find_atm",
    "menu",
    "run_filters_flow",
    "run_rebuild_flow",
    "run_spectra_flow",
    "__version__",
]
