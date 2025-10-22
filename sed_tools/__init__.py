"""High level API for working with pre-computed stellar SED libraries.

This module exposes the :class:`SED` facade together with helper classes that
wrap the existing command line tooling in a programmatic friendly interface.
Users can discover locally available model grids, load a specific grid into
memory and evaluate spectra or synthetic photometry directly from Python code.

Typical usage::

    from sed_tools import SED

    sed = SED()                             # discover default data directories
    matches = sed.find_model(5777, 4.44, 0) # inspect available grids
    model = sed.model(matches[0].name)
    spec = model(5777, 4.44, 0.0)           # interpolate a spectrum
    gaia = spec.photometry("GAIA")         # synthetic GAIA magnitudes

For range-based discovery a convenience wrapper is also provided::

    from sed_tools import find_atmospheres

    grids = find_atmospheres(teff_range=(5000, 6500), metallicity_range=(-0.5, 0.5))

The package reuses the heavy lifting that already powers the interactive tools
shipped with SED Tools so that workflows built on the CLI continue to operate
unchanged while pipelines can opt into the same functionality via imports.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import os

import sys

from .models import (
    SED,
    SEDModel,
    EvaluatedSED,
    PhotometryResult,
    ModelMatch,
    STELLAR_DIR_DEFAULT,
    FILTER_DIR_DEFAULT,
)

__version__ = "0.1.0"


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

    sed = SED(model_root=model_root, filter_root=filter_root)
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
    "STELLAR_DIR_DEFAULT",
    "FILTER_DIR_DEFAULT",
    "find_atmospheres",
    "find_atm",
    "__version__",
]


# Expose a camel-cased import alias so that ``import SED_tools`` works as sketched
# in the original API proposal without breaking the historical script module.
sys.modules.setdefault("SED_tools", sys.modules[__name__])
