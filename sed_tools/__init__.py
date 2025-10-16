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

The package reuses the heavy lifting that already powers the interactive tools
shipped with SED Tools so that workflows built on the CLI continue to operate
unchanged while pipelines can opt into the same functionality via imports.
"""

from .models import (
    SED,
    SEDModel,
    EvaluatedSED,
    PhotometryResult,
    ModelMatch,
    STELLAR_DIR_DEFAULT,
    FILTER_DIR_DEFAULT,
)

__all__ = [
    "SED",
    "SEDModel",
    "EvaluatedSED",
    "PhotometryResult",
    "ModelMatch",
    "STELLAR_DIR_DEFAULT",
    "FILTER_DIR_DEFAULT",
]
