SED_Tools documentation
=======================

SED_Tools downloads, standardizes and interpolates stellar spectral grids,
and prepares photometric filter curves and their optical-path components.

A spectral grid contains model spectra indexed by physical parameters.
Use ``sed-tools grids`` to download grids, ``SED.query()`` to discover them
from Python, and ``sed.grid`` to access a loaded ``SpectralGrid``.
Spectral libraries can also include observed templates, planetary spectra
and stellar population spectra. Earth's atmosphere transmission models
are a separate part of the photometric optical path.

The older ``spectra`` command, ``--models`` option, ``Catalog`` and
``CatalogInfo`` classes and ``sed.cat`` attribute remain supported.
The data directory retains its established ``stellar_models/`` name.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   sed_tools
   changelog

