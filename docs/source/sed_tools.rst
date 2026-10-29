sed\_tools package
==================

The code is in the subpackages below, one page section per module.
The old module paths (``sed_tools.filter_library``, ``sed_tools.spectra_cleaner``,
``sed_tools.precompute_flux_cube`` and the rest) still import: each forwards to the
module that now holds its code, so they are not listed separately.

Public Python API
-----------------

sed\_tools.api module
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.api
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.api.filters module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.api.filters
   :members:
   :show-inheritance:
   :undoc-members:

Command line
------------

sed\_tools.cli module
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cli
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.cli.grids module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cli.grids
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.cli.filters module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cli.filters
   :members:
   :show-inheritance:
   :undoc-members:

Filter curves
-------------

.. automodule:: sed_tools.filters

sed\_tools.filters.library module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.library
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.names module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.names
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.manifest module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.manifest
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.compose module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.compose
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.curve\_metadata module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.curve_metadata
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.njm module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.njm
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.njm\_catalogue module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.njm_catalogue
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.svo module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.svo
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.properties module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.properties
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.display module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.display
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.combine module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.combine
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.filters.sites module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.filters.sites
   :members:
   :show-inheritance:
   :undoc-members:

Spectral grids
--------------

.. automodule:: sed_tools.grids

sed\_tools.grids.pipeline module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.pipeline
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.njm module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.njm
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.svo module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.svo
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.msg module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.msg
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.mast module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.mast
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.clean module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.clean
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.units module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.units
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.hdf5\_wavelengths module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.hdf5_wavelengths
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.lookup module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.lookup
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.coverage module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.coverage
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.spectrum\_io module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.spectrum_io
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.grids.header\_parser module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.grids.header_parser
   :members:
   :show-inheritance:
   :undoc-members:

Flux cubes
----------

.. automodule:: sed_tools.cube

sed\_tools.cube.build module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cube.build
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.cube.variants module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cube.variants
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.cube.check module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cube.check
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.cube.fileio module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cube.fileio
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.cube.collision module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cube.collision
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.cube.tool module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cube.tool
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.cube.model module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cube.model
   :members:
   :show-inheritance:
   :undoc-members:
   :ignore-module-all:

sed\_tools.cube.combine module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cube.combine
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.cube.densify module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cube.densify
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.cube.mesa\_export module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.cube.mesa_export
   :members:
   :show-inheritance:
   :undoc-members:

Machine learning
----------------

sed\_tools.ml module
~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.ml
   :members:
   :show-inheritance:
   :undoc-members:
   :ignore-module-all:

sed\_tools.ml.completer module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.ml.completer
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.ml.generator module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.ml.generator
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.ml.common module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.ml.common
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.ml.tuning module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.ml.tuning
   :members:
   :show-inheritance:
   :undoc-members:

Shared helpers
--------------

sed\_tools.photometry module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.photometry
   :members:
   :show-inheritance:
   :undoc-members:

sed\_tools.njm\_mirror module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sed_tools.njm_mirror
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

Functions defined in the package itself. The names it re-exports are
documented above, in the modules that define them.

.. automodule:: sed_tools
   :members:
   :show-inheritance:
   :undoc-members:
   :ignore-module-all:
