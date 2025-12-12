SED Tools for MESA
==================

Automated data preparation for the MESA ``colors`` module.

This package downloads stellar atmospheres (SEDs) and photometric filter profiles, then processes them into the specific directory structure and binary formats (``.bin``, ``.h5``) required by MESA.

Installation
------------

You can install the package directly from the source code:

.. code-block:: bash

    pip install .

Or, for development (editable mode):

.. code-block:: bash

    pip install -e .

Web Interface
-------------

All data used by this tool originate from three public services: the  
`SVO Filter Profile Service <http://svo2.cab.inta-csic.es/theory/fps/>`_,  
the `MAST BOSZ Spectral Library <https://archive.stsci.edu/prepds/bosz/>`_,  
and Townsend’s `MSG Stellar Atmosphere Grids <https://www.astro.wisc.edu/~townsend/msg/>`_.  

These datasets are mirrored on a dedicated server to ensure fast, consistent access for SED Tools and for any external workflows that rely on the processed spectra and filters.

A browsable mirror of the processed SED and filter data is available here:

`SED Tools Web Interface <https://nillmill.ddns.net/sed_tools/>`_


Quick Start
-----------

Once installed, you can run the interactive menu from anywhere in your terminal:

.. code-block:: bash

    sed-tools

Alternatively, you can run it as a Python module:

.. code-block:: bash

    python -m sed_tools

(For developers: a ``run.py`` wrapper script is also provided in the root directory if you prefer not to install the package).

Workflow
--------

To prepare data for MESA, follow these steps sequentially:

Download Stellar Atmospheres (Spectra)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select option **1** in the menu or run:

.. code-block:: bash

    sed-tools spectra

* Supports **SVO**, **MSG** (Townsend), and **MAST** (BOSZ) repositories.
* Downloads raw spectra and metadata.
* Automatically cleans data and creates the ``lookup_table.csv``.

Download Filters
~~~~~~~~~~~~~~~~~~~

Select option **2** in the menu or run:

.. code-block:: bash

    sed-tools filters

* Interactive browser for the **SVO Filter Profile Service**.
* Select Facility → Instrument → Filters.
* Downloads transmission curves in the correct format.

Build Data Cubes
~~~~~~~~~~~~~~~~~~~

Select option **3** in the menu or run:

.. code-block:: bash

    sed-tools rebuild

* This step **Will** be completed automatically for any *new* download of a Stellar Atmosphere table.
* **This step is required for MESA.**
* Converts raw text spectra into binary ``flux_cube.bin`` files and HDF5 bundles.
* MESA will be looking for these ``flux_cubes``.

Install into MESA
~~~~~~~~~~~~~~~~~~~~

Once Step 3 is complete, your ``data/`` folder will mirror the structure MESA expects.

**Option A: Copy**

Copy the generated folders into MESA.

.. code-block:: bash

    cp -r data/stellar_models/Kurucz2003all $MESA_DIR/colors/data/stellar_models/
    cp -r data/filters/Generic $MESA_DIR/colors/data/filters/

**Option B: Symlink**

Link the generated data directly to your MESA directory to keep this repo as the source of truth.

.. code-block:: bash

    ln -s /path/to/SED_Tools/data/stellar_models/MyModel $MESA_DIR/colors/data/stellar_models/MyModel
    ln -s /path/to/SED_Tools/data/filters/MyInstrument $MESA_DIR/colors/data/filters/MyInstrument

Configure MESA Inlist
~~~~~~~~~~~~~~~~~~~~~~~~

Point your MESA project's ``inlist`` to the new data.

**For Spectra:**

.. code-block:: fortran

    stellar_atm = '/colors/data/stellar_models/Kurucz2003all/'

**For Filters:**

.. code-block:: fortran

    ! format: 'Facility/Instrument'
    instrument = '/colors/data/filters/Generic/Johnson'

Directory Structure
-------------------

The tool generates data in this exact hierarchy:

.. code-block:: text

    data/
    ├── stellar_models/
    │   └── Kurucz2003all/
    │       ├── flux_cube.bin       <-- (Required by MESA)
    │       ├── lookup_table.csv    <-- (Required by MESA)
    │       ├── *.txt               <-- Raw spectra
    │       └── *.h5                <-- Python bundle
    └── filters/
        └── Generic/
            └── Johnson/
                ├── B.dat           <-- Transmission curve
                ├── V.dat
                └── Johnson         <-- Index file (Required by MESA)
