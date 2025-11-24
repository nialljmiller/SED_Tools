SED Tools for MESA
==================

Automated data preparation for the MESA ``colors`` module.

This package downloads stellar atmospheres (SEDs) and photometric filter profiles, then processes them into the specific directory structure and binary formats (``.bin``, ``.h5``) required by MESA.

Quick Start
-----------

This tool is self-bootstrapping. It checks for dependencies (numpy, pandas, h5py, etc.) and installs them if missing.

.. code-block:: bash

    python run.py

Workflow
--------

To prepare data for MESA, follow these steps sequentially:

1. Download Stellar Atmospheres (Spectra)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select option **1** in the menu or run ``python run.py spectra``.

* Supports **SVO**, **MSG** (Townsend), and **MAST** (BOSZ) repositories. (TODO: create unified hosting source)
* Downloads raw spectra and metadata.
* Automatically cleans data and creates the ``lookup_table.csv``.

2. Download Filters
~~~~~~~~~~~~~~~~~~~

Select option **2** in the menu or run ``python run.py filters``.

* Interactive browser for the **SVO Filter Profile Service**.
* Select Facility → Instrument → Filters.
* Downloads transmission curves in the correct format.

3. Build Data Cubes (Crucial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select option **3** in the menu or run ``python run.py rebuild``.

* **This step is required for MESA.**
* Converts raw text spectra into binary ``flux_cube.bin`` files and HDF5 bundles.
* MESA cannot read the raw text files; it requires these binaries.

4. Install into MESA
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

5. Configure MESA Inlist
~~~~~~~~~~~~~~~~~~~~~~~~

Point your MESA project's ``inlist`` to the new data. (NB can also just point to the SED Tools repo from here)

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