# SED_Tools

Python package for downloading, processing, and standardizing stellar atmosphere models for MESA's `colors` module.

## What It Does

- Downloads stellar atmosphere spectra from SVO, MSG (Townsend), MAST (BOSZ), and NJM mirror
- Downloads photometric filter transmission curves from SVO Filter Profile Service
- Converts spectra to standardized units (wavelength in Angstroms, flux in erg/cm²/s/Å)
- Generates MESA-compatible binary flux cubes and HDF5 bundles
- Combines multiple stellar libraries into unified "omni grids"
- ML-powered extension of incomplete SEDs using neural networks

## Installation

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

## Quick Start

Interactive menu:

```bash
sed-tools
```

Direct commands:

```bash
sed-tools spectra          # Download stellar atmospheres
sed-tools filters          # Download photometric filters
sed-tools rebuild          # Build flux cubes and lookup tables
sed-tools combine          # Merge multiple grids
sed-tools ml_completer     # Train/use ML SED completer
```

## Workflow

### 1. Download Spectra

```bash
sed-tools spectra
```

Select source (SVO/MSG/MAST/NJM) and models. Downloads spectra, cleans units, generates `lookup_table.csv`.

### 2. Download Filters

```bash
sed-tools filters
```

Browse SVO Filter Profile Service by Facility → Instrument → Filters. Downloads transmission curves.

### 3. Build Data Cubes

```bash
sed-tools rebuild
```

Converts text spectra into:
- Binary `flux_cube.bin` files (required by MESA)
- HDF5 bundles
- Updated lookup tables

This step runs automatically after new spectra downloads.

### 4. Install into MESA

Copy or symlink the generated data:

```bash
# Copy
cp -r data/stellar_models/Kurucz2003all $MESA_DIR/colors/data/stellar_models/

# Or symlink
ln -s $(pwd)/data/stellar_models/Kurucz2003all $MESA_DIR/colors/data/stellar_models/
```

Configure MESA inlist:

```fortran
stellar_atm = '/colors/data/stellar_models/Kurucz2003all/'
instrument = '/colors/data/filters/Generic/Johnson'
```

## Advanced Features

### Combine Multiple Libraries

Create unified grids spanning multiple stellar atmosphere models:

```bash
sed-tools combine
```

Select models interactively or combine all with `--non-interactive`.

### ML SED Completer

Extend incomplete SEDs to broader wavelength ranges using black body baseline + neural network refinement:

```bash
sed-tools ml_completer
```

Supports training on combined grids with varying wavelength coverage. Uses masked training to handle heterogeneous data.

## Directory Structure

```
data/
├── stellar_models/
│   └── Kurucz2003all/
│       ├── flux_cube.bin       # Binary flux cube (MESA)
│       ├── lookup_table.csv    # Parameter lookup (MESA)
│       ├── *.txt               # Raw spectra
│       └── *.h5                # HDF5 bundle
└── filters/
    └── Generic/
        └── Johnson/
            ├── B.dat           # Filter transmission
            ├── V.dat
            └── Johnson         # Index file (MESA)
```

## Python API

```python
from sed_tools import SED

sed = SED()
matches = sed.find_model(5777, 4.44, 0)  # Teff, logg, [Fe/H]
model = sed.model(matches[0].name)
spec = model(5777, 4.44, 0.0)            # Interpolate spectrum
mags = spec.photometry("G", "Gbp", "Grp")  # GAIA filter names (no .dat extension)
```

Filter specifications use the filename stem only. For `data/filters/GAIA/GAIA/G.dat`, use `"G"`.

## Data Sources

- [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/)
- [MAST BOSZ Spectral Library](https://archive.stsci.edu/prepds/bosz/)
- [MSG Stellar Atmosphere Grids](http://user.astro.wisc.edu/~townsend/static.php?ref=msg-grids)
- [NJM Mirror](https://nillmill.ddns.net/sed_tools/)

## Requirements

- Python ≥3.9
- numpy, pandas, h5py, astropy, matplotlib
- TensorFlow (for ML completer)

See `pyproject.toml` for complete dependency list.