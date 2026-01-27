# SED_Tools Python API

A programmatic Python interface that provides full parity with the CLI, plus additional capabilities for building data pipelines.

## Quick Start

```python
from sed_tools.api import SED, Filters

# Query available catalogs
catalogs = SED.query()
catalogs = SED.query(teff_min=5000, logg_min=4.0)  # Filter by coverage

# Fetch and download data
sed = SED.fetch('Kurucz2003all', teff_min=4000, teff_max=8000)
sed.cat.write()  # Save to default path

# Work with local data
sed = SED.local('Kurucz2003all')
spectrum = sed(5777, 4.44, 0.0)  # Interpolate
```

## API Reference

### `SED` Class

The main entry point for all functionality.

#### Class Methods (Discovery & Fetching)

##### `SED.query(**filters) -> List[CatalogInfo]`
Query available stellar atmosphere catalogs.

```python
# All available catalogs
catalogs = SED.query()

# Filter by source
catalogs = SED.query(source='svo')  # 'svo', 'msg', 'mast', 'njm', 'local'

# Filter by parameter coverage
catalogs = SED.query(
    teff_min=5000,
    teff_max=7000,
    logg_min=3.5,
    metallicity_min=-1.0,
)

# Local only
catalogs = SED.query(include_remote=False)
```

##### `SED.fetch(catalog, **params) -> SED`
Download a stellar atmosphere catalog.

```python
# Basic fetch (tries NJM mirror first)
sed = SED.fetch('Kurucz2003all')

# Force specific source
sed = SED.fetch('Kurucz2003all', source='svo')

# Fetch with parameter filtering
sed = SED.fetch(
    'Kurucz2003all',
    teff_min=4000,
    teff_max=8000,
    logg_min=3.0,
    logg_max=5.0,
    metallicity_min=-1.0,
    metallicity_max=0.5,
    workers=8,  # Parallel downloads
)

# The catalog is in sed.cat
print(f"Downloaded {len(sed.cat)} spectra")
```

##### `SED.local(catalog) -> SED`
Load an already-installed local catalog.

```python
sed = SED.local('Kurucz2003all')

# Ready for interpolation
spectrum = sed(5777, 4.44, 0.0)
```

##### `SED.combine(catalogs, output) -> SED`
Create ensemble grids from multiple catalogs.

```python
ensemble = SED.combine(
    catalogs=['Kurucz2003all', 'PHOENIX'],
    output='my_combined_grid',
)
```

##### `SED.ml_completer() -> MLCompleter`
Get the ML-based SED completion tool.

```python
completer = SED.ml_completer()
completer.train(grid='combined_grid')
completer.extend('sparse_model')
```

#### Instance Methods

##### `sed(teff, logg, metallicity) -> EvaluatedSED`
Interpolate a spectrum.

```python
sed = SED.local('Kurucz2003all')
spectrum = sed(5777, 4.44, 0.0)

print(spectrum.wavelength)  # Array of wavelengths
print(spectrum.flux)        # Array of flux values
```

##### `sed.parameter_ranges() -> dict`
Get the parameter space covered by the model.

```python
ranges = sed.parameter_ranges()
# {'teff': (3500.0, 50000.0), 'logg': (0.0, 5.0), 'metallicity': (-5.0, 1.0)}
```

### `Catalog` Class

Container for downloaded/loaded spectra.

```python
# Access via sed.cat after fetch or local
catalog = sed.cat

# Properties
print(len(catalog))           # Number of spectra
print(catalog.teff_grid)      # Unique Teff values
print(catalog.logg_grid)      # Unique logg values
print(catalog.parameters)     # DataFrame of all parameters

# Iterate over spectra
for spec in catalog:
    print(spec.teff, spec.logg, spec.metallicity)

# Filter catalog
cool = catalog.filter(teff_max=5000)

# Save to disk
catalog.write()                    # Default path
catalog.write('/custom/path')      # Custom path
```

### `Spectrum` Class

Individual spectrum data.

```python
spec = catalog[0]

print(spec.wavelength)    # np.ndarray (Angstroms)
print(spec.flux)          # np.ndarray (erg/cm²/s/Å)
print(spec.teff)          # Effective temperature
print(spec.logg)          # Surface gravity
print(spec.metallicity)   # [M/H]
print(spec.filename)      # Original filename

# Aliases
print(spec.wl)  # Same as wavelength
print(spec.fl)  # Same as flux

# Save individual spectrum
spec.save('/path/to/spectrum.txt')
```

### `Filters` Class

Work with photometric filter profiles.

```python
# Query available filters
filters = Filters.query()
filters = Filters.query(facility='HST')

# Download filters
path = Filters.fetch('Generic', 'Johnson')
```

### `CatalogInfo` Class

Information about available catalogs.

```python
info = SED.query()[0]

print(info.name)              # 'Kurucz2003all'
print(info.source)            # 'svo', 'local', etc.
print(info.teff_range)        # (3500.0, 50000.0)
print(info.logg_range)        # (0.0, 5.0)
print(info.metallicity_range) # (-5.0, 1.0)
print(info.n_spectra)         # Number of spectra
print(info.is_local)          # True if installed

# Check coverage
info.covers(teff=5777, logg=4.44)  # Point coverage
info.covers_range(teff_min=5000, teff_max=6000)  # Range coverage
```

## Comparison with CLI

| CLI Command | Python API |
|-------------|------------|
| `sed-tools spectra` (list models) | `SED.query()` |
| `sed-tools spectra --models X` | `SED.fetch('X')` |
| `sed-tools rebuild --models X` | `sed.cat.write()` |
| `sed-tools combine` | `SED.combine([...])` |
| `sed-tools ml_completer` | `SED.ml_completer()` |
| `sed-tools filters` | `Filters.fetch(...)` |

## Example Pipelines

### Batch Processing Stars

```python
from sed_tools.api import SED
import numpy as np

# Load model
sed = SED.local('Kurucz2003all')

# Process a catalog of stars
stars = [
    {'teff': 5777, 'logg': 4.44, 'met': 0.0},   # Sun
    {'teff': 9940, 'logg': 4.30, 'met': 0.0},   # Vega
    {'teff': 3850, 'logg': 4.79, 'met': -0.1},  # Proxima Cen
]

for star in stars:
    spec = sed(star['teff'], star['logg'], star['met'])
    # Do analysis...
```

### Building Custom Grids

```python
from sed_tools.api import SED

# Fetch multiple catalogs with specific ranges
kurucz = SED.fetch('Kurucz2003all', teff_min=3500, teff_max=10000)
phoenix = SED.fetch('PHOENIX', teff_min=2500, teff_max=4000)

# Save both
kurucz.cat.write()
phoenix.cat.write()

# Combine into unified grid
combined = SED.combine(
    ['Kurucz2003all', 'PHOENIX'],
    output='cool_to_warm_stars'
)
```

### Extending Incomplete SEDs

```python
from sed_tools.api import SED

# Train ML model on complete grid
completer = SED.ml_completer()
completer.train('combined_grid', epochs=200)

# Extend sparse model
extended = completer.extend(
    'sparse_uv_model',
    wavelength_range=(100, 100000),
)

# Save extended grid
extended.write()
```

## Installation

Add to your `sed_tools/__init__.py`:

```python
from .api import (
    SED, Catalog, CatalogInfo, Spectrum, Filters, MLCompleter,
    query, fetch, local,
)
```

Then:

```python
from sed_tools import SED, query, fetch
```
