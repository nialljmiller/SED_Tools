# Python API

```python
from sed_tools.api import SED, Filters
```

The API does what the command line does, and adds the building blocks for
your own pipelines. Every name in `sed_tools.__all__` is documented by its
docstring (`help(SED.fetch)`); the
[Sphinx sources](https://github.com/nialljmiller/SED_Tools/tree/main/docs/source)
build a reference from them.

## `SED`: spectral grids

### Discovery

```python
from sed_tools.api import SED

grids = SED.query()                                       # every source; SpectralGridInfo objects
grids = SED.query(source="svo")
grids = SED.query(teff_min=5000, teff_max=7000, logg_min=3.5, metallicity_min=-1.0)
local = SED.query(include_remote=False)                   # installed grids and variants only
print([g.name for g in local])
```

`SED.query(include_remote=False)` lists a split grid's variants
(`Husfeld/Husfeld_yhe_0p10`), not its top folder.

### Downloading

```python
from sed_tools.api import SED

sed = SED.fetch("bbody")                                   # NJM first, then the other sources
sed = SED.fetch("Husfeld", source="njm", workers=8)
sed = SED.fetch("Kurucz2003all", teff_min=4000, teff_max=8000, logg_min=3.0, logg_max=5.0,
                metallicity_min=-1.0, metallicity_max=0.5)
```

`fetch(catalog, source=None, teff_min=None, ..., metallicity_max=None,
workers=5, clean=True, model_root=None, flux_unit=None)` downloads, converts
and builds the grid, prints the MESA lines and returns a `SED`. A grid taken
from SVO is downloaded whole and not built: run `sed-tools rebuild --grids
NAME` before `SED.local`. The rules for
partial downloads are in [Spectral grids](grids.md). For a grid split into
variants, load the variant or the all-axes cube with `SED.local` afterwards.
`flux_unit` gives the unit of spectra that do not declare one (the
`--flux-unit` choices).

### Loading and interpolating

```python
from sed_tools.api import SED

sed = SED.local("Kurucz2003all/Kurucz2003all_alpha_0p00_lh_1p25_vtur_2_km_s")
print(sed.parameter_ranges())                    # {'teff': (min, max), 'logg': ..., 'metallicity': ...}
spectrum = sed(teff=5777, logg=4.44, metallicity=0.0)
spectrum = sed(5777, 4.44, 0.0)                  # the same, positional
print(spectrum.wavelength)                       # Å
print(spectrum.flux)                             # erg/s/cm²/Å at the stellar surface

nd = SED.local("Kurucz2003all")                  # the all-axes cube of the split grid
spectrum = nd(5777, 4.44, 0.0, alpha=0.2)
```

`local(catalog, model_root=None, filter_root=None, fill_gaps=True)` needs the
grid's flux cube. With `fill_gaps=True` a value outside the grid is clamped to
its edge; with `False` it raises ValueError. Interpolation is cubic Hermite in
every axis.

The returned `EvaluatedSED` has `wavelength`, `flux`, `teff`, `logg`,
`metallicity` and `photometry(*filters, system="AB", vega_spectrum=None)`
([Photometry and zero points](photometry.md)). It has no `save` method; write
its arrays with numpy (see the examples below).

`sed.available_grids()` lists the installed grids, `sed.grid` is the loaded
grid's `SpectralGrid`, and `sed.model()` the underlying `SEDModel`.

### Coverage, import and combining

```python
from sed_tools.api import SED

report = SED.coverage("bbody", plot=False)      # keys: n_spectra, n_distinct_nodes, fill_fraction, axes
report = SED.coverage("Husfeld/Husfeld_yhe_0p10", out_path="coverage.png")
result = SED.import_grid("~/my_spectra", name="MyGridApi", dry_run=True)
combined = SED.combine(["bbody", "Husfeld"], output="bb_husfeld_api")
```

`import_grid(src, name=None, model_root=None, move=False, build_cube=True,
build_h5=True, dry_run=False, flux_unit=None)` and `combine(catalogs,
output="combined_models", model_root=None)` do what
`sed-tools import` and `sed-tools combine` do ([Spectral grids](grids.md)).
`SED.ml_completer()` and `SED.ml_generator()` return the ML tools
([Machine learning](ml.md)).

## `SpectralGrid` and `Spectrum`

`sed.grid` holds the grid's spectra, read from its folder:

```python
from sed_tools.api import SED

grid = SED.local("Husfeld/Husfeld_yhe_0p10").grid
print(len(grid), grid.teff_grid, grid.logg_grid)
print(grid.parameters.head())                    # DataFrame: file_name, teff, logg, metallicity, ...
hot = grid.filter(teff_min=50000)                # a new SpectralGrid
for spec in hot:
    print(spec.filename, spec.teff, spec.logg)
spec = grid[0]
print(spec.wavelength[:3], spec.flux[:3], spec.metallicity)
spec.save("one_spectrum.txt")                    # header lines, then wavelength and flux
```

`grid.write(path)` writes the spectra, a lookup table, an HDF5 bundle and a
flux cube to another folder. Each spectrum keeps the other columns of its
lookup row (alpha, yhe, ...), so a selection from a split grid is written
split, with its variant folders and all-axes cube, as the cube builder splits
one. An axis no spectrum has (log g of bbody) is left out of the table.

```python
from pathlib import Path
from sed_tools.api import SED

grid = SED.local("Husfeld").grid.filter(teff_max=40000)
grid.write(Path.home() / "husfeld_cool")
```

Without a path, `write` writes into the grid's own folder and replaces its
spectra and lookup table with the selection. A selection of one variant written
into a split grid's folder makes it one grid: its lookup table and
`flux_cube.bin`, with `variants_index.csv` removed; the variant folders,
`flux_cube_nd.bin` and the other spectra stay beside it. Give a path to keep
the split grid as it was.

`SpectralGridInfo` (the rows of `SED.query()`) describes a grid without
loading it:

```python
from sed_tools.api import SED

info = SED.query(include_remote=False)[0]
print(info.name, info.source, info.teff_range, info.logg_range, info.metallicity_range,
      info.n_spectra, info.is_local)
print(info.covers(teff=5777, logg=4.44), info.covers_range(teff_min=5000, teff_max=6000))
```

## `Filters`: filter sets

All methods are class methods and take `filter_root=` to use another folder
than `~/.sed_tools/data/filters`.

| Method | What it does |
|---|---|
| `query(facility=None, include_local=True, include_remote=True)` | Installed and remote filter sets |
| `fetch(facility, instrument, components=None, bands=None, atmosphere=None, detector_type=None, ...)` | Install a set, or compose a product; returns its folder |
| `available(facility, instrument, include_remote=True)` | What the set provides: components, recipes per band, missing files |
| `inventory(facility=None, include_remote=True)` | Every SVO, NJM, local and registered instrument, and the discovery errors |
| `atmospheres(include_remote=True)` | The atmosphere sites and their conditions |
| `compose(instrument, components, atmosphere=None, bands=None, detector_type=None, ...)` | Compose components of an installed set |
| `combine(output, *inputs, facility=None, instrument=None, on_conflict="rename")` | Combine sets into one MESA instrument folder |
| `tag_detector_types(path=None, force=False, dry_run=False)` | Tag `DetectorType` in place |
| `correct_detector_type(target, detector_type, bands=None)` | Relabel curves |
| `convert_detector_type(target, detector_type, bands=None)` | Relabel and convert curves |
| `properties(facility, instrument)`, `info(facility, instrument)` | NJM's `properties.json` and `info.json` for the set |
| `search(...)`, `cite(...)`, `citations(...)`, `plot(...)` | Search the catalogue, cite, plot |
| `fetch_vega()` | NJM's Vega spectrum; returns its path |

[Filters](filters.md) and [Photometry and zero points](photometry.md) show
them in use.

```python
from sed_tools.api import Filters

path = Filters.fetch("Generic", "Johnson")
bands = Filters.search(wavelength=5500, fwhm=(500, 1500))
props = Filters.properties("Generic", "Johnson")
info = Filters.info("Generic", "Johnson")
print(Filters.cite("Generic/Johnson"))
ax = Filters.plot("Generic/Johnson", normalize="peak", vega=True)
```

## MESA lines

```python
from pathlib import Path
from sed_tools import mesa_inlist_lines

print(mesa_inlist_lines(Path.home() / ".sed_tools/data/filters/Generic/Johnson"))
```

## Examples

### Spectra for a list of stars

```python
from pathlib import Path
import numpy as np
from sed_tools.api import SED

sed = SED.local("Kurucz2003all/Kurucz2003all_alpha_0p00_lh_1p25_vtur_2_km_s")
out = Path("output")
out.mkdir(exist_ok=True)
stars = [
    {"name": "Sun",         "teff": 5777, "logg": 4.44, "met":  0.0},
    {"name": "Procyon",     "teff": 6530, "logg": 4.00, "met":  0.0},
    {"name": "Alpha_Cen_B", "teff": 5260, "logg": 4.37, "met":  0.2},
]
for star in stars:
    spectrum = sed(star["teff"], star["logg"], star["met"])
    np.savetxt(out / f"{star['name']}.txt", np.column_stack([spectrum.wavelength, spectrum.flux]))
```

### A grid made of two selections

```python
from pathlib import Path
from sed_tools.api import SED

models = Path.home() / ".sed_tools/data/stellar_models"
hot = SED.local("Husfeld/Husfeld_yhe_0p10").grid.filter(teff_min=50000)
hot.write(models / "Husfeld_hot")
cool = SED.local("bbody").grid.filter(teff_min=20000, teff_max=50000)
cool.write(models / "bbody_warm")
combined = SED.combine(["Husfeld_hot", "bbody_warm"], output="bb_husfeld_hot")
```

## Command line and Python

| Command | Python |
|---|---|
| `sed-tools filters F/I` | `Filters.fetch("F", "I")` |
| `sed-tools filters-combine X A B` | `Filters.combine("X", "A", "B")` |
| `sed-tools filters-info F/I` | `Filters.available("F", "I")`, `Filters.info("F", "I")`, `Filters.properties("F", "I")` |
| `sed-tools filters-compose F/I --components ...` | `Filters.fetch("F", "I", components=...)` |
| `sed-tools filters-detector PATH` | `Filters.tag_detector_types(PATH)` |
| `sed-tools filters-detector PATH --correct T` / `--convert T` | `Filters.correct_detector_type(...)` / `Filters.convert_detector_type(...)` |
| `sed-tools filters-inventory` | `Filters.inventory()` |
| `sed-tools vega` | `Filters.fetch_vega()` |
| `sed-tools grids` (the list) | `SED.query()` |
| `sed-tools grids --grids X` | `SED.fetch("X")` |
| `sed-tools grids --grids X --flux-unit flam` | `SED.fetch("X", flux_unit="flam")` |
| `sed-tools rebuild --grids X` | `sed_tools.precompute_flux_cube(folder, folder + "/flux_cube.bin")` builds the cubes; `grid.write(path)` writes a copy with its own |
| `sed-tools combine --grids A B --output C` | `SED.combine(["A", "B"], output="C")` |
| `sed-tools densify --flux-cube F --output O` | `sed_tools.cube.densify.densify_grid(F, O)` |
| `sed-tools import --path P --name N` | `SED.import_grid("P", name="N")` |
| `sed-tools coverage --grids X` | `SED.coverage("X")` |
| `sed-tools mesa-path PATH` | `sed_tools.mesa_inlist_lines(PATH)` |
| `sed-tools ml_completer` (menu: train, extend) | `SED.ml_completer().train(...)`, `.extend(...)` |
| `sed-tools ml_generator` (menu: train, generate) | `SED.ml_generator().train(...)`, `.generate(...)` |
| `sed-tools ml_generator --auto --library L --output O` | `sed_tools.ml.tuning.AutoTrainer().train_best(L, O, n_trials=20, epochs=100)` |
