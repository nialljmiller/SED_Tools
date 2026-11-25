# MESA

SED_Tools prepares the data that MESA's `colors` module reads: a spectral
grid's flux cube and lookup table, a filter set, and the Vega spectrum. MESA
reads them from the folders SED_Tools writes; nothing needs to be copied.

## The inlist lines

Every download, `rebuild`, `combine`, `filters-combine` and `mesa_prepare`
prints the lines for the `&colors` namelist, for example:

```text
The MESA colors inlist path for this is:
    stellar_atm = '/home/you/.sed_tools/data/stellar_models/Kurucz2003all/Kurucz2003all_alpha_0p00_lh_1p25_vtur_2_km_s'
    stellar_atm = '/home/you/.sed_tools/data/stellar_models/Kurucz2003all/Kurucz2003all_alpha_0p40_lh_1p25_vtur_2_km_s'
  (one line per variant: MESA reads one, so use one of these 2; variants_index.csv lists them)
```

Each line is absolute, single-quoted and has no trailing slash (an apostrophe
in a path is doubled, as a Fortran namelist string needs), and can be pasted
into the inlist as it is. MESA reads a path that starts with `./` or `../`
relative to the directory it runs in, and any other path that does not start
with `/` relative to `$MESA_DIR`, which is why the lines are absolute. A complete
`&colors` section looks like this:

```fortran
&colors
    use_colors = .true.

    ! Filter set: the instrument folder, which holds the index file named after it
    instrument = '/home/you/.sed_tools/data/filters/Generic/Johnson'

    ! Spectral grid: a folder with flux_cube.bin and lookup_table.csv;
    ! for a grid split into variants, one variant's folder
    stellar_atm = '/home/you/.sed_tools/data/stellar_models/Kurucz2003all/Kurucz2003all_alpha_0p00_lh_1p25_vtur_2_km_s'

    ! Vega spectrum, for mag_system = 'Vega'
    vega_sed = '/home/you/.sed_tools/data/vega/vega_flam.csv'
/
```

MESA's own defaults are in `$MESA_DIR/colors/defaults/colors.defaults`
(`distance` is 10 pc, so magnitudes are absolute; `mag_system` is `'Vega'`).
MESA declares `stellar_atm` with 256 characters; the default data directory
gives paths of about 100.

## Printing the lines again

```bash
sed-tools mesa-path ~/.sed_tools/data/stellar_models/Kurucz2003all
sed-tools mesa-path ~/.sed_tools/data/stellar_models/bbody
sed-tools mesa-path ~/.sed_tools/data/filters/GAIA/GAIA
sed-tools mesa-path ~/.sed_tools/data/filters/LSST/LSST/products/filter
```

```python
from pathlib import Path
from sed_tools import mesa_inlist_lines

print(mesa_inlist_lines(Path.home() / ".sed_tools/data/stellar_models/Husfeld"))
```

`mesa-path` works for a grid, a split grid (one line per variant), a variant
folder, a filter set and a composed product. Relative and `~` paths are
printed absolute. A folder that is neither a grid (no `lookup_table.csv`) nor
a filter set (no index file named after it) is refused with exit 1.

## The Vega spectrum

```bash
sed-tools vega
```

```python
from sed_tools.api import Filters

print(Filters.fetch_vega())
```

Both download NJM's `vega_flam.csv`, the same file as MESA's own
`data/colors_data/stellar_models/vega_flam.csv`, to
`~/.sed_tools/data/vega/` (or `--out DIR`), and print the `vega_sed` line.

## Grids split into variants

MESA reads a 3-axis cube: Teff × log g × [M/H]. A grid with further
parameters (Kurucz2003all: [alpha/M] 0 and 0.4; Husfeld: seven helium
abundances) is split into one folder per variant, and MESA reads one variant
folder ([Spectral grids](grids.md) shows the layout). Point `stellar_atm` at a
variant folder, never at the grid's top folder.

The top folder holds `flux_cube_nd.bin`, the all-axes cube that SED_Tools
reads and MESA cannot, and no `flux_cube.bin`. A grid split or downloaded with
0.6.0 or earlier, or with a build from 19–24 Nov 2026, can still have an old
`flux_cube.bin` at its top, which MESA would read if pointed there; SED_Tools
does not move it. Rename or remove that file and run `sed-tools rebuild
--grids NAME` ([Spectral grids](grids.md)). Pointed at a top folder without
it, MESA prints
`colors: no flux_cube.bin found; using slower per-file SED loading` and loads
spectra one by one from the top-level `lookup_table.csv`, which lists every
variant; the result is not that of any one variant.

## Keeping a copy for MESA

MESA reads the printed paths wherever they are. To keep one variant as a
self-contained folder, for example inside MESA's tree or on another machine,
export it with `mesa_prepare`:

```bash
sed-tools mesa_prepare --grid Kurucz2003all \
    --variant Kurucz2003all_alpha_0p40_lh_1p25_vtur_2_km_s \
    --output ~/mesa_grids/Kurucz2003all_alpha04
```

It writes that variant's `flux_cube.bin`, its `lookup_table.csv` and a copy of
each of its spectra (real files, not links), and prints the `stellar_atm` line
for the new folder; it stops before writing anything if a spectrum or the cube
is missing. An existing output folder is written into without asking: the
files it writes replace theirs, and the folder's other files stay. Without
options it asks for the grid, the variant and the output folder. A grid that is not split is already
MESA-ready; `mesa_prepare` says so and prints the grid's own line.

Do not copy or link a whole split grid into
`$MESA_DIR/data/colors_data/stellar_models/`. MESA ships its own
`Kurucz2003all` there (alpha 0, the folder its default `stellar_atm` names);
copying SED_Tools' `Kurucz2003all` over it mixes the two grids' files and
replaces MESA's `lookup_table.csv` with one that lists both variants. A
variant folder copied with `cp -r` contains links to spectra in its parent
folder, which then point nowhere; `mesa_prepare` writes real files.

## Filter sets

`instrument` is an instrument folder: MESA opens the index file named after
the folder and loads the curves it lists. A composed product
(`products/<recipe>/`) and a combined set (`filters/Combined/<name>/`) are
instrument folders too. MESA names each history column after the curve's
file name without `.dat` (`G.dat` gives the column `G`), so two curves with
the same file name cannot be used together; `filters-combine` renames such
duplicates ([Filters](filters.md)).

MESA's photon-counting branch reads each curve's `DetectorType` from its
header row; released MESA skips that row.

## When MESA cannot find the cube

The folder `stellar_atm` names must hold both files:

```text
<stellar_atm>/
  flux_cube.bin       must exist
  lookup_table.csv    must exist
```

If it has `variants_index.csv`, it is a split grid's top folder: use one of
its variant folders. A `flux_cube.bin` in such a top folder was left by an
earlier version and is not the grid's cube; rename it and rebuild (see
"Grids split into variants" above). `sed-tools mesa-path <grid folder>` prints the right
lines.

## Reading MESA's own data from Python

SED_Tools can read MESA's shipped data directly:

```python
import os
from pathlib import Path
from sed_tools.api import SED

colors_data = Path(os.path.expandvars("$MESA_DIR")) / "data/colors_data"
sed = SED.local("Kurucz2003all", model_root=colors_data / "stellar_models",
                filter_root=colors_data / "filters")
mags = sed(teff=6000, logg=2.0, metallicity=-1.0).photometry("GAIA", system="AB")
print(mags["Gbp"].magnitude - mags["Grp"].magnitude)
```
