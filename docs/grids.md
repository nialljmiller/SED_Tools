# Spectral grids

A **stellar spectral grid** is a collection of **model spectra** indexed by
parameters such as effective temperature (Teff), surface gravity (log g) and
metallicity ([M/H]). SED_Tools uses *spectral grid* in its menus and commands;
*spectral library* is the broader term that also covers observed templates,
planetary spectra and stellar population spectra. Earth's atmosphere
transmission models, used for ground-based filter curves, are a separate thing
(see [Filters](filters.md)).

The command `sed-tools grids` and the option `--grids` were called `spectra`
and `--models` before 0.5.0; the old names still work. In Python the
preferred names are `SpectralGrid`, `SpectralGridInfo`, `sed.grid`,
`sed.available_grids()` and `find_grids`; `Catalog`, `CatalogInfo` and
`sed.cat` remain as aliases. Keyword arguments such as `catalog` and
`model_root`, and the `stellar_models/` folder, keep their names.

## Sources

| Source | `--source` | What it offers |
|---|---|---|
| [NJM mirror](https://data.nialljmiller.com/sed_tools/) | `njm` | Grids already standardised and built, cut on the server; tried first |
| [SVO Theory Server](https://svo2.cab.inta-csic.es/theory/newov2/) | `svo` | Many model grids; downloaded whole (no cuts) |
| [MSG](https://madstar.richtownsend.dev/static.php?ref=msg-grids) | `msg` | Rich Townsend's MSG grids; one HDF5 file per grid, from which the selection is extracted |
| [MAST BOSZ](https://archive.stsci.edu/prepds/bosz/) | `mast` | The BOSZ library, including the BOSZ 2024 fixed-resolution and original-resolution grids |

`--source all` uses every source. Without `--source`, NJM is tried first
and then the others; when NJM does not answer, its grids are not offered and
the others are (see [NJM mirror](njm.md)).

The BOSZ 2024 fixed-resolution products (`BOSZ-2024-r500` to
`BOSZ-2024-r50000`) are resampled spectra. SED_Tools reads their wavelengths
from the official STScI files in the archive's `bosz2024/wavelength_grids/`
folder, and never makes up a wavelength axis from row counts. If those files
cannot be reached, the download fails; `BOSZ-2024-rorig` carries its
wavelength column in each file and does not need them.

## Downloading

```bash
sed-tools grids --grids bbody                                       # a whole grid
sed-tools grids --grids Husfeld --workers 8                         # parallel downloads
sed-tools grids --source njm --grids Kurucz2003all --teff 4000 8000 --logg 3 5   # part of a grid
```

Run `sed-tools grids` with no options in a terminal to pick sources and
grids from a list; for an NJM grid it then asks for cuts. Without a terminal it
needs `--grids`, and takes the whole grid unless a cut option is given.

The download:

1. downloads the spectrum files (only those within `--teff`, `--logg`,
   `--meta` and `--wavelength`, when given);
2. converts every spectrum to wavelength in Å and flux in erg/s/cm²/Å. Spectra
   that declare their flux unit keep it. For spectra that do not, SED_Tools
   asks in a terminal, and otherwise stops unless `--flux-unit` gives it
   (`flam`, `fnu`, `fnu_jy`, `flam_si`, `flam_si_nm`, `flam_si_um` or
   `normalized`);
3. writes `lookup_table.csv`, an HDF5 bundle (`<grid>.h5`) and the flux cube
   (see Rebuilding below);
4. prints the MESA `stellar_atm` line or lines.

### Partial downloads

The same cuts are available from Python:

```python
from sed_tools.api import SED

SED.fetch("Kurucz2003all", teff_min=4000, teff_max=8000, logg_min=3.0, logg_max=5.0)
SED.fetch("bbody", source="njm", teff_min=3000, teff_max=12000)
```

With any bound, only the spectra within the bounds are downloaded. A bound of
0 is a bound, and a single bound leaves the other side open. The grid's folder
(`~/.sed_tools/data/stellar_models/<grid>`) then holds those spectra, their
lookup table, HDF5 bundle and flux cubes, so `SED.local` and MESA use that
selection.

- From NJM, the cut is made on the server: NJM's cube and bundle of the whole
  grid are not downloaded. MSG downloads its grid's single HDF5 file whole and
  extracts the spectra within the bounds. SVO cannot select spectra:
  `sed-tools grids` refuses a cut for an SVO grid, and `SED.fetch` downloads
  the whole grid, with a warning, and cuts only the `sed.grid` it returns.
  From SVO, with or without bounds, `SED.fetch` cuts nothing on disk and
  builds no flux cube; build it with `sed-tools rebuild --grids NAME`.
- A folder that already holds every spectrum within the bounds, with a flux
  cube that holds every spectrum there, is used as it stands: nothing is
  downloaded or rewritten. This is read from the cube and the lookup table, not
  from file times, so a grid copied without its times (`cp -R`, `scp`) counts
  as complete.
- A later fetch without bounds, or with wider ones, downloads the rest and
  rebuilds the lookup table, bundle and cube from every spectrum in the folder.
- Separate selections fetched into one folder (say Teff ≤ 5800 with
  log g ≤ 4.2, then Teff ≥ 6200 with log g ≥ 4.3) leave points of the cube's
  Teff × log g × [M/H] grid without a spectrum. Their flux is zero, and pulls
  interpolation near them toward zero.
- A whole grid from NJM comes with NJM's flux cube, or, for a grid split into
  variants, with each variant's folder and cube. For a grid that `SED.fetch`
  took from SVO, or a whole grid from MSG, build the cube with
  `sed-tools rebuild --grids NAME`.

## Local grids

```python
from sed_tools.api import SED

print(SED.query(include_remote=False))          # installed grids and variants
sed = SED.local("bbody")
print(sed.parameter_ranges())
spectrum = sed(teff=5777, logg=0.0, metallicity=0.0)
```

A grid that is not split loads by its folder name. A grid without a log g or
[M/H] axis (bbody has neither; Husfeld has no [M/H]) takes any value for the
missing axis.

## Split grids and the all-axes cube

Some grids have more parameters than Teff, log g and [M/H]: Kurucz2003all
has two [alpha/M] values, Husfeld seven helium abundances (`yhe`), and MSG
grids may have [alpha/Fe]. A lookup column is such an *extra axis* when it
tells apart spectra at one (Teff, log g, [M/H]) node. The columns SED_Tools
writes to describe a file (units, source, the provenance of combined and
ML-extended grids) never are, and a column that varies only between nodes
labels the spectra and splits nothing.

SED_Tools splits a grid with extra axes into one folder per combination of
their values, a **variant**:

```text
~/.sed_tools/data/stellar_models/Kurucz2003all/
  flux_cube_nd.bin          the all-axes cube: every variant, alpha as a fourth axis
  lookup_table.csv          every spectrum, with its alpha
  lookup_table_full.csv     the full grid's table as it was before the first split
  variants_index.csv        one row per variant: its values, number of spectra, folder
  Kurucz2003all.h5
  Kurucz2003all_fid*.txt    the spectra
  Kurucz2003all_alpha_0p00_lh_1p25_vtur_2_km_s/
    flux_cube.bin           this variant's 3-axis cube: what MESA reads
    lookup_table.csv
    *.txt                   links to the grid's spectra
  Kurucz2003all_alpha_0p40_lh_1p25_vtur_2_km_s/
    ...
```

The top folder holds no `flux_cube.bin` (unless an earlier version left one;
see below). Each variant folder is a complete
grid that MESA and `SED.local` read:

```python
from sed_tools.api import SED

sed = SED.local("Husfeld/Husfeld_yhe_0p10")
spectrum = sed(teff=40000, logg=4.0, metallicity=0.0)
```

The top folder loads the all-axes cube, `flux_cube_nd.bin`. It takes each extra
axis by keyword and interpolates it like Teff, log g and [M/H] (cubic Hermite);
a text-valued axis is matched exactly. A missing or unknown keyword raises
ValueError naming the axes. The file stores each node's spectrum in one piece,
so an evaluation reads only the spectra of the nodes around the point, a small
part of the file even for the largest grids.

```python
from sed_tools.api import SED

spectrum = SED.local("Husfeld")(40000, 4.0, 0.0, yhe=0.3)
spectrum = SED.local("Kurucz2003all")(5777, 4.44, 0.0, alpha=0.2)
```

MESA cannot read `flux_cube_nd.bin`; give it a variant folder
([MESA](mesa.md)).

### Split grids from earlier versions

A grid split or downloaded with SED_Tools 0.6.0 or earlier, or with a build
from 19–24 Nov 2026, can have a `flux_cube.bin` at its top: a copy of one
variant's cube, or the all-axes cube under its old name. SED_Tools does not
move or delete that file. While it is there, `SED.local` on the top folder
refuses the grid, names its variants and says to remove or rename the file;
`sed-tools rebuild` and a new download leave it in place; and MESA, if
`stellar_atm` names the top folder, reads it.

To upgrade such a grid, rename (or remove) the old file, then rebuild:

```bash
mv ~/.sed_tools/data/stellar_models/Husfeld/flux_cube.bin ~/Husfeld_old_top_flux_cube.bin
sed-tools rebuild --grids Husfeld
```

The rebuild writes `flux_cube_nd.bin`. A split grid that has neither file (split
before the all-axes cube existed) needs only the rebuild, or a new download of
the grid.

A `flux_cube_nd.bin` written by an earlier build (header version 1 or 2) is
still read and gives the same values, but it stores the spectra one wavelength
at a time, so an evaluation reads much of the file: up to 190 GB for the largest
MSG grids. The next `sed-tools rebuild --grids NAME`, download or `SED.fetch` of
the grid rewrites it in the current layout (version 3), with the same values to
the bit; the new file is written in full beside the old one and then swapped
in. This applies to `flux_cube_nd.bin` only: an old top `flux_cube.bin` stays
where it is, as described above.

## Rebuilding

`sed-tools rebuild` builds the lookup table, the HDF5 bundle and the flux
cubes from the `.txt` spectra in a grid's folder:

```bash
sed-tools rebuild --grids bbody
sed-tools rebuild --grids Husfeld --no-h5
```

| File | What it is |
|---|---|
| `lookup_table.csv` | one row per spectrum: file name and parameters |
| `<grid>.h5` | HDF5 bundle with every spectrum (`--no-h5` skips it) |
| `flux_cube.bin` | binary flux cube on the Teff × log g × [M/H] grid, read by MESA, [SED_Model](https://github.com/nialljmiller/SED_Model) and `SED.local`; for a split grid, one in each variant folder |
| `flux_cube_nd.bin` | the all-axes cube at the top of a split grid |
| `variants_index.csv` | the variants of a split grid |

`--no-cube` skips the cubes. Without `--grids`, `sed-tools rebuild` lists the
local grids and asks which to rebuild (`all`, or numbers and ranges such as
`1,3-6`).

## Combining grids

```bash
sed-tools combine --grids bbody Husfeld --output hot_and_bb
```

```python
from sed_tools.api import SED

combined = SED.combine(["bbody", "Husfeld"], output="hot_and_bb_py")
```

The combined cube's wavelengths are the union of the grids' own, so it keeps
each grid's whole coverage. Each spectrum keeps its own flux; between its
samples it follows the rule photometry uses (log F linear in λ between
positive samples, F linear next to a zero or negative one; see
[Photometry](photometry.md)). Every cube stores a spectrum's zero and negative
samples as they are. A node no spectrum fills stays at zero. A grid that lacks an axis the others have
(bbody has no log g) is used at every value of that axis. A combination whose
extra axes vary is split into variants, and the command names the variant
folders to give MESA. Combining again into the same output folder replaces
the earlier combination's spectrum copies and its `variants_index.csv`; the
earlier variant folders and `flux_cube_nd.bin` stay beside the new grid,
which is read from the new `variants_index.csv` (or the new `flux_cube.bin`).
The output may not be one of the grids combined.

`sed-tools combine` with no options asks which grids to combine;
`--non-interactive` without `--grids` combines every local grid except earlier
combined ones. Uses: extending Teff coverage with a hot and a cool grid,
filling gaps in parameter space with complementary grids.

## Densifying a flux cube

A cube with coarse Teff steps gives step-like colours in MESA. `sed-tools
densify` adds Teff planes to a 3-axis `flux_cube.bin` and writes a new cube
with the source's `lookup_table.csv` beside it:

```bash
sed-tools densify \
  --flux-cube ~/.sed_tools/data/stellar_models/Husfeld/Husfeld_yhe_0p10/flux_cube.bin \
  --output ~/.sed_tools/data/stellar_models/Husfeld_yhe_0p10_dense/flux_cube.bin \
  --teff-spacing 2500
```

New Teff nodes are filled with a Planck blackbody scaled to the nearest real
spectrum's bolometric flux; a Teff beside a node that holds no spectrum stays
empty there. With `--ml-model PATH`, a trained ML generator ([ML](ml.md))
fills the points inside its training range instead. The source cube is read
through a memory map and the new one built a block of wavelengths at a time,
sized to half the available RAM, so cubes larger than memory work. The
all-axes cube cannot be densified: densify a variant's cube.

## Importing your own spectra

`sed-tools import` copies (or with `--move`, moves) a folder of `.txt`
spectra into the grid folder and runs the same cleaning, lookup table, HDF5
bundle and flux cube steps as a download. Each file needs a header that gives
its parameters, for example:

```text
# teff = 5750.0
# logg = 4.5
# metallicity = 0.0
# wavelength_unit = Angstrom
# flux_unit = erg/cm2/s/A
3000.0  1.234e+06
3001.0  1.236e+06
...
```

To try it, copy a few spectra of an installed grid:

```bash
mkdir -p ~/my_spectra
cp ~/.sed_tools/data/stellar_models/Husfeld/Husfeld_yhe_0p10/*.txt ~/my_spectra/
sed-tools import --path ~/my_spectra --name MyGrid --dry-run     # report what it can parse
sed-tools import --path ~/my_spectra --name MyGrid --no-h5
```

```python
from sed_tools.api import SED

result = SED.import_grid("~/my_spectra", name="MyGridPy", dry_run=True)
```

`--flux-unit` gives the flux unit of spectra that do not declare one;
`--dry-run` says whether they do.

## Coverage

```bash
sed-tools coverage --grids Husfeld/Husfeld_yhe_0p10 --no-plot
sed-tools coverage --grids bbody --out bbody_coverage.png
sed-tools coverage --grids Husfeld/Husfeld_yhe_0p10 --no-plot --filters ~/.sed_tools/data/filters/Generic/Johnson/V.dat
```

`coverage` prints each axis's range and number of values, the number of
spectra and of distinct nodes, and the fill fraction of the full product of
the axes; without `--no-plot` it writes a Teff–log g and 3-D scatter plot.
`--filters` checks that the grid's wavelengths sample each band, and exits 1 if
one is not sampled. In Python:

```python
from sed_tools.api import SED

report = SED.coverage("bbody", plot=False)
print(report["n_spectra"], report["n_distinct_nodes"], report["fill_fraction"])
```
