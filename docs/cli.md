# Command line

`sed-tools` is installed with the package. `sed-tools --help` lists the
commands, and `sed-tools <command> --help` gives each one's options.

```bash
sed-tools --help
sed-tools grids --help
```

## The interactive menu

`sed-tools` with no command opens a menu:

```text
-- Filters ---------------------------------------
  1) Download Filters (NJM / SVO)
  2) Combine filter sets
  3) Filter set information
  4) Build a filter curve from its parts (filter, optics, camera, atmosphere)

-- Spectral grids --------------------------------
  5) Download spectral grids (NJM / SVO / MSG / MAST)
  6) Rebuild (lookup + HDF5 + flux cube)
  7) Combine grids into omni grid

-- ML --------------------------------------------
  8) ML SED Completer (train/extend incomplete SEDs)
  9) ML SED Generator (generate SEDs from parameters)
 10) Grid Densifier (fill coarse Teff gaps)

-- Tools -----------------------------------------
 11) Coverage (parameter-space summary + plot)
 12) Import a local spectral grid
 13) Prepare a spectral grid for MESA
 14) Config (data directory)

--------------------------------------------------
  0/q) Quit
```

Option 3 shows what `sed-tools filters-info` shows, for an instrument picked
from the lists; option 4 asks the composition questions that follow a
download. In 0.4.1 the filter options were 1 and 2 only, so 0.4.1's options
3–12 are now 5–14. The menu shows itself again after a workflow ends or fails;
Ctrl-C during a workflow returns to it.

## Commands

| Command | What it does | Guide |
|---|---|---|
| `grids` (alias `spectra`) | Download spectral grids, whole or cut by `--teff`, `--logg`, `--meta`, `--wavelength`, and build them | [Spectral grids](grids.md) |
| `rebuild` | Build the lookup table, HDF5 bundle and flux cubes from a grid's spectra | [Spectral grids](grids.md) |
| `combine` | Combine local grids into one | [Spectral grids](grids.md) |
| `densify` | Add Teff planes to a 3-axis flux cube | [Spectral grids](grids.md) |
| `import` | Bring a folder of your own `.txt` spectra into the data directory and build it | [Spectral grids](grids.md) |
| `coverage` | Report a grid's parameter-space coverage, optionally against filter bands | [Spectral grids](grids.md) |
| `filters` | Download a filter set, or browse the pickers; with `--components`, compose | [Filters](filters.md) |
| `filters-info` | Show what a filter set contains, its components, atmosphere sites, licence and bands | [Filters](filters.md) |
| `filters-combine` | Combine filter sets into one MESA instrument folder | [Filters](filters.md) |
| `filters-compose` | Compose an instrument's components and atmosphere into a product | [Filters](filters.md) |
| `filters-detector` | Tag curves' detector type, or correct or convert it in place | [Filters](filters.md) |
| `filters-inventory` | List every SVO, NJM, local and registered instrument | [Filters](filters.md) |
| `filters-build-components` | Build components and atmospheres from their original sources (mirror curation) | [NJM mirror](njm.md) |
| `vega` | Download NJM's Vega spectrum and print the `vega_sed` line | [MESA](mesa.md) |
| `mesa-path` | Print the MESA inlist line(s) for a local grid or filter set | [MESA](mesa.md) |
| `mesa_prepare` | Export one variant of a split grid as a self-contained MESA folder | [MESA](mesa.md) |
| `ml_completer` | Train and use the ML completer (in a terminal) | [Machine learning](ml.md) |
| `ml_generator` | Train and use the ML generator; `--auto` for a hyperparameter search | [Machine learning](ml.md) |
| `config` | Show or set the data directory | [Install](install.md) |

Two modules also run as commands: `python -m sed_tools.filters.label_check`
([Filters](filters.md)) and `python -m sed_tools.cube.tool --flux-cube PATH`,
which evaluates a spectrum from a flux cube, its bolometric magnitude and its
magnitudes in the installed filters, asks for anything not given, and saves a
plot to `plots/`.

Commands that ask questions (the pickers, `combine`, `rebuild` without
`--grids`, `mesa_prepare` without options, the flux-unit prompt) ask only in
a terminal. Without one, give the options; `grids` then needs `--grids` and
takes the whole grid unless a cut is given.

```bash
sed-tools filters-inventory --facility Generic
sed-tools filters-inventory --offline --json > inventory.json
```

## Common options

| Option | Commands | Meaning |
|---|---|---|
| `--base DIR` | grid and filter commands | Use this folder instead of the data directory's `stellar_models/` or `filters/` |
| `--grids NAME ...` (`--models`) | `grids`, `rebuild`, `combine`, `coverage` | The grids to work on |
| `--flux-unit UNIT` | `grids`, `rebuild`, `import` | Flux unit of spectra whose headers do not declare one: `flam`, `fnu`, `fnu_jy`, `flam_si`, `flam_si_nm`, `flam_si_um`, `normalized` |
| `--no-h5`, `--no-cube` | `grids`, `rebuild`, `import` | Skip the HDF5 bundle or the flux cube |
| `--workers N` | `grids` | Parallel downloads |
| `--components`, `--bands`, `--airmass`, `--pwv`, `--atmosphere-site`, `--detector-type` | `filters`, `filters-compose` | What to compose ([Filters](filters.md)) |
| `--offline`, `--json` | `filters-info`, `filters-inventory` | Use only local data and the cached catalogue; print JSON |

## Errors and exit codes

- A usage error prints the command's usage line and a plain message, and exits 2.
- A failed command prints `sed-tools: error: <message>` on stderr and exits 1.
- The interactive menu exits 1 at quit if any workflow failed.
- Ctrl-C prints `sed-tools: interrupted`.
- Set `SED_TOOLS_DEBUG=1` to see the Python traceback of an error.

```bash
SED_TOOLS_DEBUG=1 sed-tools config
```

## Environment variables

| Variable | Effect |
|---|---|
| `SED_DATA_DIR` | Use this data directory instead of the configured one |
| `SED_SVO_FPS_URL` | Use another SVO Filter Profile Service endpoint (`fps.php`) |
| `SED_TOOLS_DEBUG=1` | Show tracebacks |

SVO filter requests give up after a 10 s connect timeout. If the service is
unreachable, use the NJM mirror or the original-source builders, or point
SED_Tools at another endpoint:

```bash
export SED_SVO_FPS_URL=https://svo2.cab.inta-csic.es/theory/fps/fps.php
sed-tools filters-info Generic/Johnson --offline
```
