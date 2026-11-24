# SED_Tools documentation

SED_Tools downloads stellar spectral grids and photometric filter curves,
converts them to one set of units, and writes the files that MESA's `colors`
module and SED_Model read. These pages describe the command line
(`sed-tools`) and the Python API.

| Page | Contents |
|---|---|
| [Install](install.md) | Extras, installing from source, the data directory |
| [Quick start](quickstart.md) | A filter set, part of a grid, a spectrum and its magnitudes, the MESA lines |
| [Spectral grids](grids.md) | Sources, partial downloads, local grids, split grids and the all-axes cube, rebuild, combine, densify, import, coverage |
| [Filters](filters.md) | Fetching, storage, detector types, components and atmospheres, composing, combining, `label_check` |
| [Photometry and zero points](photometry.md) | AB and Vega magnitudes, band integration, filter properties, zero points, display colours |
| [MESA](mesa.md) | Inlist lines, `mesa-path`, `mesa_prepare`, Vega, what not to copy |
| [Machine learning](ml.md) | The SED completer and generator |
| [Command line](cli.md) | Every `sed-tools` command, the menu, exit codes, environment variables |
| [Python API](python-api.md) | `SED`, `SpectralGrid`, `Spectrum`, `Filters`, examples, the command-to-API table |
| [NJM mirror](njm.md) | What the mirror serves, its catalogue and cache, working without it, the builders |
| [Developing](developing.md) | Tests, golden tests, these pages, releases |
| [Citing](citing.md) | SED_Tools, the data providers, `Filters.cite` |

The changes in each release are in the
[changelog](https://github.com/nialljmiller/SED_Tools/blob/main/docs/source/changelog.rst),
and worked examples in the
[notebooks](https://github.com/nialljmiller/SED_Tools/tree/main/jupyter_notebooks).
In these pages, `~/.sed_tools/data` stands for the data directory
(`sed-tools config` shows yours).
