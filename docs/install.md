# Install

## From PyPI

```bash
pip install sed-tools
```

SED_Tools needs Python 3.10 or newer. The required packages are numpy, pandas,
tqdm, requests, beautifulsoup4, h5py, astropy and matplotlib (`tomli` as well
on Python 3.10).

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `full` | scipy, xlrd, scikit-learn, torch, dust_extinction | the ML completer and generator, `AfAv`, the Keck filter workbooks of the original-source builders |
| `extinction` | dust_extinction | only `AfAv`, the extinction ratio in the filter properties |
| `ml-tune` | optuna | `sed-tools ml_generator --auto` (together with `full`) |

```bash
pip install "sed-tools[full]"
pip install "sed-tools[extinction]"
pip install "sed-tools[full,ml-tune]"
```

The ML tools need torch and scikit-learn from `full`. `pyproject.toml` lists
every dependency.

## From source

```bash
git clone https://github.com/nialljmiller/SED_Tools.git
cd SED_Tools
pip install .          # or: pip install -e . for a development install
```

[Developing](developing.md) describes the test suite.

## The data directory

Everything SED_Tools downloads or builds is stored under one data directory,
`~/.sed_tools/data` by default:

```text
~/.sed_tools/data/
  stellar_models/     spectral grids, one folder per grid
  filters/            filter sets, filters/<Facility>/<Instrument>/
  atmospheres/        atmosphere sites used in composing filter curves
  cache/njm/          the NJM mirror's catalogue, cached
  vega/               vega_flam.csv, from `sed-tools vega`
```

```bash
sed-tools config                                   # show the data directory
sed-tools config --set ~/sed_data --move           # move it, with everything in it
```

`sed-tools config` also names the configuration file
(`~/.sed_tools/config.toml`). The environment variable `SED_DATA_DIR`, when
set, overrides the configured data directory, and `sed-tools config` says so. In these pages, `~/.sed_tools/data` stands
for the data directory; replace it if you have moved yours.

## Network access

SED_Tools reads from the NJM mirror, SVO, MSG, MAST and the instrument teams'
own sites. None of them is required for working with data that is already
installed. [NJM mirror](njm.md) says what happens when a source does not
answer.

If downloads fail or time out, lower the number of parallel workers, or name
the source:

```bash
sed-tools grids --grids bbody --workers 2
sed-tools grids --source njm --grids bbody
```
