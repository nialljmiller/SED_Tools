# The NJM mirror

The NJM mirror, [data.nialljmiller.com/sed_tools](https://data.nialljmiller.com/sed_tools/),
serves data SED_Tools has already prepared:

- spectral grids, standardised and built (flux cubes, lookup tables, HDF5
  bundles), with cuts made on the server;
- filter sets, with their components, atmosphere sites and the files
  `manifest.json`, `properties.json` and `info.json`;
- a catalogue: `index.json`, which lists every facility, instrument and band,
  and each instrument's `properties.json` and `info.json`;
- the reference spectra its properties use: `vega/vega_flam.csv` (the same
  file as MESA's) and `sun/sun_flam.csv`.

Its website lists the served grids and filter sets, with each instrument's
band properties, references and licence. It is a data archive; these pages are
the manual.

The mirror is optional. Everything works without it, from installed copies
and the other sources, as described below. It is a single server; please do
not script large repeated downloads from it.

## Order of sources

- **Filters:** an installed copy, then NJM, then the instrument team's original
  files (the original-source builders), then the SVO Filter Profile Service.
  NJM and the builders are asked under their own spelling of a name, matched
  without regard to case or punctuation, so `CTIO/S-PLUS` fetches NJM's
  `CTIO/S_PLUS`.
- **Spectral grids:** NJM, then SVO, MSG and MAST.
- **Atmospheres:** installed grids, then NJM, then the team's original files or
  ESO SkyCalc.

Facility and instrument names, their one-line descriptions, and which SVO
entries are the same product as an NJM instrument come from a catalogue shipped
with the package
([`sed_tools/instrument_catalogue.json`](https://github.com/nialljmiller/SED_Tools/blob/main/sed_tools/instrument_catalogue.json)).
A reachable NJM adds to it, so the pickers show the same names with or without
NJM.

## The catalogue cache

Catalogue files are cached in `~/.sed_tools/data/cache/njm/`. `index.json` is
checked again daily; the other files are revalidated, by ETag, only when
`index.json` changes. Without the mirror the cached copies are used.
`filters-info --offline` and `offline=True` in Python never contact it, and
`refresh=True` asks again.

```bash
sed-tools filters-info GAIA/GAIA --offline
```

```python
from sed_tools.api import Filters

print(Filters.properties("GAIA", "GAIA", refresh=True)["bands"]["G"]["WavelengthPivot"])
```

A downloaded instrument keeps its own `properties.json` and `info.json`
beside the curves. These copies are preferred, because they describe exactly
those curves.

## When NJM does not answer

- Its requests give up after a 5 s connect timeout. One warning per command
  names the sources used instead, and NJM is not asked again during that run
  (for up to five minutes); the next command tries again.
- Filter sets come from SVO or the original-source builders. An NJM set that
  SVO holds under other names (NJM's `GAIA/GAIA` is SVO's `GAIA/GAIA3`) is
  installed from SVO under NJM's names. If SVO has only some of its bands,
  those are installed and a warning names the missing ones (`GAIA/GAIA`: 4 of
  6, without `Gbp_bright` and `Gbp_faint`); `filters-info` and the picker
  summary mark the set "Incomplete", and `Filters.available()` gives the
  installed `bands` and all `expected_bands`. Fetch it again when NJM is back
  to complete it.
- `sed-tools grids` says that NJM's grids are not offered and lists the
  others. A grid from SVO is downloaded whole: `sed-tools grids` builds it;
  after `SED.fetch` with bounds, run `sed-tools rebuild --grids NAME`.
- `filters-info` says there is no catalogue description and uses the cached
  catalogue.
- Atmospheres: team sites are rebuilt from their original files, and SkyCalc
  sites generate the requested point.
- `sed-tools vega` and `Filters.fetch_vega()` need NJM or a cached copy.

If downloads from any source fail or time out, see [Install](install.md).

## Building components from the original sources

The NJM mirror runs the original-source builders ahead of time
(`sed-tools filters-build-components all`). Anyone can run them. They write
into the filter root: an instrument's components into its installed folder
(fetch the instrument first), an atmosphere into `atmospheres/<site>/`:

```bash
sed-tools filters-build-components team_atmospheres --sites apo_sdss --archive ~/evidence
```

- The builders are listed by `sed-tools filters-build-components --help`;
  `all` runs those that need no input.
- `--archive DIR` keeps every raw download with its URL, time and SHA-256.
- `--raw DIR` reads saved official files when a source is unavailable.
- For the table-based sets, `--instrument` names the instrument: `sed-tools
  filters-build-components tabular --instrument PANIC` (or `INGRID`, `NSFCam`,
  `IOO`, `HAWKI`, `NACO`, `SPHERE`, `GSAOI`, `NIFS`). Their root curves are
  measured filters, not full system throughput.
- Builders marked as large downloads need `--allow-heavy` (`allow_heavy=True`
  in Python); none is marked at present.

Builders register a team atmosphere with
`component_sources._common.SiteAtmosphere` (written by `write_site`;
`SimpleSource.site`/`sites`, or `Instrument.sites` for original-source
recipes);
[`component_sources/team_atmospheres.py`](https://github.com/nialljmiller/SED_Tools/blob/main/sed_tools/component_sources/team_atmospheres.py)
holds the registered ones. [Developing](developing.md) covers the label rules
the mirror checks.
