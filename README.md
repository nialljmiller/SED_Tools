Here’s a minimal `README.md` you can drop in:

````markdown
# SED_Tools

One-stop launcher to fetch stellar/atmosphere spectra and filters, and to build the standard products for each model.  The repository also includes tooling to inspect pre-computed flux cubes and evaluate interpolated SEDs.

**Providers**
- **SVO** (various synthetic & empirical sets)
- **MSG** (Rich Townsend’s grids; downloads `.h5`, extracts to `.txt`)
- **MAST (BOSZ 2024)** via official bulk scripts (no directory listings)

**Outputs (guaranteed per selected model)**
- `*.txt` spectra (collection)
- `lookup_table.csv`
- `<model>.h5` (bundle of all spectra)
- `flux_cube.bin`

## Flux cube inspector

Use `flux_cube_tool.py` to open a `flux_cube.bin`, interpolate an SED at arbitrary `(Teff, logg, [M/H])`, compute bolometric and synthetic filter magnitudes, and generate diagnostic plots. At its simplest you only need to supply the flux cube path (the tool accepts either the exact file or the directory that contains it)::

    python flux_cube_tool.py --flux-cube data/stellar_models/<model>/

The script will report the valid parameter ranges, prompt for any missing `Teff`, `logg`, or `[M/H]` values, automatically discover every filter curve beneath `data/filters/` (or any directories/files you pass via `--filters`), evaluate magnitudes for each, and save a plot to `plots/` by default. Additional options allow you to override the plot destination (`--plot`), persist the interpolated SED (`--save-sed`), or change the magnitude zero points via `--bolometric-reference-flux`/`--bolometric-reference-mag` and the analogous filter settings.

Default data paths (relative to this repo):
- Spectra: `data/stellar_models/`
- Filters: `data/filters/`
````
