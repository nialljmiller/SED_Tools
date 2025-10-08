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

Use `flux_cube_tool.py` to open a `flux_cube.bin`, interpolate an SED at arbitrary `(Teff, logg, [M/H])`, compute bolometric and synthetic filter magnitudes, and optionally generate diagnostic plots::

    python flux_cube_tool.py \
        --flux-cube data/stellar_models/<model>/flux_cube.bin \
        --teff 6000 --logg 4.5 --metallicity 0.0 \
        --filters data/filters/Johnson/U.dat data/filters/Johnson/B.dat \
        --plot sed.png --save-sed sed.csv

By default the magnitudes are reported relative to a reference flux of `1.0`, but alternative zero points can be provided with `--bolometric-reference-flux`/`--bolometric-reference-mag` and the analogous filter options.

Default data paths (relative to this repo):
- Spectra: `data/stellar_models/`
- Filters: `data/filters/`
````
