Here’s a minimal `README.md` you can drop in:

````markdown
# SED_Tools

One-stop launcher to fetch stellar/atmosphere spectra and filters, and to build the standard products for each model.

**Providers**
- **SVO** (various synthetic & empirical sets)
- **MSG** (Rich Townsend’s grids; downloads `.h5`, extracts to `.txt`)
- **MAST (BOSZ 2024)** via official bulk scripts (no directory listings)

**Outputs (guaranteed per selected model)**
- `*.txt` spectra (collection)
- `lookup_table.csv`
- `<model>.h5` (bundle of all spectra)
- `flux_cube.bin`

Default data paths (relative to this repo):
- Spectra: `data/stellar_models/`
- Filters: `data/filters/`
````
