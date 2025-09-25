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
- Spectra: `../data/stellar_models/`
- Filters: `../data/filters/`
````
---

## Quick start

```bash
python SED_tools.py
````

Interactive menu:

```
1) Spectra (SVO / MSG / MAST)
2) Filters (SVO)
3) Rebuild (lookup + HDF5 + flux cube)
```

* For **MAST/BOSZ**, you’ll be prompted to optionally limit metallicities (e.g. `-1.00,+0.00`) or press **Enter** for **all**.

Everything lands under `../data/stellar_models/<ModelName>/`.

---

## Command-line usage (non-interactive)

### Spectra

```bash
# interactive model picker, all providers
python SED_tools.py spectra --source all

# SVO only (choose from list)
python SED_tools.py spectra --source svo

# Specific models (mixed): use 'src:model'
python SED_tools.py spectra --models svo:bt-settl msg:sg-SPHINX mast:BOSZ-2024-r10000

# Change output location; skip flux cube; skip HDF5 bundle
python SED_tools.py spectra --base /path/to/stellar_models --no-cube --no-h5
```

### Filters (SVO)

```bash
python SED_tools.py filters
```

You’ll be asked for optional substrings (Facility / Instrument / Band) and a wavelength range.
Filters are written under `../data/filters/<Facility>/<Instrument>/<Band>.dat` (CSV-ascii).

### Rebuild (no download)

```bash
# Rebuild lookup_table.csv, HDF5, and flux cube for local models
python SED_tools.py rebuild

# Rebuild only lookup + HDF5
python SED_tools.py rebuild --no-cube

# Rebuild a specific local model folder
python SED_tools.py rebuild --models BOSZ-2024-r10000
```

---

## What gets written

Example: `../data/stellar_models/BOSZ-2024-r10000/`

```
BOSZ-2024-r10000/
  bosz2024_ms_t05000_g+4.5_m+0.00_a+0.00_c+0.00_v2_r10000_resam.txt
  ...
  lookup_table.csv
  BOSZ-2024-r10000.h5
  flux_cube.bin
```

Each `.txt` has a comment header with parsed parameters and source URL.
HDF5 bundles all spectra under `/spectra/<filename>/{lambda,flux}` (plus attributes).
`flux_cube.bin` is produced by the provided `precompute_flux_cube` tool.

---

## Dependencies

Install what you don’t already have:

```bash
pip install numpy pandas requests beautifulsoup4 h5py astropy astroquery tqdm
```

* Filters require `astropy` + `astroquery`.
* MAST/BOSZ uses `requests` + `bs4` (no astroquery needed).
* Your existing grabbers/flux cube code are used as-is.

