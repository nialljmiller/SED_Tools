# Stellar Spectra Toolkit

A Python toolkit for downloading, processing, filtering, combining, and precomputing stellar atmosphere spectra from SVO and MSG sources. Designed for integration with stellar evolution models like MESA.

## Features

- Download spectra from SVO and MSG grids.
- Filter and regenerate lookup tables.
- Combine multiple atmosphere models into unified grids.
- Precompute flux cubes for efficient access.

## Requirements

- Python 3.8+
- Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `astropy`, `astroquery`, `h5py`, `requests`, `beautifulsoup4`, `tqdm`

Install with:
```
pip install numpy scipy pandas matplotlib astropy astroquery h5py requests beautifulsoup4 tqdm
```

## Scripts and Usage

### 1. `svo_spectra_grabber.py`
Downloads stellar spectra from SVO.

```
python svo_spectra_grabber.py --output ../data/stellar_models/ --workers 5
```
- `--output`: Output directory (default: `../data/stellar_models/`).
- `--workers`: Parallel downloads (default: 5).
- `--models`: Specific models (e.g., `kurucz castelli`); interactive if omitted.

### 2. `msg_spectra_grabber.py`
Downloads and extracts spectra from MSG grids.

```
python msg_spectra_grabber.py --output ../data/stellar_models/ --workers 5
```
- Arguments mirror `svo_spectra_grabber.py`.

### 3. `svo_spectra_filter.py`
Filters lookup tables interactively.

```
python svo_spectra_filter.py
```
- Prompts for base directory and filters (e.g., Teff range).

### 4. `svo_regen_spectra_lookup.py`
Regenerates lookup tables from existing spectra files.

```
python svo_regen_spectra_lookup.py
```
- Scans `../../data/stellar_models/` by default.

### 5. `combine_stellar_atm.py`
Combines models into a unified grid.

```
python combine_stellar_atm.py --data_dir ../data/stellar_models/ --output combined_models --interactive
```
- `--data_dir`: Models directory.
- `--output`: Output subdirectory.
- `--interactive`: Select models (default: True).

### 6. `precompute_flux_cube.py`
Precomputes flux cube from a model directory.

```
python precompute_flux_cube.py --model_dir path/to/model --output flux_cube.bin
```
- `--model_dir`: Directory with lookup_table.csv and spectra.
- `--output`: Binary output file.

### 7. `svo_filter_grabber.py`
Downloads filter transmission curves from SVO.

```
python svo_filter_grabber.py
```
- Saves to `data/filters` by wavelength region.

## Directory Structure

- Spectra saved to `../data/stellar_models/<model_name>/`.
- Combined models to `../data/stellar_models/<output>/`.
- Filters to `data/filters/<facility>/<instrument>/`.

## License

MIT