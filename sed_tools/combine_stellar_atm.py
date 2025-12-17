#!/usr/bin/env python3
import argparse
import os
import shutil
import struct
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
import h5py
import glob

SIGMA = 5.670374419e-5  # erg s-1 cm-2 K-4
SPEED_OF_LIGHT_ANG_S = 2.99792458e18  # speed of light in Angstrom/s (for F_nu conversion)



def prepare_sed_with_scale(filepath, teff, scale_factor=1.0, model_dir=None):
    result = validate_and_clean(filepath, model_dir=model_dir)
    if not result.usable:
        raise ValueError(f"{filepath}: {result.reason}")

    wl = result.wavelength
    flux = result.flux

    flux = flux * scale_factor

    return wl, flux


def load_sed(filepath):
    return np.loadtxt(filepath, unpack=True)


def prepare_sed(filepath, teff, correction=1.0):
    wl, flux = load_sed(filepath)
    flux = flux * correction  # Apply BEFORE normalization
    return wl, flux




@dataclass
class FileValidation:
    """Result of validating one spectrum file."""
    usable: bool
    reason: str
    wavelength: Optional[np.ndarray] = None  # Cleaned wavelength (in Angstroms)
    flux: Optional[np.ndarray] = None        # Cleaned flux (NOT scaled, just cleaned)


def validate_and_clean(filepath: str, model_dir: str) -> FileValidation:
    """
    Validate and clean one spectrum file with comprehensive unit detection.
    
    Ensures output is:
    - Wavelength: Angstroms (Å)
    - Flux: erg/s/cm²/Å (F_lambda)
    
    Returns FileValidation with:
    - usable: True if file can be used
    - wavelength: Cleaned wavelength array in Angstroms
    - flux: Cleaned flux array in erg/s/cm²/Å
    - reason: Why file was skipped (if unusable) or conversion info
    """
    
    # Stage 1: Check file format (skip XML)
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline()
            if '<?xml' in first_line.lower():
                return FileValidation(usable=False, reason="XML file")
    except:
        return FileValidation(usable=False, reason="Cannot read")
    
    # Stage 2: Load data
    try:
        data = np.loadtxt(filepath, unpack=True)
        if data.ndim != 2 or data.shape[0] < 2:
            return FileValidation(usable=False, reason="Wrong shape")
        wl_raw = data[0].copy()
        flux_raw = data[1].copy()
    except Exception as e:
        return FileValidation(usable=False, reason=f"Load failed: {str(e)[:30]}")
    
    # Stage 3: Basic quality checks
    if len(wl_raw) < 10:
        return FileValidation(usable=False, reason="Too few points")
    
    # Stage 4: Remove NaN/inf from BOTH wavelength and flux
    good = np.isfinite(wl_raw) & np.isfinite(flux_raw)
    if np.sum(good) < 10:
        return FileValidation(usable=False, reason="Too many NaN/inf values")
    wl_raw = wl_raw[good]
    flux_raw = flux_raw[good]
    
    # Stage 5: Check if wavelength is index grid (0, 1, 2, ...)
    if _is_index_grid(wl_raw):
        # Try to recover real wavelength from HDF5
        fixed_wl = _try_recover_wavelength_from_h5(filepath, model_dir, len(wl_raw))
        if fixed_wl is not None:
            wl_raw = fixed_wl
        else:
            return FileValidation(usable=False, reason="Index grid, can't recover wavelength")
    
    # Stage 6: Sort by wavelength
    order = np.argsort(wl_raw)
    wl_raw = wl_raw[order]
    flux_raw = flux_raw[order]
    
    # Stage 7: Remove duplicate wavelengths
    unique_mask = np.concatenate([[True], np.diff(wl_raw) > 0])
    if np.sum(unique_mask) < 10:
        return FileValidation(usable=False, reason="No unique wavelengths")
    wl_raw = wl_raw[unique_mask]
    flux_raw = flux_raw[unique_mask]
    
    # === ENHANCED UNIT DETECTION AND CONVERSION ===
    
    # Stage 8a: Detect wavelength units
    wl_unit_factor = _detect_wavelength_unit_from_header(filepath)
    if wl_unit_factor is None:
        wl_unit_factor = _detect_wavelength_unit(wl_raw)
    if wl_unit_factor is None:
        return FileValidation(usable=False, reason="Can't determine wavelength units")
    
    # Stage 8b: Detect flux units
    flux_unit = _detect_flux_unit_from_header(filepath)
    if flux_unit is None:
        # Default assumption: F_lambda (most common for stellar atmospheres)
        flux_unit = 'flam'
    
    # Stage 8c: Convert wavelength to Angstroms
    wl = wl_raw * wl_unit_factor
    
    # Stage 8d: Convert flux to F_lambda (erg/s/cm²/Å)
    if flux_unit == 'fnu':
        # Convert F_nu to F_lambda
        flux = _convert_fnu_to_flam(flux_raw, wl)
    elif flux_unit == 'jy':
        # Convert Jansky to erg/s/cm²/Hz first, then to F_lambda
        flux_fnu = flux_raw * 1e-23  # 1 Jy = 1e-23 erg/s/cm²/Hz
        flux = _convert_fnu_to_flam(flux_fnu, wl)
    else:
        # Already F_lambda, just copy
        flux = flux_raw.copy()
    
    # Stage 8e: Adjust flux for wavelength unit conversion
    # F_lambda (per Å) = F_lambda (per original unit) / conversion_factor
    flux = flux / wl_unit_factor
    
    # Stage 8f: Validate converted units make sense
    is_valid, validation_msg = _validate_standard_units(wl, flux)
    if not is_valid:
        return FileValidation(usable=False, reason=f"Unit validation failed: {validation_msg}")
    
    # Stage 9: Ensure wavelength is positive and finite (should be true but double-check)
    if not (np.all(wl > 0) and np.all(np.isfinite(wl))):
        return FileValidation(usable=False, reason="Invalid wavelengths after conversion")
    
    # Stage 10: Remove negative flux values
    positive = flux > 0
    if np.sum(positive) < 10:
        return FileValidation(usable=False, reason="No positive flux values")
    wl = wl[positive]
    flux = flux[positive]
    
    # Stage 11: Remove catastrophic outliers (indicates corruption)
    flux_median = np.median(flux)
    flux_mad = np.median(np.abs(flux - flux_median))
    if flux_mad > 0:
        # Remove extreme outliers (>100 MAD from median)
        outlier_mask = np.abs(flux - flux_median) > 100 * flux_mad
        if np.sum(outlier_mask) > 0.5 * len(flux):
            return FileValidation(usable=False, reason="Majority are outliers (corrupted)")
        # Remove the outliers
        if np.sum(outlier_mask) > 0:
            wl = wl[~outlier_mask]
            flux = flux[~outlier_mask]
    
    # Final check
    if len(wl) < 10 or len(flux) < 10:
        return FileValidation(usable=False, reason="Too few valid points after cleaning")
    
    # Build reason string with conversion info
    unit_info = f"wl_factor={wl_unit_factor}, flux_unit={flux_unit}"
    
    # Return cleaned data (wavelength in Angstroms, flux in erg/s/cm²/Å)
    return FileValidation(
        usable=True,
        reason=f"OK ({unit_info})",
        wavelength=wl,
        flux=flux
    )


def _is_index_grid(wl):
    """Detect if wavelength is actually an index (0, 1, 2, ...) not physical units."""
    if len(wl) < 10:
        return False
    
    # Check if it starts near zero or one
    if not (wl[0] < 2.0):
        return False
    
    # Check if differences are close to 1
    diffs = np.diff(wl)
    median_diff = np.median(diffs)
    
    if 0.9 < median_diff < 1.1:
        # Check if most differences are close to 1
        close_to_one = np.abs(diffs - 1.0) < 0.1
        if np.sum(close_to_one) > 0.9 * len(diffs):
            return True
    
    return False


def _try_recover_wavelength_from_h5(txt_filepath: str, model_dir: str, expected_len: int) -> Optional[np.ndarray]:
    """
    Try to recover wavelength grid from HDF5 source file.
    
    This handles the case where MSG extraction produced index grids.
    """
    # Parse header to find HDF5 source info
    try:
        with open(txt_filepath, 'r') as f:
            spec_group = None
            for line in f:
                if not line.startswith('#'):
                    break
                if 'spec_group' in line and '=' in line:
                    spec_group = line.split('=', 1)[1].strip()
                    break
            if spec_group is None:
                return None
    except:
        return None
    
    # Find HDF5 file in model directory
    h5_files = glob.glob(os.path.join(model_dir, "*.h5"))
    
    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, 'r') as f:
                if spec_group not in f:
                    continue
                
                spec_g = f[spec_group]
                wl = _recover_wavelengths_from_group(spec_g, expected_len)
                
                if wl is not None and len(wl) == expected_len:
                    # Verify it's monotonic and positive
                    if np.all(wl > 0) and np.all(np.diff(wl) > 0):
                        return wl
        except:
            continue
    
    return None


def _recover_wavelengths_from_group(spec_g, expected_len=None):
    """Extract wavelength from HDF5 group."""
    # Try direct dataset
    KEYS = ("lambda", "wavelength", "wave", "wl", "wavelength_A")
    for key in KEYS:
        if key in spec_g and isinstance(spec_g[key], h5py.Dataset):
            try:
                wl = np.array(spec_g[key][()]).astype(float).squeeze()
                if wl.size > 1 and np.all(np.diff(wl) > 0):
                    return wl
            except:
                continue
    
    # Try range/x
    if "range" in spec_g and isinstance(spec_g["range"], h5py.Group):
        rg = spec_g["range"]
        if "x" in rg and isinstance(rg["x"], h5py.Dataset):
            try:
                wl = np.array(rg["x"][()]).astype(float).ravel()
                if wl.size > 1 and np.all(np.diff(wl) > 0):
                    return wl
            except:
                pass
    
    return None


def _detect_wavelength_unit(wl):
    """
    Infer wavelength units from data range.
    Returns conversion factor to Angstroms, or None if uncertain.
    """
    wl_min, wl_max = wl.min(), wl.max()
    
    # Safety check for index grids (should be caught earlier)
    if wl_min < 10 and wl_max < 1000:
        # Could be index grid or valid data in m/cm
        # Check if values look like indices
        if np.allclose(np.diff(wl), 1.0, rtol=0.1):
            return None  # Likely index grid, reject
    
    # Angstroms: typical range 1-100,000 Å (UV to far-IR)
    # Most stellar spectra are 1000-100000 Å but bbody can start at 1 Å
    if wl_min >= 1 and wl_max <= 1e6:
        return 1.0
    
    # Nanometers: typical range 100-10,000 nm
    if 10 < wl_max <= 50000 and wl_min > 5:
        # If max is in hundreds to thousands, likely nm
        if wl_max < 20000:  # Most stellar spectra don't exceed 20000 nm
            return 10.0
    
    # Micrometers: typical range 0.1-1000 μm
    if 0.05 < wl_max <= 5000 and wl_max > 0.1:
        # If max is in tens to hundreds, likely μm
        if wl_max < 1000:
            return 1e4
    
    # Centimeters: rare but possible for radio
    if 1e-6 < wl_max <= 10 and wl_max < 0.1:
        return 1e8
    
    # Meters: also rare
    if 1e-8 < wl_max <= 1 and wl_max < 0.01:
        return 1e10
    
    # If we get here, units are ambiguous
    return None


def _detect_wavelength_unit_from_header(filepath: str) -> Optional[float]:
    """
    Parse header for wavelength unit hints.
    Returns conversion factor to Angstroms, or None if not found.
    """
    try:
        with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
            for _ in range(100):  # Check more lines
                line = f.readline()
                if not line:
                    break
                if not line.startswith("#"):
                    break
                u = line.lower()
                
                # Look for wavelength-related lines
                if any(kw in u for kw in ['wavelength', 'lambda', 'wave']):
                    # Check for unit indicators
                    if 'angstrom' in u or '(a)' in u or '(å)' in u:
                        return 1.0
                    if 'nanometer' in u or '(nm)' in u:
                        return 10.0
                    if 'micron' in u or 'micrometer' in u or '(um)' in u or '(µm)' in u:
                        return 1e4
                    if '(cm)' in u:
                        return 1e8
                    if '(m)' in u and 'nm' not in u:  # meter but not nanometer
                        return 1e10
    except:
        pass
    return None


def _detect_flux_unit_from_header(filepath: str) -> Optional[str]:
    """
    Parse header for flux unit hints.
    Returns 'flam' (F_lambda), 'fnu' (F_nu), 'jy' (Jansky), or None.
    """
    try:
        with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
            for _ in range(100):
                line = f.readline()
                if not line:
                    break
                if not line.startswith("#"):
                    break
                u = line.lower()
                
                # Look for flux-related lines
                if any(kw in u for kw in ['flux', 'f_lambda', 'flam', 'f_nu', 'fnu', 'intensity']):
                    # Check for F_nu indicators
                    if any(kw in u for kw in ['f_nu', 'fnu', 'hz', 'frequency']):
                        if 'jy' in u or 'jansky' in u:
                            return 'jy'
                        return 'fnu'
                    
                    # Check for F_lambda indicators (most common)
                    if any(kw in u for kw in ['f_lambda', 'flam', 'f_lam', 'lambda']):
                        return 'flam'
                    
                    # If it just says "flux" with wavelength context, assume F_lambda
                    if 'wavelength' in u or 'angstrom' in u:
                        return 'flam'
    except:
        pass
    return None


def _convert_fnu_to_flam(flux_fnu, wl_angstrom):
    """
    Convert F_nu (erg/s/cm²/Hz) to F_lambda (erg/s/cm²/Å).
    
    Formula: F_lambda = F_nu * (c / λ²)
    where c = speed of light in Å/s, λ in Å
    """
    # Avoid division by zero
    wl_safe = np.maximum(wl_angstrom, 1.0)
    conversion_factor = SPEED_OF_LIGHT_ANG_S / (wl_safe ** 2)
    return flux_fnu * conversion_factor


def _validate_standard_units(wl, flux):
    """
    Validate that wavelength and flux are in reasonable ranges.
    
    Returns (is_valid: bool, reason: str)
    """
    # Wavelength checks (should be in Angstroms)
    if wl.min() < 0.1:  # Allow down to X-ray range
        return False, f"Wavelength too small: {wl.min():.2e} Å"
    
    if wl.max() > 1e7:  # Up to far-IR/radio
        return False, f"Wavelength too large: {wl.max():.2e} Å"
    
    # Check wavelength span is reasonable
    if wl.max() / max(wl.min(), 1e-10) < 1.1:
        return False, f"Wavelength range too narrow: {wl.min():.1f} - {wl.max():.1f} Å"
    
    # Flux checks - ONLY check for obviously broken data, NOT absolute magnitudes
    # (flux hasn't been normalized yet, so magnitude checks are premature)
    flux_finite = flux[np.isfinite(flux)]
    if len(flux_finite) == 0:
        return False, "No finite flux values"
    
    # Check for all negative flux (sign error)
    if np.all(flux_finite <= 0):
        return False, "All flux values are negative or zero"
    
    # Check for all zero flux (empty/broken file)
    if np.all(flux_finite == 0):
        return False, "All flux values are zero"
    
    # Check fraction of negative values (some noise is OK)
    neg_frac = np.sum(flux_finite < 0) / len(flux_finite)
    if neg_frac > 0.5:  # Allow up to 50% negative (generous for noisy data)
        return False, f"Too many negative flux values: {neg_frac:.1%}"
    
    return True, "OK"


def find_stellar_models(base_dir="../data/stellar_models/"):
    """Find all stellar model directories containing lookup tables."""
    model_dirs = []

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            lookup_file = os.path.join(item_path, "lookup_table.csv")
            if os.path.exists(lookup_file):
                model_dirs.append((item, item_path))

    return model_dirs


def select_models_interactive(model_dirs):
    """Present model options to user and get selection."""
    print("\nAvailable stellar atmosphere models:")
    print("-" * 50)
    for idx, (name, path) in enumerate(model_dirs, start=1):
        # Count models in directory
        lookup_file = os.path.join(path, "lookup_table.csv")
        try:
            df = pd.read_csv(lookup_file, comment="#")
            n_models = len(df)
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
            n_models = "?"
        print(f"{idx}. {name} ({n_models} models)")

    print("\nEnter the numbers of models to combine (comma-separated):")
    print("Example: 1,3,5 or 'all' for all models")

    user_input = input("> ").strip()

    if user_input.lower() == "all":
        return model_dirs

    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(",")]
        selected = [model_dirs[i] for i in indices if 0 <= i < len(model_dirs)]
        return selected
    except (ValueError, IndexError):
        print("Invalid input. Using all models.")
        return model_dirs


def load_model_data(model_path):
    """Load lookup table and extract parameter information."""
    lookup_file = os.path.join(model_path, "lookup_table.csv")

    # Read the CSV, handling the # comment character
    with open(lookup_file, "r") as f:
        # Read header line
        header = f.readline().strip()
        if header.startswith("#"):
            header = header[1:].strip()
        columns = [col.strip() for col in header.split(",")]

    # Read the data
    df = pd.read_csv(lookup_file, comment="#", names=columns, skiprows=1)

    # Extract parameters
    file_col = columns[0]  # Usually 'file_name' or 'filename'

    # Find parameter columns (case-insensitive)
    teff_col = None
    logg_col = None
    meta_col = None

    for col in columns:
        col_lower = col.lower()
        if "teff" in col_lower:
            teff_col = col
        elif "logg" in col_lower or "log(g)" in col_lower:
            logg_col = col
        elif "meta" in col_lower or "feh" in col_lower or "m/h" in col_lower:
            meta_col = col

    data = {
        "files": df[file_col].values,
        "teff": df[teff_col].values if teff_col else np.zeros(len(df)),
        "logg": df[logg_col].values if logg_col else np.zeros(len(df)),
        "meta": df[meta_col].values if meta_col else np.zeros(len(df)),
        "model_dir": model_path,
    }

    return data


def create_unified_grid(all_models_data):
    """Create unified parameter grids from all models."""
    # Collect all parameter values
    all_teff = []
    all_logg = []
    all_meta = []

    for data in all_models_data:
        all_teff.extend(data["teff"])
        all_logg.extend(data["logg"])
        all_meta.extend(data["meta"])

    # Get unique sorted values
    teff_grid = np.unique(np.sort(all_teff))
    logg_grid = np.unique(np.sort(all_logg))
    meta_grid = np.unique(np.sort(all_meta))

    # Remove values that are too close (within tolerance)
    def clean_grid(grid, tol=1e-6):
        if len(grid) <= 1:
            return grid
        cleaned = [grid[0]]
        for val in grid[1:]:
            if abs(val - cleaned[-1]) > tol:
                cleaned.append(val)
        return np.array(cleaned)

    teff_grid = clean_grid(teff_grid, tol=1.0)  # 1K tolerance for Teff
    logg_grid = clean_grid(logg_grid, tol=0.01)  # 0.01 dex for log g
    meta_grid = clean_grid(meta_grid, tol=0.01)  # 0.01 dex for [M/H]

    return teff_grid, logg_grid, meta_grid


def create_common_wavelength_grid(all_models_data, sample_size=20):
    """Create a common wavelength grid by sampling models."""
    print("\nAnalyzing wavelength coverage across models...")

    min_wave = float("inf")
    max_wave = 0
    resolutions = []

    for model_data in all_models_data:
        # Sample a few SEDs from this model
        n_sample = min(sample_size, len(model_data["files"]))
        indices = np.random.choice(len(model_data["files"]), n_sample, replace=False)

        for idx in indices:
            filepath = os.path.join(model_data["model_dir"], model_data["files"][idx])
            try:
                wavelengths, _ = load_sed(filepath)

                if len(wavelengths) > 10:
                    # Focus on optical/near-IR
                    mask = (wavelengths >= 3000) & (wavelengths <= 25000)
                    if mask.sum() > 10:
                        wl_subset = wavelengths[mask]
                        min_wave = min(min_wave, wl_subset.min())
                        max_wave = max(max_wave, wl_subset.max())

                        # Estimate resolution
                        resolution = np.median(np.diff(wl_subset))
                        resolutions.append(resolution)

            except (ValueError, IndexError, TypeError):
                continue

    # Create wavelength grid
    typical_resolution = np.median(resolutions) if resolutions else 50.0
    grid_resolution = max(50.0, typical_resolution * 2)

    n_points = int((max_wave - min_wave) / grid_resolution) + 1
    n_points = min(n_points, 5000)  # Cap at 5000 points

    wavelength_grid = np.linspace(min_wave, max_wave, n_points)

    print(f"  Wavelength range: {min_wave:.0f} - {max_wave:.0f} Å")
    print(f"  Grid points: {len(wavelength_grid)}")
    print(f"  Resolution: {grid_resolution:.1f} Å")

    return wavelength_grid



def build_combined_flux_cube(
    all_models_data, teff_grid, logg_grid, meta_grid, wavelength_grid
):
    """Build the combined flux cube from all models."""
    n_teff = len(teff_grid)
    n_logg = len(logg_grid)
    n_meta = len(meta_grid)
    n_lambda = len(wavelength_grid)

    # Initialize flux cube and tracking arrays
    flux_cube = np.zeros((n_teff, n_logg, n_meta, n_lambda))
    filled_map = np.zeros((n_teff, n_logg, n_meta), dtype=bool)
    source_map = (
        np.zeros((n_teff, n_logg, n_meta), dtype=int) - 1
    )  # Which model filled each point

    print(f"\nBuilding combined flux cube: {flux_cube.shape}")
    print(f"Memory requirement: {flux_cube.nbytes / (1024**2):.1f} MB")

    # All models are now in same units - no cross-normalization needed
    normalization_factors = {}
    for model_idx, model_data in enumerate(all_models_data):
        normalization_factors[model_idx] = np.ones(len(model_data["files"]))


    # Build KDTree for fast nearest neighbor searches
    # Normalize parameters for distance calculation
    teff_norm = (teff_grid - teff_grid.min()) / (teff_grid.max() - teff_grid.min())
    logg_norm = (logg_grid - logg_grid.min()) / (logg_grid.max() - logg_grid.min())
    meta_norm = (meta_grid - meta_grid.min()) / (meta_grid.max() - meta_grid.min())

    # Process each model
    for model_idx, model_data in enumerate(all_models_data):
        model_name = os.path.basename(model_data["model_dir"])
        print(f"\nProcessing {model_name}...")

        norm_factors = normalization_factors[model_idx]

        for i, (file, teff, logg, meta) in enumerate(
            tqdm(
                zip(
                    model_data["files"],
                    model_data["teff"],
                    model_data["logg"],
                    model_data["meta"],
                ),
                total=len(model_data["files"]),
                desc=f"Model {model_idx + 1}",
            )
        ):
            # Find grid indices
            i_teff = np.searchsorted(teff_grid, teff)
            i_logg = np.searchsorted(logg_grid, logg)
            i_meta = np.searchsorted(meta_grid, meta)

            # Clip to valid range
            i_teff = np.clip(i_teff, 0, n_teff - 1)
            i_logg = np.clip(i_logg, 0, n_logg - 1)
            i_meta = np.clip(i_meta, 0, n_meta - 1)

            # Load and interpolate SED

            filepath = os.path.join(model_data["model_dir"], file)
            scale = model_data.get("scale", 1.0)

            if "fid0" not in filepath:

                model_wavelengths, model_fluxes = prepare_sed_with_scale(
                    filepath, teff, scale, model_dir=model_data["model_dir"]
                )

                # Apply normalization factor
                model_fluxes *= norm_factors[i]

                # Interpolate to common grid (in log space for flux)
                log_fluxes = np.log10(np.maximum(model_fluxes, 1e-50))
                log_interpolated = np.interp(
                    wavelength_grid,
                    model_wavelengths,
                    log_fluxes,
                    left=log_fluxes[0],
                    right=log_fluxes[-1],
                )
                interpolated_flux = 10**log_interpolated

                # Store in cube
                flux_cube[i_teff, i_logg, i_meta, :] = interpolated_flux
                filled_map[i_teff, i_logg, i_meta] = True
                source_map[i_teff, i_logg, i_meta] = model_idx


    # Fill gaps using nearest neighbor interpolation
    empty_points = np.sum(~filled_map)
    if empty_points > 0:
        print(f"\nFilling {empty_points} empty grid points...")

        # Get filled points
        filled_indices = np.where(filled_map)
        filled_points = np.column_stack(
            [
                teff_norm[filled_indices[0]],
                logg_norm[filled_indices[1]],
                meta_norm[filled_indices[2]],
            ]
        )

        # Build KDTree
        tree = KDTree(filled_points)

        # Fill empty points
        for i_teff in range(n_teff):
            for i_logg in range(n_logg):
                for i_meta in range(n_meta):
                    if not filled_map[i_teff, i_logg, i_meta]:
                        # Find nearest filled point
                        query_point = [
                            teff_norm[i_teff],
                            logg_norm[i_logg],
                            meta_norm[i_meta],
                        ]
                        dist, idx = tree.query(query_point, k=1)

                        # Copy flux from nearest neighbor
                        src_i = filled_indices[0][idx]
                        src_j = filled_indices[1][idx]
                        src_k = filled_indices[2][idx]

                        flux_cube[i_teff, i_logg, i_meta, :] = flux_cube[
                            src_i, src_j, src_k, :
                        ]
                        filled_map[i_teff, i_logg, i_meta] = True
                        source_map[i_teff, i_logg, i_meta] = source_map[
                            src_i, src_j, src_k
                        ]

    return flux_cube, source_map





def save_combined_data(
    output_dir,
    teff_grid,
    logg_grid,
    meta_grid,
    wavelength_grid,
    flux_cube,
    all_models_data,
):
    """Save the combined data and create unified lookup table."""
    os.makedirs(output_dir, exist_ok=True)

    # Save binary flux cube
    binary_file = os.path.join(output_dir, "flux_cube.bin")
    with open(binary_file, "wb") as f:
        # Write dimensions
        f.write(
            struct.pack(
                "4i",
                len(teff_grid),
                len(logg_grid),
                len(meta_grid),
                len(wavelength_grid),
            )
        )

        # Write grid arrays
        teff_grid.astype(np.float64).tofile(f)
        logg_grid.astype(np.float64).tofile(f)
        meta_grid.astype(np.float64).tofile(f)
        wavelength_grid.astype(np.float64).tofile(f)

        # Write flux cube
        flux_cube.astype(np.float64).tofile(f)

    print(f"\nSaved binary flux cube to: {binary_file}")

    # Create combined lookup table
    lookup_data = []
    file_counter = 0
    copied_files = 0

    print("\nCopying SED files to combined directory...")

    for model_data in tqdm(all_models_data, desc="Copying models"):
        model_name = os.path.basename(model_data["model_dir"])
        for i, (orig_file, teff, logg, meta) in enumerate(
            zip(
                model_data["files"],
                model_data["teff"],
                model_data["logg"],
                model_data["meta"],
            )
        ):
            # Create new filename that includes source model
            new_filename = f"{model_name}_{file_counter:06d}.txt"
            lookup_data.append(
                {
                    "file_name": new_filename,
                    "teff": teff,
                    "logg": logg,
                    "meta": meta,
                    "source_model": model_name,
                    "original_file": orig_file,
                }
            )

            # Copy the actual file instead of creating a symlink
            src_path = os.path.join(model_data["model_dir"], orig_file)
            dst_path = os.path.join(output_dir, new_filename)

            if os.path.exists(src_path) and not os.path.exists(dst_path):
                try:
                    # copy2 preserves metadata
                    shutil.copy2(src_path, dst_path)
                    copied_files += 1
                except Exception as e:
                    print(f"\nWarning: Could not copy {src_path}: {e}")

            file_counter += 1

    print(f"Copied {copied_files} SED files to combined directory")

    # Save lookup table
    lookup_df = pd.DataFrame(lookup_data)
    lookup_file = os.path.join(output_dir, "lookup_table.csv")

    with open(lookup_file, "w") as f:
        f.write("#file_name, teff, logg, meta, source_model, original_file\n")
        lookup_df.to_csv(f, index=False, header=False)

    print(f"Saved combined lookup table to: {lookup_file}")
    print(f"Total models in combined set: {len(lookup_df)}")

    # Calculate and display disk usage
    total_size = 0
    for f in os.listdir(output_dir):
        if f.endswith(".txt"):
            total_size += os.path.getsize(os.path.join(output_dir, f))

    print(f"Total disk space used: {total_size / (1024**3):.2f} GB")

    return lookup_df



# After imports
def detect_model_scale(model_data):
    """Detect if model needs correction factor."""
    
    model_name = os.path.basename(model_data['model_dir'])
    
    n = min(15, len(model_data['files']))
    indices = np.linspace(0, len(model_data['files'])-1, n, dtype=int)
    
    ratios = []
    for idx in indices:
        fp = os.path.join(model_data['model_dir'], model_data['files'][idx])
        teff = model_data['teff'][idx]
        
        try:
            wl, flux = load_sed(fp)
            good = np.isfinite(flux) & (flux > 0) & (wl > 0)
            if np.sum(good) < 10: continue
            
            wl, flux = wl[good], flux[good]
            order = np.argsort(wl)
            
            Fbol = simps(flux[order], wl[order])
            Fbol_expected = SIGMA * teff**4
            
            if Fbol > 0 and Fbol_expected > 0:
                ratios.append(Fbol / Fbol_expected)
        except: 
            pass
    
    if not ratios: 
        print(f"  [{model_name:20s}] WARNING: No valid samples, scale = 1.0")
        return 1.0
    
    median = np.median(ratios)
    
    # Detect correction needed with TIGHT threshold around 1.0
    if 0.8 < median < 1.25:
        # Close to 1.0 - correct units
        print(f"  [{model_name:20s}] ∫Fλ/σT⁴ = {median:.3f} → scale = 1.0 ✓")
        return 1.0
    
    elif 0.25 < median < 0.8:
        # Check if it's per-steradian (ratio ≈ 1/π ≈ 0.318)
        test = median * np.pi
        if 0.8 < test < 1.25:
            print(f"  [{model_name:20s}] ∫Fλ/σT⁴ = {median:.3f} → scale = π (per-steradian, test={test:.3f})")
            return np.pi
        else:
            # In range but not per-steradian - might need other correction
            print(f"  [{model_name:20s}] ∫Fλ/σT⁴ = {median:.3f} → scale = 1.0 (unclear)")
            return 1.0
    
    elif median < 0.25:
        # Very low - probably missing scale factor
        for factor in [10, 100, 1000, 10000]:
            test = median * factor
            if 0.8 < test < 1.25:
                print(f"  [{model_name:20s}] ∫Fλ/σT⁴ = {median:.3e} → scale = {factor} (too low, test={test:.3f})")
                return factor
        print(f"  [{model_name:20s}] ∫Fλ/σT⁴ = {median:.3e} → scale = 10000 (way too low)")
        return 10000
    elif median > 1.25:
        # Flux too high - try dividing
        for factor in [0.1, 0.01, 0.001, 0.0001]:
            test = median * factor
            if 0.8 < test < 1.25:
                print(f"  [{model_name:20s}] ∫Fλ/σT⁴ = {median:.3e} → scale = {factor} (too high, test={test:.3f})")
                return factor
        print(f"  [{model_name:20s}] ∫Fλ/σT⁴ = {median:.3e} → scale = 0.0001 (way too high)")
        return 0.0001
    
    print(f"  [{model_name:20s}] ∫Fλ/σT⁴ = {median:.3f} → scale = 1.0 (default)")
    return 1.0








def visualize_parameter_space(
    teff_grid, logg_grid, meta_grid, source_map, all_models_data, output_dir
):
    """Create visualizations of the parameter space coverage."""
    print("\nCreating parameter space visualizations...")

    # Get model names
    model_names = [os.path.basename(data["model_dir"]) for data in all_models_data]

    # Create color map for models
    cmap = plt.cm.tab10 if len(model_names) <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, len(model_names)))

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(
        2, 3,
        wspace=0.20,
        hspace=0.20,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
    )

    # ---------- Top row ----------
    # 1) 3D scatter
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")

    for model_idx, model_name in enumerate(model_names):
        mask = source_map == model_idx
        if np.any(mask):
            indices = np.where(mask)
            ax1.scatter(
                teff_grid[indices[0]],
                logg_grid[indices[1]],
                meta_grid[indices[2]],
                c=[colors[model_idx]],
                label=model_name,
                alpha=0.6,
                s=10,
            )

    ax1.set_xlabel("Teff (K)")
    ax1.set_ylabel("log g")
    ax1.set_zlabel("[M/H]")
    ax1.set_title("3D Coverage")
    #ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, frameon=False)

    # 2) Normalisation check (span top middle + top right)
    ax2 = fig.add_subplot(gs[0, 1:3])
    target = (5777, 4.44, 0.0)  # solar-like reference

    for idx, model_name in enumerate(model_names):
        data = all_models_data[idx]

        valid_mask = (
            np.isfinite(data["teff"])
            & np.isfinite(data["logg"])
            & np.isfinite(data["meta"])
        )
        if not valid_mask.any():
            continue

        valid_teff = np.array(data["teff"])[valid_mask]
        valid_logg = np.array(data["logg"])[valid_mask]
        valid_meta = np.array(data["meta"])[valid_mask]
        valid_files = np.array(data["files"])[valid_mask]

        dist = (
            ((valid_teff - target[0]) / 1000) ** 2
            + (valid_logg - target[1]) ** 2
            + (valid_meta - target[2]) ** 2
        )

        if len(dist) == 0:
            continue

        j = np.argmin(dist)
        fpath = os.path.join(data["model_dir"], valid_files[j])

        # Quick guard against XML/FITS/binary junk
        if not fpath.lower().endswith((".txt", ".dat", ".sed")):
            continue

        try:
            if not os.path.exists(fpath):
                continue

            with open(fpath, "rb") as fh:
                first = fh.read(256).lstrip()[:1]
            if first == b"<":
                continue

            wl, fl = prepare_sed(fpath, valid_teff[j])
            mask = (wl > 3000) & (wl < 10000)
            ax2.plot(
                wl[mask],
                wl[mask] ** 2 * fl[mask],
                label=model_name,
                color=colors[idx],
                alpha=0.85,
                linewidth=1.2,
            )

        except Exception as e:
            print(f"  ⚠ Skipping {model_name}: {e}")
            continue

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Wavelength (Å)")
    ax2.set_ylabel(r"$\lambda^2 F_\lambda$ (arb. units)")
    ax2.set_title("Normalisation Check (closest-to-solar SED per model)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8, frameon=False, ncol=2)

    # ---------- Bottom row ----------
    from matplotlib.colors import BoundaryNorm

    def _count_unique_models(arr):
        arr = arr[arr >= 0]
        return len(np.unique(arr))

    # Quantized colormap + shared norm for all bottom panels (0..N unique models)
    n_levels = len(model_names) + 1
    dens_cmap = plt.get_cmap("YlOrRd", n_levels)
    bounds = np.arange(-0.5, n_levels + 0.5, 1.0)
    dens_norm = BoundaryNorm(bounds, dens_cmap.N)

    # 3) Teff vs log g (marginalized over [M/H])
    ax4 = fig.add_subplot(gs[1, 0])
    dens_tg = np.zeros((len(teff_grid), len(logg_grid)))
    for i in range(len(teff_grid)):
        for j in range(len(logg_grid)):
            dens_tg[i, j] = _count_unique_models(source_map[i, j, :])

    im4 = ax4.imshow(
        dens_tg.T,
        origin="lower",
        aspect="auto",
        extent=[teff_grid.min(), teff_grid.max(), logg_grid.min(), logg_grid.max()],
        cmap=dens_cmap,
        norm=dens_norm,
    )
    ax4.set_xlabel("Teff (K)")
    ax4.set_ylabel("log g")
    ax4.set_title("Model Density: Teff vs log g")

    # 4) Teff vs [M/H] (marginalized over log g)
    ax5 = fig.add_subplot(gs[1, 1])
    dens_tm = np.zeros((len(teff_grid), len(meta_grid)))
    for i in range(len(teff_grid)):
        for k in range(len(meta_grid)):
            dens_tm[i, k] = _count_unique_models(source_map[i, :, k])

    im5 = ax5.imshow(
        dens_tm.T,
        origin="lower",
        aspect="auto",
        extent=[teff_grid.min(), teff_grid.max(), meta_grid.min(), meta_grid.max()],
        cmap=dens_cmap,
        norm=dens_norm,
    )
    ax5.set_xlabel("Teff (K)")
    ax5.set_ylabel("[M/H]")
    ax5.set_title("Model Density: Teff vs [M/H]")

    # 5) log g vs [M/H] (marginalized over Teff)
    ax6 = fig.add_subplot(gs[1, 2])
    dens_gm = np.zeros((len(logg_grid), len(meta_grid)))
    for j in range(len(logg_grid)):
        for k in range(len(meta_grid)):
            dens_gm[j, k] = _count_unique_models(source_map[:, j, k])

    im6 = ax6.imshow(
        dens_gm.T,
        origin="lower",
        aspect="auto",
        extent=[logg_grid.min(), logg_grid.max(), meta_grid.min(), meta_grid.max()],
        cmap=dens_cmap,
        norm=dens_norm,
    )
    ax6.set_xlabel("log g")
    ax6.set_ylabel("[M/H]")
    ax6.set_title("Model Density: log g vs [M/H]")

    # Single shared colorbar for bottom row
    fig.colorbar(
        im6,
        ax=[ax4, ax5, ax6],
        fraction=0.046,
        pad=0.04,
        ticks=np.arange(0, n_levels, 1),
    )

    plot_file = os.path.join(output_dir, "parameter_space_visualization.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {plot_file}")

    # Print summary statistics (unchanged)
    print("\n" + "=" * 60)
    print("COMBINED MODEL STATISTICS")
    print("=" * 60)
    print(f"Total number of source models: {len(all_models_data)}")
    print(
        f"Total grid points: {len(teff_grid)} × {len(logg_grid)} × {len(meta_grid)} = "
        f"{len(teff_grid) * len(logg_grid) * len(meta_grid):,}"
    )
    print("\nParameter ranges:")
    print(f"  Teff: {teff_grid.min():.0f} - {teff_grid.max():.0f} K")
    print(f"  log g: {logg_grid.min():.2f} - {logg_grid.max():.2f}")
    print(f"  [M/H]: {meta_grid.min():.2f} - {meta_grid.max():.2f}")

    print("\nPer-model contributions:")
    for model_idx, model_name in enumerate(model_names):
        n_points = np.sum(source_map == model_idx)
        pct = 100 * n_points / source_map.size
        print(f"  {model_name}: {n_points:,} grid points ({pct:.1f}%)")