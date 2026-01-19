#!/usr/bin/env python3
"""
spectra_cleaner.py — Standardize stellar spectra for MESA

Ensures all SEDs have:
  - Wavelength in Angstroms (Å)
  - Flux in erg/cm²/s/Å (F_lambda)
  - Consistent metadata tags alongside Teff/logg/[M/H]

KEY PRINCIPLE: A catalog has ONE unit system, not per-file units.

Processing:
  1. Sample 10% of catalog to determine units (consensus)
  2. Apply those units uniformly to ALL files
  3. Convert wavelength → Angstroms, flux → F_lambda
  4. Validate and write with standardized header

The tag `# units_standardized = True` prevents reprocessing.
"""

from __future__ import annotations

import glob
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ============================================================================
# CONSTANTS
# ============================================================================

SPEED_OF_LIGHT_ANGSTROM = 2.99792458e18  # c in Å/s

# Wavelength conversion factors → Angstroms
WAVELENGTH_FACTORS = {
    'angstrom': 1.0,
    'nm': 10.0,
    'um': 1e4,
    'cm': 1e8,
    'm': 1e10,
}

# Standard header format
STANDARD_HEADER_TEMPLATE = """\
# source = {source}
# teff = {teff}
# logg = {logg}
# metallicity = {metallicity}
# wavelength_unit = Angstrom
# flux_unit = erg/cm2/s/A
# units_standardized = True
# original_wavelength_unit = {orig_wl_unit}
# original_flux_unit = {orig_flux_unit}
# conversion_confidence = {confidence}
"""


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class UnitInfo:
    """Detected unit information."""
    wavelength_unit: str      # 'angstrom', 'nm', 'um', 'cm', 'm', 'index'
    wavelength_factor: float  # multiply raw values by this to get Angstroms
    flux_type: str            # 'flam', 'fnu', 'normalized'
    flux_factor: float        # scaling factor before F_nu→F_lambda conversion
    confidence: str           # 'high', 'medium', 'low', 'catalog'
    detection_source: str     # 'header', 'range', 'catalog_consensus'


@dataclass 
class SpectrumMeta:
    """Parsed spectrum metadata."""
    source: str = "unknown"
    teff: float = float('nan')
    logg: float = float('nan')
    metallicity: float = float('nan')
    spec_group: str = ""           # For MSG HDF5 recovery
    units_standardized: bool = False
    original_header: str = ""      # Preserve non-standard header lines
    extra: Dict[str, str] = None   # Any other key=value pairs
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


# ============================================================================
# HEADER PARSING
# ============================================================================

def parse_header(filepath: str) -> Tuple[SpectrumMeta, str]:
    """
    Parse header metadata from a spectrum file.
    
    Returns: (SpectrumMeta, raw_header_text)
    """
    meta = SpectrumMeta()
    header_lines = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip().startswith('#'):
                    break
                header_lines.append(line)
                
                # Parse key=value pairs
                if '=' in line:
                    key_part, _, value_part = line.partition('=')
                    key = key_part.strip('#').strip().lower()
                    value = value_part.strip()
                    
                    # Extract numeric part if present
                    num_match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value)
                    num_value = num_match.group(0) if num_match else value
                    
                    if key in ('teff', 't_eff', 'effective_temperature'):
                        try:
                            meta.teff = float(num_value)
                        except ValueError:
                            pass
                    elif key in ('logg', 'log_g', 'log(g)', 'surface_gravity'):
                        try:
                            meta.logg = float(num_value)
                        except ValueError:
                            pass
                    elif key in ('metallicity', 'meta', 'feh', '[fe/h]', '[m/h]', 'm/h'):
                        try:
                            meta.metallicity = float(num_value)
                        except ValueError:
                            pass
                    elif key == 'source':
                        meta.source = value
                    elif key == 'spec_group':
                        meta.spec_group = value
                    elif key == 'units_standardized':
                        meta.units_standardized = value.lower() in ('true', '1', 'yes')
                    else:
                        # Store other metadata
                        meta.extra[key] = value
    
    except Exception:
        pass
    
    return meta, ''.join(header_lines)


def read_spectrum_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read wavelength and flux columns from a spectrum file."""
    wl, flux = [], []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                
                parts = s.replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        wl.append(float(parts[0]))
                        flux.append(float(parts[1]))
                    except ValueError:
                        continue
    except Exception:
        pass
    
    return np.asarray(wl, dtype=float), np.asarray(flux, dtype=float)


# ============================================================================
# UNIT DETECTION FROM HEADER
# ============================================================================

def detect_units_from_header(header_text: str) -> Dict[str, Optional[str]]:
    """
    Extract unit information from header text.
    
    Returns dict with 'wavelength_unit' and 'flux_unit' keys.
    """
    result = {'wavelength_unit': None, 'flux_unit': None}
    header_lower = header_text.lower()
    
    # --- Wavelength unit detection ---
    wl_patterns = [
        (r'wavelength[_\s]*unit\s*[=:]\s*(\S+)', None),
        (r'wavelength\s*\(\s*(\S+)\s*\)', None),
        (r'wave\s*\(\s*(\S+)\s*\)', None),
        (r'lambda\s*\(\s*(\S+)\s*\)', None),
        (r'columns?\s*[=:].*wavelength[_\s]*\(?([aå]ngstrom|nm|um|µm|micron)\)?', None),
    ]
    
    for pattern, default in wl_patterns:
        match = re.search(pattern, header_lower)
        if match:
            unit_str = match.group(1).lower().strip('()[]')
            if unit_str in ('angstrom', 'ang', 'a', 'aa', 'å'):
                result['wavelength_unit'] = 'angstrom'
            elif unit_str in ('nm', 'nanometer', 'nanometre'):
                result['wavelength_unit'] = 'nm'
            elif unit_str in ('um', 'µm', 'micron', 'micrometer'):
                result['wavelength_unit'] = 'um'
            elif unit_str in ('cm', 'centimeter'):
                result['wavelength_unit'] = 'cm'
            elif unit_str in ('m', 'meter'):
                result['wavelength_unit'] = 'm'
            break
    
    # --- Flux unit detection ---
    flux_patterns = [
        # F_lambda patterns
        (r'erg\s*/?\s*s\s*/?\s*cm\s*\^?2?\s*/?\s*(a|å|angstrom)', 'flam'),
        (r'erg\s*/?\s*cm\s*\^?2?\s*/?\s*s\s*/?\s*(a|å|angstrom)', 'flam'),
        (r'f_?lambda', 'flam'),
        (r'flam\b', 'flam'),
        # F_nu patterns
        (r'erg\s*/?\s*s\s*/?\s*cm\s*\^?2?\s*/?\s*hz', 'fnu'),
        (r'erg\s*/?\s*cm\s*\^?2?\s*/?\s*s\s*/?\s*hz', 'fnu'),
        (r'f_?nu', 'fnu'),
        (r'fnu\b', 'fnu'),
        # Jansky
        (r'\bjansky\b|\bjy\b', 'fnu_jy'),
        # SI units
        (r'w\s*/?\s*m\s*\^?2?\s*/?\s*(a|å|angstrom|nm|um)', 'flam_si'),
        # Normalized
        (r'normalized|norm\b', 'normalized'),
    ]
    
    for pattern, flux_type in flux_patterns:
        if re.search(pattern, header_lower):
            result['flux_unit'] = flux_type
            break
    
    return result


# ============================================================================
# UNIT DETECTION FROM DATA
# ============================================================================

def detect_wavelength_unit_from_range(wl: np.ndarray) -> Tuple[str, float, str]:
    """
    Infer wavelength unit from data range.
    
    Returns: (unit_name, conversion_factor, confidence)
    """
    if wl.size < 2:
        return 'angstrom', 1.0, 'low'
    
    wl_min, wl_max = wl.min(), wl.max()
    wl_span = wl_max / max(wl_min, 1e-10)
    
    # Index grid detection: values are 0, 1, 2, ... N-1
    if wl_min < 1 and np.allclose(wl, np.arange(wl.size), atol=0.01):
        return 'index', 0.0, 'index_grid'
    
    # Near-index grid: starts at 0 or 1, uniform step ~1
    if wl_min < 10 and wl_max < len(wl) * 2:
        steps = np.diff(wl)
        if len(steps) > 0 and np.allclose(steps, 1.0, rtol=0.1):
            return 'index', 0.0, 'index_grid'
    
    # Angstroms: typical optical/IR range 100 - 1,000,000 Å
    if 500 <= wl_max <= 1e6 and wl_min >= 10:
        conf = 'high' if wl_min > 500 else 'medium'
        return 'angstrom', 1.0, conf
    
    # Nanometers: typical range 50 - 100,000 nm
    if 50 <= wl_max <= 1e5 and wl_min >= 5 and wl_max < 50000:
        return 'nm', 10.0, 'medium'
    
    # Micrometers: typical range 0.1 - 1000 µm
    if 0.05 <= wl_max <= 1000 and wl_min >= 0.01 and wl_max < 100:
        return 'um', 1e4, 'medium'
    
    # Centimeters: rare, radio wavelengths
    if 1e-5 <= wl_max <= 10 and wl_span < 1000:
        return 'cm', 1e8, 'low'
    
    # Meters: very rare
    if 1e-8 <= wl_max <= 0.1 and wl_span < 1000:
        return 'm', 1e10, 'low'
    
    # Ambiguous range 10-500: could be nm or Å
    if 10 <= wl_max <= 500:
        if wl_max > 100 and wl_min > 10:
            return 'nm', 10.0, 'low'
    
    # Default: assume Angstroms
    return 'angstrom', 1.0, 'low'


def detect_flux_unit_from_range(flux: np.ndarray) -> Tuple[str, float, str]:
    """
    Infer flux unit from data magnitude.
    
    Returns: (flux_type, scaling_factor, confidence)
    """
    flux_valid = flux[np.isfinite(flux) & (flux > 0)]
    if flux_valid.size == 0:
        return 'flam', 1.0, 'low'
    
    median = np.median(flux_valid)
    log_median = np.log10(median)
    
    # F_lambda in erg/cm²/s/Å: typical range 1e-17 to 1e-5 for stellar surfaces
    if -17 < log_median < -4:
        conf = 'high' if -15 < log_median < -6 else 'medium'
        return 'flam', 1.0, conf
    
    # F_nu in erg/cm²/s/Hz: typically 1e-30 to 1e-18
    if -32 < log_median < -18:
        return 'fnu', 1.0, 'low'
    
    # Jansky (1 Jy = 1e-23 erg/cm²/s/Hz): typically 1e-6 to 1e6 Jy
    if -6 < log_median < 6:
        if -1 < log_median < 2:
            return 'normalized', 1.0, 'low'
        return 'fnu', 1e-23, 'low'  # Assume Jansky
    
    # Very small values: possibly SI units (W/m²/Å)
    if log_median < -17:
        return 'flam', 1e3, 'low'  # W/m² to erg/cm²/s
    
    # Very large or normalized
    if log_median > 0:
        return 'normalized', 1.0, 'low'
    
    return 'flam', 1.0, 'low'


def detect_units_single_file(filepath: str) -> Optional[UnitInfo]:
    """
    Detect units from a single file (for sampling).
    
    Returns UnitInfo or None if file is invalid/already standardized.
    """
    meta, header_text = parse_header(filepath)
    
    # Skip if already standardized
    if meta.units_standardized:
        return None
    
    wl, flux = read_spectrum_data(filepath)
    
    if wl.size < 2 or flux.size < 2:
        return None
    
    # Clean data
    valid = np.isfinite(wl) & np.isfinite(flux) & (wl > 0)
    wl, flux = wl[valid], flux[valid]
    
    if wl.size < 2:
        return None
    
    # Try header first
    header_units = detect_units_from_header(header_text)
    
    # Wavelength
    if header_units['wavelength_unit']:
        wl_unit = header_units['wavelength_unit']
        wl_factor = WAVELENGTH_FACTORS.get(wl_unit, 1.0)
        wl_conf = 'high'
        wl_source = 'header'
    else:
        wl_unit, wl_factor, wl_conf = detect_wavelength_unit_from_range(wl)
        wl_source = 'range'
    
    # Flux
    if header_units['flux_unit']:
        flux_type = header_units['flux_unit']
        flux_factor = 1.0
        if flux_type == 'fnu_jy':
            flux_type = 'fnu'
            flux_factor = 1e-23
        elif flux_type == 'flam_si':
            flux_type = 'flam'
            flux_factor = 1e3
        flux_conf = 'high'
    else:
        flux_type, flux_factor, flux_conf = detect_flux_unit_from_range(flux)
    
    # Combined confidence
    if wl_conf == 'high' and flux_conf == 'high':
        confidence = 'high'
    elif wl_conf in ('high', 'medium') and flux_conf in ('high', 'medium'):
        confidence = 'medium'
    elif wl_conf == 'index_grid':
        confidence = 'index_grid'
    else:
        confidence = 'low'
    
    return UnitInfo(
        wavelength_unit=wl_unit,
        wavelength_factor=wl_factor,
        flux_type=flux_type,
        flux_factor=flux_factor,
        confidence=confidence,
        detection_source=wl_source
    )


# ============================================================================
# CATALOG-LEVEL UNIT DETECTION (10% SAMPLE)
# ============================================================================

def detect_catalog_units(txt_files: List[str], sample_fraction: float = 0.10) -> Tuple[Optional[UnitInfo], Dict]:
    """
    Detect units from a sample of catalog files.
    
    Samples 10% of files (min 1, max 100), determines consensus units.
    
    Returns: (UnitInfo or None, stats_dict)
    """
    if not txt_files:
        return None, {'error': 'no files'}
    
    # Calculate sample size: 10%, min 1, max 100
    n_total = len(txt_files)
    n_sample = max(1, min(1000, int(n_total * sample_fraction)))
    
    # Random sample for better representation
    if n_sample >= n_total:
        sample_files = txt_files
    else:
        sample_files = random.sample(txt_files, n_sample)
    
    # Detect units from each sample file
    detections = []
    skipped_standardized = 0
    skipped_invalid = 0
    index_grids = 0
    
    for filepath in sample_files:
        unit_info = detect_units_single_file(filepath)
        
        if unit_info is None:
            # Check why it was skipped
            meta, _ = parse_header(filepath)
            if meta.units_standardized:
                skipped_standardized += 1
            else:
                skipped_invalid += 1
            continue
        
        if unit_info.confidence == 'index_grid':
            index_grids += 1
            continue
        
        detections.append(unit_info)
    
    stats = {
        'total_files': n_total,
        'sample_size': n_sample,
        'valid_detections': len(detections),
        'skipped_standardized': skipped_standardized,
        'skipped_invalid': skipped_invalid,
        'index_grids': index_grids,
    }
    
    # If all sampled files are already standardized, skip the catalog
    if skipped_standardized == n_sample:
        stats['status'] = 'already_standardized'
        return None, stats
    
    # If no valid detections, can't determine units
    if not detections:
        stats['status'] = 'no_valid_detections'
        return None, stats
    
    # Find consensus: most common wavelength unit and flux type
    wl_units = Counter(d.wavelength_unit for d in detections)
    flux_types = Counter(d.flux_type for d in detections)
    
    consensus_wl = wl_units.most_common(1)[0][0]
    consensus_flux = flux_types.most_common(1)[0][0]
    
    # Get representative UnitInfo with consensus values
    # Use the first detection that matches consensus for factors
    representative = None
    for d in detections:
        if d.wavelength_unit == consensus_wl and d.flux_type == consensus_flux:
            representative = d
            break
    
    if representative is None:
        # Fallback: use first detection with consensus wavelength
        for d in detections:
            if d.wavelength_unit == consensus_wl:
                representative = d
                break
    
    if representative is None:
        representative = detections[0]
    
    # Check consensus strength
    wl_agreement = wl_units[consensus_wl] / len(detections)
    flux_agreement = flux_types[consensus_flux] / len(detections)
    
    stats['wavelength_consensus'] = consensus_wl
    stats['wavelength_agreement'] = f"{wl_agreement:.0%}"
    stats['flux_consensus'] = consensus_flux
    stats['flux_agreement'] = f"{flux_agreement:.0%}"
    stats['status'] = 'success'
    
    # Determine confidence based on agreement
    if wl_agreement >= 0.9 and flux_agreement >= 0.9:
        confidence = 'catalog_high'
    elif wl_agreement >= 0.7 and flux_agreement >= 0.7:
        confidence = 'catalog_medium'
    else:
        confidence = 'catalog_low'
    
    catalog_units = UnitInfo(
        wavelength_unit=consensus_wl,
        wavelength_factor=WAVELENGTH_FACTORS.get(consensus_wl, 1.0),
        flux_type=consensus_flux,
        flux_factor=representative.flux_factor,
        confidence=confidence,
        detection_source='catalog_consensus'
    )
    
    return catalog_units, stats


# ============================================================================
# UNIT CONVERSION
# ============================================================================

def convert_to_standard_units(
    wl: np.ndarray,
    flux: np.ndarray,
    unit_info: UnitInfo
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert wavelength to Angstroms and flux to F_lambda (erg/cm²/s/Å).
    """
    # Convert wavelength to Angstroms
    wl_ang = wl * unit_info.wavelength_factor
    
    # Convert flux
    flux_scaled = flux * unit_info.flux_factor
    
    if unit_info.flux_type in ('fnu', 'fnu_jy'):
        # F_nu (erg/cm²/s/Hz) → F_lambda (erg/cm²/s/Å)
        # F_lambda = F_nu * c / λ²
        with np.errstate(divide='ignore', invalid='ignore'):
            flux_flam = flux_scaled * SPEED_OF_LIGHT_ANGSTROM / (wl_ang ** 2)
            flux_flam = np.where(np.isfinite(flux_flam), flux_flam, 0.0)
    elif unit_info.flux_type == 'normalized':
        # Keep as-is, no physical conversion possible
        flux_flam = flux_scaled
    else:
        # Already F_lambda (or close enough)
        flux_flam = flux_scaled
    
    return wl_ang, flux_flam


# ============================================================================
# VALIDATION
# ============================================================================

def validate_converted_spectrum(wl: np.ndarray, flux: np.ndarray) -> Tuple[bool, str]:
    """
    Validate converted spectrum has sensible values.
    """
    if wl.size == 0 or flux.size == 0:
        return False, "empty arrays"
    
    if not np.all(np.isfinite(wl)):
        return False, "non-finite wavelengths"
    
    if wl.min() <= 0:
        return False, f"non-positive wavelength: {wl.min()}"
    
    # Wavelength range check (1 Å to 10 million Å = 1mm)
    if wl.min() < 1 or wl.max() > 1e7:
        return False, f"wavelength out of range: {wl.min():.1f}-{wl.max():.1f}"
    
    # Flux can have some issues but shouldn't be all bad
    flux_valid = np.isfinite(flux)
    if np.sum(flux_valid) < len(flux) * 0.5:
        return False, f"too many non-finite flux values: {np.sum(~flux_valid)}/{len(flux)}"
    
    return True, "ok"


# ============================================================================
# FILE WRITING
# ============================================================================

def write_standardized_spectrum(
    filepath: str,
    wl: np.ndarray,
    flux: np.ndarray,
    meta: SpectrumMeta,
    unit_info: UnitInfo,
    backup: bool = True
) -> None:
    """Write standardized spectrum with proper header."""
    
    # Create backup if requested
    if backup:
        backup_path = filepath + '.bak'
        if not os.path.exists(backup_path):
            try:
                os.rename(filepath, backup_path)
            except Exception:
                pass
    
    # Build header
    header = STANDARD_HEADER_TEMPLATE.format(
        source=meta.source,
        teff=meta.teff if np.isfinite(meta.teff) else 'nan',
        logg=meta.logg if np.isfinite(meta.logg) else 'nan',
        metallicity=meta.metallicity if np.isfinite(meta.metallicity) else 'nan',
        orig_wl_unit=unit_info.wavelength_unit,
        orig_flux_unit=unit_info.flux_type,
        confidence=unit_info.confidence,
    )
    
    # Add spec_group if present (for HDF5 recovery)
    if meta.spec_group:
        header += f"# spec_group = {meta.spec_group}\n"
    
    # Add extra metadata
    for key, value in meta.extra.items():
        if key not in ('wavelength_unit', 'flux_unit', 'units_standardized',
                       'original_wavelength_unit', 'original_flux_unit', 'conversion_confidence'):
            header += f"# {key} = {value}\n"
    
    # Write file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header)
        for w, fl in zip(wl, flux):
            f.write(f"{w:.6f} {fl:.8e}\n")


# ============================================================================
# HDF5 WAVELENGTH RECOVERY (for MSG index grids)
# ============================================================================

WAVE_KEYS = ('lambda', 'wavelength', 'wave', 'wl', 'wavelength_A', 'x')

def recover_wavelengths_from_hdf5(
    model_dir: str,
    spec_group: str,
    expected_len: int
) -> Optional[np.ndarray]:
    """
    Attempt to recover wavelength array from HDF5 file for index grids.
    """
    if not HAS_H5PY:
        return None
    
    h5_files = glob.glob(os.path.join(model_dir, "*.h5"))
    
    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, 'r') as f:
                if spec_group not in f:
                    continue
                
                group = f[spec_group]
                
                # Direct wavelength dataset
                for key in WAVE_KEYS:
                    if key in group:
                        wl = np.array(group[key]).astype(float).ravel()
                        if wl.size >= expected_len and np.all(np.diff(wl[:expected_len]) > 0):
                            return wl[:expected_len]
                
                # Check range/x structure
                if 'range' in group and isinstance(group['range'], h5py.Group):
                    rg = group['range']
                    if 'x' in rg:
                        wl = np.array(rg['x']).astype(float).ravel()
                        if wl.size >= expected_len and np.all(np.diff(wl[:expected_len]) > 0):
                            return wl[:expected_len]
                    
                    # Concatenated segments
                    segs = []
                    seg_names = sorted([k for k in rg.keys() if 'range' in k.lower()],
                                       key=lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else 0)
                    
                    for sn in seg_names:
                        if not isinstance(rg[sn], h5py.Group):
                            continue
                        sg = rg[sn]
                        
                        for key in WAVE_KEYS:
                            if key in sg:
                                segs.append(np.array(sg[key]).astype(float).ravel())
                                break
                    
                    if segs:
                        wl = np.concatenate(segs)
                        if wl.size >= expected_len:
                            return wl[:expected_len]
        
        except Exception:
            continue
    
    return None


def is_index_grid(wl: np.ndarray) -> bool:
    """Detect if wavelength array is actually an index grid (0, 1, 2, ...)."""
    if wl.size < 4:
        return False
    
    # Must start near 0 or 1
    if wl[0] > 2:
        return False
    
    # Check if it matches 0..N-1 or 1..N
    if np.allclose(wl, np.arange(wl.size), atol=0.01):
        return True
    if np.allclose(wl, np.arange(1, wl.size + 1), atol=0.01):
        return True
    
    # Check uniform step of ~1
    steps = np.diff(wl)
    if np.allclose(steps, 1.0, atol=0.01):
        return True
    
    return False


# ============================================================================
# MAIN CLEANING LOGIC
# ============================================================================

def clean_spectrum_file(
    filepath: str,
    catalog_units: Optional[UnitInfo] = None,
    try_h5_recovery: bool = True,
    backup: bool = True
) -> Tuple[str, str]:
    """
    Clean and standardize a single spectrum file.
    
    Args:
        filepath: Path to spectrum file
        catalog_units: Pre-determined units for the catalog (from 10% sample).
                       If None, detects units per-file (legacy behavior).
        try_h5_recovery: Attempt HDF5 wavelength recovery for index grids
        backup: Create .bak backup file
    
    Returns: (status, detail)
        status: 'skipped_already', 'skipped_index', 'skipped_invalid', 
                'converted', 'recovered', 'error'
    """
    # Parse header
    meta, header_text = parse_header(filepath)
    
    # Skip if already standardized
    if meta.units_standardized:
        return 'skipped_already', 'already standardized'
    
    # Read data
    wl, flux = read_spectrum_data(filepath)
    
    if wl.size == 0 or flux.size == 0:
        return 'skipped_invalid', 'empty data'
    
    # Clean: remove NaN/inf and non-positive wavelengths
    valid = np.isfinite(wl) & np.isfinite(flux) & (wl > 0)
    wl, flux = wl[valid], flux[valid]
    
    if wl.size < 2:
        return 'skipped_invalid', 'no valid data points'
    
    # Sort by wavelength
    order = np.argsort(wl)
    wl, flux = wl[order], flux[order]
    
    # Remove duplicates (keep first)
    unique_mask = np.concatenate([[True], np.diff(wl) > 0])
    wl, flux = wl[unique_mask], flux[unique_mask]
    
    # Check for index grid
    recovered_wl = None
    if is_index_grid(wl):
        if try_h5_recovery and meta.spec_group and 'msg' in meta.source.lower():
            model_dir = os.path.dirname(filepath)
            recovered_wl = recover_wavelengths_from_hdf5(model_dir, meta.spec_group, len(flux))
        
        if recovered_wl is None:
            return 'skipped_index', 'index grid without HDF5 recovery'
        
        wl = recovered_wl
        status = 'recovered'
    else:
        status = 'converted'
    
    # Use catalog units if provided, otherwise detect per-file
    if catalog_units is not None:
        unit_info = catalog_units
    else:
        unit_info = detect_units_single_file(filepath)
        if unit_info is None:
            # Fallback detection
            header_units = detect_units_from_header(header_text)
            wl_unit = header_units.get('wavelength_unit') or 'angstrom'
            flux_type = header_units.get('flux_unit') or 'flam'
            unit_info = UnitInfo(
                wavelength_unit=wl_unit,
                wavelength_factor=WAVELENGTH_FACTORS.get(wl_unit, 1.0),
                flux_type=flux_type,
                flux_factor=1.0,
                confidence='low',
                detection_source='fallback'
            )
    
    try:
        wl_std, flux_std = convert_to_standard_units(wl, flux, unit_info)
    except Exception as e:
        return 'error', str(e)
    
    # Validate
    valid, msg = validate_converted_spectrum(wl_std, flux_std)
    if not valid:
        return 'skipped_invalid', f'validation failed: {msg}'
    
    # Write standardized file
    write_standardized_spectrum(filepath, wl_std, flux_std, meta, unit_info, backup=backup)
    
    detail = f"{wl_std[0]:.0f}-{wl_std[-1]:.0f}Å, {unit_info.wavelength_unit}→Å, conf={unit_info.confidence}"
    return status, detail


def rebuild_lookup_table(model_dir: str) -> str:
    """Rebuild lookup_table.csv from spectrum file headers."""
    txt_files = sorted(glob.glob(os.path.join(model_dir, "*.txt")))
    if not txt_files:
        return ""
    
    rows = []
    all_keys = {'file_name'}
    
    for filepath in txt_files:
        meta, _ = parse_header(filepath)
        row = {'file_name': os.path.basename(filepath)}
        
        if np.isfinite(meta.teff):
            row['teff'] = meta.teff
        if np.isfinite(meta.logg):
            row['logg'] = meta.logg
        if np.isfinite(meta.metallicity):
            row['metallicity'] = meta.metallicity
        
        row['source'] = meta.source
        row['units_standardized'] = str(meta.units_standardized)
        
        for k, v in meta.extra.items():
            row[k] = v
        
        rows.append(row)
        all_keys.update(row.keys())
    
    # Write CSV
    out_path = os.path.join(model_dir, "lookup_table.csv")
    
    # Ordered columns: file_name, teff, logg, metallicity first
    priority = ['file_name', 'teff', 'logg', 'metallicity', 'source', 'units_standardized']
    columns = [c for c in priority if c in all_keys]
    columns += sorted(k for k in all_keys if k not in priority)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('#' + ','.join(columns) + '\n')
        for row in rows:
            values = [str(row.get(c, '')) for c in columns]
            f.write(','.join(values) + '\n')
    
    return out_path


# ============================================================================
# PUBLIC API
# ============================================================================

def clean_model_dir(
    model_dir: str,
    try_h5_recovery: bool = True,
    backup: bool = True,
    rebuild_lookup: bool = True,
    sample_fraction: float = 0.10
) -> Dict:
    """
    Clean all spectra in a model directory.
    
    KEY: Detects units from 10% sample first, then applies uniformly.
    
    Args:
        model_dir: Directory containing *.txt spectrum files
        try_h5_recovery: Attempt HDF5 wavelength recovery for index grids
        backup: Create .bak files before modifying
        rebuild_lookup: Rebuild lookup_table.csv after cleaning
        sample_fraction: Fraction of files to sample for unit detection (default 10%)
    
    Returns:
        Summary dict with counts and file lists
    """
    txt_files = sorted(glob.glob(os.path.join(model_dir, "*.txt")))
    
    summary = {
        'total': len(txt_files),
        'converted': [],
        'recovered': [],
        'skipped_already': [],
        'skipped_index': [],
        'skipped_invalid': [],
        'error': [],
        'lookup_updated': False,
        'lookup_path': '',
        'catalog_units': None,
        'detection_stats': {},
    }
    
    if not txt_files:
        return summary
    
    # =========================================================
    # STEP 1: Detect catalog-wide units from 10% sample
    # =========================================================
    catalog_units, detection_stats = detect_catalog_units(txt_files, sample_fraction)
    summary['detection_stats'] = detection_stats
    
    if catalog_units:
        summary['catalog_units'] = {
            'wavelength': catalog_units.wavelength_unit,
            'flux': catalog_units.flux_type,
            'confidence': catalog_units.confidence,
        }
        print(f"[clean] Catalog units (from {detection_stats.get('sample_size', '?')} samples): "
              f"λ={catalog_units.wavelength_unit} ({detection_stats.get('wavelength_agreement', '?')}), "
              f"flux={catalog_units.flux_type} ({detection_stats.get('flux_agreement', '?')})")
    elif detection_stats.get('status') == 'already_standardized':
        print(f"[clean] Catalog already standardized (all {detection_stats.get('sample_size', '?')} samples)")
    else:
        print(f"[clean] Warning: Could not determine catalog units. Status: {detection_stats.get('status', 'unknown')}")
    
    # =========================================================
    # STEP 2: Apply catalog units to ALL files
    # =========================================================
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        try:
            status, detail = clean_spectrum_file(
                filepath,
                catalog_units=catalog_units,  # Pass catalog-wide units
                try_h5_recovery=try_h5_recovery,
                backup=backup
            )
            
            if status in summary:
                summary[status].append(filename)
            else:
                summary['error'].append(filename)
                
        except Exception as e:
            summary['error'].append(filename)
    
    # =========================================================
    # STEP 3: Rebuild lookup table
    # =========================================================
    if rebuild_lookup:
        lookup_path = rebuild_lookup_table(model_dir)
        if lookup_path:
            summary['lookup_updated'] = True
            summary['lookup_path'] = lookup_path
    
    return summary
