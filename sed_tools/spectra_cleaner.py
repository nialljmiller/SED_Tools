#!/usr/bin/env python3
"""
spectra_cleaner.py — Standardize stellar spectra for MESA

Ensures all SEDs have:
  - Wavelength in Angstroms (Å)
  - Flux in erg/cm²/s/Å (F_lambda)
  - Consistent metadata tags alongside Teff/logg/[M/H]

KEY PRINCIPLE: A spectral grid has ONE unit system, not per-file units.

Processing:
  1. Sample 10% of the spectral grid to determine units (consensus)
     A flux unit not declared in the headers is not guessed: cleaning stops
     and names the best guess, unless the caller gives the unit (flux_unit).
  2. Apply those units uniformly to ALL files
  3. Convert wavelength → Angstroms, flux → F_lambda
  4. Validate, renormalise a spectrum whose absolute flux scale is wrong
     (see _bolometric_renorm_ratio), and write with standardized header

The tag `# units_standardized = True` prevents reprocessing.
"""

from __future__ import annotations

import dataclasses
import glob
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from ._constants import C, SIGMA, planck_flam
from ._lookup_io import write_lookup_csv
from .header_parser import parse_header as _parse_header_dict
from .spectrum_io import read_text_spectrum

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

SPEED_OF_LIGHT_ANGSTROM = C * 1e8  # c in Å/s — from _constants

# A spectrum whose integrated flux is more than 3 times a blackbody's at its
# Teff, over the same wavelengths, has a wrong absolute flux scale (SVO's NextGen
# T >= 5000 K entries are about 100x too large). It is only checked when those
# wavelengths hold at least 5 % of sigma*T^4. Spectra far below the blackbody are
# left alone: cool giants' optical flux really is 2-10 % of it.
_RENORM_MAX_RATIO = 3.0
_RENORM_MIN_COVERAGE = 0.05

# Wavelength conversion factors → Angstroms
WAVELENGTH_FACTORS = {
    'angstrom': 1.0,
    'nm': 10.0,
    'um': 1e4,
    'cm': 1e8,
    'm': 1e10,
}

# The flux units a caller can give with flux_unit:
# name -> (flux type, factor applied first, description). The names are the
# ones detect_units_from_header gives a declared unit.
FLUX_UNITS = {
    'flam': ('flam', 1.0, 'erg/s/cm²/Å (F_λ)'),
    'fnu': ('fnu', 1.0, 'erg/s/cm²/Hz (F_ν)'),
    'fnu_jy': ('fnu', 1e-23, 'Jansky (F_ν)'),
    'flam_si': ('flam', 1e3, 'W/m²/Å'),
    'normalized': ('normalized', 1.0, 'arbitrary scale, left as it is'),
}

# How detect_flux_unit_from_range guesses. Model-atmosphere surface fluxes in
# erg/s/cm²/Å are 'flam', but it reads them as Jansky or as normalised.
FLUX_GUESS_WHY = "guessed from the size of the flux values, assuming flux received at Earth"

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
    flux_declared: bool = True  # False when the flux unit was guessed from the data


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

def _load_header(filepath: str) -> Tuple[SpectrumMeta, str]:
    """Internal adapter: calls header_parser and returns (SpectrumMeta, raw_header_text)."""
    d = _parse_header_dict(filepath)
    meta = SpectrumMeta(
        source=d.get("source", "unknown"),
        teff=d["teff"],
        logg=d["logg"],
        metallicity=d["metallicity"],
        spec_group=d.get("spec_group", ""),
        units_standardized=str(d.get("units_standardized", "")).lower() in ("true", "1", "yes"),
    )
    # everything not consumed above goes into extra
    skip = {"teff", "logg", "metallicity", "source", "spec_group",
            "units_standardized", "wavelength_unit", "flux_unit",
            "original_wavelength_unit", "original_flux_unit", "conversion_confidence"}
    meta.extra = {k: v for k, v in d.items() if k not in skip}

    # re-read raw header text for unit detection
    header_lines = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip().startswith("#"):
                break
            header_lines.append(line)

    return meta, "".join(header_lines)


def read_spectrum_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read wavelength and flux columns from a spectrum file."""
    return read_text_spectrum(filepath)


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
    meta, header_text = _load_header(filepath)
    
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
        detection_source=wl_source,
        flux_declared=header_units['flux_unit'] is not None,
    )


# ============================================================================
# SPECTRAL GRID UNIT DETECTION (SAMPLED)
# ============================================================================

def detect_catalog_units(txt_files: List[str], sample_fraction: float = 0.10) -> Tuple[Optional[UnitInfo], Dict]:
    """
    Detect units from a sample of model spectrum files.
    
    Samples ``sample_fraction`` of the files at random (min 1, max 1000) and
    takes the most common wavelength unit and flux type as the grid's units.
    
    Returns: (UnitInfo or None, stats_dict)
    """
    if not txt_files:
        return None, {'error': 'no files'}
    
    n_total = len(txt_files)
    n_sample = max(1, min(1000, int(n_total * sample_fraction)))
    
    # Random sample for better representation
    if n_sample >= n_total:
        sample_files = txt_files
    else:
        sample_files = random.sample(txt_files, n_sample)
    
    detections, skipped_standardized, skipped_invalid, index_grids = _detect_sample(sample_files)
    stats = {
        'total_files': n_total,
        'sample_size': n_sample,
        'valid_detections': len(detections),
        'skipped_standardized': skipped_standardized,
        'skipped_invalid': skipped_invalid,
        'index_grids': index_grids,
    }
    
    # If all sampled files are already standardized, skip the spectral grid
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
        flux_factor=_representative(detections, consensus_wl, consensus_flux).flux_factor,
        confidence=confidence,
        detection_source='catalog_consensus',
        flux_declared=all(d.flux_declared for d in detections),
    )
    
    return catalog_units, stats


def undeclared_flux_unit(txt_files: List[str], sample_fraction: float = 0.10) -> Optional[str]:
    """The best guess at the spectra's flux unit (a FLUX_UNITS name) when a sample of them
    does not declare it, as clean_model_dir would find; None when they do, or have nothing to clean.
    """
    catalog_units, _ = detect_catalog_units(txt_files, sample_fraction)
    if catalog_units is None or catalog_units.flux_declared:
        return None
    return _flux_unit_name(catalog_units)


def check_flux_unit(grid: str, txt_files: List[str], flux_unit: Optional[str] = None,
                    sample_fraction: float = 0.10) -> None:
    """Raise ValueError if clean_model_dir would refuse these spectra's flux unit before cleaning:
    flux_unit is not a FLUX_UNITS name, or is None while a sample of them does not declare theirs.
    """
    _check_flux_unit_name(flux_unit)
    if flux_unit is None:
        guess = undeclared_flux_unit(txt_files, sample_fraction)
        if guess is not None:
            raise ValueError(_undeclared_flux_unit_message(grid, guess))


def _check_flux_unit_name(flux_unit: Optional[str]) -> None:
    if flux_unit is not None and flux_unit not in FLUX_UNITS:
        raise ValueError(f"flux_unit={flux_unit!r} is not one of {', '.join(FLUX_UNITS)}")


def _undeclared_flux_unit_message(grid: str, guess: str) -> str:
    """The error for spectra that do not declare their flux unit: the best guess and how to pass one."""
    return (
        f"Flux unit of {grid} is not declared; best guess: '{guess}', {FLUX_UNITS[guess][2]} "
        f"({FLUX_GUESS_WHY}). Pass the unit as flux_unit='{guess}', or as another of "
        f"{', '.join(FLUX_UNITS)}, to clean_model_dir, SED.fetch or SED.import_grid. "
        "sed-tools asks for it when run in a terminal."
    )


def _flux_unit_name(unit_info: UnitInfo) -> str:
    return next(name for name, (flux_type, factor, _) in FLUX_UNITS.items()
                if (flux_type, factor) == (unit_info.flux_type, unit_info.flux_factor))


def _with_flux_unit(grid: str, unit_info: UnitInfo, flux_unit: Optional[str]) -> UnitInfo:
    """unit_info with the caller's flux unit if given, else as detected when the spectra declare theirs.

    Raises ValueError naming the best guess when they do not and none was given.
    """
    if flux_unit is not None:
        flux_type, flux_factor, _ = FLUX_UNITS[flux_unit]
        return dataclasses.replace(unit_info, flux_type=flux_type, flux_factor=flux_factor,
                                   flux_declared=True)
    if not unit_info.flux_declared:
        raise ValueError(_undeclared_flux_unit_message(grid, _flux_unit_name(unit_info)))
    return unit_info


def _detect_sample(sample_files: List[str]) -> Tuple[List[UnitInfo], int, int, int]:
    """Detect units in each sampled file.

    Returns (detections, n_already_standardized, n_invalid, n_index_grids).
    """
    detections = []
    skipped_standardized = 0
    skipped_invalid = 0
    index_grids = 0
    for filepath in sample_files:
        unit_info = detect_units_single_file(filepath)
        if unit_info is None:
            # Check why it was skipped
            meta, _ = _load_header(filepath)
            if meta.units_standardized:
                skipped_standardized += 1
            else:
                skipped_invalid += 1
        elif unit_info.confidence == 'index_grid':
            index_grids += 1
        else:
            detections.append(unit_info)
    return detections, skipped_standardized, skipped_invalid, index_grids


def _representative(detections: List[UnitInfo], wl_unit: str, flux_type: str) -> UnitInfo:
    """The detection whose flux factor the grid uses.

    The first detection matching both consensus units, else the first with the
    consensus wavelength unit (one always exists: the consensus came from them).
    """
    for d in detections:
        if d.wavelength_unit == wl_unit and d.flux_type == flux_type:
            return d
    return next(d for d in detections if d.wavelength_unit == wl_unit)


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
) -> None:
    """Write standardized spectrum with proper header."""
    
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
    
    # Stack first so a length mismatch fails before the file is truncated.
    data = np.column_stack([wl, flux])
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header)
    with open(filepath, 'ab') as f:
        np.savetxt(f, data, fmt='%.6f %.8e')


# ============================================================================
# HDF5 WAVELENGTH RECOVERY (for MSG index grids)
# ============================================================================

WAVE_KEYS = ('lambda', 'wavelength', 'wave', 'wl', 'wavelength_A', 'x')

def _is_physical_wavelength_array(wl: np.ndarray) -> bool:
    """Return True if wl looks like a physical wavelength grid in Angstroms."""
    if wl.size < 2:
        return False
    if wl[0] < 1.0 or wl[-1] > 1e9:
        return False
    # Reject pure integer index sequences
    if wl[0] < 2 and np.allclose(wl, np.arange(wl.size, dtype=float), atol=0.5):
        return False
    if wl[0] < 2 and np.allclose(wl, np.arange(1, wl.size + 1, dtype=float), atol=0.5):
        return False
    if np.allclose(np.diff(wl), 1.0, atol=0.01):
        return False
    return True


def recover_wavelengths_from_hdf5(
    model_dir: str,
    spec_group: str,
    expected_len: int
) -> Optional[np.ndarray]:
    """
    Attempt to recover wavelength array from HDF5 file for index grids.

    Searches every *.h5 in model_dir, in priority order:
      1. Direct wavelength dataset within spec_group
      2. range/x structure within spec_group
      3. Top-level wavelength arrays in the HDF5 root
      4. Global search for any physically plausible array of the right length

    An unreadable .h5 is reported with a warning and the next one is tried.
    Returns None if no physical wavelength grid is found.
    """
    for h5_path in glob.glob(os.path.join(model_dir, "*.h5")):
        try:
            with h5py.File(h5_path, 'r') as f:
                wl = _wavelengths_in_h5(f, spec_group, expected_len)
        except OSError as exc:
            logger.warning("Skipping unreadable HDF5 file %s while recovering wavelengths "
                           "for %s: %s", h5_path, spec_group, exc)
            continue
        if wl is not None:
            return wl
    return None


def _wavelengths_in_h5(f, spec_group: str, expected_len: int) -> Optional[np.ndarray]:
    """Apply the four recovery searches to one open HDF5 file."""
    if spec_group in f:
        wl = _wavelengths_in_group(f[spec_group], expected_len)
        if wl is not None:
            return wl
    wl = _top_level_wavelengths(f, expected_len)
    if wl is not None:
        return wl
    return _search_all_datasets(f, expected_len)


def _increasing_physical_prefix(wl: np.ndarray, expected_len: int) -> Optional[np.ndarray]:
    """wl[:expected_len] if wl is long enough and that prefix is a rising physical grid."""
    if (wl.size >= expected_len and
            np.all(np.diff(wl[:expected_len]) > 0) and
            _is_physical_wavelength_array(wl[:expected_len])):
        return wl[:expected_len]
    return None


def _wavelengths_in_group(group, expected_len: int) -> Optional[np.ndarray]:
    """Searches 1 and 2: a wavelength dataset in the spectrum's group, then its range/ subgroup."""
    for key in WAVE_KEYS:
        if key in group and isinstance(group[key], h5py.Dataset):
            wl = _increasing_physical_prefix(np.array(group[key]).astype(float).ravel(), expected_len)
            if wl is not None:
                return wl
    if 'range' in group and isinstance(group['range'], h5py.Group):
        return _wavelengths_in_range_group(group['range'], expected_len)
    return None


def _segment_number(name: str) -> int:
    match = re.search(r'\d+', name)
    return int(match.group()) if match else 0


def _wavelengths_in_range_group(rg, expected_len: int) -> Optional[np.ndarray]:
    """range/x, else the range segments' wavelength datasets concatenated in numeric order."""
    if 'x' in rg and isinstance(rg['x'], h5py.Dataset):
        wl = _increasing_physical_prefix(np.array(rg['x']).astype(float).ravel(), expected_len)
        if wl is not None:
            return wl

    segs = []
    seg_names = sorted([k for k in rg.keys() if 'range' in k.lower()], key=_segment_number)
    for sn in seg_names:
        if not isinstance(rg[sn], h5py.Group):
            continue
        sg = rg[sn]
        for key in WAVE_KEYS:
            if key in sg and isinstance(sg[key], h5py.Dataset):
                segs.append(np.array(sg[key]).astype(float).ravel())
                break
    if segs:
        wl = np.concatenate(segs)
        if wl.size >= expected_len and _is_physical_wavelength_array(wl[:expected_len]):
            return wl[:expected_len]
    return None


def _top_level_wavelengths(f, expected_len: int) -> Optional[np.ndarray]:
    """Search 3: a root-level wavelength dataset within two points of the expected length."""
    for key in WAVE_KEYS:
        if key in f and isinstance(f[key], h5py.Dataset):
            wl = np.array(f[key]).astype(float).ravel()
            if (wl.size >= 2 and np.all(np.diff(wl) > 0) and
                    _is_physical_wavelength_array(wl) and
                    abs(wl.size - expected_len) <= 2):
                return wl[:expected_len] if wl.size > expected_len else wl
    return None


def _iter_datasets(group):
    for item in group.values():
        if isinstance(item, h5py.Dataset):
            yield item
        elif isinstance(item, h5py.Group):
            yield from _iter_datasets(item)


def _search_all_datasets(f, expected_len: int) -> Optional[np.ndarray]:
    """Search 4: any rising physical numeric dataset of exactly the expected length."""
    for dataset in _iter_datasets(f):
        try:
            arr = np.array(dataset[()]).astype(float).ravel()
        except (TypeError, ValueError):
            continue  # string and compound datasets cannot be wavelength grids
        if (arr.size == expected_len and
                np.all(np.diff(arr) > 0) and
                _is_physical_wavelength_array(arr)):
            return arr
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
) -> Tuple[str, str]:
    """
    Clean and standardize a single spectrum file.
    
    Args:
        filepath: Path to spectrum file
        catalog_units: Pre-determined units for the spectral grid (from the sample).
                       If None, detects units per-file (legacy behavior); a flux
                       unit the header does not declare then raises ValueError.
        try_h5_recovery: Attempt HDF5 wavelength recovery for index grids
    
    Returns: (status, detail)
        status: 'skipped_already', 'skipped_index', 'skipped_invalid', 
                'converted', 'recovered', or 'converted_renormed' /
                'recovered_renormed' when the flux scale was also corrected
                (see _bolometric_renorm_ratio); detail then gives the factor.

    Raises RuntimeError if rewriting the file fails; the file may then hold
    only its new header.
    """
    status, detail, renorm_ratio = _clean_spectrum(filepath, catalog_units, try_h5_recovery)
    if renorm_ratio is not None:
        status += '_renormed'
    return status, detail


def _clean_spectrum(
    filepath: str,
    catalog_units: Optional[UnitInfo],
    try_h5_recovery: bool,
    flux_unit: Optional[str] = None,
) -> Tuple[str, str, Optional[float]]:
    """clean_spectrum_file's work. Also returns the factor the flux was divided by, or None.

    flux_unit is used when catalog_units is None (see clean_model_dir).
    """
    # Parse header
    meta, _ = _load_header(filepath)
    
    # Skip if already standardized
    if meta.units_standardized:
        return 'skipped_already', 'already standardized', None
    
    # Read data
    wl, flux = read_spectrum_data(filepath)
    
    if wl.size == 0 or flux.size == 0:
        return 'skipped_invalid', 'empty data', None
    
    # Clean: remove NaN/inf and non-positive wavelengths
    valid = np.isfinite(wl) & np.isfinite(flux) & (wl > 0)
    wl, flux = wl[valid], flux[valid]
    
    if wl.size < 2:
        return 'skipped_invalid', 'no valid data points', None
    
    wl, flux = _sorted_unique(wl, flux)
    
    if is_index_grid(wl):
        wl = _recovered_wavelengths(filepath, meta, len(flux), try_h5_recovery)
        if wl is None:
            return 'skipped_index', 'index grid without HDF5 recovery', None
        status = 'recovered'
    else:
        status = 'converted'
    
    # Use catalog units if provided, otherwise detect per-file (the file has
    # at least two valid points by now, so per-file detection always succeeds).
    if catalog_units is not None:
        unit_info = catalog_units
    else:
        unit_info = _with_flux_unit("this spectrum", detect_units_single_file(filepath), flux_unit)
    wl_std, flux_std = convert_to_standard_units(wl, flux, unit_info)
    
    # Validate
    valid, msg = validate_converted_spectrum(wl_std, flux_std)
    if not valid:
        return 'skipped_invalid', f'validation failed: {msg}', None

    renorm_ratio = _bolometric_renorm_ratio(wl_std, flux_std, meta.teff)
    if renorm_ratio is not None:
        flux_std = flux_std / renorm_ratio

    try:
        write_standardized_spectrum(filepath, wl_std, flux_std, meta, unit_info)
    except OSError as exc:
        raise RuntimeError(
            f"Failed while rewriting {filepath}: {exc}. The file may now hold only its new "
            "header, which marks it as standardized, so later runs would skip it. Restore "
            "it from the original download, then run the cleaning again."
        ) from exc
    
    detail = f"{wl_std[0]:.0f}-{wl_std[-1]:.0f}Å, {unit_info.wavelength_unit}→Å, conf={unit_info.confidence}"
    if renorm_ratio is not None:
        detail += f", flux divided by {renorm_ratio:.4g} (bolometric renormalisation)"
    return status, detail, renorm_ratio


def _bolometric_renorm_ratio(wl: np.ndarray, flux: np.ndarray, teff: float) -> Optional[float]:
    """The factor to divide the flux by when its absolute scale is wrong, else None.

    That factor is the spectrum's integrated flux over pi*B_lambda(Teff)
    integrated on the same wavelengths, so both integrals share one quadrature
    and a correctly scaled spectrum gives about 1 however it is sampled.
    Spectra without a Teff, or whose wavelengths hold too little of sigma*T^4,
    are not checked.

    The first version differed in two ways, both of which rescaled correct
    spectra. It integrated the blackbody on a 2000-point linear grid, which
    misses a hot star's peak on a wide wavelength range (correct 30000 K
    blackbodies over 10 Å-1e7 Å came out 11 times too bright). And it also
    rescaled spectra below 0.1 times the blackbody, which multiplied correct
    3000-3500 K giants observed only in the optical by 10-50.
    """
    if not (np.isfinite(teff) and teff > 0):
        return None
    expected = float(np.trapezoid(planck_flam(wl, teff), wl))
    if expected < _RENORM_MIN_COVERAGE * SIGMA * teff ** 4:
        return None
    ratio = float(np.trapezoid(np.maximum(flux, 0), wl)) / expected
    if ratio > _RENORM_MAX_RATIO:
        return ratio
    return None


def _sorted_unique(wl: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by wavelength and drop repeated wavelengths, keeping the first."""
    order = np.argsort(wl)
    wl, flux = wl[order], flux[order]
    unique_mask = np.concatenate([[True], np.diff(wl) > 0])
    return wl[unique_mask], flux[unique_mask]


def _recovered_wavelengths(filepath: str, meta: SpectrumMeta, n_points: int,
                           try_h5_recovery: bool) -> Optional[np.ndarray]:
    """Real wavelengths for an index-grid MSG spectrum from the grid's HDF5 files, or None."""
    if not (try_h5_recovery and meta.spec_group and 'msg' in meta.source.lower()):
        return None
    return recover_wavelengths_from_hdf5(os.path.dirname(filepath), meta.spec_group, n_points)


def rebuild_lookup_table(model_dir: str) -> str:
    """Rebuild lookup_table.csv from spectrum file headers."""
    txt_files = sorted(glob.glob(os.path.join(model_dir, "*.txt")))
    if not txt_files:
        return ""
    
    rows = []
    all_keys = {'file_name'}
    
    for filepath in txt_files:
        meta, _ = _load_header(filepath)
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
    
    out_path = os.path.join(model_dir, "lookup_table.csv")
    priority = ['file_name', 'teff', 'logg', 'metallicity', 'source', 'units_standardized']
    columns = [c for c in priority if c in all_keys]
    columns += sorted(k for k in all_keys if k not in priority)
    write_lookup_csv(rows, out_path, columns=columns)
    return out_path


# ============================================================================
# PUBLIC API
# ============================================================================

def clean_model_dir(
    model_dir: str,
    try_h5_recovery: bool = True,
    rebuild_lookup: bool = True,
    sample_fraction: float = 0.10,
    flux_unit: Optional[str] = None,
) -> Dict:
    """
    Clean all spectra in a model directory.
    
    KEY: Detects units from a random sample first (default 10%), then applies uniformly.
    
    Args:
        model_dir: Directory containing *.txt spectrum files
        try_h5_recovery: Attempt HDF5 wavelength recovery for index grids
        rebuild_lookup: Rebuild lookup_table.csv after cleaning
        sample_fraction: Fraction of files to sample for unit detection (default 10%)
        flux_unit: The spectra's flux unit, a name in FLUX_UNITS ('flam', 'fnu',
            'fnu_jy', 'flam_si', 'normalized'). It replaces the unit their
            headers declare, and is required when they declare none.
    
    Returns:
        Summary dict with counts and file lists. When any spectrum's flux scale
        was corrected it also has 'renormed': {file name: factor its flux was
        divided by}; those files are listed under 'converted' or 'recovered' too.

    Raises:
        ValueError: before any file is changed, if flux_unit is not a FLUX_UNITS
            name, or is None while the sampled headers do not all declare the
            flux unit; the message gives the best guess.
        RuntimeError: if any spectrum is left unconverted (invalid, unrecoverable
            index grid, unreadable, or without a declared flux unit when no
            grid-wide units were found). Raised after every file has been tried
            and the lookup table rebuilt, listing each such file and why.
            Raised at once if rewriting a file fails.
    """
    _check_flux_unit_name(flux_unit)
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
    # STEP 1: Detect catalog-wide units from a sample
    # =========================================================
    catalog_units, detection_stats = detect_catalog_units(txt_files, sample_fraction)
    if catalog_units is not None:
        catalog_units = _with_flux_unit(model_dir, catalog_units, flux_unit)
    summary['detection_stats'] = detection_stats
    if catalog_units:
        summary['catalog_units'] = {
            'wavelength': catalog_units.wavelength_unit,
            'flux': catalog_units.flux_type,
            'confidence': catalog_units.confidence,
        }
    _print_detected_units(catalog_units, detection_stats)
    
    # =========================================================
    # STEP 2: Apply catalog units to ALL files
    # =========================================================
    failures = _clean_all_files(txt_files, catalog_units, try_h5_recovery, summary, flux_unit)
    n_per_file = len(summary['converted']) + len(summary['recovered'])
    if catalog_units is None and n_per_file:
        logger.warning(
            "%s: the sampled files gave no grid-wide units (status: %s); %d spectra were "
            "converted using units detected file by file.",
            model_dir, detection_stats.get('status', 'unknown'), n_per_file,
        )
    if 'renormed' in summary:
        _warn_renormed(model_dir, summary['renormed'])
    
    # =========================================================
    # STEP 3: Rebuild lookup table
    # =========================================================
    if rebuild_lookup:
        lookup_path = rebuild_lookup_table(model_dir)
        if lookup_path:
            summary['lookup_updated'] = True
            summary['lookup_path'] = lookup_path
    
    if failures:
        raise RuntimeError(
            _unconverted_report(model_dir, len(txt_files), failures, summary['lookup_updated'])
        )
    return summary


# Outcomes that leave a spectrum in its original units.
_UNCONVERTED = ('skipped_invalid', 'skipped_index', 'error')


def _print_detected_units(catalog_units: Optional[UnitInfo], stats: Dict) -> None:
    if catalog_units:
        print(f"[clean] Spectral grid units (from {stats.get('sample_size', '?')} samples): "
              f"λ={catalog_units.wavelength_unit} ({stats.get('wavelength_agreement', '?')}), "
              f"flux={catalog_units.flux_type} ({stats.get('flux_agreement', '?')})")
    elif stats.get('status') == 'already_standardized':
        print(f"[clean] Spectral grid already standardized (all {stats.get('sample_size', '?')} samples)")
    else:
        print(f"[clean] Warning: Could not determine spectral grid units. Status: {stats.get('status', 'unknown')}")


def _clean_all_files(txt_files: List[str], catalog_units: Optional[UnitInfo],
                     try_h5_recovery: bool, summary: Dict,
                     flux_unit: Optional[str]) -> List[Tuple[str, str]]:
    """Clean every file, appending each file name to its status list in *summary*.

    A renormalised spectrum is listed under its status ('converted' or
    'recovered') and in summary['renormed'] with its factor; that key is added
    only when there is one.

    Returns (file name, reason) for every spectrum left unconverted. A file that
    cannot be read is recorded and the batch goes on; a failed rewrite raises.
    """
    failures = []
    renormed = {}
    t_start = time.time()
    for i, filepath in enumerate(txt_files):
        filename = os.path.basename(filepath)
        try:
            status, detail, renorm_ratio = _clean_spectrum(filepath, catalog_units, try_h5_recovery,
                                                           flux_unit)
        except (OSError, ValueError) as exc:
            # Raised before the file is rewritten (rewrite failures raise RuntimeError).
            status, detail, renorm_ratio = 'error', f'{type(exc).__name__}: {exc}', None
        summary[status].append(filename)
        if renorm_ratio is not None:
            renormed[filename] = renorm_ratio
        if status in _UNCONVERTED:
            failures.append((filename, detail))
        _print_progress(i + 1, len(txt_files), t_start)
    print()
    if renormed:
        summary['renormed'] = renormed
    return failures


def _warn_renormed(model_dir: str, renormed: Dict[str, float]) -> None:
    ratios = list(renormed.values())
    logger.warning(
        "%s: %d spectra integrated to %.3g-%.3g times a blackbody at their Teff, so each "
        "one's flux was divided by its factor (bolometric renormalisation). The summary's "
        "'renormed' entry lists every file and factor.",
        model_dir, len(ratios), min(ratios), max(ratios),
    )


def _print_progress(done: int, total: int, t_start: float) -> None:
    elapsed = time.time() - t_start
    rate = done / elapsed if elapsed > 0 else 1.0
    eta = (total - done) / rate
    print(
        f"\r  [clean] {done}/{total} ({100*done/total:.1f}%)  "
        f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s   ",
        end="", flush=True,
    )


def _unconverted_report(model_dir: str, n_total: int, failures: List[Tuple[str, str]],
                        lookup_rebuilt: bool) -> str:
    """Error text listing every unconverted spectrum, grouped by reason."""
    by_reason = defaultdict(list)
    for filename, reason in failures:
        by_reason[reason].append(filename)
    lines = [f"{len(failures)} of {n_total} spectra in {model_dir} were not converted "
             "and are still in their original units:"]
    for reason, filenames in by_reason.items():
        lines.append(f"  {reason} ({len(filenames)}): {', '.join(filenames)}")
    if len(failures) < n_total:
        rebuilt = " and lookup_table.csv was rebuilt" if lookup_rebuilt else ""
        lines.append(f"The other spectra were standardized in place{rebuilt}; a re-run skips them.")
    lines.append("Fix or remove the listed files, then clean the directory again.")
    return "\n".join(lines)
