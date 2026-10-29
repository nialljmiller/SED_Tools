"""Spectrum units for the cleaner: unit tables, detection from headers and data, the flux-unit decision and conversion.

The cleaner (clean.py) applies these to a whole grid; spectra_cleaner is its old import path.
"""

from __future__ import annotations

import dataclasses
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .._constants import C
from .header_parser import parse_header as _parse_header_dict
from .spectrum_io import read_text_spectrum

logger = logging.getLogger("sed_tools.spectra_cleaner")


SPEED_OF_LIGHT_ANGSTROM = C * 1e8  # c in Å/s — from _constants

# Wavelength conversion factors → Angstroms
WAVELENGTH_FACTORS = {
    'angstrom': 1.0,
    'nm': 10.0,
    'um': 1e4,
    'cm': 1e8,
    'm': 1e10,
}

# Wavelength unit names a header may give -> the WAVELENGTH_FACTORS name
_WAVELENGTH_NAMES = {
    **dict.fromkeys(['angstrom', 'ang', 'a', 'aa', 'å'], 'angstrom'),
    **dict.fromkeys(['nm', 'nanometer', 'nanometre'], 'nm'),
    **dict.fromkeys(['um', 'µm', 'micron', 'micrometer'], 'um'),
    **dict.fromkeys(['cm', 'centimeter'], 'cm'),
    **dict.fromkeys(['m', 'meter'], 'm'),
}

# The flux units a caller can give with flux_unit:
# name -> (flux type, factor applied first, description). The names are the
# ones detect_units_from_header gives a declared unit.
# 1 W/m² = 1e3 erg/s/cm², so W/m² per Å, nm and µm are 1e3, 1e2 and 0.1 erg/s/cm²/Å.
FLUX_UNITS = {
    'flam': ('flam', 1.0, 'erg/s/cm²/Å (F_λ)'),
    'fnu': ('fnu', 1.0, 'erg/s/cm²/Hz (F_ν)'),
    'fnu_jy': ('fnu', 1e-23, 'Jansky (F_ν)'),
    'flam_si': ('flam', 1e3, 'W/m²/Å'),
    'flam_si_nm': ('flam', 1e2, 'W/m²/nm'),
    'flam_si_um': ('flam', 0.1, 'W/m²/µm'),
    'normalized': ('normalized', 1.0, 'arbitrary scale, left as it is'),
}

# How detect_flux_unit_from_range guesses. Model-atmosphere surface fluxes in
# erg/s/cm²/Å are 'flam', but it reads them as Jansky or as normalised.
FLUX_GUESS_WHY = "guessed from the size of the flux values, assuming flux received at Earth"


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
        r'wavelength[_\s]*unit\s*[=:]\s*(\S+)',
        r'wavelength\s*\(\s*(\S+)\s*\)',
        r'wave\s*\(\s*(\S+)\s*\)',
        r'lambda\s*\(\s*(\S+)\s*\)',
        r'columns?\s*[=:].*wavelength[_\s]*\(?([aå]ngstrom|nm|um|µm|micron)\)?',
    ]
    
    for pattern in wl_patterns:
        match = re.search(pattern, header_lower)
        if match:
            result['wavelength_unit'] = _WAVELENGTH_NAMES.get(match.group(1).lower().strip('()[]'))
            break
    
    # --- Flux unit detection ---
    # Units come before the bare quantity names, so that 'F_nu (Jy)' is read
    # as Jansky and 'F_lambda (W/m2/nm)' as W/m²/nm. The (?<![a-z]) and
    # (?![a-z]) keep text such as 'erg/s/cm2/arcsec2' or 'low mass' from
    # matching a unit.
    per_angstrom = r'(angstroms?|angs|ang|aa|a|å)(?![a-z])'
    flux_patterns = [
        # F_lambda in erg/s/cm²/Å
        (r'erg\s*/?\s*s\s*/?\s*cm\s*\^?2?\s*/?\s*' + per_angstrom, 'flam'),
        (r'erg\s*/?\s*cm\s*\^?2?\s*/?\s*s\s*/?\s*' + per_angstrom, 'flam'),
        # F_nu in erg/s/cm²/Hz
        (r'erg\s*/?\s*s\s*/?\s*cm\s*\^?2?\s*/?\s*hz', 'fnu'),
        (r'erg\s*/?\s*cm\s*\^?2?\s*/?\s*s\s*/?\s*hz', 'fnu'),
        # Jansky
        (r'\bjansky\b|\bjy\b', 'fnu_jy'),
        # SI F_lambda in W/m² per Å, nm or µm
        (r'(?<![a-z])w\s*/?\s*m\s*\^?2?\s*/?\s*' + per_angstrom, 'flam_si'),
        (r'(?<![a-z])w\s*/?\s*m\s*\^?2?\s*/?\s*nm(?![a-z])', 'flam_si_nm'),
        (r'(?<![a-z])w\s*/?\s*m\s*\^?2?\s*/?\s*um(?![a-z])', 'flam_si_um'),
        # Quantity names with no unit, taken as erg/s/cm²/Å or erg/s/cm²/Hz
        (r'f_?lambda', 'flam'),
        (r'flam\b', 'flam'),
        (r'f_?nu', 'fnu'),
        (r'fnu\b', 'fnu'),
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
        flux_type, flux_factor, _ = FLUX_UNITS[header_units['flux_unit']]
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
        f"{', '.join(FLUX_UNITS)}, to clean_model_dir, SED.fetch or SED.import_grid; on the "
        f"command line, pass --flux-unit {guess} to sed-tools grids, rebuild or import. "
        "sed-tools asks for it when run in a terminal."
    )


def _flux_unit_name(unit_info: UnitInfo) -> str:
    return next(name for name, (flux_type, factor, _) in FLUX_UNITS.items()
                if (flux_type, factor) == (unit_info.flux_type, unit_info.flux_factor))


def _with_flux_unit(grid: str, unit_info: UnitInfo, flux_unit: Optional[str], *,
                    report: bool = True) -> UnitInfo:
    """unit_info as detected when the spectra declare their flux unit, else with the caller's.

    A declared unit is never replaced: a flux_unit given as well is not used,
    and a warning says so (unless ``report`` is False).  Raises ValueError
    naming the best guess when the spectra declare none and none was given.
    """
    if unit_info.flux_declared:
        if flux_unit is not None and report:
            logger.warning("%s: the spectra declare their flux unit (%s); flux_unit=%r is not used",
                           grid, _flux_unit_name(unit_info), flux_unit)
        return unit_info
    if flux_unit is None:
        raise ValueError(_undeclared_flux_unit_message(grid, _flux_unit_name(unit_info)))
    flux_type, flux_factor, _ = FLUX_UNITS[flux_unit]
    return dataclasses.replace(unit_info, flux_type=flux_type, flux_factor=flux_factor, flux_declared=True)


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
