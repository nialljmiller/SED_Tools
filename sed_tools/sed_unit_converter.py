#!/usr/bin/env python3
"""
Enhanced unit detection and conversion for stellar atmosphere models.

Ensures all SEDs are converted to:
  - Wavelength: Angstroms (Å)
  - Flux: erg/s/cm²/Å (F_lambda)
"""

import re
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

# Physical constants
SPEED_OF_LIGHT = 2.99792458e18  # Angstroms/s
SIGMA = 5.670374419e-5  # erg s-1 cm-2 K-4


@dataclass
class UnitDetectionResult:
    """Results of unit detection."""
    wavelength_unit: str  # 'angstrom', 'nm', 'um', 'cm', 'm'
    wavelength_factor: float  # multiply by this to get Angstroms
    flux_unit: str  # 'flam', 'fnu', 'normalized', 'unknown'
    flux_factor: float  # Additional scaling factor (for cgs vs SI, etc.)
    confidence: str  # 'high', 'medium', 'low'
    source: str  # 'header', 'range', 'unknown'


def detect_units_from_header(filepath: str) -> Dict[str, any]:
    """
    Parse header lines for unit information.
    
    Returns dict with keys:
        - 'wavelength_unit': detected wavelength unit string
        - 'flux_unit': detected flux unit string  
        - 'wavelength_column': column name for wavelength
        - 'flux_column': column name for flux
    """
    result = {
        'wavelength_unit': None,
        'flux_unit': None,
        'wavelength_column': None,
        'flux_column': None
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            # Read first 100 lines or until data starts
            for line_num, line in enumerate(f):
                if line_num > 100:
                    break
                    
                # Skip non-comment lines
                if not line.startswith('#'):
                    continue
                    
                line_lower = line.lower()
                
                # Look for wavelength units
                if 'wavelength' in line_lower or 'lambda' in line_lower or 'wave' in line_lower:
                    if 'angstrom' in line_lower or '(a)' in line_lower or '(å)' in line_lower:
                        result['wavelength_unit'] = 'angstrom'
                    elif 'nanometer' in line_lower or '(nm)' in line_lower:
                        result['wavelength_unit'] = 'nm'
                    elif 'micron' in line_lower or 'micrometer' in line_lower or '(um)' in line_lower or '(µm)' in line_lower:
                        result['wavelength_unit'] = 'um'
                    elif '(cm)' in line_lower:
                        result['wavelength_unit'] = 'cm'
                    elif '(m)' in line_lower and 'nm' not in line_lower:
                        result['wavelength_unit'] = 'm'
                
                # Look for flux units
                if 'flux' in line_lower or 'f_lambda' in line_lower or 'flam' in line_lower:
                    if 'erg' in line_lower and 'cm' in line_lower and 's' in line_lower:
                        if '/a' in line_lower or 'angstrom' in line_lower:
                            result['flux_unit'] = 'flam_cgs'  # erg/s/cm²/Å
                    elif 'erg' in line_lower and ('hz' in line_lower or 'frequency' in line_lower):
                        result['flux_unit'] = 'fnu_cgs'  # erg/s/cm²/Hz
                    elif 'jy' in line_lower or 'jansky' in line_lower:
                        result['flux_unit'] = 'fnu_jy'  # Jansky
                    elif 'w' in line_lower and 'm' in line_lower:
                        result['flux_unit'] = 'flam_si'  # W/m²/Å or similar
                    elif 'normalized' in line_lower or 'norm' in line_lower:
                        result['flux_unit'] = 'normalized'
                
                # Look for column descriptions (e.g., "# columns = wavelength_A flux continuum")
                if 'columns' in line_lower or 'column' in line_lower:
                    # Extract column names
                    parts = line.split('=')
                    if len(parts) > 1:
                        cols = [c.strip() for c in parts[1].split()]
                        if len(cols) >= 2:
                            result['wavelength_column'] = cols[0]
                            result['flux_column'] = cols[1]
    
    except Exception:
        pass
    
    return result


def detect_wavelength_unit(wl: np.ndarray, header_info: Dict[str, any] = None) -> Tuple[str, float, str]:
    """
    Detect wavelength units from data range and header.
    
    Returns:
        (unit_name, conversion_factor, confidence)
        where conversion_factor converts to Angstroms
    """
    # First check header
    if header_info and header_info.get('wavelength_unit'):
        unit = header_info['wavelength_unit']
        factors = {
            'angstrom': 1.0,
            'nm': 10.0,
            'um': 1e4,
            'cm': 1e8,
            'm': 1e10
        }
        return unit, factors.get(unit, 1.0), 'high'
    
    # Infer from data range
    wl_max = wl.max()
    wl_min = wl.min()
    
    # Index grid detection (should be caught earlier, but safety check)
    if wl_min < 10 and wl_max < 1000 and np.allclose(np.diff(wl), 1.0, rtol=0.1):
        return 'index', None, 'index_grid'
    
    # Angstroms: typical range 1000-100,000 Å (UV to far-IR)
    if 1000 <= wl_max <= 1e6:
        if wl_min > 100:  # Definitely Angstroms
            return 'angstrom', 1.0, 'high'
        else:
            return 'angstrom', 1.0, 'medium'
    
    # Nanometers: typical range 100-10,000 nm
    if 100 < wl_max <= 10000 and wl_min > 50:
        return 'nm', 10.0, 'medium'
    
    # Micrometers: typical range 0.1-1000 μm
    if 0.1 < wl_max <= 1000 and wl_min > 0.05:
        return 'um', 1e4, 'medium'
    
    # Centimeters: very rare but possible for radio
    if 1e-5 < wl_max <= 100 and wl_max < wl_min * 100:
        return 'cm', 1e8, 'low'
    
    # Meters: also rare
    if 1e-7 < wl_max <= 1 and wl_max < wl_min * 100:
        return 'm', 1e10, 'low'
    
    # Default to Angstroms if uncertain
    return 'angstrom', 1.0, 'low'


def detect_flux_unit(flux: np.ndarray, wl: np.ndarray, 
                     header_info: Dict[str, any] = None) -> Tuple[str, float, str]:
    """
    Detect flux units from data characteristics and header.
    
    Returns:
        (unit_name, additional_factor, confidence)
    """
    # Check header first
    if header_info and header_info.get('flux_unit'):
        unit = header_info['flux_unit']
        
        if unit == 'flam_cgs':
            return 'flam', 1.0, 'high'
        elif unit == 'fnu_cgs':
            return 'fnu', 1.0, 'high'
        elif unit == 'fnu_jy':
            # Convert Jansky to erg/s/cm²/Hz: 1 Jy = 1e-23 erg/s/cm²/Hz
            return 'fnu', 1e-23, 'high'
        elif unit == 'flam_si':
            # W/m²/Å to erg/s/cm²/Å: 1 W/m² = 1e3 erg/s/cm²
            return 'flam', 1e3, 'medium'
        elif unit == 'normalized':
            return 'normalized', 1.0, 'high'
    
    # Analyze flux magnitudes to infer units
    flux_median = np.median(flux[np.isfinite(flux)])
    flux_max = np.max(flux[np.isfinite(flux)])
    
    # F_lambda in erg/s/cm²/Å typically ranges from ~1e-30 to ~1e-5 for stellar atmospheres
    # F_nu in erg/s/cm²/Hz typically ranges from ~1e-30 to ~1e-5 as well
    # But F_lambda and F_nu differ by factor of c/λ²
    
    # If flux is order 1e-6 to 1e-15, likely F_lambda in cgs
    if 1e-15 < flux_median < 1e-5:
        return 'flam', 1.0, 'medium'
    
    # If flux is order 1e-25 to 1e-35, could be F_nu or very faint F_lambda
    if 1e-35 < flux_median < 1e-20:
        return 'fnu', 1.0, 'low'
    
    # If flux is order 0.1 to 10, likely normalized
    if 0.01 < flux_median < 100:
        return 'normalized', 1.0, 'low'
    
    # Default assumption: F_lambda in cgs
    return 'flam', 1.0, 'low'


def convert_fnu_to_flam(flux_fnu: np.ndarray, wl_angstrom: np.ndarray) -> np.ndarray:
    """
    Convert F_nu (erg/s/cm²/Hz) to F_lambda (erg/s/cm²/Å).
    
    F_lambda = F_nu * (c / λ²)
    
    where:
        c = speed of light in Å/s
        λ = wavelength in Å
    """
    # c / λ² = (2.998e18 Å/s) / (λ²)
    conversion = SPEED_OF_LIGHT / (wl_angstrom ** 2)
    flux_flam = flux_fnu * conversion
    return flux_flam


def convert_to_standard_units(
    wl: np.ndarray, 
    flux: np.ndarray,
    filepath: str = None
) -> Tuple[np.ndarray, np.ndarray, UnitDetectionResult]:
    """
    Convert arbitrary wavelength and flux arrays to standard units.
    
    Standard units:
        - Wavelength: Angstroms (Å)
        - Flux: erg/s/cm²/Å (F_lambda)
    
    Returns:
        (wavelength_angstrom, flux_flam, detection_result)
    """
    
    # Parse header if filepath provided
    header_info = {}
    if filepath:
        header_info = detect_units_from_header(filepath)
    
    # Detect wavelength units
    wl_unit, wl_factor, wl_confidence = detect_wavelength_unit(wl, header_info)
    
    if wl_unit == 'index':
        raise ValueError("Wavelength appears to be index grid, not physical units")
    
    # Convert wavelength to Angstroms
    wl_angstrom = wl * wl_factor
    
    # Detect flux units
    flux_unit, flux_factor, flux_confidence = detect_flux_unit(flux, wl_angstrom, header_info)
    
    # Apply flux scaling factor
    flux_scaled = flux * flux_factor
    
    # Convert F_nu to F_lambda if needed
    if flux_unit == 'fnu':
        flux_flam = convert_fnu_to_flam(flux_scaled, wl_angstrom)
    elif flux_unit == 'normalized':
        # Keep normalized flux as-is (will need external info to scale properly)
        flux_flam = flux_scaled
    else:
        # Already F_lambda
        flux_flam = flux_scaled
    
    # Also adjust flux for wavelength unit conversion
    # If wavelength units changed, F_lambda changes inversely
    # F_lambda (per Å) = F_lambda (per original unit) / conversion_factor
    flux_flam = flux_flam / wl_factor
    
    # Create detection result
    overall_confidence = 'high' if (wl_confidence == 'high' and flux_confidence == 'high') else \
                        'medium' if (wl_confidence in ['high', 'medium'] and flux_confidence in ['high', 'medium']) else \
                        'low'
    
    result = UnitDetectionResult(
        wavelength_unit=wl_unit,
        wavelength_factor=wl_factor,
        flux_unit=flux_unit,
        flux_factor=flux_factor,
        confidence=overall_confidence,
        source='header' if header_info.get('wavelength_unit') or header_info.get('flux_unit') else 'range'
    )
    
    return wl_angstrom, flux_flam, result


def validate_standard_units(wl: np.ndarray, flux: np.ndarray, 
                           teff: float = None) -> Tuple[bool, str]:
    """
    Validate that wavelength and flux are in reasonable ranges for standard units.
    
    Returns:
        (is_valid, reason)
    """
    # Wavelength checks (should be in Angstroms)
    if wl.min() < 1 or wl.max() > 1e7:
        return False, f"Wavelength out of expected range: {wl.min():.1f} - {wl.max():.1f} Å"
    
    # Typical stellar spectrum range
    if not (100 < wl.min() < 5000):
        return False, f"Wavelength minimum unusual: {wl.min():.1f} Å"
    
    # Flux checks (should be F_lambda in erg/s/cm²/Å)
    flux_median = np.median(flux[np.isfinite(flux)])
    
    # For stellar atmospheres, typical F_lambda ranges from ~1e-20 to ~1e-5
    if not (1e-30 < flux_median < 1e0):
        return False, f"Flux median out of expected range: {flux_median:.2e}"
    
    # Check for all negative flux (sign error)
    if np.all(flux < 0):
        return False, "All flux values are negative"
    
    # Check for unreasonable number of negative values
    neg_frac = np.sum(flux < 0) / len(flux)
    if neg_frac > 0.1:
        return False, f"Too many negative flux values: {neg_frac:.1%}"
    
    # If Teff provided, check integrated flux makes sense
    if teff is not None:
        try:
            from scipy.integrate import simpson
            fbol = simpson(flux, wl)
            fbol_expected = SIGMA * teff ** 4
            
            # Allow 10 orders of magnitude range (very permissive)
            if abs(np.log10(fbol / fbol_expected)) > 10:
                return False, f"Integrated flux inconsistent with Teff: {fbol:.2e} vs {fbol_expected:.2e}"
        except:
            pass  # Skip this check if integration fails
    
    return True, "OK"


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing unit detection and conversion...\n")
    
    # Test 1: Data in nanometers
    wl_nm = np.linspace(300, 1000, 100)  # 300-1000 nm
    flux_flam = np.ones(100) * 1e-10  # erg/s/cm²/Å (but per nm)
    
    wl_out, flux_out, result = convert_to_standard_units(wl_nm, flux_flam)
    print(f"Test 1 (nm input):")
    print(f"  Input: {wl_nm[0]:.1f} - {wl_nm[-1]:.1f} nm")
    print(f"  Output: {wl_out[0]:.1f} - {wl_out[-1]:.1f} Å")
    print(f"  Detection: {result.wavelength_unit} -> Å (factor: {result.wavelength_factor})")
    print(f"  Confidence: {result.confidence}\n")
    
    # Test 2: Data in Angstroms with F_nu
    wl_ang = np.linspace(3000, 10000, 100)
    flux_fnu = np.ones(100) * 1e-20  # erg/s/cm²/Hz
    
    wl_out2, flux_out2, result2 = convert_to_standard_units(wl_ang, flux_fnu)
    print(f"Test 2 (Å input, F_nu):")
    print(f"  Input: {wl_ang[0]:.1f} - {wl_ang[-1]:.1f} Å (F_nu)")
    print(f"  Output: {wl_out2[0]:.1f} - {wl_out2[-1]:.1f} Å (F_lambda)")
    print(f"  Detection: {result2.flux_unit}")
    print(f"  Flux conversion: {flux_fnu[50]:.2e} -> {flux_out2[50]:.2e}\n")
    
    # Test 3: Validation
    valid, reason = validate_standard_units(wl_out, flux_out)
    print(f"Validation: {valid} - {reason}")