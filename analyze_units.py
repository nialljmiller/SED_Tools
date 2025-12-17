#!/usr/bin/env python3
"""
Systematic unit analyzer - no guessing, actual analysis
"""
import os
import re
import numpy as np
from scipy.integrate import simpson
from pathlib import Path

SIGMA = 5.670374419e-5  # erg s-1 cm-2 K-4

def read_spectrum_with_header(filepath):
    """Read spectrum and extract header metadata."""
    header_lines = []
    data_lines = []
    
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#'):
                header_lines.append(stripped)
            elif stripped:
                data_lines.append(stripped)
    
    # Parse header for metadata
    metadata = {}
    for line in header_lines:
        # Look for key = value patterns
        if '=' in line:
            parts = line.lstrip('#').split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip().lower()
                value = parts[1].strip()
                metadata[key] = value
    
    # Parse data
    try:
        data = []
        for line in data_lines[:100]:  # Sample first 100 lines
            parts = line.split()
            if len(parts) >= 2:
                data.append([float(parts[0]), float(parts[1])])
        
        if not data:
            return None, None, metadata
        
        data = np.array(data)
        wavelength = data[:, 0]
        flux = data[:, 1]
        
        return wavelength, flux, metadata
    except:
        return None, None, metadata


def analyze_wavelength_units(wavelength):
    """Determine wavelength units from values."""
    if wavelength is None or len(wavelength) == 0:
        return None, None
    
    min_wl = np.min(wavelength)
    max_wl = np.max(wavelength)
    median_wl = np.median(wavelength)
    
    # Determine units based on typical ranges
    # UV-Visible-IR range is ~100nm to ~10000nm = ~1000Å to ~100000Å
    
    if 100 < median_wl < 100000:
        # Likely Angstroms
        return "Angstrom", 1.0
    elif 10 < median_wl < 10000:
        # Likely nanometers
        return "nanometer", 10.0  # nm to Angstrom
    elif 0.1 < median_wl < 10:
        # Likely micrometers
        return "micrometer", 10000.0  # um to Angstrom
    elif median_wl < 0.1:
        # Likely meters
        return "meter", 1e10  # m to Angstrom
    else:
        return "unknown", 1.0


def analyze_flux_units(wavelength, flux, teff, metadata):
    """Determine flux units and needed correction."""
    if wavelength is None or flux is None or len(flux) == 0:
        return None, 1.0, None
    
    # Clean data
    good = np.isfinite(wavelength) & np.isfinite(flux) & (flux > 0)
    if np.sum(good) < 10:
        return None, 1.0, None
    
    wl = wavelength[good]
    fl = flux[good]
    
    # Integrate to get bolometric flux
    fbol_model = simpson(fl, wl)
    fbol_expected = SIGMA * teff**4
    
    ratio = fbol_model / fbol_expected if fbol_expected > 0 else 0
    
    # Check metadata for unit hints
    unit_hint = None
    for key in ['flux_unit', 'flux_units', 'unit', 'units']:
        if key in metadata:
            unit_hint = metadata[key].lower()
            break
    
    # Analyze ratio to determine what's wrong
    analysis = {
        'ratio': ratio,
        'fbol_model': fbol_model,
        'fbol_expected': fbol_expected,
        'unit_hint': unit_hint,
        'diagnosis': None,
        'correction': 1.0
    }
    
    if 0.8 <= ratio <= 1.2:
        analysis['diagnosis'] = "Correct units: erg/s/cm²/Å"
        analysis['correction'] = 1.0
    
    elif 0.25 <= ratio <= 0.35:
        # Close to 1/π - likely per steradian
        analysis['diagnosis'] = "Per steradian (needs ×π)"
        analysis['correction'] = np.pi
    
    elif 0.075 <= ratio <= 0.085:
        # Close to 1/4π - likely isotropic flux
        analysis['diagnosis'] = "Per 4π steradians (needs ×4π)"
        analysis['correction'] = 4 * np.pi
    
    elif ratio < 0.001:
        # Way too small - likely missing scale factor
        # Check magnitudes
        if ratio < 1e-6:
            analysis['diagnosis'] = f"Way too small ({ratio:.2e}), likely missing large scale factor"
            analysis['correction'] = 1.0 / ratio  # Just report what's needed
        else:
            # Try powers of 10
            for power in [2, 3, 4, 5]:
                test_ratio = ratio * (10 ** power)
                if 0.8 <= test_ratio <= 1.2:
                    analysis['diagnosis'] = f"Missing 10^{power} factor"
                    analysis['correction'] = 10 ** power
                    break
    
    elif ratio > 10:
        # Too large
        for power in [1, 2, 3, 4]:
            test_ratio = ratio / (10 ** power)
            if 0.8 <= test_ratio <= 1.2:
                analysis['diagnosis'] = f"Extra 10^{power} factor (needs ÷10^{power})"
                analysis['correction'] = 1.0 / (10 ** power)
                break
    
    else:
        analysis['diagnosis'] = f"Unclear (ratio={ratio:.3f})"
        analysis['correction'] = 1.0
    
    return analysis['diagnosis'], analysis['correction'], analysis


def analyze_model(model_dir, model_name, n_samples=10):
    """Analyze a model directory."""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")
    
    # Find lookup table
    lookup_file = os.path.join(model_dir, 'lookup_table.csv')
    if not os.path.exists(lookup_file):
        print(f"  No lookup_table.csv found")
        return
    
    # Read lookup table
    import pandas as pd
    try:
        df = pd.read_csv(lookup_file, comment='#')
    except:
        print(f"  Could not read lookup_table.csv")
        return
    
    print(f"  Total spectra: {len(df)}")
    
    # Sample spectra
    n = min(n_samples, len(df))
    sample_indices = np.linspace(0, len(df)-1, n, dtype=int)
    
    results = []
    
    for idx in sample_indices:
        row = df.iloc[idx]
        
        # Get filename and parameters
        filename = row.get('file_name', row.get('filename', None))
        if filename is None:
            continue
        
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            continue
        
        # Get Teff
        teff = None
        for key in ['teff', 'Teff', 'T_eff']:
            if key in row:
                try:
                    teff = float(row[key])
                    break
                except:
                    pass
        
        if teff is None or teff <= 0:
            continue
        
        # Read spectrum
        wl, flux, metadata = read_spectrum_with_header(filepath)
        
        if wl is None:
            continue
        
        # Analyze wavelength units
        wl_unit, wl_conversion = analyze_wavelength_units(wl)
        
        # Convert wavelength to Angstroms if needed
        wl_angstrom = wl * wl_conversion if wl_conversion else wl
        
        # Analyze flux units
        diagnosis, correction, analysis = analyze_flux_units(wl_angstrom, flux, teff, metadata)
        
        results.append({
            'filename': filename,
            'teff': teff,
            'wl_unit': wl_unit,
            'wl_conversion': wl_conversion,
            'diagnosis': diagnosis,
            'correction': correction,
            'ratio': analysis['ratio'] if analysis else None
        })
    
    if not results:
        print("  No valid spectra analyzed")
        return
    
    # Print results
    print(f"\n  Analyzed {len(results)} spectra:")
    print(f"\n  {'File':<40} {'Teff':>8} {'λ unit':<12} {'Ratio':>10} {'Diagnosis':<40}")
    print(f"  {'-'*120}")
    
    for r in results:
        ratio_str = f"{r['ratio']:.3f}" if r['ratio'] is not None else "N/A"
        print(f"  {r['filename']:<40} {r['teff']:>8.0f} {r['wl_unit']:<12} {ratio_str:>10} {r['diagnosis']:<40}")
    
    # Summary
    corrections = [r['correction'] for r in results if r['correction'] != 1.0]
    if corrections:
        median_corr = np.median(corrections)
        print(f"\n  SUMMARY:")
        print(f"    Wavelength units: {results[0]['wl_unit']} (×{results[0]['wl_conversion']} to Angstrom)")
        print(f"    Flux correction needed: ×{median_corr:.4f}")
        print(f"    Most common diagnosis: {results[0]['diagnosis']}")
    else:
        print(f"\n  SUMMARY: Units appear correct (no correction needed)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_units.py <data_dir>")
        print("Example: python analyze_units.py ../data/stellar_models")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    # Find all model directories
    models = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            lookup_file = os.path.join(item_path, 'lookup_table.csv')
            if os.path.exists(lookup_file):
                models.append((item, item_path))
    
    print(f"Found {len(models)} models with lookup tables")
    
    for name, path in models:
        analyze_model(path, name)
