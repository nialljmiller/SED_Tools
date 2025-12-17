#!/usr/bin/env python3
"""
Unit analysis based on ACTUAL header information
"""
import os
import numpy as np
import pandas as pd
from scipy.integrate import simpson

SIGMA = 5.670374419e-5  # erg s-1 cm-2 K-4

def analyze_file(filepath, teff):
    """Analyze one file and return unit info."""
    
    # Read header
    header_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('#'):
                header_lines.append(line.strip())
            else:
                break
    
    # Check if it's XML
    if any('<?xml' in line for line in header_lines[:5]):
        return None, "XML file (not readable)"
    
    # Find unit info from header
    flux_unit = None
    wl_unit = None
    
    for line in header_lines:
        if 'column 2:' in line.lower():
            # Extract flux unit
            if 'LUMINOSITY' in line:
                flux_unit = "LUMINOSITY (erg/s/Å)"
            elif 'FLUX' in line:
                if 'ERG/CM2/S/A' in line or 'ERG/CM²/S/Å' in line:
                    flux_unit = "FLUX (erg/cm²/s/Å)"
                else:
                    flux_unit = "FLUX (unknown units)"
        
        if 'column 1:' in line.lower():
            if 'ANGSTROM' in line:
                wl_unit = "Angstrom"
    
    # Load data
    try:
        data = np.loadtxt(filepath, unpack=True)
        if data.ndim != 2 or data.shape[0] < 2:
            return None, "Wrong data format"
        
        wl = data[0]
        flux = data[1]
        
        # Clean
        good = np.isfinite(wl) & np.isfinite(flux) & (wl > 0) & (flux > 0)
        if np.sum(good) < 10:
            return None, "No valid data points"
        
        wl = wl[good]
        flux = flux[good]
        
        # Compute ratio
        fbol = simpson(flux, wl)
        fbol_expected = SIGMA * teff**4
        ratio = fbol / fbol_expected
        
        return {
            'wl_unit': wl_unit or "unknown",
            'flux_unit': flux_unit or "unknown",
            'ratio': ratio,
            'wl_range': (wl.min(), wl.max()),
            'flux_range': (flux.min(), flux.max()),
            'n_points': len(wl)
        }, None
        
    except Exception as e:
        return None, f"Load failed: {str(e)}"


def analyze_model(model_dir, model_name, n_samples=5):
    """Analyze a model."""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")
    
    # Read lookup
    lookup_file = os.path.join(model_dir, 'lookup_table.csv')
    with open(lookup_file, 'r') as f:
        first_line = f.readline().strip()
    
    if first_line.startswith('#'):
        df = pd.read_csv(lookup_file, skiprows=0)
        df.columns = [col.lstrip('#').strip() for col in df.columns]
    else:
        df = pd.read_csv(lookup_file)
    
    # Sample files
    n = min(n_samples, len(df))
    sample_indices = np.linspace(0, len(df)-1, n, dtype=int)
    
    results = []
    for idx in sample_indices:
        row = df.iloc[idx]
        
        # Get filename
        filename = None
        for col in ['file_name', 'filename']:
            if col in df.columns:
                filename = row[col]
                break
        
        if filename is None:
            continue
        
        # Get teff
        teff = None
        for col in ['teff', 'Teff']:
            if col in df.columns:
                try:
                    teff = float(row[col])
                    if not np.isnan(teff) and teff > 0:
                        break
                except:
                    pass
        
        if teff is None:
            continue
        
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            continue
        
        result, error = analyze_file(filepath, teff)
        if error:
            print(f"  {filename}: {error}")
        else:
            results.append((filename, teff, result))
    
    if not results:
        print("  No valid files analyzed")
        return
    
    # Print results
    print(f"\n  Analyzed {len(results)} files:")
    print(f"  {'File':<30} {'Teff':>8} {'Wavelength':<12} {'Flux Unit':<30} {'Ratio':>10}")
    print(f"  {'-'*100}")
    
    for fname, teff, r in results:
        print(f"  {fname:<30} {teff:>8.0f} {r['wl_unit']:<12} {r['flux_unit']:<30} {r['ratio']:>10.3f}")
    
    # Summary
    print(f"\n  SUMMARY:")
    print(f"    Wavelength unit: {results[0][2]['wl_unit']}")
    print(f"    Flux unit: {results[0][2]['flux_unit']}")
    
    ratios = [r[2]['ratio'] for r in results]
    median_ratio = np.median(ratios)
    print(f"    Median ∫Fλ dλ / σT⁴ ratio: {median_ratio:.3f}")
    
    # Determine what's needed
    if 0.8 <= median_ratio <= 1.2:
        print(f"    ✓ Units are correct!")
    elif 0.25 <= median_ratio <= 0.35:
        print(f"    → Needs ×π correction (per steradian flux)")
    elif 0.075 <= median_ratio <= 0.085:
        print(f"    → Needs ×4π correction (isotropic flux)")
    elif median_ratio < 0.01:
        # Check if it's a power of 10
        for power in [2, 3, 4, 5]:
            test = median_ratio * (10 ** power)
            if 0.8 <= test <= 1.2:
                print(f"    → Needs ×{10**power} correction")
                break
        else:
            print(f"    → Needs ×{1.0/median_ratio:.2e} correction (unclear cause)")
    elif 'LUMINOSITY' in results[0][2]['flux_unit']:
        print(f"    ⚠ This is LUMINOSITY, not FLUX!")
        print(f"      Cannot be directly compared without distance/radius info")
    else:
        print(f"    ? Unclear (ratio={median_ratio:.3f})")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python unit_check.py <data_dir>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    # Find models
    models = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            lookup_file = os.path.join(item_path, 'lookup_table.csv')
            if os.path.exists(lookup_file):
                models.append((item, item_path))
    
    print(f"Found {len(models)} models\n")
    
    for name, path in models:
        try:
            analyze_model(path, name)
        except Exception as e:
            print(f"\nMODEL: {name}")
            print(f"  Error: {e}")
