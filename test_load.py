#!/usr/bin/env python3
"""
Simple diagnostic - show exactly why files fail to load
"""
import os
import numpy as np
import pandas as pd

def test_one_file(model_dir, model_name):
    """Try to load just ONE file from a model and show what goes wrong."""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"Path: {model_dir}")
    
    # Check lookup table
    lookup_file = os.path.join(model_dir, 'lookup_table.csv')
    if not os.path.exists(lookup_file):
        print("  ✗ No lookup_table.csv")
        return
    
    print(f"  ✓ Found lookup_table.csv")
    
    # Read it
    try:
        with open(lookup_file, 'r') as f:
            first_line = f.readline().strip()
        
        if first_line.startswith('#'):
            df = pd.read_csv(lookup_file, skiprows=0)
            df.columns = [col.lstrip('#').strip() for col in df.columns]
        else:
            df = pd.read_csv(lookup_file)
        
        print(f"  ✓ Read lookup table: {len(df)} entries")
        print(f"    Columns: {list(df.columns)}")
    except Exception as e:
        print(f"  ✗ Could not read lookup table: {e}")
        return
    
    # Get first row
    if len(df) == 0:
        print("  ✗ Lookup table is empty")
        return
    
    row = df.iloc[0]
    print(f"\n  First row data:")
    for col in df.columns:
        print(f"    {col}: {row[col]}")
    
    # Try to find filename
    filename = None
    for col in ['file_name', 'filename', 'file']:
        if col in df.columns:
            filename = row[col]
            print(f"\n  ✓ Found filename in column '{col}': {filename}")
            break
    
    if filename is None:
        print(f"\n  ✗ No filename column found (tried: file_name, filename, file)")
        return
    
    # Check if file exists
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        print(f"  ✗ File does not exist: {filepath}")
        
        # List what IS in the directory
        files = os.listdir(model_dir)
        txt_files = [f for f in files if f.endswith('.txt')][:5]
        print(f"\n  Files in directory (showing first 5 .txt):")
        for f in txt_files:
            print(f"    - {f}")
        return
    
    print(f"  ✓ File exists: {filepath}")
    
    # Try to read it
    print(f"\n  Attempting to read file...")
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()[:20]
        
        print(f"  ✓ File readable, {len(lines)} lines (showing first 20):")
        for i, line in enumerate(lines[:10], 1):
            print(f"    {i}: {line.rstrip()}")
        
        # Try to load as numpy
        print(f"\n  Attempting numpy load...")
        data = np.loadtxt(filepath, unpack=True)
        print(f"  ✓ Numpy loaded: shape = {data.shape}")
        
        if data.ndim == 2:
            wl = data[0]
            flux = data[1]
            print(f"    Wavelength: {len(wl)} points, range [{wl.min():.2f}, {wl.max():.2f}]")
            print(f"    Flux: {len(flux)} points, range [{flux.min():.2e}, {flux.max():.2e}]")
        else:
            print(f"  ✗ Unexpected shape: {data.shape}")
        
    except Exception as e:
        print(f"  ✗ Failed to read: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_load.py <data_dir>")
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
    
    # Test each one
    for name, path in models:
        test_one_file(path, name)
