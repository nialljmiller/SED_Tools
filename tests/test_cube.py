#!/usr/bin/env python3
"""
Fixed integration test for stellar atmosphere models:
1. Display all available atmosphere tables
2. Download 10% of Kurucz2003 models (actually working now)
3. Create data cube
4. Create 2D and 3D plots using package plotting functionality
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import stellar_colors as sc


def test_comprehensive_sed_workflow():
    """Fixed comprehensive integration test for SED workflow."""
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        models_dir = temp_path / "models"
        plots_dir = temp_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("FIXED COMPREHENSIVE SED WORKFLOW TEST")
        print("=" * 80)
        
        # Step 1: Display all available atmosphere tables
        print("\n1. DISCOVERING AVAILABLE ATMOSPHERE MODELS")
        print("-" * 50)
        
        try:
            models = sc.discover_models()
            print(f"Found {len(models)} available atmosphere models:")
            print()
            
            # Display models in a nice table format
            for i, model in enumerate(models, 1):
                print(f"{i:2d}. {model}")
                
            print(f"\nTotal models available: {len(models)}")
            
        except Exception as e:
            print(f"Error discovering models: {e}")
            return False
        
        # Step 2: Download 10% of Kurucz2003 models (using fixed method)
        print("\n\n2. DOWNLOADING KURUCZ2003 SAMPLE (10%)")
        print("-" * 50)
        
        model_name = "Kurucz2003"
        
        try:
            # Use the fixed AtmosphereGrabber with working download method
            grabber = sc.AtmosphereGrabber(cache_dir=models_dir)
            
            # First discover all available spectra
            print(f"Discovering spectra for {model_name}...")
            spectra_info = grabber._discover_spectra(model_name)
            total_spectra = len(spectra_info)
            print(f"Found {total_spectra} total spectra")
            
            if total_spectra == 0:
                raise ValueError("No spectra found")
            
            # Download 10% sample (minimum 20, maximum 100 for reasonable test time)
            sample_size = max(20, min(100, int(total_spectra * 0.1)))
            print(f"Downloading sample of {sample_size} spectra ({100*sample_size/total_spectra:.1f}%)")
            
            # Download the sample using the FIXED download method
            model_dir = grabber.download_model(
                model_name,
                max_spectra=sample_size,
                show_progress=True
            )
            
            print(f"Downloaded to: {model_dir}")
            
            # Verify files were actually downloaded
            downloaded_files = list(model_dir.glob("*.txt"))
            lookup_file = model_dir / "lookup_table.csv"
            
            print(f"Verification:")
            print(f"  - Downloaded spectrum files: {len(downloaded_files)}")
            print(f"  - Lookup table exists: {lookup_file.exists()}")
            
            if len(downloaded_files) == 0:
                raise RuntimeError("No files were actually downloaded!")
                
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            print("CRITICAL: Download failed, cannot continue with data cube test")
            return False
        
        # Step 3: Create data cube
        print("\n\n3. CREATING FLUX DATA CUBE")
        print("-" * 50)
        
        cube_file = temp_path / "kurucz_test_cube.h5"
        
        try:
            # Initialize cube builder
            builder = sc.DataCubeBuilder(model_dir)
            
            # Analyze grid structure
            print("Analyzing model grid structure...")
            analysis = builder.analyze_grid_structure()
            
            for param, info in analysis.items():
                print(f"{param}: {info['n_unique']} unique values, "
                      f"range: {info['range'][0]:.1f} to {info['range'][1]:.1f}")
            
            # Build the cube
            print(f"\nBuilding flux cube: {cube_file}")
            cube_path = builder.build_cube(
                cube_file,
                wavelength_range=(3000, 10000),  # Optical + near-IR
                compression=True,
                show_progress=True
            )
            
            print(f"Flux cube created: {cube_path}")
            
            # Load the cube for analysis
            cube = sc.FluxCube(cube_path)
            print(f"Cube shape: {cube.flux_cube.shape}")
            
        except Exception as e:
            print(f"Error creating flux cube: {e}")
            return False
        
        # Step 4: Create visualizations using package plotting functions
        print("\n\n4. CREATING PARAMETER SPACE VISUALIZATIONS")
        print("-" * 50)
        
        try:
            # Use the package plotting functionality
            plot_files = sc.create_comprehensive_plots(
                cube, 
                plots_dir, 
                prefix="kurucz2003_test"
            )
            
            print(f"Plots created in: {plots_dir}")
            for plot_type, plot_file in plot_files.items():
                print(f"  - {plot_type}: {plot_file.name}")
            
            # Create additional focused plots
            
            # Get flattened parameter arrays for plotting
            teff_coords, logg_coords, meta_coords = np.meshgrid(
                cube.teff_grid, cube.logg_grid, cube.meta_grid, indexing='ij'
            )
            teff_flat = teff_coords.flatten()
            logg_flat = logg_coords.flatten()
            meta_flat = meta_coords.flatten()
            
            # Create detailed 3D plot
            print("Creating detailed 3D parameter space plot...")
            fig_3d = sc.plot_parameter_space_3d(
                teff_flat, logg_flat, meta_flat,
                color_by='teff',
                save_path=plots_dir / "detailed_3d_parameter_space.png"
            )
            
            # Create 2D projections
            print("Creating 2D parameter space projections...")
            fig_2d = sc.plot_parameter_space_2d(
                teff_flat, logg_flat, meta_flat,
                save_path=plots_dir / "parameter_space_projections.png"
            )
            
            # Create model coverage analysis
            print("Creating model coverage analysis...")
            lookup_table = pd.read_csv(model_dir / "lookup_table.csv", comment='#')
            fig_coverage = sc.plot_model_coverage(
                lookup_table,
                save_path=plots_dir / "model_coverage_analysis.png"
            )
            
            # Close figures to free memory
            import matplotlib.pyplot as plt
            plt.close('all')
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            return False
        
        # Step 5: Summary and validation
        print("\n\n5. WORKFLOW SUMMARY")
        print("-" * 50)
        
        success = True
        
        # Check downloaded models
        if model_dir.exists():
            model_files = list(model_dir.glob("*.txt"))
            lookup_file = model_dir / "lookup_table.csv"
            
            print(f"✓ Model directory: {model_dir}")
            print(f"✓ Downloaded spectrum files: {len(model_files)}")
            print(f"✓ Lookup table: {'EXISTS' if lookup_file.exists() else 'MISSING'}")
            
            if len(model_files) == 0:
                print("✗ ERROR: No spectrum files found!")
                success = False
        else:
            print("✗ ERROR: Model directory missing!")
            success = False
        
        # Check flux cube
        if cube_file.exists():
            size_mb = cube_file.stat().st_size / (1024**2)
            print(f"✓ Flux cube: {cube_file.name} ({size_mb:.1f} MB)")
        else:
            print("✗ ERROR: Flux cube missing!")
            success = False
        
        # Check plots
        plot_files = list(plots_dir.glob("*.png"))
        print(f"✓ Generated plots: {len(plot_files)}")
        for plot_file in plot_files:
            print(f"    - {plot_file.name}")
        
        if len(plot_files) == 0:
            print("✗ ERROR: No plots generated!")
            success = False
        
        # Final summary
        print("\n" + "=" * 80)
        if success:
            print("WORKFLOW COMPLETED SUCCESSFULLY!")
            print("✓ All steps completed without errors")
        else:
            print("WORKFLOW COMPLETED WITH ERRORS!")
            print("✗ Some steps failed - check output above")
        print("=" * 80)
        
        return success


def main():
    """Run the integration test."""
    success = test_comprehensive_sed_workflow()
    
    if success:
        print("\n🎉 Integration test PASSED!")
        return 0
    else:
        print("\n❌ Integration test FAILED!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())