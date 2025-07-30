# stellar-colors

[![PyPI version](https://badge.fury.io/py/stellar-colors.svg)](https://badge.fury.io/py/stellar-colors)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Astropy](https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

A comprehensive Python package for stellar atmosphere modeling and synthetic photometry, fully integrated with the Astropy ecosystem.

## Features

**Stellar Atmosphere Models**
- Automated discovery and download from SVO Theoretical Spectra Database
- Support for multiple model grids (KURUCZ, PHOENIX, ATLAS, etc.)
- Intelligent parameter space analysis and validation

**Photometric Filters**
- Access to 10,000+ filter transmission curves via SVO Filter Profile Service
- Organized collections by facility and instrument
- Custom filter collections for specific projects

**Data Cubes & Interpolation**
- Build 4D flux cubes (Teff, log g, [M/H], wavelength) for fast interpolation
- Multiple interpolation methods (linear, nearest, cubic)
- Memory-efficient HDF5 storage with compression

**Synthetic Photometry**
- Compute synthetic magnitudes in any photometric system
- Bolometric corrections and luminosity calculations
- Support for both Vega and AB magnitude systems

**Performance Optimized**
- Parallel downloads and processing
- Intelligent caching and data management
- Astropy integration for units and coordinate handling

## Installation

```bash
# From PyPI (recommended)
pip install stellar-colors

# Development version from GitHub
pip install git+https://github.com/yourusername/stellar-colors.git

# With all optional dependencies
pip install stellar-colors[dev,docs]
```

## Quick Start

```python
import stellar_colors as sc
from astropy import units as u

# 1. Discover and download stellar atmosphere models
models = sc.discover_models()
print(f"Available models: {models[:5]}")

# Download KURUCZ models
model_dir = sc.download_model_grid('KURUCZ2003', max_spectra=100)

# 2. Download photometric filters
filters = sc.discover_filters(facility='HST')
filter_dir = sc.download_filter_collection('HST_Collection', ['HST'])

# 3. Build a flux cube for fast interpolation
cube_file = sc.build_flux_cube(model_dir, 'kurucz_cube.h5')

# 4. Compute synthetic photometry
photometry = sc.SyntheticPhotometry(cube_file, filter_dir)

# Compute V-band magnitude for a solar-type star
v_mag = photometry.compute_magnitude(
    teff=5777,      # K
    logg=4.44,      # log g
    metallicity=0.0, # [M/H]
    filter_id='Generic/Johnson/V',
    distance=10.0,   # pc (for absolute magnitude)
    radius=1.0       # solar radii
)

print(f"V-band magnitude: {v_mag:.2f}")

# Compute color index
b_v_color = photometry.compute_color(
    5777, 4.44, 0.0, 
    'Generic/Johnson/B', 'Generic/Johnson/V',
    distance=10.0, radius=1.0
)

print(f"B-V color: {b_v_color:.3f}")
```

## Detailed Usage

### Working with Stellar Atmosphere Models

```python
# Discover available models
grabber = sc.AtmosphereGrabber()
models = grabber.discover_models()

# Get detailed information about a model
info = grabber.get_model_info('KURUCZ2003')
print(f"Model has {info['n_spectra']} spectra")
print(f"Teff range: {info['parameter_ranges']['teff']['range']}")

# Download with filtering
model_dir = grabber.download_model(
    'KURUCZ2003',
    parameter_filter={
        'teff': (4000, 8000),
        'logg': (3.5, 5.0),
        'meta': (-1.0, 0.5)
    },
    show_progress=True
)

# Validate the downloaded collection
validation = sc.validate_model_collection(model_dir)
if validation['valid']:
    print("Model collection is valid!")
else:
    print(f"Issues found: {validation['issues']}")
```

### Working with Photometric Filters

```python
# Discover filters
grabber = sc.FilterGrabber()
facilities = grabber.discover_facilities()
print(f"Available facilities: {facilities[:10]}")

# Search for specific filters
hst_filters = grabber.search_filters(facility='HST', instrument='WFC3')
optical_filters = grabber.search_filters(
    wavelength_range=(4000*u.AA, 7000*u.AA)
)

# Download facility filters
hst_dir = grabber.download_facility_filters('HST')

# Create custom filter collection
custom_collection = [
    {'facility': 'HST', 'instrument': 'WFC3'},
    {'facility': 'Gaia'},
    {'wavelength_range': (5000*u.AA, 6000*u.AA)}
]

collection_dir = grabber.download_filter_collection(
    'MyProject', custom_collection
)
```

### Building and Using Flux Cubes

```python
# Build a flux cube
builder = sc.DataCubeBuilder(model_dir)

# Analyze the parameter space
analysis = builder.analyze_grid_structure()
print(f"Teff coverage: {analysis['teff']['n_unique']} points")

# Build cube with custom wavelength range
cube_file = builder.build_cube(
    'stellar_cube.h5',
    wavelength_range=(3000, 10000),  # Angstroms
    compression=True,
    fill_missing=True
)

# Load and use the cube
cube = sc.FluxCube(cube_file)
print(f"Cube shape: {cube.shape}")
print(f"Parameter ranges: {cube.parameter_ranges}")

# Interpolate a spectrum
wavelengths, fluxes = cube.interpolate_spectrum(
    teff=5500, logg=4.0, metallicity=0.2
)

# Interpolate at specific wavelengths
target_waves = [5000, 5500, 6000]  # Angstroms
fluxes = cube.interpolate_at_wavelength(
    target_waves, teff=5500, logg=4.0, metallicity=0.2
)
```

### Advanced Photometry

```python
# Initialize photometry system
photometry = sc.SyntheticPhotometry(cube_file, filter_dir)

# Compute magnitude grid for parameter study
grid_results = photometry.compute_magnitude_grid(
    teff_range=(4000, 8000),
    logg_range=(3.0, 5.0),
    metallicity_range=(-1.0, 0.5),
    filter_id='Generic/Johnson/V',
    n_teff=20, n_logg=10, n_metallicity=5
)

# Extract results
teff_grid = grid_results['teff_grid']
magnitudes = grid_results['magnitudes']

# Bolometric corrections
bolometric = sc.BolometricCorrections(cube_file)

# Compute bolometric magnitude
m_bol = bolometric.compute_bolometric_magnitude(
    5777, 4.44, 0.0, distance=10.0, radius=1.0
)

# Compute luminosity
luminosity = bolometric.compute_luminosity(
    5777, 4.44, 0.0, radius=1.0
)
print(f"Luminosity: {luminosity:.3f} L_sun")

# Bolometric correction
bc_v = bolometric.compute_bolometric_correction(
    5777, 4.44, 0.0, 'Generic/Johnson/V', photometry
)
print(f"BC_V: {bc_v:.3f}")
```

## Configuration

The package uses Astropy's configuration system:

```python
import stellar_colors as sc

# View current configuration
print(sc.conf)

# Modify settings
sc.conf.max_download_workers = 10
sc.conf.default_interpolation_method = 'cubic'
sc.conf.magnitude_system = 'ab'

# Get data directories
data_dir = sc.conf.get_data_dir()
models_dir = sc.conf.get_models_dir()
```

## Performance Tips

1. **Use flux cubes for repeated calculations**:
   ```python
   # Build once, use many times
   cube_file = sc.build_flux_cube(model_dir, 'my_cube.h5')
   photometry = sc.SyntheticPhotometry(cube_file, filter_dir)
   ```

2. **Enable parallel processing**:
   ```python
   sc.conf.enable_parallel_processing = True
   sc.conf.max_download_workers = 8
   ```

3. **Use appropriate interpolation methods**:
   ```python
   # 'linear' for accuracy, 'nearest' for speed
   cube.interpolate_spectrum(5777, 4.44, 0.0, method='linear')
   ```

4. **Optimize wavelength ranges**:
   ```python
   # Only include wavelengths you need
   builder = sc.DataCubeBuilder(
       model_dir, 
       wavelength_range=(4000, 9000)  # Skip UV and far-IR
   )
   ```

## Examples

See the `examples/` directory for complete workflows:

- `basic_photometry.py` - Computing synthetic magnitudes
- `color_evolution.py` - Stellar evolution color tracks  
- `survey_simulation.py` - Large-scale survey simulation
- `custom_filters.py` - Working with custom filter sets
- `bolometric_corrections.py` - Computing BC tables

## API Reference

### Core Classes

- **`AtmosphereGrabber`** - Download stellar atmosphere models
- **`FilterGrabber`** - Download photometric filters
- **`DataCubeBuilder`** - Build interpolatable flux cubes
- **`FluxCube`** - Query precomputed flux cubes
- **`SyntheticPhotometry`** - Compute synthetic magnitudes
- **`BolometricCorrections`** - Bolometric quantities

### Convenience Functions

- **`discover_models()`** - List available atmosphere models
- **`download_model_grid()`** - Download a model collection
- **`discover_filters()`** - Search for photometric filters
- **`build_flux_cube()`** - Create a flux cube from models
- **`compute_synthetic_magnitude()`** - One-off magnitude calculation

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/stellar-colors.git
cd stellar-colors

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black stellar_colors/
isort stellar_colors/

# Type checking
mypy stellar_colors/
```

## Support

- **Documentation**: https://stellar-colors.readthedocs.io
- **Issues**: https://github.com/yourusername/stellar-colors/issues
- **Discussions**: https://github.com/yourusername/stellar-colors/discussions

## Citation

If you use stellar-colors in your research, please cite:

```bibtex
@software{stellar_colors,
  author = {{Stellar Colors Development Team}},
  title = {stellar-colors: Stellar atmosphere modeling and synthetic photometry},
  url = {https://github.com/yourusername/stellar-colors},
  version = {1.0.0},
  year = {2024}
}
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **SVO Team** for providing the theoretical spectra and filter databases
- **Astropy Project** for the foundational astronomy tools
- **MESA Team** for inspiration from the original colors module
- **Community contributors** for feedback and improvements

---

**stellar-colors**: Making stellar atmosphere modeling and synthetic photometry accessible to the astronomy community! 🌟

---

# examples/basic_photometry.py
"""
Basic synthetic photometry example.

This example demonstrates how to:
1. Set up stellar atmosphere models and filters
2. Build a flux cube for interpolation
3. Compute synthetic magnitudes and colors
"""

import stellar_colors as sc
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Stellar Colors - Basic Photometry Example")
    print("=" * 50)
    
    # 1. Download a small model grid for testing
    print("1. Setting up stellar atmosphere models...")
    
    grabber = sc.AtmosphereGrabber()
    
    # Download KURUCZ models (limit for demo)
    model_dir = grabber.download_model(
        'KURUCZ2003',
        max_spectra=50,  # Small sample for demo
        parameter_filter={
            'teff': (4000, 8000),
            'logg': (3.5, 5.0),
            'meta': (-0.5, 0.5)
        }
    )
    
    print(f"Downloaded models to: {model_dir}")
    
    # 2. Download some standard filters
    print("\n2. Setting up photometric filters...")
    
    filter_grabber = sc.FilterGrabber()
    
    # Download Johnson-Cousins filters
    johnson_filters = filter_grabber.search_filters(facility='Generic')
    print(f"Found {len(johnson_filters)} Generic filters")
    
    filter_dir = filter_grabber.download_facility_filters('Generic')
    print(f"Downloaded filters to: {filter_dir}")
    
    # 3. Build flux cube
    print("\n3. Building flux cube...")
    
    cube_file = sc.build_flux_cube(
        model_dir, 
        'demo_cube.h5',
        wavelength_range=(3500, 9000)  # Optical + near-IR
    )
    
    print(f"Built flux cube: {cube_file}")
    
    # 4. Initialize photometry system
    print("\n4. Computing synthetic photometry...")
    
    photometry = sc.SyntheticPhotometry(cube_file, filter_dir)
    
    # List available filters
    filters = photometry.list_filters()
    print(f"Available filters: {filters[:5]}...")  # Show first 5
    
    # 5. Compute magnitudes for different stellar types
    stellar_types = [
        ('M5V', 3500, 5.0, 0.0),
        ('K5V', 4500, 4.5, 0.0),
        ('G2V', 5777, 4.44, 0.0),  # Sun
        ('F5V', 6500, 4.0, 0.0),
        ('A5V', 8000, 4.0, 0.0)
    ]
    
    print("\nSynthetic magnitudes (absolute, 10 pc):")
    print("Type    Teff   logg  [M/H]    V      B-V    V-I")
    print("-" * 50)
    
    results = []
    for spec_type, teff, logg, meta in stellar_types:
        try:
            # Compute magnitudes
            v_mag = photometry.compute_magnitude(
                teff, logg, meta, 'Generic/Johnson/V',
                distance=10.0, radius=1.0  # 10 pc, 1 solar radius
            )
            
            b_v = photometry.compute_color(
                teff, logg, meta,
                'Generic/Johnson/B', 'Generic/Johnson/V',
                distance=10.0, radius=1.0
            )
            
            v_i = photometry.compute_color(
                teff, logg, meta,
                'Generic/Johnson/V', 'Generic/Cousins/I',
                distance=10.0, radius=1.0
            )
            
            print(f"{spec_type:<6} {teff:5.0f} {logg:5.2f} {meta:5.1f} "
                  f"{v_mag:7.2f} {b_v:6.3f} {v_i:6.3f}")
            
            results.append((spec_type, teff, v_mag, b_v, v_i))
            
        except Exception as e:
            print(f"{spec_type:<6} - Error: {e}")
    
    # 6. Create color-magnitude diagram
    print("\n5. Creating color-magnitude diagram...")
    
    if len(results) > 1:
        spec_types, teffs, v_mags, b_vs, v_is = zip(*results)
        
        plt.figure(figsize=(10, 6))
        
        # B-V vs V plot
        plt.subplot(1, 2, 1)
        plt.scatter(b_vs, v_mags, c=teffs, cmap='RdYlBu_r', s=100)
        plt.colorbar(label='Teff (K)')
        plt.xlabel('B - V')
        plt.ylabel('V (mag)')
        plt.title('Color-Magnitude Diagram')
        plt.gca().invert_yaxis()
        
        # Add labels
        for i, spec_type in enumerate(spec_types):
            plt.annotate(spec_type, (b_vs[i], v_mags[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # V-I vs B-V plot  
        plt.subplot(1, 2, 2)
        plt.scatter(b_vs, v_is, c=teffs, cmap='RdYlBu_r', s=100)
        plt.colorbar(label='Teff (K)')
        plt.xlabel('B - V')
        plt.ylabel('V - I')
        plt.title('Color-Color Diagram')
        
        # Add labels
        for i, spec_type in enumerate(spec_types):
            plt.annotate(spec_type, (b_vs[i], v_is[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig('color_diagrams.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Saved color diagrams to: color_diagrams.png")
    
    # 7. Bolometric corrections
    print("\n6. Computing bolometric corrections...")
    
    bolometric = sc.BolometricCorrections(cube_file)
    
    print("Bolometric corrections:")
    print("Type    Teff   BC_V   L_bol/L_sun")
    print("-" * 35)
    
    for spec_type, teff, logg, meta in stellar_types[:3]:  # Just first 3
        try:
            bc_v = bolometric.compute_bolometric_correction(
                teff, logg, meta, 'Generic/Johnson/V', photometry
            )
            
            luminosity = bolometric.compute_luminosity(
                teff, logg, meta, radius=1.0
            )
            
            print(f"{spec_type:<6} {teff:5.0f} {bc_v:6.3f} {luminosity:10.3f}")
            
        except Exception as e:
            print(f"{spec_type:<6} - Error: {e}")
    
    print("\nBasic photometry example completed!")
    print("Check the generated files:")
    print(f"- Flux cube: {cube_file}")
    print(f"- Models: {model_dir}")
    print(f"- Filters: {filter_dir}")
    print("- color_diagrams.png")

if __name__ == "__main__":
    main()


# examples/survey_simulation.py
"""
Large-scale survey simulation example.

This example demonstrates how to:
1. Create synthetic observations for a large stellar population
2. Add realistic photometric errors
3. Compare with real survey data
"""

import stellar_colors as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def generate_stellar_population(n_stars=10000):
    """Generate realistic stellar population."""
    
    # Mass function (Salpeter-like)
    masses = np.random.pareto(2.35, n_stars) * 0.1 + 0.1
    masses = np.clip(masses, 0.1, 10.0)  # 0.1 to 10 solar masses
    
    # Mass-Teff relation (approximate main sequence)
    teff = 5780 * (masses ** 0.5)  # Very approximate
    teff = np.clip(teff, 3000, 30000)
    
    # Mass-logg relation
    logg = 4.44 + 0.4 * np.log10(masses)  # Approximate
    logg = np.clip(logg, 3.0, 5.5)
    
    # Metallicity distribution (local disk)
    metallicity = np.random.normal(-0.1, 0.2, n_stars)
    metallicity = np.clip(metallicity, -2.0, 0.5)
    
    # Distance distribution (exponential disk)
    distances = np.random.exponential(1000, n_stars)  # pc
    distances = np.clip(distances, 10, 5000)
    
    # Stellar radii (mass-radius relation)
    radii = masses ** 0.8  # Approximate main sequence
    
    return pd.DataFrame({
        'mass': masses,
        'teff': teff,
        'logg': logg,
        'metallicity': metallicity,
        'distance': distances,
        'radius': radii
    })

def add_photometric_errors(magnitudes, survey_type='ground'):
    """Add realistic photometric errors."""
    
    if survey_type == 'ground':
        # Ground-based survey errors
        bright_limit = 12.0
        faint_limit = 22.0
        min_error = 0.01
        max_error = 0.3
    elif survey_type == 'space':
        # Space-based survey errors (e.g., Gaia)
        bright_limit = 6.0
        faint_limit = 20.0
        min_error = 0.001
        max_error = 0.1
    
    # Error model: increases with magnitude
    errors = min_error * 10**((magnitudes - bright_limit) / 5.0)
    errors = np.clip(errors, min_error, max_error)
    
    # Add errors to magnitudes
    noisy_mags = magnitudes + np.random.normal(0, errors)
    
    # Create detection flags
    detected = (magnitudes > bright_limit - 2) & (magnitudes < faint_limit)
    
    return noisy_mags, errors, detected

def main():
    print("Stellar Colors - Survey Simulation Example")
    print("=" * 50)
    
    # Setup (assuming we have models and filters already)
    print("1. Setting up photometry system...")
    
    # Use pre-built cube (you'd build this first)
    cube_file = 'demo_cube.h5'
    filter_dir = 'filters/Generic'
    
    try:
        photometry = sc.SyntheticPhotometry(cube_file, filter_dir)
    except FileNotFoundError:
        print("Error: Run basic_photometry.py first to create the flux cube and filters")
        return
    
    # 2. Generate stellar population
    print("2. Generating stellar population...")
    
    n_stars = 5000  # Reduced for demo
    population = generate_stellar_population(n_stars)
    
    print(f"Generated {n_stars} stars:")
    print(f"  Mass range: {population['mass'].min():.2f} - {population['mass'].max():.2f} M_sun")
    print(f"  Teff range: {population['teff'].min():.0f} - {population['teff'].max():.0f} K")
    print(f"  Distance range: {population['distance'].min():.0f} - {population['distance'].max():.0f} pc")
    
    # 3. Compute synthetic photometry
    print("3. Computing synthetic photometry...")
    
    # Define filter set
    filters = ['Generic/Johnson/B', 'Generic/Johnson/V', 'Generic/Cousins/I']
    
    # Initialize results arrays
    magnitudes = {}
    for filt in filters:
        magnitudes[filt] = np.full(n_stars, np.nan)
    
    # Compute magnitudes for each star
    valid_count = 0
    for i, row in population.iterrows():
        if i % 1000 == 0:
            print(f"  Processing star {i+1}/{n_stars}")
        
        try:
            # Check if parameters are within model grid
            cube = photometry.flux_cube
            ranges = cube.parameter_ranges
            
            if not (ranges['teff'][0] <= row['teff'] <= ranges['teff'][1]):
                continue
            if not (ranges['logg'][0] <= row['logg'] <= ranges['logg'][1]):
                continue
            if not (ranges['metallicity'][0] <= row['metallicity'] <= ranges['metallicity'][1]):
                continue
            
            # Compute magnitudes in each filter
            for filt in filters:
                mag = photometry.compute_magnitude(
                    row['teff'], row['logg'], row['metallicity'], filt,
                    distance=row['distance'], radius=row['radius']
                )
                magnitudes[filt][i] = mag
            
            valid_count += 1
            
        except Exception as e:
            continue  # Skip problematic stars
    
    print(f"Successfully computed photometry for {valid_count}/{n_stars} stars")
    
    # 4. Add observational effects
    print("4. Adding observational effects...")
    
    survey_results = {}
    
    for survey_name, survey_type in [('Ground-based', 'ground'), ('Space-based', 'space')]:
        print(f"  Simulating {survey_name} survey...")
        
        survey_mags = {}
        survey_errors = {}
        survey_detected = {}
        
        for filt in filters:
            valid_mask = np.isfinite(magnitudes[filt])
            if not np.any(valid_mask):
                continue
                
            # Add errors only to valid magnitudes
            noisy_mags = magnitudes[filt].copy()
            errors = np.full_like(magnitudes[filt], np.nan)
            detected = np.zeros_like(magnitudes[filt], dtype=bool)
            
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                noisy_valid, errors_valid, detected_valid = add_photometric_errors(
                    magnitudes[filt][valid_indices], survey_type
                )
                
                noisy_mags[valid_indices] = noisy_valid
                errors[valid_indices] = errors_valid
                detected[valid_indices] = detected_valid
            
            survey_mags[filt] = noisy_mags
            survey_errors[filt] = errors
            survey_detected[filt] = detected
        
        survey_results[survey_name] = {
            'magnitudes': survey_mags,
            'errors': survey_errors,
            'detected': survey_detected
        }
    
    # 5. Create diagnostic plots
    print("5. Creating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: HR diagram
    ax = axes[0, 0]
    valid = np.isfinite(magnitudes['Generic/Johnson/B']) & np.isfinite(magnitudes['Generic/Johnson/V'])
    if np.any(valid):
        b_v = magnitudes['Generic/Johnson/B'][valid] - magnitudes['Generic/Johnson/V'][valid]
        v_mag = magnitudes['Generic/Johnson/V'][valid]
        
        ax.scatter(b_v, v_mag, alpha=0.5, s=1)
        ax.set_xlabel('B - V')
        ax.set_ylabel('V (mag)')
        ax.set_title('Synthetic HR Diagram')
        ax.invert_yaxis()
    
    # Plot 2: Color-color diagram
    ax = axes[0, 1]
    valid = (np.isfinite(magnitudes['Generic/Johnson/B']) & 
             np.isfinite(magnitudes['Generic/Johnson/V']) & 
             np.isfinite(magnitudes['Generic/Cousins/I']))
    if np.any(valid):
        b_v = magnitudes['Generic/Johnson/B'][valid] - magnitudes['Generic/Johnson/V'][valid]
        v_i = magnitudes['Generic/Johnson/V'][valid] - magnitudes['Generic/Cousins/I'][valid]
        
        ax.scatter(b_v, v_i, alpha=0.5, s=1)
        ax.set_xlabel('B - V')
        ax.set_ylabel('V - I')
        ax.set_title('Color-Color Diagram')
    
    # Plot 3: Distance vs magnitude
    ax = axes[0, 2]
    valid = np.isfinite(magnitudes['Generic/Johnson/V'])
    if np.any(valid):
        ax.scatter(population['distance'][valid], magnitudes['Generic/Johnson/V'][valid], 
                  alpha=0.5, s=1)
        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('V (mag)')
        ax.set_title('Distance vs Magnitude')
        ax.set_xscale('log')
    
    # Plot 4-6: Survey comparisons
    for i, (survey_name, survey_data) in enumerate(survey_results.items()):
        ax = axes[1, i]
        
        # Plot detected vs undetected
        v_mags = survey_data['magnitudes']['Generic/Johnson/V']
        detected = survey_data['detected']['Generic/Johnson/V']
        
        if np.any(detected):
            ax.hist(v_mags[detected], bins=30, alpha=0.7, label='Detected', density=True)
        if np.any(~detected & np.isfinite(v_mags)):
            ax.hist(v_mags[~detected & np.isfinite(v_mags)], bins=30, alpha=0.7, 
                   label='Missed', density=True)
        
        ax.set_xlabel('V magnitude')
        ax.set_ylabel('Normalized count')
        ax.set_title(f'{survey_name} Detection')
        ax.legend()
    
    # Hide unused subplot
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('survey_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 6. Generate summary statistics
    print("\n6. Survey simulation summary:")
    print("-" * 40)
    
    for survey_name, survey_data in survey_results.items():
        print(f"\n{survey_name} Survey:")
        
        for filt in filters:
            detected = survey_data['detected'][filt]
            if np.any(detected):
                n_detected = np.sum(detected)
                detection_rate = n_detected / valid_count * 100
                
                errors = survey_data['errors'][filt][detected]
                median_error = np.median(errors)
                
                print(f"  {filt.split('/')[-1]} band:")
                print(f"    Detected: {n_detected}/{valid_count} ({detection_rate:.1f}%)")
                print(f"    Median error: {median_error:.3f} mag")
    
    # 7. Export results
    print("\n7. Exporting results...")
    
    # Create output DataFrame
    output_data = population.copy()
    
    # Add synthetic magnitudes
    for filt in filters:
        band_name = filt.split('/')[-1]
        output_data[f'{band_name}_true'] = magnitudes[filt]
        
        # Add survey results
        for survey_name, survey_data in survey_results.items():
            survey_short = survey_name.replace('-based', '').replace(' ', '_').lower()
            output_data[f'{band_name}_{survey_short}'] = survey_data['magnitudes'][filt]
            output_data[f'{band_name}_{survey_short}_err'] = survey_data['errors'][filt]
            output_data[f'{band_name}_{survey_short}_det'] = survey_data['detected'][filt]
    
    # Save to file
    output_file = 'synthetic_survey.csv'
    output_data.to_csv(output_file, index=False)
    
    print(f"Results saved to: {output_file}")
    print("Plots saved to: survey_simulation.png")
    
    print("\nSurvey simulation completed!")

if __name__ == "__main__":
    main()