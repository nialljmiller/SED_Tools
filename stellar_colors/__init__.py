# stellar_colors/__init__.py
"""
Stellar Colors: A comprehensive package for stellar atmosphere modeling and synthetic photometry.

This package provides tools for:
- Downloading stellar atmosphere models and filter transmission curves
- Building interpolatable data cubes from model collections
- Computing synthetic photometry and bolometric corrections
- Creating visualizations of parameter spaces and flux cubes
- Integration with astropy for astronomical applications

Examples
--------
Basic usage for synthetic photometry:

>>> import stellar_colors as sc
>>> from astropy import units as u

# Download some stellar atmosphere models
>>> models = sc.discover_models()
>>> sc.download_model_grid('KURUCZ2003') 

# Download photometric filters  
>>> filters = sc.discover_filters(facility='HST')
>>> sc.download_filter_collection('HST_Collection', ['HST'])

# Build a flux cube for fast interpolation
>>> cube_file = sc.build_flux_cube('models/KURUCZ2003/', 'kurucz_cube.h5')

# Create visualizations
>>> cube = sc.FluxCube(cube_file)
>>> sc.plot_parameter_space_3d(cube.teff_grid, cube.logg_grid, cube.meta_grid)
>>> sc.create_comprehensive_plots(cube, 'plots/')

# Compute synthetic photometry
>>> photometry = sc.SyntheticPhotometry(cube_file, 'filters/HST_Collection/')
>>> magnitude = photometry.compute_magnitude(5777, 4.44, 0.0, 'HST/WFC3/F555W')
"""

# Version information
from .version import __version__

# Core imports - make main functionality easily accessible
from .atmosphere.grabber import AtmosphereGrabber, discover_models, download_model_grid
from .filters.grabber import FilterGrabber, discover_filters, download_filter_collection
from .cube.builder import DataCubeBuilder, FluxCube, build_flux_cube

# Plotting functionality
from .plotting import (
    plot_parameter_space_2d,
    plot_parameter_space_3d,
    plot_flux_cube_analysis,
    plot_spectrum_comparison,
    plot_model_coverage,
    create_comprehensive_plots
)

#from .photometry.synthetic import SyntheticPhotometry, compute_synthetic_magnitude
#from .photometry.bolometric import BolometricCorrections, compute_bolometric_correction

# Configuration
from .config import conf

# Make key classes and functions available at package level
__all__ = [
    # Version
    '__version__',
    
    # Main classes
    'AtmosphereGrabber',
    'FilterGrabber', 
    'DataCubeBuilder',
    'FluxCube',
    'SyntheticPhotometry',
    'BolometricCorrections',
    
    # Convenience functions
    'discover_models',
    'download_model_grid',
    'discover_filters',
    'download_filter_collection',
    'build_flux_cube',
    'compute_synthetic_magnitude',
    'compute_bolometric_correction',
    
    # Plotting functions
    'plot_parameter_space_2d',
    'plot_parameter_space_3d', 
    'plot_flux_cube_analysis',
    'plot_spectrum_comparison',
    'plot_model_coverage',
    'create_comprehensive_plots',
    
    # Configuration
    'conf',
]

# Package metadata
__author__ = "Stellar Colors Development Team"
__email__ = "stellar-colors@example.com"
__license__ = "BSD-3-Clause"
__description__ = "Stellar atmosphere modeling and synthetic photometry for astronomy"