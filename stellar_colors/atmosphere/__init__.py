"""
Stellar atmosphere module for stellar-colors package.

This module provides tools for downloading, managing, and interpolating
stellar atmosphere model grids for synthetic photometry calculations.
"""

from .grabber import SVOSpectraGrabber, download_atmosphere_models
from .models import (
    AtmosphereModel,
    KuruczModel, 
    PhoenixModel,
    AtlasModel,
    load_atmosphere_model
)
from .interpolation import (
    LinearInterpolator,
    HermiteInterpolator, 
    KNNInterpolator,
    interpolate_spectrum
)

__all__ = [
    # Grabber functions
    'SVOSpectraGrabber',
    'download_atmosphere_models',
    
    # Model classes
    'AtmosphereModel',
    'KuruczModel',
    'PhoenixModel', 
    'AtlasModel',
    'load_atmosphere_model',
    
    # Interpolation functions
    'LinearInterpolator',
    'HermiteInterpolator',
    'KNNInterpolator', 
    'interpolate_spectrum',
]