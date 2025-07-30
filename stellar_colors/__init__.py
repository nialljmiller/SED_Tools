# stellar_colors/__init__.py
"""
Stellar Colors: A comprehensive package for stellar atmosphere modeling and synthetic photometry.

This package provides tools for:
- Downloading stellar atmosphere models and filter transmission curves
- Building interpolatable data cubes from model collections
- Computing synthetic photometry and bolometric corrections
- Integration with astropy for astronomical applications

Examples
--------
Basic usage for synthetic photometry:

>>> import stellar_colors as sc
>>> from astropy import units as u

# Download some stellar atmosphere models
>>> models = sc.discover_models()
>>> sc.download_model_grid('KURUCZ2003')cp 

# Download photometric filters  
>>> filters = sc.discover_filters(facility='HST')
>>> sc.download_filter_collection('HST_Collection', ['HST'])

# Build a flux cube for fast interpolation
>>> cube_file = sc.build_flux_cube('models/KURUCZ2003/', 'kurucz_cube.h5')

# Compute synthetic photometry
>>> photometry = sc.SyntheticPhotometry(cube_file, 'filters/HST_Collection/')
>>> magnitude = photometry.compute_magnitude(5777, 4.44, 0.0, 'HST/WFC3/F555W')
"""

# Version information
from .version import __version__


# Core imports - make main functionality easily accessible
from .atmosphere.grabber import AtmosphereGrabber, discover_models, download_model_grid
from .filters.grabber import FilterGrabber, discover_filters, download_filter_collection
from .cube.builder import DataCubeBuilder,  build_flux_cube
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
    
    # Configuration
    'conf',
]

# Package metadata
__author__ = "Stellar Colors Development Team"
__email__ = "stellar-colors@example.com"
__license__ = "BSD-3-Clause"
__description__ = "Stellar atmosphere modeling and synthetic photometry for astronomy"


# stellar_colors/photometry/synthetic.py
"""
Synthetic photometry calculations using stellar atmosphere models and filter transmission curves.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from astropy.io import ascii
from scipy.integrate import trapz, simpson

from .cube.builder import FluxCube
from .utils.integration import adaptive_integration

__all__ = ['SyntheticPhotometry', 'compute_synthetic_magnitude']


class SyntheticPhotometry:
    """
    A class for computing synthetic photometry from stellar atmosphere models.
    
    This class provides methods to compute synthetic magnitudes by convolving
    stellar atmosphere model spectra with photometric filter transmission curves.
    
    Parameters
    ----------
    flux_cube : str, Path, or FluxCube
        Path to flux cube file or FluxCube instance
    filter_dir : str or Path
        Directory containing filter transmission curves
    vega_spectrum : str or Path, optional
        Path to Vega spectrum for magnitude zero points
    """
    
    def __init__(
        self,
        flux_cube: Union[str, Path, FluxCube],
        filter_dir: Union[str, Path],
        vega_spectrum: Optional[Union[str, Path]] = None
    ):
        # Load flux cube
        if isinstance(flux_cube, FluxCube):
            self.flux_cube = flux_cube
        else:
            self.flux_cube = FluxCube(flux_cube)
        
        self.filter_dir = Path(filter_dir)
        if not self.filter_dir.exists():
            raise FileNotFoundError(f"Filter directory not found: {filter_dir}")
        
        # Load Vega spectrum if provided
        self.vega_spectrum = None
        if vega_spectrum:
            self.vega_spectrum = self._load_vega_spectrum(vega_spectrum)
        
        # Cache for loaded filters
        self._filter_cache = {}
        
        # Discover available filters
        self._discover_filters()
    
    def _discover_filters(self):
        """Discover available filters in the filter directory."""
        self.available_filters = []
        
        # Look for filter files (various formats)
        filter_extensions = ['.dat', '.txt', '.csv', '.fits']
        
        for ext in filter_extensions:
            for filter_file in self.filter_dir.rglob(f'*{ext}'):
                # Skip catalog files
                if 'catalog' in filter_file.name.lower():
                    continue
                
                # Create filter ID from path structure
                rel_path = filter_file.relative_to(self.filter_dir)
                filter_id = str(rel_path.with_suffix(''))
                self.available_filters.append(filter_id)
        
        print(f"Discovered {len(self.available_filters)} filters")
    
    def list_filters(self) -> List[str]:
        """
        List all available filters.
        
        Returns
        -------
        List[str]
            List of available filter identifiers
        """
        return self.available_filters.copy()
    
    def compute_magnitude(
        self,
        teff: float,
        logg: float,
        metallicity: float,
        filter_id: str,
        distance: Optional[float] = None,
        radius: Optional[float] = None,
        interpolation_method: str = 'linear'
    ) -> float:
        """
        Compute synthetic magnitude for given stellar parameters.
        
        Parameters
        ----------
        teff : float
            Effective temperature in K
        logg : float
            Surface gravity (log g)
        metallicity : float
            Metallicity [M/H]
        filter_id : str
            Filter identifier
        distance : float, optional
            Distance in parsecs (for apparent magnitudes)
        radius : float, optional
            Stellar radius in solar radii (for flux scaling)
        interpolation_method : str, optional
            Interpolation method for flux cube
            
        Returns
        -------
        float
            Synthetic magnitude
        """
        # Get stellar spectrum from flux cube
        wavelengths, fluxes = self.flux_cube.interpolate_spectrum(
            teff, logg, metallicity, method=interpolation_method
        )
        
        # Apply distance and radius scaling if provided
        if distance is not None and radius is not None:
            # Convert surface flux to observed flux
            distance_cm = distance * u.pc.to(u.cm)
            radius_cm = radius * const.R_sun.to(u.cm).value
            flux_scale = (radius_cm / distance_cm) ** 2
            fluxes = fluxes * flux_scale
        
        # Load filter transmission
        filter_wavelengths, filter_transmission = self._load_filter(filter_id)
        
        # Compute synthetic flux
        synthetic_flux = self._compute_synthetic_flux(
            wavelengths, fluxes, filter_wavelengths, filter_transmission
        )
        
        # Convert to magnitude using Vega zero point
        if self.vega_spectrum is not None:
            vega_flux = self._compute_synthetic_flux(
                self.vega_spectrum['wavelength'],
                self.vega_spectrum['flux'],
                filter_wavelengths,
                filter_transmission
            )
            magnitude = -2.5 * np.log10(synthetic_flux / vega_flux)
        else:
            # Use AB magnitude system as fallback
            # AB magnitude = -2.5 * log10(flux) - 48.6
            magnitude = -2.5 * np.log10(synthetic_flux) - 48.6
            warnings.warn("No Vega spectrum provided, using AB magnitude system")
        
        return magnitude
    
    def compute_color(
        self,
        teff: float,
        logg: float,
        metallicity: float,
        filter1_id: str,
        filter2_id: str,
        **kwargs
    ) -> float:
        """
        Compute color index (filter1 - filter2).
        
        Parameters
        ----------
        teff : float
            Effective temperature in K
        logg : float
            Surface gravity (log g)
        metallicity : float
            Metallicity [M/H]
        filter1_id : str
            First filter identifier
        filter2_id : str
            Second filter identifier
        **kwargs
            Additional arguments passed to compute_magnitude
            
        Returns
        -------
        float
            Color index (magnitude1 - magnitude2)
        """
        mag1 = self.compute_magnitude(teff, logg, metallicity, filter1_id, **kwargs)
        mag2 = self.compute_magnitude(teff, logg, metallicity, filter2_id, **kwargs)
        
        return mag1 - mag2
    
    def compute_magnitude_grid(
        self,
        teff_range: Tuple[float, float],
        logg_range: Tuple[float, float],
        metallicity_range: Tuple[float, float],
        filter_id: str,
        n_teff: int = 20,
        n_logg: int = 20,
        n_metallicity: int = 10,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute synthetic magnitudes on a regular parameter grid.
        
        Parameters
        ----------
        teff_range : tuple
            (min_teff, max_teff) in K
        logg_range : tuple
            (min_logg, max_logg)
        metallicity_range : tuple
            (min_metallicity, max_metallicity)
        filter_id : str
            Filter identifier
        n_teff : int, optional
            Number of Teff grid points
        n_logg : int, optional
            Number of log g grid points
        n_metallicity : int, optional
            Number of metallicity grid points
        **kwargs
            Additional arguments passed to compute_magnitude
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing parameter grids and magnitude array
        """
        # Create parameter grids
        teff_grid = np.linspace(teff_range[0], teff_range[1], n_teff)
        logg_grid = np.linspace(logg_range[0], logg_range[1], n_logg)
        metallicity_grid = np.linspace(metallicity_range[0], metallicity_range[1], n_metallicity)
        
        # Initialize magnitude array
        magnitudes = np.zeros((n_teff, n_logg, n_metallicity))
        
        # Compute magnitudes
        for i, teff in enumerate(teff_grid):
            for j, logg in enumerate(logg_grid):
                for k, metallicity in enumerate(metallicity_grid):
                    try:
                        mag = self.compute_magnitude(
                            teff, logg, metallicity, filter_id, **kwargs
                        )
                        magnitudes[i, j, k] = mag
                    except Exception as e:
                        magnitudes[i, j, k] = np.nan
                        warnings.warn(f"Failed to compute magnitude at "
                                    f"Teff={teff}, logg={logg}, [M/H]={metallicity}: {e}")
        
        return {
            'teff_grid': teff_grid,
            'logg_grid': logg_grid,
            'metallicity_grid': metallicity_grid,
            'magnitudes': magnitudes
        }
    
    def _load_filter(self, filter_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load filter transmission curve."""
        if filter_id in self._filter_cache:
            return self._filter_cache[filter_id]
        
        # Find filter file
        filter_file = None
        for ext in ['.dat', '.txt', '.csv']:
            candidate = self.filter_dir / f"{filter_id}{ext}"
            if candidate.exists():
                filter_file = candidate
                break
        
        if filter_file is None:
            raise FileNotFoundError(f"Filter {filter_id} not found in {self.filter_dir}")
        
        try:
            # Try different loading methods
            if filter_file.suffix == '.csv':
                data = pd.read_csv(filter_file, comment='#')
                wavelengths = data.iloc[:, 0].values
                transmission = data.iloc[:, 1].values
            else:
                data = np.loadtxt(filter_file, comments='#')
                wavelengths = data[:, 0]
                transmission = data[:, 1]
            
            # Validate and sort
            if len(wavelengths) != len(transmission):
                raise ValueError("Wavelength and transmission arrays must have same length")
            
            # Sort by wavelength
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            transmission = transmission[sort_idx]
            
            # Normalize transmission to [0, 1]
            transmission = np.maximum(transmission, 0)  # No negative transmission
            if transmission.max() > 1.1:  # Likely in percentage
                transmission = transmission / 100.0
            
            # Cache the result
            self._filter_cache[filter_id] = (wavelengths, transmission)
            
            return wavelengths, transmission
            
        except Exception as e:
            raise RuntimeError(f"Failed to load filter {filter_id}: {e}")
    
    def _load_vega_spectrum(self, vega_file: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load Vega spectrum."""
        vega_file = Path(vega_file)
        
        try:
            if vega_file.suffix == '.csv':
                data = pd.read_csv(vega_file, comment='#')
                wavelengths = data.iloc[:, 0].values
                fluxes = data.iloc[:, 1].values
            else:
                data = np.loadtxt(vega_file, comments='#')
                wavelengths = data[:, 0]
                fluxes = data[:, 1]
            
            # Sort by wavelength
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            fluxes = fluxes[sort_idx]
            
            return {'wavelength': wavelengths, 'flux': fluxes}
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Vega spectrum: {e}")
    
    def _compute_synthetic_flux(
        self,
        spec_wavelengths: np.ndarray,
        spec_fluxes: np.ndarray,
        filter_wavelengths: np.ndarray,
        filter_transmission: np.ndarray
    ) -> float:
        """
        Compute synthetic flux by convolving spectrum with filter.
        
        Uses the standard formula:
        F = ∫ F_λ(λ) * T(λ) * λ dλ / ∫ T(λ) * λ dλ
        """
        # Find wavelength overlap
        wave_min = max(spec_wavelengths.min(), filter_wavelengths.min())
        wave_max = min(spec_wavelengths.max(), filter_wavelengths.max())
        
        if wave_min >= wave_max:
            raise ValueError("No wavelength overlap between spectrum and filter")
        
        # Create common wavelength grid
        # Use the finer of the two grids
        spec_resolution = np.median(np.diff(spec_wavelengths))
        filter_resolution = np.median(np.diff(filter_wavelengths))
        resolution = min(spec_resolution, filter_resolution)
        
        n_points = int((wave_max - wave_min) / resolution) + 1
        n_points = min(n_points, 10000)  # Cap for memory
        
        common_wavelengths = np.linspace(wave_min, wave_max, n_points)
        
        # Interpolate spectrum and filter to common grid
        interp_flux = np.interp(common_wavelengths, spec_wavelengths, spec_fluxes)
        interp_transmission = np.interp(
            common_wavelengths, filter_wavelengths, filter_transmission
        )
        
        # Compute integrals
        numerator = trapz(
            interp_flux * interp_transmission * common_wavelengths,
            common_wavelengths
        )
        denominator = trapz(
            interp_transmission * common_wavelengths,
            common_wavelengths
        )
        
        if denominator == 0:
            raise ValueError("Filter transmission integrates to zero")
        
        return numerator / denominator


def compute_synthetic_magnitude(
    teff: float,
    logg: float,
    metallicity: float,
    filter_id: str,
    flux_cube_file: Union[str, Path],
    filter_dir: Union[str, Path],
    **kwargs
) -> float:
    """
    Convenience function to compute a synthetic magnitude.
    
    Parameters
    ----------
    teff : float
        Effective temperature in K
    logg : float
        Surface gravity (log g)
    metallicity : float
        Metallicity [M/H]
    filter_id : str
        Filter identifier
    flux_cube_file : str or Path
        Path to flux cube file
    filter_dir : str or Path
        Directory containing filter transmission curves
    **kwargs
        Additional arguments passed to SyntheticPhotometry.compute_magnitude
        
    Returns
    -------
    float
        Synthetic magnitude
    """
    photometry = SyntheticPhotometry(flux_cube_file, filter_dir)
    return photometry.compute_magnitude(teff, logg, metallicity, filter_id, **kwargs)


# stellar_colors/utils/integration.py
"""
Numerical integration utilities for stellar colors calculations.
"""

import warnings
from typing import Callable, Tuple

import numpy as np
from scipy.integrate import quad, trapz, simpson


def adaptive_integration(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'auto'
) -> float:
    """
    Perform adaptive numerical integration.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable array
    y : np.ndarray
        Dependent variable array
    method : str, optional
        Integration method ('auto', 'trapz', 'simpson')
        
    Returns
    -------
    float
        Integrated value
    """
    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 points for integration")
    
    if method == 'auto':
        # Choose method based on array size and regularity
        if len(x) < 5:
            method = 'trapz'
        else:
            # Check if grid is regular
            dx = np.diff(x)
            if np.allclose(dx, dx[0], rtol=1e-3):
                method = 'simpson'
            else:
                method = 'trapz'
    
    if method == 'trapz':
        return trapz(y, x)
    elif method == 'simpson':
        if len(x) % 2 == 0:
            # Simpson's rule requires odd number of points
            # Use composite rule for even number
            result = simpson(y[:-1], x[:-1])
            result += trapz(y[-2:], x[-2:])  # Add last interval with trapz
            return result
        else:
            return simpson(y, x)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def romberg_integration(
    x: np.ndarray,
    y: np.ndarray,
    max_levels: int = 10
) -> float:
    """
    Romberg integration for improved accuracy.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable array (must be uniformly spaced)
    y : np.ndarray
        Dependent variable array
    max_levels : int, optional
        Maximum number of Romberg levels
        
    Returns
    -------
    float
        Integrated value
    """
    n = len(x)
    if n < 3:
        return trapz(y, x)
    
    # Check if uniformly spaced
    dx = np.diff(x)
    if not np.allclose(dx, dx[0], rtol=1e-6):
        warnings.warn("Romberg integration requires uniform spacing, falling back to trapz")
        return trapz(y, x)
    
    h = dx[0]
    
    # Initialize Romberg table
    R = np.zeros((max_levels, max_levels))
    
    # First column: trapezoidal rule with successive halvings
    R[0, 0] = 0.5 * h * (y[0] + y[-1])  # Single interval
    
    for i in range(1, min(max_levels, int(np.log2(n-1)) + 1)):
        # Refine with half the step size
        step = 2 ** i
        if step >= n:
            break
            
        # Add intermediate points
        sum_intermediate = 0
        for k in range(1, step, 2):
            if k < n:
                sum_intermediate += y[k * (n-1) // step]
        
        R[i, 0] = 0.5 * R[i-1, 0] + (h / step) * sum_intermediate
        
        # Richardson extrapolation
        for j in range(1, i+1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
    
    # Return the most accurate estimate
    max_i = min(max_levels-1, int(np.log2(n-1)))
    return R[max_i, max_i] if max_i > 0 else R[0, 0]