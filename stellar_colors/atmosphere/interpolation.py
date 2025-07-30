"""
Interpolation methods for stellar atmosphere model grids.

This module provides various interpolation techniques for generating
stellar spectra at arbitrary stellar parameters from model grids.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator, interp1d
import logging

logger = logging.getLogger(__name__)


class BaseInterpolator(ABC):
    """
    Abstract base class for stellar atmosphere interpolators.
    """
    
    def __init__(self, atmosphere_model):
        """
        Initialize the interpolator with an atmosphere model.
        
        Parameters
        ----------
        atmosphere_model : AtmosphereModel
            The loaded atmosphere model to interpolate
        """
        self.model = atmosphere_model
        if not self.model._loaded:
            self.model.load_model_grid()
    
    @abstractmethod
    def interpolate(self, teff: float, logg: float, metallicity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate a spectrum at given stellar parameters.
        
        Parameters
        ----------
        teff : float
            Effective temperature in K
        logg : float
            Surface gravity (log g)
        metallicity : float
            Metallicity [M/H]
            
        Returns
        -------
        wavelength : np.ndarray
            Wavelength array in Angstroms
        flux : np.ndarray
            Interpolated flux array in erg/s/cm²/Å
        """
        pass


class LinearInterpolator(BaseInterpolator):
    """
    Linear interpolation in 3D parameter space (Teff, log g, [M/H]).
    
    Uses scipy's RegularGridInterpolator for fast linear interpolation
    on regular grids.
    """
    
    def __init__(self, atmosphere_model):
        super().__init__(atmosphere_model)
        self._setup_grid()
    
    def _setup_grid(self):
        """Set up the regular grid for interpolation."""
        # Get unique parameter values
        self.teff_grid = np.sort(self.model.lookup_table['teff'].unique())
        self.logg_grid = np.sort(self.model.lookup_table['logg'].unique())
        self.meta_grid = np.sort(self.model.lookup_table['meta'].unique())
        
        # Load a reference spectrum to get wavelength grid
        first_file = self.model.lookup_table.iloc[0]['file_name']
        self.wavelength, _ = self.model._load_spectrum_file(first_file)
        
        logger.info(f"Set up linear interpolation grid: "
                   f"{len(self.teff_grid)} × {len(self.logg_grid)} × {len(self.meta_grid)} models")
    
    def interpolate(self, teff: float, logg: float, metallicity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform linear interpolation."""
        # Check if parameters are in grid
        if not self.model.is_in_grid(teff, logg, metallicity):
            logger.warning(f"Parameters ({teff}, {logg}, {metallicity}) outside grid bounds")
        
        # Find the 8 closest grid points for trilinear interpolation
        flux_values = self._get_flux_values_for_interpolation(teff, logg, metallicity)
        
        if flux_values is None:
            # Fall back to nearest neighbor
            return self._nearest_neighbor(teff, logg, metallicity)
        
        # Perform trilinear interpolation
        interpolated_flux = self._trilinear_interpolation(teff, logg, metallicity, flux_values)
        
        return self.wavelength, interpolated_flux
    
    def _get_flux_values_for_interpolation(self, teff: float, logg: float, metallicity: float):
        """Get flux values from the 8 surrounding grid points."""
        # Find surrounding grid indices
        teff_idx = np.searchsorted(self.teff_grid, teff)
        logg_idx = np.searchsorted(self.logg_grid, logg)
        meta_idx = np.searchsorted(self.meta_grid, metallicity)
        
        # Ensure we have valid ranges
        teff_idx = max(1, min(len(self.teff_grid) - 1, teff_idx))
        logg_idx = max(1, min(len(self.logg_grid) - 1, logg_idx))
        meta_idx = max(1, min(len(self.meta_grid) - 1, meta_idx))
        
        # Get the 8 corner points
        flux_cube = np.zeros((2, 2, 2, len(self.wavelength)))
        
        for i, ti in enumerate([teff_idx - 1, teff_idx]):
            for j, gi in enumerate([logg_idx - 1, logg_idx]):
                for k, mi in enumerate([meta_idx - 1, meta_idx]):
                    # Find model with these exact parameters
                    mask = (
                        (self.model.lookup_table['teff'] == self.teff_grid[ti]) &
                        (self.model.lookup_table['logg'] == self.logg_grid[gi]) &
                        (self.model.lookup_table['meta'] == self.meta_grid[mi])
                    )
                    
                    if mask.any():
                        filename = self.model.lookup_table[mask].iloc[0]['file_name']
                        _, flux = self.model._load_spectrum_file(filename)
                        
                        # Interpolate to common wavelength grid if needed
                        if len(flux) != len(self.wavelength):
                            flux = np.interp(self.wavelength, _, flux)
                        
                        flux_cube[i, j, k, :] = flux
                    else:
                        return None  # Missing model, fall back to nearest neighbor
        
        return flux_cube, (teff_idx, logg_idx, meta_idx)
    
    def _trilinear_interpolation(self, teff: float, logg: float, metallicity: float, flux_data):
        """Perform trilinear interpolation."""
        flux_cube, (teff_idx, logg_idx, meta_idx) = flux_data
        
        # Calculate interpolation weights
        t_weight = (teff - self.teff_grid[teff_idx - 1]) / (self.teff_grid[teff_idx] - self.teff_grid[teff_idx - 1])
        g_weight = (logg - self.logg_grid[logg_idx - 1]) / (self.logg_grid[logg_idx] - self.logg_grid[logg_idx - 1])
        m_weight = (metallicity - self.meta_grid[meta_idx - 1]) / (self.meta_grid[meta_idx] - self.meta_grid[meta_idx - 1])
        
        # Clamp weights to [0, 1]
        t_weight = np.clip(t_weight, 0, 1)
        g_weight = np.clip(g_weight, 0, 1)  
        m_weight = np.clip(m_weight, 0, 1)
        
        # Trilinear interpolation
        # First interpolate along Teff axis
        c00 = flux_cube[0, 0, 0, :] * (1 - t_weight) + flux_cube[1, 0, 0, :] * t_weight
        c01 = flux_cube[0, 0, 1, :] * (1 - t_weight) + flux_cube[1, 0, 1, :] * t_weight
        c10 = flux_cube[0, 1, 0, :] * (1 - t_weight) + flux_cube[1, 1, 0, :] * t_weight
        c11 = flux_cube[0, 1, 1, :] * (1 - t_weight) + flux_cube[1, 1, 1, :] * t_weight
        
        # Then along log g axis
        c0 = c00 * (1 - g_weight) + c10 * g_weight
        c1 = c01 * (1 - g_weight) + c11 * g_weight
        
        # Finally along metallicity axis
        interpolated_flux = c0 * (1 - m_weight) + c1 * m_weight
        
        return interpolated_flux
    
    def _nearest_neighbor(self, teff: float, logg: float, metallicity: float):
        """Fall back to nearest neighbor interpolation."""
        logger.debug("Using nearest neighbor interpolation")
        return self.model.get_spectrum(teff, logg, metallicity)


class KNNInterpolator(BaseInterpolator):
    """
    K-Nearest Neighbors interpolation using distance-weighted averaging.
    
    This method finds the k closest models in parameter space and
    computes a weighted average of their spectra.
    """
    
    def __init__(self, atmosphere_model, k: int = 4):
        """
        Initialize KNN interpolator.
        
        Parameters
        ----------
        k : int
            Number of nearest neighbors to use (default: 4)
        """
        super().__init__(atmosphere_model)
        self.k = k
        self._setup_kdtree()
    
    def _setup_kdtree(self):
        """Set up KDTree for efficient nearest neighbor searches."""
        # Normalize parameters for distance calculations
        params = self.model.lookup_table[['teff', 'logg', 'meta']].values
        
        # Normalize each parameter to [0, 1] range
        self.param_mins = params.min(axis=0)
        self.param_maxs = params.max(axis=0)
        self.param_ranges = self.param_maxs - self.param_mins
        
        # Handle case where range is zero
        self.param_ranges[self.param_ranges == 0] = 1.0
        
        normalized_params = (params - self.param_mins) / self.param_ranges
        
        # Build KDTree
        self.kdtree = KDTree(normalized_params)
        
        # Load reference wavelength grid
        first_file = self.model.lookup_table.iloc[0]['file_name']
        self.wavelength, _ = self.model._load_spectrum_file(first_file)
        
        logger.info(f"Set up KNN interpolation with k={self.k}")
    
    def interpolate(self, teff: float, logg: float, metallicity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Perform KNN interpolation."""
        # Normalize query point
        query = np.array([teff, logg, metallicity])
        normalized_query = (query - self.param_mins) / self.param_ranges
        
        # Find k nearest neighbors
        distances, indices = self.kdtree.query(normalized_query, k=self.k)
        
        # Handle single point case
        if np.isscalar(distances):
            distances = np.array([distances])
            indices = np.array([indices])
        
        # Load spectra for nearest neighbors
        spectra = []
        for idx in indices:
            filename = self.model.lookup_table.iloc[idx]['file_name']
            _, flux = self.model._load_spectrum_file(filename)
            
            # Interpolate to common wavelength grid if needed
            if len(flux) != len(self.wavelength):
                flux = np.interp(self.wavelength, _, flux)
            
            spectra.append(flux)
        
        spectra = np.array(spectra)
        
        # Calculate weights (inverse distance weighting)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        weights = 1.0 / (distances + epsilon)
        weights /= weights.sum()
        
        # Weighted average of spectra
        interpolated_flux = np.average(spectra, axis=0, weights=weights)
        
        return self.wavelength, interpolated_flux


class HermiteInterpolator(BaseInterpolator):
    """
    Hermite interpolation using derivatives for smooth interpolation.
    
    This method uses Hermite polynomials with derivative information
    for smoother interpolation than linear methods.
    """
    
    def __init__(self, atmosphere_model):
        super().__init__(atmosphere_model)
        self._setup_grid()
    
    def _setup_grid(self):
        """Set up the grid and compute derivatives."""
        # Get unique parameter values
        self.teff_grid = np.sort(self.model.lookup_table['teff'].unique())
        self.logg_grid = np.sort(self.model.lookup_table['logg'].unique())
        self.meta_grid = np.sort(self.model.lookup_table['meta'].unique())
        
        # Load reference wavelength grid
        first_file = self.model.lookup_table.iloc[0]['file_name']
        self.wavelength, _ = self.model._load_spectrum_file(first_file)
        
        logger.info(f"Set up Hermite interpolation grid: "
                   f"{len(self.teff_grid)} × {len(self.logg_grid)} × {len(self.meta_grid)} models")
    
    def interpolate(self, teff: float, logg: float, metallicity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Hermite interpolation.
        
        Note: This is a simplified implementation. Full Hermite interpolation
        requires derivative information which may not be available in all grids.
        Falls back to linear interpolation for now.
        """
        logger.warning("Hermite interpolation not fully implemented, using linear fallback")
        
        # For now, use linear interpolation as fallback
        linear_interp = LinearInterpolator(self.model)
        return linear_interp.interpolate(teff, logg, metallicity)


def interpolate_spectrum(atmosphere_model, teff: float, logg: float, metallicity: float, 
                        method: str = "linear", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to interpolate a spectrum using specified method.
    
    Parameters
    ----------
    atmosphere_model : AtmosphereModel
        The loaded atmosphere model
    teff : float
        Effective temperature in K
    logg : float
        Surface gravity (log g)
    metallicity : float
        Metallicity [M/H]
    method : str
        Interpolation method ('linear', 'knn', 'hermite')
    **kwargs
        Additional arguments for the interpolator
        
    Returns
    -------
    wavelength : np.ndarray
        Wavelength array in Angstroms
    flux : np.ndarray
        Interpolated flux array in erg/s/cm²/Å
    """
    if method.lower() == "linear":
        interpolator = LinearInterpolator(atmosphere_model)
    elif method.lower() == "knn":
        k = kwargs.get('k', 4)
        interpolator = KNNInterpolator(atmosphere_model, k=k)
    elif method.lower() == "hermite":
        interpolator = HermiteInterpolator(atmosphere_model)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return interpolator.interpolate(teff, logg, metallicity)