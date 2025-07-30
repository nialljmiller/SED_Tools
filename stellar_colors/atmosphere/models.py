"""
Stellar atmosphere model classes for different model grids.

This module provides base classes and specific implementations for 
different stellar atmosphere model grids like Kurucz, PHOENIX, ATLAS, etc.
"""

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import logging

logger = logging.getLogger(__name__)


class AtmosphereModel(ABC):
    """
    Abstract base class for stellar atmosphere models.
    
    This class defines the interface that all stellar atmosphere models
    should implement for consistent usage across the package.
    """
    
    def __init__(self, model_dir: Union[str, Path]):
        """
        Initialize the atmosphere model.
        
        Parameters
        ----------
        model_dir : str or Path
            Directory containing the model files and lookup table
        """
        self.model_dir = Path(model_dir)
        self.lookup_table = None
        self.parameter_ranges = {}
        self._loaded = False
        
    @abstractmethod
    def load_model_grid(self) -> None:
        """Load the model grid and lookup table."""
        pass
    
    @abstractmethod
    def get_spectrum(self, teff: float, logg: float, metallicity: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a spectrum for given stellar parameters.
        
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
            Flux array in erg/s/cm²/Å
        """
        pass
    
    def is_in_grid(self, teff: float, logg: float, metallicity: float) -> bool:
        """
        Check if stellar parameters are within the model grid bounds.
        
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
        bool
            True if parameters are within grid bounds
        """
        if not self._loaded:
            self.load_model_grid()
            
        teff_ok = (self.parameter_ranges['teff_min'] <= teff <= self.parameter_ranges['teff_max'])
        logg_ok = (self.parameter_ranges['logg_min'] <= logg <= self.parameter_ranges['logg_max'])
        meta_ok = (self.parameter_ranges['meta_min'] <= metallicity <= self.parameter_ranges['meta_max'])
        
        return teff_ok and logg_ok and meta_ok
    
    def get_parameter_ranges(self) -> Dict[str, float]:
        """
        Get the parameter ranges covered by this model grid.
        
        Returns
        -------
        dict
            Dictionary with parameter ranges
        """
        if not self._loaded:
            self.load_model_grid()
        return self.parameter_ranges.copy()
    
    def _load_lookup_table(self, lookup_file: str = "lookup_table.csv") -> pd.DataFrame:
        """
        Load the lookup table from CSV file.
        
        Parameters
        ----------
        lookup_file : str
            Name of the lookup table file
            
        Returns
        -------
        pd.DataFrame
            Lookup table with model parameters
        """
        lookup_path = self.model_dir / lookup_file
        
        if not lookup_path.exists():
            raise FileNotFoundError(f"Lookup table not found: {lookup_path}")
        
        # Read CSV, handling comments
        df = pd.read_csv(lookup_path, comment='#')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Ensure we have the required columns
        required_cols = ['file_name', 'teff', 'logg']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in lookup table")
        
        # Handle metallicity column (might be 'meta', 'feh', or '[M/H]')
        meta_cols = ['meta', 'feh', '[M/H]', 'metallicity']
        meta_col = next((col for col in meta_cols if col in df.columns), None)
        
        if meta_col and meta_col != 'meta':
            df['meta'] = df[meta_col]
        elif 'meta' not in df.columns:
            logger.warning("No metallicity column found, setting to 0.0")
            df['meta'] = 0.0
        
        return df
    
    def _compute_parameter_ranges(self) -> None:
        """Compute parameter ranges from the lookup table."""
        if self.lookup_table is None:
            raise ValueError("Lookup table not loaded")
        
        self.parameter_ranges = {
            'teff_min': self.lookup_table['teff'].min(),
            'teff_max': self.lookup_table['teff'].max(),
            'logg_min': self.lookup_table['logg'].min(),
            'logg_max': self.lookup_table['logg'].max(),
            'meta_min': self.lookup_table['meta'].min(),
            'meta_max': self.lookup_table['meta'].max(),
        }


class KuruczModel(AtmosphereModel):
    """
    Kurucz stellar atmosphere model implementation.
    
    Handles Kurucz ATLAS model grids with standard format.
    """
    
    def __init__(self, model_dir: Union[str, Path]):
        super().__init__(model_dir)
        self.model_name = "Kurucz"
    
    def load_model_grid(self) -> None:
        """Load the Kurucz model grid."""
        logger.info(f"Loading Kurucz model grid from {self.model_dir}")
        
        # Load lookup table
        self.lookup_table = self._load_lookup_table()
        
        # Compute parameter ranges
        self._compute_parameter_ranges()
        
        self._loaded = True
        logger.info(f"Loaded {len(self.lookup_table)} Kurucz models")
    
    def get_spectrum(self, teff: float, logg: float, metallicity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get spectrum by finding closest model in grid."""
        if not self._loaded:
            self.load_model_grid()
        
        # Find closest model
        distances = (
            ((self.lookup_table['teff'] - teff) / 1000) ** 2 +
            (self.lookup_table['logg'] - logg) ** 2 +
            (self.lookup_table['meta'] - metallicity) ** 2
        )
        
        closest_idx = distances.idxmin()
        filename = self.lookup_table.loc[closest_idx, 'file_name']
        
        # Load spectrum file
        return self._load_spectrum_file(filename)
    
    def _load_spectrum_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a spectrum file."""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Spectrum file not found: {filepath}")
        
        # Load spectrum data (wavelength, flux)
        data = np.loadtxt(filepath, unpack=True)
        
        if len(data) == 2:
            wavelength, flux = data
        else:
            raise ValueError(f"Expected 2 columns in spectrum file, got {len(data)}")
        
        return wavelength, flux


class PhoenixModel(AtmosphereModel):
    """
    PHOENIX stellar atmosphere model implementation.
    
    Handles PHOENIX model grids with their specific format.
    """
    
    def __init__(self, model_dir: Union[str, Path]):
        super().__init__(model_dir)
        self.model_name = "PHOENIX"
    
    def load_model_grid(self) -> None:
        """Load the PHOENIX model grid."""
        logger.info(f"Loading PHOENIX model grid from {self.model_dir}")
        
        # Load lookup table
        self.lookup_table = self._load_lookup_table()
        
        # Compute parameter ranges
        self._compute_parameter_ranges()
        
        self._loaded = True
        logger.info(f"Loaded {len(self.lookup_table)} PHOENIX models")
    
    def get_spectrum(self, teff: float, logg: float, metallicity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get spectrum by finding closest model in grid."""
        if not self._loaded:
            self.load_model_grid()
        
        # Find closest model (similar to Kurucz but could have different weighting)
        distances = (
            ((self.lookup_table['teff'] - teff) / 1000) ** 2 +
            (self.lookup_table['logg'] - logg) ** 2 +
            (self.lookup_table['meta'] - metallicity) ** 2
        )
        
        closest_idx = distances.idxmin()
        filename = self.lookup_table.loc[closest_idx, 'file_name']
        
        # Load spectrum file
        return self._load_spectrum_file(filename)
    
    def _load_spectrum_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a PHOENIX spectrum file."""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Spectrum file not found: {filepath}")
        
        # PHOENIX files might have headers or different formats
        try:
            # Try standard format first
            data = np.loadtxt(filepath, unpack=True)
            if len(data) == 2:
                wavelength, flux = data
            else:
                raise ValueError(f"Expected 2 columns, got {len(data)}")
        except:
            # Try reading with pandas to handle headers
            df = pd.read_csv(filepath, delim_whitespace=True, comment='#')
            if len(df.columns) >= 2:
                wavelength = df.iloc[:, 0].values
                flux = df.iloc[:, 1].values
            else:
                raise ValueError("Could not parse PHOENIX spectrum file")
        
        return wavelength, flux


class AtlasModel(AtmosphereModel):
    """
    ATLAS stellar atmosphere model implementation.
    
    Handles ATLAS model grids.
    """
    
    def __init__(self, model_dir: Union[str, Path]):
        super().__init__(model_dir)
        self.model_name = "ATLAS"
    
    def load_model_grid(self) -> None:
        """Load the ATLAS model grid."""
        logger.info(f"Loading ATLAS model grid from {self.model_dir}")
        
        # Load lookup table
        self.lookup_table = self._load_lookup_table()
        
        # Compute parameter ranges
        self._compute_parameter_ranges()
        
        self._loaded = True
        logger.info(f"Loaded {len(self.lookup_table)} ATLAS models")
    
    def get_spectrum(self, teff: float, logg: float, metallicity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get spectrum by finding closest model in grid."""
        if not self._loaded:
            self.load_model_grid()
        
        # Find closest model
        distances = (
            ((self.lookup_table['teff'] - teff) / 1000) ** 2 +
            (self.lookup_table['logg'] - logg) ** 2 +
            (self.lookup_table['meta'] - metallicity) ** 2
        )
        
        closest_idx = distances.idxmin()
        filename = self.lookup_table.loc[closest_idx, 'file_name']
        
        # Load spectrum file
        return self._load_spectrum_file(filename)
    
    def _load_spectrum_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load an ATLAS spectrum file."""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Spectrum file not found: {filepath}")
        
        # Load spectrum data
        data = np.loadtxt(filepath, unpack=True)
        
        if len(data) == 2:
            wavelength, flux = data
        else:
            raise ValueError(f"Expected 2 columns in spectrum file, got {len(data)}")
        
        return wavelength, flux


def load_atmosphere_model(model_dir: Union[str, Path], model_type: str = "auto") -> AtmosphereModel:
    """
    Load an atmosphere model based on the model type.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing the model files
    model_type : str
        Type of model ('kurucz', 'phoenix', 'atlas', or 'auto')
        
    Returns
    -------
    AtmosphereModel
        Loaded atmosphere model instance
    """
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Auto-detect model type if not specified
    if model_type.lower() == "auto":
        model_name = model_dir.name.lower()
        if "kurucz" in model_name:
            model_type = "kurucz"
        elif "phoenix" in model_name:
            model_type = "phoenix"
        elif "atlas" in model_name:
            model_type = "atlas"
        else:
            logger.warning(f"Could not auto-detect model type for {model_dir}, using Kurucz")
            model_type = "kurucz"
    
    # Create appropriate model instance
    if model_type.lower() == "kurucz":
        return KuruczModel(model_dir)
    elif model_type.lower() == "phoenix":
        return PhoenixModel(model_dir)
    elif model_type.lower() == "atlas":
        return AtlasModel(model_dir)
    else:
        raise ValueError(f"Unknown model type: {model_type}")