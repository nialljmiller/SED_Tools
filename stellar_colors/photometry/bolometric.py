# stellar_colors/photometry/bolometric.py
"""
Bolometric corrections and luminosity calculations from stellar atmosphere models.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.integrate import trapz

from ..cube.builder import FluxCube
from ..utils.integration import adaptive_integration

__all__ = ['BolometricCorrections', 'compute_bolometric_correction']


class BolometricCorrections:
    """
    A class for computing bolometric corrections and related quantities.
    
    This class provides methods to compute bolometric magnitudes, luminosities,
    and bolometric corrections using stellar atmosphere models.
    
    Parameters
    ----------
    flux_cube : str, Path, or FluxCube
        Path to flux cube file or FluxCube instance
    """
    
    def __init__(self, flux_cube: Union[str, Path, FluxCube]):
        # Load flux cube
        if isinstance(flux_cube, FluxCube):
            self.flux_cube = flux_cube
        else:
            self.flux_cube = FluxCube(flux_cube)
    
    def compute_bolometric_magnitude(
        self,
        teff: float,
        logg: float,
        metallicity: float,
        distance: Optional[float] = None,
        radius: Optional[float] = None,
        interpolation_method: str = 'linear'
    ) -> float:
        """
        Compute bolometric magnitude for given stellar parameters.
        
        Parameters
        ----------
        teff : float
            Effective temperature in K
        logg : float
            Surface gravity (log g)
        metallicity : float
            Metallicity [M/H]
        distance : float, optional
            Distance in parsecs (for apparent magnitudes)
        radius : float, optional
            Stellar radius in solar radii (for absolute magnitudes)
        interpolation_method : str, optional
            Interpolation method for flux cube
            
        Returns
        -------
        float
            Bolometric magnitude
        """
        # Get stellar spectrum
        wavelengths, fluxes = self.flux_cube.interpolate_spectrum(
            teff, logg, metallicity, method=interpolation_method
        )
        
        # Integrate to get bolometric flux
        bolometric_flux = self._integrate_bolometric_flux(wavelengths, fluxes)
        
        # Apply distance and radius scaling if provided
        if distance is not None and radius is not None:
            # Convert surface flux to observed flux
            distance_cm = distance * u.pc.to(u.cm)
            radius_cm = radius * const.R_sun.to(u.cm).value
            flux_scale = (radius_cm / distance_cm) ** 2
            bolometric_flux = bolometric_flux * flux_scale
            
            # Apparent bolometric magnitude
            # Using standard zero point: M_bol,sun = 4.83
            solar_lum = 3.828e33  # erg/s
            solar_flux_at_10pc = solar_lum / (4 * np.pi * (10 * u.pc.to(u.cm))**2)
            
            magnitude = -2.5 * np.log10(bolometric_flux / solar_flux_at_10pc) + 4.83
            
        elif radius is not None:
            # Absolute bolometric magnitude
            # Convert surface flux to luminosity
            radius_cm = radius * const.R_sun.to(u.cm).value
            luminosity = bolometric_flux * 4 * np.pi * radius_cm**2
            
            # Solar luminosity and absolute bolometric magnitude
            solar_lum = 3.828e33  # erg/s
            magnitude = -2.5 * np.log10(luminosity / solar_lum) + 4.83
            
        else:
            # Just surface flux - need to specify what this represents
            warnings.warn("No distance or radius provided, returning surface flux magnitude")
            # Arbitrary zero point for surface flux
            magnitude = -2.5 * np.log10(bolometric_flux) - 10.0
        
        return magnitude
    
    def compute_luminosity(
        self,
        teff: float,
        logg: float,
        metallicity: float,
        radius: float,
        interpolation_method: str = 'linear'
    ) -> float:
        """
        Compute bolometric luminosity in solar units.
        
        Parameters
        ----------
        teff : float
            Effective temperature in K
        logg : float
            Surface gravity (log g)
        metallicity : float
            Metallicity [M/H]
        radius : float
            Stellar radius in solar radii
        interpolation_method : str, optional
            Interpolation method for flux cube
            
        Returns
        -------
        float
            Bolometric luminosity in solar luminosities
        """
        # Get stellar spectrum
        wavelengths, fluxes = self.flux_cube.interpolate_spectrum(
            teff, logg, metallicity, method=interpolation_method
        )
        
        # Integrate to get bolometric flux
        bolometric_flux = self._integrate_bolometric_flux(wavelengths, fluxes)
        
        # Convert to luminosity
        radius_cm = radius * const.R_sun.to(u.cm).value
        luminosity = bolometric_flux * 4 * np.pi * radius_cm**2
        
        # Convert to solar units
        solar_lum = 3.828e33  # erg/s
        return luminosity / solar_lum
    
    def compute_bolometric_correction(
        self,
        teff: float,
        logg: float,
        metallicity: float,
        filter_id: str,
        photometry_instance,
        interpolation_method: str = 'linear'
    ) -> float:
        """
        Compute bolometric correction BC = M_bol - M_filter.
        
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
        photometry_instance : SyntheticPhotometry
            Instance for computing filter magnitudes
        interpolation_method : str, optional
            Interpolation method
            
        Returns
        -------
        float
            Bolometric correction in magnitudes
        """
        # Compute bolometric magnitude (absolute, so set radius=1 for comparison)
        m_bol = self.compute_bolometric_magnitude(
            teff, logg, metallicity, radius=1.0, 
            interpolation_method=interpolation_method
        )
        
        # Compute filter magnitude (absolute, so set radius=1 for comparison)
        m_filter = photometry_instance.compute_magnitude(
            teff, logg, metallicity, filter_id,
            distance=10.0, radius=1.0,  # 10 pc for absolute magnitude
            interpolation_method=interpolation_method
        )
        
        return m_bol - m_filter
    
    def compute_effective_temperature(
        self,
        teff: float,
        logg: float,
        metallicity: float,
        interpolation_method: str = 'linear'
    ) -> float:
        """
        Compute effective temperature from integrated flux using Stefan-Boltzmann law.
        
        This can be used to validate models or compare different definitions.
        
        Parameters
        ----------
        teff : float
            Model effective temperature in K
        logg : float
            Surface gravity (log g)
        metallicity : float
            Metallicity [M/H]
        interpolation_method : str, optional
            Interpolation method
            
        Returns
        -------
        float
            Effective temperature derived from integrated flux
        """
        # Get stellar spectrum
        wavelengths, fluxes = self.flux_cube.interpolate_spectrum(
            teff, logg, metallicity, method=interpolation_method
        )
        
        # Integrate to get bolometric flux
        bolometric_flux = self._integrate_bolometric_flux(wavelengths, fluxes)
        
        # Apply Stefan-Boltzmann law: F = σ T_eff^4
        sigma_sb = 5.670374419e-5  # erg cm^-2 s^-1 K^-4
        teff_derived = (bolometric_flux / sigma_sb) ** 0.25
        
        return teff_derived
    
    def compute_bc_grid(
        self,
        teff_range: Tuple[float, float],
        logg_range: Tuple[float, float],
        metallicity_range: Tuple[float, float],
        filter_id: str,
        photometry_instance,
        n_teff: int = 20,
        n_logg: int = 20,
        n_metallicity: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute bolometric corrections on a regular parameter grid.
        
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
        photometry_instance : SyntheticPhotometry
            Instance for computing filter magnitudes
        n_teff : int, optional
            Number of Teff grid points
        n_logg : int, optional
            Number of log g grid points
        n_metallicity : int, optional
            Number of metallicity grid points
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing parameter grids and BC array
        """
        # Create parameter grids
        teff_grid = np.linspace(teff_range[0], teff_range[1], n_teff)
        logg_grid = np.linspace(logg_range[0], logg_range[1], n_logg)
        metallicity_grid = np.linspace(metallicity_range[0], metallicity_range[1], n_metallicity)
        
        # Initialize BC array
        bc_values = np.zeros((n_teff, n_logg, n_metallicity))
        
        # Compute BCs
        for i, teff in enumerate(teff_grid):
            for j, logg in enumerate(logg_grid):
                for k, metallicity in enumerate(metallicity_grid):
                    try:
                        bc = self.compute_bolometric_correction(
                            teff, logg, metallicity, filter_id, photometry_instance
                        )
                        bc_values[i, j, k] = bc
                    except Exception as e:
                        bc_values[i, j, k] = np.nan
                        warnings.warn(f"Failed to compute BC at "
                                    f"Teff={teff}, logg={logg}, [M/H]={metallicity}: {e}")
        
        return {
            'teff_grid': teff_grid,
            'logg_grid': logg_grid,
            'metallicity_grid': metallicity_grid,
            'bc_values': bc_values
        }
    
    def _integrate_bolometric_flux(
        self, 
        wavelengths: np.ndarray, 
        fluxes: np.ndarray
    ) -> float:
        """
        Integrate spectrum to get bolometric flux.
        
        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength array in Angstroms
        fluxes : np.ndarray
            Flux array in erg/s/cm²/Å
            
        Returns
        -------
        float
            Bolometric flux in erg/s/cm²
        """
        # Validate inputs
        if len(wavelengths) != len(fluxes):
            raise ValueError("Wavelength and flux arrays must have same length")
        
        if np.any(fluxes < 0):
            warnings.warn("Negative flux values found, setting to zero")
            fluxes = np.maximum(fluxes, 0)
        
        # Check for sufficient wavelength coverage
        wave_range = wavelengths.max() - wavelengths.min()
        if wave_range < 1000:  # Less than 1000 Å
            warnings.warn("Limited wavelength coverage may affect bolometric flux accuracy")
        
        # Integrate using adaptive method
        try:
            bolometric_flux = adaptive_integration(wavelengths, fluxes)
        except Exception:
            # Fallback to simple trapezoidal rule
            bolometric_flux = trapz(fluxes, wavelengths)
        
        return bolometric_flux


def compute_bolometric_correction(
    teff: float,
    logg: float,
    metallicity: float,
    filter_id: str,
    flux_cube_file: Union[str, Path],
    filter_dir: Union[str, Path],
    **kwargs
) -> float:
    """
    Convenience function to compute a bolometric correction.
    
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
        Additional arguments
        
    Returns
    -------
    float
        Bolometric correction
    """
    from .synthetic import SyntheticPhotometry
    
    bolometric = BolometricCorrections(flux_cube_file)
    photometry = SyntheticPhotometry(flux_cube_file, filter_dir)
    
    return bolometric.compute_bolometric_correction(
        teff, logg, metallicity, filter_id, photometry, **kwargs
    )


# stellar_colors/config.py
"""
Configuration management for stellar-colors package.
"""

import os
from pathlib import Path
from typing import Optional, Union

from astropy.config import ConfigNamespace, ConfigItem

__all__ = ['conf']

# Create configuration namespace
conf = ConfigNamespace()

# Data directories
conf.data_dir = ConfigItem(
    default=str(Path.home() / '.stellar_colors'),
    description='Default directory for storing stellar colors data',
    cfgtype='string'
)

conf.models_dir = ConfigItem(
    default='models',
    description='Subdirectory for stellar atmosphere models (relative to data_dir)',
    cfgtype='string'
)

conf.filters_dir = ConfigItem(
    default='filters',
    description='Subdirectory for filter transmission curves (relative to data_dir)',
    cfgtype='string'
)

conf.cubes_dir = ConfigItem(
    default='cubes',
    description='Subdirectory for flux cubes (relative to data_dir)',
    cfgtype='string'
)

# Download settings
conf.max_download_workers = ConfigItem(
    default=5,
    description='Maximum number of parallel download workers',
    cfgtype='integer'
)

conf.download_timeout = ConfigItem(
    default=30.0,
    description='Timeout for individual downloads in seconds',
    cfgtype='float'
)

conf.cache_duration = ConfigItem(
    default=24,
    description='Cache duration for filter lists in hours',
    cfgtype='integer'
)

# Interpolation settings
conf.default_interpolation_method = ConfigItem(
    default='linear',
    description='Default interpolation method for flux cubes',
    cfgtype='option',
    options=['linear', 'nearest', 'cubic']
)

conf.wavelength_resolution = ConfigItem(
    default=None,
    description='Default wavelength resolution for flux cubes in Angstroms',
    cfgtype='float'
)

# Integration settings
conf.integration_method = ConfigItem(
    default='auto',
    description='Default numerical integration method',
    cfgtype='option',
    options=['auto', 'trapz', 'simpson', 'romberg']
)

# Photometry settings
conf.magnitude_system = ConfigItem(
    default='vega',
    description='Default magnitude system',
    cfgtype='option',
    options=['vega', 'ab']
)

conf.vega_spectrum_url = ConfigItem(
    default='http://ssb.stsci.edu/cdbs/current_calspec/alpha_lyr_stis_010.fits',
    description='URL for default Vega spectrum',
    cfgtype='string'
)

# Performance settings
conf.max_cube_memory_gb = ConfigItem(
    default=4.0,
    description='Maximum memory usage for flux cubes in GB',
    cfgtype='float'
)

conf.enable_parallel_processing = ConfigItem(
    default=True,
    description='Enable parallel processing where possible',
    cfgtype='boolean'
)

# Logging settings
conf.log_level = ConfigItem(
    default='INFO',
    description='Logging level',
    cfgtype='option',
    options=['DEBUG', 'INFO', 'WARNING', 'ERROR']
)

conf.show_progress_bars = ConfigItem(
    default=True,
    description='Show progress bars for long operations',
    cfgtype='boolean'
)


def get_data_dir() -> Path:
    """Get the configured data directory, creating it if necessary."""
    data_dir = Path(conf.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_models_dir() -> Path:
    """Get the models directory, creating it if necessary."""
    models_dir = get_data_dir() / conf.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_filters_dir() -> Path:
    """Get the filters directory, creating it if necessary."""
    filters_dir = get_data_dir() / conf.filters_dir
    filters_dir.mkdir(parents=True, exist_ok=True)
    return filters_dir


def get_cubes_dir() -> Path:
    """Get the cubes directory, creating it if necessary."""
    cubes_dir = get_data_dir() / conf.cubes_dir
    cubes_dir.mkdir(parents=True, exist_ok=True)
    return cubes_dir


# stellar_colors/atmosphere/models.py
"""
Stellar atmosphere model handling and validation.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import ascii

__all__ = ['ModelCollection', 'validate_model_collection']


class ModelCollection:
    """
    A class for managing collections of stellar atmosphere models.
    
    This class provides methods to validate, analyze, and organize
    stellar atmosphere model collections.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing the model collection
    """
    
    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        self.lookup_file = self.model_dir / 'lookup_table.csv'
        if not self.lookup_file.exists():
            raise FileNotFoundError(f"Lookup table not found: {self.lookup_file}")
        
        self._load_lookup_table()
    
    def _load_lookup_table(self):
        """Load and validate the lookup table."""
        try:
            self.lookup_table = pd.read_csv(self.lookup_file, comment='#')
        except Exception as e:
            raise RuntimeError(f"Failed to load lookup table: {e}")
        
        # Standardize column names
        self._standardize_columns()
        
        # Validate required columns
        required_columns = ['filename', 'teff', 'logg', 'metallicity']
        missing_columns = [col for col in required_columns if col not in self.lookup_table.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        self.lookup_table = self.lookup_table.dropna(subset=required_columns)
    
    def _standardize_columns(self):
        """Standardize column names across different model formats."""
        column_mapping = {
            'file_name': 'filename',
            'teff': 'teff',
            'logg': 'logg', 
            'log_g': 'logg',
            'meta': 'metallicity',
            'feh': 'metallicity',
            '[M/H]': 'metallicity',
            '[Fe/H]': 'metallicity'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in self.lookup_table.columns:
                self.lookup_table = self.lookup_table.rename(columns={old_name: new_name})
    
    def validate_collection(self) -> Dict[str, any]:
        """
        Validate the model collection for completeness and consistency.
        
        Returns
        -------
        Dict[str, any]
            Validation report with issues and statistics
        """
        report = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'missing_files': [],
            'parameter_coverage': {}
        }
        
        # Check file existence
        for _, row in self.lookup_table.iterrows():
            file_path = self.model_dir / row['filename']
            if not file_path.exists():
                report['missing_files'].append(row['filename'])
                report['valid'] = False
        
        # Check parameter ranges
        for param in ['teff', 'logg', 'metallicity']:
            values = self.lookup_table[param].values
            
            # Check for invalid values
            invalid_mask = ~np.isfinite(values)
            if np.any(invalid_mask):
                n_invalid = np.sum(invalid_mask)
                report['issues'].append(f"{n_invalid} invalid {param} values")
                report['valid'] = False
            
            # Parameter statistics
            valid_values = values[~invalid_mask]
            if len(valid_values) > 0:
                report['parameter_coverage'][param] = {
                    'min': float(valid_values.min()),
                    'max': float(valid_values.max()),
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std()),
                    'n_unique': len(np.unique(valid_values))
                }
        
        # Check for reasonable parameter ranges
        self._check_parameter_ranges(report)
        
        # Check wavelength coverage (sample a few files)
        self._check_wavelength_coverage(report)
        
        # Overall statistics
        report['statistics'] = {
            'n_models': len(self.lookup_table),
            'n_missing_files': len(report['missing_files']),
            'n_issues': len(report['issues']),
            'n_warnings': len(report['warnings'])
        }
        
        return report
    
    def _check_parameter_ranges(self, report: Dict):
        """Check if parameter ranges are reasonable for stellar atmospheres."""
        # Typical ranges for stellar atmosphere models
        typical_ranges = {
            'teff': (2000, 50000),
            'logg': (-1.0, 6.0),
            'metallicity': (-5.0, 1.0)
        }
        
        for param, (min_expected, max_expected) in typical_ranges.items():
            if param in report['parameter_coverage']:
                coverage = report['parameter_coverage'][param]
                
                if coverage['min'] < min_expected:
                    report['warnings'].append(
                        f"{param} minimum ({coverage['min']:.2f}) below typical range"
                    )
                
                if coverage['max'] > max_expected:
                    report['warnings'].append(
                        f"{param} maximum ({coverage['max']:.2f}) above typical range"
                    )
    
    def _check_wavelength_coverage(self, report: Dict):
        """Check wavelength coverage by sampling a few models."""  
        sample_size = min(5, len(self.lookup_table))
        sample_models = self.lookup_table.sample(sample_size)
        
        wavelength_ranges = []
        for _, row in sample_models.iterrows():
            try:
                file_path = self.model_dir / row['filename']
                if file_path.exists():
                    # Try to load the spectrum
                    data = np.loadtxt(file_path, comments='#', max_rows=10)
                    if data.shape[1] >= 2:
                        wavelengths = data[:, 0]
                        # Try to load full spectrum for range
                        full_data = np.loadtxt(file_path, comments='#')
                        full_wavelengths = full_data[:, 0]
                        wavelength_ranges.append((
                            float(full_wavelengths.min()),
                            float(full_wavelengths.max())
                        ))
            except Exception:
                continue
        
        if wavelength_ranges:
            min_waves, max_waves = zip(*wavelength_ranges)
            report['wavelength_coverage'] = {
                'min': min(min_waves),
                'max': max(max_waves),
                'common_min': max(min_waves),
                'common_max': min(max_waves)
            }
            
            # Check for reasonable wavelength coverage
            total_range = max(max_waves) - min(min_waves)
            if total_range < 1000:  # Less than 1000 Å
                report['warnings'].append("Limited wavelength coverage detected")
        else:
            report['issues'].append("Could not determine wavelength coverage")
    
    def get_parameter_grid_info(self) -> Dict[str, any]:
        """
        Analyze the parameter space grid structure.
        
        Returns
        -------
        Dict[str, any]
            Information about parameter grid regularity and coverage
        """
        grid_info = {}
        
        for param in ['teff', 'logg', 'metallicity']:
            values = self.lookup_table[param].values
            unique_values = np.unique(values)
            
            # Check if values form a regular grid
            if len(unique_values) > 2:
                spacings = np.diff(unique_values)
                is_regular = np.allclose(spacings, spacings[0], rtol=0.1)
                typical_spacing = np.median(spacings)
            else:
                is_regular = True
                typical_spacing = 0 if len(unique_values) <= 1 else spacings[0]
            
            grid_info[param] = {
                'unique_values': unique_values.tolist(),
                'n_unique': len(unique_values),
                'is_regular': is_regular,
                'typical_spacing': float(typical_spacing),
                'range': (float(unique_values.min()), float(unique_values.max()))
            }
        
        # Check for complete grid coverage
        n_combinations = (
            grid_info['teff']['n_unique'] *
            grid_info['logg']['n_unique'] *
            grid_info['metallicity']['n_unique']
        )
        
        grid_info['completeness'] = {
            'n_expected': n_combinations,
            'n_actual': len(self.lookup_table),
            'completeness_fraction': len(self.lookup_table) / n_combinations if n_combinations > 0 else 0
        }
        
        return grid_info
    
    def suggest_improvements(self) -> List[str]:
        """
        Suggest improvements for the model collection.
        
        Returns
        -------
        List[str]
            List of suggested improvements
        """
        suggestions = []
        
        # Validate collection first
        validation = self.validate_collection()
        
        if validation['missing_files']:
            suggestions.append(f"Fix {len(validation['missing_files'])} missing model files")
        
        # Check parameter coverage
        grid_info = self.get_parameter_grid_info()
        
        for param in ['teff', 'logg', 'metallicity']:
            if grid_info[param]['n_unique'] < 5:
                suggestions.append(f"Increase {param} coverage (only {grid_info[param]['n_unique']} values)")
        
        # Check completeness
        completeness = grid_info['completeness']['completeness_fraction']
        if completeness < 0.8:
            suggestions.append(f"Fill missing parameter combinations (grid is {completeness:.1%} complete)")
        
        # Check wavelength coverage
        if 'wavelength_coverage' in validation:
            wl_coverage = validation['wavelength_coverage']
            total_range = wl_coverage['max'] - wl_coverage['min']
            if total_range < 5000:  # Less than 5000 Å
                suggestions.append("Extend wavelength coverage for better photometry accuracy")
        
        return suggestions


def validate_model_collection(model_dir: Union[str, Path]) -> Dict[str, any]:
    """
    Convenience function to validate a stellar atmosphere model collection.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing the model collection
        
    Returns
    -------
    Dict[str, any]
        Validation report
    """
    collection = ModelCollection(model_dir)
    return collection.validate_collection()