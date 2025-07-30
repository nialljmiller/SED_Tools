"""
Data cube builder for stellar atmosphere model interpolation.

This module provides tools to create and manage 4D flux cubes (Teff, log g, [M/H], wavelength)
from stellar atmosphere model collections, enabling fast interpolation for synthetic photometry.
"""

import struct
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import ascii
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from tqdm import tqdm

__all__ = ['DataCubeBuilder', 'FluxCube', 'build_flux_cube']


class FluxCube:
    """
    A class for managing and querying stellar atmosphere flux cubes.
    
    This class provides efficient access to precomputed stellar atmosphere model
    grids, enabling fast interpolation for synthetic photometry calculations.
    
    Parameters
    ----------
    cube_file : str or Path
        Path to the flux cube file (HDF5 or binary format)
    """
    
    def __init__(self, cube_file: Union[str, Path]):
        self.cube_file = Path(cube_file)
        
        if not self.cube_file.exists():
            raise FileNotFoundError(f"Cube file not found: {cube_file}")
        
        self._load_cube()
    
    def _load_cube(self):
        """Load the flux cube from file."""
        if self.cube_file.suffix == '.h5':
            self._load_hdf5()
        elif self.cube_file.suffix == '.bin':
            self._load_binary()
        else:
            raise ValueError(f"Unsupported cube format: {self.cube_file.suffix}")
    
    def _load_hdf5(self):
        """Load from HDF5 format."""
        with h5py.File(self.cube_file, 'r') as f:
            self.teff_grid = f['grids/teff'][:]
            self.logg_grid = f['grids/logg'][:]
            self.meta_grid = f['grids/metallicity'][:]
            self.wavelength_grid = f['grids/wavelength'][:]
            
            # Load flux cube in chunks to manage memory
            self.flux_cube = f['flux_cube'][:]
            
            # Load metadata
            self.metadata = dict(f.attrs)
    
    def _load_binary(self):
        """Load from binary format (legacy support)."""
        with open(self.cube_file, 'rb') as f:
            # Read dimensions
            n_teff, n_logg, n_meta, n_lambda = struct.unpack('4i', f.read(16))
            
            # Read grids
            self.teff_grid = np.frombuffer(f.read(8 * n_teff), dtype=np.float64)
            self.logg_grid = np.frombuffer(f.read(8 * n_logg), dtype=np.float64)
            self.meta_grid = np.frombuffer(f.read(8 * n_meta), dtype=np.float64)
            self.wavelength_grid = np.frombuffer(f.read(8 * n_lambda), dtype=np.float64)
            
            # Read flux cube
            flux_size = n_teff * n_logg * n_meta * n_lambda
            self.flux_cube = np.frombuffer(
                f.read(8 * flux_size), dtype=np.float64
            ).reshape(n_teff, n_logg, n_meta, n_lambda)
            
            self.metadata = {'format': 'binary'}
    
    @property
    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges of the cube."""
        return {
            'teff': (self.teff_grid.min(), self.teff_grid.max()),
            'logg': (self.logg_grid.min(), self.logg_grid.max()),
            'metallicity': (self.meta_grid.min(), self.meta_grid.max()),
            'wavelength': (self.wavelength_grid.min(), self.wavelength_grid.max())
        }
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Get the shape of the flux cube."""
        return self.flux_cube.shape
    
    def interpolate_spectrum(
        self,
        teff: float,
        logg: float,
        metallicity: float,
        method: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        method : str, optional
            Interpolation method ('linear', 'nearest', 'cubic')
            
        Returns
        -------
        wavelengths : np.ndarray
            Wavelength array in Angstroms
        fluxes : np.ndarray
            Flux array in erg/s/cm²/Å
        """
        # Check if parameters are within grid bounds
        ranges = self.parameter_ranges
        if not (ranges['teff'][0] <= teff <= ranges['teff'][1]):
            warnings.warn(f"Teff {teff} outside grid range {ranges['teff']}")
        if not (ranges['logg'][0] <= logg <= ranges['logg'][1]):
            warnings.warn(f"log g {logg} outside grid range {ranges['logg']}")
        if not (ranges['metallicity'][0] <= metallicity <= ranges['metallicity'][1]):
            warnings.warn(f"[M/H] {metallicity} outside grid range {ranges['metallicity']}")
        
        # Create interpolator for each wavelength point
        interpolated_flux = np.zeros(len(self.wavelength_grid))
        
        if method == 'linear':
            # Use scipy's RegularGridInterpolator for linear interpolation
            for i, wavelength in enumerate(self.wavelength_grid):
                flux_slice = self.flux_cube[:, :, :, i]
                
                interpolator = RegularGridInterpolator(
                    (self.teff_grid, self.logg_grid, self.meta_grid),
                    flux_slice,
                    method='linear',
                    bounds_error=False,
                    fill_value=0.0
                )
                
                interpolated_flux[i] = interpolator([teff, logg, metallicity])[0]
        
        elif method == 'nearest':
            # Find nearest grid point
            teff_idx = np.argmin(np.abs(self.teff_grid - teff))
            logg_idx = np.argmin(np.abs(self.logg_grid - logg))
            meta_idx = np.argmin(np.abs(self.meta_grid - metallicity))
            
            interpolated_flux = self.flux_cube[teff_idx, logg_idx, meta_idx, :]
        
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
        
        return self.wavelength_grid.copy(), interpolated_flux
    
    def interpolate_at_wavelength(
        self,
        wavelengths: np.ndarray,
        teff: float,
        logg: float,
        metallicity: float,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        Interpolate flux at specific wavelengths.
        
        Parameters
        ----------
        wavelengths : np.ndarray
            Desired wavelength points in Angstroms
        teff : float
            Effective temperature in K
        logg : float
            Surface gravity (log g)
        metallicity : float
            Metallicity [M/H]
        method : str, optional
            Interpolation method
            
        Returns
        -------
        np.ndarray
            Interpolated flux values
        """
        # First interpolate the full spectrum
        cube_wavelengths, cube_fluxes = self.interpolate_spectrum(
            teff, logg, metallicity, method
        )
        
        # Then interpolate to desired wavelengths
        return np.interp(wavelengths, cube_wavelengths, cube_fluxes)


class DataCubeBuilder:
    """
    A class for building stellar atmosphere flux cubes from model collections.
    
    This class takes a collection of stellar atmosphere models and creates a
    regularly gridded 4D flux cube for efficient interpolation.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing stellar atmosphere models and lookup table
    wavelength_range : tuple, optional
        Wavelength range to include (min_wave, max_wave) in Angstroms
    wavelength_resolution : float, optional
        Wavelength resolution in Angstroms. Default is adaptive.
    """
    
    def __init__(
        self,
        model_dir: Union[str, Path],
        wavelength_range: Optional[Tuple[float, float]] = None,
        wavelength_resolution: Optional[float] = None
    ):
        self.model_dir = Path(model_dir)
        self.wavelength_range = wavelength_range
        self.wavelength_resolution = wavelength_resolution
        
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
        column_mapping = {
            'file_name': 'filename',
            'teff': 'teff',
            'logg': 'logg',
            'meta': 'metallicity',
            'feh': 'metallicity',
            '[M/H]': 'metallicity'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in self.lookup_table.columns:
                self.lookup_table = self.lookup_table.rename(columns={old_name: new_name})
        
        # Validate required columns
        required_columns = ['filename', 'teff', 'logg', 'metallicity']
        missing_columns = [col for col in required_columns if col not in self.lookup_table.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        self.lookup_table = self.lookup_table.dropna(subset=required_columns)
        
        print(f"Loaded lookup table with {len(self.lookup_table)} models")
    
    def analyze_grid_structure(self) -> Dict:
        """
        Analyze the parameter space coverage of the model collection.
        
        Returns
        -------
        Dict
            Grid analysis including parameter ranges and sampling
        """
        analysis = {}
        
        for param in ['teff', 'logg', 'metallicity']:
            values = self.lookup_table[param].values
            unique_values = np.unique(values)
            
            analysis[param] = {
                'range': (values.min(), values.max()),
                'n_unique': len(unique_values),
                'unique_values': unique_values,
                'median_spacing': np.median(np.diff(unique_values)) if len(unique_values) > 1 else 0
            }
        
        # Analyze wavelength coverage by sampling a few models
        sample_models = self.lookup_table.sample(min(10, len(self.lookup_table)))
        wavelength_ranges = []
        
        for _, row in sample_models.iterrows():
            try:
                spec_file = self.model_dir / row['filename']
                wavelengths, _ = self._load_spectrum(spec_file)
                wavelength_ranges.append((wavelengths.min(), wavelengths.max()))
            except Exception:
                continue
        
        if wavelength_ranges:
            min_waves, max_waves = zip(*wavelength_ranges)
            analysis['wavelength'] = {
                'range': (max(min_waves), min(max_waves)),  # Common overlap
                'full_range': (min(min_waves), max(max_waves))
            }
        
        return analysis
    
    def create_regular_grid(
        self,
        teff_points: Optional[int] = None,
        logg_points: Optional[int] = None,
        metallicity_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create regular parameter grids for the flux cube.
        
        Parameters
        ----------
        teff_points : int, optional
            Number of Teff grid points. If None, uses unique values from data.
        logg_points : int, optional
            Number of log g grid points. If None, uses unique values from data.
        metallicity_points : int, optional
            Number of metallicity grid points. If None, uses unique values from data.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Regular grids for Teff, log g, and metallicity
        """
        def create_grid(param_name: str, n_points: Optional[int] = None):
            values = self.lookup_table[param_name].values
            if n_points is None:
                # Use unique values from data
                return np.unique(values)
            else:
                # Create regular grid
                return np.linspace(values.min(), values.max(), n_points)
        
        teff_grid = create_grid('teff', teff_points)
        logg_grid = create_grid('logg', logg_points)
        metallicity_grid = create_grid('metallicity', metallicity_points)
        
        return teff_grid, logg_grid, metallicity_grid
    
    def create_wavelength_grid(
        self,
        n_points: Optional[int] = None,
        sample_size: int = 20
    ) -> np.ndarray:
        """
        Create a common wavelength grid from the model collection.
        
        Parameters
        ----------
        n_points : int, optional
            Number of wavelength points. If None, determined adaptively.
        sample_size : int, optional
            Number of models to sample for wavelength analysis
            
        Returns
        -------
        np.ndarray
            Common wavelength grid in Angstroms
        """
        # Sample models to determine wavelength coverage
        sample_models = self.lookup_table.sample(
            min(sample_size, len(self.lookup_table))
        )
        
        min_wave = float('inf')
        max_wave = 0
        resolutions = []
        
        for _, row in sample_models.iterrows():
            try:
                spec_file = self.model_dir / row['filename']
                wavelengths, _ = self._load_spectrum(spec_file)
                
                # Apply wavelength range filter if specified
                if self.wavelength_range:
                    mask = (
                        (wavelengths >= self.wavelength_range[0]) &
                        (wavelengths <= self.wavelength_range[1])
                    )
                    wavelengths = wavelengths[mask]
                
                if len(wavelengths) > 10:
                    min_wave = min(min_wave, wavelengths.min())
                    max_wave = max(max_wave, wavelengths.max())
                    
                    # Estimate resolution
                    resolution = np.median(np.diff(wavelengths))
                    resolutions.append(resolution)
                    
            except Exception as e:
                warnings.warn(f"Failed to load {row['filename']}: {e}")
                continue
        
        if not resolutions:
            raise RuntimeError("Could not determine wavelength grid from any models")
        
        # Use specified resolution or adaptive resolution
        if self.wavelength_resolution:
            grid_resolution = self.wavelength_resolution
        else:
            grid_resolution = max(1.0, np.median(resolutions) * 2)  # 2x median for safety
        
        # Determine number of points
        if n_points is None:
            n_points = int((max_wave - min_wave) / grid_resolution) + 1
            n_points = min(n_points, 10000)  # Cap at reasonable size
        
        wavelength_grid = np.linspace(min_wave, max_wave, n_points)
        
        print(f"Created wavelength grid: {min_wave:.1f}-{max_wave:.1f} Å")
        print(f"Grid points: {len(wavelength_grid)}, Resolution: {grid_resolution:.1f} Å")
        
        return wavelength_grid
    
    def build_cube(
        self,
        output_file: Union[str, Path],
        format: str = 'hdf5',
        interpolation_method: str = 'linear',
        fill_missing: bool = True,
        compression: bool = True
    ) -> Path:
        """
        Build the flux cube from the model collection.
        
        Parameters
        ----------
        output_file : str or Path
            Path for the output cube file
        format : str, optional
            Output format ('hdf5' or 'binary'). Default is 'hdf5'.
        interpolation_method : str, optional
            Method for filling missing grid points ('linear', 'nearest')
        fill_missing : bool, optional
            Whether to fill missing grid points using interpolation
        compression : bool, optional
            Whether to compress the output file (HDF5 only)
            
        Returns
        -------
        Path
            Path to the created cube file
        """
        output_file = Path(output_file)
        
        # Create parameter grids
        teff_grid, logg_grid, metallicity_grid = self.create_regular_grid()
        wavelength_grid = self.create_wavelength_grid()
        
        print(f"Building flux cube with shape: "
              f"{len(teff_grid)} × {len(logg_grid)} × {len(metallicity_grid)} × {len(wavelength_grid)}")
        
        # Initialize flux cube
        flux_cube = np.zeros((
            len(teff_grid), len(logg_grid), len(metallicity_grid), len(wavelength_grid)
        ))
        filled_mask = np.zeros((
            len(teff_grid), len(logg_grid), len(metallicity_grid)
        ), dtype=bool)
        
        # Build KDTree for efficient nearest neighbor searches
        tree_points = []
        tree_indices = []
        
        for _, row in tqdm(self.lookup_table.iterrows(), 
                          total=len(self.lookup_table), 
                          desc="Loading spectra"):
            # Find grid indices
            teff_idx = np.argmin(np.abs(teff_grid - row['teff']))
            logg_idx = np.argmin(np.abs(logg_grid - row['logg']))
            meta_idx = np.argmin(np.abs(metallicity_grid - row['metallicity']))
            
            try:
                spec_file = self.model_dir / row['filename']
                wavelengths, fluxes = self._load_spectrum(spec_file)
                
                # Interpolate to common wavelength grid
                interpolated_flux = np.interp(
                    wavelength_grid, wavelengths, np.maximum(fluxes, 1e-99)
                )
                
                # Store in cube
                flux_cube[teff_idx, logg_idx, meta_idx, :] = interpolated_flux
                filled_mask[teff_idx, logg_idx, meta_idx] = True
                
                # Store for KDTree
                tree_points.append([row['teff'], row['logg'], row['metallicity']])
                tree_indices.append((teff_idx, logg_idx, meta_idx))
                
            except Exception as e:
                warnings.warn(f"Failed to load {row['filename']}: {e}")
                continue
        
        # Fill missing grid points if requested
        if fill_missing and len(tree_points) > 0:
            self._fill_missing_points(
                flux_cube, filled_mask, 
                teff_grid, logg_grid, metallicity_grid,
                tree_points, tree_indices, interpolation_method
            )
        
        # Save the cube
        if format.lower() == 'hdf5':
            self._save_hdf5(
                output_file, teff_grid, logg_grid, metallicity_grid,
                wavelength_grid, flux_cube, compression
            )
        elif format.lower() == 'binary':
            self._save_binary(
                output_file, teff_grid, logg_grid, metallicity_grid,
                wavelength_grid, flux_cube
            )
        else:
            raise ValueError(f"Unsupported format: {format