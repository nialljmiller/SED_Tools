# stellar_colors/cube/builder.py
"""
Data cube builder and flux cube classes for stellar atmosphere interpolation.
"""

import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ['DataCubeBuilder', 'FluxCube', 'build_flux_cube']


class FluxCube:
    """
    A class for querying precomputed stellar flux cubes with interpolation.
    
    This class loads flux cube data from HDF5 files and provides methods
    for interpolating stellar spectra at arbitrary stellar parameters.
    
    Parameters
    ----------
    cube_file : str or Path
        Path to HDF5 flux cube file
    """
    
    def __init__(self, cube_file: Union[str, Path]):
        self.cube_file = Path(cube_file)
        
        if not self.cube_file.exists():
            raise FileNotFoundError(f"Flux cube file not found: {cube_file}")
        
        # Load cube data from HDF5
        self._load_cube_data()
        
        # Set up interpolators
        self._setup_interpolators()
    
    def _load_cube_data(self):
        """Load flux cube data from HDF5 file."""
        logger.info(f"Loading flux cube from {self.cube_file}")
        
        try:
            with h5py.File(self.cube_file, 'r') as f:
                # Load parameter grids
                grids = f['grids']
                self.teff_grid = grids['teff'][:]
                self.logg_grid = grids['logg'][:]
                self.meta_grid = grids['metallicity'][:]
                self.wavelength_grid = grids['wavelength'][:]
                
                # Load flux cube
                self.flux_cube = f['flux_cube'][:]
                
                # Load metadata if available
                self.format_version = f.attrs.get('format_version', '1.0')
                
                logger.info(f"Loaded flux cube: "
                           f"{len(self.teff_grid)} × {len(self.logg_grid)} × "
                           f"{len(self.meta_grid)} models, "
                           f"{len(self.wavelength_grid)} wavelength points")
                
        except Exception as e:
            raise IOError(f"Failed to load flux cube: {e}") from e
    
    def _setup_interpolators(self):
        """Set up scipy interpolators for different methods."""
        # Create coordinate grids for RegularGridInterpolator
        self._coordinates = (self.teff_grid, self.logg_grid, self.meta_grid)
        
        # Store grid info for bounds checking
        self._teff_min, self._teff_max = self.teff_grid.min(), self.teff_grid.max()
        self._logg_min, self._logg_max = self.logg_grid.min(), self.logg_grid.max()
        self._meta_min, self._meta_max = self.meta_grid.min(), self.meta_grid.max()
        
        logger.debug("Set up interpolators for flux cube")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the flux cube."""
        return self.flux_cube.shape
    
    @property
    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter ranges covered by the flux cube.
        
        Returns
        -------
        dict
            Dictionary with parameter ranges
        """
        return {
            'teff': (self._teff_min, self._teff_max),
            'logg': (self._logg_min, self._logg_max),
            'metallicity': (self._meta_min, self._meta_max),
            'wavelength': (self.wavelength_grid.min(), self.wavelength_grid.max())
        }
    
    def _validate_parameters(self, teff: float, logg: float, metallicity: float):
        """Validate that parameters are within cube bounds."""
        ranges = self.parameter_ranges
        
        if not (ranges['teff'][0] <= teff <= ranges['teff'][1]):
            logger.warning(f"Teff {teff} outside cube range {ranges['teff']}")
        
        if not (ranges['logg'][0] <= logg <= ranges['logg'][1]):
            logger.warning(f"log g {logg} outside cube range {ranges['logg']}")
        
        if not (ranges['metallicity'][0] <= metallicity <= ranges['metallicity'][1]):
            logger.warning(f"Metallicity {metallicity} outside cube range {ranges['metallicity']}")
    
    def interpolate_spectrum(
        self, 
        teff: float, 
        logg: float, 
        metallicity: float,
        method: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate a stellar spectrum at given parameters.
        
        Parameters
        ----------
        teff : float
            Effective temperature in K
        logg : float
            Surface gravity (log g)
        metallicity : float
            Metallicity [M/H]
        method : str
            Interpolation method ('linear', 'nearest', 'cubic')
            
        Returns
        -------
        wavelengths : np.ndarray
            Wavelength array in Angstroms
        fluxes : np.ndarray
            Interpolated flux array in erg/s/cm²/Å
        """
        self._validate_parameters(teff, logg, metallicity)
        
        # Query point
        query_point = np.array([teff, logg, metallicity])
        
        if method == 'nearest':
            fluxes = self._nearest_neighbor_interpolation(query_point)
        elif method in ['linear', 'cubic']:
            fluxes = self._regular_grid_interpolation(query_point, method)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        return self.wavelength_grid.copy(), fluxes
    
    def _nearest_neighbor_interpolation(self, query_point: np.ndarray) -> np.ndarray:
        """Perform nearest neighbor interpolation."""
        # Find nearest indices
        teff_idx = np.argmin(np.abs(self.teff_grid - query_point[0]))
        logg_idx = np.argmin(np.abs(self.logg_grid - query_point[1]))
        meta_idx = np.argmin(np.abs(self.meta_grid - query_point[2]))
        
        return self.flux_cube[teff_idx, logg_idx, meta_idx, :].copy()
    
    def _regular_grid_interpolation(self, query_point: np.ndarray, method: str) -> np.ndarray:
        """Perform regular grid interpolation using scipy."""
        # Clamp query point to grid bounds
        clamped_point = np.array([
            np.clip(query_point[0], self._teff_min, self._teff_max),
            np.clip(query_point[1], self._logg_min, self._logg_max),
            np.clip(query_point[2], self._meta_min, self._meta_max)
        ])
        
        # Interpolate each wavelength separately (more memory efficient)
        n_wavelength = len(self.wavelength_grid)
        interpolated_fluxes = np.zeros(n_wavelength)
        
        for i in range(n_wavelength):
            # Create interpolator for this wavelength
            interpolator = RegularGridInterpolator(
                self._coordinates,
                self.flux_cube[:, :, :, i],
                method=method,
                bounds_error=False,
                fill_value=None
            )
            
            # Interpolate at query point
            interpolated_fluxes[i] = interpolator(clamped_point)
        
        return interpolated_fluxes
    
    def interpolate_at_wavelength(
        self,
        wavelengths: Union[float, np.ndarray],
        teff: float,
        logg: float,
        metallicity: float,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        Interpolate flux at specific wavelengths.
        
        Parameters
        ----------
        wavelengths : float or array
            Target wavelength(s) in Angstroms
        teff : float
            Effective temperature in K
        logg : float
            Surface gravity (log g)
        metallicity : float
            Metallicity [M/H]
        method : str
            Interpolation method ('linear', 'nearest', 'cubic')
            
        Returns
        -------
        fluxes : float or np.ndarray
            Interpolated flux(es) in erg/s/cm²/Å
        """
        # Get full interpolated spectrum
        wave_grid, flux_grid = self.interpolate_spectrum(teff, logg, metallicity, method)
        
        # Interpolate to requested wavelengths
        target_wavelengths = np.atleast_1d(wavelengths)
        interpolated_fluxes = np.interp(
            target_wavelengths, wave_grid, flux_grid,
            left=0.0, right=0.0  # Zero flux outside wavelength range
        )
        
        # Return scalar if input was scalar
        if np.isscalar(wavelengths):
            return float(interpolated_fluxes[0])
        else:
            return interpolated_fluxes


class DataCubeBuilder:
    """
    A class for building flux cubes from stellar atmosphere model collections.
    
    This class reads model collections, creates regular parameter grids,
    and builds HDF5 flux cubes for efficient interpolation.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing stellar atmosphere models
    """
    
    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load lookup table
        self._load_lookup_table()
        
        logger.info(f"Initialized DataCubeBuilder with {len(self.lookup_table)} models")
    
    def _load_lookup_table(self):
        """Load the model lookup table."""
        lookup_file = self.model_dir / 'lookup_table.csv'
        
        if not lookup_file.exists():
            raise FileNotFoundError(f"Lookup table not found: {lookup_file}")
        
        try:
            self.lookup_table = pd.read_csv(lookup_file, comment='#')
        except Exception as e:
            raise IOError(f"Failed to load lookup table: {e}") from e
        
        # Validate required columns
        required_columns = ['filename', 'teff', 'logg', 'metallicity']
        missing_columns = set(required_columns) - set(self.lookup_table.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def analyze_grid_structure(self) -> Dict:
        """
        Analyze the parameter space structure of the model collection.
        
        Returns
        -------
        dict
            Analysis of parameter coverage
        """
        analysis = {}
        
        for param in ['teff', 'logg', 'metallicity']:
            values = self.lookup_table[param].values
            analysis[param] = {
                'range': (values.min(), values.max()),
                'n_unique': len(np.unique(values)),
                'unique_values': sorted(np.unique(values)),
                'n_total': len(values)
            }
        
        return analysis
    
    def create_regular_grid(
        self,
        teff_points: Optional[int] = None,
        logg_points: Optional[int] = None,
        metallicity_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create regular parameter grids for interpolation.
        
        Parameters
        ----------
        teff_points : int, optional
            Number of Teff grid points (default: use unique values)
        logg_points : int, optional
            Number of log g grid points (default: use unique values)
        metallicity_points : int, optional
            Number of metallicity grid points (default: use unique values)
            
        Returns
        -------
        teff_grid : np.ndarray
            Temperature grid
        logg_grid : np.ndarray
            Surface gravity grid
        metallicity_grid : np.ndarray
            Metallicity grid
        """
        # Use unique values by default
        if teff_points is None:
            teff_grid = np.sort(self.lookup_table['teff'].unique())
        else:
            teff_min, teff_max = self.lookup_table['teff'].min(), self.lookup_table['teff'].max()
            teff_grid = np.linspace(teff_min, teff_max, teff_points)
        
        if logg_points is None:
            logg_grid = np.sort(self.lookup_table['logg'].unique())
        else:
            logg_min, logg_max = self.lookup_table['logg'].min(), self.lookup_table['logg'].max()
            logg_grid = np.linspace(logg_min, logg_max, logg_points)
        
        if metallicity_points is None:
            metallicity_grid = np.sort(self.lookup_table['metallicity'].unique())
        else:
            meta_min, meta_max = self.lookup_table['metallicity'].min(), self.lookup_table['metallicity'].max()
            metallicity_grid = np.linspace(meta_min, meta_max, metallicity_points)
        
        return teff_grid, logg_grid, metallicity_grid
    
    def create_wavelength_grid(
        self,
        n_points: Optional[int] = None,
        wavelength_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Create a common wavelength grid from the model collection.
        
        Parameters
        ----------
        n_points : int, optional
            Number of wavelength points (default: use first model)
        wavelength_range : tuple, optional
            Wavelength range as (min, max) in Angstroms
            
        Returns
        -------
        wavelength_grid : np.ndarray
            Common wavelength grid in Angstroms
        """
        # Load first model to get wavelength range
        first_file = self.lookup_table.iloc[0]['filename']
        first_path = self.model_dir / first_file
        
        wavelengths, _ = self._load_spectrum_file(first_path)
        
        if wavelength_range:
            wave_min, wave_max = wavelength_range
        else:
            wave_min, wave_max = wavelengths.min(), wavelengths.max()
        
        if n_points is None:
            # Use original grid within range
            mask = (wavelengths >= wave_min) & (wavelengths <= wave_max)
            return wavelengths[mask]
        else:
            return np.linspace(wave_min, wave_max, n_points)
    
    def _load_spectrum_file(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load wavelength and flux from a spectrum file."""
        try:
            data = np.loadtxt(filepath)
            if data.shape[1] < 2:
                raise ValueError(f"Spectrum file must have at least 2 columns: {filepath}")
            
            wavelengths = data[:, 0]
            fluxes = data[:, 1]
            
            return wavelengths, fluxes
        
        except Exception as e:
            raise IOError(f"Failed to load spectrum {filepath}: {e}") from e
    
    def build_cube(
        self,
        output_file: Union[str, Path],
        format: str = 'hdf5',
        wavelength_range: Optional[Tuple[float, float]] = None,
        compression: bool = True,
        fill_missing: bool = True,
        show_progress: bool = True
    ) -> Path:
        """
        Build a flux cube from the model collection.
        
        Parameters
        ----------
        output_file : str or Path
            Output file path
        format : str
            Output format ('hdf5' or 'binary')
        wavelength_range : tuple, optional
            Wavelength range to include (min, max) in Angstroms
        compression : bool
            Enable compression for HDF5 output
        fill_missing : bool
            Fill missing grid points with interpolated values
        show_progress : bool
            Show progress bars
            
        Returns
        -------
        Path
            Path to created flux cube file
        """
        output_file = Path(output_file)
        
        logger.info(f"Building flux cube: {output_file}")
        
        # Create parameter grids
        teff_grid, logg_grid, meta_grid = self.create_regular_grid()
        wavelength_grid = self.create_wavelength_grid(wavelength_range=wavelength_range)
        
        # Initialize flux cube
        cube_shape = (len(teff_grid), len(logg_grid), len(meta_grid), len(wavelength_grid))
        flux_cube = np.zeros(cube_shape)
        filled_cube = np.zeros(cube_shape[:3], dtype=bool)
        
        logger.info(f"Flux cube shape: {cube_shape}")
        
        # Fill cube with model data
        iterator = tqdm(self.lookup_table.iterrows(), total=len(self.lookup_table)) if show_progress else self.lookup_table.iterrows()
        
        for idx, row in iterator:
            try:
                # Load spectrum
                filepath = self.model_dir / row['filename']
                wavelengths, fluxes = self._load_spectrum_file(filepath)
                
                # Interpolate to common wavelength grid
                interp_fluxes = np.interp(wavelength_grid, wavelengths, fluxes, left=0, right=0)
                
                # Find grid indices
                teff_idx = np.argmin(np.abs(teff_grid - row['teff']))
                logg_idx = np.argmin(np.abs(logg_grid - row['logg']))
                meta_idx = np.argmin(np.abs(meta_grid - row['metallicity']))
                
                # Store in cube
                flux_cube[teff_idx, logg_idx, meta_idx, :] = interp_fluxes
                filled_cube[teff_idx, logg_idx, meta_idx] = True
                
            except Exception as e:
                logger.warning(f"Failed to process {row['filename']}: {e}")
                continue
        
        # Fill missing grid points if requested
        if fill_missing:
            self._fill_missing_points(flux_cube, filled_cube, teff_grid, logg_grid, meta_grid)
        
        # Save cube
        if format.lower() == 'hdf5':
            self._save_hdf5_cube(
                output_file, teff_grid, logg_grid, meta_grid, wavelength_grid, 
                flux_cube, compression
            )
        elif format.lower() == 'binary':
            self._save_binary_cube(
                output_file, teff_grid, logg_grid, meta_grid, wavelength_grid, flux_cube
            )
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Successfully built flux cube: {output_file}")
        
        return output_file
    
    def _fill_missing_points(self, flux_cube, filled_cube, teff_grid, logg_grid, meta_grid):
        """Fill missing grid points using nearest neighbor interpolation."""
        logger.info("Filling missing grid points...")
        
        # Find filled and missing points
        filled_indices = np.where(filled_cube)
        missing_indices = np.where(~filled_cube)
        
        if len(missing_indices[0]) == 0:
            logger.info("No missing points to fill")
            return
        
        # Create coordinate arrays for filled points
        filled_coords = np.column_stack([
            teff_grid[filled_indices[0]],
            logg_grid[filled_indices[1]],
            meta_grid[filled_indices[2]]
        ])
        
        # Coordinates for missing points
        missing_coords = np.column_stack([
            teff_grid[missing_indices[0]],
            logg_grid[missing_indices[1]],
            meta_grid[missing_indices[2]]
        ])
        
        # Build KDTree for nearest neighbor search
        tree = cKDTree(filled_coords)
        
        # Find nearest neighbors for missing points
        _, nearest_indices = tree.query(missing_coords)
        
        # Fill missing points with nearest neighbor values
        for i, (teff_idx, logg_idx, meta_idx) in enumerate(zip(*missing_indices)):
            nearest_filled_idx = nearest_indices[i]
            nearest_teff_idx = filled_indices[0][nearest_filled_idx]
            nearest_logg_idx = filled_indices[1][nearest_filled_idx]
            nearest_meta_idx = filled_indices[2][nearest_filled_idx]
            
            flux_cube[teff_idx, logg_idx, meta_idx, :] = flux_cube[nearest_teff_idx, nearest_logg_idx, nearest_meta_idx, :]
        
        logger.info(f"Filled {len(missing_indices[0])} missing grid points")
    
    def _save_hdf5_cube(self, output_file, teff_grid, logg_grid, meta_grid, wavelength_grid, flux_cube, compression):
        """Save flux cube to HDF5 format."""
        compression_opts = {'compression': 'gzip', 'compression_opts': 9} if compression else {}
        
        with h5py.File(output_file, 'w') as f:
            # Create groups
            grids_group = f.create_group('grids')
            
            # Save parameter grids
            grids_group.create_dataset('teff', data=teff_grid, **compression_opts)
            grids_group.create_dataset('logg', data=logg_grid, **compression_opts)
            grids_group.create_dataset('metallicity', data=meta_grid, **compression_opts)
            grids_group.create_dataset('wavelength', data=wavelength_grid, **compression_opts)
            
            # Save flux cube
            f.create_dataset('flux_cube', data=flux_cube, **compression_opts)
            
            # Save metadata
            f.attrs['format_version'] = '1.0'
            f.attrs['cube_shape'] = flux_cube.shape
            f.attrs['parameter_ranges'] = {
                'teff': [teff_grid.min(), teff_grid.max()],
                'logg': [logg_grid.min(), logg_grid.max()],
                'metallicity': [meta_grid.min(), meta_grid.max()],
                'wavelength': [wavelength_grid.min(), wavelength_grid.max()]
            }
    
    def _save_binary_cube(self, output_file, teff_grid, logg_grid, meta_grid, wavelength_grid, flux_cube):
        """Save flux cube to binary format (MESA compatible)."""
        with open(output_file, 'wb') as f:
            # Write dimensions
            f.write(struct.pack('4i', len(teff_grid), len(logg_grid), len(meta_grid), len(wavelength_grid)))
            
            # Write grids
            teff_grid.astype(np.float64).tofile(f)
            logg_grid.astype(np.float64).tofile(f)
            meta_grid.astype(np.float64).tofile(f)
            wavelength_grid.astype(np.float64).tofile(f)
            
            # Write flux cube (transpose for FORTRAN column-major order)
            flux_cube.transpose(3, 2, 1, 0).astype(np.float64).tofile(f)


def build_flux_cube(
    model_dir: Union[str, Path],
    output_file: Union[str, Path],
    **kwargs
) -> Path:
    """
    Convenience function to build a flux cube from a model directory.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing stellar atmosphere models
    output_file : str or Path
        Output flux cube file path
    **kwargs
        Additional arguments passed to DataCubeBuilder.build_cube()
        
    Returns
    -------
    Path
        Path to created flux cube file
    """
    builder = DataCubeBuilder(model_dir)
    return builder.build_cube(output_file, **kwargs)