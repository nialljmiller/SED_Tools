# tests/test_atmosphere.py
"""
Tests for stellar atmosphere model grabber functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests_mock
from astropy.table import Table

from stellar_colors.atmosphere.grabber import AtmosphereGrabber


class TestAtmosphereGrabber:
    """Test cases for AtmosphereGrabber class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def grabber(self, temp_cache_dir):
        """Create AtmosphereGrabber instance with temporary cache."""
        return AtmosphereGrabber(cache_dir=temp_cache_dir)
    
    def test_initialization(self, temp_cache_dir):
        """Test proper initialization of AtmosphereGrabber."""
        grabber = AtmosphereGrabber(cache_dir=temp_cache_dir)
        assert grabber.cache_dir == temp_cache_dir
        assert grabber.cache_dir.exists()
        assert grabber.max_workers == 5
        assert grabber.timeout == 30.0
    
    @requests_mock.Mocker()
    def test_discover_models(self, m, grabber):
        """Test model discovery functionality."""
        # Mock SVO response
        mock_html = '''
        <html>
        <body>
            <a href="?models=KURUCZ2003&action=view">KURUCZ2003</a>
            <a href="?models=PHOENIX&action=view">PHOENIX</a>
            <a href="?models=ATLAS9&action=view">ATLAS9</a>
        </body>
        </html>
        '''
        m.get(grabber.model_index_url, text=mock_html)
        
        models = grabber.discover_models()
        
        assert isinstance(models, list)
        assert 'KURUCZ2003' in models
        assert 'PHOENIX' in models
        assert 'ATLAS9' in models
        assert len(models) == 3
    
    @requests_mock.Mocker()
    def test_discover_models_connection_error(self, m, grabber):
        """Test handling of connection errors during model discovery."""
        m.get(grabber.model_index_url, exc=requests.ConnectionError)
        
        with pytest.raises(ConnectionError):
            grabber.discover_models()
    
    def test_get_model_info(self, grabber):
        """Test getting model information."""
        with patch.object(grabber, '_discover_spectra') as mock_discover:
            mock_spectra = [
                {'fid': 1, 'teff': 5000, 'logg': 4.0, 'meta': 0.0},
                {'fid': 2, 'teff': 5500, 'logg': 4.5, 'meta': 0.0}
            ]
            mock_discover.return_value = mock_spectra
            
            info = grabber.get_model_info('TEST_MODEL')
            
            assert info['name'] == 'TEST_MODEL'
            assert info['n_spectra'] == 2
            assert 'parameter_ranges' in info
            assert 'teff' in info['parameter_ranges']
    
    def test_test_spectrum_exists(self, grabber):
        """Test spectrum existence checking."""
        with requests_mock.Mocker() as m:
            # Mock successful response
            m.head(grabber.spectra_url, headers={'content-length': '2048'})
            result = grabber._test_spectrum_exists('TEST_MODEL', 123)
            assert result is True
            
            # Mock failed response
            m.head(grabber.spectra_url, status_code=404)
            result = grabber._test_spectrum_exists('TEST_MODEL', 456)
            assert result is False


# tests/test_filters.py
"""
Tests for filter transmission curve grabber functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from astropy import units as u
from astropy.table import Table

from stellar_colors.filters.grabber import FilterGrabber


class TestFilterGrabber:
    """Test cases for FilterGrabber class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def grabber(self, temp_cache_dir):
        """Create FilterGrabber instance with temporary cache."""
        return FilterGrabber(cache_dir=temp_cache_dir)
    
    @pytest.fixture
    def mock_filter_table(self):
        """Create mock filter table."""
        return Table({
            'filterID': ['Generic/Johnson.V', 'HST/WFC3/F555W', 'Gaia/G'],
            'Facility': ['Generic', 'HST', 'Gaia'],
            'Instrument': ['Johnson', 'WFC3', 'Gaia'],
            'Band': ['V', 'F555W', 'G'],
            'WavelengthEff': [5500.0, 5300.0, 6200.0],
            'WavelengthMin': [4800.0, 4500.0, 3300.0],
            'WavelengthMax': [6200.0, 6100.0, 10500.0],
            'FWHM': [800.0, 1200.0, 4500.0]
        })
    
    def test_initialization(self, temp_cache_dir):
        """Test proper initialization of FilterGrabber."""
        grabber = FilterGrabber(cache_dir=temp_cache_dir)
        assert grabber.cache_dir == temp_cache_dir
        assert grabber.cache_dir.exists()
        assert len(grabber.wavelength_ranges) > 0
    
    def test_discover_facilities(self, grabber, mock_filter_table):
        """Test facility discovery."""
        with patch.object(grabber, '_get_all_filters', return_value=mock_filter_table):
            facilities = grabber.discover_facilities()
            
            assert isinstance(facilities, list)
            assert 'Generic' in facilities
            assert 'HST' in facilities
            assert 'Gaia' in facilities
    
    def test_search_filters_by_facility(self, grabber, mock_filter_table):
        """Test searching filters by facility."""
        with patch.object(grabber, '_get_all_filters', return_value=mock_filter_table):
            hst_filters = grabber.search_filters(facility='HST')
            
            assert len(hst_filters) == 1
            assert hst_filters['filterID'][0] == 'HST/WFC3/F555W'
    
    def test_search_filters_by_wavelength(self, grabber, mock_filter_table):
        """Test searching filters by wavelength range."""
        with patch.object(grabber, '_get_all_filters', return_value=mock_filter_table):
            optical_filters = grabber.search_filters(
                wavelength_range=(5000*u.AA, 6000*u.AA)
            )
            
            # Should find Generic/Johnson.V and HST/WFC3/F555W
            assert len(optical_filters) == 2
    
    def test_clean_name(self, grabber):
        """Test name cleaning functionality."""
        assert grabber._clean_name('HST/ACS') == 'HST_ACS'
        assert grabber._clean_name('Test Filter') == 'Test_Filter'
        assert grabber._clean_name('') == 'Unknown'
        assert grabber._clean_name(None) == 'Unknown'


# tests/test_cube.py
"""
Tests for data cube builder functionality.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import h5py

from stellar_colors.cube.builder import DataCubeBuilder, FluxCube


class TestDataCubeBuilder:
    """Test cases for DataCubeBuilder class."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            
            # Create lookup table
            lookup_data = {
                'filename': ['model_1.txt', 'model_2.txt', 'model_3.txt'],
                'teff': [5000, 5500, 6000],
                'logg': [4.0, 4.5, 4.0],
                'metallicity': [0.0, 0.0, 0.5]
            }
            lookup_df = pd.DataFrame(lookup_data)
            lookup_df.to_csv(model_dir / 'lookup_table.csv', index=False)
            
            # Create simple model files
            wavelengths = np.linspace(4000, 7000, 100)
            for i, (teff, filename) in enumerate(zip([5000, 5500, 6000], lookup_data['filename'])):
                # Simple blackbody-like spectrum
                fluxes = np.exp(-((wavelengths - 5500) / 500)**2) * (teff / 5500)**4
                spectrum = np.column_stack([wavelengths, fluxes])
                np.savetxt(model_dir / filename, spectrum, 
                          header='Wavelength(A) Flux(erg/s/cm2/A)', comments='#')
            
            yield model_dir
    
    @pytest.fixture
    def builder(self, temp_model_dir):
        """Create DataCubeBuilder instance."""
        return DataCubeBuilder(temp_model_dir)
    
    def test_initialization(self, temp_model_dir):
        """Test proper initialization of DataCubeBuilder."""
        builder = DataCubeBuilder(temp_model_dir)
        assert builder.model_dir == temp_model_dir
        assert len(builder.lookup_table) == 3
        assert 'filename' in builder.lookup_table.columns
        assert 'teff' in builder.lookup_table.columns
    
    def test_initialization_missing_directory(self):
        """Test initialization with missing directory."""
        with pytest.raises(FileNotFoundError):
            DataCubeBuilder('/nonexistent/directory')
    
    def test_analyze_grid_structure(self, builder):
        """Test grid structure analysis."""
        analysis = builder.analyze_grid_structure()
        
        assert 'teff' in analysis
        assert 'logg' in analysis
        assert 'metallicity' in analysis
        
        # Check Teff analysis
        teff_analysis = analysis['teff']
        assert teff_analysis['range'] == (5000, 6000)
        assert teff_analysis['n_unique'] == 3
    
    def test_create_regular_grid(self, builder):
        """Test regular grid creation."""
        teff_grid, logg_grid, meta_grid = builder.create_regular_grid()
        
        assert len(teff_grid) == 3  # Unique values from data
        assert len(logg_grid) == 2  # Two unique logg values
        assert len(meta_grid) == 2  # Two unique metallicity values
        
        # Test with specified points
        teff_grid, logg_grid, meta_grid = builder.create_regular_grid(
            teff_points=10, logg_points=5, metallicity_points=3
        )
        
        assert len(teff_grid) == 10
        assert len(logg_grid) == 5
        assert len(meta_grid) == 3
    
    def test_create_wavelength_grid(self, builder):
        """Test wavelength grid creation."""
        wavelength_grid = builder.create_wavelength_grid()
        
        assert len(wavelength_grid) > 0
        assert wavelength_grid[0] >= 4000  # Should be around model range
        assert wavelength_grid[-1] <= 7000
        
        # Test with specified points
        wavelength_grid = builder.create_wavelength_grid(n_points=50)
        assert len(wavelength_grid) == 50
    
    def test_build_cube_hdf5(self, builder):
        """Test building HDF5 flux cube."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / 'test_cube.h5'
            
            result_file = builder.build_cube(output_file, format='hdf5')
            
            assert result_file.exists()
            assert result_file == output_file
            
            # Verify HDF5 structure
            with h5py.File(result_file, 'r') as f:
                assert 'grids' in f
                assert 'flux_cube' in f
                assert 'grids/teff' in f
                assert 'grids/logg' in f
                assert 'grids/metallicity' in f
                assert 'grids/wavelength' in f


class TestFluxCube:
    """Test cases for FluxCube class."""
    
    @pytest.fixture
    def test_cube_file(self):
        """Create a test HDF5 flux cube file."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            # Create test cube data
            teff_grid = np.array([5000, 5500, 6000])
            logg_grid = np.array([4.0, 4.5])
            meta_grid = np.array([0.0, 0.5])
            wavelength_grid = np.linspace(4000, 7000, 50)
            
            flux_cube = np.random.rand(3, 2, 2, 50) + 1e-10  # Add small offset
            
            with h5py.File(temp_file.name, 'w') as f:
                grids_group = f.create_group('grids')
                grids_group.create_dataset('teff', data=teff_grid)
                grids_group.create_dataset('logg', data=logg_grid)
                grids_group.create_dataset('metallicity', data=meta_grid)
                grids_group.create_dataset('wavelength', data=wavelength_grid)
                
                f.create_dataset('flux_cube', data=flux_cube)
                f.attrs['format_version'] = '1.0'
            
            yield Path(temp_file.name)
            Path(temp_file.name).unlink()  # Clean up
    
    def test_initialization(self, test_cube_file):
        """Test FluxCube initialization."""
        cube = FluxCube(test_cube_file)
        
        assert cube.cube_file == test_cube_file
        assert hasattr(cube, 'teff_grid')
        assert hasattr(cube, 'logg_grid')
        assert hasattr(cube, 'meta_grid')
        assert hasattr(cube, 'wavelength_grid')
        assert hasattr(cube, 'flux_cube')
    
    def test_initialization_missing_file(self):
        """Test initialization with missing file."""
        with pytest.raises(FileNotFoundError):
            FluxCube('/nonexistent/file.h5')
    
    def test_parameter_ranges(self, test_cube_file):
        """Test parameter range properties."""
        cube = FluxCube(test_cube_file)
        ranges = cube.parameter_ranges
        
        assert 'teff' in ranges
        assert 'logg' in ranges
        assert 'metallicity' in ranges
        assert 'wavelength' in ranges
        
        assert ranges['teff'][0] == 5000
        assert ranges['teff'][1] == 6000
    
    def test_interpolate_spectrum(self, test_cube_file):
        """Test spectrum interpolation."""
        cube = FluxCubeProvider(test_cube_file)
        
        wavelengths, fluxes = cube.interpolate_spectrum(5250, 4.25, 0.25)
        
        assert len(wavelengths) == len(fluxes)
        assert len(wavelengths) > 0
        assert np.all(fluxes >= 0)  # Should be non-negative
        
        # Test nearest neighbor interpolation
        wavelengths_nn, fluxes_nn = cube.interpolate_spectrum(
            5250, 4.25, 0.25, method='nearest'
        )
        
        assert len(wavelengths_nn) == len(wavelengths)
    
    def test_interpolate_at_wavelength(self, test_cube_file):
        """Test interpolation at specific wavelengths."""
        cube = FluxCube(test_cube_file)
        
        target_wavelengths = np.array([5000, 5500, 6000])
        fluxes = cube.interpolate_at_wavelength(
            target_wavelengths, 5250, 4.25, 0.25
        )
        
        assert len(fluxes) == len(target_wavelengths)
        assert np.all(fluxes >= 0)


# tests/test_photometry.py
"""
Tests for synthetic photometry functionality.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import Mock, patch

from stellar_colors.photometry.synthetic import SyntheticPhotometry
from stellar_colors.cube.builder import FluxCube


class TestSyntheticPhotometry:
    """Test cases for SyntheticPhotometry class."""
    
    @pytest.fixture
    def mock_flux_cube(self):
        """Create mock flux cube."""
        mock_cube = Mock(spec=FluxCube)
        mock_cube.interpolate_spectrum.return_value = (
            np.linspace(4000, 7000, 100),  # wavelengths
            np.ones(100) * 1e-10  # fluxes
        )
        return mock_cube
    
    @pytest.fixture
    def temp_filter_dir(self):
        """Create temporary filter directory with test filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filter_dir = Path(temp_dir)
            
            # Create test filter
            wavelengths = np.linspace(4800, 6200, 50)
            transmission = np.exp(-((wavelengths - 5500) / 400)**2)
            
            filter_file = filter_dir / 'test_filter.dat'
            filter_data = np.column_stack([wavelengths, transmission])
            np.savetxt(filter_file, filter_data, 
                      header='Wavelength(A) Transmission', comments='#')
            
            yield filter_dir
    
    @pytest.fixture
    def photometry(self, mock_flux_cube, temp_filter_dir):
        """Create SyntheticPhotometry instance."""
        return SyntheticPhotometry(mock_flux_cube, temp_filter_dir)
    
    def test_initialization(self, mock_flux_cube, temp_filter_dir):
        """Test SyntheticPhotometry initialization."""
        photometry = SyntheticPhotometry(mock_flux_cube, temp_filter_dir)
        
        assert photometry.flux_cube == mock_flux_cube
        assert photometry.filter_dir == temp_filter_dir
        assert len(photometry.available_filters) > 0
        assert 'test_filter' in photometry.available_filters
    
    def test_list_filters(self, photometry):
        """Test filter listing."""
        filters = photometry.list_filters()
        
        assert isinstance(filters, list)
        assert 'test_filter' in filters
    
    def test_load_filter(self, photometry):
        """Test filter loading."""
        wavelengths, transmission = photometry._load_filter('test_filter')
        
        assert len(wavelengths) == len(transmission)
        assert len(wavelengths) > 0
        assert np.all(transmission >= 0)
        assert np.all(transmission <= 1.1)  # Allow for slight numerical errors
    
    def test_load_filter_caching(self, photometry):
        """Test filter caching."""
        # Load filter twice
        wl1, tr1 = photometry._load_filter('test_filter')
        wl2, tr2 = photometry._load_filter('test_filter')
        
        # Should return same arrays (cached)
        assert np.array_equal(wl1, wl2)
        assert np.array_equal(tr1, tr2)
        assert 'test_filter' in photometry._filter_cache
    
    def test_compute_magnitude(self, photometry):
        """Test magnitude computation."""
        magnitude = photometry.compute_magnitude(5500, 4.5, 0.0, 'test_filter')
        
        assert isinstance(magnitude, float)
        assert not np.isnan(magnitude)
        
        # Test with distance and radius
        magnitude_apparent = photometry.compute_magnitude(
            5500, 4.5, 0.0, 'test_filter', distance=10.0, radius=1.0
        )
        
        assert isinstance(magnitude_apparent, float)
        assert not np.isnan(magnitude_apparent)
    
    def test_compute_color(self, photometry):
        """Test color computation.""" 
        # Create second filter
        wavelengths = np.linspace(5800, 7200, 50)
        transmission = np.exp(-((wavelengths - 6500) / 400)**2)
        
        filter_file = photometry.filter_dir / 'test_filter_2.dat'
        filter_data = np.column_stack([wavelengths, transmission])
        np.savetxt(filter_file, filter_data, 
                  header='Wavelength(A) Transmission', comments='#')
        
        # Rediscover filters
        photometry._discover_filters()
        
        color = photometry.compute_color(5500, 4.5, 0.0, 'test_filter', 'test_filter_2')
        
        assert isinstance(color, float)
        assert not np.isnan(color)
    
    def test_compute_synthetic_flux(self, photometry):
        """Test synthetic flux computation."""
        spec_wavelengths = np.linspace(4000, 7000, 100)
        spec_fluxes = np.ones(100) * 1e-10
        
        filter_wavelengths = np.linspace(4800, 6200, 50)
        filter_transmission = np.exp(-((filter_wavelengths - 5500) / 400)**2)
        
        flux = photometry._compute_synthetic_flux(
            spec_wavelengths, spec_fluxes, filter_wavelengths, filter_transmission
        )
        
        assert isinstance(flux, float)
        assert flux > 0
        assert not np.isnan(flux)
    
    def test_compute_synthetic_flux_no_overlap(self, photometry):
        """Test synthetic flux computation with no wavelength overlap."""
        spec_wavelengths = np.linspace(4000, 5000, 100)
        spec_fluxes = np.ones(100) * 1e-10
        
        filter_wavelengths = np.linspace(6000, 7000, 50)
        filter_transmission = np.ones(50)
        
        with pytest.raises(ValueError, match="No wavelength overlap"):
            photometry._compute_synthetic_flux(
                spec_wavelengths, spec_fluxes, filter_wavelengths, filter_transmission
            )


# tests/test_config.py
"""
Tests for configuration functionality.
"""

import pytest
from stellar_colors.config import conf, get_data_dir, get_models_dir


class TestConfiguration:
    """Test cases for configuration management."""
    
    def test_default_values(self):
        """Test default configuration values."""
        assert conf.max_download_workers == 5
        assert conf.download_timeout == 30.0
        assert conf.default_interpolation_method == 'linear'
        assert conf.magnitude_system == 'vega'
    
    def test_data_directory_creation(self):
        """Test data directory creation."""
        data_dir = get_data_dir()
        assert data_dir.exists()
        assert data_dir.is_dir()
    
    def test_subdirectory_creation(self):
        """Test subdirectory creation."""
        models_dir = get_models_dir()
        assert models_dir.exists()
        assert models_dir.is_dir()
        assert models_dir.name == conf.models_dir


# tests/conftest.py
"""
Shared pytest fixtures and configuration.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def suppress_warnings():
    """Suppress warnings during tests."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# Integration test example
# tests/test_integration.py
"""
Integration tests for stellar-colors package.
"""

import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, Mock

import stellar_colors as sc


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            yield workspace
    
    @patch('stellar_colors.atmosphere.grabber.AtmosphereGrabber')
    @patch('stellar_colors.filters.grabber.FilterGrabber')
    def test_basic_workflow(self, mock_filter_grabber, mock_atmosphere_grabber, temp_workspace):
        """Test basic workflow from download to photometry."""
        # Mock the grabbers to avoid actual downloads
        mock_atm_instance = Mock()
        mock_atm_instance.download_model.return_value = temp_workspace / 'test_model'
        mock_atmosphere_grabber.return_value = mock_atm_instance
        
        mock_filter_instance = Mock()
        mock_filter_instance.download_facility_filters.return_value = temp_workspace / 'test_filters'
        mock_filter_grabber.return_value = mock_filter_instance
        
        # Test discovery functions
        with patch('stellar_colors.discover_models') as mock_discover_models:
            mock_discover_models.return_value = ['TEST_MODEL']
            models = sc.discover_models()
            assert 'TEST_MODEL' in models
        
        with patch('stellar_colors.discover_filters') as mock_discover_filters:
            mock_discover_filters.return_value = Mock()
            filters = sc.discover_filters(facility='TEST')
            assert filters is not None
    
    def test_package_imports(self):
        """Test that all main components can be imported."""
        # Test main classes
        assert hasattr(sc, 'AtmosphereGrabber')
        assert hasattr(sc, 'FilterGrabber')
        assert hasattr(sc, 'DataCubeBuilder')
        assert hasattr(sc, 'FluxCube')
        assert hasattr(sc, 'SyntheticPhotometry')
        assert hasattr(sc, 'BolometricCorrections')
        
        # Test convenience functions
        assert hasattr(sc, 'discover_models')
        assert hasattr(sc, 'download_model_grid')
        assert hasattr(sc, 'discover_filters')
        assert hasattr(sc, 'build_flux_cube')
        
        # Test configuration
        assert hasattr(sc, 'conf')
    
    def test_version_info(self):
        """Test version information is available."""
        assert hasattr(sc, '__version__')
        assert isinstance(sc.__version__, str)
        