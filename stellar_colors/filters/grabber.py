"""
Photometric filter transmission curve grabber for SVO Filter Profile Service.

This module provides tools to discover, download, and manage photometric filter
transmission curves from the SVO Filter Profile Service.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table, unique, vstack
from astropy.utils.data import download_file
from astroquery.svo_fps import SvoFps
from tqdm import tqdm

__all__ = ['FilterGrabber', 'discover_filters', 'download_filter_collection']


class FilterGrabber:
    """
    A class for discovering and downloading photometric filter transmission curves.
    
    This class provides a high-level interface to interact with the SVO Filter Profile
    Service, allowing users to discover available filters, download transmission curves,
    and organize filter collections by facility and instrument.
    
    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to store downloaded filters. Default is ~/.stellar_colors/filters/
    timeout : float, optional
        Request timeout in seconds. Default is 300.
        
    Examples
    --------
    >>> grabber = FilterGrabber()
    >>> facilities = grabber.discover_facilities()
    >>> grabber.download_facility_filters('HST')
    """
    
    def __init__(
        self, 
        cache_dir: Optional[Union[str, Path]] = None,
        timeout: float = 300.0
    ):
        if cache_dir is None:
            cache_dir = Path.home() / '.stellar_colors' / 'filters'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure SVO FPS with longer timeout
        SvoFps.TIMEOUT = timeout
        
        # Define wavelength ranges for systematic discovery
        self.wavelength_ranges = [
            ("X-ray", 0.1 * u.AA, 100 * u.AA),
            ("UV", 100 * u.AA, 4000 * u.AA),
            ("Optical", 4000 * u.AA, 7000 * u.AA),
            ("NIR", 7000 * u.AA, 25000 * u.AA),
            ("MIR", 25000 * u.AA, 250000 * u.AA),
            ("FIR", 250000 * u.AA, 1e7 * u.AA),
            ("Radio", 1e7 * u.AA, 1e8 * u.AA),
        ]
    
    def discover_facilities(self) -> List[str]:
        """
        Discover all available facilities with photometric filters.
        
        Returns
        -------
        List[str]
            Sorted list of facility names
        """
        all_filters = self._get_all_filters()
        facilities = list(set(all_filters['Facility'].tolist()))
        return sorted([f for f in facilities if f and f != 'Unknown'])
    
    def discover_instruments(self, facility: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Discover instruments for a specific facility or all facilities.
        
        Parameters
        ----------
        facility : str, optional
            Facility name. If None, returns instruments for all facilities.
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping facility names to lists of instrument names
        """
        all_filters = self._get_all_filters()
        
        if facility:
            facility_filters = all_filters[all_filters['Facility'] == facility]
            instruments = list(set(facility_filters['Instrument'].tolist()))
            return {facility: sorted([i for i in instruments if i])}
        
        instruments_by_facility = {}
        for fac in self.discover_facilities():
            facility_filters = all_filters[all_filters['Facility'] == fac]
            instruments = list(set(facility_filters['Instrument'].tolist()))
            instruments_by_facility[fac] = sorted([i for i in instruments if i])
        
        return instruments_by_facility
    
    def search_filters(
        self,
        facility: Optional[str] = None,
        instrument: Optional[str] = None,
        wavelength_range: Optional[Tuple[u.Quantity, u.Quantity]] = None,
        band_name: Optional[str] = None
    ) -> Table:
        """
        Search for filters matching specific criteria.
        
        Parameters
        ----------
        facility : str, optional
            Facility name (e.g., 'HST', 'Gaia', 'SDSS')
        instrument : str, optional
            Instrument name (e.g., 'WFC3', 'ACS')
        wavelength_range : tuple of Quantity, optional
            Wavelength range as (min_wavelength, max_wavelength)
        band_name : str, optional
            Band/filter name (e.g., 'V', 'g', 'F555W')
            
        Returns
        -------
        astropy.table.Table
            Table of matching filters with metadata
            
        Examples
        --------
        >>> grabber = FilterGrabber()
        >>> hst_filters = grabber.search_filters(facility='HST')
        >>> optical_filters = grabber.search_filters(
        ...     wavelength_range=(4000*u.AA, 7000*u.AA)
        ... )
        """
        all_filters = self._get_all_filters()
        
        # Apply filters
        mask = np.ones(len(all_filters), dtype=bool)
        
        if facility:
            mask &= all_filters['Facility'] == facility
        
        if instrument:
            mask &= all_filters['Instrument'] == instrument
        
        if band_name:
            # Case-insensitive search in band names
            band_mask = np.array([
                band_name.lower() in str(band).lower() 
                for band in all_filters['Band']
            ])
            mask &= band_mask
        
        if wavelength_range:
            min_wave, max_wave = wavelength_range
            # Convert to Angstroms for comparison
            min_aa = min_wave.to(u.AA).value
            max_aa = max_wave.to(u.AA).value
            
            wave_mask = (
                (all_filters['WavelengthEff'] >= min_aa) & 
                (all_filters['WavelengthEff'] <= max_aa)
            )
            mask &= wave_mask
        
        return all_filters[mask]
    
    def download_filter(
        self, 
        filter_id: str, 
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Download a specific filter transmission curve.
        
        Parameters
        ----------
        filter_id : str
            SVO filter identifier (e.g., 'Generic/Johnson.V')
        output_dir : str or Path, optional
            Output directory. If None, uses organized structure in cache_dir
            
        Returns
        -------
        Path
            Path to the downloaded filter file
        """
        # Get filter metadata
        try:
            filter_info = SvoFps.get_filter_index(filterID=filter_id)
            if len(filter_info) == 0:
                raise ValueError(f"Filter {filter_id} not found")
            
            filter_row = filter_info[0]
        except Exception as e:
            raise ValueError(f"Failed to get filter info for {filter_id}: {e}")
        
        # Determine output path
        if output_dir is None:
            facility = self._clean_name(filter_row.get('Facility', 'Unknown'))
            instrument = self._clean_name(filter_row.get('Instrument', 'Unknown'))
            output_dir = self.cache_dir / facility / instrument
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        band = self._clean_name(filter_row.get('Band', filter_id.split('.')[-1]))
        filename = f"{band}.dat"
        filepath = output_dir / filename
        
        # Download transmission data
        try:
            transmission_data = SvoFps.get_transmission_data(filter_id)
            if transmission_data is None or len(transmission_data) == 0:
                raise ValueError(f"No transmission data available for {filter_id}")
            
            # Save with metadata header
            self._save_filter_with_metadata(
                filepath, transmission_data, filter_row, filter_id
            )
            
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"Failed to download {filter_id}: {e}")
    
    def download_facility_filters(
        self, 
        facility: str,
        instruments: Optional[List[str]] = None,
        wavelength_range: Optional[Tuple[u.Quantity, u.Quantity]] = None,
        max_filters: Optional[int] = None,
        show_progress: bool = True
    ) -> Path:
        """
        Download all filters for a specific facility.
        
        Parameters
        ----------
        facility : str
            Facility name
        instruments : List[str], optional
            Specific instruments to download. If None, downloads all.
        wavelength_range : tuple of Quantity, optional
            Wavelength range filter
        max_filters : int, optional
            Maximum number of filters to download (for testing)
        show_progress : bool, optional
            Show progress bar
            
        Returns
        -------
        Path
            Path to the facility filter directory
        """
        # Search for filters
        filters_table = self.search_filters(
            facility=facility,
            wavelength_range=wavelength_range
        )
        
        if instruments:
            instrument_mask = np.isin(filters_table['Instrument'], instruments)
            filters_table = filters_table[instrument_mask]
        
        if len(filters_table) == 0:
            raise ValueError(f"No filters found for facility {facility}")
        
        if max_filters:
            filters_table = filters_table[:max_filters]
        
        print(f"Downloading {len(filters_table)} filters for {facility}")
        
        # Download filters in parallel
        successful_downloads = []
        facility_dir = self.cache_dir / self._clean_name(facility)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_filter = {
                executor.submit(self._download_single_filter, row): row
                for row in filters_table
            }
            
            if show_progress:
                progress_bar = tqdm(
                    total=len(future_to_filter),
                    desc=f"Downloading {facility} filters"
                )
            
            for future in as_completed(future_to_filter):
                filter_row = future_to_filter[future]
                if show_progress:
                    progress_bar.update(1)
                
                try:
                    filepath = future.result()
                    if filepath:
                        successful_downloads.append({
                            'filter_id': filter_row['filterID'],
                            'facility': filter_row['Facility'],
                            'instrument': filter_row['Instrument'],
                            'band': filter_row['Band'],
                            'filepath': str(filepath.relative_to(facility_dir))
                        })
                except Exception as e:
                    warnings.warn(f"Failed to download {filter_row['filterID']}: {e}")
            
            if show_progress:
                progress_bar.close()
        
        # Create facility catalog
        if successful_downloads:
            catalog_df = pd.DataFrame(successful_downloads)
            catalog_path = facility_dir / 'filter_catalog.csv'
            catalog_df.to_csv(catalog_path, index=False)
            print(f"Created filter catalog: {catalog_path}")
        
        print(f"Successfully downloaded {len(successful_downloads)} filters to {facility_dir}")
        return facility_dir
    
    def download_filter_collection(
        self,
        collection_name: str,
        filter_specs: List[Dict],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Download a custom collection of filters.
        
        Parameters
        ----------
        collection_name : str
            Name for the filter collection
        filter_specs : List[Dict]
            List of filter specifications, each containing search criteria
        output_dir : str or Path, optional
            Output directory for the collection
            
        Returns
        -------
        Path
            Path to the collection directory
            
        Examples
        --------
        >>> collection = [
        ...     {'facility': 'HST', 'instrument': 'WFC3'},
        ...     {'facility': 'Gaia', 'band_name': 'G'},
        ...     {'wavelength_range': (5000*u.AA, 6000*u.AA)}
        ... ]
        >>> grabber.download_filter_collection('MyCollection', collection)
        """
        if output_dir is None:
            output_dir = self.cache_dir / 'collections' / collection_name
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_filters = []
        for spec in filter_specs:
            filters = self.search_filters(**spec)
            all_filters.append(filters)
        
        # Combine and remove duplicates
        if all_filters:
            combined_filters = vstack(all_filters)
            combined_filters = unique(combined_filters, keys='filterID')
        else:
            raise ValueError("No filters found matching the specifications")
        
        # Download each filter
        successful_downloads = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_filter = {
                executor.submit(
                    self.download_filter, 
                    row['filterID'], 
                    output_dir
                ): row for row in combined_filters
            }
            
            progress_bar = tqdm(
                total=len(future_to_filter),
                desc=f"Building {collection_name} collection"
            )
            
            for future in as_completed(future_to_filter):
                filter_row = future_to_filter[future]
                progress_bar.update(1)
                
                try:
                    filepath = future.result()
                    successful_downloads.append({
                        'filter_id': filter_row['filterID'],
                        'facility': filter_row['Facility'],
                        'instrument': filter_row['Instrument'],
                        'band': filter_row['Band'],
                        'filename': filepath.name,
                        'wavelength_eff': filter_row['WavelengthEff']
                    })
                except Exception as e:
                    warnings.warn(f"Failed to download {filter_row['filterID']}: {e}")
            
            progress_bar.close()
        
        # Create collection catalog
        if successful_downloads:
            catalog_df = pd.DataFrame(successful_downloads)
            catalog_df = catalog_df.sort_values('wavelength_eff')
            catalog_path = output_dir / 'collection_catalog.csv'
            catalog_df.to_csv(catalog_path, index=False)
            
            # Create collection info file
            info = {
                'name': collection_name,
                'n_filters': len(successful_downloads),
                'specifications': filter_specs,
                'wavelength_range': {
                    'min': float(catalog_df['wavelength_eff'].min()),
                    'max': float(catalog_df['wavelength_eff'].max())
                }
            }
            
            import json
            info_path = output_dir / 'collection_info.json'
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
        
        print(f"Created filter collection '{collection_name}' with {len(successful_downloads)} filters")
        return output_dir
    
    def _get_all_filters(self) -> Table:
        """Get all available filters from SVO, cached for efficiency."""
        cache_file = self.cache_dir / 'all_filters_cache.fits'
        
        # Check if cache exists and is recent (1 day)
        if cache_file.exists():
            import time
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < 24:
                try:
                    return Table.read(cache_file)
                except Exception:
                    pass  # Cache corrupted, rebuild
        
        print("Discovering all available filters (this may take a few minutes)...")
        
        all_filters = []
        for region_name, wavelength_min, wavelength_max in self.wavelength_ranges:
            try:
                filters_table = SvoFps.get_filter_index(
                    wavelength_eff_min=wavelength_min,
                    wavelength_eff_max=wavelength_max
                )
                print(f"Found {len(filters_table)} filters in {region_name} region")
                all_filters.append(filters_table)
            except Exception as e:
                warnings.warn(f"Failed to get filters for {region_name}: {e}")
        
        if all_filters:
            combined_filters = vstack(all_filters)
            combined_filters = unique(combined_filters, keys='filterID')
            
            # Cache the results
            try:
                combined_filters.write(cache_file, overwrite=True)
            except Exception as e:
                warnings.warn(f"Failed to cache filter list: {e}")
            
            return combined_filters
        else:
            raise RuntimeError("Failed to discover any filters")
    
    def _download_single_filter(self, filter_row) -> Optional[Path]:
        """Download a single filter, handling errors gracefully."""
        try:
            return self.download_filter(filter_row['filterID'])
        except Exception:
            return None
    
    def _save_filter_with_metadata(
        self, 
        filepath: Path, 
        transmission_data: Table, 
        filter_info: dict, 
        filter_id: str
    ):
        """Save filter with comprehensive metadata header."""
        with open(filepath, 'w') as f:
            # Write metadata header
            f.write(f"# Filter: {filter_id}\n")
            f.write(f"# Facility: {filter_info.get('Facility', 'Unknown')}\n")
            f.write(f"# Instrument: {filter_info.get('Instrument', 'Unknown')}\n")
            f.write(f"# Band: {filter_info.get('Band', 'Unknown')}\n")
            f.write(f"# WavelengthEff: {filter_info.get('WavelengthEff', 'Unknown')} AA\n")
            f.write(f"# WavelengthMin: {filter_info.get('WavelengthMin', 'Unknown')} AA\n")
            f.write(f"# WavelengthMax: {filter_info.get('WavelengthMax', 'Unknown')} AA\n")
            f.write(f"# FWHM: {filter_info.get('FWHM', 'Unknown')} AA\n")
            f.write("# Columns: Wavelength(AA) Transmission\n")
            
            # Write transmission data
            for row in transmission_data:
                f.write(f"{row[0]:.3f} {row[1]:.6f}\n")
    
    def _clean_name(self, name: str) -> str:
        """Clean a name for use as a directory/filename."""
        if not name or str(name).lower() in ['unknown', 'nan', 'none']:
            return 'Unknown'
        
        # Replace problematic characters
        clean = str(name)
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
            clean = clean.replace(char, '_')
        
        return clean.strip('_')


def discover_filters(
    facility: Optional[str] = None,
    wavelength_range: Optional[Tuple[u.Quantity, u.Quantity]] = None
) -> Table:
    """
    Convenience function to discover available photometric filters.
    
    Parameters
    ----------
    facility : str, optional
        Facility name to search for
    wavelength_range : tuple of Quantity, optional
        Wavelength range to search within
        
    Returns
    -------
    astropy.table.Table
        Table of available filters
    """
    grabber = FilterGrabber()
    return grabber.search_filters(
        facility=facility,
        wavelength_range=wavelength_range
    )


def download_filter_collection(
    collection_name: str,
    facilities: Optional[List[str]] = None,
    wavelength_range: Optional[Tuple[u.Quantity, u.Quantity]] = None,
    **kwargs
) -> Path:
    """
    Convenience function to download a collection of filters.
    
    Parameters
    ----------
    collection_name : str
        Name for the filter collection
    facilities : List[str], optional
        List of facilities to include
    wavelength_range : tuple of Quantity, optional
        Wavelength range to limit the collection
    **kwargs
        Additional arguments passed to FilterGrabber
        
    Returns
    -------
    Path
        Path to the downloaded collection directory
    """
    grabber = FilterGrabber(**kwargs)
    
    # Build filter specifications
    filter_specs = []
    
    if facilities:
        for facility in facilities:
            spec = {'facility': facility}
            if wavelength_range:
                spec['wavelength_range'] = wavelength_range
            filter_specs.append(spec)
    elif wavelength_range:
        filter_specs.append({'wavelength_range': wavelength_range})
    else:
        # Download some common filter systems
        common_facilities = ['Generic', 'HST', 'Gaia', 'SDSS', '2MASS']
        filter_specs = [{'facility': fac} for fac in common_facilities]
    
    return grabber.download_filter_collection(collection_name, filter_specs)