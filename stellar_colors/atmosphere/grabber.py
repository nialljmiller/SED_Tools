# stellar_colors/atmosphere/grabber.py
"""
Stellar atmosphere model grabber for SVO (Spanish Virtual Observatory).

This module provides tools to discover, download, and manage stellar atmosphere 
models from the SVO theoretical spectra database.
"""

import json
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from astropy.io import ascii
from astropy.table import Table
from astropy.utils.data import download_file
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ['AtmosphereGrabber', 'discover_models', 'download_model_grid']


def _parse_json_response(response):
    """Parse JSON response from SVO."""
    try:
        data = response.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'spectra' in data:
            return data['spectra']
        return []
    except (json.JSONDecodeError, KeyError):
        return []


class AtmosphereGrabber:
    """
    A class for discovering and downloading stellar atmosphere models from SVO.
    
    This class provides a high-level interface to interact with the SVO theoretical
    spectra database, allowing users to discover available models, download specific
    grids, and manage local model collections.
    
    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to store downloaded models. Default is ~/.stellar_colors/models/
    max_workers : int, optional
        Maximum number of concurrent download threads. Default is 5.
    timeout : float, optional
        Request timeout in seconds. Default is 30.
    
    Examples
    --------
    >>> grabber = AtmosphereGrabber()
    >>> models = grabber.discover_models()
    >>> grabber.download_model('KURUCZ2003')
    """
    
    def __init__(
        self, 
        cache_dir: Optional[Union[str, Path]] = None,
        max_workers: int = 5,
        timeout: float = 30.0
    ):
        if cache_dir is None:
            cache_dir = Path.home() / '.stellar_colors' / 'models'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.timeout = timeout
        
        # SVO endpoints
        self.base_url = "http://svo2.cab.inta-csic.es/theory/newov2/"
        self.model_index_url = urljoin(self.base_url, "index.php")
        self.spectra_url = urljoin(self.base_url, "ssap.php")
        self.metadata_url = urljoin(self.base_url, "getmeta.php")
        
        # Setup session with retry strategy
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'stellar-colors/1.0 (Astropy Compatible Package)'
        })
    
    def discover_models(self) -> List[str]:
        """
        Discover all available stellar atmosphere models from SVO.
        
        Returns
        -------
        List[str]
            List of available model names
            
        Examples
        --------
        >>> grabber = AtmosphereGrabber()
        >>> models = grabber.discover_models()
        >>> print(f"Found {len(models)} models: {models[:5]}")
        """
        try:
            response = self.session.get(self.model_index_url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch model index: {e}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        models = []
        
        # Look for model links in the page
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if 'models=' in href:
                model_name = href.split('models=')[1].split('&')[0]
                if model_name and model_name not in models:
                    models.append(model_name)
        
        return sorted(models)
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        Get detailed information about a specific model.
        
        Parameters
        ----------
        model_name : str
            Name of the stellar atmosphere model
            
        Returns
        -------
        Dict
            Dictionary containing model information including parameter ranges,
            number of spectra, and metadata
        """
        spectra_info = self._discover_spectra(model_name)
        
        info = {
            'name': model_name,
            'n_spectra': len(spectra_info),
            'spectra_info': spectra_info,
            'parameter_ranges': self._analyze_parameter_ranges(spectra_info)
        }
        
        return info
    
    def download_model(
        self, 
        model_name: str, 
        output_dir: Optional[Union[str, Path]] = None,
        max_spectra: Optional[int] = None,
        parameter_filter: Optional[Dict] = None,
        show_progress: bool = True
    ) -> Path:
        """
        Download a complete stellar atmosphere model grid.
        
        Parameters
        ----------
        model_name : str
            Name of the model to download
        output_dir : str or Path, optional
            Directory to save the model. Default is cache_dir/model_name
        max_spectra : int, optional
            Maximum number of spectra to download (for testing)
        parameter_filter : dict, optional
            Dictionary to filter spectra by parameter ranges
        show_progress : bool, optional
            Show progress bars
            
        Returns
        -------
        Path
            Path to the downloaded model directory
        """
        if output_dir is None:
            output_dir = self.cache_dir / model_name
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading model {model_name} to {output_dir}")
        
        # Discover available spectra
        spectra_info = self._discover_spectra(model_name)
        
        if not spectra_info:
            raise ValueError(f"No spectra found for model {model_name}")
        
        # Apply parameter filtering if provided
        if parameter_filter:
            spectra_info = self._filter_spectra(spectra_info, parameter_filter)
        
        # Limit number of spectra if requested
        if max_spectra:
            spectra_info = spectra_info[:max_spectra]
        
        logger.info(f"Downloading {len(spectra_info)} spectra for {model_name}")
        
        # Download spectra
        metadata_rows = []
        successful_downloads = 0
        
        # Set up progress bar
        iterator = spectra_info
        if show_progress:
            iterator = tqdm(spectra_info, desc=f"Downloading {model_name}")
        
        # Download in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_spectrum = {}
            
            for spectrum in iterator:
                fid = spectrum.get('fid')
                if fid is None:
                    continue
                
                filename = f"{model_name}_fid{fid}.txt"
                output_path = output_dir / filename
                
                # Skip if already exists
                if output_path.exists():
                    metadata = self._parse_spectrum_metadata(output_path)
                    metadata['file_name'] = filename
                    metadata_rows.append(metadata)
                    successful_downloads += 1
                    continue
                
                future = executor.submit(
                    self._download_single_spectrum, model_name, fid, output_path
                )
                future_to_spectrum[future] = (fid, filename, output_path)
            
            # Collect results
            for future in as_completed(future_to_spectrum):
                fid, filename, output_path = future_to_spectrum[future]
                
                try:
                    success = future.result()
                    if success:
                        metadata = self._parse_spectrum_metadata(output_path)
                        metadata['file_name'] = filename
                        metadata_rows.append(metadata)
                        successful_downloads += 1
                    else:
                        # Clean up failed download
                        if output_path.exists():
                            output_path.unlink()
                except Exception as e:
                    logger.warning(f"Error processing FID {fid}: {e}")
        
        # Create lookup table
        if metadata_rows:
            self._create_lookup_table(output_dir, metadata_rows)
            logger.info(f"Successfully downloaded {successful_downloads} spectra")
        else:
            raise RuntimeError(f"No spectra downloaded for {model_name}")
        
        return output_dir
    
    def _discover_spectra(self, model_name: str) -> List[Dict]:
        """Discover available spectra for a model using multiple methods."""
        logger.info(f"Discovering spectra for model {model_name}")
        
        # Method 1: VOTable catalog query (NEW - handles KURUCZ2003 type models)
        try:
            spectra_info = self._discover_spectra_votable(model_name)
            if spectra_info:
                logger.info(f"Found {len(spectra_info)} spectra via VOTable catalog")
                return spectra_info
        except Exception as e:
            logger.warning(f"VOTable discovery failed: {e}")
        
        # Method 2: Direct metadata query
        try:
            resp = self.session.get(
                self.metadata_url, 
                params={"model": model_name, "format": "json"}, 
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = _parse_json_response(resp)
                if data:
                    logger.info(f"Found {len(data)} spectra via metadata query")
                    return data
        except requests.RequestException as e:
            logger.warning(f"Metadata query failed: {e}")

        # Method 3: SSAP query
        try:
            spectra_info = self._discover_spectra_ssap(model_name)
            if spectra_info:
                logger.info(f"Found {len(spectra_info)} spectra via SSAP")
                return spectra_info
        except Exception as e:
            logger.warning(f"SSAP discovery failed: {e}")

        # Method 4: Brute force fallback
        logger.info(f"Falling back to brute-force discovery for '{model_name}'...")
        return self._discover_spectra_brute_force(model_name)


    def _discover_spectra_votable(self, model_name: str) -> list:
        """Discover spectra using VOTable with case sensitivity handling."""

        from astropy.io.votable import parse_single_table
        from io import BytesIO

        name_variations = [
            model_name,
            model_name.lower(),
            model_name.upper(),
            model_name.capitalize(),
            model_name.title(),
        ]

        for variant in name_variations:
            try:
                print(f"Trying model name variant: '{variant}'")
                params = {"model": variant, "REQUEST": "queryData"}
                resp = self.session.get(self.spectra_url, params=params, timeout=self.timeout)

                if resp.status_code != 200 or len(resp.content) < 1000:
                    continue

                print(f"*** SUCCESS with variant '{variant}' - {len(resp.content)} bytes ***")

                votable_data = BytesIO(resp.content)
                table = parse_single_table(votable_data).to_table()
                print(f"Parsed VOTable: {len(table)} entries")

                url_column = None
                for col in table.colnames:
                    simplified = col.strip().lower().replace('_', '').replace('.', '')
                    if 'access' in simplified and 'reference' in simplified:
                        url_column = col
                        break

                if url_column is None:
                    print(f"No usable URL column found among: {table.colnames}")
                    continue

                spectra_info = []
                for row in table:
                    try:

                        access_url = str(row[url_column])
                        fid_match = re.search(r'fid=(\d+)', access_url)
                        if fid_match:
                            entry = {
                                'fid': int(fid_match.group(1)),
                                'access_url': access_url
                            }
                            for param_col in ['teff', 'logg', 'meta']:
                                if param_col in table.colnames:
                                    val = row[param_col]
                                    if hasattr(val, 'item'):
                                        val = val.item()
                                    entry[param_col] = val
                            spectra_info.append(entry)

                    except Exception:
                        continue

                print(f"Extracted {len(spectra_info)} spectra")
                if spectra_info:
                    spectra_info.sort(key=lambda s: s['fid'])
                    self._fid_offset = spectra_info[0]['fid']  # save lowest FID universally
                return spectra_info

            except Exception as e:
                print(f"Variant '{variant}' failed: {e}")
                continue

        return []



    def _discover_spectra_ssap(self, model_name: str) -> List[Dict]:
        """Discover spectra using SSAP protocol."""
        spectra_info = []

        try:
            resp = self.session.get(
                self.spectra_url, 
                params={"model": model_name, "REQUEST": "queryData", "FORMAT": "metadata"}, 
                timeout=self.timeout
            )
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, 'html.parser')
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if "fid=" in href:
                    fid_match = re.search(r'fid=(\d+)', href)
                    if fid_match:
                        fid = int(fid_match.group(1))
                        spectra_info.append({"fid": fid})

        except requests.RequestException as e:
            logger.warning(f"SSAP request failed: {e}")

        return spectra_info

    def _discover_spectra_brute_force(
        self, 
        model_name: str, 
        max_fid: int = 50000,  # Increased to handle KURUCZ2003-type models
        batch_size: int = 100,
        max_failures: int = 1000  # Increased failure tolerance
    ) -> List[Dict]:
        """Brute-force spectrum discovery via parallel probing of FIDs."""
        logger.info(f"Starting brute-force discovery for {model_name}")
        
        spectra_info = []
        consecutive_failures = 0
        total_failures = 0
        
        # Define search ranges based on model type
        # Some models (like KURUCZ2003) have real spectra at high FIDs
        if any(pattern in model_name.upper() for pattern in ['KURUCZ2003', 'ATLAS', 'PHOENIX']):
            # For these models, try high FID ranges first
            search_ranges = [
                range(10000, 15000),  # High range first
                range(1, 1000),       # Then low range
                range(1000, 5000),    # Medium range
                range(15000, max_fid, 1000)  # Very high range (sparse)
            ]
            logger.info(f"Using high-FID search pattern for {model_name}")
        else:
            # Standard search pattern for other models
            search_ranges = [
                range(1, 2000),       # Low range
                range(2000, 10000),   # Medium range  
                range(10000, max_fid, 1000)  # High range (sparse)
            ]
        
        # Test FID ranges
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for search_range in search_ranges:
                logger.info(f"Testing FID range {search_range.start}-{search_range.stop-1} (step {search_range.step})")
                
                # Break range into batches
                range_list = list(search_range)
                for start_idx in range(0, len(range_list), batch_size):
                    end_idx = min(start_idx + batch_size, len(range_list))
                    fids = range_list[start_idx:end_idx]
                    
                    # Submit batch of tests
                    future_to_fid = {
                        executor.submit(self._test_spectrum_exists, model_name, fid): fid
                        for fid in fids
                    }
                    
                    batch_found = 0
                    batch_failures = 0
                    
                    # Collect results
                    for future in as_completed(future_to_fid):
                        fid = future_to_fid[future]
                        try:
                            exists = future.result()
                            if exists:
                                spectra_info.append({"fid": fid})
                                batch_found += 1
                                consecutive_failures = 0
                            else:
                                batch_failures += 1
                                consecutive_failures += 1
                                total_failures += 1
                                
                        except Exception as e:
                            logger.debug(f"Error testing FID {fid}: {e}")
                            batch_failures += 1
                            consecutive_failures += 1
                            total_failures += 1
                    
                    if batch_found > 0:
                        logger.info(f"Batch {fids[0]}-{fids[-1]}: found {batch_found}, failed {batch_failures}")
                    
                    # If we found spectra in this range, continue with this range
                    if batch_found > 0:
                        consecutive_failures = 0
                    
                    # Stop if too many consecutive failures within a range
                    if consecutive_failures > 500:
                        logger.info(f"Stopping range after {consecutive_failures} consecutive failures")
                        break
                    
                    # Progress update
                    if len(spectra_info) > 0 and len(spectra_info) % 100 == 0:
                        logger.info(f"Found {len(spectra_info)} spectra so far...")
                
                # If we found enough spectra, no need to test other ranges
                if len(spectra_info) > 50:
                    logger.info(f"Found sufficient spectra ({len(spectra_info)}), stopping search")
                    break
                
                # Reset consecutive failures for next range
                consecutive_failures = 0

        logger.info(f"Brute-force discovery found {len(spectra_info)} spectra")
        return spectra_info



    def _test_spectrum_exists(self, model_name: str, fid: int) -> bool:
        """Try downloading the spectrum and check that it's real data, not a template."""
        import re

        params = {
            "model": model_name,
            "fid": fid,
            "format": "ascii"
        }

        try:
            r = self.session.get(self.spectra_url, params=params, timeout=10)
            if r.status_code != 200 or len(r.content) < 500:
                return False

            content = r.text

            # Template-like patterns — bad
            if re.search(r'teff\s*=\s*K', content) or \
               re.search(r'logg\s*=\s*log', content) or \
               re.search(r'meta\s*=\s*$', content):
                return False

            # Looks real
            return True

        except Exception:
            return False

    def _download_single_spectrum(self, model_name: str, fid: int, output_path: Path) -> bool:
        """Download a single spectrum file."""
        try:
            params = {"model": model_name, "fid": fid, "format": "ascii"}
            response = self.session.get(
                self.spectra_url, 
                params=params, 
                timeout=self.timeout,
                stream=True
            )

            if response.status_code == 200 and len(response.content) > 1024:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                return False

        except Exception as e:
            logger.warning(f"Error downloading FID {fid}: {e}")
            return False

    def _parse_spectrum_metadata(self, filepath: Path) -> Dict:
        """Parse metadata from a downloaded spectrum file."""
        metadata = {}
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') and '=' in line:
                        try:
                            key, value = line.split('=', 1)
                            key = key.strip('#').strip()
                            value = value.split('(')[0].strip()

                            # Clean numerical values
                            if key != 'file_name':
                                match = re.search(r'[-+]?\d*\.?\d+', value)
                                value = match.group() if match else '999.9'

                            metadata[key] = value
                        except ValueError:
                            continue
                    elif not line.startswith('#'):
                        break  # Stop at first data line
        except Exception as e:
            logger.warning(f"Error parsing metadata in {filepath}: {e}")

        return metadata

    def _create_lookup_table(self, output_dir: Path, metadata_rows: List[Dict]) -> None:
        """Create a lookup table CSV from metadata."""
        if not metadata_rows:
            return
        
        # Create DataFrame
        df = pd.DataFrame(metadata_rows)
        
        # Ensure required columns exist
        required_columns = ['file_name', 'teff', 'logg', 'metallicity']
        for col in required_columns:
            if col not in df.columns:
                if col == 'metallicity' and 'meta' in df.columns:
                    df['metallicity'] = df['meta']
                elif col == 'file_name':
                    df['file_name'] = df.index.map(lambda i: f"spectrum_{i}.txt")
                else:
                    df[col] = 999.9  # Default value for missing parameters
        
        # Clean up data types
        for col in ['teff', 'logg', 'metallicity']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(999.9)
        
        # Sort by parameters
        df = df.sort_values(['teff', 'logg', 'metallicity'])
        
        # Save lookup table
        lookup_file = output_dir / 'lookup_table.csv'
        with open(lookup_file, 'w') as f:
            f.write('#file_name,teff,logg,metallicity\n')
            df[required_columns].to_csv(f, index=False, header=False)
        
        logger.info(f"Created lookup table: {lookup_file}")

    def _filter_spectra(self, spectra_info: List[Dict], parameter_filter: Dict) -> List[Dict]:
        """Filter spectra based on parameter ranges."""
        filtered = []
        
        for spectrum in spectra_info:
            keep = True
            for param, (min_val, max_val) in parameter_filter.items():
                value = spectrum.get(param)
                if value is not None:
                    try:
                        value = float(value)
                        if not (min_val <= value <= max_val):
                            keep = False
                            break
                    except (ValueError, TypeError):
                        continue
            if keep:
                filtered.append(spectrum)
        
        return filtered

    def _analyze_parameter_ranges(self, spectra_info: List[Dict]) -> Dict:
        """Analyze parameter ranges from spectra info."""
        ranges = {}
        
        if not spectra_info:
            return ranges
        
        # Collect all parameter keys
        all_keys = set()
        for spectrum in spectra_info:
            all_keys.update(spectrum.keys())
        
        # Analyze ranges for numerical parameters
        for key in all_keys:
            if key in ['fid', 'file_name']:
                continue
            
            values = []
            for spectrum in spectra_info:
                if key in spectrum:
                    try:
                        val = float(spectrum[key])
                        if not np.isnan(val):
                            values.append(val)
                    except (ValueError, TypeError):
                        continue
            
            if values:
                ranges[key] = {
                    'min': min(values),
                    'max': max(values),
                    'count': len(values),
                    'unique': len(set(values))
                }
        
        return ranges


def discover_models() -> List[str]:
    """
    Convenience function to discover available stellar atmosphere models.
    
    Returns
    -------
    List[str]
        List of available model names
    """
    grabber = AtmosphereGrabber()
    return grabber.discover_models()


def download_model_grid(
    model_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Path:
    """
    Convenience function to download a stellar atmosphere model grid.
    
    Parameters
    ----------
    model_name : str
        Name of the model to download
    output_dir : str or Path, optional
        Directory to save the model
    **kwargs
        Additional arguments passed to AtmosphereGrabber.download_model()
        
    Returns
    -------
    Path
        Path to the downloaded model directory
    """
    grabber = AtmosphereGrabber()
    return grabber.download_model(model_name, output_dir, **kwargs)


    def download_model(
    self, 
    model_name: str, 
    output_dir: Optional[Union[str, Path]] = None,
    max_spectra: Optional[int] = None,
    parameter_filter: Optional[Dict] = None,
    show_progress: bool = True
) -> Path:
    """
    Download a complete stellar atmosphere model grid.
    
    Parameters
    ----------
    model_name : str
        Name of the model to download
    output_dir : str or Path, optional
        Directory to save the model. Default is cache_dir/model_name
    max_spectra : int, optional
        Maximum number of spectra to download (for testing)
    parameter_filter : dict, optional
        Dictionary to filter spectra by parameter ranges
    show_progress : bool, optional
        Show progress bars
        
    Returns
    -------
    Path
        Path to the downloaded model directory
    """
    if output_dir is None:
        output_dir = self.cache_dir / model_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading model {model_name} to {output_dir}")
    
    # Discover available spectra 
    spectra_info = self._discover_spectra(model_name)
    
    if not spectra_info:
        raise ValueError(f"No spectra found for model {model_name}")
    
    # Apply parameter filtering if provided
    if parameter_filter:
        spectra_info = self._filter_spectra(spectra_info, parameter_filter)
    
    # Limit number of spectra if requested
    if max_spectra:
        spectra_info = spectra_info[:max_spectra]
    
    logger.info(f"Downloading {len(spectra_info)} spectra for {model_name}")
    
    # Download spectra using ASCII format (the method that actually works)
    metadata_rows = []
    successful_downloads = 0
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        # Submit download tasks
        future_to_fid = {}
        
        for spectrum in spectra_info:
            fid = spectrum.get('fid')
            if not fid:
                continue
                
            filename = f"{model_name}_fid{fid}.txt"
            output_path = output_dir / filename
            
            # Skip if already exists
            if output_path.exists():
                metadata = self._parse_spectrum_metadata(output_path)
                metadata['filename'] = filename
                metadata['file_name'] = filename
                metadata_rows.append(metadata)
                successful_downloads += 1
                continue
            
            future = executor.submit(self._download_single_spectrum, model_name, fid, output_path)
            future_to_fid[future] = (fid, filename, output_path)
        
        # Process completed downloads with progress bar
        if show_progress:
            futures_iter = tqdm(as_completed(future_to_fid), 
                              total=len(future_to_fid),
                              desc=f"Downloading {model_name}")
        else:
            futures_iter = as_completed(future_to_fid)
            
        for future in futures_iter:
            fid, filename, output_path = future_to_fid[future]
            
            try:
                success = future.result()
                if success:
                    # Parse metadata from downloaded file
                    metadata = self._parse_spectrum_metadata(output_path)
                    metadata['filename'] = filename
                    metadata['file_name'] = filename
                    metadata_rows.append(metadata)
                    successful_downloads += 1
                else:
                    # Clean up failed download
                    if output_path.exists():
                        output_path.unlink()
                        
            except Exception as e:
                logger.warning(f"Error processing FID {fid}: {e}")
                if output_path.exists():
                    output_path.unlink()
    
    # Create lookup table with the correct column names expected by DataCubeBuilder
    if metadata_rows:
        self._create_lookup_table(output_dir, metadata_rows)
        logger.info(f"Successfully downloaded {successful_downloads} spectra")
    else:
        raise RuntimeError(f"No spectra downloaded for {model_name}")
    
    return output_dir


def _parse_spectrum_metadata(self, filepath: Path) -> Dict:
    """Parse metadata from a downloaded spectrum file."""
    metadata = {}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') and '=' in line:
                    try:
                        key, value = line.split('=', 1)
                        key = key.strip('#').strip().lower()
                        value = value.split('(')[0].strip()

                        # Map SVO parameter names to standard names
                        key_mapping = {
                            'teff': 'teff',
                            'teffs': 'teff', 
                            'temp': 'teff',
                            'temperature': 'teff',
                            'logg': 'logg',
                            'log_g': 'logg',
                            'grav': 'logg',
                            'meta': 'metallicity',
                            'metallicity': 'metallicity',
                            'm_h': 'metallicity',
                            'feh': 'metallicity',
                            '[m/h]': 'metallicity',
                            '[fe/h]': 'metallicity'
                        }
                        
                        standard_key = key_mapping.get(key, key)
                        
                        # Clean numerical values
                        if standard_key in ['teff', 'logg', 'metallicity']:
                            match = re.search(r'[-+]?\d*\.?\d+', value)
                            value = float(match.group()) if match else 0.0

                        metadata[standard_key] = value
                        
                    except (ValueError, AttributeError):
                        continue
                elif not line.startswith('#'):
                    break  # Stop at first data line
    except Exception as e:
        logger.warning(f"Error parsing metadata in {filepath}: {e}")

    return metadata


def _create_lookup_table(self, output_dir: Path, metadata_rows: List[Dict]) -> None:
    """Create a lookup table CSV from metadata with correct column names."""
    lookup_file = output_dir / 'lookup_table.csv'
    
    # Ensure all required columns exist with defaults
    required_columns = ['filename', 'teff', 'logg', 'metallicity', 'file_name']
    
    for row in metadata_rows:
        for col in required_columns:
            if col not in row:
                if col == 'file_name' and 'filename' in row:
                    row[col] = row['filename']
                elif col == 'filename' and 'file_name' in row:
                    row[col] = row['file_name']
                else:
                    row[col] = 0.0 if col in ['teff', 'logg', 'metallicity'] else 'unknown'
    
    # Create DataFrame and save
    df = pd.DataFrame(metadata_rows)
    
    # Reorder columns to match expected format
    column_order = ['filename', 'teff', 'logg', 'metallicity', 'file_name']
    existing_cols = [col for col in column_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in column_order]
    final_columns = existing_cols + other_cols
    
    df = df[final_columns]
    
    # Save with header comment
    with open(lookup_file, 'w') as f:
        f.write('#' + ','.join(final_columns) + '\n')
        df.to_csv(f, index=False, header=False)
    
    logger.info(f"Created lookup table: {lookup_file}")