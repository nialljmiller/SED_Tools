"""
Stellar atmosphere model grabber for SVO (Spanish Virtual Observatory).

This module provides tools to discover, download, and manage stellar atmosphere 
models from the SVO theoretical spectra database.
"""

import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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




__all__ = ['AtmosphereGrabber', 'discover_models', 'download_model_grid']


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
            Directory to save the model. If None, uses cache_dir/model_name
        max_spectra : int, optional
            Maximum number of spectra to download (for testing)
        parameter_filter : dict, optional
            Filter spectra by parameter ranges, e.g., 
            {'teff': (4000, 8000), 'logg': (3.5, 5.0)}
        show_progress : bool, optional
            Show download progress bar
            
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
        
        # Discover available spectra
        spectra_info = self._discover_spectra(model_name)
        
        if not spectra_info:
            raise ValueError(f"No spectra found for model {model_name}")
        
        # Apply filters
        if parameter_filter:
            spectra_info = self._filter_spectra(spectra_info, parameter_filter)
        
        if max_spectra:
            spectra_info = spectra_info[:max_spectra]
        
        # Download spectra
        successful_downloads = self._download_spectra_parallel(
            model_name, spectra_info, output_dir, show_progress
        )
        
        # Create metadata table
        self._create_lookup_table(output_dir, successful_downloads)
        
        print(f"Successfully downloaded {len(successful_downloads)} spectra to {output_dir}")
        return output_dir


    def _parse_json_response(response) -> Optional[List[Dict]]:
        try:
            data = response.json()
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return None


    def _discover_spectra(self, model_name: str) -> List[Dict]:
        """Discover available spectra for a model using multiple methods."""
        spectra_info = []

        # Method 1: Direct metadata query
        try:
            resp = self.session.get(
                self.metadata_url, 
                params={"model": model_name, "format": "json"}, 
                timeout=self.timeout
            )
            if resp.status_code == 200:
                data = _parse_json_response(resp)
                if data:
                    return data
        except requests.RequestException as e:
            print(f"[WARN] Metadata query failed: {e}")

        # Method 2: SSAP query
        try:
            spectra_info = self._discover_spectra_ssap(model_name)
            if spectra_info:
                return spectra_info
        except Exception as e:
            print(f"[WARN] SSAP discovery failed: {e}")

        # Method 3: Brute force fallback
        print(f"[INFO] Falling back to brute-force discovery for '{model_name}'...")
        return self._discover_spectra_brute_force(model_name)


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

            soup = BeautifulSoup(resp.text, features="html.parser")  # Use 'xml' if appropriate
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if "fid=" in href:
                    fid = href.split("fid=")[1].split("&")[0]
                    if fid.isdigit():
                        spectra_info.append({"fid": int(fid)})

        except requests.RequestException as e:
            print(f"[WARN] SSAP request failed: {e}")

        return spectra_info


    def _discover_spectra_brute_force(
        self, 
        model_name: str, 
        max_fid: int = 20000, 
        batch_size: int = 50
    ) -> List[Dict]:
        """Brute-force spectrum discovery via parallel probing of FIDs."""
        spectra_info = []
        consecutive_failures = 0
        max_failures = 100

        with ThreadPoolExecutor(max_workers=min(10, batch_size)) as executor:
            for start in tqdm(range(1, max_fid, batch_size), desc=f"Brute scan {model_name}"):
                fids = list(range(start, min(start + batch_size, max_fid)))

                futures = {
                    executor.submit(self._test_spectrum_exists, model_name, fid): fid
                    for fid in fids
                }

                found = False
                for future in as_completed(futures):
                    fid = futures[future]
                    try:
                        if future.result():
                            spectra_info.append({"fid": fid})
                            consecutive_failures = 0
                            found = True
                        else:
                            consecutive_failures += 1
                    except Exception:
                        consecutive_failures += 1

                if not found and consecutive_failures > max_failures:
                    print(f"[INFO] Exceeded max failures, stopping at fid {start}")
                    break

        return spectra_info


    def _test_spectrum_exists(self, model_name: str, fid: int) -> bool:
        """Test if a spectrum exists without downloading it."""
        try:
            params = {'model': model_name, 'fid': fid, 'format': 'ascii'}
            response = self.session.head(
                self.spectra_url, params=params, timeout=10
            )
            return (
                response.status_code == 200 and 
                int(response.headers.get('content-length', '0')) > 1024
            )
        except (requests.RequestException, ValueError, KeyError):
            return False
    
    def _filter_spectra(self, spectra_info: List[Dict], filters: Dict) -> List[Dict]:
        """Filter spectra based on parameter ranges."""
        filtered = []
        
        for spectrum in spectra_info:
            include = True
            for param, (min_val, max_val) in filters.items():
                if param in spectrum:
                    value = float(spectrum[param])
                    if not (min_val <= value <= max_val):
                        include = False
                        break
            
            if include:
                filtered.append(spectrum)
        
        return filtered
    
    def _download_spectra_parallel(
        self, 
        model_name: str, 
        spectra_info: List[Dict], 
        output_dir: Path,
        show_progress: bool = True
    ) -> List[Dict]:
        """Download spectra in parallel with progress tracking."""
        successful_downloads = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_spectrum = {}
            for spectrum in spectra_info:
                fid = spectrum['fid']
                filename = f"{model_name}_fid{fid:06d}.txt"
                filepath = output_dir / filename
                
                # Skip if already exists
                if filepath.exists() and filepath.stat().st_size > 1024:
                    # Parse metadata from existing file
                    metadata = self._parse_spectrum_metadata(filepath)
                    metadata['file_name'] = filename
                    successful_downloads.append(metadata)
                    continue
                
                future = executor.submit(
                    self._download_single_spectrum, model_name, fid, filepath
                )
                future_to_spectrum[future] = (spectrum, filename, filepath)
            
            # Process completed downloads
            progress_bar = tqdm(
                total=len(future_to_spectrum),
                desc=f"Downloading {model_name}",
                disable=not show_progress
            )
            
            for future in as_completed(future_to_spectrum):
                spectrum, filename, filepath = future_to_spectrum[future]
                progress_bar.update(1)
                
                try:
                    success = future.result()
                    if success:
                        metadata = self._parse_spectrum_metadata(filepath)
                        metadata['file_name'] = filename
                        successful_downloads.append(metadata)
                    else:
                        # Clean up failed download
                        if filepath.exists():
                            filepath.unlink()
                except Exception as e:
                    warnings.warn(f"Error processing FID {spectrum['fid']}: {e}")
            
            progress_bar.close()
        
        return successful_downloads
    
    def _download_single_spectrum(
        self, 
        model_name: str, 
        fid: int, 
        filepath: Path
    ) -> bool:
        """Download a single spectrum file."""
        try:
            params = {'model': model_name, 'fid': fid, 'format': 'ascii'}
            response = self.session.get(
                self.spectra_url, params=params, timeout=self.timeout, stream=True
            )
            
            if response.status_code == 200 and len(response.content) > 1024:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return True
            
            return False
            
        except Exception:
            return False
    
    def _parse_spectrum_metadata(self, filepath: Path) -> Dict:
        """Parse metadata from a spectrum file."""
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
                            
                            # Extract numerical values
                            if key.lower() != 'file_name':
                                match = re.search(r'[-+]?\d*\.?\d+', value)
                                value = float(match.group()) if match else np.nan
                            
                            metadata[key] = value
                        except (ValueError, AttributeError):
                            continue
                    elif not line.startswith('#'):
                        break
        except Exception as e:
            warnings.warn(f"Error parsing metadata from {filepath}: {e}")
        
        return metadata
    
    def _create_lookup_table(self, output_dir: Path, metadata_list: List[Dict]):
        """Create a lookup table CSV for the downloaded model."""
        if not metadata_list:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(metadata_list)
        
        # Ensure file_name is first column
        columns = ['file_name']
        columns.extend([col for col in df.columns if col != 'file_name'])
        df = df[columns]
        
        # Save to CSV
        lookup_path = output_dir / 'lookup_table.csv'
        df.to_csv(lookup_path, index=False)
        
        print(f"Created lookup table with {len(df)} entries: {lookup_path}")
    
    def _analyze_parameter_ranges(self, spectra_info: List[Dict]) -> Dict:
        """Analyze parameter ranges from spectra metadata."""
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