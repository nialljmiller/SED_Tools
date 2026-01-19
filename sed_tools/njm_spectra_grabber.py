#!/usr/bin/env python3
"""
NJM Mirror Grabber - Downloads from nillmill.ddns.net mirror
Downloads ALL files: individual .txt spectra, flux_cube.bin, .h5 bundle, lookup_table.csv
All files go through spectra_cleaner for unit standardization.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
# Suppress SSL warnings when verification is disabled
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


class NJMSpectraGrabber:
    """Grabber for data from NJM mirror server. Downloads ALL files including individual spectra."""
    
    def __init__(self, base_dir: str = "../data/stellar_models/", max_workers: int = 8):
        self.base_dir = base_dir
        self.max_workers = max_workers
        os.makedirs(base_dir, exist_ok=True)
        
        # Use HTTPS
        self.base_url = "https://nillmill.ddns.net/sed_tools"
        self.stellar_models_url = f"{self.base_url}/stellar_models"
        
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SED_Tools/1.0 (NJM Mirror)"})
        
        # Check if server is available
        self._available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if the NJM mirror is available and responding."""
        try:
            response = self.session.head(self.base_url, timeout=5)
            response.raise_for_status()
            return True
        except requests.exceptions.SSLError:
            try:
                response = self.session.head(self.base_url, timeout=5, verify=False)
                response.raise_for_status()
                self.session.verify = False
                print("  [njm] Note: SSL verification disabled (certificate issue)")
                return True
            except Exception:
                return False
        except Exception:
            return False
    
    def is_available(self) -> bool:
        """Return True if the mirror is available."""
        return self._available
    
    def discover_models(self) -> List[str]:
        """Discover available models from the mirror."""
        if not self._available:
            return []
        
        try:
            # Try to get the index.json first
            index_url = f"{self.base_url}/index.json"
            response = self.session.get(index_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            if models:
                return models
            
        except Exception:
            pass
        
        # Fallback: parse directory listing
        try:
            return self._parse_directory_listing(self.stellar_models_url)
        except Exception as e:
            print(f"[njm] Could not fetch model list: {e}")
            return []
    
    def _parse_directory_listing(self, url: str) -> List[str]:
        """Parse Apache directory listing HTML to extract subdirectories/files."""
        if not _HAS_BS4:
            print("[njm] BeautifulSoup not available, cannot parse directory listing")
            return []
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            items = []
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href and href not in ['..', '/', '../', '?', '']:
                    # Remove trailing slash
                    item = href.rstrip('/')
                    # Skip parent links, query strings, and sorting links
                    if not item.startswith('..') and not item.startswith('/') and not item.startswith('?'):
                        # Skip column sorting links
                        if item not in ['Name', 'Last modified', 'Size', 'Description']:
                            items.append(item)
            
            return items
            
        except Exception as e:
            print(f"[njm] Directory listing failed: {e}")
            return []
    
    def _discover_model_files(self, model_name: str) -> Dict[str, List[str]]:
        """Discover all files in a model directory, categorized by type."""
        model_url = f"{self.stellar_models_url}/{model_name}/"
        
        all_files = self._parse_directory_listing(model_url)
        
        # Categorize files
        result = {
            'txt_files': [],
            'bin_files': [],
            'h5_files': [],
            'csv_files': [],
            'other_files': []
        }
        
        for f in all_files:
            f_lower = f.lower()
            if f_lower.endswith('.txt'):
                result['txt_files'].append(f)
            elif f_lower.endswith('.bin'):
                result['bin_files'].append(f)
            elif f_lower.endswith('.h5'):
                result['h5_files'].append(f)
            elif f_lower.endswith('.csv'):
                result['csv_files'].append(f)
            else:
                result['other_files'].append(f)
        
        return result
    
    def get_model_metadata(self, model_name: str) -> Dict[str, any]:
        """Return metadata for a model, including list of all files."""
        if not self._available:
            return {}
        
        print(f"  Discovering files for {model_name} on NJM mirror...")
        
        # Discover all files in the model directory
        files = self._discover_model_files(model_name)
        
        total = sum(len(v) for v in files.values())
        if total == 0:
            print(f"    No files found for {model_name}")
            return {}
        
        print(f"    Found: {len(files['txt_files'])} spectra, {len(files['bin_files'])} .bin, "
              f"{len(files['h5_files'])} .h5, {len(files['csv_files'])} .csv")
        
        model_url = f"{self.stellar_models_url}/{model_name}/"
        
        return {
            "provider": "njm-mirror",
            "model_name": model_name,
            "model_url": model_url,
            "txt_files": files['txt_files'],
            "bin_files": files['bin_files'],
            "h5_files": files['h5_files'],
            "csv_files": files['csv_files'],
            # CRITICAL: Set to False so ALL files go through cleaning/unit standardization
            "pre_processed": False,
        }
    
    def _download_one_file(self, url: str, output_path: str) -> bool:
        """Download a single file. Returns True on success."""
        try:
            response = self.session.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            
            return os.path.getsize(output_path) > 0
            
        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    def download_model_spectra(self, model_name: str, metadata: Dict[str, any]) -> int:
        """Download ALL files from the mirror for a model."""
        if not self._available:
            print("[njm] Mirror not available")
            return 0
        
        if not metadata:
            print(f"[njm] No metadata for {model_name}")
            return 0
        
        model_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        model_url = metadata.get("model_url", f"{self.stellar_models_url}/{model_name}/")
        
        txt_files = metadata.get("txt_files", [])
        bin_files = metadata.get("bin_files", [])
        h5_files = metadata.get("h5_files", [])
        csv_files = metadata.get("csv_files", [])
        
        all_files = txt_files + bin_files + h5_files + csv_files
        
        if not all_files:
            print(f"[njm] No files found for {model_name}")
            return 0
        
        print(f"[njm] Downloading {len(all_files)} files for {model_name}...")
        
        downloaded = 0
        skipped = 0
        failed = 0
        
        def download_task(filename: str) -> tuple:
            """Download a single file. Returns (status, filename)."""
            url = urljoin(model_url, filename)
            output_path = os.path.join(model_dir, filename)
            
            # Skip if already exists with reasonable size
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                min_size = 100 if filename.endswith('.csv') else 1024
                if size > min_size:
                    return ('skip', filename)
            
            if self._download_one_file(url, output_path):
                return ('ok', filename)
            else:
                return ('fail', filename)
        
        # Download .txt files first (parallel) - these are the primary data
        if txt_files:
            print(f"  Downloading {len(txt_files)} spectrum files...")
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = {ex.submit(download_task, f): f for f in txt_files}
                
                if _HAS_TQDM:
                    iterator = tqdm(as_completed(futures), total=len(futures), desc="  Spectra")
                else:
                    iterator = as_completed(futures)
                    print(f"    Processing {len(futures)} files...")
                
                for fut in iterator:
                    status, fname = fut.result()
                    if status == 'ok':
                        downloaded += 1
                    elif status == 'skip':
                        skipped += 1
                    else:
                        failed += 1
        
        # Download auxiliary files (sequential)
        # These will be REGENERATED after cleaning to ensure consistency
        aux_files = bin_files + h5_files + csv_files
        if aux_files:
            print(f"  Downloading {len(aux_files)} auxiliary files...")
            for filename in aux_files:
                status, fname = download_task(filename)
                if status == 'ok':
                    downloaded += 1
                    print(f"    [ok] {filename}")
                elif status == 'skip':
                    skipped += 1
                    print(f"    [skip] {filename}")
                else:
                    failed += 1
                    print(f"    [fail] {filename}")
        
        print(f"[njm] {model_name}: {downloaded} downloaded, {skipped} skipped, {failed} failed")
        
        # Return count of txt files (consistent with other grabbers)
        # The cleaning pipeline will regenerate .h5, .bin, .csv from cleaned .txt files
        return len(txt_files)