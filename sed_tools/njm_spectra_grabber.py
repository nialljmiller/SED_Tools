#!/usr/bin/env python3
"""
NJM Mirror Grabber - Downloads from nillmill.ddns.net mirror
Now with HTTPS support and SSL handling
"""

import os
import json
import requests
from typing import List, Dict, Optional
from urllib.parse import urljoin

class NJMSpectraGrabber:
    """Grabber for pre-processed data from NJM mirror server."""
    
    def __init__(self, base_dir: str = "../data/stellar_models/", max_workers: int = 5):
        self.base_dir = base_dir
        self.max_workers = max_workers
        os.makedirs(base_dir, exist_ok=True)
        
        # Use HTTPS (your server redirects HTTP → HTTPS anyway)
        self.base_url = "https://nillmill.ddns.net/sed_tools"
        self.stellar_models_url = f"{self.base_url}/stellar_models"
        
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SED_Tools/1.0 (NJM Mirror)"})
        
        # Check if server is available
        self._available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if the NJM mirror is available and responding."""
        try:
            # Try with SSL verification first
            response = self.session.head(self.base_url, timeout=5)
            response.raise_for_status()
            return True
        except requests.exceptions.SSLError:
            # SSL certificate issue - try without verification
            try:
                response = self.session.head(self.base_url, timeout=5, verify=False)
                response.raise_for_status()
                # Disable SSL verification for this session
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
            return data.get("models", [])
            
        except Exception as e:
            print(f"[njm] Could not fetch model list: {e}")
            return []
    
    def get_model_metadata(self, model_name: str) -> Dict[str, any]:
        """Return metadata for a model."""
        if not self._available:
            return {}
        
        print(f"  Model available on NJM mirror: {model_name}")
        
        # Check what files are available
        model_url = f"{self.stellar_models_url}/{model_name}/"
        
        available_files = {
            "flux_cube": f"{model_url}flux_cube.bin",
            "lookup_table": f"{model_url}lookup_table.csv",
            "h5_bundle": f"{model_url}{model_name}.h5",
        }
        
        return {
            "provider": "njm-mirror",
            "model_name": model_name,
            "urls": available_files,
            "pre_processed": True,  # Data from NJM is already processed
        }
    
    def download_model_spectra(self, model_name: str, metadata: Dict[str, any]) -> int:
        """Download pre-processed data from the mirror."""
        if not self._available:
            print("[njm] Mirror not available")
            return 0
        
        model_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        urls = metadata.get("urls", {})
        downloaded = 0
        
        print(f"[njm] Downloading from NJM mirror: {model_name}")
        
        # Download key files
        files_to_download = [
            ("flux_cube", "flux_cube.bin"),
            ("lookup_table", "lookup_table.csv"),
            ("h5_bundle", f"{model_name}.h5"),
        ]
        
        for key, filename in files_to_download:
            if key not in urls:
                continue
            
            url = urls[key]
            output_path = os.path.join(model_dir, filename)
            
            # Skip if already exists
            if os.path.exists(output_path):
                print(f"  [skip] {filename} already exists")
                downloaded += 1
                continue
            
            try:
                print(f"  [download] {filename}...")
                response = self.session.get(url, timeout=300, stream=True)
                response.raise_for_status()
                
                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                
                with open(output_path, 'wb') as f:
                    if total_size > 0:
                        downloaded_size = 0
                        chunk_size = 1024 * 1024  # 1MB chunks
                        
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                percent = (downloaded_size / total_size) * 100
                                print(f"    {percent:.1f}%", end='\r')
                        print()  # New line after progress
                    else:
                        f.write(response.content)
                
                print(f"  [saved] {filename}")
                downloaded += 1
                
            except Exception as e:
                print(f"  [error] Failed to download {filename}: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
        
        if downloaded > 0:
            print(f"[njm] Downloaded {downloaded} files from Niall J Millers mirror")
        
        return downloaded


# Suppress SSL warnings when verification is disabled
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
