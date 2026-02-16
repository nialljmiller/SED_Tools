#!/usr/bin/env python3
"""
NJM Mirror Grabber - Downloads from nillmill.ddns.net mirror
Downloads ALL files: individual .txt spectra, flux_cube.bin, .h5 bundle, lookup_table.csv
All files go through spectra_cleaner for unit standardization.

Supports axis-cutting: filter by Teff, logg, [M/H], and wavelength ranges
before/during download to avoid transferring unwanted data.
"""

import csv
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
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
    """Grabber for data from NJM mirror server. Downloads ALL files including individual spectra.
    
    Supports optional axis cuts to filter which spectra are downloaded:
        teff_range:  (min, max) or None
        logg_range:  (min, max) or None
        meta_range:  (min, max) or None
        wl_range:    (min, max) or None  — applied post-download by trimming each spectrum
    """
    
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
    
    # =========================================================================
    # AXIS CUTTING — fetch lookup table and filter file lists
    # =========================================================================

    def _fetch_remote_lookup(self, model_name: str, model_url: str) -> Optional[Dict[str, Dict]]:
        """Download lookup_table.csv from the mirror and parse it.
        
        Returns a dict mapping filename -> {teff, logg, meta} or None on failure.
        """
        lookup_url = urljoin(model_url, "lookup_table.csv")
        try:
            response = self.session.get(lookup_url, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"  [njm] Could not fetch lookup_table.csv: {e}")
            return None
        
        text = response.text
        
        # Parse CSV — handle the #-prefixed header that our pipeline writes
        lines = text.splitlines()
        if not lines:
            return None
        
        # Strip leading '#' from header
        header_line = lines[0].lstrip('#').strip()
        reader = csv.DictReader([header_line] + lines[1:])
        
        # Find columns
        fieldnames_lower = {f.strip().lower(): f for f in (reader.fieldnames or [])}
        
        file_col = None
        for candidate in ['file_name', 'filename', 'file']:
            if candidate in fieldnames_lower:
                file_col = fieldnames_lower[candidate]
                break
        
        teff_col = None
        for candidate in ['teff', 't_eff']:
            if candidate in fieldnames_lower:
                teff_col = fieldnames_lower[candidate]
                break
        
        logg_col = None
        for candidate in ['logg', 'log_g', 'gravity']:
            if candidate in fieldnames_lower:
                logg_col = fieldnames_lower[candidate]
                break
        
        meta_col = None
        for candidate in ['metallicity', 'meta', 'feh', '[m/h]', '[fe/h]', 'm_h']:
            if candidate in fieldnames_lower:
                meta_col = fieldnames_lower[candidate]
                break
        
        if not file_col:
            print("  [njm] lookup_table.csv has no recognizable filename column")
            return None
        
        result = {}
        for row in reader:
            fname = row.get(file_col, '').strip()
            if not fname:
                continue
            
            params = {}
            if teff_col:
                try:
                    params['teff'] = float(row[teff_col])
                except (ValueError, TypeError):
                    params['teff'] = float('nan')
            if logg_col:
                try:
                    params['logg'] = float(row[logg_col])
                except (ValueError, TypeError):
                    params['logg'] = float('nan')
            if meta_col:
                try:
                    params['meta'] = float(row[meta_col])
                except (ValueError, TypeError):
                    params['meta'] = float('nan')
            
            result[fname] = params
        
        return result if result else None

    def _filter_files_by_params(
        self,
        txt_files: List[str],
        lookup: Dict[str, Dict],
        teff_range: Optional[Tuple[float, float]] = None,
        logg_range: Optional[Tuple[float, float]] = None,
        meta_range: Optional[Tuple[float, float]] = None,
    ) -> List[str]:
        """Filter txt file list using parameter ranges from the lookup table.
        
        Files not found in the lookup table are KEPT (conservative — don't
        accidentally drop spectra we can't classify).
        """
        if not teff_range and not logg_range and not meta_range:
            return txt_files
        
        kept = []
        cut = 0
        unknown = 0
        
        for fname in txt_files:
            params = lookup.get(fname)
            
            if params is None:
                # Not in lookup — keep it (conservative)
                kept.append(fname)
                unknown += 1
                continue
            
            # Apply each axis cut
            skip = False
            
            if teff_range:
                teff = params.get('teff', float('nan'))
                if np.isfinite(teff):
                    if teff < teff_range[0] or teff > teff_range[1]:
                        skip = True
            
            if logg_range and not skip:
                logg = params.get('logg', float('nan'))
                if np.isfinite(logg):
                    if logg < logg_range[0] or logg > logg_range[1]:
                        skip = True
            
            if meta_range and not skip:
                meta = params.get('meta', float('nan'))
                if np.isfinite(meta):
                    if meta < meta_range[0] or meta > meta_range[1]:
                        skip = True
            
            if skip:
                cut += 1
            else:
                kept.append(fname)
        
        print(f"  [njm] Axis cut: {len(kept)} kept, {cut} removed", end="")
        if unknown:
            print(f" ({unknown} not in lookup — kept)")
        else:
            print()
        
        return kept

    def _apply_wavelength_cut(self, model_dir: str, txt_files: List[str],
                               wl_range: Tuple[float, float]) -> int:
        """Trim downloaded spectra to the given wavelength range.
        
        Rewrites each .txt file in-place, preserving headers.
        Returns the number of files trimmed.
        """
        wl_min, wl_max = wl_range
        trimmed = 0
        
        for fname in txt_files:
            fpath = os.path.join(model_dir, fname)
            if not os.path.exists(fpath):
                continue
            
            header_lines = []
            data_lines = []
            
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip().startswith('#') or not line.strip():
                            header_lines.append(line)
                        else:
                            # Parse wavelength from first column
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    wl = float(parts[0])
                                    if wl_min <= wl <= wl_max:
                                        data_lines.append(line)
                                except ValueError:
                                    data_lines.append(line)  # Keep unparseable lines
                            else:
                                data_lines.append(line)
                
                # Only rewrite if we actually cut something
                # (compare original data line count with filtered count)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    original_data_count = sum(
                        1 for line in f 
                        if line.strip() and not line.strip().startswith('#')
                    )
                
                if len(data_lines) < original_data_count:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        for line in header_lines:
                            f.write(line)
                        for line in data_lines:
                            f.write(line)
                    trimmed += 1
                    
            except Exception:
                continue  # Skip files that can't be read
        
        return trimmed

    # =========================================================================
    # GRID INFO — report parameter ranges before cutting
    # =========================================================================

    def get_grid_info(self, model_name: str, model_url: Optional[str] = None) -> Optional[Dict]:
        """Fetch the lookup table and report available parameter ranges.
        
        Returns dict with keys: teff_min, teff_max, logg_min, logg_max,
        meta_min, meta_max, n_spectra, or None on failure.
        """
        if model_url is None:
            model_url = f"{self.stellar_models_url}/{model_name}/"
        
        lookup = self._fetch_remote_lookup(model_name, model_url)
        if not lookup:
            return None
        
        teffs = [p['teff'] for p in lookup.values() if 'teff' in p and np.isfinite(p['teff'])]
        loggs = [p['logg'] for p in lookup.values() if 'logg' in p and np.isfinite(p['logg'])]
        metas = [p['meta'] for p in lookup.values() if 'meta' in p and np.isfinite(p['meta'])]
        
        info = {'n_spectra': len(lookup)}
        
        if teffs:
            info['teff_min'] = min(teffs)
            info['teff_max'] = max(teffs)
            info['teff_unique'] = len(set(teffs))
        if loggs:
            info['logg_min'] = min(loggs)
            info['logg_max'] = max(loggs)
            info['logg_unique'] = len(set(loggs))
        if metas:
            info['meta_min'] = min(metas)
            info['meta_max'] = max(metas)
            info['meta_unique'] = len(set(metas))
        
        return info
    
    # =========================================================================
    # DOWNLOAD
    # =========================================================================

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
    
    def download_model_spectra(
        self,
        model_name: str,
        metadata: Dict[str, any],
        teff_range: Optional[Tuple[float, float]] = None,
        logg_range: Optional[Tuple[float, float]] = None,
        meta_range: Optional[Tuple[float, float]] = None,
        wl_range: Optional[Tuple[float, float]] = None,
    ) -> int:
        """Download files from the mirror for a model, with optional axis cuts.
        
        Parameters
        ----------
        model_name : str
            Name of the model to download.
        metadata : dict
            Metadata from get_model_metadata().
        teff_range : tuple of (min, max), optional
            Only download spectra with Teff in this range.
        logg_range : tuple of (min, max), optional
            Only download spectra with logg in this range.
        meta_range : tuple of (min, max), optional
            Only download spectra with [M/H] in this range.
        wl_range : tuple of (min, max), optional
            Trim downloaded spectra to this wavelength range (Angstroms).
            Applied post-download since wavelength is within each file.
        
        Returns
        -------
        int
            Number of spectrum (.txt) files downloaded/available.
        """
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
        
        has_param_cuts = any([teff_range, logg_range, meta_range])
        
        # ── AXIS CUTTING: filter txt_files by stellar parameters ──
        if has_param_cuts and txt_files:
            print(f"  [njm] Applying parameter cuts...")
            if teff_range:
                print(f"    Teff:  {teff_range[0]:.0f} – {teff_range[1]:.0f} K")
            if logg_range:
                print(f"    logg:  {logg_range[0]:.2f} – {logg_range[1]:.2f}")
            if meta_range:
                print(f"    [M/H]: {meta_range[0]:+.2f} – {meta_range[1]:+.2f}")
            
            lookup = self._fetch_remote_lookup(model_name, model_url)
            if lookup:
                original_count = len(txt_files)
                txt_files = self._filter_files_by_params(
                    txt_files, lookup,
                    teff_range=teff_range,
                    logg_range=logg_range,
                    meta_range=meta_range,
                )
                print(f"  [njm] {original_count} → {len(txt_files)} spectra after cuts")
            else:
                print("  [njm] Warning: Could not fetch lookup table — downloading all spectra")
        
        all_files = txt_files + bin_files + h5_files + csv_files
        
        if not all_files:
            print(f"[njm] No files to download for {model_name}")
            return 0
        
        # When cutting by params, skip downloading pre-built .bin and .h5
        # since they contain the full grid and will be rebuilt from the
        # filtered .txt files anyway
        if has_param_cuts:
            skipped_aux = []
            if bin_files:
                skipped_aux.append(f"{len(bin_files)} .bin")
            if h5_files:
                skipped_aux.append(f"{len(h5_files)} .h5")
            if skipped_aux:
                print(f"  [njm] Skipping pre-built {', '.join(skipped_aux)} (will rebuild from filtered spectra)")
            bin_files = []
            h5_files = []
        
        print(f"[njm] Downloading {len(txt_files)} spectra + {len(bin_files + h5_files + csv_files)} auxiliary for {model_name}...")
        
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
        
        # ── WAVELENGTH CUT: trim spectra post-download ──
        if wl_range and txt_files:
            print(f"  [njm] Applying wavelength cut: {wl_range[0]:.1f} – {wl_range[1]:.1f} Å")
            trimmed = self._apply_wavelength_cut(model_dir, txt_files, wl_range)
            if trimmed:
                print(f"  [njm] Trimmed wavelength range in {trimmed} files")
            else:
                print(f"  [njm] All spectra already within requested range")
        
        # Return count of txt files (consistent with other grabbers)
        # The cleaning pipeline will regenerate .h5, .bin, .csv from cleaned .txt files
        return len(txt_files)
