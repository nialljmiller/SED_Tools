#!/usr/bin/env python3
"""
NJM Mirror Grabber - Downloads from nillmill.ddns.net mirror
Downloads ALL files: individual .txt spectra, flux_cube.bin, .h5 bundle, lookup_table.csv
All files go through spectra_cleaner for unit standardization.

Supports axis-cutting: filter by Teff, logg, [M/H], and wavelength ranges
before/during download to avoid transferring unwanted data.
"""

import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import requests
import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm

from .njm_filter_grabber import NJM_CONNECT_TIMEOUT

logger = logging.getLogger(__name__)

#: Connect and read timeouts of the availability check: a mirror that is down fails fast.
_TIMEOUT = (NJM_CONNECT_TIMEOUT, 10)


class NJMSpectraGrabber:
    """Grabber for data from NJM mirror server. Downloads ALL files including individual spectra.
    
    Supports optional axis cuts to filter which spectra are downloaded:
        teff_range:  (min, max) or None
        logg_range:  (min, max) or None
        meta_range:  (min, max) or None
        wl_range:    (min, max) or None  — applied post-download by trimming each spectrum
    """
    
    def __init__(self, base_dir: str = None, max_workers: int = 8):
        if base_dir is None:
            from .models import STELLAR_DIR_DEFAULT
            base_dir = str(STELLAR_DIR_DEFAULT)
        
        self.base_dir = base_dir
        self.max_workers = max_workers
        os.makedirs(base_dir, exist_ok=True)
        
        # Use HTTPS
        self.base_url = "https://nillmill.ddns.net/sed_tools"
        self.stellar_models_url = f"{self.base_url}/stellar_models"
        
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SED_Tools/1.0 (NJM Mirror)"})
        
        # Cached index data (populated by _check_availability)
        self._index_data = None
        
        # Check if server is available
        self._available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if the NJM mirror is available, failing fast when it does not answer.

        GET index.json, which is also cached for discover_models.  A failed
        certificate check is retried without verification, with a warning; an
        HTTP error status is confirmed with a HEAD of the base URL (any answer
        below 500 means the server is up).  No answer at all (refused, or no
        connection within NJM_CONNECT_TIMEOUT) means the mirror is down, and
        nothing else is tried.  Plain HTTP is tried only when HTTPS fails its
        TLS handshake with and without verification.  If the mirror is down,
        the warning lists each attempt's error.
        """
        index_url = f"{self.base_url}/index.json"
        errors: List[str] = []
        for verify in (True, False):
            try:
                resp = self.session.get(index_url, timeout=_TIMEOUT, verify=verify)
            except requests.exceptions.SSLError as exc:
                errors.append(f"GET {index_url}: {exc}")
                continue  # try without certificate verification
            except requests.RequestException as exc:
                errors.append(f"GET {index_url}: {exc}")
                return self._unreachable(errors)
            if resp.status_code == 200:
                self._cache_index(resp)
            else:
                errors.append(f"GET {index_url}: HTTP {resp.status_code}")
                if not self._answers_head(verify, errors):
                    return self._unreachable(errors)
            if not verify:
                self._disable_ssl_verification(errors)
            return True
        return self._plain_http(index_url, errors)

    def _answers_head(self, verify: bool, errors: List[str]) -> bool:
        """Whether a HEAD of the base URL gets an answer below 500 (even a 403 means the server is up)."""
        try:
            resp = self.session.head(self.base_url, timeout=_TIMEOUT, verify=verify)
        except requests.RequestException as exc:
            errors.append(f"HEAD {self.base_url}: {exc}")
            return False
        if resp.status_code < 500:
            return True
        errors.append(f"HEAD {self.base_url}: HTTP {resp.status_code}")
        return False

    def _plain_http(self, index_url: str, errors: List[str]) -> bool:
        """Use plain HTTP when HTTPS cannot complete its TLS handshake at all."""
        http_index_url = index_url.replace("https://", "http://")
        try:
            resp = self.session.get(http_index_url, timeout=_TIMEOUT, verify=False)
        except requests.RequestException as exc:
            errors.append(f"GET {http_index_url}: {exc}")
            return self._unreachable(errors)
        if resp.status_code != 200:
            errors.append(f"GET {http_index_url}: HTTP {resp.status_code}")
            return self._unreachable(errors)
        self._cache_index(resp)
        self.base_url = self.base_url.replace("https://", "http://")
        self.stellar_models_url = self.stellar_models_url.replace("https://", "http://")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.session.verify = False
        logger.warning("NJM mirror: HTTPS failed (%s); using plain HTTP at %s", "; ".join(errors), self.base_url)
        return True

    @staticmethod
    def _unreachable(errors: List[str]) -> bool:
        logger.warning("NJM mirror is unreachable, so its spectral grids are not offered. Attempts: %s",
                       "; ".join(errors))
        return False

    def _cache_index(self, resp: requests.Response) -> None:
        """Keep index.json for discover_models; a body that is not JSON just isn't cached."""
        try:
            self._index_data = resp.json()
        except ValueError:
            pass  # server is alive; discover_models fetches the index again and reports why it fails

    def _disable_ssl_verification(self, errors: List[str]) -> None:
        """Switch the session to skip certificate checks, saying why."""
        logger.warning("NJM mirror: certificate check failed (%s); continuing without certificate verification",
                       "; ".join(errors))
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.session.verify = False

    def is_available(self) -> bool:
        """Return True if the mirror is available."""
        return self._available
    
    def discover_models(self) -> List[str]:
        """Discover available models from the mirror using cached index.

        Falls back to fetching index.json again, then to the directory listing.
        If the mirror is unreachable or none of these works, returns [] after
        a warning, so the other sources can still be offered.
        """
        if not self._available:
            return []  # _check_availability has already warned
        
        # Use cached index data from availability check
        if self._index_data and self._index_data.get("models"):
            return _visible(self._index_data["models"])
        
        index_url = f"{self.base_url}/index.json"
        try:
            response = self.session.get(index_url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            index_problem = str(exc)
        else:
            self._index_data = data
            if data.get("models"):
                return _visible(data["models"])
            index_problem = "it lists no models"
            
        logger.warning("NJM mirror: could not read the grid list from %s (%s); using the directory listing instead",
                       index_url, index_problem)
        try:
            models = _visible(self._parse_directory_listing(self.stellar_models_url))
        except requests.RequestException as exc:
            logger.warning("NJM mirror: directory listing %s failed too (%s); NJM spectral grids are not offered",
                           self.stellar_models_url, exc)
            return []
        if not models:
            logger.warning("NJM mirror: directory listing %s is empty; NJM spectral grids are not offered",
                           self.stellar_models_url)
        return models
            
    def _parse_directory_listing(self, url: str) -> List[str]:
        """Parse Apache directory listing HTML to extract subdirectories/files.

        Raises requests.RequestException if the listing cannot be fetched.
        """
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
    
    def _discover_model_files(self, model_name: str) -> Dict[str, List[str]]:
        """Discover all files in a model directory, categorized by type.
        
        Tries directory listing first. If that fails (e.g. 403 because
        Apache has Options -Indexes), warns and falls back to:
          1) lookup_table.csv for txt filenames
          2) probing known auxiliary files directly
        Raises RuntimeError if lookup_table.csv cannot be read either.
        """
        model_url = f"{self.stellar_models_url}/{model_name}/"
        
        # Categorize files
        result = {
            'txt_files': [],
            'bin_files': [],
            'h5_files': [],
            'csv_files': [],
            'other_files': []
        }
        
        try:
            all_files = self._parse_directory_listing(model_url)
            listing_problem = "it is empty"
        except requests.RequestException as exc:
            all_files = []
            listing_problem = str(exc)
        
        if all_files:
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
        
        logger.warning("NJM mirror: no directory listing for %s (%s); taking spectrum names from lookup_table.csv",
                       model_url, listing_problem)
        result['txt_files'] = list(self._fetch_remote_lookup(model_name, model_url))
        
        # Probe for known auxiliary files
        for aux_name, category in [
            ("lookup_table.csv", 'csv_files'),
            (f"{model_name}.h5", 'h5_files'),
            ("flux_cube.bin", 'bin_files'),
        ]:
            if self._remote_file_exists(urljoin(model_url, aux_name)):
                result[category].append(aux_name)
        
        return result
    
    def _remote_file_exists(self, url: str) -> bool:
        """True if the mirror serves *url*.

        Asks with HEAD; if HEAD errors, with a GET for one byte. Raises
        RuntimeError if neither gets an answer.
        """
        try:
            return self.session.head(url, timeout=5).status_code == 200
        except requests.RequestException as head_error:
            try:
                resp = self.session.get(url, timeout=5, headers={"Range": "bytes=0-0"})
            except requests.RequestException as get_error:
                raise RuntimeError(f"NJM mirror: could not check {url}: HEAD failed ({head_error}), "
                                   f"GET failed ({get_error})") from get_error
            return resp.status_code in (200, 206)

    def get_model_metadata(self, model_name: str) -> Dict[str, any]:
        """Return metadata for a model, including list of all files.

        Raises RuntimeError if the mirror is unreachable or has no files for
        the model.
        """
        if not self._available:
            raise RuntimeError(f"NJM mirror {self.base_url} is unreachable; cannot fetch {model_name}")
        
        print(f"  Discovering files for {model_name} on NJM mirror...")
        
        # Discover all files in the model directory
        files = self._discover_model_files(model_name)
        model_url = f"{self.stellar_models_url}/{model_name}/"
        
        total = sum(len(v) for v in files.values())
        if total == 0:
            raise RuntimeError(f"NJM mirror has no files for {model_name} at {model_url}")
        
        print(f"    Found: {len(files['txt_files'])} spectra, {len(files['bin_files'])} .bin, "
              f"{len(files['h5_files'])} .h5, {len(files['csv_files'])} .csv")
        
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

    def _fetch_remote_lookup(self, model_name: str, model_url: str) -> Dict[str, Dict]:
        """Download lookup_table.csv from the mirror and parse it.
        
        Returns a dict mapping filename -> {teff, logg, meta}; a parameter
        that is not a number is NaN. Raises RuntimeError if the table cannot
        be fetched, has no file-name column, or lists no spectra.
        """
        lookup_url = urljoin(model_url, "lookup_table.csv")
        try:
            response = self.session.get(lookup_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"NJM mirror: could not fetch {lookup_url} for {model_name}: {exc}") from exc
        
        # Parse CSV — handle the #-prefixed header that our pipeline writes
        lines = response.text.splitlines()
        if not lines:
            raise RuntimeError(f"NJM mirror: {lookup_url} is empty")
        
        # Strip leading '#' from header
        header_line = lines[0].lstrip('#').strip()
        reader = csv.DictReader([header_line] + lines[1:])
        
        # Find columns
        fieldnames_lower = {f.strip().lower(): f for f in (reader.fieldnames or [])}
        file_col = _find_column(fieldnames_lower, ['file_name', 'filename', 'file'])
        teff_col = _find_column(fieldnames_lower, ['teff', 't_eff'])
        logg_col = _find_column(fieldnames_lower, ['logg', 'log_g', 'gravity'])
        meta_col = _find_column(fieldnames_lower, ['metallicity', 'meta', 'feh', '[m/h]', '[fe/h]', 'm_h'])
        
        if not file_col:
            raise RuntimeError(f"NJM mirror: {lookup_url} has no recognizable filename column")
        
        result = {}
        for row in reader:
            fname = row.get(file_col, '').strip()
            if not fname:
                continue
            
            params = {}
            if teff_col:
                params['teff'] = _float_or_nan(row[teff_col])
            if logg_col:
                params['logg'] = _float_or_nan(row[logg_col])
            if meta_col:
                params['meta'] = _float_or_nan(row[meta_col])
            
            result[fname] = params
        
        if not result:
            raise RuntimeError(f"NJM mirror: {lookup_url} lists no spectra")
        return result

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
                               wl_range: Tuple[float, float]) -> Tuple[int, List[str]]:
        """Trim downloaded spectra to the given wavelength range.
        
        Rewrites each .txt file in-place, preserving headers. Files that are
        not on disk (failed downloads, which the caller reports) are skipped.
        Returns the number of files trimmed and a 'file: error' entry for
        each file that could not be read or rewritten.
        """
        trimmed = 0
        failures = []
        
        for fname in txt_files:
            fpath = os.path.join(model_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                if _trim_spectrum_file(fpath, wl_range):
                    trimmed += 1
            except OSError as exc:
                failures.append(f"{fname}: wavelength cut failed: {exc}")
            
        return trimmed, failures

    # =========================================================================
    # GRID INFO — report parameter ranges before cutting
    # =========================================================================

    def get_grid_info(self, model_name: str, model_url: Optional[str] = None) -> Optional[Dict]:
        """Fetch the lookup table and report available parameter ranges.
        
        Returns dict with keys: teff_min, teff_max, logg_min, logg_max,
        meta_min, meta_max, n_spectra, or None (after a warning saying why)
        if the lookup table cannot be read.
        """
        if model_url is None:
            model_url = f"{self.stellar_models_url}/{model_name}/"
        
        try:
            lookup = self._fetch_remote_lookup(model_name, model_url)
        except RuntimeError as exc:
            logger.warning("No grid summary for %s: %s", model_name, exc)
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

    def _download_one_file(self, url: str, output_path: str) -> None:
        """Download a single file. Raises on failure, leaving no partial file."""
        try:
            response = self.session.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        except Exception:
            # A partial file larger than the skip threshold would pass as complete on the next run
            if os.path.exists(output_path):
                os.remove(output_path)
            raise

        if os.path.getsize(output_path) == 0:
            raise ValueError(f"{url} returned an empty file")

    def _fetch_file(self, model_dir: str, model_url: str, filename: str,
                    wl_query: str = "", force: bool = False) -> Tuple[str, Optional[str]]:
        """Download one model file unless a complete copy is already on disk.

        Returns (status, error): status is 'ok', 'skip' or 'fail', and error
        says why a file failed.
        """
        output_path = os.path.join(model_dir, filename)

        # Skip if already exists (unless forced for wl_cut re-downloads)
        if not force and os.path.exists(output_path):
            size = os.path.getsize(output_path)
            min_size = 100 if filename.endswith('.csv') else 1024
            if size > min_size:
                return 'skip', None

        # Server-side wl_cut: query params on the normal file URL — Apache rewrites to PHP
        url = urljoin(model_url, filename)
        if filename.endswith('.txt'):
            url += wl_query

        try:
            self._download_one_file(url, output_path)
        except (requests.RequestException, OSError, ValueError) as exc:
            return 'fail', f"{filename}: {exc}"
        return 'ok', None

    def _download_files(self, model_dir: str, model_url: str, txt_files: List[str],
                        aux_files: List[str], wl_query: str,
                        force_txt: bool) -> Tuple[Dict[str, int], List[str]]:
        """Download the spectra (in parallel), then the auxiliary files.

        Returns the count of each status ('ok', 'skip', 'fail') and a
        'file: error' entry for each failure.
        """
        counts = {'ok': 0, 'skip': 0, 'fail': 0}
        failures: List[str] = []
        
        # Download .txt files first (parallel) - these are the primary data
        # Force re-download when server-side wl_cut is active (existing files
        # may have a different wavelength range from a prior download)
        if txt_files:
            print(f"  Downloading {len(txt_files)} spectrum files...")
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = [ex.submit(self._fetch_file, model_dir, model_url, f, wl_query, force_txt)
                           for f in txt_files]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="  Spectra"):
                    status, error = fut.result()
                    counts[status] += 1
                    if error:
                        failures.append(error)
        
        # Download auxiliary files (sequential)
        # These will be REGENERATED after cleaning to ensure consistency
        if aux_files:
            print(f"  Downloading {len(aux_files)} auxiliary files...")
            for filename in aux_files:
                status, error = self._fetch_file(model_dir, model_url, filename)
                counts[status] += 1
                print(f"    [{status}] {filename}")
                if error:
                    failures.append(error)
        
        return counts, failures

    def _server_wl_cut_available(self, model_name: str, probe_file: str,
                                 wl_range: Tuple[float, float]) -> bool:
        """Ask for one spectrum with wl_min/wl_max; True if the mirror cuts server-side.

        If the server has the rewrite rule + wl_cut.php, the response
        will include an X-WL-Cut header. Without it, the raw file is
        served (no header). Each file URL is unique, so mod_evasive
        treats them as different pages — no rate-limit issues.
        """
        probe_url = (f"{self.stellar_models_url}/{model_name}/{probe_file}"
                     f"?wl_min={wl_range[0]}&wl_max={wl_range[1]}")
        try:
            resp = self.session.get(probe_url, timeout=10, stream=True)
        except requests.RequestException as exc:
            logger.warning("NJM mirror: could not check for server-side wavelength cuts (%s); "
                           "trimming spectra after download instead", exc)
            return False
        available = resp.status_code == 200 and 'X-WL-Cut' in resp.headers
        resp.close()
        return available

    def _select_spectra(self, model_name: str, model_url: str, txt_files: List[str],
                        teff_range, logg_range, meta_range) -> List[str]:
        """Apply the parameter cuts to the spectrum list using the remote lookup table.

        Raises RuntimeError if the lookup table cannot be read, rather than
        downloading spectra outside the requested ranges.
        """
        print(f"  [njm] Applying parameter cuts...")
        if teff_range:
            print(f"    Teff:  {teff_range[0]:.0f} – {teff_range[1]:.0f} K")
        if logg_range:
            print(f"    logg:  {logg_range[0]:.2f} – {logg_range[1]:.2f}")
        if meta_range:
            print(f"    [M/H]: {meta_range[0]:+.2f} – {meta_range[1]:+.2f}")

        lookup = self._fetch_remote_lookup(model_name, model_url)
        selected = self._filter_files_by_params(
            txt_files, lookup,
            teff_range=teff_range,
            logg_range=logg_range,
            meta_range=meta_range,
        )
        print(f"  [njm] {len(txt_files)} → {len(selected)} spectra after cuts")
        return selected
    
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
            Uses server-side wl_cut.php if available, otherwise trims
            client-side after download.
        
        Returns
        -------
        int
            Number of spectrum (.txt) files downloaded/available.

        Raises
        ------
        RuntimeError
            If the mirror is unreachable, the parameter cuts cannot be applied
            (lookup table unreadable), or any file could not be downloaded or
            trimmed. Every other file is still downloaded first, and the error
            lists each failure.
        """
        if not self._available:
            raise RuntimeError(f"NJM mirror {self.base_url} is unreachable; cannot download {model_name}")
        
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
        if has_param_cuts and txt_files:
            txt_files = self._select_spectra(model_name, model_url, txt_files,
                                             teff_range, logg_range, meta_range)
        
        all_files = txt_files + bin_files + h5_files + csv_files
        
        if not all_files:
            print(f"[njm] No files to download for {model_name}")
            return 0
        
        # When cutting by params or wavelength, skip downloading pre-built
        # .bin and .h5 since they contain the full grid and will be rebuilt
        # from the filtered .txt files anyway
        if has_param_cuts or wl_range:
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
        
        server_wl_cut = False
        if wl_range and txt_files:
            server_wl_cut = self._server_wl_cut_available(model_name, txt_files[0], wl_range)
            if server_wl_cut:
                print(f"  [njm] Server-side wavelength cut active ({wl_range[0]:.1f} – {wl_range[1]:.1f} Å)")
        wl_query = f"?wl_min={wl_range[0]}&wl_max={wl_range[1]}" if server_wl_cut else ""

        aux_files = bin_files + h5_files + csv_files
        counts, failures = self._download_files(model_dir, model_url, txt_files, aux_files,
                                                wl_query, force_txt=server_wl_cut)
        
        print(f"[njm] {model_name}: {counts['ok']} downloaded, {counts['skip']} skipped, {counts['fail']} failed")
        
        # ── WAVELENGTH CUT: trim spectra post-download (only if server-side cut unavailable) ──
        if wl_range and txt_files and not server_wl_cut:
            print(f"  [njm] Applying wavelength cut (client-side): {wl_range[0]:.1f} – {wl_range[1]:.1f} Å")
            trimmed, trim_failures = self._apply_wavelength_cut(model_dir, txt_files, wl_range)
            failures.extend(trim_failures)
            if trimmed:
                print(f"  [njm] Trimmed wavelength range in {trimmed} files")
            else:
                print(f"  [njm] All spectra already within requested range")

        if failures:
            raise RuntimeError(
                f"NJM: {len(failures)} file(s) of {model_name} failed; the rest are in {model_dir}. "
                f"Rerun to retry the failed files:\n  " + "\n  ".join(failures)
            )
        
        # Return count of txt files (consistent with other grabbers)
        # The cleaning pipeline will regenerate .h5, .bin, .csv from cleaned .txt files
        return len(txt_files)


def _visible(names: List[str]) -> List[str]:
    """Drop hidden entries (names starting with '.')."""
    return [m for m in names if not m.startswith('.')]


def _find_column(fieldnames_lower: Dict[str, str], candidates: List[str]) -> Optional[str]:
    """Return the original name of the first candidate column present, or None."""
    for candidate in candidates:
        if candidate in fieldnames_lower:
            return fieldnames_lower[candidate]
    return None


def _float_or_nan(value) -> float:
    """A lookup-table cell as a float; NaN (an unknown parameter) if empty or not a number."""
    try:
        return float(value)
    except (ValueError, TypeError):  # TypeError: csv gives None for a short row
        return float('nan')


def _trim_spectrum_file(fpath: str, wl_range: Tuple[float, float]) -> bool:
    """Drop data rows outside wl_range, rewriting the file; True if anything was cut.

    Comment and blank lines are written first, then the kept data rows.
    Rows whose first field is not a number are kept.
    """
    wl_min, wl_max = wl_range
    header_lines = []
    data_lines = []
    n_data = 0

    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#') or not stripped:
                header_lines.append(line)
                continue
            n_data += 1
            parts = stripped.split()
            if len(parts) < 2:
                data_lines.append(line)
                continue
            try:
                wl = float(parts[0])
            except ValueError:
                data_lines.append(line)  # Keep unparseable lines
                continue
            if wl_min <= wl <= wl_max:
                data_lines.append(line)

    if len(data_lines) == n_data:
        return False
    with open(fpath, 'w', encoding='utf-8') as f:
        f.writelines(header_lines)
        f.writelines(data_lines)
    return True
