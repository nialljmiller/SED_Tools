#!/usr/bin/env python3
"""
NJM Filter Grabber - Downloads filter profiles from nillmill.ddns.net mirror
Complements the spectra grabber for complete data access
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup

from .filter_metadata import SOURCE_ASSUMED, SVOMetadataClient, migrate_filter_file, tag_filter_file

logger = logging.getLogger(__name__)

class NJMFilterGrabber:
    """Grabber for filter profiles from NJM mirror server."""
    
    def __init__(self, base_dir: str = "../data/filters/", max_workers: int = 5):
        self.base_dir = base_dir
        self.max_workers = max_workers
        os.makedirs(base_dir, exist_ok=True)

        self.base_url = "https://nillmill.ddns.net/sed_tools"
        self.filters_url = f"{self.base_url}/filters"

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SED_Tools/1.0 (NJM Mirror)"})

        self._cached_facilities: Optional[List[str]] = None
        self._metadata_client = SVOMetadataClient()
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
                self._disable_ssl_verification()
                print("  [njm] Note: SSL verification disabled (certificate issue)")
                return True
            except requests.RequestException:
                return False
        except requests.RequestException:
            return False
    
    def _disable_ssl_verification(self) -> None:
        """Suppress SSL warnings and switch the session to skip certificate checks."""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.session.verify = False

    def is_available(self) -> bool:
        """Return True if the mirror is available."""
        return self._available

    def _get(self, url: str, *, timeout: int = 30) -> requests.Response:
        """Retry transient mirror failures, including Apache's temporary 403 block."""
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in (403, 429, 500, 502, 503, 504):
                    raise
                if attempt == 2:
                    raise
                delay = (1, 60)[attempt] if status == 403 else (1, 3)[attempt]
                if status == 429:
                    retry_after = getattr(exc.response, "headers", {}).get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = min(int(retry_after), 60)
                logger.warning("NJM request failed (%s); retrying in %s s: %s", status or type(exc).__name__, delay, url)
                time.sleep(delay)

    def _save_curve(self, url: str, target: Path, *, timeout: int = 30) -> None:
        """Save a mirror curve, moving any 0.3.0 ``#`` metadata lines onto its column row."""
        response = self._get(url, timeout=timeout)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(response.text, encoding="utf-8")
        migrate_filter_file(target)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Lowercase and strip separators for fuzzy name comparison."""
        return re.sub(r'[-_\s]', '', name).lower()

    def find_facility(self, svo_key: str) -> Optional[str]:
        """Return the NJM directory name that best matches an SVO facility key.

        Tries exact match first, then normalized prefix match to handle
        differences like 'Bepi-Colom' (SVO) vs 'Bepi_Colombo' (NJM).
        """
        facilities = self.discover_facilities()
        if svo_key in facilities:
            return svo_key
        svo_norm = self._normalize_name(svo_key)
        for name in facilities:
            njm_norm = self._normalize_name(name)
            if njm_norm.startswith(svo_norm) or svo_norm.startswith(njm_norm):
                return name
        return None

    def get_exclusive_facilities(self, svo_keys: List[str]) -> List[str]:
        """Return NJM facility names that have no matching SVO facility.

        Uses the same normalized prefix match as find_facility so that, e.g.,
        'Bepi_Colombo' (NJM) correctly matches 'Bepi-Colom' (SVO) and is NOT
        returned as exclusive.
        """
        svo_norms = {self._normalize_name(k) for k in svo_keys}
        exclusive = []
        for fac in self.discover_facilities():
            fac_norm = self._normalize_name(fac)
            matched = any(
                fac_norm.startswith(s) or s.startswith(fac_norm)
                for s in svo_norms
            )
            if not matched:
                exclusive.append(fac)
        return exclusive

    def find_instrument(self, njm_facility: str, svo_instrument: str) -> Optional[str]:
        """Return the NJM instrument name that best matches an SVO instrument key."""
        instruments = self.discover_instruments(njm_facility)
        if svo_instrument in instruments:
            return svo_instrument
        # Exact (normalised) matches only: a prefix match would hand back a
        # different instrument, e.g. GAIA0 (DR1) -> the mirror's GAIA set.
        svo_norm = self._normalize_name(svo_instrument)
        for name in instruments:
            if self._normalize_name(name) == svo_norm:
                return name
        return None

    def discover_facilities(self) -> List[str]:
        """Discover available filter facilities from the mirror (cached)."""
        if not self._available:
            return []
        if self._cached_facilities is not None:
            return self._cached_facilities

        try:
            index_url = f"{self.base_url}/index.json"
            response = self.session.get(index_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            facilities = data.get("filters", {}).get("facilities", [])
            if not facilities:
                facilities = self._parse_directory_listing(self.filters_url)
        except (requests.RequestException, ValueError) as e:
            logger.debug("[njm] Could not fetch facility list: %s", e, exc_info=True)
            facilities = self._parse_directory_listing(self.filters_url)

        self._cached_facilities = facilities
        return facilities
    
    def discover_instruments(self, facility: str) -> List[str]:
        """Discover available instruments for a facility."""
        if not self._available:
            return []
        
        return self._parse_directory_listing(f"{self.filters_url}/{facility}/")
    
    def discover_filters(self, facility: str, instrument: str) -> List[str]:
        """Discover available filters for a facility/instrument."""
        if not self._available:
            return []
        
        instrument_url = f"{self.filters_url}/{facility}/{instrument}/"
        try:
            response = self._get(f"{instrument_url}{instrument}", timeout=10)
        except requests.RequestException:
            logger.debug("No index for %s/%s; using directory listing", facility, instrument)
        else:
            return [line.strip() for line in response.text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]

        return [name.removesuffix(".dat") for name in self._parse_directory_listing(instrument_url)
                if name.endswith(".dat")]

    def _parse_directory_listing(self, url: str) -> List[str]:
        """Parse Apache directory listing HTML to extract subdirectories/files."""
        try:
            response = self._get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            items = []
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href and href not in ['..', '/', '../'] and not href.startswith('?'):
                    # Remove trailing slash for directories
                    item = href.rstrip('/')
                    # Skip parent directory links
                    if not item.startswith('..') and not item.startswith('/'):
                        items.append(item)
            
            return items
            
        except requests.RequestException:
            return []
    
    def download_filters(self, facility: str, instrument: str, 
                        filters: Optional[List[str]] = None) -> int:
        """
        Download filters for a facility/instrument.
        
        Args:
            facility: Facility name (e.g., 'Generic', 'GAIA')
            instrument: Instrument name (e.g., 'Johnson', 'G3')
            filters: Optional list of specific filters to download.
                    If None, downloads all available filters.
        
        Returns:
            Number of filters downloaded
        """
        if not self._available:
            print("[njm] Mirror not available")
            return 0
        
        # Discover available filters if not specified
        if filters is None:
            filters = self.discover_filters(facility, instrument)
        
        if not filters:
            print(f"[njm] No filters found for {facility}/{instrument}")
            return 0
        
        # Create output directory
        output_dir = os.path.join(self.base_dir, facility, instrument)
        os.makedirs(output_dir, exist_ok=True)
        
        downloaded = 0
        skipped = 0
        failed = 0
        
        print(f"[njm] Downloading filters: {facility}/{instrument}")
        print(f"  Found {len(filters)} filters")
        
        for filter_name in filters:
            # Handle both with and without .dat extension
            filter_filename = filter_name if filter_name.endswith('.dat') else f"{filter_name}.dat"
            filter_url = f"{self.filters_url}/{facility}/{instrument}/{filter_filename}"
            output_path = os.path.join(output_dir, filter_filename)
            
            # Skip if already exists
            if os.path.exists(output_path):
                tag_filter_file(output_path, client=self._metadata_client, facility=facility)
                skipped += 1
                continue
            
            try:
                self._save_curve(filter_url, Path(output_path))

                # Mirror files should already carry a DetectorType; this is a
                # no-op for those and resolves it against SVO otherwise.
                result = tag_filter_file(output_path, client=self._metadata_client, facility=facility)
                if result.source == SOURCE_ASSUMED:
                    print(f"  [warn] {filter_name}: detector type unverified, assuming {result.detector_type}"
                          + (f" ({result.note})" if result.note else ""))

                downloaded += 1
                
            except requests.RequestException as e:
                print(f"  [error] {filter_name}: {e}")
                failed += 1
                if os.path.exists(output_path):
                    os.remove(output_path)
        
        # Only the mirror (or its manifest) can declare the full filter set.
        # An index built from the files seen so far would certify a partial set.
        index_path = Path(output_dir) / instrument
        remote_index = None
        try:
            response = self._get(f"{self.filters_url}/{facility}/{instrument}/{instrument}", timeout=10)
        except requests.RequestException:
            logger.warning("NJM index unavailable for %s/%s; completeness cannot be verified", facility, instrument)
        else:
            index_path.write_text(response.text, encoding="utf-8")
            remote_index = [line.strip() for line in response.text.splitlines() if line.strip()]

        print(f"[njm] {facility}/{instrument}: {downloaded} downloaded, {skipped} skipped, {failed} failed")
        self.download_components(facility, instrument)
        if remote_index and not (Path(output_dir) / "manifest.json").is_file():
            from .filter_components import InstrumentManifest

            manifest = InstrumentManifest.from_published(output_dir)
            manifest.bands = sorted(name.removesuffix(".dat") for name in remote_index if name.endswith(".dat"))
            manifest.save(output_dir)
        return downloaded
    
    def download_all_filters(self, facility: Optional[str] = None) -> int:
        """
        Download all available filters, optionally limited to a facility.
        
        Args:
            facility: Optional facility name. If None, downloads from all facilities.
        
        Returns:
            Total number of filters downloaded
        """
        if not self._available:
            print("[njm] Mirror not available")
            return 0
        
        total_downloaded = 0
        
        # Discover facilities
        if facility:
            facilities = [facility]
        else:
            facilities = self.discover_facilities()
        
        if not facilities:
            print("[njm] No facilities found")
            return 0
        
        print(f"[njm] Downloading filters from {len(facilities)} facilities...")
        
        for fac in facilities:
            print(f"\n[njm] Facility: {fac}")
            instruments = self.discover_instruments(fac)
            
            for inst in instruments:
                downloaded = self.download_filters(fac, inst)
                total_downloaded += downloaded
        
        print(f"\n[njm] Total filters downloaded: {total_downloaded}")
        return total_downloaded
    
    # ------------------------------------------------------------------
    # Separable components (see sed_tools.filter_components)
    # ------------------------------------------------------------------
    def fetch_manifest(self, facility: str, instrument: str) -> Optional[Dict]:
        """Return the remote ``manifest.json`` for an instrument, or ``None``."""
        if not self._available:
            return None
        try:
            response = self._get(f"{self.filters_url}/{facility}/{instrument}/manifest.json", timeout=15)
            if response.status_code != 200:
                return None
            manifest = response.json()
            from .filter_components import InstrumentManifest

            parsed = InstrumentManifest.from_dict(manifest)
            if (parsed.facility, parsed.instrument) != (facility, instrument):
                raise ValueError("NJM manifest identity does not match the requested instrument")
            return manifest
        except (requests.RequestException, ValueError, TypeError):
            logger.debug("No NJM manifest for %s/%s", facility, instrument, exc_info=True)
            return None

    def download_components(self, facility: str, instrument: str) -> int:
        """Download ``manifest.json`` and any component files it lists.

        Returns the number of component files downloaded (0 if the mirror has
        no components for this instrument).
        """
        manifest = self.fetch_manifest(facility, instrument)
        if manifest is None:
            return 0
        output_dir = Path(self.base_dir) / facility / instrument
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        rel_paths = sorted({
            rel for entries in manifest.get("components", {}).values() for rel in entries.values()
            if "/" in rel  # root-level entries are the published curves themselves
        })
        count = 0
        for rel in rel_paths:
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                print(f"  [skip] component path outside the instrument: {rel}")
                continue
            target = output_dir / rel
            if not target.resolve().is_relative_to(output_dir.resolve()):
                print(f"  [skip] component resolves outside the instrument: {rel}")
                continue
            if target.exists():
                migrate_filter_file(target)
                continue
            try:
                self._save_curve(f"{self.filters_url}/{facility}/{instrument}/{rel}", target)
                count += 1
            except requests.RequestException as exc:
                print(f"  [error] component {rel}: {exc}")
        if count:
            print(f"[njm] {facility}/{instrument}: {count} component files downloaded")
        return count

    def download_atmosphere(self, atmosphere_root: str, site: str, filename: Optional[str] = None) -> bool:
        """Fetch an atmosphere grid manifest (and optionally one grid file) for ``site``."""
        if not self._available:
            return False
        site_dir = Path(atmosphere_root) / site
        base = f"{self.base_url}/atmospheres/{site}"
        try:
            response = self.session.get(f"{base}/manifest.json", timeout=15)
            if response.status_code != 200:
                return False
            site_dir.mkdir(parents=True, exist_ok=True)
            (site_dir / "manifest.json").write_text(response.text, encoding="utf-8")
            if filename and not (site_dir / filename).exists():
                self._save_curve(f"{base}/{filename}", site_dir / filename, timeout=120)
            elif filename:
                migrate_filter_file(site_dir / filename)
            return True
        except requests.RequestException as exc:
            print(f"[njm] atmosphere {site}: {exc}")
            return False

    def list_atmosphere_sites(self) -> List[str]:
        if not self._available:
            return []
        return [s for s in self._parse_directory_listing(f"{self.base_url}/atmospheres/") if "." not in s]

    def get_filter_info(self, facility: str, instrument: str) -> Dict[str, any]:
        """Get information about available filters for an instrument."""
        if not self._available:
            return {}
        
        filters = self.discover_filters(facility, instrument)
        
        return {
            "facility": facility,
            "instrument": instrument,
            "filter_count": len(filters),
            "filters": filters,
            "url": f"{self.filters_url}/{facility}/{instrument}/",
        }


def main():
    """Standalone CLI for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download filters from NJM mirror")
    parser.add_argument("--facility", help="Specific facility to download")
    parser.add_argument("--instrument", help="Specific instrument to download (requires --facility)")
    parser.add_argument("--list", action="store_true", help="List available facilities/instruments")
    parser.add_argument("--output", default="../data/filters/", help="Output directory")
    
    args = parser.parse_args()
    
    grabber = NJMFilterGrabber(base_dir=args.output)
    
    if not grabber.is_available():
        print("NJM mirror is not available")
        return 1
    
    if args.list:
        print("\nAvailable facilities:")
        facilities = grabber.discover_facilities()
        for fac in facilities:
            print(f"\n  {fac}:")
            instruments = grabber.discover_instruments(fac)
            for inst in instruments:
                filters = grabber.discover_filters(fac, inst)
                print(f"    - {inst} ({len(filters)} filters)")
        return 0
    
    if args.instrument and not args.facility:
        print("Error: --instrument requires --facility")
        return 1
    
    if args.facility and args.instrument:
        grabber.download_filters(args.facility, args.instrument)
    elif args.facility:
        grabber.download_all_filters(facility=args.facility)
    else:
        grabber.download_all_filters()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
