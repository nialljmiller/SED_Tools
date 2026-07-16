#!/usr/bin/env python3
"""
NJM Filter Grabber - Downloads filter profiles from nillmill.ddns.net mirror
Complements the spectra grabber for complete data access
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
import urllib3

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup as _BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False


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
            except Exception:
                return False
        except Exception:
            return False
    
    def _disable_ssl_verification(self) -> None:
        """Suppress SSL warnings and switch the session to skip certificate checks."""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.session.verify = False

    def is_available(self) -> bool:
        """Return True if the mirror is available."""
        return self._available

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
        svo_norm = self._normalize_name(svo_instrument)
        for name in instruments:
            njm_norm = self._normalize_name(name)
            if njm_norm.startswith(svo_norm) or svo_norm.startswith(njm_norm):
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
        except Exception as e:
            logger.debug("[njm] Could not fetch facility list: %s", e, exc_info=True)
            facilities = self._parse_directory_listing(self.filters_url)

        self._cached_facilities = facilities
        return facilities
    
    def discover_instruments(self, facility: str) -> List[str]:
        """Discover available instruments for a facility."""
        if not self._available:
            return []
        
        try:
            facility_url = f"{self.filters_url}/{facility}/"
            return self._parse_directory_listing(facility_url)
        except Exception as e:
            print(f"[njm] Could not fetch instruments for {facility}: {e}")
            return []
    
    def discover_filters(self, facility: str, instrument: str) -> List[str]:
        """Discover available filters for a facility/instrument."""
        if not self._available:
            return []
        
        try:
            instrument_url = f"{self.filters_url}/{facility}/{instrument}/"
            
            # Try to get instrument index file first
            index_file_url = f"{instrument_url}{instrument}"
            try:
                response = self.session.get(index_file_url, timeout=10)
                if response.status_code == 200:
                    # Parse index file (one filter per line)
                    filters = [line.strip() for line in response.text.splitlines() 
                              if line.strip() and not line.startswith('#')]
                    return filters
            except Exception:
                logger.debug("Could not fetch instrument index file for %s/%s", facility, instrument, exc_info=True)

            # Fallback: parse directory listing
            all_files = self._parse_directory_listing(instrument_url)
            # Filter for .dat files and remove the extension
            filters = [f.replace('.dat', '') for f in all_files if f.endswith('.dat')]
            return filters
            
        except Exception as e:
            print(f"[njm] Could not fetch filters for {facility}/{instrument}: {e}")
            return []
    
    def _parse_directory_listing(self, url: str) -> List[str]:
        """Parse Apache directory listing HTML to extract subdirectories/files."""
        if not _HAS_BS4:
            return []
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = _BeautifulSoup(response.text, 'html.parser')
            
            items = []
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href and href not in ['..', '/', '../']:
                    # Remove trailing slash for directories
                    item = href.rstrip('/')
                    # Skip parent directory links
                    if not item.startswith('..') and not item.startswith('/'):
                        items.append(item)
            
            return items
            
        except Exception:
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
                skipped += 1
                continue
            
            try:
                response = self.session.get(filter_url, timeout=30)
                response.raise_for_status()
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                downloaded += 1
                
            except Exception as e:
                print(f"  [error] {filter_name}: {e}")
                failed += 1
                if os.path.exists(output_path):
                    os.remove(output_path)
        
        # Download instrument index file
        try:
            index_url = f"{self.filters_url}/{facility}/{instrument}/{instrument}"
            index_path = os.path.join(output_dir, instrument)
            
            response = self.session.get(index_url, timeout=10)
            if response.status_code == 200:
                with open(index_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"  [saved] {instrument} index file")
        except Exception:
            # Create index file from downloaded filters
            filter_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.dat')])
            if filter_files:
                index_path = os.path.join(output_dir, instrument)
                with open(index_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(filter_files) + '\n')
                print(f"  [created] {instrument} index file")
        
        print(f"[njm] {facility}/{instrument}: {downloaded} downloaded, {skipped} skipped, {failed} failed")
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
