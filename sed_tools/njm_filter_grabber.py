#!/usr/bin/env python3
"""
NJM Filter Grabber - Downloads filter profiles from nillmill.ddns.net mirror
Complements the spectra grabber for complete data access
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from time import monotonic
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
import urllib3
from bs4 import BeautifulSoup

from .filter_components import InstrumentManifest
from .filter_metadata import SOURCE_ASSUMED, SVOMetadataClient, migrate_filter_file, tag_filter_file

logger = logging.getLogger(__name__)

# Answers that mean "ask again later", not "no such file": NJM's Apache answers a burst of
# requests from one address with 403 until the block lifts, within about a minute.
RETRY_STATUSES = (403, 429, 500, 502, 503, 504)
# Answers that, still given after the retries, mean NJM will not serve this client for now.
BLOCKED_STATUSES = (403, 429)
#: Seconds to wait for NJM to accept a connection: a mirror that is down fails fast.
NJM_CONNECT_TIMEOUT = 5.0
#: Seconds NJM is left alone after it stops answering.
NJM_PAUSE = 300
#: Seconds a process keeps NJM's index.json before reading it again.
NJM_INDEX_MAX_AGE = 3600
FALLBACK_SOURCES = "installed copies, the original-source builders and SVO"


def plain_reason(reason) -> str:
    """Why NJM could not be reached, in a few plain words rather than a raw exception's text."""
    text = str(reason)
    if len(text) <= 60 and "(" not in text:
        return text
    lowered = text.lower()
    if isinstance(reason, requests.exceptions.Timeout) or "timed out" in lowered:
        return "it did not answer in time"
    if isinstance(reason, requests.exceptions.SSLError):
        return "its certificate or TLS handshake failed"
    for words, plain in (("refused", "connection refused"), ("reset by peer", "connection reset"),
                         ("aborted", "connection reset"), ("no route", "no route to host"),
                         ("name or service not known", "host name not found"),
                         ("nodename nor servname", "host name not found"),
                         ("name resolution", "host name not found")):
        if words in lowered:
            return plain
    return "could not connect"
#: Which installed curves a download found to be NJM's, kept under ``<filter root>/../cache/njm/filters/F/I/``.
VERIFIED_CURVES = "verified_curves.json"


class _MirrorState:
    """What this process knows about NJM, shared by every :class:`NJMFilterGrabber`.

    A new grabber (every FilterLibrary and FilterCatalogue makes one) reuses
    the last availability check instead of asking NJM again, and a mirror that
    stopped answering is left alone by all of them for ``NJM_PAUSE`` seconds,
    reported once per outage.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.available = False  # a check or a request found NJM answering
        self.unverified_ssl = False  # NJM's certificate could not be verified; requests skip the check
        self.unreachable: Optional[str] = None  # why NJM is not asked, until `until` (monotonic())
        self.until = 0.0
        self.reported = False  # the current outage has been reported
        self.index: Optional[Dict] = None  # NJM's index.json, read at `index_read` (monotonic())
        self.index_read = 0.0


_STATE = _MirrorState()


def _status(exc: Exception) -> Optional[int]:
    """The HTTP status of a failed request, or None when there was no answer."""
    return getattr(getattr(exc, "response", None), "status_code", None)


def curve_file(name: str) -> str:
    """The file of a curve named in an NJM index (``g.dat``) or listing (``g``)."""
    return name if name.endswith(".dat") else f"{name}.dat"


def _samples(text: str) -> List[str]:
    """A curve's sample rows: every line but ``#`` lines and the first other line (the column row).

    Tagging rewrites only the column row, so these identify the curve itself.
    """
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    return lines[1:]


def _same_curve(path: Path, served: str) -> bool:
    """Whether the installed curve at ``path`` has the samples of the mirror's copy ``served``."""
    try:
        return _samples(path.read_text(encoding="utf-8")) == _samples(served)
    except UnicodeDecodeError:  # not a text curve: it cannot be the mirror's
        return False


@dataclass
class SetDownload:
    """What :meth:`NJMFilterGrabber._download_set` did for one instrument."""

    listed: List[str]  # the curves NJM's index lists (empty: NJM does not serve the instrument)
    downloaded: int = 0
    verified: int = 0  # installed curves found to be NJM's
    curve_failures: List[str] = field(default_factory=list)  # curves not downloaded or not verified
    other_failures: List[str] = field(default_factory=list)  # e.g. its components


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
        self._index_data: Optional[Dict] = None
        self._index_unreadable = False
        self._metadata_client = SVOMetadataClient()
        self._available = self._mirror_available()

    def _mirror_available(self) -> bool:
        """Whether NJM answers, checked once in this process and again once a pause has expired."""
        if self._paused():
            return False
        if _STATE.available:
            if _STATE.unverified_ssl:
                self._skip_ssl_verification()
            return True
        if not self._check_availability():
            return False
        _STATE.available, _STATE.reported = True, False
        return True

    def _check_availability(self) -> bool:
        """Check if the NJM mirror is available and responding.

        No answer, or a definite refusal (e.g. 404), means the mirror is down.
        A 403 is NJM's temporary block on a burst of requests from this address
        (shared by every client behind it), and 429/5xx are transient too: those
        are confirmed by fetching ``index.json`` with the usual retries, which
        wait the block out, rather than reported as an outage.
        """
        try:
            # Try with SSL verification first
            response = self.session.head(self.base_url, timeout=(NJM_CONNECT_TIMEOUT, 10))
        except requests.exceptions.SSLError:
            # SSL certificate issue - try without verification
            try:
                response = self.session.head(self.base_url, timeout=(NJM_CONNECT_TIMEOUT, 10), verify=False)
            except requests.RequestException as exc:
                return self._unavailable(exc)
            self._disable_ssl_verification()
            logger.warning("NJM's SSL certificate could not be verified; using %s without verification",
                           self.base_url)
        except requests.RequestException as exc:
            return self._unavailable(exc)
        if response.status_code < 400:
            return True
        if response.status_code not in RETRY_STATUSES:
            return self._unavailable(f"HTTP {response.status_code}")
        logger.warning("NJM answered %s; confirming with index.json (a burst block lifts within about a minute)",
                       response.status_code)
        try:
            response = self._get(f"{self.base_url}/index.json", timeout=10)
        except requests.RequestException as exc:
            return self._unavailable(exc)
        try:
            data = response.json()
            self._index_data = data if isinstance(data, dict) else None
        except ValueError:
            pass  # the mirror answered; _index fetches the index again
        return True

    def _unavailable(self, reason) -> bool:
        reason = plain_reason(reason)
        self._pause(reason, f"NJM mirror {self.base_url} is unavailable ({reason});\n"
                            f"  it will not be used for {NJM_PAUSE // 60} minutes, and filters come from "
                            f"installed copies,\n  the original-source builders and SVO")
        return False

    @staticmethod
    def _pause(reason: str, message: str, log: logging.Logger = logger) -> None:
        """Leave NJM alone for ``NJM_PAUSE`` seconds in this whole process; ``message`` is warned once per outage."""
        with _STATE.lock:
            _STATE.available = False
            _STATE.unreachable, _STATE.until = reason, monotonic() + NJM_PAUSE
            report, _STATE.reported = not _STATE.reported, True
        if report:
            log.warning(message)

    def _disable_ssl_verification(self) -> None:
        """Switch this and every later grabber in the process to skip certificate checks."""
        _STATE.unverified_ssl = True
        self._skip_ssl_verification()

    def _skip_ssl_verification(self) -> None:
        """Suppress SSL warnings and switch the session to skip certificate checks."""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.session.verify = False

    def is_available(self) -> bool:
        """Return True if the mirror is available."""
        return self._available and not self._paused()

    @staticmethod
    def _paused() -> bool:
        """Whether NJM stopped answering, in this process, less than ``NJM_PAUSE`` seconds ago."""
        with _STATE.lock:
            if _STATE.unreachable is not None and monotonic() >= _STATE.until:
                _STATE.unreachable = None
            return _STATE.unreachable is not None

    def _get(self, url: str, *, timeout: int = 30, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        """Retry transient mirror failures, including Apache's temporary 403 block.

        A request NJM still does not answer after the retries (no connection,
        a timeout, or a block that has not lifted) marks NJM unreachable for
        ``NJM_PAUSE`` seconds in the whole process, with one warning per
        outage; requests until then fail at once.
        """
        if self._paused():
            raise requests.ConnectionError(f"NJM is unreachable ({_STATE.unreachable}); "
                                           f"not asked again for {NJM_PAUSE // 60} minutes")
        timeouts = (NJM_CONNECT_TIMEOUT, timeout)
        for attempt in range(3):
            try:
                response = (self.session.get(url, timeout=timeouts, headers=headers) if headers
                            else self.session.get(url, timeout=timeouts))
                response.raise_for_status()
                _STATE.reported = False  # NJM answers: a later outage is news again
                return response
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in RETRY_STATUSES:
                    raise
                if attempt == 2:
                    if status is None or status in BLOCKED_STATUSES:
                        self._pause(f"{status or type(exc).__name__} for {url}",
                                    f"NJM mirror {self.base_url} stopped answering ({exc}); it is not asked "
                                    f"again for {NJM_PAUSE // 60} minutes, and filters come from {FALLBACK_SOURCES}")
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
        self._write_curve(self._get(url, timeout=timeout).text, target)

    @staticmethod
    def _write_curve(text: str, target: Path) -> None:
        """Write a served curve whole (an interrupted write leaves no partial file) and migrate it."""
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.part")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
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

    def _index(self, unreadable: str = "listing the filters directory instead") -> Optional[Dict]:
        """NJM's ``index.json``, read once per process (again after an hour); None when it cannot be read.

        A failure is warned about once per grabber, followed by ``unreadable``
        (what is done without it).
        """
        if self._index_data is not None and not isinstance(self._index_data.get("filters"), dict):
            self._index_data = None  # what the availability check fetched is no filter index
        if self._index_data is None:
            with _STATE.lock:
                if _STATE.index is not None and monotonic() - _STATE.index_read < NJM_INDEX_MAX_AGE:
                    self._index_data = _STATE.index
        if self._index_data is None and not self._index_unreadable:
            try:
                data = self._get(f"{self.base_url}/index.json", timeout=10).json()
                if not isinstance(data, dict) or not isinstance(data.get("filters"), dict):
                    raise ValueError("it has no 'filters' section")
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Cannot read NJM's index.json (%s); %s", exc, unreadable)
                self._index_unreadable = True
            else:
                self._index_data = data
                with _STATE.lock:
                    _STATE.index, _STATE.index_read = data, monotonic()
        return self._index_data

    def discover_facilities(self) -> List[str]:
        """Discover available filter facilities from the mirror (cached)."""
        if not self._available:
            return []
        if self._cached_facilities is None:
            facilities = (self._index() or {"filters": {}})["filters"].get("facilities")
            facilities = [f for f in facilities if isinstance(f, str)] if isinstance(facilities, list) else []
            if not facilities:
                facilities = self._parse_directory_listing(self.filters_url, directories_only=True)
            self._cached_facilities = facilities
        return self._cached_facilities

    def discover_instruments(self, facility: str) -> List[str]:
        """Discover available instruments for a facility."""
        return list(self._instrument_records(facility))

    def _instrument_records(self, facility: str) -> Dict[str, Dict]:
        """NJM's instruments of ``facility``: name -> its ``index.json`` record.

        A record may carry ``description`` and ``svo_instruments`` (the SVO
        ``Facility/Instrument`` products it is the same as).  Only when
        ``index.json`` cannot be read are the names taken from the directory
        listing, with a warning and empty records.
        """
        if not self._available:
            return {}
        detailed = (self._index() or {"filters": {}})["filters"].get("facilities_detailed")
        if isinstance(detailed, dict):
            record = detailed.get(facility, {})
            instruments = record.get("instruments", {}) if isinstance(record, dict) else None
            if not isinstance(instruments, dict):
                logger.warning("NJM's index.json describes %s in an unexpected form; its instruments are "
                               "not listed from NJM", facility)
                return {}
            return {name: meta if isinstance(meta, dict) else {} for name, meta in instruments.items()}
        logger.warning("NJM's index.json lists no instruments; reading the directory listing of %s instead",
                       facility)
        names = self._parse_directory_listing(f"{self.filters_url}/{facility}/", directories_only=True)
        return {name: {} for name in names}

    def discover_filters(self, facility: str, instrument: str) -> List[str]:
        """Discover available filters for a facility/instrument."""
        if not self._available:
            return []
        
        instrument_url = f"{self.filters_url}/{facility}/{instrument}/"
        try:
            response = self._get(f"{instrument_url}{instrument}", timeout=10)
        except requests.RequestException as exc:
            # Without an index file (404) the directory listing is the list of curves.
            if _status(exc) != 404:
                logger.warning("Cannot read NJM's index of %s/%s (%s); listing its directory instead",
                               facility, instrument, exc)
        else:
            return [line.strip() for line in response.text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]

        return [name.removesuffix(".dat") for name in self._parse_directory_listing(instrument_url)
                if name.endswith(".dat")]

    def _parse_directory_listing(self, url: str, *, directories_only: bool = False) -> List[str]:
        """Parse Apache directory listing HTML to extract subdirectories/files.

        ``directories_only`` keeps only links with a trailing slash, so a file
        beside the instruments is never taken for one.  Empty when NJM has
        nothing at ``url`` (404); other failures raise ``requests.RequestException``.
        """
        try:
            response = self._get(url, timeout=10)
        except requests.RequestException as exc:
            if _status(exc) == 404:
                return []
            raise
        soup = BeautifulSoup(response.text, 'html.parser')

        items = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if directories_only and not href.endswith('/'):
                continue
            if href and href not in ['..', '/', '../'] and not href.startswith('?'):
                # Remove trailing slash for directories
                item = href.rstrip('/')
                # Skip parent directory links
                if not item.startswith('..') and not item.startswith('/'):
                    items.append(item)
        return items

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

        Every curve that can be downloaded is saved, with the mirror's index,
        catalogue files, components and manifest.  An installed curve is kept
        only if its samples match the mirror's copy; otherwise it is replaced.
        Then RuntimeError lists every curve (and the components, if they
        failed) that could not be downloaded or checked.
        """
        if not self._available:
            print("[njm] Mirror not available")
            return 0
        result = self._download_set(facility, instrument, filters)
        if not result.listed:
            print(f"[njm] No filters found for {facility}/{instrument}")
            return 0
        failed = result.curve_failures + result.other_failures
        if failed:
            raise RuntimeError(f"NJM download of {facility}/{instrument} is incomplete; run again to retry:\n  "
                               + "\n  ".join(failed))
        return result.downloaded

    def _download_set(self, facility: str, instrument: str, filters: Optional[List[str]] = None, *,
                      output_dir: Optional[Path] = None) -> SetDownload:
        """Download an instrument's curves and what comes with them, and report every failure.

        ``listed`` is empty when NJM does not serve the instrument.  Raises
        ``requests.RequestException`` only when NJM cannot say which curves it serves.
        ``output_dir`` is where they go when the instrument is installed under
        another spelling (``Gemini/GMOS-N`` for NJM's ``Gemini/GMOS_N``); by
        default ``<base_dir>/<facility>/<instrument>``.
        """
        if filters is None:
            filters = self.discover_filters(facility, instrument)
        if not filters:
            return SetDownload(listed=[])
        default_dir = Path(self.base_dir) / facility / instrument
        output_dir = Path(output_dir) if output_dir is not None else default_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        elsewhere = output_dir != default_dir

        print(f"[njm] Downloading filters: {facility}/{instrument}")
        print(f"  Found {len(filters)} filters")
        downloaded, skipped, failed = self._download_curves(facility, instrument, filters, output_dir)
        result = SetDownload(listed=list(filters), downloaded=downloaded, verified=skipped, curve_failures=failed)
        if self._paused():
            result.other_failures.append("its index, catalogue files and components: NJM stopped answering")
            return result

        remote_index = self._save_index(facility, instrument, output_dir)
        print(f"[njm] {facility}/{instrument}: {downloaded} downloaded, {skipped} already installed, "
              f"{len(failed)} failed")
        if downloaded:
            # Only beside curves fetched now: the mirror's files describe its current curves.
            if elsewhere:
                self._save_catalogue_files(facility, instrument, output_dir)
            else:
                self.download_catalogue_files(facility, instrument)
        try:
            if elsewhere:
                self._save_components(facility, instrument, output_dir)
            else:
                self.download_components(facility, instrument)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            result.other_failures.append(f"components: {exc}")
        if remote_index and not (output_dir / "manifest.json").is_file():
            manifest = InstrumentManifest.from_published(output_dir)
            manifest.bands = sorted(name.removesuffix(".dat") for name in remote_index if name.endswith(".dat"))
            manifest.save(output_dir)
        return result

    def _download_curves(self, facility: str, instrument: str, filters: List[str],
                         output_dir: Path) -> Tuple[int, int, List[str]]:
        """Save each listed curve; returns (downloaded, already installed, failures).

        An installed curve is checked against the mirror's copy: it is kept if
        its samples match (its header may hold tags added here), and replaced
        otherwise, e.g. by SVO's curve of the same name written by an earlier release.
        It is not fetched again when an earlier download found it to be NJM's
        and neither it nor NJM's record of the instrument has changed since
        (:meth:`_verified_curves`).
        """
        downloaded, skipped, failed, njms = 0, 0, [], []
        installed = any((output_dir / curve_file(name)).is_file() for name in filters)
        trusted = self._verified_curves(facility, instrument, output_dir) if installed else set()
        for number, filter_name in enumerate(filters):
            output_path = output_dir / curve_file(filter_name)
            if output_path.name in trusted:
                skipped += 1
                njms.append(output_path.name)
                continue
            try:
                served = self._get(f"{self.filters_url}/{facility}/{instrument}/{output_path.name}", timeout=30).text
            except requests.RequestException as e:  # raised before anything is written
                if self._paused():  # one line for the rest, not one per curve
                    failed.append(f"{len(filters) - number} curves from {filter_name} on: {e}")
                    print(f"  [error] {failed[-1]}")
                    break
                print(f"  [error] {filter_name}: {e}")
                failed.append(f"{filter_name}: {e}")
                continue
            if output_path.exists() and _same_curve(output_path, served):
                self._tag(output_path, facility)
                skipped += 1
                njms.append(output_path.name)
                continue
            if output_path.exists():
                logger.warning("%s differs from NJM's copy; replacing it with NJM's", output_path)
            self._write_curve(served, output_path)
            result = self._tag(output_path, facility)
            if result is not None and result.source == SOURCE_ASSUMED:
                print(f"  [warn] {filter_name}: detector type unverified, assuming {result.detector_type}"
                      + (f" ({result.note})" if result.note else ""))
            downloaded += 1
            njms.append(output_path.name)
        if njms:
            self._record_verified(facility, instrument, output_dir, njms)
        return downloaded, skipped, failed

    # ------------------------------------------------------------------
    # Installed curves already found to be NJM's
    # ------------------------------------------------------------------
    def _verified_path(self, facility: str, instrument: str) -> Path:
        from .filter_catalogue import cache_root

        return cache_root(self.base_dir) / "filters" / facility / instrument / VERIFIED_CURVES

    def _njm_record(self, facility: str, instrument: str) -> Optional[str]:
        """NJM's ``index.json`` record of the instrument's curves (names and size), or None if it has none.

        It changes whenever NJM's set does, so installed curves verified under
        it need not be fetched again while it stays the same.  While NJM is
        paused nothing is fetched for it: the curves are then checked next time.
        """
        if self._index_data is None and self._paused():
            return None
        index = self._index(f"the installed curves of {facility}/{instrument} are checked against NJM's copies")
        detailed = (index or {"filters": {}})["filters"].get("facilities_detailed")
        facility_record = detailed.get(facility) if isinstance(detailed, dict) else None
        instruments = facility_record.get("instruments") if isinstance(facility_record, dict) else None
        record = instruments.get(instrument) if isinstance(instruments, dict) else None
        if not isinstance(record, dict) or "size_bytes" not in record:
            return None
        return json.dumps({"filters": record.get("filters"), "size_bytes": record["size_bytes"]}, sort_keys=True)

    def _verified_curves(self, facility: str, instrument: str, output_dir: Path) -> Set[str]:
        """Installed curve files an earlier download found to be NJM's, unchanged since on both sides."""
        record = self._njm_record(facility, instrument)
        path = self._verified_path(facility, instrument)
        if record is None or not path.is_file():
            return set()
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("%s is unreadable (%s); every installed curve of %s/%s is checked against NJM's copy",
                           path, exc, facility, instrument)
            return set()
        files = saved.get("files") if isinstance(saved, dict) and saved.get("njm_record") == record else None
        if not isinstance(files, dict):
            return set()
        return {name for name, size in files.items()
                if (output_dir / name).is_file() and (output_dir / name).stat().st_size == size}

    def _record_verified(self, facility: str, instrument: str, output_dir: Path, names: List[str]) -> None:
        """Note that these installed curve files are NJM's, under NJM's current record of the instrument."""
        record = self._njm_record(facility, instrument)
        if record is None:
            return
        path = self._verified_path(facility, instrument)
        files = {name: (output_dir / name).stat().st_size for name in names}
        if path.is_file():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                saved = {}  # rewritten below
            if isinstance(saved, dict) and saved.get("njm_record") == record and isinstance(saved.get("files"), dict):
                files = {**saved["files"], **files}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.part")
        tmp.write_text(json.dumps({"njm_record": record, "files": files}, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)

    def _tag(self, path: Path, facility: str):
        """Fill in what the curve's header lacks from SVO; a no-op for curves the mirror tagged fully.

        Returns the :class:`~sed_tools.filter_metadata.DetectorTypeResult`, or
        None when SVO could not be asked; the mirror's header is then kept as
        served, with a warning.
        """
        try:
            return tag_filter_file(path, client=self._metadata_client, facility=facility)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("%s: could not check its header against SVO (%s); kept the mirror's header", path, exc)
            return None

    def _save_index(self, facility: str, instrument: str, output_dir: Path) -> Optional[List[str]]:
        """Save the mirror's index of the instrument and return its lines; None when it cannot be read.

        Only the mirror (or its manifest) can declare the full filter set: an
        index built from the files seen so far would certify a partial set.
        The file is named after ``output_dir``, as MESA looks for it there.
        """
        try:
            response = self._get(f"{self.filters_url}/{facility}/{instrument}/{instrument}", timeout=10)
        except requests.RequestException as exc:
            logger.warning("NJM index unavailable for %s/%s (%s); completeness cannot be verified",
                           facility, instrument, exc)
            return None
        (output_dir / output_dir.name).write_text(response.text, encoding="utf-8")
        return [line.strip() for line in response.text.splitlines() if line.strip()]

    def download_all_filters(self, facility: Optional[str] = None) -> int:
        """
        Download all available filters, optionally limited to a facility.

        Args:
            facility: Optional facility name. If None, downloads from all facilities.

        Returns:
            Total number of filters downloaded

        Every instrument is attempted; then RuntimeError lists those whose
        download was incomplete.
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

        failed = []
        for fac in facilities:
            print(f"\n[njm] Facility: {fac}")
            try:
                instruments = self.discover_instruments(fac)
            except requests.RequestException as exc:
                failed.append(f"{fac}: {exc}")
                continue

            for inst in instruments:
                try:
                    total_downloaded += self.download_filters(fac, inst)
                except (requests.RequestException, RuntimeError) as exc:
                    failed.append(f"{fac}/{inst}: {exc}")

        print(f"\n[njm] Total filters downloaded: {total_downloaded}")
        if failed:
            raise RuntimeError(f"{len(failed)} NJM downloads are incomplete:\n" + "\n".join(failed))
        return total_downloaded

    # ------------------------------------------------------------------
    # Catalogue files (see sed_tools.filter_catalogue)
    # ------------------------------------------------------------------
    CATALOGUE_FILES = ("properties.json", "info.json")

    def download_catalogue_files(self, facility: str, instrument: str) -> List[str]:
        """Save the instrument's ``properties.json`` and ``info.json`` beside its curves.

        Best effort: a file the mirror does not serve is skipped, and one that
        cannot be fetched or read is skipped with a warning (catalogue lookups
        then use NJM's current copy).  Returns the names saved.
        """
        return self._save_catalogue_files(facility, instrument, Path(self.base_dir) / facility / instrument)

    def _save_catalogue_files(self, facility: str, instrument: str, output_dir: Path) -> List[str]:
        """:meth:`download_catalogue_files` into ``output_dir`` (see :meth:`_download_set`)."""
        if not self._available:
            return []
        saved = []
        for name in self.CATALOGUE_FILES:
            try:
                content = self._catalogue_file(facility, instrument, name)
            except (requests.RequestException, ValueError) as exc:
                if _status(exc) != 404:
                    logger.warning("NJM's %s for %s/%s was not saved beside its curves (%s); "
                                   "catalogue lookups will use NJM's current copy", name, facility, instrument, exc)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            tmp = output_dir / f".{name}.part"
            tmp.write_bytes(content)
            os.replace(tmp, output_dir / name)
            saved.append(name)
        return saved

    def _catalogue_file(self, facility: str, instrument: str, name: str) -> bytes:
        """The served bytes of one catalogue file; ValueError if it is not an NJM catalogue file."""
        response = self._get(f"{self.filters_url}/{facility}/{instrument}/{name}", timeout=30)
        content = getattr(response, "content", None)
        if not isinstance(content, bytes):
            content = response.text.encode("utf-8")
        data = json.loads(content.decode("utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("bands"), dict):
            raise ValueError(f"not an NJM {name}")
        return content

    # ------------------------------------------------------------------
    # Separable components (see sed_tools.filter_components)
    # ------------------------------------------------------------------
    def fetch_manifest(self, facility: str, instrument: str) -> Optional[Dict]:
        """Return the remote ``manifest.json`` for an instrument, or ``None`` when NJM serves none.

        Raises ``requests.RequestException`` when NJM cannot be asked, and
        ValueError when the manifest it serves cannot be read or describes
        another instrument.
        """
        if not self._available:
            return None
        try:
            response = self._get(f"{self.filters_url}/{facility}/{instrument}/manifest.json", timeout=15)
        except requests.RequestException as exc:
            if _status(exc) == 404:
                return None
            raise
        if response.status_code != 200:
            return None
        try:
            manifest = response.json()
            parsed = InstrumentManifest.from_dict(manifest)
        except (ValueError, TypeError, AttributeError) as exc:  # not JSON, not an object, or fields missing
            raise ValueError(f"NJM serves an unreadable manifest.json for {facility}/{instrument}: {exc}") from exc
        if (parsed.facility, parsed.instrument) != (facility, instrument):
            raise ValueError(f"NJM's manifest.json for {facility}/{instrument} describes "
                             f"{parsed.facility}/{parsed.instrument}")
        return manifest

    def download_components(self, facility: str, instrument: str) -> int:
        """Download ``manifest.json`` and any component files it lists.

        Returns the number of component files downloaded (0 if the mirror has
        no components for this instrument).  Every file that can be downloaded
        is saved; then RuntimeError lists those that could not be.
        """
        return self._save_components(facility, instrument, Path(self.base_dir) / facility / instrument)

    def _save_components(self, facility: str, instrument: str, output_dir: Path) -> int:
        """:meth:`download_components` into ``output_dir`` (see :meth:`_download_set`)."""
        manifest = self.fetch_manifest(facility, instrument)
        if manifest is None:
            return 0
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        listed = [rel for entries in manifest.get("components", {}).values() for rel in entries.values()]
        # Alternative curves (e.g. QE per detector epoch) are fetched too, so
        # they can be selected offline.
        listed += [rel for named in (manifest.get("variants") or {}).values()
                   for entry in named.values() for rel in (entry.get("files") or {}).values()]
        rel_paths = sorted({
            rel for rel in listed
            if isinstance(rel, str) and "/" in rel  # root-level entries are the published curves themselves
        })
        count = 0
        failed = []
        for rel in rel_paths:
            # fetch_manifest refused absolute and ".." paths; a symlink could still lead outside.
            target = output_dir / rel
            if not target.resolve().is_relative_to(output_dir.resolve()):
                failed.append(f"{rel}: resolves outside the instrument directory")
                continue
            if target.exists():
                migrate_filter_file(target)
                continue
            try:
                self._save_curve(f"{self.filters_url}/{facility}/{instrument}/{rel}", target)
                count += 1
            except requests.RequestException as exc:
                print(f"  [error] component {rel}: {exc}")
                failed.append(f"{rel}: {exc}")
                if self._paused():  # the rest would fail the same way
                    break
        if count:
            print(f"[njm] {facility}/{instrument}: {count} component files downloaded")
        if failed:
            raise RuntimeError(f"{len(failed)} component files of {facility}/{instrument} were not downloaded:\n  "
                               + "\n  ".join(failed))
        return count

    def download_atmosphere(self, atmosphere_root: str, site: str, filename: Optional[str] = None) -> bool:
        """Fetch an atmosphere grid manifest (and optionally one grid file) for ``site``.

        False when NJM has no atmosphere for ``site``; raises
        ``requests.RequestException`` when NJM cannot provide what it lists.
        An installed manifest is brought up to NJM's, keeping what only it
        has (see :func:`_update_atmosphere_manifest`); grid files already
        installed are never replaced.
        """
        if not self._available:
            return False
        site_dir = Path(atmosphere_root) / site
        base = f"{self.base_url}/atmospheres/{site}"
        try:
            response = self._get(f"{base}/manifest.json", timeout=15)
        except requests.RequestException as exc:
            if _status(exc) == 404:
                return False
            raise
        site_dir.mkdir(parents=True, exist_ok=True)
        _update_atmosphere_manifest(site_dir, response.text)
        if filename and not (site_dir / filename).exists():
            self._save_curve(f"{base}/{filename}", site_dir / filename, timeout=120)
        elif filename:
            migrate_filter_file(site_dir / filename)
        return True

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


def _update_atmosphere_manifest(site_dir: Path, served: str) -> None:
    """Save NJM's manifest of an atmosphere site (``served``), keeping what only the installed one has.

    Rows (``entries``, ``unstated``) for installed files NJM does not list
    stay: points built here with SkyCalc, say.  A row NJM describes otherwise
    stays as installed when its grid file is installed, since it describes
    that file.
    Every other field takes NJM's value.  A change is warned about, naming
    the fields; nothing is written when the manifest is already NJM's.
    """
    path = site_dir / "manifest.json"
    if not path.is_file():
        _write_atomic(path, served)
        return
    installed, latest = json.loads(path.read_text(encoding="utf-8")), json.loads(served)
    merged, kept = dict(latest), []
    for key in ("entries", "unstated"):
        rows = [dict(row) for row in latest.get(key) or []]
        files = [row.get("file") for row in rows]
        for row in installed.get(key) or []:
            name = row.get("file")
            installed_file = (site_dir / str(name)).is_file()
            if name not in files:
                if installed_file:
                    rows.append(row)  # a point of the installed grid that NJM does not have
            elif rows[files.index(name)] != row and installed_file:
                rows[files.index(name)] = row
                kept.append(str(name))
        if rows or key in latest:
            merged[key] = rows
    if merged == installed:
        return
    changed = sorted(key for key in set(merged) | set(installed) if merged.get(key) != installed.get(key))
    _write_atomic(path, served if merged == latest else json.dumps(merged, indent=2, sort_keys=True) + "\n")
    logger.warning("Updated the %s atmosphere's manifest to NJM's (%s)%s", site_dir.name, ", ".join(changed),
                   f"; kept the installed description of {', '.join(kept)}, which NJM describes otherwise "
                   "(remove those files to fetch NJM's)" if kept else "")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


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
