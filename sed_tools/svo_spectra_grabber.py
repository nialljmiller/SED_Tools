#!/usr/bin/env python3
# svo_spectra_grabber.py — robust SVO discovery & downloader

import csv
import io
import json
import logging
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .header_parser import parse_header

logger = logging.getLogger(__name__)

#: Connect and read timeouts for a source's grid index.  Discovery asks every source before
#: anything is downloaded, so a source that is down or hanging must cost seconds, not minutes:
#: the index is asked once, without the retries a download gets (see ``mount_discovery``).
DISCOVERY_TIMEOUT = (5, 10)


def mount_discovery(session, url):
    """Make ``session`` ask ``url`` (a grid index) once: no retries, unlike the rest of the source."""
    session.mount(url, HTTPAdapter(max_retries=0))


def discovery_failure(exc):
    """Why a grid index could not be read, in a few plain words."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return f"answered HTTP {status}"
    if isinstance(exc, requests.Timeout):
        return f"did not answer within {DISCOVERY_TIMEOUT[0]:g}+{DISCOVERY_TIMEOUT[1]:g} s"
    if isinstance(exc, requests.ConnectionError):
        return "could not be reached"
    return str(exc)


class SVOSpectraGrabber:
    """
    Robust SVO discovery & downloader.

    Discovery strategy (in order):
      1) SSAP VOTable stream  → regex over raw bytes → FIDs
      2) Model index page scrape for 'fid='
      3) Bounded sparse HEAD probe + local expansion

    Download:
      - ASCII spectra to <base>/<model>/
      - Validates content is actual spectral data (not an HTML error page)
      - Retries transient failures (connection errors, 429 and 5xx) with exponential backoff
      - Prints progress every 50 files
      - Builds lookup_table.csv from parsed headers
    """

    def __init__(self, base_dir="data/stellar_models/", max_workers=8):
        self.base_dir = base_dir
        self.max_workers = max_workers
        os.makedirs(base_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SED_Tools/1.0 (SVO)"})

        # Retry adapter: 5 retries, exponential backoff, retry on common server errors
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Endpoints
        self.model_index_url = "http://svo2.cab.inta-csic.es/theory/newov2/index.php"
        self.spectra_base_url = "http://svo2.cab.inta-csic.es/theory/newov2/ssap.php"
        mount_discovery(self.session, self.model_index_url)  # also each grid's page there, a discovery fallback

        # Budgets
        self.ssap_timeout    = 60          # seconds — SVO can be slow to start responding
        self.ssap_max_bytes  = 52_428_800  # 50 MB — enough for the largest spectral grids
        self.head_timeout    = 10          # seconds per HEAD
        self.dl_timeout      = 90          # seconds per spectrum download
        self.bruteforce_budget = 350
        self.expand_window   = 12
        self.expand_max_gap  = 20
        self.progress_every  = 50          # print progress every N completions

        # Cache
        self.cache_root = os.path.join(self.base_dir, ".cache", "svo_fids")
        os.makedirs(self.cache_root, exist_ok=True)

    # -------------------- model list --------------------

    def discover_models(self):
        """Return the spectral grids SVO lists; [] after a warning if the index cannot be read."""
        print("Discovering spectral grids from SVO...")
        try:
            r = self.session.get(self.model_index_url, timeout=DISCOVERY_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning("SVO: spectral grid index %s %s; SVO spectral grids are not offered",
                           self.model_index_url, discovery_failure(e))
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        models = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "models=" in href:
                val = href.split("models=", 1)[1].split("&", 1)[0]
                if val:
                    models.add(val)
        if not models:
            for m in re.findall(r"models=([A-Za-z0-9._\-]+)", r.text):
                models.add(m)
        out = sorted(models)
        if not out:
            logger.warning("SVO: no spectral grids found on the index page %s; SVO spectral grids are not offered",
                           self.model_index_url)
        return out

    # -------------------- spectra discovery --------------------

    def get_model_metadata(self, model_name):
        """Return [{'fid': n}, ...] for every spectrum of the grid on SVO.

        Discovery tries the SSAP query, then the model index page, then a
        bounded probe, warning at each fallback. Raises RuntimeError if none
        finds a spectrum, if the SSAP response reaches its size cap, or if a
        probe request fails. Only a complete FID list is cached.
        """
        print(f"  Fetching metadata for {model_name}...")

        cache_path = os.path.join(self.cache_root, f"{model_name}.json")
        cached = self._read_fid_cache(cache_path)
        if cached:
            print(f"    (cache) {len(cached)} spectra")
            return [{"fid": fid} for fid in cached]

        fids = self._discover_fids(model_name)
        out = [{"fid": int(fid)} for fid in sorted(fids)]
        print(f"    Found {len(out)} spectra for {model_name}")

        try:
            with open(cache_path, "w") as f:
                json.dump([int(d["fid"]) for d in out], f)
        except OSError as e:
            logger.warning("SVO: could not cache the FID list at %s (%s); it will be discovered again next time",
                           cache_path, e)

        return out

    def _read_fid_cache(self, cache_path):
        """FIDs saved by an earlier discovery, or None (warned about if the cache is unreadable)."""
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r") as f:
                return sorted(set(int(fid) for fid in json.load(f)))
        except (OSError, ValueError, TypeError) as e:
            logger.warning("SVO: ignoring unreadable FID cache %s (%s); discovering the spectra again", cache_path, e)
            return None

    def _discover_fids(self, model_name):
        """FIDs from the SSAP query, else the index page, else a bounded probe; one warning per fallback."""
        fids, ssap_problem = self._discover_ssap_stream(model_name)
        if fids:
            return fids

        fids, index_problem = self._discover_index_page(model_name)
        if fids:
            logger.warning("SVO: SSAP query for %s found no spectra (%s); using the model index page",
                           model_name, ssap_problem)
            return fids
        logger.warning("SVO: neither the SSAP query (%s) nor the model index page (%s) lists the spectra "
                       "of %s; probing FIDs directly", ssap_problem, index_problem, model_name)

        fids = self._bounded_probe(model_name)
        if fids:
            return fids
        raise RuntimeError(f"SVO has no spectra for {model_name}: SSAP query: {ssap_problem}; "
                           f"index page: {index_problem}; probing found none")

    # ---- A) SSAP stream ----

    def _discover_ssap_stream(self, model_name):
        """
        Stream the SSAP VOTable and extract FIDs via regex.

        SVO's VOTable XML contains unclosed CDATA sections which cause astropy
        and any standard XML parser to fail. Regex over raw bytes is the only
        reliable approach and is actually faster for this use case.

        Returns (fids, problem): problem says why fids is empty.
        """
        params = {"model": model_name, "REQUEST": "queryData", "FORMAT": "votable"}
        try:
            with self.session.get(
                self.spectra_base_url, params=params,
                timeout=self.ssap_timeout, stream=True
            ) as r:
                if r.status_code != 200:
                    return set(), f"HTTP {r.status_code}"
                raw = self._read_capped(r, model_name)
        except requests.RequestException as e:
            return set(), str(e)

        text = raw.decode("utf-8", "ignore")
        fids = set(int(x) for x in re.findall(r"[?&]fid=(\d+)", text))
        if not fids:
            return fids, "the response lists none"
        return fids, None

    def _read_capped(self, r, model_name):
        """The SSAP response body; RuntimeError if it reaches ssap_max_bytes (the FID list would be cut short)."""
        buf = io.BytesIO()
        total = 0
        for chunk in r.iter_content(chunk_size=131072):
            if not chunk:
                break
            buf.write(chunk)
            total += len(chunk)
            if total >= self.ssap_max_bytes:
                raise RuntimeError(
                    f"SVO: the SSAP response for {model_name} reached the "
                    f"{self.ssap_max_bytes // 1_048_576} MB cap, so its spectrum list would be "
                    f"incomplete; raise the grabber's ssap_max_bytes and retry")
        return buf.getvalue()

    # ---- B) Index page scrape ----

    def _discover_index_page(self, model_name):
        """FIDs linked from the grid's index page, as (fids, problem)."""
        url = f"{self.model_index_url}?models={urllib.parse.quote(model_name)}"
        try:
            r = self.session.get(url, timeout=20)
        except requests.RequestException as e:
            return set(), str(e)
        if r.status_code != 200:
            return set(), f"HTTP {r.status_code}"
        fids = set(int(x) for x in re.findall(r"[?&]fid=(\d+)", r.text))
        if not fids:
            return fids, "the page links none"
        return fids, None

    # ---- C) Bounded sparse probe + expansion ----

    def _bounded_probe(self, model_name):
        print("    Probing sparsely (bounded)...")
        probes = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            12, 15, 20, 25, 30, 40, 50, 60, 75, 100,
            125, 150, 200, 250, 300, 400, 500, 650, 800, 1000,
            1200, 1500, 2000, 2500, 3000, 4000, 5000, 6500, 8000, 10000,
        ]
        hits = []
        budget = min(self.bruteforce_budget, len(probes))
        with ThreadPoolExecutor(max_workers=min(self.max_workers, 16)) as ex:
            futs = {
                ex.submit(self._head_exists, model_name, fid): fid
                for fid in probes[:budget]
            }
            for fut in as_completed(futs):
                if fut.result():
                    hits.append(futs[fut])
        if not hits:
            return set()
        found = set(hits)
        for seed in sorted(hits):
            found |= self._expand_around(model_name, seed)
        return found

    def _head_exists(self, model_name, fid):
        """True if SVO serves a spectrum for this FID; RuntimeError if the request fails."""
        params = {"model": model_name, "fid": int(fid), "format": "ascii"}
        try:
            r = self.session.head(
                self.spectra_base_url, params=params,
                timeout=self.head_timeout, allow_redirects=True
            )
            if r.status_code != 200:
                return False
            cl = r.headers.get("content-length")
            if cl is None or cl == "0":
                # HEAD gave no content-length info — do a small GET to verify
                g = self.session.get(
                    self.spectra_base_url, params=params,
                    timeout=self.head_timeout, stream=True
                )
                if g.status_code != 200:
                    return False
                for chunk in g.iter_content(chunk_size=512):
                    if chunk:
                        return _is_spectral_content(chunk[:512])
                return False
            return int(cl) > 1024
        except requests.RequestException as e:
            # Treating this FID as absent would silently cut the FID list short
            raise RuntimeError(f"SVO: could not check FID {fid} of {model_name}: {e}") from e

    def _expand_around(self, model_name, seed):
        found = {seed}
        for direction in (-1, +1):
            misses = 0
            step = 1
            while misses < self.expand_max_gap:
                fid = seed + direction * (self.expand_window + step - 1)
                step += 1
                if fid <= 0:
                    misses += 1
                    continue
                if self._head_exists(model_name, fid):
                    found.add(fid)
                    misses = 0
                else:
                    misses += 1
        return found

    # -------------------- download + lookup --------------------

    def download_model_spectra(self, model_name, spectra_info,
                               teff_range=None, logg_range=None,
                               meta_range=None, wl_range=None):
        """Download the spectra listed by get_model_metadata and write lookup_table.csv.

        Returns the number of spectra on disk. SVO serves whole grids, so a
        Teff, logg, [M/H] or wavelength range raises ValueError. If any
        spectrum fails, the rest are still downloaded, lookup_table.csv is
        not written, and RuntimeError lists each failure.
        """
        if any(rng is not None for rng in (teff_range, logg_range, meta_range, wl_range)):
            raise ValueError(f"SVO downloads of {model_name} cannot be cut by Teff, logg, [M/H] or "
                             "wavelength; leave the ranges unset and filter the downloaded grid")

        out_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        if not spectra_info:
            print(f"  No spectra to download for {model_name}.")
            return 0

        total = len(spectra_info)
        print(f"Downloading {total} spectra for {model_name}...")

        rows = []
        to_download = []
        for spec in spectra_info:
            fid = int(spec["fid"])
            fname = f"{model_name}_fid{fid}.txt"
            fpath = os.path.join(out_dir, fname)

            if os.path.exists(fpath) and os.path.getsize(fpath) > 1024:
                # Validate cached file is actually spectral data
                if _file_is_spectral(fpath):
                    rows.append(_lookup_row(fpath, fname))
                    continue
                # Corrupted/HTML cached file — re-download
                os.remove(fpath)
            to_download.append((fid, fname, fpath))

        failures = []
        done = len(rows)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._download_spectrum, model_name, fid, fpath): (fid, fname, fpath)
                       for fid, fname, fpath in to_download}
            for fut in as_completed(futures):
                fid, fname, fpath = futures[fut]
                try:
                    fut.result()
                except (requests.RequestException, OSError, ValueError) as e:
                    failures.append(f"FID {fid}: {e}")
                    if os.path.exists(fpath):
                        os.remove(fpath)  # a stale cached file too small to trust
                else:
                    rows.append(_lookup_row(fpath, fname))
                done += 1
                if done % self.progress_every == 0 or done == total:
                    print(f"    {done}/{total} processed, {len(rows)} ok")

        if failures:
            raise RuntimeError(
                f"SVO: {len(failures)} of {total} spectra of {model_name} failed to download, so "
                f"lookup_table.csv was not written; the others are in {out_dir}. Rerun to retry:\n  "
                + "\n  ".join(sorted(failures)))

        self._write_lookup(out_dir, rows)
        print(f"  Successfully downloaded {len(rows)}/{total} spectra")
        return len(rows)

    def _download_spectrum(self, model_name, fid, out_path):
        """Download one spectrum to out_path; the session's adapter retries transient failures.

        Raises the request's error, or ValueError if what SVO sent is not a spectrum.
        """
        params = {"model": model_name, "fid": int(fid), "format": "ascii"}
        tmp_path = out_path + ".tmp"
        try:
            with self.session.get(self.spectra_base_url, params=params,
                                  timeout=self.dl_timeout, stream=True) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            if os.path.getsize(tmp_path) < 512:
                raise ValueError("file too small")
            if not _file_is_spectral(tmp_path):
                raise ValueError("content is not spectral data (HTML error?)")
            os.replace(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


    def _write_lookup(self, out_dir, rows):
        path = os.path.join(out_dir, "lookup_table.csv")
        keys = set()
        for r in rows:
            keys.update(r.keys())
        header = ["file_name"] + sorted(k for k in keys if k != "file_name")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("#" + ",".join(header) + "\n")
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writerows(rows)
        print(f"    Lookup table saved: {path}")


# -------------------- module-level helpers --------------------

def _lookup_row(fpath: str, fname: str) -> dict:
    """The spectrum's header parameters plus its file name, for lookup_table.csv."""
    meta = parse_header(fpath)
    meta["file_name"] = fname
    return meta


def _is_spectral_content(data: bytes) -> bool:
    """
    Return True if the first bytes look like SVO ASCII spectrum output.
    SVO spectra start with comment lines (#) or a numeric wavelength value.
    HTML error pages start with '<', 'E', etc.
    """
    if not data:
        return False
    # Strip leading whitespace/BOM
    stripped = data.lstrip(b"\xef\xbb\xbf \t\r\n")
    if not stripped:
        return False
    first = stripped[0:1]
    # Valid: starts with '#' (comment) or a digit / sign (wavelength)
    if first in (b"#", b"-", b"+") or first.isdigit():
        return True
    return False


def _file_is_spectral(path: str) -> bool:
    """Read the first 512 bytes of a file and check if it's spectral data."""
    with open(path, "rb") as f:
        return _is_spectral_content(f.read(512))
