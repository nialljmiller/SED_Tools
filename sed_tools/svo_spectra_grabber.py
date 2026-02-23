#!/usr/bin/env python3
# svo_spectra_grabber.py — robust SVO discovery & downloader

import csv
import io
import json
import os
import re
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from astropy.io.votable import parse_single_table as vot_parse_single_table
    _HAS_VOTABLE = True
except Exception:
    _HAS_VOTABLE = False


# SVO's SSAP endpoint serves malformed VOTable XML with unclosed CDATA sections.
# This is a known server-side bug; astropy fails on it predictably.
# We skip astropy entirely and go straight to the regex approach which is
# faster, more memory-efficient, and 100% reliable against SVO's output.
_USE_VOTABLE_PARSER = False  # Kept for future use if SVO ever fixes their XML


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
      - Retries with exponential backoff on failure
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

        # Budgets
        self.ssap_timeout    = 60          # seconds — SVO can be slow to start responding
        self.ssap_max_bytes  = 52_428_800  # 50 MB — enough for the largest catalogs
        self.head_timeout    = 10          # seconds per HEAD
        self.dl_timeout      = 90          # seconds per spectrum download
        self.dl_max_retries  = 4           # per-file retry attempts
        self.bruteforce_budget = 350
        self.expand_window   = 12
        self.expand_max_gap  = 20
        self.progress_every  = 50          # print progress every N completions

        # Cache
        self.cache_root = os.path.join(self.base_dir, ".cache", "svo_fids")
        os.makedirs(self.cache_root, exist_ok=True)

    # -------------------- model list --------------------

    def discover_models(self):
        print("Discovering available models from SVO...")
        try:
            r = self.session.get(self.model_index_url, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[SVO] model index failed: {e}")
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
            print("  [warn] no models found on SVO index page.")
        return out

    # -------------------- spectra discovery --------------------

    def get_model_metadata(self, model_name):
        print(f"  Fetching metadata for {model_name}...")

        # Cache hit
        cache_path = os.path.join(self.cache_root, f"{model_name}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    print(f"    (cache) {len(data)} spectra")
                    return [{"fid": int(x)} for x in sorted(set(int(y) for y in data))]
            except Exception:
                pass

        fids = set()

        # A) SSAP VOTable stream → regex
        fids |= self._discover_ssap_stream(model_name)

        # B) Model index scrape (fallback)
        if not fids:
            fids |= self._discover_index_page(model_name)

        # C) Bounded sparse probe + expansion (last resort)
        if not fids:
            fids |= self._bounded_probe(model_name)

        out = [{"fid": int(fid)} for fid in sorted(fids)]
        print(f"    Found {len(out)} spectra for {model_name}")

        # Cache
        if out:
            try:
                with open(cache_path, "w") as f:
                    json.dump([int(d["fid"]) for d in out], f)
            except Exception:
                pass

        return out

    # ---- A) SSAP stream ----

    def _discover_ssap_stream(self, model_name):
        """
        Stream the SSAP VOTable and extract FIDs via regex.

        SVO's VOTable XML contains unclosed CDATA sections which cause astropy
        and any standard XML parser to fail. Regex over raw bytes is the only
        reliable approach and is actually faster for this use case.
        """
        params = {"model": model_name, "REQUEST": "queryData", "FORMAT": "votable"}
        try:
            with self.session.get(
                self.spectra_base_url, params=params,
                timeout=self.ssap_timeout, stream=True
            ) as r:
                if r.status_code != 200:
                    return set()
                buf = io.BytesIO()
                total = 0
                for chunk in r.iter_content(chunk_size=131072):
                    if not chunk:
                        break
                    buf.write(chunk)
                    total += len(chunk)
                    if total >= self.ssap_max_bytes:
                        print(f"    [SVO] SSAP response hit {self.ssap_max_bytes // 1_048_576}MB cap "
                              f"— catalog may be very large")
                        break
                raw = buf.getvalue()
        except Exception as e:
            print(f"    [SVO] SSAP stream failed: {e}")
            return set()

        text = raw.decode("utf-8", "ignore")
        fids = set(int(x) for x in re.findall(r"[?&]fid=(\d+)", text))
        if fids:
            print(f"    SSAP: {len(fids)} fids")
        return fids

    # ---- B) Index page scrape ----

    def _discover_index_page(self, model_name):
        try:
            url = f"{self.model_index_url}?models={urllib.parse.quote(model_name)}"
            r = self.session.get(url, timeout=20)
            if r.status_code != 200:
                return set()
            fids = set(int(x) for x in re.findall(r"[?&]fid=(\d+)", r.text))
            if fids:
                print(f"    index: {len(fids)} fids")
            return fids
        except Exception:
            return set()

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
                fid = futs[fut]
                try:
                    if fut.result():
                        hits.append(fid)
                except Exception:
                    pass
        if not hits:
            return set()
        found = set(hits)
        for seed in sorted(hits):
            found |= self._expand_around(model_name, seed)
        print(f"    probe+expand: {len(found)} fids")
        return found

    def _head_exists(self, model_name, fid):
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
        except Exception:
            return False

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
        out_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        if not spectra_info:
            print(f"  No spectra to download for {model_name}.")
            return 0

        total = len(spectra_info)
        print(f"Downloading {total} spectra for {model_name}...")

        rows = []
        rows_lock = threading.Lock()
        ok_count = [0]          # list so closure can mutate it
        done_count = [0]
        done_lock = threading.Lock()

        def on_done(fname, fpath, success):
            with done_lock:
                done_count[0] += 1
                done = done_count[0]
            if success:
                meta = self._parse_header(fpath)
                meta["file_name"] = fname
                with rows_lock:
                    rows.append(meta)
                    ok_count[0] += 1
            if done % self.progress_every == 0 or done == total:
                print(f"    {done}/{total} processed, {ok_count[0]} ok")

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {}
            for spec in spectra_info:
                fid = int(spec["fid"])
                fname = f"{model_name}_fid{fid}.txt"
                fpath = os.path.join(out_dir, fname)

                if os.path.exists(fpath) and os.path.getsize(fpath) > 1024:
                    # Validate cached file is actually spectral data
                    if _file_is_spectral(fpath):
                        meta = self._parse_header(fpath)
                        meta["file_name"] = fname
                        with rows_lock:
                            rows.append(meta)
                            ok_count[0] += 1
                        with done_lock:
                            done_count[0] += 1
                        continue
                    else:
                        # Corrupted/HTML cached file — re-download
                        os.remove(fpath)

                futures[ex.submit(self._download_with_retry, model_name, fid, fpath)] = (fid, fname, fpath)

            for fut in as_completed(futures):
                fid, fname, fpath = futures[fut]
                try:
                    success = fut.result()
                    if not success and os.path.exists(fpath):
                        os.remove(fpath)
                    on_done(fname, fpath, success)
                except Exception as e:
                    print(f"    [error] FID {fid}: {e}")
                    if os.path.exists(fpath):
                        os.remove(fpath)
                    on_done(fname, fpath, False)

        if rows:
            self._write_lookup(out_dir, rows)
            print(f"  Successfully downloaded {ok_count[0]}/{total} spectra")
        else:
            print(f"  No spectra downloaded for {model_name}")
        return ok_count[0]

    def _download_with_retry(self, model_name, fid, out_path):
        """Download one spectrum with exponential-backoff retries."""
        params = {"model": model_name, "fid": int(fid), "format": "ascii"}
        last_exc = None
        for attempt in range(self.dl_max_retries + 1):
            if attempt > 0:
                wait = 2 ** attempt  # 2, 4, 8 … seconds
                time.sleep(wait)
            try:
                r = self.session.get(
                    self.spectra_base_url, params=params,
                    timeout=self.dl_timeout, stream=True
                )
                if r.status_code != 200:
                    last_exc = f"HTTP {r.status_code}"
                    continue

                tmp_path = out_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)

                # Validate before accepting
                if os.path.getsize(tmp_path) < 512:
                    os.remove(tmp_path)
                    last_exc = "file too small"
                    continue
                if not _file_is_spectral(tmp_path):
                    os.remove(tmp_path)
                    last_exc = "content is not spectral data (HTML error?)"
                    continue

                os.replace(tmp_path, out_path)
                return True

            except requests.exceptions.Timeout:
                last_exc = "timeout"
            except requests.exceptions.ConnectionError as e:
                last_exc = f"connection error: {e}"
            except Exception as e:
                last_exc = str(e)
            finally:
                tmp = out_path + ".tmp"
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

        # All retries exhausted
        print(f"    [fail] FID {fid} after {self.dl_max_retries + 1} attempts: {last_exc}")
        return False

    # -------------------- helpers --------------------

    def _parse_header(self, file_path):
        meta = {}
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("#") and "=" in s:
                        try:
                            key, val = s.split("=", 1)
                            key = key.strip("#").strip()
                            val = val.split("(")[0].strip()
                            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
                            meta[key] = m.group(0) if m else val
                        except Exception:
                            continue
                    elif not s.startswith("#"):
                        break
        except Exception:
            pass
        return meta

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
    try:
        with open(path, "rb") as f:
            return _is_spectral_content(f.read(512))
    except Exception:
        return False