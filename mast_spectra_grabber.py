#!/usr/bin/env python3
# mast_spectra_grabber.py
"""
MASTSpectraGrabber
------------------
Provider for BOSZ@MAST HLSP (2024 version). Produces per-spectrum .txt files
(with headers), a lookup_table.csv, and returns a count of spectra written.
Designed to match the interface used by SED_tools.py for SVO/MSG grabbers.

This implementation uses direct HTTP directory scraping of the BOSZ HLSP page
documented at STScI/MAST. For resampled resolutions (r500..r50000), wavelengths
are read from the corresponding `bosz2024_wave_rXXXX.txt` file. For original
resolution ('rorig'), wavelengths are present in each file.

References:
- BOSZ HLSP page (file layout, naming, grids, and wave files)
  https://archive.stsci.edu/hlsp/bosz

Output directory structure (under base_dir):
    <base_dir>/<model_name>/
        *.txt    (one per BOSZ spectrum)
        lookup_table.csv

Each .txt file includes a comment header (lines starting with '#') that records
the parsed parameters and the source URL to facilitate reproducibility.
"""

from __future__ import annotations

import os
import io
import re
import csv
import gzip
import time
import math
import queue
import shutil
import string
import random
import urllib.parse
import concurrent.futures as cf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup
import urllib.parse
import posixpath
# ---- constants ----

BOSZ_BASE = "https://archive.stsci.edu/hlsp/bosz/bosz2024"
RES_KEYS  = ["r500","r1000","r2000","r5000","r10000","r20000","r50000","rorig"]
METALS    = ["m+0.00","m+0.25","m+0.50","m+0.75",
             "m-0.25","m-0.50","m-0.75","m-1.00","m-1.25","m-1.50","m-1.75",
             "m-2.00","m-2.25","m-2.50"]
MET_DIR_RE = re.compile(r"^m[+-]\d\.\d{2}/?$", re.IGNORECASE)

HEAD_RE = re.compile(
    r"^bosz2024_(?P<atmos>ap|mp|ms)_t(?P<teff>\d{4,5})_g(?P<gsgn>[+-])(?P<gval>\d\.\d)"
    r"_m(?P<metsgn>[+-])(?P<metval>\d\.\d{2})_a(?P<asgn>[+-])(?P<aval>\d\.\d{2})"
    r"_c(?P<csgn>[+-])(?P<cval>\d\.\d{2})_v(?P<vturb>\d)_(?P<res>r\d+|rorig)_(?P<prod>resam|noresam|lineid)"
)

# ---- helpers ----



def _list_metallicity_dirs(res_url: str, session: requests.Session) -> List[str]:
    """
    Given the URL of a resolution directory (e.g., .../r10000/),
    return absolute URLs of metallicity subdirectories (m+0.00/, m-1.00/, ...).
    """
    links = _list_dir_links(res_url, session=session)
    out = []
    for u in links:
        # Keep only immediate child dirs under res_url
        parsed = urllib.parse.urlparse(u)
        if not u.endswith("/"):
            continue
        # must be a child like .../r10000/m+0.00/
        if not parsed.path.startswith(urllib.parse.urlparse(res_url).path):
            continue
        tail = parsed.path.rstrip("/").split("/")[-1] + "/"
        if MET_DIR_RE.match(tail):
            out.append(u)
    return sorted(set(out))

def _is_txt_like(name: str) -> bool:
    n = name.lower()
    return n.endswith(".txt") or n.endswith(".txt.gz")

def _want_product(name: str, reskey: str) -> bool:
    n = name.lower()
    if reskey == "rorig":
        return n.endswith("_noresam.txt.gz") or n.endswith("_noresam.txt")
    else:
        return n.endswith("_resam.txt.gz") or n.endswith("_resam.txt")

def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _http_get(url: str, session: Optional[requests.Session] = None, timeout: int = 60) -> requests.Response:
    sess = session or requests.Session()
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    return r

def _list_dir_links(url: str, session: Optional[requests.Session] = None) -> List[str]:
    """Return hrefs found at url (directory listing or HTML table of links)."""
    try:
        r = _http_get(url, session=session, timeout=60)
    except Exception as e:
        print(f"[mast] list failed {url} : {e}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    hrefs = []
    for a in soup.find_all("a", href=True):
        hrefs.append(urllib.parse.urljoin(url, a["href"]))
    return sorted(set(hrefs))

def _download_text_gz(url: str, session: Optional[requests.Session] = None) -> str:
    try:
        r = _http_get(url, session=session, timeout=180)
        # some servers set content-encoding: gzip; some provide .gz file
        raw = r.content
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                return gz.read().decode("utf-8", errors="ignore")
        except OSError:
            # not gzipped; return as text
            return raw.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[mast] gz fetch failed {url}: {e}")
        return ""

def _download_text(url: str, session: Optional[requests.Session] = None) -> str:
    try:
        r = _http_get(url, session=session, timeout=120)
        return r.text
    except Exception as e:
        print(f"[mast] text fetch failed {url}: {e}")
        return ""

def _parse_wavefile(text: str) -> np.ndarray:
    xs = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            xs.append(float(s.split()[0]))
        except Exception:
            continue
    return np.asarray(xs, dtype=float)

def _parse_resam_txt(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (flux, cont) arrays from a resampled BOSZ .txt (no wavelength column)."""
    fx, ct = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                fx.append(float(parts[0]))
                ct.append(float(parts[1]))
            except Exception:
                continue
    return np.asarray(fx, dtype=float), np.asarray(ct, dtype=float)

def _parse_rorig_txt(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (wave, H_or_F, cont) from original-res txt with 3 columns."""
    wl, fx, ct = [], [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                wl.append(float(parts[0]))
                fx.append(float(parts[1]))
                ct.append(float(parts[2]))
            except Exception:
                continue
    return (np.asarray(wl, dtype=float),
            np.asarray(fx, dtype=float),
            np.asarray(ct, dtype=float))

def _sanitize_filename(name: str) -> str:
    keep = f"-_.() {string.ascii_letters}{string.digits}"
    return "".join(ch if ch in keep else "_" for ch in name)

def _parse_params_from_name(fname: str) -> Dict[str, str]:
    """Parse BOSZ filename into parameter dict; returns {} if not matched."""
    m = HEAD_RE.match(os.path.basename(fname))
    if not m:
        return {}
    d = m.groupdict()
    teff = int(d["teff"])
    logg = float(("-" if d["gsgn"] == "-" else "+") + d["gval"])
    mh   = float(("-" if d["metsgn"] == "-" else "+") + d["metval"])
    am   = float(("-" if d["asgn"] == "-" else "+") + d["aval"])
    cm   = float(("-" if d["csgn"] == "-" else "+") + d["cval"])
    res  = d["res"]
    vt   = int(d["vturb"])
    return {
        "teff": str(teff),
        "logg": f"{logg:.2f}",
        "mh":   f"{mh:.2f}",
        "am":   f"{am:.2f}",
        "cm":   f"{cm:.2f}",
        "vturb": str(vt),
        "res": res,
        "atmos": d["atmos"],
    }


def _is_monotonic_increasing(a: np.ndarray) -> bool:
    return np.all(np.diff(a) > 0)

def _detect_inline_wave(text: str) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Inspect a BOSZ txt to see if it already contains wavelength.
    Returns (has_wave, wave, flux, cont) where arrays may be None if not parsed here.
    """
    wl, f1, f2 = [], [], []
    data_rows = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                nums = []
                break
        if not nums:
            continue
        data_rows += 1
        if len(nums) >= 3:
            wl.append(nums[0]); f1.append(nums[1]); f2.append(nums[2])
        elif len(nums) == 2:
            # Could be (wl, flux) OR (flux, cont). We'll disambiguate later.
            wl.append(nums[0]); f1.append(nums[1])
        else:
            # single column → definitely not inline wave
            return (False, None, None, None)
        if data_rows > 50:  # enough to decide
            break

    if data_rows == 0:
        return (False, None, None, None)

    wl = np.asarray(wl, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float) if len(f2) else None

    # If we saw 3+ columns, first is almost certainly wavelength.
    if f2 is not None and wl.size and _is_monotonic_increasing(wl):
        return (True, wl, f1, f2)

    # If we saw exactly 2 columns in the sample, check if first looks like wavelength:
    # heuristic: positive, strictly increasing, and spans at least ~10% range
    if wl.size and _is_monotonic_increasing(wl):
        span = wl[-1] / max(wl[0], 1e-12)
        if span > 1.1:
            # treat as (wave, flux); no continuum
            return (True, wl, f1, None)

    return (False, None, None, None)

def _try_wave_urls(reskey: str, session: requests.Session) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Try a few plausible wavelength-grid URLs for resampled BOSZ.
    Returns (wave_array, url_used) or (None, None).
    """
    candidates = [
        f"{BOSZ_BASE}/bosz2024_wave_{reskey}.txt",
        f"{BOSZ_BASE}/waves/bosz2024_wave_{reskey}.txt",
        f"{BOSZ_BASE}/{reskey}/bosz2024_wave_{reskey}.txt",
        f"{BOSZ_BASE}/bosz2024_{reskey}_wave.txt",
    ]
    for url in candidates:
        txt = _download_text(url, session=session)
        if txt:
            w = _parse_wavefile(txt)
            if w.size > 10 and _is_monotonic_increasing(w):
                return (w, url)
    # Last resort: scrape root for anything containing 'wave' and the reskey
    root_links = _list_dir_links(BOSZ_BASE + "/", session=session)
    for u in root_links:
        name = os.path.basename(urllib.parse.urlparse(u).path).lower()
        if "wave" in name and reskey in name:
            txt = _download_text(u, session=session)
            if txt:
                w = _parse_wavefile(txt)
                if w.size > 10 and _is_monotonic_increasing(w):
                    return (w, u)
    return (None, None)


# ---- main class ----

class MASTSpectraGrabber:
    """
    BOSZ@MAST spectra grabber.
    - discover_models(): list supported BOSZ grids by resolution
    - get_model_metadata(model_name): pack chosen resolution and selected metallicities
    - download_model_spectra(model_name, info): write .txt + lookup; return count
    """

    def __init__(self, base_dir: str, max_workers: int = 5, session: Optional[requests.Session] = None):
        self.base_dir = base_dir
        self.max_workers = max_workers
        self.session = session or requests.Session()

    # ----- public API (used by SED_tools.py) -----

    def discover_models(self) -> List[str]:
        """
        Return list of BOSZ 2024 resolution "models":
            ["BOSZ-2024-r500", ..., "BOSZ-2024-r50000", "BOSZ-2024-rorig"]
        """
        print("Discovering available models from MAST (BOSZ 2024)...")
        return [f"BOSZ-2024-{rk}" for rk in RES_KEYS]

    def get_model_metadata(self, model_name: str) -> Dict[str, any]:
        """
        Parse `model_name` to get resolution key and optionally select metallicities.
        This call is interactive to let the user subset metallicities, otherwise 'all'.
        """
        m = re.match(r"^BOSZ-2024-(r\d+|rorig)$", model_name)
        if not m:
            raise ValueError(f"Unrecognized BOSZ model name: {model_name}")
        reskey = m.group(1)

        print(f"[MAST:BOSZ] Selected resolution: {reskey}")
        print("  Limit metallicities? e.g., '-1.00,+0.00' or leave blank for ALL")
        raw = input("> ").strip()
        if raw:
            metals = []
            for tok in raw.split(","):
                try:
                    val = float(tok)
                except Exception:
                    continue
                # format to BOSZ directory string form
                sgn = "+" if val >= 0 else "-"
                metals.append(f"m{sgn}{abs(val):.2f}")
            metals = [m for m in metals if m in METALS]
            if not metals:
                metals = METALS
        else:
            metals = METALS

        return {"provider": "mast-bosz", "version": "2024", "reskey": reskey, "metals": metals}


    def download_model_spectra(self, model_name: str, info: Dict[str, any]) -> int:
        reskey: str = info["reskey"]
        metals: List[str] = info["metals"]
        model_dir = os.path.join(self.base_dir, model_name)
        _safe_makedirs(model_dir)
        # 1) enumerate product URLs by listing the resolution dir,
        #    then walking into each metallicity subdir
        res_url = f"{BOSZ_BASE}/{reskey}/"
        res_children = _list_metallicity_dirs(res_url, session=self.session)
        if not res_children:
            print(f"[MAST] no metallicity subdirs visible under {res_url}")
            return 0

        # optional user filter: restrict to chosen metals
        if metals and len(metals) != len(METALS):
            # normalize to the server’s encoding form (directories may show as m%2B0.00 or m+0.00)
            keep_tags = set(metals)  # e.g., {'m+0.00', 'm-1.00'}
            filtered = []
            for u in res_children:
                seg = urllib.parse.unquote(urllib.parse.urlparse(u).path.rstrip("/").split("/")[-1])
                if seg in keep_tags:
                    filtered.append(u)
            res_children = filtered or res_children  # if match empty, keep all

        urls = []
        for met_dir in res_children:
            hrefs = _list_dir_links(met_dir, session=self.session)
            for u in hrefs:
                if _is_txt_like(u) and _want_product(u, reskey):
                    urls.append(u)
        urls = sorted(set(urls))

        # Fallback to download_scripts only if nothing found
        if not urls:
            ds_url = f"{BOSZ_BASE}/download_scripts/"
            scripts = [u for u in _list_dir_links(ds_url, session=self.session) if u.lower().endswith(".sh")]
            for s in scripts:
                name = os.path.basename(urllib.parse.urlparse(s).path).lower()
                if reskey in name:
                    stxt = _download_text(s, session=self.session)
                    for line in stxt.splitlines():
                        sline = line.strip()
                        if "https://" in sline and ".txt" in sline:
                            cand = sline.split()[-1]
                            if _is_txt_like(cand) and _want_product(cand, reskey):
                                urls.append(cand)
            urls = sorted(set(urls))


        # Peek at the first file to decide whether wavelength is inline.
        first_url = urls[0]
        first_text = _download_text_gz(first_url, session=self.session)
        has_inline, inline_w, inline_f, inline_c = _detect_inline_wave(first_text)

        wave = None
        wave_url = None
        if reskey != "rorig" and not has_inline:
            wave, wave_url = _try_wave_urls(reskey, self.session)
            if wave is None:
                print(f"[MAST] failed to locate wavelength grid for {reskey}")
                return 0

        def write_txt(out_path: str, wl: np.ndarray, fx: np.ndarray, ct: Optional[np.ndarray],
                      src_url: str, params: Dict[str, str], wave_url_used: Optional[str]) -> None:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("# source = BOSZ@MAST (HLSP)\n")
                f.write("# url = " + src_url + "\n")
                if wave_url_used:
                    f.write("# wave_grid = " + wave_url_used + "\n")
                for k,v in params.items():
                    f.write(f"# {k} = {v}\n")
                f.write("# columns = wavelength_A flux continuum\n")
                if ct is None:
                    # synthesize continuum = 1 if absent
                    ct = np.ones_like(fx)
                for x,y,z in zip(wl, fx, ct):
                    f.write(f"{x:.6f} {y:.8e} {z:.8e}\n")

        def parse_any(text: str, reskey_local: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
            # Try rorig parser first (3 cols), then resampled parsers
            wl3, f3, c3 = _parse_rorig_txt(text)
            if wl3.size:
                return wl3, f3, c3
            # resampled: either (flux, cont) OR inline (wave, flux) 2-col
            fx, ct = _parse_resam_txt(text)
            if fx.size and (wave is not None):
                return wave[:fx.size], fx, (ct if ct.size == fx.size else None)
            # inline 2-col case (wave, flux) detected centrally for first file,
            # but if we get here (no wave grid), assume first col is wavelength.
            wl_guess = []
            fx_guess = []
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) == 2:
                    try:
                        a = float(parts[0]); b = float(parts[1])
                    except Exception:
                        continue
                    wl_guess.append(a); fx_guess.append(b)
            wl_guess = np.asarray(wl_guess, float)
            fx_guess = np.asarray(fx_guess, float)
            if wl_guess.size and _is_monotonic_increasing(wl_guess):
                return wl_guess, fx_guess, None
            return np.array([]), np.array([]), None

        # Write the first file using the prefetched content, then parallelize the rest
        names: List[str] = []

        def handle_one(u: str, pre_text: Optional[str] = None) -> Optional[str]:
            try:
                text = pre_text if pre_text is not None else _download_text_gz(u, session=self.session)
                if not text:
                    return None
                base = os.path.basename(urllib.parse.urlparse(u).path)
                params = _parse_params_from_name(base)
                if not params:
                    return None
                oname = _sanitize_filename(base.replace(".txt.gz", ".txt"))
                out_path = os.path.join(model_dir, oname)

                if has_inline and pre_text is not None:
                    # Use inline parse for the first file
                    if inline_w is not None and inline_f is not None:
                        ct = inline_c if inline_c is not None else np.ones_like(inline_f)
                        write_txt(out_path, inline_w, inline_f, ct, u, params, None)
                        return oname

                wl, fx, ct = parse_any(text, reskey)
                if wl.size == 0 or fx.size == 0:
                    return None
                write_txt(out_path, wl, fx, ct, u, params, wave_url)
                return oname
            except Exception as e:
                print(f"[MAST] failed on {u}: {e}")
                return None

        first_name = handle_one(first_url, pre_text=first_text)
        if first_name:
            names.append(first_name)

        rest = urls[1:]
        if rest:
            with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                for res_name in ex.map(handle_one, rest):
                    if res_name:
                        names.append(res_name)

        # lookup table
        if names:
            out_csv = os.path.join(model_dir, "lookup_table.csv")
            rows = []
            keys = set(["file_name"])
            for n in names:
                pd = _parse_params_from_name(n)
                pd["file_name"] = n
                rows.append(pd)
                keys.update(pd.keys())
            header = ["file_name"] + sorted([k for k in keys if k != "file_name"])
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f); w.writerow(header)
                for r in rows:
                    w.writerow([r.get(k, "") for k in header])
            print(f"[MAST] wrote lookup_table.csv with {len(rows)} rows")

        return len(names)
