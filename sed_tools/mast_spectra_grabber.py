#!/usr/bin/env python3
# mast_spectra_grabber.py — robust BOSZ@MAST provider (BOSZ 2024 HLSP)
"""
MASTSpectraGrabber
------------------
- Does NOT rely on directory listings (which 404 for some branches).
- Deterministically constructs and fetches the official BOSZ bulk download scripts
  under /hlsps/.../download_scripts, extracts direct .txt/.txt.gz URLs, and filters
  by chosen resolution (e.g., r10000 or rorig) and product suffix (_resam vs _noresam).
- For resampled products (r500..r50000) it fetches the wavelength grid file
  (bosz2024_wave_<res>.txt). For original resolution ('rorig') it reads wavelengths
  from the files directly (3 columns).
- Emits per-spectrum .txt files with a clear header and builds lookup_table.csv.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import os
import re
import string
import urllib.parse
import zlib
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import requests

# ----------------- constants -----------------
DATA_BASE   = "https://archive.stsci.edu/hlsps/bosz/bosz2024"
SCRIPTS_DIR = f"{DATA_BASE}/download_scripts"
RES_KEYS    = ["r500","r1000","r2000","r5000","r10000","r20000","r50000","rorig"]
METALS      = ["m+0.00","m+0.25","m+0.50","m+0.75",
               "m-0.25","m-0.50","m-0.75","m-1.00","m-1.25","m-1.50","m-1.75",
               "m-2.00","m-2.25","m-2.50"]

# BOSZ 2024 filename pattern
HEAD_RE = re.compile(
    r"^bosz2024_(?P<atmos>ap|mp|ms)_t(?P<teff>\d{4,5})_g(?P<gsgn>[+-])(?P<gval>\d\.\d)"
    r"_m(?P<metsgn>[+-])(?P<metval>\d\.\d{2})_a(?P<asgn>[+-])(?P<aval>\d\.\d{2})"
    r"_c(?P<csgn>[+-])(?P<cval>\d\.\d{2})_v(?P<vturb>\d)_(?P<res>r\d+|rorig)_(?P<prod>resam|noresam|lineid)"
)

SCRIPT_URL_RE = re.compile(r'https?://\S+?\.txt(?:\.gz)?', re.IGNORECASE)

# Errors that make one spectrum fail without stopping the rest of the download
_SPECTRUM_ERRORS = (requests.RequestException, OSError, ValueError)

# ----------------- small utils -----------------
def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _http_get(url: str, session: Optional[requests.Session] = None, timeout: int = 120) -> requests.Response:
    sess = session or requests.Session()
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    return r

def _download_text(url: str, session: Optional[requests.Session] = None, timeout: int = 180) -> str:
    """Fetch a text file; "" if the server says it does not exist (404). Raises on any other failure."""
    try:
        return _http_get(url, session, timeout).text
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return ""
        raise

def _download_text_gz(url: str, session: Optional[requests.Session] = None) -> str:
    """Fetch a spectrum file and gunzip it if it is gzip data. Raises on failure."""
    raw = _http_get(url, session, timeout=300).content
    if raw[:2] != b"\x1f\x8b":  # served uncompressed
        return raw.decode("utf-8", errors="ignore")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            return gz.read().decode("utf-8", errors="ignore")
    except (OSError, EOFError, zlib.error) as e:
        raise ValueError(f"corrupt gzip data from {url}: {e}") from e

def _sanitize_filename(name: str) -> str:
    keep = f"-_.() {string.ascii_letters}{string.digits}"
    return "".join(ch if ch in keep else "_" for ch in name)

def _is_monotonic_increasing(a: np.ndarray) -> bool:
    return a.size > 1 and np.all(np.diff(a) > 0)

def _in_range(value: float, bounds: Optional[Tuple[float, float]]) -> bool:
    if bounds is None:
        return True
    lo, hi = bounds
    if lo > hi:
        lo, hi = hi, lo
    return lo <= value <= hi

def _metal_tag_to_float(tag: str) -> Optional[float]:
    m = re.match(r"^m([+\-])(?P<val>\d\.\d{2})$", tag)
    if not m:
        return None
    sign = -1.0 if m.group(1) == "-" else 1.0
    return sign * float(m.group("val"))

def _metal_tag(token: str) -> str:
    """The BOSZ tag ('m-1.00') for a typed metallicity; ValueError if BOSZ has no such value."""
    try:
        val = float(token)
    except ValueError:
        raise ValueError(f"Metallicity '{token}' is not a number") from None
    sgn = "+" if val >= 0 else "-"
    tag = f"m{sgn}{abs(val):.2f}"
    if tag not in METALS:
        raise ValueError(f"BOSZ 2024 has no [M/H] = {token}; choose from "
                         + ", ".join(m[1:] for m in METALS))
    return tag

def _filter_metals_by_range(
    metals: List[str],
    meta_range: Optional[Tuple[float, float]],
) -> List[str]:
    if meta_range is None:
        return metals
    return [
        met for met in metals
        if (val := _metal_tag_to_float(met)) is not None and _in_range(val, meta_range)
    ]

def _filter_urls_by_params(
    urls: List[str],
    teff_range: Optional[Tuple[float, float]] = None,
    logg_range: Optional[Tuple[float, float]] = None,
) -> List[str]:
    if teff_range is None and logg_range is None:
        return urls

    kept = []
    for url in urls:
        params = _parse_params_from_name(os.path.basename(urllib.parse.urlparse(url).path))
        if not params:
            kept.append(url)  # cannot be classified; the download reports it
            continue
        if _in_range(float(params["teff"]), teff_range) and _in_range(float(params["logg"]), logg_range):
            kept.append(url)
    return kept

# ----------------- parsing helpers -----------------
def _parse_params_from_name(fname: str) -> Dict[str, str]:
    m = HEAD_RE.match(os.path.basename(fname))
    if not m:
        return {}
    d = m.groupdict()
    teff = int(d["teff"])
    logg = float(("-" if d["gsgn"] == "-" else "+") + d["gval"])
    mh   = float(("-" if d["metsgn"] == "-" else "+") + d["metval"])
    am   = float(("-" if d["asgn"]   == "-" else "+") + d["aval"])
    cm   = float(("-" if d["csgn"]   == "-" else "+") + d["cval"])
    res  = d["res"]; vt = int(d["vturb"])
    return {
        "teff": str(teff), "logg": f"{logg:.2f}",
        "mh": f"{mh:.2f}", "am": f"{am:.2f}", "cm": f"{cm:.2f}",
        "vturb": str(vt), "res": res, "atmos": d["atmos"],
    }

def _parse_wavefile(text: str) -> np.ndarray:
    xs = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            xs.append(float(s.split()[0]))
        except ValueError:
            pass  # a label line, not a wavelength
    return np.asarray(xs, dtype=float)

def _parse_rorig_txt(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wl, fx, ct = [], [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                wl.append(float(parts[0])); fx.append(float(parts[1])); ct.append(float(parts[2]))
            except ValueError:
                continue
    return np.asarray(wl, float), np.asarray(fx, float), np.asarray(ct, float)

def _parse_resam_txt(text: str) -> Tuple[np.ndarray, np.ndarray]:
    fx, ct = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                fx.append(float(parts[0])); ct.append(float(parts[1]))
            except ValueError:
                continue
    return np.asarray(fx, float), np.asarray(ct, float)

def _detect_inline_wave(text: str) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    wl, f1, f2 = [], [], []
    rows = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        nums = []
        for p in parts:
            try: nums.append(float(p))
            except ValueError: nums = []; break
        if not nums: continue
        rows += 1
        if len(nums) >= 3:
            wl.append(nums[0]); f1.append(nums[1]); f2.append(nums[2])
        elif len(nums) == 2:
            wl.append(nums[0]); f1.append(nums[1])
        else:
            return (False, None, None, None)
        if rows > 50: break
    if rows == 0: return (False, None, None, None)
    wl = np.asarray(wl, float); f1 = np.asarray(f1, float)
    f2 = np.asarray(f2, float) if len(f2) else None
    if f2 is not None and _is_monotonic_increasing(wl):
        return (True, wl, f1, f2)
    if _is_monotonic_increasing(wl) and wl[-1]/max(wl[0],1e-12) > 1.1:
        return (True, wl, f1, None)
    return (False, None, None, None)

def _try_wave_urls(reskey: str, session: requests.Session) -> Tuple[np.ndarray, str]:
    """The BOSZ wavelength grid for a resolution, as (wavelengths, url).

    Raises RuntimeError saying what each candidate location gave if none
    holds a usable grid.
    """
    # Official locations under /hlsps/
    candidates = [
        f"{DATA_BASE}/wavelength_grids/bosz2024_wave_{reskey}.txt",
        f"{DATA_BASE}/bosz2024_wave_{reskey}.txt",
        f"{DATA_BASE}/waves/bosz2024_wave_{reskey}.txt",
        f"{DATA_BASE}/{reskey}/bosz2024_wave_{reskey}.txt",
        f"{DATA_BASE}/bosz2024_{reskey}_wave.txt",
    ]
    tried = []
    for url in candidates:
        txt = _download_text(url, session=session)
        if not txt:
            tried.append(f"{url}: not found")
            continue
        w = _parse_wavefile(txt)
        if w.size > 10 and _is_monotonic_increasing(w):
            return (w, url)
        tried.append(f"{url}: not an increasing wavelength grid ({w.size} values)")
    raise RuntimeError(f"No BOSZ wavelength grid found for {reskey}:\n  " + "\n  ".join(tried))

# ----------------- URL harvesting (no listings) -----------------
def _gather_urls_from_scripts(reskey: str, metals: List[str], session: requests.Session) -> List[str]:
    """
    Build each official bulk script URL deterministically:
      {SCRIPTS_DIR}/hlsp_bosz_bosz2024_sim_<reskey>_<metal>_v1_bulkdl.sh
    Extract all direct .txt/.txt.gz URLs; filter by product (_resam vs _noresam).

    Raises RuntimeError if any script cannot be read (the grid would lack
    that metallicity) or the scripts list no spectra.
    """
    want_metals = metals if metals and len(metals) != len(METALS) else METALS
    urls = set()
    unread = []
    for met in want_metals:
        met_enc = urllib.parse.quote(met, safe="")  # 'm+0.00' -> 'm%2B0.00'
        script = f"{SCRIPTS_DIR}/hlsp_bosz_bosz2024_sim_{reskey}_{met_enc}_v1_bulkdl.sh"
        try:
            stxt = _download_text(script, session=session)
        except requests.RequestException as e:
            unread.append(f"{script}: {e}")
            continue
        if not stxt:
            unread.append(f"{script}: not found")
            continue
        urls.update(_script_spectrum_urls(stxt, reskey))
    if unread:
        raise RuntimeError(f"Could not read {len(unread)} BOSZ download script(s) for {reskey}, so those "
                           "metallicities would be missing:\n  " + "\n  ".join(unread))
    if not urls:
        raise RuntimeError(f"The BOSZ download scripts for {reskey} list no spectra")
    return sorted(urls)

def _script_spectrum_urls(stxt: str, reskey: str) -> List[str]:
    """Spectrum URLs in a bulk script for this resolution: _noresam for rorig, else _resam."""
    product = "_noresam.txt" if reskey == "rorig" else "_resam.txt"
    found = []
    for u in SCRIPT_URL_RE.findall(stxt):
        upath = urllib.parse.urlparse(u).path.lower()
        if f"/{reskey}/" in upath and (product in upath or product + ".gz" in upath):
            found.append(u)
    return found

# ----------------- spectrum files -----------------
def _apply_wl_range(
    wl: np.ndarray,
    fx: np.ndarray,
    ct: Optional[np.ndarray],
    wl_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if wl_range is None:
        return wl, fx, ct
    lo, hi = wl_range
    if lo > hi:
        lo, hi = hi, lo
    mask = (wl >= lo) & (wl <= hi)
    if not np.any(mask):
        return np.array([]), np.array([]), None
    return wl[mask], fx[mask], (ct[mask] if ct is not None and ct.size == wl.size else ct)

def _write_spectrum_txt(out_path: str, wl: np.ndarray, fx: np.ndarray, ct: Optional[np.ndarray],
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
            ct = np.ones_like(fx)
        for x,y,z in zip(wl, fx, ct):
            f.write(f"{x:.6f} {y:.8e} {z:.8e}\n")

def _parse_any(text: str, inline_w: Optional[np.ndarray], wave: Optional[np.ndarray],
               reskey: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Wavelength, flux and continuum from a BOSZ file; empty arrays if none can be read.

    3-column files carry their wavelengths; 2-column (flux, continuum) files
    take them from *inline_w* (detected in the first file) or the grid *wave*.
    """
    wl3, f3, c3 = _parse_rorig_txt(text)
    if wl3.size:
        return wl3, f3, c3
    fx, ct = _parse_resam_txt(text)
    if fx.size:
        if inline_w is not None:
            n = min(fx.size, inline_w.size)
            return inline_w[:n], fx[:n], (ct[:n] if ct.size >= n else None)
        if wave is not None:
            if fx.size != wave.size:
                raise ValueError(
                    f"BOSZ {reskey} wavelength-grid length mismatch: "
                    f"wave={wave.size}, flux={fx.size}"
                )
            continuum = ct if ct.size == fx.size else None
            return wave, fx, continuum
    # last-ditch: 2-col (wave, flux)
    wl_guess, fx_guess = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        parts = s.split()
        if len(parts) == 2:
            try: a = float(parts[0]); b = float(parts[1])
            except ValueError: continue
            wl_guess.append(a); fx_guess.append(b)
    wl_guess = np.asarray(wl_guess, float)
    fx_guess = np.asarray(fx_guess, float)
    if _is_monotonic_increasing(wl_guess):
        return wl_guess, fx_guess, None
    return np.array([]), np.array([]), None

def _write_lookup_table(model_dir: str, names: List[str]) -> None:
    out_csv = os.path.join(model_dir, "lookup_table.csv")
    rows = []; keys = set(["file_name"])
    for n in names:
        pd = _parse_params_from_name(n); pd["file_name"] = n
        rows.append(pd); keys.update(pd.keys())
    header = ["file_name"] + sorted([k for k in keys if k != "file_name"])
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow([r.get(k, "") for k in header])
    print(f"[MAST] wrote lookup_table.csv with {len(rows)} rows")

# ----------------- main class -----------------
class MASTSpectraGrabber:
    def __init__(self, base_dir: str, max_workers: int = 5, session: Optional[requests.Session] = None):
        self.base_dir = base_dir
        self.max_workers = max_workers
        self.session = session or requests.Session()

    # API expected by SED_tools.py
    def discover_models(self) -> List[str]:
        print("Discovering spectral grids from MAST (BOSZ 2024)...")
        return [f"BOSZ-2024-{rk}" for rk in RES_KEYS]

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Ask which metallicities to fetch and return the download plan.

        Raises ValueError for an unknown model name, or a typed metallicity
        that is not a number or not a BOSZ 2024 value.
        """
        m = re.match(r"^BOSZ-2024-(r\d+|rorig)$", model_name)
        if not m:
            raise ValueError(f"Unrecognized BOSZ model name: {model_name}")
        reskey = m.group(1)
        print(f"[MAST:BOSZ] Selected resolution: {reskey}")
        print("  Limit metallicities? e.g., '-1.00,+0.00' or leave blank for ALL")
        raw = input("> ").strip()
        metals = METALS
        if raw:
            sel = [_metal_tag(tok.strip()) for tok in raw.split(",") if tok.strip()]
            if sel:
                metals = sel
        return {"provider": "mast-bosz", "version": "2024", "reskey": reskey, "metals": metals}

    def download_model_spectra(
        self,
        model_name: str,
        info: Dict[str, Any],
        teff_range: Optional[Tuple[float, float]] = None,
        logg_range: Optional[Tuple[float, float]] = None,
        meta_range: Optional[Tuple[float, float]] = None,
        wl_range: Optional[Tuple[float, float]] = None,
        **_ignored: Any,
    ) -> int:
        """Download the BOSZ spectra in the plan from get_model_metadata.

        Returns the number written. Raises RuntimeError if the download
        scripts or the wavelength grid cannot be read; if any spectrum
        fails, the rest are still written, lookup_table.csv is not, and
        RuntimeError lists each failure.
        """
        import concurrent.futures as cf

        reskey: str = info["reskey"]
        metals: List[str] = info["metals"]

        metals = _filter_metals_by_range(metals, meta_range)
        if not metals:
            print(f"[MAST] No BOSZ metallicities match meta_range={meta_range}")
            return 0

        model_dir = os.path.join(self.base_dir, model_name)
        _safe_makedirs(model_dir)

        # -------- harvest URLs from official scripts (no listings, no API) --------
        urls = _gather_urls_from_scripts(reskey, metals, self.session)
        urls = _filter_urls_by_params(urls, teff_range=teff_range, logg_range=logg_range)
        if not urls:
            print(
                f"[MAST] No BOSZ spectra URLs found for {reskey} "
                f"with teff_range={teff_range}, logg_range={logg_range}, "
                f"meta_range={meta_range}"
            )
            return 0
        print(f"[MAST] harvest = download_scripts ({len(urls)} urls)")

        # -------- detect inline wavelength; if not, fetch wave grid (resampled only) --------
        first_url = urls[0]
        try:
            first_text = _download_text_gz(first_url, session=self.session)
        except _SPECTRUM_ERRORS:
            first_text = None  # handle_one fetches it again and reports the failure with the rest
        has_inline, inline_w, inline_f, inline_c = _detect_inline_wave(first_text or "")

        wave = None; wave_url = None
        if reskey != "rorig" and not has_inline:
            wave, wave_url = _try_wave_urls(reskey, self.session)

        def handle_one(u: str, pre_text: Optional[str] = None) -> Optional[str]:
            """Write one spectrum; its file name, or None if wl_range leaves no points.

            Raises one of _SPECTRUM_ERRORS if it cannot be fetched, named or read.
            """
            text = pre_text if pre_text is not None else _download_text_gz(u, session=self.session)
            if not text:
                raise ValueError("empty file")
            base = os.path.basename(urllib.parse.urlparse(u).path)
            params = _parse_params_from_name(base)
            if not params:
                raise ValueError("file name does not follow the BOSZ 2024 pattern")
            oname = _sanitize_filename(base.replace(".txt.gz", ".txt"))
            out_path = os.path.join(model_dir, oname)

            if has_inline and pre_text is not None and inline_w is not None and inline_f is not None:
                ct = inline_c if inline_c is not None else np.ones_like(inline_f)
                wl, fx, ct = _apply_wl_range(inline_w, inline_f, ct, wl_range)
                if wl.size == 0 or fx.size == 0:
                    return None
                _write_spectrum_txt(out_path, wl, fx, ct, u, params, None)
                return oname

            wl, fx, ct = _parse_any(text, inline_w if has_inline else None, wave, reskey)
            if wl.size == 0:
                raise ValueError("no spectrum could be read from the file")
            wl, fx, ct = _apply_wl_range(wl, fx, ct, wl_range)
            if wl.size == 0 or fx.size == 0: return None
            _write_spectrum_txt(out_path, wl, fx, ct, u, params, wave_url)
            return oname

        def try_one(u: str, pre_text: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
            try:
                return handle_one(u, pre_text), None
            except _SPECTRUM_ERRORS as e:
                return None, f"{u}: {e}"

        # write first (prefetched) then parallelize the rest
        results = [try_one(first_url, pre_text=first_text)]
        rest = urls[1:]
        if rest:
            with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                results.extend(ex.map(try_one, rest))
        names = [name for name, _ in results if name]
        failures = [error for _, error in results if error]

        if failures:
            raise RuntimeError(
                f"MAST: {len(failures)} of {len(urls)} BOSZ spectra for {model_name} failed, so "
                f"lookup_table.csv was not written; the others are in {model_dir}:\n  "
                + "\n  ".join(failures))

        if names:
            _write_lookup_table(model_dir, names)

        return len(names)
