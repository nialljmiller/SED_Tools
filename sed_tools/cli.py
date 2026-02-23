#!/usr/bin/env python3
"""
SED_tools.py — one-stop launcher for spectra & filters

Pipeline per selected model:
  - SVO or MSG (Rich Townsend) model discovery
  - Download (SVO) or extract (MSG) spectra -> *.txt collection
  - Ensure lookup_table.csv
  - Build/ensure HDF5 bundle of spectra
  - Precompute flux cube file

Filters (SVO only):
  - Interactive substring filter over Facility/Instrument/Band
  - Saves to data/filters/<Facility>/<Instrument>/<Band>.dat
"""

import argparse
import glob
import math
import os
import re
import shutil
import sys
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from .mast_spectra_grabber import MASTSpectraGrabber
# --- Standard Package Imports (The "Safe" Way) ---
from .models import FILTER_DIR_DEFAULT, STELLAR_DIR_DEFAULT
from .msg_spectra_grabber import MSGSpectraGrabber
from .njm_spectra_grabber import NJMSpectraGrabber
from .precompute_flux_cube import precompute_flux_cube
from .spectra_cleaner import clean_model_dir
from .svo_filter_grabber import run_interactive as _run_filter_cli
from .svo_regen_spectra_lookup import parse_metadata, regenerate_lookup_table
from .svo_spectra_grabber import SVOSpectraGrabber
from .ui_utils import _prompt_choice
# ------------ Small Utils ------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_txt_spectra(model_dir: str) -> List[str]:
    return sorted([f for f in os.listdir(model_dir) if f.lower().endswith(".txt")])


def load_txt_spectrum(txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    wl, fl = [], []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                wl.append(float(parts[0]))
                fl.append(float(parts[1]))
    return np.asarray(wl, dtype=float), np.asarray(fl, dtype=float)


def numeric_from(meta: Dict[str, str], key_candidates: List[str], default: float = np.nan) -> float:
    """Extract first numeric token from any of the candidate keys (case-insensitive)."""
    lower = {k.lower(): v for k, v in meta.items()}
    for ck in key_candidates:
        if ck.lower() in lower:
            val = lower[ck.lower()]
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
            if m:
                return float(m.group(0))
    return default


# ------------ HDF5 Bundling ------------

def build_h5_bundle_from_txt(model_dir: str, out_h5: str) -> None:
    """Create an HDF5 file bundling all .txt spectra."""
    txt_files = list_txt_spectra(model_dir)
    if not txt_files:
        print(f"[H5 bundle] No .txt spectra found in {model_dir}; skipping.")
        return

    ensure_dir(os.path.dirname(out_h5))
    with h5py.File(out_h5, "w") as h5:
        spectra_grp = h5.create_group("spectra")
        for fname in txt_files:
            path = os.path.join(model_dir, fname)
            wl, fl = load_txt_spectrum(path)
            g = spectra_grp.create_group(fname)
            g.create_dataset("lambda", data=wl, dtype="f8")
            g.create_dataset("flux",   data=fl, dtype="f8")

            meta = parse_metadata(path)
            teff = numeric_from(meta, ["Teff", "teff", "T_eff"])
            logg = numeric_from(meta, ["logg", "Logg", "log_g"])
            feh = numeric_from(meta, ["FeH", "feh", "metallicity", "[Fe/H]", "meta"])
            if not np.isnan(teff): g.attrs["teff"] = teff
            if not np.isnan(logg): g.attrs["logg"] = logg
            if not np.isnan(feh):  g.attrs["feh"] = feh
            for k, v in meta.items():
                g.attrs[f"raw:{k}"] = v

    print(f"[H5 bundle] Wrote {out_h5}")


# ------------ UI Helpers (Your Full Interactive Menu) ------------

class _Opt:
    __slots__ = ("src", "name", "label")

    def __init__(self, src, name):
        self.src, self.name = src, name
        self.label = f"{name} [{src}]"


def _prompt_choice(
    options: Sequence,
    label: str,
    allow_back: bool = False,
    page_size: int = 100,
    max_label: int = -1,
    min_cols: int = 1,
    max_cols: int = 3,
    use_color: bool = True,
    multi: bool = False,
) -> Union[int, List[int], None]:
    """
    Plain-ASCII picker with stable IDs, paging, grid columns, and simple filters.
    If multi=True, allows input like "1, 3-5" and returns a List[int] of indices.
    Otherwise returns a single int.
    Returns None for quit, -1 for back (if allowed).
    """
    if not options:
        print(f"No {label} options available.")
        return None

    if max_label < 0:
        max_label = max(len(getattr(x, "label", str(x))) for x in options) + 4

    labels: List[str] = [getattr(o, "label", str(o)) for o in options]
    N = len(labels)
    page = 1
    filt: Optional[Tuple[str, str]] = None

    use_color = use_color and sys.stdout.isatty() and ("NO_COLOR" not in os.environ)
    BOLD = "\x1b[1m" if use_color else ""
    DIM = "\x1b[2m" if use_color else ""
    CYAN = "\x1b[36m" if use_color else ""
    YELL = "\x1b[33m" if use_color else ""
    GREEN = "\x1b[32m" if use_color else ""
    RESET = "\x1b[0m" if use_color else ""

    def term_width() -> int:
        return shutil.get_terminal_size().columns

    def apply_filter(idx: List[int]) -> List[int]:
        if filt is None: return idx
        kind, patt = filt
        if kind == "substr":
            p = patt.lower()
            return [i for i in idx if p in labels[i].lower()]
        if kind == "neg":
            p = patt.lower()
            return [i for i in idx if p not in labels[i].lower()]
        rx = re.compile(patt, re.I)
        return [i for i in idx if rx.search(labels[i])]

    def page_slice(total: int, p: int) -> slice:
        pmax = max(1, math.ceil(total / page_size))
        if p > pmax: p = pmax
        if p < 1: p = 1
        a = (p - 1) * page_size
        b = min(a + page_size, total)
        return slice(a, b)

    def ellipsize(s: str) -> str:
        return s if len(s) <= max_label else s[:max_label - 1] + "…"

    def hl(s: str) -> str:
        if not use_color or filt is None: return s
        kind, patt = filt
        if kind != "substr" or not patt: return s
        rx = re.compile(re.escape(patt), re.I)
        return rx.sub(lambda m: f"{YELL}{m.group(0)}{RESET}", s)

    def grid_print(visible_ids: List[int]) -> None:
        width = max(80, term_width())
        names = [labels[i] for i in visible_ids]
        col_w = 12 + max_label + 2
        cols = max(min_cols, min(max_cols, max(1, width // col_w)))
        cells = [f"[{GREEN}{i+1:4d}{RESET}] {hl(ellipsize(s))}" for i, s in zip(visible_ids, names)]
        while len(cells) % cols: cells.append("")
        rows = [cells[k:k+cols] for k in range(0, len(cells), cols)]
        print(f"\n{BOLD}{label}{RESET} ({CYAN}{len(all_idx)}{RESET} total):")
        print("─" * min(80, width))
        for r in rows:
            print("".join(x.ljust(col_w - 2) for x in r))

    all_idx = list(range(N))
    while True:
        kept = apply_filter(all_idx)
        sl = page_slice(len(kept), page)
        view = kept[sl]
        start, end = sl.start + 1, sl.stop
        grid_print(view)

        ftxt = ""
        if filt:
            kind, patt = filt
            ftxt = f' {DIM}filter="/{patt}"{RESET}' if kind == "substr" else (
                   f' {DIM}filter="!{patt}"{RESET}' if kind == "neg" else
                   f' {DIM}filter="//{patt}"{RESET}')
        
        controls = f"{DIM}"
        if end < len(kept):
            print(f"{DIM}Showing {start}–{end} of {len(kept)}{ftxt}{RESET}")
            controls += "n, p, g <page>, "
        
        controls += "/text, !text, //regex, "
        if multi:
            controls += "list (1,3) or range (1-5), "
        else:
            controls += "id <N> (or just N), "

        controls += "clear"
        if allow_back: controls += ", b"
        controls += ", q" + RESET
        print(controls)

        inp = input("> ").strip()
        if not inp: continue
        low = inp.lower()

        if low in ("q", "quit", "exit"): return None
        if allow_back and low in ("b", "back"): return -1
        if low == "n": page += 1; continue
        if low == "p": page -= 1; continue
        if low.startswith("g "):
            parts = low.split()
            if len(parts) == 2 and parts[1].isdigit(): page = int(parts[1])
            continue
        if low == "clear": filt = None; page = 1; continue
        if low.startswith("//"): patt = inp[2:].strip(); filt = ("regex", patt) if patt else None; page = 1; continue
        if low.startswith("!"):  patt = inp[1:].strip();  filt = ("neg", patt)   if patt else None; page = 1; continue
        if low.startswith("/"):  patt = inp[1:].strip();  filt = ("substr", patt)if patt else None; page = 1; continue
        if low.startswith("id "):
            parts = low.split()
            if len(parts) == 2 and parts[1].isdigit():
                k = int(parts[1])
                if 1 <= k <= N: return k - 1
            continue

        # --- Multi Selection Logic ---
        if multi:
            # Check if input looks like a numeric selection string (digits, comma, hyphen, space)
            # This prevents treating search terms like "C3K" as a failed selection attempt
            if re.match(r"^[\d\s,-]+$", inp):
                selected_indices = []
                parts = inp.split(',')
                is_range = False
                for p in parts:
                    p = p.strip()
                    if not p: continue
                    if '-' in p:
                        is_range = True
                        a, b = p.split('-', 1)
                        # Handle cases like "1 - 5" or "1-5"
                        a, b = int(a), int(b)
                        selected_indices.extend(range(a, b + 1))
                    else:
                        selected_indices.append(int(p))
                
                # Deduplicate and validate
                valid = [x - 1 for x in sorted(list(set(selected_indices))) if 1 <= x <= N]
                if valid:
                    return valid

        # --- Single Selection Logic ---
        if not multi and inp.isdigit():
            k = int(inp)
            if 1 <= k <= N: return k - 1
            continue

        # Default to substring filter
        filt = ("substr", inp); page = 1




def _parse_range(raw: str) -> Optional[Tuple[float, float]]:
    """Parse a user-entered range string like '3500,8000' or '3500 8000' or '3500-8000'.
    
    Returns (min, max) tuple or None if empty/invalid.
    """
    raw = raw.strip()
    if not raw:
        return None
    
    # Try comma-separated
    for sep in [',', ' ', '-', '..', ':']:
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep, 1) if p.strip()]
            if len(parts) == 2:
                try:
                    lo, hi = float(parts[0]), float(parts[1])
                    if lo > hi:
                        lo, hi = hi, lo
                    return (lo, hi)
                except ValueError:
                    pass
    
    # Single value — no range
    print(f"  Could not parse range from '{raw}' — need two values (e.g. '3500,8000')")
    return None


def prompt_njm_axis_cuts(
    model_name: str,
    grabber,  # NJMSpectraGrabber instance
    model_url: Optional[str] = None,
) -> Dict:
    """Interactively prompt the user for axis cuts on an NJM download.
    
    Shows the available parameter ranges from the remote lookup table,
    then asks if the user wants to cut on each axis.
    
    Returns a dict with keys: teff_range, logg_range, meta_range, wl_range
    (each either a (min, max) tuple or None).
    """
    cuts = {
        'teff_range': None,
        'logg_range': None,
        'meta_range': None,
        'wl_range': None,
    }
    
    print(f"\n  ── Axis Cuts for {model_name} ──")
    print(f"  You can restrict the download to a subset of the grid.")
    print(f"  Leave blank to download everything.\n")
    
    # Try to fetch grid info for context
    info = grabber.get_grid_info(model_name, model_url=model_url)
    
    if info:
        print(f"  Available grid ({info['n_spectra']} spectra):")
        if 'teff_min' in info:
            print(f"    Teff:  {info['teff_min']:.0f} – {info['teff_max']:.0f} K  ({info['teff_unique']} values)")
        if 'logg_min' in info:
            print(f"    logg:  {info['logg_min']:.2f} – {info['logg_max']:.2f}    ({info['logg_unique']} values)")
        if 'meta_min' in info:
            print(f"    [M/H]: {info['meta_min']:+.2f} – {info['meta_max']:+.2f}    ({info['meta_unique']} values)")
        print()
    else:
        print("  (Could not fetch grid info — cuts will still work if lookup_table.csv is available)\n")
    
    # Ask about each axis
    raw = input("  Teff range (e.g. '3500,8000') or blank for all: ").strip()
    cuts['teff_range'] = _parse_range(raw)
    
    raw = input("  logg range (e.g. '3.5,5.0') or blank for all: ").strip()
    cuts['logg_range'] = _parse_range(raw)
    
    raw = input("  [M/H] range (e.g. '-1.0,0.5') or blank for all: ").strip()
    cuts['meta_range'] = _parse_range(raw)
    
    raw = input("  Wavelength range in Å (e.g. '3000,10000') or blank for all: ").strip()
    cuts['wl_range'] = _parse_range(raw)
    
    # Summary
    any_cuts = any(v is not None for v in cuts.values())
    if any_cuts:
        print(f"\n  Applied cuts:")
        if cuts['teff_range']:
            print(f"    Teff:       {cuts['teff_range'][0]:.0f} – {cuts['teff_range'][1]:.0f} K")
        if cuts['logg_range']:
            print(f"    logg:       {cuts['logg_range'][0]:.2f} – {cuts['logg_range'][1]:.2f}")
        if cuts['meta_range']:
            print(f"    [M/H]:      {cuts['meta_range'][0]:+.2f} – {cuts['meta_range'][1]:+.2f}")
        if cuts['wl_range']:
            print(f"    Wavelength: {cuts['wl_range'][0]:.1f} – {cuts['wl_range'][1]:.1f} Å")
        print()
    else:
        print("\n  No cuts — downloading full grid.\n")
    
    return cuts



# ------------ Workflow Runners ------------

def run_rebuild_flow(
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models: Optional[List[str]] = None,
    rebuild_h5: bool = True,
    rebuild_flux_cube: bool = True
) -> None:
    ensure_dir(base_dir)

    cand = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p):
            continue
        has_txt = any(fn.lower().endswith(".txt") for fn in os.listdir(p))
        has_h5 = any(fn.lower().endswith(".h5") for fn in os.listdir(p))
        if has_txt or has_h5:
            cand.append(name)

    if not cand:
        print(f"No local models found under {base_dir}")
        return

    if models is None:
        print("\nLocal models available for rebuild:")
        print("-" * 64)
        for i, m in enumerate(cand, 1):
            print(f"{i:3d}. {m}")
        print("\nSelect indices (comma / ranges like 3-6) or 'all':")
        raw = input("> ").strip().lower()
        if raw == "all":
            selected = cand
        else:
            idxs = []
            for token in raw.split(","):
                token = token.strip()
                if "-" in token:
                    a, b = token.split("-", 1)
                    idxs += list(range(int(a), int(b)+1))
                else:
                    if token:
                        idxs.append(int(token))
            selected = [cand[i-1] for i in idxs if 1 <= i <= len(cand)]
    else:
        selected = models

    if not selected:
        print("Nothing selected.")
        return

    for model_name in selected:
        print("\n" + "=" * 64)
        print(f"[rebuild] {model_name}")
        model_dir = os.path.join(base_dir, model_name)

        summary = clean_model_dir(model_dir, try_h5_recovery=True, backup=True, rebuild_lookup=True)
        print(f"[clean] {model_name}: total={summary['total']}")

        txts = glob.glob(os.path.join(model_dir, "*.txt"))
        if not txts:
            print(f"[rebuild] no spectra (.txt) present after cleaning; skipping {model_name}.")
            continue

        regenerate_lookup_table(model_dir)

        if rebuild_h5:
            out_h5 = os.path.join(model_dir, f"{model_name}.h5")
            build_h5_bundle_from_txt(model_dir, out_h5)

        if rebuild_flux_cube:
            out_flux = os.path.join(model_dir, "flux_cube.bin")
            precompute_flux_cube(model_dir, out_flux)

    print("\nRebuild complete.")






def _parse_range(raw: str):
    """Parse a user-entered range like '3500,8000' into (min, max) or None."""
    raw = raw.strip()
    if not raw:
        return None
    for sep in [',', ' ', '..', ':']:
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep, 1) if p.strip()]
            if len(parts) == 2:
                try:
                    lo, hi = float(parts[0]), float(parts[1])
                    if lo > hi:
                        lo, hi = hi, lo
                    return (lo, hi)
                except ValueError:
                    pass
    print(f"  Could not parse range from '{raw}' — need two values (e.g. '3500,8000')")
    return None


def _prompt_axis_cuts(name, grabber, model_url=None):
    """Prompt user for axis cuts on an NJM download. Returns dict of ranges."""
    cuts = {'teff_range': None, 'logg_range': None, 'meta_range': None, 'wl_range': None}

    print(f"\n  ── Axis Cuts for {name} ──")
    print(f"  Restrict the download to a subset of the grid.")
    print(f"  Leave blank to download everything.\n")

    info = grabber.get_grid_info(name, model_url=model_url)
    if info:
        print(f"  Available grid ({info['n_spectra']} spectra):")
        if 'teff_min' in info:
            print(f"    Teff:  {info['teff_min']:.0f} – {info['teff_max']:.0f} K  ({info['teff_unique']} values)")
        if 'logg_min' in info:
            print(f"    logg:  {info['logg_min']:.2f} – {info['logg_max']:.2f}    ({info['logg_unique']} values)")
        if 'meta_min' in info:
            print(f"    [M/H]: {info['meta_min']:+.2f} – {info['meta_max']:+.2f}    ({info['meta_unique']} values)")
        print()
    else:
        print("  (Could not fetch grid info)\n")

    cuts['teff_range'] = _parse_range(input("  Teff range (e.g. '3500,8000') or blank for all: "))
    cuts['logg_range'] = _parse_range(input("  logg range (e.g. '3.5,5.0') or blank for all: "))
    cuts['meta_range'] = _parse_range(input("  [M/H] range (e.g. '-1.0,0.5') or blank for all: "))
    cuts['wl_range']   = _parse_range(input("  Wavelength range in Å (e.g. '3000,10000') or blank for all: "))

    any_cuts = any(v is not None for v in cuts.values())
    if any_cuts:
        print(f"\n  Applied cuts:")
        if cuts['teff_range']:
            print(f"    Teff:       {cuts['teff_range'][0]:.0f} – {cuts['teff_range'][1]:.0f} K")
        if cuts['logg_range']:
            print(f"    logg:       {cuts['logg_range'][0]:.2f} – {cuts['logg_range'][1]:.2f}")
        if cuts['meta_range']:
            print(f"    [M/H]:      {cuts['meta_range'][0]:+.2f} – {cuts['meta_range'][1]:+.2f}")
        if cuts['wl_range']:
            print(f"    Wavelength: {cuts['wl_range'][0]:.1f} – {cuts['wl_range'][1]:.1f} Å")
    else:
        print("\n  No cuts — downloading full grid.")

    return cuts





def run_spectra_flow(
    source: str,
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models: Optional[List[str]] = None,
    workers: int = 5,
    force_bundle_h5: bool = True,
    build_flux_cube: bool = True
) -> None:
    ensure_dir(base_dir)
    
    # Parse source list - now includes 'njm'
    src_list = []
    if source.lower() == "all":
        src_list = ["njm", "svo", "msg", "mast"]  # NJM first!
    elif source.lower() == "both":
        src_list = ["njm", "svo", "msg"]
    else:
        if "," in source:
            src_list = [s.strip().lower() for s in source.split(",")]
        else:
            src_list = [source.lower()]

    # Initialize grabbers - NJM first to check availability
    grabs = {}
    if "njm" in src_list:
        grabs["njm"] = NJMSpectraGrabber(base_dir=base_dir, max_workers=workers)
        if not grabs["njm"].is_available():
            print("[njm] Mirror unavailable, using other sources")
            del grabs["njm"]
            if "njm" in src_list:
                src_list.remove("njm")
    
    if "svo" in src_list:
        grabs["svo"] = SVOSpectraGrabber(base_dir=base_dir, max_workers=workers)
    if "msg" in src_list:
        grabs["msg"] = MSGSpectraGrabber(base_dir=base_dir, max_workers=workers)
    if "mast" in src_list:
        grabs["mast"] = MASTSpectraGrabber(base_dir=base_dir, max_workers=workers)

    # Discover models from all sources
    model_sources = {}
    for s in src_list:
        if s in grabs:
            for model_name in grabs[s].discover_models():
                if model_name not in model_sources:
                    model_sources[model_name] = []
                model_sources[model_name].append(s)
    
    if not model_sources:
        print("No models discovered.")
        return

    # Build display options
    class ModelOption:
        def __init__(self, name, sources):
            self.name = name
            self.sources = sources
            source_tags = "".join(f"[{s}]" for s in sources)
            self.label = f"{name} {source_tags}"
    
    all_models = [ModelOption(name, sources) for name, sources in sorted(model_sources.items())]

    # Selection logic
    chosen = []
    if models is not None:
        if len(models) == 1 and models[0].lower() == "all":
            chosen = [(opt.sources, opt.name) for opt in all_models]
        else:
            for m in models:
                if ":" in m:
                    src, name = m.split(":", 1)
                    src, name = src.strip().lower(), name.strip()
                    chosen.append(([src], name))
                else:
                    name = m.strip()
                    matching = [opt for opt in all_models if opt.name == name]
                    if not matching:
                        print(f"[skip] Model '{name}' not found")
                        continue
                    chosen.append((matching[0].sources, name))
    else:
        idxs = _prompt_choice(all_models, label="Spectral models", allow_back=True, multi=True)
        
        if idxs is None:
            return
        if idxs == -1:
            print("No model selected.")
            return
            
        if isinstance(idxs, int):
            idxs = [idxs]

        chosen = [(all_models[i].sources, all_models[i].name) for i in idxs]

    # Download and process each selected model
    for sources, name in chosen:
        print("\\n" + "=" * 64)
        source_tags = "".join(f"[{s}]" for s in sources)
        print(f"{source_tags} {name}")
        print("=" * 64)
        
        model_dir = os.path.join(base_dir, name)
        ensure_dir(model_dir)

        # Try each source in order until one returns metadata
        meta = None
        g = None
        src = None
        for try_src in sources:
            g = grabs.get(try_src)
            if not g:
                continue
            meta = g.get_model_metadata(name)
            if meta:
                src = try_src
                break
            print(f"  [{try_src}] No metadata — trying next source...")
        
        if not meta or not g or not src:
            print(f"No metadata available from any source")
            continue
        
        print(f"  Using source: {src}")
        
        # Check if pre-processed (NJM)
        is_preprocessed = isinstance(meta, dict) and meta.get("pre_processed", False)

        # ── Axis cuts (NJM only) ──
        njm_cuts = {}
        if src == "njm":
            njm_cuts = _prompt_axis_cuts(name, g, model_url=meta.get("model_url"))

        n_written = g.download_model_spectra(
            name, meta,
            teff_range=njm_cuts.get('teff_range'),
            logg_range=njm_cuts.get('logg_range'),
            meta_range=njm_cuts.get('meta_range'),
            wl_range=njm_cuts.get('wl_range'),
        )
        print(f"Downloaded {n_written} spectra{model_dir}")

        # Skip cleaning for pre-processed NJM data
        if is_preprocessed:
            print(f"Pre-processed data (skipping cleaning)")
            essential = ["flux_cube.bin", "lookup_table.csv"]
            missing = [f for f in essential if not os.path.exists(os.path.join(model_dir, f))]
            if missing:
                print(f"Missing: {', '.join(missing)}")
            else:
                print(f"All essential files present")
            continue

        # --- Cleaning ---
        summary = clean_model_dir(model_dir, try_h5_recovery=True, backup=True, rebuild_lookup=True)
        
        # ─────────────────────────────────────────────────────────────
        # DETAILED REPORTING
        # ─────────────────────────────────────────────────────────────
        
        n_total = summary['total']
        
        # Get detection info
        det_stats = summary.get('detection_stats', {})
        cat_units = summary.get('catalog_units', {})
        
        # Determine detected units
        detected_wl = cat_units.get('wavelength', 'unknown') if cat_units else 'unknown'
        detected_flux = cat_units.get('flux', 'unknown') if cat_units else 'unknown'
        confidence = cat_units.get('confidence', 'unknown') if cat_units else 'unknown'
        
        # Check if already standard (no conversion needed)
        units_already_standard = (detected_wl == 'angstrom' and detected_flux == 'flam')
        
        # ── Unit Detection Report ──
        
        sample_size = det_stats.get('sample_size', '?')
        wl_agree = det_stats.get('wavelength_agreement', '?')
        fl_agree = det_stats.get('flux_agreement', '?')
        det_status = det_stats.get('status', '')
        
        if det_status == 'already_standardized':
            print(f" Method    : Sampled {sample_size}/{n_total} files")
            print(f" Result    : All samples already standardized")
            print(f" Wavelength: angstrom (Å)")
            print(f" Flux      : F_lambda (erg/s/cm²/Å)")
        elif cat_units:
            print(f" Method    : Catalog consensus ({sample_size}/{n_total} files sampled)")
            print(f" Wavelength: {detected_wl} ({wl_agree} agreement)")
            print(f" Flux      : {detected_flux} ({fl_agree} agreement)")
            print(f" Confidence: {confidence}")
        else:
            print(f" Status    : {det_status or 'No unit detection performed'}")
            print(f" Wavelength: {detected_wl}")
            print(f" Flux      : {detected_flux}")
        
        # ── Processing Report ──
        # Extract counts - handle both old and new key names
        n_converted = len(summary.get('converted', []))
        n_tagged = len(summary.get('tagged', []))  # new: header-only tagging
        n_recovered = len(summary.get('recovered', []))
        n_skipped = len(summary.get('skipped_already', []) or summary.get('skipped', []))
        n_invalid = len(summary.get('skipped_invalid', []) or summary.get('invalid', []))
        n_index = len(summary.get('skipped_index', []))
        n_error = len(summary.get('error', []))
        
        # If 'converted' is used but units were already standard, treat as 'tagged'
        if n_converted > 0 and units_already_standard:
            n_tagged = n_converted
            n_converted = 0
        
        print(f" Total files: {n_total}")
        
        # Show what actually happened
        if n_converted > 0:
            # Actual unit conversion occurred
            src_wl = detected_wl if detected_wl != 'angstrom' else 'original'
            src_fl = detected_flux if detected_flux != 'flam' else 'original'
            print(f"Converted     : {n_converted} files")
            if detected_wl != 'angstrom':
                print(f"   λ: {detected_wl}angstrom")
            if detected_flux != 'flam':
                print(f"   F: {detected_flux}F_lambda")
        
        if n_tagged > 0:
            # No conversion, just cleaning + header tagging
            print(f"Cleaned       : {n_tagged} files")
            print(f"   (units already Å + F_λ, added standardized tag)")
        
        if n_recovered > 0:
            print(f"Recovered     : {n_recovered} files")
            print(f"   (wavelengths restored from HDF5)")
        
        if n_skipped > 0:
            print(f" - Already done  : {n_skipped} files")
            print(f"   (had units_standardized header)")
        
        if n_invalid > 0:
            print(f"Invalid       : {n_invalid} files")
            print(f"   (empty, corrupt, or λ ≤ 0)")
        
        if n_index > 0:
            print(f"Index grids   : {n_index} files")
            print(f"   (λ=0,1,2... without HDF5 source)")
        
        if n_error > 0:
            print(f" Errors        : {n_error} files")
        
        # Summary line
        n_usable = n_converted + n_tagged + n_recovered + n_skipped
        print()
        if n_usable == n_total:
            print(f"All {n_usable} files ready for use")
        elif n_usable > 0:
            print(f"{n_usable}/{n_total} files usable")
        else:
            print(f"No usable files")

        if not glob.glob(os.path.join(model_dir, "*.txt")):
            print(f"\n  No spectra remaining after cleaning")
            continue

        # ── Data Products ──
        
        # HDF5 Bundle
        if src == "msg":
            out_h5 = os.path.join(model_dir, f"{name}_bundle.h5")
        else:
            out_h5 = os.path.join(model_dir, f"{name}.h5")
        
        if force_bundle_h5 or not os.path.exists(out_h5):
            build_h5_bundle_from_txt(model_dir, out_h5)
            print(f"HDF5 bundle   : {os.path.basename(out_h5)}")
        else:
            print(f" - HDF5 bundle   : {os.path.basename(out_h5)} (exists)")

        # Lookup Table
        regenerate_lookup_table(model_dir)
        print(f"Lookup table  : lookup_table.csv")

        # Flux Cube
        if build_flux_cube:
            precompute_flux_cube(model_dir, os.path.join(model_dir, "flux_cube.bin"))
            print(f"Flux cube     : flux_cube.bin")

    print("\n" + "─" * 64)
    print("Done.")



def run_filters_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> None:
    """
    Interactive filter downloader with automatic NJMSVO fallback.
    
    Shows full SVO catalog, user selects what they want, then automatically
    downloads from NJM mirror if available, falling back to SVO if not.
    """
    ensure_dir(base_dir)
    
    from .njm_filter_grabber import NJMFilterGrabber
    from .svo_filter_grabber import SVOFilterBrowser

    # Initialize both sources
    svo = SVOFilterBrowser(base_dir=base_dir)
    njm = NJMFilterGrabber(base_dir=base_dir)
    njm_available = njm.is_available()
    
    # Browse SVO catalog (authoritative source)
    print("\nBrowsing SVO Filter Profile Service...")
    
    facilities = svo.list_facilities()
    if not facilities:
        print("No facilities found.")
        return
    
    # Facility selection loop
    while True:
        fac_idx = _prompt_choice(facilities, "Filter Facilities")
        if fac_idx is None:
            return
        
        facility = facilities[fac_idx]
        instruments = svo.list_instruments(facility.key)
        
        if not instruments:
            print(f"No instruments found for {facility.label}.")
            continue
        
        # Instrument selection loop
        while True:
            inst_idx = _prompt_choice(
                instruments,
                f"Instruments for {facility.label}",
                allow_back=True
            )
            
            if inst_idx is None:
                return
            if inst_idx == -1:
                break  # Back to facilities
            
            instrument = instruments[inst_idx]
            filters = svo.list_filters(facility.key, instrument.key)
            
            if not filters:
                print(f"No filters found for {instrument.label}.")
                continue
            
            # Show selection
            print(f"\n{facility.label} / {instrument.label}")
            print(f"Found {len(filters)} filters")
            
            confirm = input("Download all filters? [Y/n] ").strip().lower()
            if confirm and not confirm.startswith('y'):
                continue
            
            # Smart download: Try NJM first, fall back to SVO
            downloaded = False
            
            if njm_available:
                # Check if NJM has this facility/instrument
                njm_facilities = njm.discover_facilities()
                if facility.key in njm_facilities:
                    njm_instruments = njm.discover_instruments(facility.key)
                    if instrument.key in njm_instruments:
                        # NJM has it - download from there
                        print(f"\n[njm] Downloading from mirror...")
                        count = njm.download_filters(facility.key, instrument.key)
                        if count > 0:
                            print(f"[njm]  Downloaded {count} filters")
                            downloaded = True
            
            # Fall back to SVO if NJM didn't work
            if not downloaded:
                print(f"\n[svo] Downloading from SVO...")
                svo.download_filters(filters)
                print(f"[svo]  Downloaded {len(filters)} filters")
            
            again = input("\nDownload another instrument? [y/N] ").strip().lower()
            if not again.startswith('y'):
                break


def run_ml_generator_flow(
    base_dir: str = STELLAR_DIR_DEFAULT,
    models_dir: str = "models"
) -> None:
    """Run ML SED Generator interactive workflow."""
    from .ml_sed_generator import run_interactive_workflow
    run_interactive_workflow(base_dir=base_dir, models_dir=models_dir)





def menu() -> str:
    print("\nWhat would you like to run?")
    print("1) Spectra (NJM / SVO / MSG / MAST)")
    print("2) Filters (NJM / SVO)")
    print("3) Rebuild (lookup + HDF5 + flux cube)")
    print("4) Combine grids into omni grid")
    print("5) ML SED Completer (train/extend incomplete SEDs)")
    print("6) ML SED Generator (generate SEDs from parameters)")  # NEW
    print("0) Quit")
    choice = input("> ").strip()
    mapping = {
        "1": "spectra", 
        "2": "filters", 
        "3": "rebuild",
        "4": "combine",
        "5": "ml_completer",
        "6": "ml_generator",  # NEW
        "0": "quit"
    }
    return mapping.get(choice, "")


#!/usr/bin/env python3
"""
ADD THIS FUNCTION TO sed_tools/__init__.py or sed_tools/cli.py

Place it after run_rebuild_flow() function.
"""



def run_combine_flow(
    base_dir: str = STELLAR_DIR_DEFAULT,
    output_name: str = "combined_models",
    interactive: bool = True
) -> None:
    """
    Combine multiple stellar atmosphere grids into unified omni grid.

    This creates a single flux cube spanning the parameter space of all
    selected models.

    Args:
        base_dir: Base directory containing stellar_models/ subdirectories
        output_name: Name for the output combined model directory
        interactive: If True, prompt user to select models
    """
    from .combine_stellar_atm import (build_combined_flux_cube,
                                      create_common_wavelength_grid,
                                      create_unified_grid, find_stellar_models,
                                      load_model_data, save_combined_data,
                                      select_models_interactive,
                                      visualize_parameter_space)

    ensure_dir(base_dir)

    model_dirs = find_stellar_models(base_dir)
    if not model_dirs:
        print(f"No stellar models found in {base_dir}")
        return

    selected_models = select_models_interactive(model_dirs) if interactive else model_dirs
    if not selected_models:
        print("No models selected.")
        return


    if interactive:
        output_name = input(f"What should the combined model be called? [{output_name}]") or output_name

    print("\nModel will be saved as:", output_name, "\n")


    print(f"\nSelected {len(selected_models)} models to combine:")
    for name, _ in selected_models:
        print(f"- {name}")

    print("\nLoading model data...")
    all_models_data = []
    for name, path in selected_models:
        print(f"Loading {name}...")
        all_models_data.append(load_model_data(path))

    print("\nCreating unified parameter grids...")
    teff_grid, logg_grid, meta_grid = create_unified_grid(all_models_data)
    wavelength_grid = create_common_wavelength_grid(all_models_data)

    flux_cube, source_map = build_combined_flux_cube(
        all_models_data, teff_grid, logg_grid, meta_grid, wavelength_grid
    )

    output_dir = os.path.join(base_dir, output_name)
    save_combined_data(
        output_dir,
        teff_grid,
        logg_grid,
        meta_grid,
        wavelength_grid,
        flux_cube,
        all_models_data,
    )

    visualize_parameter_space(
        teff_grid, logg_grid, meta_grid, source_map, all_models_data, output_dir,
        wavelength_grid=wavelength_grid, flux_cube=flux_cube
    )
    
    print(f"\nSuccessfully combined {len(selected_models)} stellar atmosphere models!")
    print(f"Output saved to: {output_dir}")
    print("You can now use this combined model in MESA by setting:")
    print(f"stellar_atm = '{output_dir}/'")


def run_ml_completer_flow(
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models_dir: str = "models"
) -> None:
    """
    Launch the interactive ML SED Completer workflow.
    
    This is a thin wrapper that delegates to ml_sed_completer.run_interactive_workflow().
    All training and inference logic lives in the ml_sed_completer module.
    """
    from .ml_sed_completer import run_interactive_workflow
    run_interactive_workflow(base_dir, models_dir)





# ------------ Main CLI ------------

def main():
    parser = argparse.ArgumentParser(description="SED Tools — spectra & filters")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # spectra
    sp = sub.add_parser("spectra", help="Download/build spectra products")
    sp.add_argument("--source", 
                    choices=["njm", "svo", "msg", "mast", "both", "all"], 
                    default="all",
                    help="Which provider(s) to use. NJM mirror is checked first when available.")
    sp.add_argument("--models", nargs="*", default=None,
                    help="Model names to process.")
    sp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Output base for spectra")
    sp.add_argument("--workers", type=int, default=5, help="Parallel workers")
    sp.add_argument("--no-h5", action="store_true", help="Do not force-create HDF5 bundle")
    sp.add_argument("--no-cube", action="store_true", help="Skip flux cube build")

    # filters
    fp = sub.add_parser("filters", help="Download SVO filters")
    fp.add_argument("--base", default=str(FILTER_DIR_DEFAULT),
                    help="Output base for filters")

    # rebuild
    rp = sub.add_parser("rebuild", help="Rebuild lookup table + HDF5 + flux cube")
    rp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base models dir")
    rp.add_argument("--models", nargs="*", default=None,
                    help="Names of local model folders to rebuild")
    rp.add_argument("--no-h5", action="store_true", help="Skip HDF5 bundle")
    rp.add_argument("--no-cube", action="store_true", help="Skip flux cube")


    # combine
    cp = sub.add_parser("combine", help="Combine multiple grids into omni grid")
    cp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base models directory")
    cp.add_argument("--output", default="combined_models",
                    help="Name for output combined model")
    cp.add_argument("--non-interactive", action="store_true",
                    help="Combine all models without prompting")

    # ml_completer
    mp = sub.add_parser("ml_completer", help="Train/use ML SED completer")
    mp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base models directory")
    mp.add_argument("--models-dir", default="models",
                    help="Directory for trained ML models")


    # ml_generator
    gp = sub.add_parser("ml_generator", help="Train/use ML SED generator")
    gp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base models directory")
    gp.add_argument("--models-dir", default="models",
                    help="Directory for trained ML models")
    gp.add_argument("--auto", action="store_true",
                    help="Run optimized auto-training with hyperparameter search")
    gp.add_argument("--library", help="Library path for auto-training")
    gp.add_argument("--output", help="Output directory for auto-training")


    args = parser.parse_args()

    if args.cmd == "spectra":
        run_spectra_flow(
            source=args.source,
            base_dir=args.base,
            models=args.models,
            workers=args.workers,
            force_bundle_h5=not args.no_h5,
            build_flux_cube=not args.no_cube
        )
    elif args.cmd == "filters":
        run_filters_flow(base_dir=args.base)
    elif args.cmd == "rebuild":
        run_rebuild_flow(
            base_dir=args.base,
            models=args.models,
            rebuild_h5=not args.no_h5,
            rebuild_flux_cube=not args.no_cube
        )

    elif args.cmd == "combine":
        run_combine_flow(
            base_dir=args.base,
            output_name=args.output,
            interactive=not args.non_interactive
        )

    elif args.cmd == "ml_completer":
        run_ml_completer_flow(
            base_dir=args.base,
            models_dir=args.models_dir
        )


    elif args.cmd == "ml_generator":
        if args.auto:
            if not args.library or not args.output:
                print("Error: --library and --output are required for --auto training.")
                sys.exit(1)
            from .ml import auto_train_generator
            auto_train_generator(library=args.library, output=args.output)
        else:
            run_ml_generator_flow(
                base_dir=args.base,
                models_dir=args.models_dir
            )


    else:
        # Interactive mode
        while True:
            choice = menu()
            if choice == "filters":
                run_filters_flow()
            elif choice == "rebuild":
                run_rebuild_flow()
            elif choice == "spectra":
                run_spectra_flow(source="all")
            elif choice == "combine":
                run_combine_flow()
            elif choice == "ml_completer":
                run_ml_completer_flow()
            elif choice == "ml_generator":
                run_ml_generator_flow()                
            else:
                sys.exit(0)


if __name__ == "__main__":
    main()
