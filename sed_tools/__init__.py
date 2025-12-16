"""High level API for working with pre-computed stellar SED libraries.

This module exposes the :class:`SED` facade together with helper classes that
wrap the existing command line tooling in a programmatic friendly interface.
Users can discover locally available model grids, load a specific grid into
memory and evaluate spectra or synthetic photometry directly from Python code.

Typical usage::

    from sed_tools import SED

    sed = SED()                             # discover default data directories
    matches = sed.find_model(5777, 4.44, 0) # inspect available grids
    model = sed.model(matches[0].name)
    spec = model(5777, 4.44, 0.0)           # interpolate a spectrum
    gaia = spec.photometry("GAIA")         # synthetic GAIA magnitudes

For range-based discovery a convenience wrapper is also provided::

    from sed_tools import find_atmospheres

    grids = find_atmospheres(teff_range=(5000, 6500), metallicity_range=(-0.5, 0.5))

The package reuses the heavy lifting that already powers the interactive tools
shipped with SED Tools so that workflows built on the CLI continue to operate
unchanged while pipelines can opt into the same functionality via imports.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import os
import sys
import h5py
import numpy as np

from .models import (
    SED,
    SEDModel,
    EvaluatedSED,
    PhotometryResult,
    ModelMatch,
    DATA_DIR_DEFAULT,
    STELLAR_DIR_DEFAULT,
    FILTER_DIR_DEFAULT,
)

from .svo_spectra_grabber import SVOSpectraGrabber  # SVO spectra → .txt + lookup_table.csv
from .msg_spectra_grabber import MSGSpectraGrabber  # MSG (Townsend) .h5 → .txt + lookup_table.csv
from .njm_spectra_grabber import NJMSpectraGrabber  # Niall J Miller
from .precompute_flux_cube import precompute_flux_cube  # builds flux cube from lookup + .txt
from .svo_regen_spectra_lookup import parse_metadata, regenerate_lookup_table
from .mast_spectra_grabber import MASTSpectraGrabber
from .spectra_cleaner import clean_model_dir
from .njm_filter_grabber import NJMFilterGrabber

__version__ = "0.1.0"


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
                try:
                    wl.append(float(parts[0]))
                    fl.append(float(parts[1]))
                except ValueError:
                    continue
    return np.asarray(wl, dtype=float), np.asarray(fl, dtype=float)

def numeric_from(meta: Dict[str, str], key_candidates: List[str], default: float = np.nan) -> float:
    """Extract first numeric token from any of the candidate keys (case-insensitive)."""
    lower = {k.lower(): v for k, v in meta.items()}
    for ck in key_candidates:
        if ck.lower() in lower:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lower[ck.lower()])
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    pass
    return default
# ------------ HDF5 bundling ------------

def build_h5_bundle_from_txt(model_dir: str, out_h5: str) -> None:
    """
    Create an HDF5 file bundling all .txt spectra under groups:
      /spectra/<filename>/lambda (float64)
      /spectra/<filename>/flux   (float64)
    And store key metadata as HDF5 attributes on each group:
      attrs: teff, logg, feh (if found), plus all raw header pairs in attrs["raw:<key>"]
    """
    txt_files = list_txt_spectra(model_dir)
    if not txt_files:
        print(f"[H5 bundle] No .txt spectra found in {model_dir}; skipping.")
        return

    ensure_dir(os.path.dirname(out_h5))
    with h5py.File(out_h5, "w") as h5:
        spectra_grp = h5.create_group("spectra")
        for fname in txt_files:
            path = os.path.join(model_dir, fname)
            try:
                wl, fl = load_txt_spectrum(path)
                if wl.size == 0 or fl.size == 0:
                    print(f"[H5 bundle] Empty or invalid spectrum: {fname}")
                    continue
                g = spectra_grp.create_group(fname)
                g.create_dataset("lambda", data=wl, dtype="f8")
                g.create_dataset("flux",   data=fl, dtype="f8")

                meta = parse_metadata(path)
                # canonical numeric attrs
                teff = numeric_from(meta, ["Teff", "teff", "T_eff"])
                logg = numeric_from(meta, ["logg", "Logg", "log_g"])
                feh  = numeric_from(meta, ["FeH", "feh", "metallicity", "[Fe/H]", "meta"])
                if not np.isnan(teff): g.attrs["teff"] = teff
                if not np.isnan(logg): g.attrs["logg"] = logg
                if not np.isnan(feh):  g.attrs["feh"]  = feh
                # stash raw header lines so nothing is lost
                for k, v in meta.items():
                    g.attrs[f"raw:{k}"] = v
            except Exception as e:
                print(f"[H5 bundle] Error on {fname}: {e}")

    print(f"[H5 bundle] Wrote {out_h5}")

def run_rebuild_flow(base_dir: str = STELLAR_DIR_DEFAULT,
                     models: List[str] = None,
                     rebuild_h5: bool = True,
                     rebuild_flux_cube: bool = True) -> None:
    """
    Rebuild lookup_table.csv (+ optional HDF5 bundle and flux cube)
    for existing local model directories. Now cleans spectra first.
    """
    import glob
    ensure_dir(base_dir)

    # discover local model dirs by presence of .txt or .h5
    cand = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p):
            continue
        has_txt = any(fn.lower().endswith(".txt") for fn in os.listdir(p))
        has_h5  = any(fn.lower().endswith(".h5") for fn in os.listdir(p))
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

    # optional SVO lookup regen helper
    regen = None
    try:
        from .svo_regen_spectra_lookup import regenerate_lookup_table as regen  # type: ignore
    except Exception:
        regen = None

    # cleaner
    try:
        from .spectra_cleaner import clean_model_dir
    except Exception as e:
        clean_model_dir = None
        print(f"[clean] cleaner unavailable: {e}")

    for model_name in selected:
        print("\n" + "="*64)
        print(f"[rebuild] {model_name}")
        model_dir = os.path.join(base_dir, model_name)

        # 0) CLEAN FIRST (fix λ<=0, repair index grids via HDF5 when possible)
        if clean_model_dir:
            try:
                summary = clean_model_dir(model_dir, try_h5_recovery=True, backup=True, rebuild_lookup=True)
                print(f"[clean] {model_name}: total={summary['total']}, "
                      f"fixed={len(summary['fixed'])}, recovered={len(summary['recovered'])}, "
                      f"skipped={len(summary['skipped'])}, deleted={len(summary['deleted'])}")
            except Exception as e:
                print(f"[clean] failed for {model_name}: {e}")

        # If no .txt remain, skip this model
        txts = glob.glob(os.path.join(model_dir, "*.txt"))
        if not txts:
            print(f"[rebuild] no spectra (.txt) present after cleaning; skipping {model_name}.")
            continue

        # 1) lookup table
        try:
            if regen:
                regen(model_dir)
                print("[rebuild] lookup_table.csv via svo_regen_spectra_lookup")
            else:
                regenerate_lookup_table(model_dir)
        except Exception as e:
            print(f"[rebuild] lookup failed: {e}")

        # 2) HDF5 bundle (from .txt), ensure it exists
        if rebuild_h5:
            out_h5 = os.path.join(model_dir, f"{model_name}.h5")
            try:
                build_h5_bundle_from_txt(model_dir, out_h5)
            except Exception as e:
                print(f"[rebuild] h5 bundle failed: {e}")

        # 3) flux cube
        if rebuild_flux_cube:
            out_flux = os.path.join(model_dir, "flux_cube.bin")
            try:
                precompute_flux_cube(model_dir, out_flux)
            except Exception as e:
                print(f"[rebuild] flux cube failed: {e}")

    print("\nRebuild complete.")

from typing import List, Tuple
import os, glob
# assumes in scope:
# ensure_dir, STELLAR_DIR_DEFAULT
# SVOSpectraGrabber, MSGSpectraGrabber, MASTSpectraGrabber
# build_h5_bundle_from_txt, regenerate_lookup_table, precompute_flux_cube
# and: from SED_tools.cli import _prompt_choice

def _srcs(s: str) -> List[str]:
    s = s.lower()
    return {"svo":["svo"], "msg":["msg"], "mast":["mast"], "both":["svo","msg"], "all":["svo","msg","mast"]}[s]

class _Opt:
    __slots__ = ("src","name","label")
    def __init__(self, src, name):
        self.src, self.name = src, name
        self.label = f"{name} [{src}]"

def run_spectra_flow(
    source: str,
    base_dir: str = STELLAR_DIR_DEFAULT,
    models: List[str] = None,
    workers: int = 5,
    force_bundle_h5: bool = True,
    build_flux_cube: bool = True
) -> None:
    from .spectra_cleaner import clean_model_dir

    ensure_dir(base_dir)
    src_list = _srcs(source)

    grabs = {}
    if "svo" in src_list:  grabs["svo"]  = SVOSpectraGrabber(base_dir=base_dir, max_workers=workers)
    if "msg" in src_list:  grabs["msg"]  = MSGSpectraGrabber(base_dir=base_dir, max_workers=workers)
    if "mast" in src_list: grabs["mast"] = MASTSpectraGrabber(base_dir=base_dir, max_workers=workers)

    discovered: List[Tuple[str,str]] = [(s, m) for s in src_list for m in grabs[s].discover_models()]
    if models is None and not discovered:
        print("No models discovered."); return

    if models is not None:
        if len(models)==1 and models[0].lower()=="all":
            chosen = discovered
        else:
            chosen = []
            for m in models:
                if ":" in m:
                    src, name = m.split(":",1); src, name = src.strip().lower(), name.strip()
                else:
                    if len(src_list)!=1: raise ValueError(f"Ambiguous '{m}' with source='{source}'. Use 'src:model'.")
                    src, name = src_list[0], m.strip()
                if (src,name) not in discovered: raise ValueError(f"Model '{name}' not found in '{src}'.")
                chosen.append((src,name))
    else:
        opts = [_Opt(s,n) for s,n in discovered]
        idx = _prompt_choice(opts, label="Spectral models", allow_back=True)
        if idx is None or idx==-1: print("No model selected."); return
        sel = opts[idx]; chosen = [(sel.src, sel.name)]

    for src, name in chosen:
        print("\n"+"="*64); print(f"[{src}] {name}")
        model_dir = os.path.join(base_dir, name); ensure_dir(model_dir)

        g = grabs[src]
        meta = g.get_model_metadata(name)
        if not meta: print(f"[{src}] No metadata for {name}; skip."); continue
        n_written = g.download_model_spectra(name, meta)
        print(f"[{src}] wrote {n_written} spectra -> {model_dir}")

        summary = clean_model_dir(model_dir, try_h5_recovery=True, backup=True, rebuild_lookup=True)
        print(f"[clean] total={summary['total']} fixed={len(summary['fixed'])} "
              f"recovered={len(summary['recovered'])} skipped={len(summary['skipped'])} "
              f"deleted={len(summary['deleted'])}")

        if not glob.glob(os.path.join(model_dir, "*.txt")):
            print(f"[{src}] no .txt after cleaning; skip downstream."); continue

        if src == "msg":
            out_h5 = os.path.join(model_dir, f"{name}_bundle.h5")
            if force_bundle_h5 and not os.path.exists(out_h5):
                build_h5_bundle_from_txt(model_dir, out_h5)
        else:
            out_h5 = os.path.join(model_dir, f"{name}.h5")
            if force_bundle_h5 or not os.path.exists(out_h5):
                build_h5_bundle_from_txt(model_dir, out_h5)

        print("[lookup] rebuilding lookup_table.csv"); regenerate_lookup_table(model_dir)

        if build_flux_cube:
            precompute_flux_cube(model_dir, os.path.join(model_dir,"flux_cube.bin"))

    print("\nDone.")


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
    selected models, normalized to a reference (preferably Kurucz).
    
    Args:
        base_dir: Base directory containing stellar_models/ subdirectories
        output_name: Name for the output combined model directory  
        interactive: If True, prompt user to select models
    """
    from .combine_stellar_atm import (
        find_stellar_models,
        select_models_interactive,
        load_model_data,
        create_unified_grid,
        create_common_wavelength_grid,
        build_combined_flux_cube,
        save_combined_data,
        visualize_parameter_space
    )
    
    ensure_dir(base_dir)
    
    # Find available models
    model_dirs = find_stellar_models(base_dir)
    
    if not model_dirs:
        print(f"No stellar models found in {base_dir}")
        print("Download some models first using option 1 (Spectra)")
        return
    
    print(f"\nFound {len(model_dirs)} stellar atmosphere models")
    
    # Select models to combine
    if interactive:
        selected = select_models_interactive(model_dirs)
    else:
        selected = model_dirs
    
    if not selected:
        print("No models selected.")
        return
    
    if len(selected) < 2:
        print("Warning: Need at least 2 models for meaningful combination.")
        if interactive:
            confirm = input("Continue with 1 model? [y/N] ").strip().lower()
            if not confirm.startswith('y'):
                return
    
    print(f"\nSelected {len(selected)} models to combine:")
    for name, _ in selected:
        print(f"  - {name}")
    
    # Get output name
    if interactive:
        user_output = input(f"\nOutput directory name [{output_name}]: ").strip()
        if user_output:
            output_name = user_output
    
    # Load model data
    print("\nLoading model data...")
    all_models = []
    for name, path in selected:
        print(f"  Loading {name}...")
        all_models.append(load_model_data(path))
    
    # Create unified grids
    print("\nCreating unified parameter grids...")
    teff_grid, logg_grid, meta_grid = create_unified_grid(all_models)
    wavelength_grid = create_common_wavelength_grid(all_models)
    
    # Show grid info
    total_points = len(teff_grid) * len(logg_grid) * len(meta_grid)
    print(f"\nUnified grid dimensions:")
    print(f"  Teff:  {len(teff_grid)} points  " +
          f"({teff_grid.min():.0f} - {teff_grid.max():.0f} K)")
    print(f"  log g: {len(logg_grid)} points  " +
          f"({logg_grid.min():.2f} - {logg_grid.max():.2f})")
    print(f"  [M/H]: {len(meta_grid)} points  " +
          f"({meta_grid.min():.2f} - {meta_grid.max():.2f})")
    print(f"  λ:     {len(wavelength_grid)} points " +
          f"({wavelength_grid.min():.0f} - {wavelength_grid.max():.0f} Å)")
    print(f"\nTotal grid points: {total_points:,}")
    
    # Confirm before proceeding
    if interactive:
        confirm = input("\nProceed with building combined flux cube? [Y/n] ").strip().lower()
        if confirm and not confirm.startswith('y'):
            print("Cancelled.")
            return
    
    # Build combined flux cube
    print("\nBuilding combined flux cube...")
    print("(This may take several minutes for large grids)")
    flux_cube, source_map = build_combined_flux_cube(
        all_models, teff_grid, logg_grid, meta_grid, wavelength_grid
    )
    
    # Save combined data
    output_dir = os.path.join(base_dir, output_name)
    print(f"\nSaving combined model to: {output_dir}")
    save_combined_data(
        output_dir,
        teff_grid,
        logg_grid, 
        meta_grid,
        wavelength_grid,
        flux_cube,
        all_models
    )
    
    # Create visualizations
    print("\nCreating parameter space visualizations...")
    visualize_parameter_space(
        teff_grid, logg_grid, meta_grid, source_map, all_models, output_dir
    )
    
    # Success message
    print("\n" + "="*70)
    print("SUCCESS: Omni grid created!")
    print("="*70)
    print(f"\nYour combined model is ready at:")
    print(f"  {output_dir}")
    print(f"\nContents:")
    print(f"  • flux_cube.bin - MESA-ready flux cube")
    print(f"  • lookup_table.csv - Combined model inventory")
    print(f"  • parameter_space_visualization.png - Coverage maps")
    print(f"  • Original spectra from all source models")
    print(f"\nTo use in MESA, set in your inlist:")
    print(f"  stellar_atm = '/colors/data/stellar_models/{output_name}/'")
    print()





def menu() -> str:
    print("\nWhat would you like to run?")
    print("  1) Spectra (NJM / SVO / MSG / MAST)")
    print("  2) Filters (NJM / SVO)")
    print("  3) Rebuild (lookup + HDF5 + flux cube)")
    print("  4) Combine grids into omni grid")  # ← ADD THIS LINE
    print("  0) Quit")
    choice = input("> ").strip()
    mapping = {
        "1": "spectra", 
        "2": "filters", 
        "3": "rebuild",
        "4": "combine",  # ← ADD THIS LINE
        "0": "quit"
    }
    return mapping.get(choice, "")


def run_filters_flow(base_dir: str = FILTER_DIR_DEFAULT) -> None:
    """Interactive nested filter downloader that mirrors the spectra workflow."""
    ensure_dir(base_dir)
    try:
        from .svo_filter_grabber import run_interactive as _run_filter_cli
    except Exception as exc:
        print("This feature needs requests/bs4/astroquery available:", exc)
        return

    _run_filter_cli(base_dir)



def find_atmospheres(
    *,
    teff_range: Optional[Sequence[float]] = None,
    logg_range: Optional[Sequence[float]] = None,
    metallicity_range: Optional[Sequence[float]] = None,
    limit: Optional[int] = None,
    allow_partial: bool = False,
    model_root: Optional[Union[str, os.PathLike[str]]] = None,
    filter_root: Optional[Union[str, os.PathLike[str]]] = None,
) -> list[ModelMatch]:
    """Discover locally available model grids matching the requested ranges."""

    sed = SED(model_root=model_root, filter_root=filter_root)
    return sed.find_atmospheres(
        teff_range=teff_range,
        logg_range=logg_range,
        metallicity_range=metallicity_range,
        limit=limit,
        allow_partial=allow_partial,
    )


def find_atm(**kwargs) -> list[ModelMatch]:
    """Alias for :func:`find_atmospheres` matching the historical API sketch."""

    for legacy_key in ("Z_range", "z_range"):
        if legacy_key in kwargs and "metallicity_range" not in kwargs:
            kwargs["metallicity_range"] = kwargs.pop(legacy_key)
    if "logg_range" not in kwargs and "log_g_range" in kwargs:
        kwargs["logg_range"] = kwargs.pop("log_g_range")
    return find_atmospheres(**kwargs)


__all__ = [
    "SED",
    "SEDModel",
    "EvaluatedSED",
    "PhotometryResult",
    "ModelMatch",    
    "run_combine_flow",  # ← ADD THIS
    "DATA_DIR_DEFAULT",
    "STELLAR_DIR_DEFAULT",
    "FILTER_DIR_DEFAULT",
    "find_atmospheres",
    "find_atm",
    "__version__",
]









from typing import Sequence, Optional, Any, List, Tuple
import re, math, shutil, sys, os

def _prompt_choice(
    options: Sequence,
    label: str,
    allow_back: bool = False,
    page_size: int = 100,
    max_label: int = -1,
    min_cols: int = 1,
    max_cols: int = 3,
    use_color: bool = True,
) -> Optional[int]:
    """
    Plain-ASCII picker with stable IDs, paging, grid columns, and simple filters.
    - Stable IDs: 1..N (do not renumber after filtering/paging)
    - Pagination: n / p / g <page>
    - Filters: /text (substring), !text (negated), //regex (case-insensitive)
    - Select by ID: 'id <N>' or just '<N>'
    Returns 0-based index, -1 for back (if allowed), None for quit.
    """
    if not options:
        print(f"No {label} options available.")
        return None


    if max_label < 0:
        max_label = max(len(getattr(x, "label", str(x))) for x in options) + 4


    # ---- setup ----
    labels: List[str] = [getattr(o, "label", str(o)) for o in options]
    N = len(labels)
    page = 1
    filt: Optional[Tuple[str, str]] = None  # ('substr'|'neg'|'regex', pattern)

    # color toggles
    use_color = use_color and sys.stdout.isatty() and ("NO_COLOR" not in os.environ)
    BOLD = "\x1b[1m" if use_color else ""
    DIM = "\x1b[2m" if use_color else ""
    CYAN = "\x1b[36m" if use_color else ""
    YELL = "\x1b[33m" if use_color else ""
    RED = "\x1b[31m" if use_color else ""
    GREEN = "\x1b[32m" if use_color else ""
    YELLOW = "\x1b[33m" if use_color else ""
    BLUE = "\x1b[34m" if use_color else ""
    MAGENTA = "\x1b[35m" if use_color else ""
    WHITE = "\x1b[37m" if use_color else ""
    RESET = "\x1b[0m" if use_color else ""

    def term_width() -> int:
        try: return shutil.get_terminal_size().columns
        except: return 80

    def apply_filter(idx: List[int]) -> List[int]:
        if filt is None: return idx
        kind, patt = filt
        if kind == "substr":
            p = patt.lower()
            return [i for i in idx if p in labels[i].lower()]
        if kind == "neg":
            p = patt.lower()
            return [i for i in idx if p not in labels[i].lower()]
        # regex
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
        col_w = 6 + max_label + 2     # "[####] " + label + gap
        cols = max(min_cols, min(max_cols, max(1, width // col_w)))
        cells = [f"[{GREEN}{i+1:4d}{RESET}] {hl(ellipsize(s))}" for i, s in zip(visible_ids, names)]
        while len(cells) % cols: cells.append("")
        rows = [cells[k:k+cols] for k in range(0, len(cells), cols)]
        print(f"\n{BOLD}{label}{RESET} ({CYAN}{len(all_idx)}{RESET} total):")
        print("─" * min(80, width))
        for r in rows:
            print("  ".join(x.ljust(col_w - 2) for x in r))

    # ---- loop ----
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
        if end < len(kept):
            print(f"{DIM}Showing {start}–{end} of {len(kept)}{ftxt}{RESET}")
            controls = f"{DIM}n, p, g <page>, /text, !text, //regex, id <N> (or just N), clear"
            if allow_back: controls += ", b"
        else:
            controls = f"{DIM}/text, !text, //regex, id <N> (or just N), clear"
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
        if inp.isdigit():
            k = int(inp)
            if 1 <= k <= N: return k - 1
            continue

        # default: substring filter
        filt = ("substr", inp); page = 1








