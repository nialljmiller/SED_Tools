#!/usr/bin/env python3
"""
SED_tools.py — one-stop launcher for spectra & filters

Pipeline per selected model:
  - SVO or MSG (Rich Townsend) model discovery
  - Download (SVO) or extract (MSG) spectra -> *.txt collection
  - Ensure lookup_table.csv
  - Build/ensure HDF5 bundle of spectra (for SVO we create one; for MSG we keep the original .h5 + still write a bundle for consistency)
  - Precompute flux cube file

Filters (SVO only):
  - Interactive substring filter over Facility/Instrument/Band (or 'all')
  - Saves to data/filters/<Facility>/<Instrument>/<Band>.dat

Defaults:
  STELLAR_DIR = data/stellar_models/
  FILTER_DIR  = data/filters/

Dependencies (already used in your codebase):
  requests, bs4, h5py, numpy, tqdm, pandas, astroquery, astropy
"""

import argparse
import os
import sys
import re
from typing import List, Dict, Any, Tuple

# local modules you already have
from svo_spectra_grabber import SVOSpectraGrabber  # SVO spectra → .txt + lookup_table.csv
from msg_spectra_grabber import MSGSpectraGrabber  # MSG (Townsend) .h5 → .txt + lookup_table.csv
from precompute_flux_cube import precompute_flux_cube  # builds flux cube from lookup + .txt
from svo_regen_spectra_lookup import parse_metadata, regenerate_lookup_table
from mast_spectra_grabber import MASTSpectraGrabber
import h5py
import numpy as np
from spectra_cleaner import clean_model_dir


STELLAR_DIR_DEFAULT = os.path.normpath(os.path.join(os.path.dirname(__file__), "data/stellar_models"))
FILTER_DIR_DEFAULT  = os.path.normpath(os.path.join(os.path.dirname(__file__), "data/filters"))

# ------------ small utils ------------

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
        from svo_regen_spectra_lookup import regenerate_lookup_table as regen  # type: ignore
    except Exception:
        regen = None

    # cleaner
    try:
        from spectra_cleaner import clean_model_dir
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
    from spectra_cleaner import clean_model_dir

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
        idx = _prompt_choice(opts, label="Spectral models", allow_back=True, page_size=30, prefer_grid=True)
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


# ------------ filters (SVO only) ------------

def run_filters_flow(base_dir: str = FILTER_DIR_DEFAULT) -> None:
    """Interactive nested filter downloader that mirrors the spectra workflow."""
    ensure_dir(base_dir)
    try:
        from svo_filter_grabber import run_interactive as _run_filter_cli
    except Exception as exc:
        print("This feature needs requests/bs4/astroquery available:", exc)
        return

    _run_filter_cli(base_dir)



def menu() -> str:
    print("\nThis tool will download an SED library and then process the files to be used with MESA-Custom Colors")
    print("\nThe data will be stored in the data/ folder and will have the same structure as the data seen in $MESA_DIR/colors/data")
    print("\nWhat would you like to run?")
    print("  1) Spectra (SVO / MSG / MAST)")
    print("  2) Filters (SVO)")
    print("  3) Rebuild (lookup + HDF5 + flux cube)")
    print("  4) Flux cube inspector / photometry")
    print("  0) Quit")
    choice = input("> ").strip()
    mapping = {
        "1": "spectra", "2": "filters", "3": "rebuild", "4": "fluxcube",
        "0": "quit"
    }
    return mapping.get(choice, "")


# ------------ CLI ------------

def main():
    p = argparse.ArgumentParser(description="SED Tools — spectra & filters")
    sub = p.add_subparsers(dest="cmd", required=False)

    # spectra
    sp = sub.add_parser("spectra", help="Download/build spectra products")
    # spectra subparser:
    sp.add_argument("--source", choices=["svo","msg","mast","both","all"], default="all",
                    help="Which provider(s) to use. 'both'=SVO+MSG, 'all'=SVO+MSG+MAST")

    sp.add_argument("--models", nargs="*", default=None,
                    help="Model names to process. For mixed sources, allow 'src:model' entries.")
    sp.add_argument("--base", default=STELLAR_DIR_DEFAULT,
                    help=f"Output base for spectra (default {STELLAR_DIR_DEFAULT})")
    sp.add_argument("--workers", type=int, default=5, help="Parallel workers for grabs/extracts (default 5)")
    sp.add_argument("--no-h5", action="store_true", help="Do not force-create HDF5 bundle from .txt")
    sp.add_argument("--no-cube", action="store_true", help="Skip flux cube build")

    # filters
    fp = sub.add_parser("filters", help="Download SVO filters (interactive facility/instrument browser)")
    fp.add_argument("--base", default=FILTER_DIR_DEFAULT,
                    help=f"Output base for filters (default {FILTER_DIR_DEFAULT})")

    # after filters subparser
    rp = sub.add_parser("rebuild", help="Rebuild lookup table + HDF5 + flux cube for local models")
    rp.add_argument("--base", default=STELLAR_DIR_DEFAULT,
                    help=f"Base models dir (default {STELLAR_DIR_DEFAULT})")
    rp.add_argument("--models", nargs="*", default=None,
                    help="Names of local model folders to rebuild (default: interactive)")
    rp.add_argument("--no-h5", action="store_true", help="Skip rebuilding the HDF5 bundle")
    rp.add_argument("--no-cube", action="store_true", help="Skip rebuilding the flux cube")



    # default (no subcommand) → interactive spectra chooser then offer filters
    args = p.parse_args()

    if args.cmd == "spectra":
        run_spectra_flow(source=args.source,
                         base_dir=args.base,
                         models=args.models,
                         workers=args.workers,
                         force_bundle_h5=not args.no_h5,
                         build_flux_cube=not args.no_cube)
    elif args.cmd == "filters":
        run_filters_flow(base_dir=args.base)

    elif args.cmd == "rebuild":
        run_rebuild_flow(base_dir=args.base,
                         models=args.models,
                         rebuild_h5=not args.no_h5,
                         rebuild_flux_cube=not args.no_cube)

    else:




        choice = menu()

        if choice == "filters":
            run_filters_flow(base_dir=FILTER_DIR_DEFAULT)
        elif choice == "rebuild":
            run_rebuild_flow(base_dir=STELLAR_DIR_DEFAULT)
        elif choice == "spectra":
            run_spectra_flow(source="all",
                             base_dir=STELLAR_DIR_DEFAULT,
                             models=None,
                             workers=5,
                             force_bundle_h5=True,
                             build_flux_cube=True)
        else:
            exit()



























from typing import Sequence, Optional, Callable, Any, Dict, List, Tuple
import math
import re
import shutil

# Optional rich support
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    _RICH = True
    _console = Console()
except Exception:
    _RICH = False
    _console = None






def _prompt_choice(options: Sequence, label: str, allow_back: bool = False) -> Optional[int]:
    if not options:
        print(f"No {label} options available.")
        return None

    filtered = list(range(len(options)))
    while True:
        print(f"\nAvailable {label} ({len(filtered)} shown):")
        print("-" * 64)
        for idx, opt_index in enumerate(filtered, 1):
            opt = options[opt_index]
            display = getattr(opt, "label", str(opt))
            print(f"{idx:3d}. {display}")
        print("\nEnter number to select", end="")
        if allow_back:
            print(", 'b' to go back", end="")
        print(", 'q' to quit, or text to filter list.")
        resp = input("> ").strip()
        if not resp:
            continue
        if resp.lower() in ("q", "quit", "exit"):
            return None
        if allow_back and resp.lower() in ("b", "back"):
            return -1
        if resp.isdigit():
            idx = int(resp)
            if 1 <= idx <= len(filtered):
                return filtered[idx - 1]
            print("Invalid index.")
            continue
        # treat as substring filter
        query = resp.lower()
        new_filtered = [i for i in range(len(options)) if query in (getattr(options[i], "label", str(options[i])).lower())]
        if not new_filtered:
            print(f"No matches for '{resp}'.")
            continue
        filtered = new_filtered




def _prompt_choice(
    options: Sequence,
    label: str,
    allow_back: bool = False,
    group_by: Optional[Callable[[Any], str]] = None,
    page_size: int = 40,
    grid_min_cols: int = 1,
    grid_max_cols: int = 3,
    prefer_grid: bool = True,
) -> Optional[int]:
    """
    Interactive selector with:
      - Stable IDs (1..N) independent of filters/pagination
      - Pagination controls (n/p/g <page>)
      - Filters: /text (substring), !text (negated substring), //regex (case-ins regex)
      - Optional two-stage grouping (B) via group_by callable
      - Multi-column grid layout (C) that adapts to terminal width
      - Rich-rendered UI if 'rich' is available; ASCII otherwise

    Returns:
      0-based index into 'options' on selection,
      -1 if user goes back (when allow_back=True),
      None if user quits.

    Commands (item mode):
      q                  quit
      b                  back (if allow_back)
      n / p              next / previous page
      g <page>           go to page number (1-based)
      id <N>             select by stable ID (1..N)
      /text              set substring filter (case-insensitive)
      !text              set negated substring filter
      //regex            set regex filter (case-insensitive)
      clear              clear filter
      groups             enter group selection (if group_by provided)
      all                clear group constraint

    Commands (group mode):
      q                  quit
      b                  back to items (without changing group)
      n / p / g <page>   paginate groups
      id <N>             pick group by its stable group ID
      /text, !text, //re filter group names
      clear              clear group filter

    Notes:
      - Displayed numbers in lists are stable IDs, not row indices.
      - Filtering matches against the display label: getattr(opt, "label", str(opt)).
    """
    if not options:
        print(f"No {label} options available.")
        return None

    # --------- Core data -----------
    N = len(options)
    ids = list(range(1, N + 1))  # stable 1-based IDs
    def _disp(o: Any) -> str:
        return getattr(o, "label", str(o))

    labels = [_disp(o) for o in options]
    labels_lc = [s.lower() for s in labels]

    # Grouping (optional)
    groups: Dict[str, List[int]] = {}
    group_order: List[str] = []
    if group_by is not None:
        for i, o in enumerate(options):
            g = group_by(o)
            if g not in groups:
                groups[g] = []
                group_order.append(g)
            groups[g].append(i)  # store 0-based indices

    # --------- State -----------
    mode = "group" if (group_by is not None) else "item"
    current_group: Optional[str] = None
    page = 1
    group_page = 1
    current_filter: Optional[Tuple[str, str]] = None  # ('substr'| 'neg' | 'regex', pattern)

    # --------- Helpers -----------
    def _term_width() -> int:
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 80

    def _apply_filter_to_names(names: List[str], names_lc: List[str]) -> List[int]:
        if current_filter is None:
            return list(range(len(names)))
        ftype, patt = current_filter
        if ftype == "substr":
            q = patt.lower()
            return [i for i, s in enumerate(names_lc) if q in s]
        if ftype == "neg":
            q = patt.lower()
            return [i for i, s in enumerate(names_lc) if q not in s]
        if ftype == "regex":
            try:
                rx = re.compile(patt, flags=re.IGNORECASE)
            except re.error:
                # keep previous filter if regex invalid; show nothing to force correction
                return []
            return [i for i, s in enumerate(names) if rx.search(s)]
        return list(range(len(names)))

    def _filtered_item_indices() -> List[int]:
        # start from all or a group
        base_idx: List[int]
        if current_group is None:
            base_idx = list(range(N))
        else:
            base_idx = groups.get(current_group, [])

        # build name arrays for those
        sub_names = [labels[i] for i in base_idx]
        sub_names_lc = [labels_lc[i] for i in base_idx]

        keep_rel = _apply_filter_to_names(sub_names, sub_names_lc)
        return [base_idx[k] for k in keep_rel]

    def _filtered_group_names() -> List[str]:
        if group_by is None:
            return []
        group_names = group_order
        group_names_lc = [g.lower() for g in group_names]
        keep = _apply_filter_to_names(group_names, group_names_lc)
        return [group_names[k] for k in keep]

    def _page_slices(total: int, page_num: int, psize: int) -> slice:
        page_max = max(1, math.ceil(total / psize))
        p = max(1, min(page_num, page_max))
        start = (p - 1) * psize
        end = min(start + psize, total)
        return slice(start, end)

    def _grid_columns(current_rows: List[str]) -> int:
        if not prefer_grid:
            return 1
        width = max(40, _term_width())
        # conservative col width: ID “[####] ” + up to 28 chars + gap
        # but adapt to longest visible label up to 32 chars
        max_label = min(32, max((len(s) for s in current_rows), default=10))
        col_w = 6 + max_label + 2
        cols = max(grid_min_cols, min(grid_max_cols, max(1, width // col_w)))
        return cols

    def _ellipsize(s: str, maxlen: int) -> str:
        return s if len(s) <= maxlen else (s[: max(0, maxlen - 1)] + "…")

    def _id_of_index(idx0: int) -> int:
        return idx0 + 1

    def _index_of_id(id1: int) -> Optional[int]:
        # validate stable ID in range
        if 1 <= id1 <= N:
            return id1 - 1
        return None

    def _status_line(total: int, shown_start: int, shown_end: int) -> str:
        f = ""
        if current_filter is not None:
            t, p = current_filter
            if t == "substr":
                f = f' filter="/{p}"'
            elif t == "neg":
                f = f' filter="!{p}"'
            elif t == "regex":
                f = f' filter="//{p}"'
        g = f' group="{current_group}"' if current_group else ""
        return f"Showing {shown_start}–{shown_end} of {total}{g}{f}"

    # --------- Renderers -----------
    def _render_items(indices: List[int], page_num: int):
        # Prepare rows
        rows = [labels[i] for i in indices]
        total = len(rows)
        if total == 0:
            msg = "No matches."
            if _RICH:
                _console.print(Panel(Text(msg, style="bold red"), title=f"{label}", expand=False))
            else:
                print(f"\n{label}: {msg}")
            return

        sl = _page_slices(total, page_num, page_size)
        view = rows[sl]
        view_idx = indices[sl]
        start = sl.start + 1
        end = sl.stop

        # Decide grid
        cols = _grid_columns(view)
        if cols <= 1:
            # Single column
            if _RICH:
                table = Table(title=Text(f"{label}", style="bold"), show_lines=False)
                table.add_column("ID", justify="right", no_wrap=True, style="cyan")
                table.add_column("Label", overflow="fold")
                for idx0, name in zip(view_idx, view):
                    table.add_row(str(_id_of_index(idx0)), name)
                _console.print(table)
                _console.print(Text(_status_line(total, start, end), style="dim"))
            else:
                print(f"\n{label} ({total} total):")
                print("-" * 64)
                for idx0, name in zip(view_idx, view):
                    print(f"[{_id_of_index(idx0):4d}] {name}")
                print(_status_line(total, start, end))
        else:
            # Grid layout
            # Compute per-cell label width
            max_label = min(32, max(len(s) for s in view))
            cell_label_w = max_label
            entries = [f"[{_id_of_index(i):4d}] {_ellipsize(n, cell_label_w)}" for i, n in zip(view_idx, view)]
            # Pad to full rows
            while len(entries) % cols != 0:
                entries.append("")
            rows_of_cols = [entries[i:i+cols] for i in range(0, len(entries), cols)]

            if _RICH:
                table = Table(title=Text(f"{label}", style="bold"), show_lines=False, pad_edge=False)
                for c in range(cols):
                    table.add_column(justify="left", overflow="fold")
                for r in rows_of_cols:
                    table.add_row(*r)
                _console.print(table)
                _console.print(Text(_status_line(total, start, end), style="dim"))
            else:
                print(f"\n{label} ({total} total):")
                print("-" * 64)
                for r in rows_of_cols:
                    print("   ".join(x.ljust(cell_label_w + 7) for x in r))
                print(_status_line(total, start, end))

        # Controls line
        if _RICH:
            controls = "[n]ext [p]rev [g <page>] [/text] [!text] [//regex] [id <N>] [clear]"
            if group_by is not None:
                controls += " [groups]"
            if allow_back:
                controls += " [b]"
            controls += " [q]"
            _console.print(Text(controls, style="italic dim"))
        else:
            controls = "Commands: n, p, g <page>, /text, !text, //regex, id <N>, clear"
            if group_by is not None:
                controls += ", groups"
            if allow_back:
                controls += ", b"
            controls += ", q"
            print(controls)

    def _render_groups(group_names: List[str], page_num: int):
        total = len(group_names)
        if total == 0:
            msg = "No groups match."
            if _RICH:
                _console.print(Panel(Text(msg, style="bold red"), title=f"{label} groups", expand=False))
            else:
                print(f"\n{label} groups: {msg}")
            return

        # stable group IDs: 1..len(group_order) regardless of filter
        # we need to map visible names to their stable group IDs
        name_to_gid = {name: i + 1 for i, name in enumerate(group_order)}
        sl = _page_slices(total, page_num, page_size)
        view = group_names[sl]
        start = sl.start + 1
        end = sl.stop

        if _RICH:
            table = Table(title=Text(f"{label} groups", style="bold"))
            table.add_column("GrpID", justify="right", style="magenta")
            table.add_column("Group")
            table.add_column("Count", justify="right", style="cyan")
            for name in view:
                gid = name_to_gid[name]
                cnt = len(groups[name])
                table.add_row(str(gid), name, str(cnt))
            _console.print(table)
            _console.print(Text(_status_line(total, start, end), style="dim"))
            _console.print(Text("Commands: n, p, g <page>, /text, !text, //regex, id <GrpID>, clear, b, q", style="italic dim"))
        else:
            print(f"\n{label} groups ({total} total):")
            print("-" * 64)
            for name in view:
                gid = name_to_gid[name]
                cnt = len(groups[name])
                print(f"[{gid:4d}] {name} ({cnt})")
            print(_status_line(total, start, end))
            print("Commands: n, p, g <page>, /text, !text, //regex, id <GrpID>, clear, b, q")

    # --------- Main loop -----------
    while True:
        if mode == "group":
            # Show groups
            gnames = _filtered_group_names()
            _render_groups(gnames, group_page)
            resp = input("> ").strip()
            if not resp:
                continue
            low = resp.lower()

            if low in ("q", "quit", "exit"):
                return None
            if low in ("b", "back"):
                # leave group mode without changing current_group
                mode = "item"
                continue
            if low.startswith("n"):
                group_page += 1
                continue
            if low.startswith("p"):
                group_page = max(1, group_page - 1)
                continue
            if low.startswith("g "):
                parts = low.split()
                if len(parts) == 2 and parts[1].isdigit():
                    group_page = max(1, int(parts[1]))
                continue
            if low.startswith("id "):
                parts = low.split()
                if len(parts) == 2 and parts[1].isdigit():
                    gid = int(parts[1])
                    if 1 <= gid <= len(group_order):
                        # select this group
                        current_group = group_order[gid - 1]
                        mode = "item"
                        page = 1
                continue
            if low == "clear":
                current_filter = None
                continue
            if low.startswith("//"):
                patt = resp[2:].strip()
                current_filter = ("regex", patt) if patt else None
                group_page = 1
                continue
            if low.startswith("!"):
                patt = resp[1:].strip()
                current_filter = ("neg", patt) if patt else None
                group_page = 1
                continue
            if low.startswith("/"):
                patt = resp[1:].strip()
                current_filter = ("substr", patt) if patt else None
                group_page = 1
                continue
            # default: treat as substring filter
            current_filter = ("substr", resp)
            group_page = 1
            continue

        # mode == "item"
        items = _filtered_item_indices()
        _render_items(items, page)

        resp = input("> ").strip()
        if not resp:
            continue
        low = resp.lower()

        if low in ("q", "quit", "exit"):
            return None
        if allow_back and low in ("b", "back"):
            return -1
        if low.startswith("n"):
            page += 1
            continue
        if low.startswith("p"):
            page = max(1, page - 1)
            continue
        if low.startswith("g "):
            parts = low.split()
            if len(parts) == 2 and parts[1].isdigit():
                page = max(1, int(parts[1]))
            continue
        if low == "groups" and group_by is not None:
            mode = "group"
            group_page = 1
            continue
        if low == "all":
            current_group = None
            page = 1
            continue
        if low.startswith("id "):
            parts = low.split()
            if len(parts) == 2 and parts[1].isdigit():
                sel_id = int(parts[1])
                idx0 = _index_of_id(sel_id)
                if idx0 is not None:
                    return idx0
            continue
        if low == "clear":
            current_filter = None
            page = 1
            continue
        if low.startswith("//"):
            patt = resp[2:].strip()
            current_filter = ("regex", patt) if patt else None
            page = 1
            continue
        if low.startswith("!"):
            patt = resp[1:].strip()
            current_filter = ("neg", patt) if patt else None
            page = 1
            continue
        if low.startswith("/"):
            patt = resp[1:].strip()
            current_filter = ("substr", patt) if patt else None
            page = 1
            continue
        # If purely digits, interpret as stable ID selection for convenience
        if resp.isdigit():
            sel_id = int(resp)
            idx0 = _index_of_id(sel_id)
            if idx0 is not None:
                return idx0
            # fall through to set as filter if out-of-range? No. Just ignore invalid.
            continue

        # default: treat input as substring filter
        current_filter = ("substr", resp)
        page = 1






























if __name__ == "__main__":
    main()
