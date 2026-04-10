#!/usr/bin/env python3
"""
precompute_flux_cube.py

Builds a MESA-compatible flux_cube.bin from a directory of standardised
SED .txt files and a lookup_table.csv.

On-collision behaviour (multiple files at the same Teff/logg/[M/H] node)
is controlled by CollisionConfig loaded from:
    <root>/sed_tools.defaults          (global defaults)
    <model_dir>/mesa_config.toml       (per-model override)
    Python API override_dict argument  (highest priority)

Strategies
----------
all-warn  Build mean cube in model_dir + one cube per extra-axis
          combination in fluxcube_library/.  Warns about collisions.
all       Same as all-warn but silent.
mean      Build only the mean cube in model_dir.  No library.
filter    Filter to a specific extra-axis slice, then build normally.
"""

import argparse
import csv
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._resample import resample_to_grid
from .collision_config import (
    CollisionConfig,
    copy_global_config_to_model,
    discover_extra_axes,
    load_config,
    write_default_config,
)

try:
    import psutil
    def _available_ram_bytes():
        return psutil.virtual_memory().available
except ImportError:
    def _available_ram_bytes():
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        return 2 * (1024 ** 3)


# ---------------------------------------------------------------------------
# SED loading
# ---------------------------------------------------------------------------

def load_sed(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    wl, fl = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    wl.append(float(parts[0]))
                    fl.append(float(parts[1]))
            except (ValueError, IndexError):
                pass
    return np.array(wl), np.array(fl)


# ---------------------------------------------------------------------------
# Lookup table loading — returns all columns
# ---------------------------------------------------------------------------

def load_lookup_table(lookup_file: str) -> Dict[str, List]:
    """
    Load lookup_table.csv and return a dict of {column_name: [values]}.
    All columns are returned so callers can discover extra axes.
    The header line may begin with '#'.
    """
    with open(lookup_file, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline().lstrip("#").strip()
        cols = [c.strip() for c in header_line.split(",")]
        reader = csv.reader(f)
        rows = [row for row in reader if row and not row[0].strip().startswith("#")]

    result: Dict[str, List] = {c: [] for c in cols}
    for row in rows:
        for i, col in enumerate(cols):
            result[col].append(row[i].strip() if i < len(row) else "")

    print(f"Lookup table: {len(rows)} entries, columns: {', '.join(cols)}")
    return result


def _find_col(lookup: Dict[str, List], candidates: List[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive)."""
    lc_lookup = {k.lower(): k for k in lookup}
    for c in candidates:
        if c.lower() in lc_lookup:
            return lc_lookup[c.lower()]
    return None


def _float_col(lookup: Dict[str, List], col: Optional[str]) -> np.ndarray:
    if col is None:
        return np.zeros(len(next(iter(lookup.values()))))
    vals = []
    for v in lookup[col]:
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            vals.append(0.0)
    return np.array(vals)


# ---------------------------------------------------------------------------
# Grid building
# ---------------------------------------------------------------------------

def build_grids(
    teff_vals: np.ndarray,
    logg_vals: np.ndarray,
    meta_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    teff_grid = np.unique(teff_vals)
    logg_grid = np.unique(logg_vals)
    meta_grid = np.unique(meta_vals)
    print(f"Grid: {len(teff_grid)} Teff × {len(logg_grid)} logg × {len(meta_grid)} meta")
    return teff_grid, logg_grid, meta_grid


def compute_common_wavelengths(model_dir: str, file_names: List[str], n_sample: int = 5) -> np.ndarray:
    n = len(file_names)
    step = max(1, n // n_sample)
    subset = [file_names[i] for i in range(0, n, step)][:n_sample]
    wl = []
    for fname in subset:
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            continue
        w, _ = load_sed(fpath)
        if w.size:
            wl.append(w)
    if not wl:
        raise RuntimeError("Could not determine wavelength grid — all sampled files empty.")
    grid = np.unique(np.concatenate(wl))
    print(f"Wavelength grid: {len(grid)} points ({grid[0]:.1f}–{grid[-1]:.1f} Å)")
    return grid


# ---------------------------------------------------------------------------
# Flux cube allocation + writing
# ---------------------------------------------------------------------------

def _allocate_cube(
    teff_grid, logg_grid, meta_grid, wavelengths, tmp_path=None
) -> np.ndarray:
    shape = (len(teff_grid), len(logg_grid), len(meta_grid), len(wavelengths))
    size_gib = np.prod(shape) * 8 / 1024**3
    if tmp_path:
        print(f"  Allocating {shape} ({size_gib:.2f} GiB, disk-backed)")
        return np.memmap(tmp_path, dtype=np.float64, mode="w+", shape=shape)
    print(f"  Allocating {shape} ({size_gib:.2f} GiB, in RAM)")
    return np.zeros(shape, dtype=np.float64)


def _write_cube(
    output_file: str,
    teff_grid, logg_grid, meta_grid, wavelengths,
    flux_cube: np.ndarray,
    usable_ram: int,
) -> None:
    nt, nl, nm, nw = len(teff_grid), len(logg_grid), len(meta_grid), len(wavelengths)
    tlm = nt * nl * nm
    w_chunk = max(1, usable_ram // (tlm * 2 * 8))
    n_chunks = (nw + w_chunk - 1) // w_chunk
    total_gib = nw * tlm * 8 / 1024**3

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    t0 = time.time()
    print(f"Writing {output_file} ({total_gib:.2f} GiB, {n_chunks} chunk(s))...")
    with open(output_file, "wb") as f:
        f.write(struct.pack("4i", nt, nl, nm, nw))
        teff_grid.astype(np.float64).tofile(f)
        logg_grid.astype(np.float64).tofile(f)
        meta_grid.astype(np.float64).tofile(f)
        wavelengths.astype(np.float64).tofile(f)
        for ci in range(n_chunks):
            w0, w1 = ci * w_chunk, min((ci + 1) * w_chunk, nw)
            chunk = np.array(flux_cube[:, :, :, w0:w1])
            chunk_t = np.ascontiguousarray(np.transpose(chunk, (3, 2, 1, 0)))
            del chunk
            chunk_t.astype(np.float64).tofile(f)
            del chunk_t
            if n_chunks > 1:
                elapsed = time.time() - t0
                eta = (n_chunks - ci - 1) / ((ci + 1) / elapsed) if elapsed > 0 else 0
                print(f"\r  chunk {ci+1}/{n_chunks}  ETA {eta:.0f}s   ", end="", flush=True)
    if n_chunks > 1:
        print()
    print(f"  Done ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Core population — builds one cube from a filtered/full file list
# ---------------------------------------------------------------------------

def _populate_and_write(
    model_dir:   str,
    file_names:  List[str],
    teff_vals:   np.ndarray,
    logg_vals:   np.ndarray,
    meta_vals:   np.ndarray,
    wavelengths: np.ndarray,
    output_file: str,
    usable_ram:  int,
    label:       str = "",
) -> None:
    """Build and write one flux cube from the given file list."""
    teff_grid, logg_grid, meta_grid = build_grids(teff_vals, logg_vals, meta_vals)

    cube_bytes = len(teff_grid) * len(logg_grid) * len(meta_grid) * len(wavelengths) * 8
    tmp_path = None
    if cube_bytes > usable_ram:
        tmp_path = output_file + ".tmp.memmap"

    flux_cube = _allocate_cube(teff_grid, logg_grid, meta_grid, wavelengths, tmp_path)

    # Build index maps — use dict for O(1) lookup
    t2i = {v: i for i, v in enumerate(teff_grid)}
    l2i = {v: i for i, v in enumerate(logg_grid)}
    m2i = {v: i for i, v in enumerate(meta_grid)}

    # Accumulator for mean: track sum and count per node
    count_cube = np.zeros(
        (len(teff_grid), len(logg_grid), len(meta_grid)), dtype=np.int32
    )

    n = len(file_names)
    skipped = 0
    t0 = time.time()
    prefix = f"[{label}] " if label else ""

    for i, fname in enumerate(file_names):
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            skipped += 1
            continue

        # Snap to nearest grid point using round-trip through dict
        teff_key = _snap(teff_vals[i], teff_grid)
        logg_key = _snap(logg_vals[i], logg_grid)
        meta_key = _snap(meta_vals[i], meta_grid)

        i_t = t2i.get(teff_key)
        i_l = l2i.get(logg_key)
        i_m = m2i.get(meta_key)
        if i_t is None or i_l is None or i_m is None:
            skipped += 1
            continue

        wl, fl = load_sed(fpath)
        try:
            resampled = resample_to_grid(wl, fl, teff_vals[i], wavelengths)
        except Exception:
            skipped += 1
            continue

        # Accumulate for mean
        flux_cube[i_t, i_l, i_m, :] += resampled
        count_cube[i_t, i_l, i_m]   += 1

        elapsed = time.time() - t0
        done    = i + 1
        rate    = done / elapsed if elapsed > 0 else 1.0
        eta     = (n - done) / rate
        print(
            f"\r  {prefix}spectra {done}/{n} ({100*done/n:.1f}%)  "
            f"skipped {skipped}  ETA {eta:.0f}s   ",
            end="", flush=True,
        )

    print(f"\n  {prefix}Done — {n - skipped}/{n} loaded, {skipped} skipped.")

    # Divide accumulated flux by count to get mean
    nonzero = count_cube > 0
    for axis_idx in range(len(wavelengths)):
        flux_cube[:, :, :, axis_idx] = np.where(
            nonzero,
            flux_cube[:, :, :, axis_idx] / np.maximum(count_cube, 1),
            0.0,
        )

    try:
        _write_cube(output_file, teff_grid, logg_grid, meta_grid,
                    wavelengths, flux_cube, usable_ram)
    finally:
        del flux_cube
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _snap(value: float, grid: np.ndarray) -> float:
    """Return the nearest value in grid to value."""
    idx = int(np.argmin(np.abs(grid - value)))
    return float(grid[idx])


# ---------------------------------------------------------------------------
# Collision handling
# ---------------------------------------------------------------------------

def _group_by_extra_axes(
    lookup:       Dict[str, List],
    extra_cols:   List[str],
    file_col:     str,
    teff_col:     str,
    logg_col:     str,
    meta_col:     str,
) -> Dict[Tuple, Dict]:
    """
    Group lookup rows by their unique extra-axis combination.

    Returns {extra_key_tuple: {"file_names": [...], "teff": [...], ...}}
    where extra_key_tuple is a tuple of (col_name, value) pairs sorted by
    col_name for deterministic ordering.
    """
    n = len(lookup[file_col])
    groups: Dict[Tuple, Dict] = {}

    for i in range(n):
        extra_key = tuple(
            sorted((col, lookup[col][i]) for col in extra_cols)
        )
        if extra_key not in groups:
            groups[extra_key] = {
                "file_names": [], "teff": [], "logg": [], "meta": [],
                "extra_vals": dict(extra_key),
            }
        g = groups[extra_key]
        g["file_names"].append(lookup[file_col][i])
        try:
            g["teff"].append(float(lookup[teff_col][i]))
        except (ValueError, KeyError):
            g["teff"].append(0.0)
        try:
            g["logg"].append(float(lookup[logg_col][i]))
        except (ValueError, KeyError):
            g["logg"].append(0.0)
        try:
            g["meta"].append(float(lookup[meta_col][i]))
        except (ValueError, KeyError):
            g["meta"].append(0.0)

    return groups


def _extra_key_to_filename(extra_key: Tuple) -> str:
    """Convert an extra-axis key tuple to a safe directory/filename fragment."""
    parts = []
    for col, val in extra_key:
        safe_col = col.strip().replace("/", "_").replace("[", "").replace("]", "").replace(" ", "_")
        safe_val = str(val).strip().replace(" ", "_").replace("/", "_")
        parts.append(f"{safe_col}_{safe_val}")
    return "__".join(parts) if parts else "default"


def _write_library_index(
    library_dir: Path,
    entries: List[Dict[str, Any]],
    extra_cols: List[str],
) -> None:
    """Write fluxcube_library/index.csv."""
    index_path = library_dir / "index.csv"
    fieldnames = ["cube_file"] + extra_cols + ["n_spectra", "teff_min", "teff_max",
                                                "logg_min", "logg_max", "meta_min", "meta_max"]
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        f.write("#" + ",".join(fieldnames) + "\n")
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for e in entries:
            w.writerow(e)
    print(f"Library index written: {index_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def precompute_flux_cube(
    model_dir:     str,
    output_file:   str,
    root_dir:      Optional[str] = None,
    override_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Build flux cube(s) for model_dir.

    Parameters
    ----------
    model_dir     : directory containing .txt spectra and lookup_table.csv
    output_file   : path for the primary (MESA-facing) flux_cube.bin
    root_dir      : SED_Tools root (for sed_tools.defaults)
    override_dict : Python-API config override (highest priority)
    """
    model_dir   = str(model_dir)
    output_file = str(output_file)

    # --- Config ---
    cfg = load_config(
        model_dir     = Path(model_dir),
        root_dir      = Path(root_dir) if root_dir else None,
        override_dict = override_dict,
    )
    print(f"On-collision strategy: {cfg.strategy}  (source: {cfg.source})")

    # Copy global config into model dir for reproducibility
    if root_dir:
        copy_global_config_to_model(Path(root_dir), Path(model_dir))

    # --- Lookup table ---
    lookup_file = os.path.join(model_dir, "lookup_table.csv")
    if not os.path.isfile(lookup_file):
        cands = [p for p in os.listdir(model_dir)
                 if p.lower().startswith("lookup_table") and p.lower().endswith(".csv")]
        if cands:
            lookup_file = os.path.join(model_dir, cands[0])
        else:
            raise FileNotFoundError(f"lookup_table.csv not found in {model_dir}")

    lookup = load_lookup_table(lookup_file)

    file_col = _find_col(lookup, ["file_name", "filename", "file"]) or list(lookup.keys())[0]
    teff_col = _find_col(lookup, ["teff", "t_eff"])
    logg_col = _find_col(lookup, ["logg", "log_g"])
    meta_col = _find_col(lookup, ["metallicity", "meta", "feh", "[fe/h]", "[m/h]"])

    if not lookup[file_col]:
        raise RuntimeError("No entries in lookup table.")

    # --- Discover extra axes ---
    extra_cols = discover_extra_axes(list(lookup.keys()))
    if extra_cols:
        print(f"Extra axes detected: {', '.join(extra_cols)}")
    else:
        print("No extra axes detected — no collisions possible.")

    # --- Wavelengths ---
    wavelengths = compute_common_wavelengths(model_dir, lookup[file_col])

    # --- RAM budget ---
    available = _available_ram_bytes()
    usable    = int(available * 0.75)
    print(f"RAM: {available/1024**3:.1f} GiB available, using up to {usable/1024**3:.1f} GiB")

    # ===================================================================
    # No extra axes → build exactly as before, no collision handling needed
    # ===================================================================
    if not extra_cols:
        file_names = lookup[file_col]
        teff_vals  = _float_col(lookup, teff_col)
        logg_vals  = _float_col(lookup, logg_col)
        meta_vals  = _float_col(lookup, meta_col)
        _populate_and_write(
            model_dir, file_names, teff_vals, logg_vals, meta_vals,
            wavelengths, output_file, usable,
        )
        _print_summary(output_file, teff_vals, logg_vals, meta_vals)
        return

    # ===================================================================
    # Extra axes present — group files by extra-axis combination
    # ===================================================================
    groups = _group_by_extra_axes(
        lookup, extra_cols, file_col,
        teff_col or "", logg_col or "", meta_col or "",
    )

    n_groups    = len(groups)
    n_colliding = sum(
        1 for g in groups.values()
        if len(set(zip(g["teff"], g["logg"], g["meta"]))) <
           len(g["file_names"])   # more files than unique nodes within group
    )

    if n_groups == 1:
        # All files share the same extra-axis values — no real collision
        g = next(iter(groups.values()))
        _populate_and_write(
            model_dir,
            g["file_names"],
            np.array(g["teff"]),
            np.array(g["logg"]),
            np.array(g["meta"]),
            wavelengths, output_file, usable,
        )
        _print_summary(output_file,
                       np.array(g["teff"]),
                       np.array(g["logg"]),
                       np.array(g["meta"]))
        return

    # Multiple extra-axis groups
    print(f"\n{n_groups} extra-axis variant(s) found across {sum(len(g['file_names']) for g in groups.values())} files.")

    if cfg.strategy in ("all-warn", "all"):
        if cfg.strategy == "all-warn":
            print("\nWARNING: Multiple extra-axis variants will be averaged into the "
                  "MESA cube. Individual variant cubes are being built in fluxcube_library/.")
            for key, g in sorted(groups.items()):
                print(f"  Variant: {dict(key)}  ({len(g['file_names'])} spectra)")

        # Build mean cube (all files, mean over extra axes)
        all_files = []
        all_teff, all_logg, all_meta = [], [], []
        for g in groups.values():
            all_files.extend(g["file_names"])
            all_teff.extend(g["teff"])
            all_logg.extend(g["logg"])
            all_meta.extend(g["meta"])

        print(f"\nBuilding mean cube → {output_file}")
        _populate_and_write(
            model_dir, all_files,
            np.array(all_teff), np.array(all_logg), np.array(all_meta),
            wavelengths, output_file, usable, label="mean",
        )

        # Build per-variant library
        library_dir = Path(model_dir) / "fluxcube_library"
        library_dir.mkdir(exist_ok=True)
        index_entries = []

        for key, g in sorted(groups.items()):
            variant_name = _extra_key_to_filename(key)
            variant_file = str(library_dir / f"flux_cube__{variant_name}.bin")
            print(f"\nBuilding variant cube: {variant_name}")
            _populate_and_write(
                model_dir,
                g["file_names"],
                np.array(g["teff"]),
                np.array(g["logg"]),
                np.array(g["meta"]),
                wavelengths, variant_file, usable, label=variant_name,
            )
            entry = {"cube_file": os.path.basename(variant_file), "n_spectra": len(g["file_names"])}
            entry.update(g["extra_vals"])
            tarr = np.array(g["teff"])
            larr = np.array(g["logg"])
            marr = np.array(g["meta"])
            entry.update({
                "teff_min": float(tarr.min()), "teff_max": float(tarr.max()),
                "logg_min": float(larr.min()), "logg_max": float(larr.max()),
                "meta_min": float(marr.min()), "meta_max": float(marr.max()),
            })
            index_entries.append(entry)

        _write_library_index(library_dir, index_entries, extra_cols)

    elif cfg.strategy == "mean":
        # Mean only — no library
        all_files = []
        all_teff, all_logg, all_meta = [], [], []
        for g in groups.values():
            all_files.extend(g["file_names"])
            all_teff.extend(g["teff"])
            all_logg.extend(g["logg"])
            all_meta.extend(g["meta"])

        print(f"Building mean cube (no library) → {output_file}")
        _populate_and_write(
            model_dir, all_files,
            np.array(all_teff), np.array(all_logg), np.array(all_meta),
            wavelengths, output_file, usable, label="mean",
        )

    elif cfg.strategy == "filter":
        # Resolve filter specs against available extra-axis values
        extra_unique: Dict[str, List] = {}
        for col in extra_cols:
            extra_unique[col] = sorted(set(lookup[col]))

        resolved, errors = cfg.resolve_filter_values(extra_unique)
        if errors:
            raise ValueError(
                "Filter config errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        print(f"Filter: {resolved}")

        # Find matching group
        matched_group = None
        for key, g in groups.items():
            key_dict = dict(key)
            match = True
            for col, val in resolved.items():
                if col not in key_dict:
                    match = False
                    break
                # compare as strings (handles float/str uniformly)
                if str(key_dict[col]).strip().lower() != str(val).strip().lower():
                    match = False
                    break
            if match:
                matched_group = g
                break

        if matched_group is None:
            available_variants = "\n".join(
                f"  {dict(k)}" for k in sorted(groups.keys())
            )
            raise ValueError(
                f"No variant matches filter {resolved}.\n"
                f"Available variants:\n{available_variants}"
            )

        print(f"Matched variant: {matched_group['extra_vals']} "
              f"({len(matched_group['file_names'])} spectra)")
        _populate_and_write(
            model_dir,
            matched_group["file_names"],
            np.array(matched_group["teff"]),
            np.array(matched_group["logg"]),
            np.array(matched_group["meta"]),
            wavelengths, output_file, usable,
        )

    _print_summary(output_file,
                   _float_col(lookup, teff_col),
                   _float_col(lookup, logg_col),
                   _float_col(lookup, meta_col))


def _print_summary(output_file, teff_vals, logg_vals, meta_vals):
    print(f"\nFlux cube: {output_file}")
    if teff_vals.size:
        print(f"  Teff : {teff_vals.min():.0f} – {teff_vals.max():.0f} K")
    if logg_vals.size:
        print(f"  logg : {logg_vals.min():.2f} – {logg_vals.max():.2f}")
    if meta_vals.size:
        print(f"  [M/H]: {meta_vals.min():.2f} – {meta_vals.max():.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Precompute flux cube for stellar atmosphere models")
    parser.add_argument("--model_dir", type=str, required=False)
    parser.add_argument("--output",    type=str, default=None)
    parser.add_argument("--root_dir",  type=str, default=None,
                        help="SED_Tools root directory (for sed_tools.defaults)")
    args = parser.parse_args()

    def _discover_model_dirs(root="data/stellar_models"):
        cands = []
        for base in [root, "./" + root, "../" + root]:
            if os.path.isdir(base):
                for name in sorted(os.listdir(base)):
                    p = os.path.join(base, name)
                    if not os.path.isdir(p):
                        continue
                    has_lookup = os.path.isfile(os.path.join(p, "lookup_table.csv"))
                    has_txt    = any(fn.lower().endswith(".txt") for fn in os.listdir(p))
                    if has_lookup or has_txt:
                        cands.append(p)
        return sorted(set(cands))

    if not args.model_dir:
        print("No --model_dir provided. Discovering local model directories…")
        cands = _discover_model_dirs()
        if not cands:
            sys.exit("No model directories found.")
        for i, p in enumerate(cands, 1):
            print(f"{i}. {p}")
        sel = input("Select a model directory by number: ").strip()
        if not sel.isdigit() or not (1 <= int(sel) <= len(cands)):
            sys.exit("Invalid selection.")
        args.model_dir = cands[int(sel) - 1]

    if not args.output:
        args.output = os.path.join(args.model_dir, "flux_cube.bin")
        ans = input(f"Output file [{args.output}]: ").strip()
        if ans:
            args.output = ans

    precompute_flux_cube(args.model_dir, args.output, root_dir=args.root_dir)
