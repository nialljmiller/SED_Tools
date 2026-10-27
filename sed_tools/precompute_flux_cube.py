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
split     Auto-split extra physical axes into separate MESA-ready subgrid
          directories.  Each subgrid varies only over Teff, logg, metallicity.
          This is the default and the physically safe choice.
all-warn  Alias for split (backward compatible).
all       Alias for split (backward compatible).
mean      Build a single mean cube by averaging over extra axes.
          Physically unsafe — use only if you understand the implications.
filter    Filter to a specific extra-axis slice, then build normally.
"""

import argparse
import csv
import logging
import math
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._lookup_io import (
    find_file_column, find_logg_column, find_metallicity_column, find_teff_column,
    write_lookup_csv,
)
from ._resample import resample_to_grid
from .collision_config import (
    copy_global_config_to_model,
    discover_extra_axes,
    load_config,
)
from .spectrum_io import read_text_spectrum

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

# Auto-split when the number of extra-axis combinations is at or below this.
# Above this threshold the user is prompted before creating many directories.
SPLIT_THRESHOLD = 25

_MEMINFO = "/proc/meminfo"
_ASSUMED_RAM_BYTES = 2 * (1024 ** 3)


def _available_ram_bytes() -> int:
    """Available RAM in bytes.

    It only decides whether the cube is held in RAM or on disk and how many
    wavelengths are written per chunk; the cube's contents do not depend on it.
    """
    if psutil is not None:
        return psutil.virtual_memory().available
    try:
        with open(_MEMINFO) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        reason = f"{_MEMINFO} has no MemAvailable line"
    except OSError as exc:
        reason = f"psutil is not installed and {_MEMINFO} could not be read ({exc})"
    logger.warning(
        "Could not measure available RAM: %s. Assuming %d GiB; this affects "
        "memory use only, not the flux cube.", reason, _ASSUMED_RAM_BYTES // 1024**3,
    )
    return _ASSUMED_RAM_BYTES


# ---------------------------------------------------------------------------
# SED loading
# ---------------------------------------------------------------------------

def load_sed(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    return read_text_spectrum(filepath)


# ---------------------------------------------------------------------------
# Lookup table loading — returns all columns
# ---------------------------------------------------------------------------

def _find_lookup_file(model_dir: str) -> str:
    lookup_file = os.path.join(model_dir, "lookup_table.csv")
    if os.path.isfile(lookup_file):
        return lookup_file
    cands = [p for p in os.listdir(model_dir)
             if p.lower().startswith("lookup_table") and p.lower().endswith(".csv")]
    if not cands:
        raise FileNotFoundError(f"lookup_table.csv not found in {model_dir}")
    lookup_file = os.path.join(model_dir, cands[0])
    logger.warning("lookup_table.csv not found in %s; using %s instead", model_dir, lookup_file)
    return lookup_file


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


def _float_col(
    lookup: Dict[str, List], col: Optional[str], lookup_file: str, file_col: str,
) -> np.ndarray:
    """Parse one parameter column; a missing column (``col`` None) reads as 0.0 throughout."""
    if col is None:
        return np.zeros(len(next(iter(lookup.values()))))
    vals = []
    for text, fname in zip(lookup[col], lookup[file_col]):
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        if not math.isfinite(value):
            raise ValueError(
                f"{lookup_file}: {col} = {text!r} for {fname} is not a finite number. "
                "Correct that row. A grid without a logg or metallicity axis should "
                "omit that column, which then reads as 0."
            )
        vals.append(value)
    return np.array(vals)


def _parse_axes(
    lookup: Dict[str, List], lookup_file: str, file_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Teff, logg and [M/H] of every lookup row.

    Teff is required. A grid without a logg or metallicity column (e.g.
    blackbodies) gets a single 0.0 node on that axis.
    """
    teff_col = find_teff_column(lookup.keys())
    if teff_col is None:
        raise ValueError(
            f"{lookup_file} has no Teff column (teff, t_eff, temperature or temp); "
            "cannot build a flux cube."
        )
    logg_col = find_logg_column(lookup.keys())
    meta_col = find_metallicity_column(lookup.keys())
    return (
        _float_col(lookup, teff_col, lookup_file, file_col),
        _float_col(lookup, logg_col, lookup_file, file_col),
        _float_col(lookup, meta_col, lookup_file, file_col),
    )


def _require_spectra(model_dir: str, file_names: List[str], lookup_file: str) -> None:
    """Raise FileNotFoundError listing every spectrum in the lookup table that is missing."""
    missing = [f for f in file_names if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} spectra listed in {lookup_file} are missing from {model_dir}; "
            "no flux cube was written. Restore them or remove their rows:\n  "
            + "\n  ".join(missing)
        )


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
        w, _ = load_sed(os.path.join(model_dir, fname))
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
    """Build and write one flux cube from the given file list.

    Spectra at the same (Teff, logg, [M/H]) node are averaged; nodes with no
    spectrum are zero.
    """
    teff_grid, logg_grid, meta_grid = build_grids(teff_vals, logg_vals, meta_vals)

    cube_bytes = len(teff_grid) * len(logg_grid) * len(meta_grid) * len(wavelengths) * 8
    tmp_path = None
    if cube_bytes > usable_ram:
        tmp_path = output_file + ".tmp.memmap"

    flux_cube = _allocate_cube(teff_grid, logg_grid, meta_grid, wavelengths, tmp_path)
    try:
        count_cube = _accumulate_spectra(
            flux_cube, model_dir, file_names,
            (teff_vals, logg_vals, meta_vals), (teff_grid, logg_grid, meta_grid),
            wavelengths, label,
        )

        # Divide accumulated flux by count to get mean
        nonzero = count_cube > 0
        for axis_idx in range(len(wavelengths)):
            flux_cube[:, :, :, axis_idx] = np.where(
                nonzero,
                flux_cube[:, :, :, axis_idx] / np.maximum(count_cube, 1),
                0.0,
            )

        _write_cube(output_file, teff_grid, logg_grid, meta_grid,
                    wavelengths, flux_cube, usable_ram)
    finally:
        del flux_cube
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _accumulate_spectra(
    flux_cube:   np.ndarray,
    model_dir:   str,
    file_names:  List[str],
    params:      Tuple[np.ndarray, np.ndarray, np.ndarray],
    grids:       Tuple[np.ndarray, np.ndarray, np.ndarray],
    wavelengths: np.ndarray,
    label:       str,
) -> np.ndarray:
    """Add every spectrum into its node of *flux_cube*; return the spectra count per node.

    After reading every file, raises ValueError listing the spectra that
    held no numeric data.
    """
    teff_vals, logg_vals, meta_vals = params
    teff_grid, logg_grid, meta_grid = grids
    t2i = {v: i for i, v in enumerate(teff_grid)}
    l2i = {v: i for i, v in enumerate(logg_grid)}
    m2i = {v: i for i, v in enumerate(meta_grid)}
    count_cube = np.zeros((len(teff_grid), len(logg_grid), len(meta_grid)), dtype=np.int32)

    n = len(file_names)
    empty: List[str] = []
    t0 = time.time()
    prefix = f"[{label}] " if label else ""

    for i, fname in enumerate(file_names):
        fpath = os.path.join(model_dir, fname)
        wl, fl = load_sed(fpath)
        if wl.size == 0:
            empty.append(fpath)
            continue

        i_t, i_l, i_m = t2i[teff_vals[i]], l2i[logg_vals[i]], m2i[meta_vals[i]]
        flux_cube[i_t, i_l, i_m, :] += resample_to_grid(wl, fl, teff_vals[i], wavelengths)
        count_cube[i_t, i_l, i_m] += 1

        elapsed = time.time() - t0
        done = i + 1
        rate = done / elapsed if elapsed > 0 else 1.0
        eta = (n - done) / rate
        print(
            f"\r  {prefix}spectra {done}/{n} ({100*done/n:.1f}%)  ETA {eta:.0f}s   ",
            end="", flush=True,
        )

    print(f"\n  {prefix}Done — {n - len(empty)}/{n} loaded.")
    if empty:
        raise ValueError(
            f"{len(empty)} spectra hold no numeric data; no flux cube was written. "
            "Re-download or remove them and their lookup_table.csv rows:\n  "
            + "\n  ".join(empty)
        )
    return count_cube


# ---------------------------------------------------------------------------
# Collision handling
# ---------------------------------------------------------------------------

def _group_by_extra_axes(
    lookup:     Dict[str, List],
    extra_cols: List[str],
    file_col:   str,
    teff_vals:  np.ndarray,
    logg_vals:  np.ndarray,
    meta_vals:  np.ndarray,
) -> Dict[Tuple, Dict]:
    """
    Group lookup rows by their unique extra-axis combination.

    Returns {extra_key_tuple: {"file_names": [...], "teff": [...], ...}}
    where extra_key_tuple is a tuple of (col_name, value) pairs sorted by
    col_name for deterministic ordering.
    """
    groups: Dict[Tuple, Dict] = {}

    for i, fname in enumerate(lookup[file_col]):
        extra_key = tuple(
            sorted((col, lookup[col][i]) for col in extra_cols)
        )
        if extra_key not in groups:
            groups[extra_key] = {
                "file_names": [], "teff": [], "logg": [], "meta": [],
                "extra_vals": dict(extra_key),
            }
        g = groups[extra_key]
        g["file_names"].append(fname)
        g["teff"].append(float(teff_vals[i]))
        g["logg"].append(float(logg_vals[i]))
        g["meta"].append(float(meta_vals[i]))

    return groups


def _extra_key_to_filename(extra_key: Tuple) -> str:
    """Convert an extra-axis key tuple to a safe directory/filename fragment."""
    parts = []
    for col, val in extra_key:
        safe_col = col.strip().replace("/", "_").replace("[", "").replace("]", "").replace(" ", "_")
        safe_val = str(val).strip().replace(" ", "_").replace("/", "_")
        parts.append(f"{safe_col}_{safe_val}")
    return "__".join(parts) if parts else "default"


# ---------------------------------------------------------------------------
# Subgrid splitting — physically safe extra-axis handling
# ---------------------------------------------------------------------------

def _value_to_safe_str(val: str) -> str:
    """Convert a parameter value to a filesystem-safe string.

    Numeric:
        "0"      -> "0p00"
        "0.01"   -> "0p01"
        "0.1"    -> "0p10"
        "0.25"   -> "0p25"
        "-0.5"   -> "m0p50"
        "+0.5"   -> "0p50"
    Non-numeric:
        "h-rich" -> "h_rich"
    """
    val = str(val).strip()
    try:
        f = float(val)
    except ValueError:
        return re.sub(r"[^a-zA-Z0-9]+", "_", val).strip("_")
    prefix = "m" if f < 0 else ""
    formatted = f"{abs(f):.2f}".replace(".", "p")
    return prefix + formatted


def _subdir_name(model_name: str, extra_key: Tuple) -> str:
    """Generate a MESA-ready subgrid directory name from an extra-axis key.

    Example:
        model_name="Husfeld", key=(("yhe", "0.10"),) -> "Husfeld_yhe_0p10"
        two axes: key=(("alpha","0.4"),("yhe","0.10")) -> "Husfeld_alpha_0p40_yhe_0p10"
    """
    parts = []
    for col, val in extra_key:
        safe_col = re.sub(r"[^a-zA-Z0-9]+", "_", col.strip()).strip("_")
        safe_val = _value_to_safe_str(val)
        parts.append(f"{safe_col}_{safe_val}")
    return f"{model_name}_" + "_".join(parts) if parts else model_name


def _filter_lookup_by_files(
    lookup: Dict[str, List],
    file_names: List[str],
    file_col: str,
) -> Dict[str, List]:
    """Return a copy of lookup restricted to rows whose file_name is in file_names."""
    file_set = set(file_names)
    indices = [i for i, f in enumerate(lookup[file_col]) if f in file_set]
    return {col: [lookup[col][i] for i in indices] for col in lookup}


def _write_variants_index(
    parent_dir: str,
    entries: List[Dict[str, Any]],
    extra_cols: List[str],
) -> None:
    """Write variants_index.csv to the parent directory."""
    path = os.path.join(parent_dir, "variants_index.csv")
    fieldnames = ["variant_name"] + extra_cols + ["n_spectra", "path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("#" + ",".join(fieldnames) + "\n")
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for e in entries:
            w.writerow(e)
    print(f"Variants index: {path}")


def _print_split_announcement(
    groups: Dict,
    extra_cols: List[str],
    model_name: str,
    threshold: int,
) -> None:
    """Print the structured 'Extra physical axis detected' message."""
    n_combos = len(groups)
    axis_label = ", ".join(extra_cols)

    print(f"\nExtra physical axis detected: {axis_label}")
    print("\nMESA-compatible cubes can only vary over:")
    print("  Teff, logg, metallicity")
    print(f"\nThis grid contains {n_combos} {axis_label} variant(s):")
    for key, g in sorted(groups.items()):
        vals_str = "  ".join(f"{col} = {val}" for col, val in key)
        print(f"  {vals_str:<40}  {len(g['file_names'])} spectra")

    if n_combos <= threshold:
        print("\nWriting separate MESA-ready subgrids:")
        for key in sorted(groups):
            print(f"  {_subdir_name(model_name, key)}/")
        print("\nThe parent directory will retain the full source grid.")
        print("Use one of the subdirectories as the spectral grid in MESA Colors.")
    else:
        print(f"\n{n_combos} combinations exceed the auto-split threshold ({threshold}).")


def _link_spectra(model_dir: str, subdir: str, file_names: List[str]) -> None:
    """Symlink a variant's spectra from the parent so the subgrid can be rebuilt on its own."""
    n_linked = 0
    failures: List[str] = []
    for fname in file_names:
        dst = os.path.join(subdir, fname)
        if os.path.lexists(dst):
            continue
        try:
            os.symlink(os.path.join("..", fname), dst)
            n_linked += 1
        except OSError as exc:
            failures.append(f"{fname}: {exc}")
    if n_linked:
        print(f"  Linked {n_linked} spectrum files to parent directory")
    if failures:
        # The cube is built from the parent directory, so it is unaffected.
        logger.warning(
            "Could not link %d spectra into %s, so that subgrid cannot be rebuilt "
            "on its own (its flux cube is complete). First failure: %s",
            len(failures), subdir, failures[0],
        )


def _build_split_subgrids(
    model_dir: str,
    model_name: str,
    groups: Dict,
    extra_cols: List[str],
    lookup: Dict[str, List],
    file_col: str,
    wavelengths: np.ndarray,
    usable_ram: int,
) -> List[str]:
    """Build a MESA-ready subgrid directory for every extra-axis combination.

    For each variant:
      - Creates a subdirectory named {ModelName}_{axis}_{value}/
      - Writes a filtered lookup_table.csv
      - Symlinks .txt spectra from parent (warns if symlinks are unsupported)
      - Builds flux_cube.bin from the variant's spectra

    Then writes variants_index.csv in the parent directory and copies
    lookup_table.csv to lookup_table_full.csv as the authoritative full-grid record.

    Returns the list of subgrid directory paths created.
    """
    subdir_paths: List[str] = []
    index_entries: List[Dict[str, Any]] = []

    for key, g in sorted(groups.items()):
        name = _subdir_name(model_name, key)
        subdir = os.path.join(model_dir, name)
        os.makedirs(subdir, exist_ok=True)

        print(f"\n  [{name}]  {len(g['file_names'])} spectra")

        filtered = _filter_lookup_by_files(lookup, g["file_names"], file_col)
        write_lookup_csv(filtered, os.path.join(subdir, "lookup_table.csv"))
        _link_spectra(model_dir, subdir, g["file_names"])

        # Build flux cube (reads txt files from model_dir = parent)
        cube_path = os.path.join(subdir, "flux_cube.bin")
        teff_arr = np.array(g["teff"])
        logg_arr = np.array(g["logg"])
        meta_arr = np.array(g["meta"])
        _populate_and_write(
            model_dir, g["file_names"],
            teff_arr, logg_arr, meta_arr,
            wavelengths, cube_path, usable_ram,
            label=name,
        )
        _print_summary(cube_path, teff_arr, logg_arr, meta_arr)

        entry: Dict[str, Any] = {
            "variant_name": name,
            "n_spectra": len(g["file_names"]),
            "path": name,
        }
        entry.update(g["extra_vals"])
        index_entries.append(entry)
        subdir_paths.append(subdir)

    # Write parent-level inventory
    _write_variants_index(model_dir, index_entries, extra_cols)

    # Preserve the full-grid lookup under a permanent name
    src_lookup = os.path.join(model_dir, "lookup_table.csv")
    full_lookup = os.path.join(model_dir, "lookup_table_full.csv")
    if os.path.exists(src_lookup) and not os.path.exists(full_lookup):
        shutil.copy2(src_lookup, full_lookup)
        print("Full-grid lookup preserved: lookup_table_full.csv")

    return subdir_paths


# ---------------------------------------------------------------------------
# Strategies for grids with several extra-axis variants
# ---------------------------------------------------------------------------

def _confirm_large_split(model_dir: str, n_groups: int) -> bool:
    """Ask before splitting more than SPLIT_THRESHOLD variants; True to go ahead."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"{model_dir}: {n_groups} extra-axis combinations exceed the auto-split "
            f"threshold ({SPLIT_THRESHOLD}); no flux cube was built. Run interactively "
            "to confirm the split, or set strategy = \"filter\" or \"mean\" under "
            "[on_collision] in mesa_config.toml."
        )
    print("\nOptions:")
    print(f"  s) Split all {n_groups} combinations into subgrid directories")
    print("  a) Abort — no flux cube built")
    return input("  Choice [s/a]: ").strip().lower() == "s"


def _split_into_subgrids(
    model_dir:   str,
    groups:      Dict,
    extra_cols:  List[str],
    lookup:      Dict[str, List],
    file_col:    str,
    wavelengths: np.ndarray,
    usable_ram:  int,
) -> None:
    """Strategy 'split': one MESA-ready subgrid per variant, no cube in the parent."""
    model_name = os.path.basename(os.path.abspath(model_dir))
    _print_split_announcement(groups, extra_cols, model_name, SPLIT_THRESHOLD)

    if len(groups) > SPLIT_THRESHOLD and not _confirm_large_split(model_dir, len(groups)):
        print("Aborted. No flux cube built.")
        return

    _build_split_subgrids(
        model_dir, model_name, groups, extra_cols,
        lookup, file_col, wavelengths, usable_ram,
    )
    print("\nDone. The parent directory is the source collection.")
    print("Point MESA Colors to one of the subgrid directories listed above.")
    # The parent gets no flux_cube.bin: it is not MESA-ready.


def _build_mean_cube(
    model_dir:   str,
    groups:      Dict,
    wavelengths: np.ndarray,
    output_file: str,
    usable_ram:  int,
) -> None:
    """Strategy 'mean': one cube averaging every variant at each node."""
    print(
        "\nWARNING: strategy='mean' collapses extra axes by averaging. "
        "This is physically unsafe.\n"
        "Consider strategy='split' (the default) to get per-variant MESA-ready subgrids."
    )
    all_files: List[str] = []
    all_teff: List[float] = []
    all_logg: List[float] = []
    all_meta: List[float] = []
    for g in groups.values():
        all_files.extend(g["file_names"])
        all_teff.extend(g["teff"])
        all_logg.extend(g["logg"])
        all_meta.extend(g["meta"])

    print(f"Building mean cube (no subgrids) → {output_file}")
    _populate_and_write(
        model_dir, all_files,
        np.array(all_teff), np.array(all_logg), np.array(all_meta),
        wavelengths, output_file, usable_ram, label="mean",
    )


def _build_filtered_cube(
    cfg,
    model_dir:   str,
    groups:      Dict,
    extra_cols:  List[str],
    lookup:      Dict[str, List],
    wavelengths: np.ndarray,
    output_file: str,
    usable_ram:  int,
) -> None:
    """Strategy 'filter': one cube from the single variant the config selects."""
    extra_unique = {col: sorted(set(lookup[col])) for col in extra_cols}
    resolved, errors = cfg.resolve_filter_values(extra_unique)
    if errors:
        raise ValueError(
            "Filter config errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    print(f"Filter: {resolved}")

    matched_group = None
    for key, g in groups.items():
        key_dict = dict(key)
        match = all(
            col in key_dict
            and str(key_dict[col]).strip().lower() == str(val).strip().lower()
            for col, val in resolved.items()
        )
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
        wavelengths, output_file, usable_ram,
    )


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

    Raises before writing anything if the lookup table has no Teff column,
    holds a non-numeric or non-finite Teff/logg/[M/H], or lists a spectrum
    that does not exist; raises without writing that cube if a spectrum holds
    no data.
    """
    model_dir   = str(model_dir)
    output_file = str(output_file)

    cfg = load_config(
        model_dir     = Path(model_dir),
        root_dir      = Path(root_dir) if root_dir else None,
        override_dict = override_dict,
    )
    print(f"On-collision strategy: {cfg.strategy}  (source: {cfg.source})")

    # Copy global config into model dir for reproducibility
    if root_dir:
        copy_global_config_to_model(Path(root_dir), Path(model_dir))

    lookup_file = _find_lookup_file(model_dir)
    lookup = load_lookup_table(lookup_file)
    file_col = find_file_column(lookup.keys()) or list(lookup.keys())[0]
    file_names = lookup[file_col]
    if not file_names:
        raise RuntimeError("No entries in lookup table.")
    teff_vals, logg_vals, meta_vals = _parse_axes(lookup, lookup_file, file_col)
    _require_spectra(model_dir, file_names, lookup_file)

    extra_cols = discover_extra_axes(list(lookup.keys()))
    if extra_cols:
        print(f"Extra axes detected: {', '.join(extra_cols)}")
    else:
        print("No extra axes detected — no collisions possible.")

    wavelengths = compute_common_wavelengths(model_dir, file_names)

    available = _available_ram_bytes()
    usable    = int(available * 0.75)
    print(f"RAM: {available/1024**3:.1f} GiB available, using up to {usable/1024**3:.1f} GiB")

    groups = {}
    if extra_cols:
        groups = _group_by_extra_axes(lookup, extra_cols, file_col, teff_vals, logg_vals, meta_vals)

    # No extra axes, or every file shares the same extra-axis values: one cube.
    if len(groups) <= 1:
        _populate_and_write(
            model_dir, file_names, teff_vals, logg_vals, meta_vals,
            wavelengths, output_file, usable,
        )
        _print_summary(output_file, teff_vals, logg_vals, meta_vals)
        return

    print(f"\n{len(groups)} extra-axis variant(s) across {len(file_names)} spectra.")

    if cfg.strategy in ("split", "all", "all-warn"):
        _split_into_subgrids(model_dir, groups, extra_cols, lookup, file_col, wavelengths, usable)
        return
    if cfg.strategy == "mean":
        _build_mean_cube(model_dir, groups, wavelengths, output_file, usable)
    elif cfg.strategy == "filter":
        _build_filtered_cube(cfg, model_dir, groups, extra_cols, lookup, wavelengths, output_file, usable)

    _print_summary(output_file, teff_vals, logg_vals, meta_vals)


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
    parser = argparse.ArgumentParser(description="Precompute flux cube for stellar spectral grids")
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
