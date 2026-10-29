"""A grid folder's lookup_table.csv and .txt spectra to a MESA flux_cube.bin (precompute_flux_cube), and the
resampling every cube builder uses. When the table's extra columns (alpha, vturb, ...) vary, cube/variants.py
applies the [on_collision] strategy of cube/collision.py. spectra_not_in_cube checks a written cube.
"""

import argparse
import csv
import logging
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._constants import available_memory_bytes as _available_ram_bytes
from ..grids.lookup import (  # noqa: F401  (write_lookup_csv: importable here, as always)
    find_file_column, find_logg_column, find_metallicity_column, find_teff_column,
    write_lookup_csv,
)
from .collision import copy_global_config_to_model, discover_extra_axes, load_config
from .fileio import flux_offset, write_header
# _extra_key_to_filename, _subdir_name and _value_to_safe_str: importable here, as always.
from .variants import (_build_filtered_cube, _build_mean_cube, _extra_key_to_filename,  # noqa: F401
                       _group_by_extra_axes, _split_into_subgrids, _subdir_name, _value_to_safe_str)
from ..grids.spectrum_io import read_text_spectrum

logger = logging.getLogger("sed_tools.precompute_flux_cube")

# Up to this many extra-axis variants are split without asking; above it the user is asked first.
SPLIT_THRESHOLD = 25


# --- Spectra and the lookup table ---

def load_sed(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    return read_text_spectrum(filepath)


_FLUX_FLOOR = 1e-300


# Quoted annotations keep the signature help() has always shown for sed_tools._resample.resample_to_grid.
def resample_to_grid(
    wl_src:  "np.ndarray",
    fl_src:  "np.ndarray",
    teff:    "float",
    wl_tgt:  "np.ndarray",
) -> "np.ndarray":
    """
    Resample a stellar spectrum onto *wl_tgt*: the one rule every flux cube builder uses.

    Inside the spectrum's wavelength coverage the flux is interpolated
    linearly in log flux; outside it is zero. No flux is invented outside a
    spectrum's native coverage: the bolometric integral is dominated by the
    covered range, and a synthetic (e.g. blackbody) tail would inflate it
    without a physically justified normalization.

    Parameters
    ----------
    wl_src : array
        Source wavelengths in Angstroms, must be strictly increasing.
    fl_src : array
        Source flux in erg/cm²/s/Å, same length as wl_src.
    teff : float
        Effective temperature of the star in Kelvin.  Not currently used by
        this function's extrapolation (out-of-coverage flux is zero-filled
        regardless of teff); retained in the signature for interface
        stability with callers.
    wl_tgt : array
        Target wavelength grid in Angstroms.

    Returns
    -------
    fl_tgt : array
        Flux on *wl_tgt* in erg/cm²/s/Å.  Always >= 0.
    """
    wl_src = np.asarray(wl_src, dtype=np.float64)
    fl_src = np.asarray(fl_src, dtype=np.float64)
    wl_tgt = np.asarray(wl_tgt, dtype=np.float64)

    log_fl = np.log10(np.maximum(fl_src, _FLUX_FLOOR))
    in_cov = (wl_tgt >= float(wl_src[0])) & (wl_tgt <= float(wl_src[-1]))
    fl_tgt = np.zeros(len(wl_tgt), dtype=np.float64)
    if in_cov.any():
        fl_tgt[in_cov] = 10.0 ** np.interp(wl_tgt[in_cov], wl_src, log_fl)
    return fl_tgt


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
    """Every column of lookup_table.csv as {column_name: [values]}, so callers can find extra axes.

    The header line may begin with '#'. Prints the row count and the columns.
    """
    result, cols, n_rows = _read_lookup_table(lookup_file)
    print(f"Lookup table: {n_rows} entries, columns: {', '.join(cols)}")
    return result


def _read_lookup_table(lookup_file: str) -> Tuple[Dict[str, List], List[str], int]:
    """:func:`load_lookup_table` without its message: the table, its header's columns and its row count."""
    with open(lookup_file, "r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline().lstrip("#").strip()
        cols = [c.strip() for c in header_line.split(",")]
        rows = [row for row in csv.reader(f) if row and not row[0].strip().startswith("#")]

    result: Dict[str, List] = {c: [] for c in cols}
    for row in rows:
        for i, col in enumerate(cols):
            result[col].append(row[i].strip() if i < len(row) else "")
    return result, cols, len(rows)


def _float_col(lookup: Dict[str, List], col: Optional[str], lookup_file: str, file_col: str) -> np.ndarray:
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


def _parse_axes(lookup: Dict[str, List], lookup_file: str, file_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Teff (required), logg and [M/H] of every lookup row; a missing logg or [M/H] column (blackbodies) is 0.0."""
    teff_col = find_teff_column(lookup.keys())
    if teff_col is None:
        raise ValueError(
            f"{lookup_file} has no Teff column (teff, t_eff, temperature or temp); "
            "cannot build a flux cube."
        )
    columns = (teff_col, find_logg_column(lookup.keys()), find_metallicity_column(lookup.keys()))
    return tuple(_float_col(lookup, col, lookup_file, file_col) for col in columns)


def _require_spectra(model_dir: str, file_names: List[str], lookup_file: str) -> None:
    """Raise FileNotFoundError listing every spectrum in the lookup table that is missing."""
    missing = [f for f in file_names if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} spectra listed in {lookup_file} are missing from {model_dir}; "
            "no flux cube was written. Restore them or remove their rows:\n  "
            + "\n  ".join(missing)
        )


# --- Building and writing one cube ---

def build_grids(teff_vals: np.ndarray, logg_vals: np.ndarray,
                meta_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    teff_grid, logg_grid, meta_grid = np.unique(teff_vals), np.unique(logg_vals), np.unique(meta_vals)
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


def _allocate_cube(teff_grid, logg_grid, meta_grid, wavelengths, tmp_path=None) -> np.ndarray:
    shape = (len(teff_grid), len(logg_grid), len(meta_grid), len(wavelengths))
    size_gib = np.prod(shape) * 8 / 1024**3
    if tmp_path:
        print(f"  Allocating {shape} ({size_gib:.2f} GiB, disk-backed)")
        return np.memmap(tmp_path, dtype=np.float64, mode="w+", shape=shape)
    print(f"  Allocating {shape} ({size_gib:.2f} GiB, in RAM)")
    return np.zeros(shape, dtype=np.float64)


def _write_cube(output_file: str, teff_grid, logg_grid, meta_grid, wavelengths, flux_cube: np.ndarray,
                usable_ram: int) -> None:
    nt, nl, nm, nw = len(teff_grid), len(logg_grid), len(meta_grid), len(wavelengths)
    tlm = nt * nl * nm
    w_chunk = max(1, usable_ram // (tlm * 2 * 8))
    n_chunks = (nw + w_chunk - 1) // w_chunk
    total_gib = nw * tlm * 8 / 1024**3

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    t0 = time.time()
    print(f"Writing {output_file} ({total_gib:.2f} GiB, {n_chunks} chunk(s))...")
    with open(output_file, "wb") as f:
        write_header(f, teff_grid, logg_grid, meta_grid, wavelengths)
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


# --- Which listed spectra a written cube holds ---

#: Wavelength planes read to tell a grid point holding a spectrum from a zero-filled one.
_SAMPLED_PLANES = 8


def spectra_not_in_cube(cube_path: str, lookup_file: str) -> Tuple[List[str], List[str]]:
    """The spectra ``lookup_file`` lists, and those of them the flux cube at ``cube_path`` does not hold.

    A spectrum is held when its (Teff, logg, [M/H]), read as :func:`precompute_flux_cube` reads the
    table, is a point of the cube's axes and that point holds flux (a point no spectrum filled is zero
    everywhere). Only the header, the axes and a few wavelength planes are read, plus, for a point that is
    zero in all of those, its own spectrum up to its first non-zero value; once one such point is zero
    throughout, the remaining such points count as not held. Raises ValueError (or OSError) when either
    file cannot be read.
    """
    lookup, cols, _ = _read_lookup_table(lookup_file)
    file_col = find_file_column(cols) or cols[0]
    names = list(lookup[file_col])
    if not names:
        return [], []
    teff, logg, meta = _parse_axes(lookup, lookup_file, file_col)
    with open(cube_path, "rb") as fh:
        header = fh.read(16)
        if len(header) != 16:
            raise ValueError(f"{cube_path}: the header is truncated")
        nt, nl, nm, nw = struct.unpack("4i", header)
        # A negative count would make np.fromfile read the whole file; a short file fails in _filled_points.
        if min(nt, nl, nm, nw) < 1:
            raise ValueError(f"{cube_path}: the header gives dimensions {nt}, {nl}, {nm}, {nw}")
        axes = np.fromfile(fh, dtype=np.float64, count=nt + nl + nm)
        data_offset = flux_offset(nt, nl, nm, nw)
        filled = _filled_points(fh, (nt, nl, nm, nw), data_offset, cube_path)
    it = _axis_positions(axes[:nt], teff)
    il = _axis_positions(axes[nt:nt + nl], logg)
    im = _axis_positions(axes[nt + nl:], meta)
    on_axes = (it >= 0) & (il >= 0) & (im >= 0)
    points = list(zip(im, il, it))
    unsampled = sorted({points[i] for i in np.flatnonzero(on_axes) if not filled[points[i]]})
    if unsampled:
        flux = np.memmap(cube_path, dtype=np.float64, mode="r", offset=data_offset, shape=(nw, nm, nl, nt))
        for point in unsampled:
            if not _holds_flux(flux, point):
                break  # a point no spectrum filled: the cube lacks the spectra there
            filled[point] = True
        del flux
    held = [ok and bool(filled[point]) for ok, point in zip(on_axes, points)]
    return names, [name for name, ok in zip(names, held) if not ok]


def _holds_flux(flux: np.ndarray, point: Tuple[int, int, int], chunk: int = 4096) -> bool:
    """Whether the cube's spectrum at ``point`` (i_meta, i_logg, i_teff) is non-zero; reads up to its first non-zero."""
    i_m, i_l, i_t = point
    for w0 in range(0, flux.shape[0], chunk):
        if np.any(flux[w0:w0 + chunk, i_m, i_l, i_t] != 0):
            return True
    return False


def _axis_positions(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Where each value is on ``axis``, or -1 where it is none of the axis's values (to rounding)."""
    unique, inverse = np.unique(values, return_inverse=True)
    close = np.isclose(unique[:, None], axis[None, :], rtol=1e-9, atol=1e-9)
    return np.where(close.any(axis=1), close.argmax(axis=1), -1)[inverse.ravel()]


def _filled_points(fh, dims: Tuple[int, int, int, int], data_offset: int, cube_path: str) -> np.ndarray:
    """Whether each grid point holds flux, indexed [i_meta, i_logg, i_teff], from a few wavelength planes
    (each wavelength's plane of points is contiguous on disk, so each plane is one read)."""
    nt, nl, nm, nw = dims
    n_points = nt * nl * nm
    filled = np.zeros((nm, nl, nt), dtype=bool)
    planes = np.unique(np.linspace(0, nw - 1, _SAMPLED_PLANES + 2)[1:-1].round().astype(int))
    for iw in planes:
        fh.seek(data_offset + int(iw) * n_points * 8)
        flux = np.fromfile(fh, dtype=np.float64, count=n_points)
        if flux.size != n_points:
            raise ValueError(f"{cube_path} ends before the {nw} wavelengths its header gives")
        filled |= flux.reshape(nm, nl, nt) != 0
    return filled


# --- One cube from a list of spectra ---

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
    """Build and write one flux cube from the given files: spectra at one node are averaged, empty nodes are zero."""
    teff_grid, logg_grid, meta_grid = build_grids(teff_vals, logg_vals, meta_vals)

    cube_bytes = len(teff_grid) * len(logg_grid) * len(meta_grid) * len(wavelengths) * 8
    tmp_path = output_file + ".tmp.memmap" if cube_bytes > usable_ram else None

    flux_cube = _allocate_cube(teff_grid, logg_grid, meta_grid, wavelengths, tmp_path)
    try:
        count_cube = _accumulate_spectra(flux_cube, model_dir, file_names, (teff_vals, logg_vals, meta_vals),
                                         (teff_grid, logg_grid, meta_grid), wavelengths, label)

        nonzero = count_cube > 0
        for axis_idx in range(len(wavelengths)):
            flux_cube[:, :, :, axis_idx] = np.where(nonzero, flux_cube[:, :, :, axis_idx] / np.maximum(count_cube, 1), 0.0)

        _write_cube(output_file, teff_grid, logg_grid, meta_grid, wavelengths, flux_cube, usable_ram)
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
    """Add every spectrum into its node of *flux_cube* and return the count per node; after reading every
    file, raises ValueError listing the spectra that held no numeric data."""
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
        print(f"\r  {prefix}spectra {done}/{n} ({100*done/n:.1f}%)  ETA {eta:.0f}s   ", end="", flush=True)

    print(f"\n  {prefix}Done — {n - len(empty)}/{n} loaded.")
    if empty:
        raise ValueError(
            f"{len(empty)} spectra hold no numeric data or could not be read; "
            "no flux cube was written. "
            "Re-download or remove them and their lookup_table.csv rows:\n  "
            + "\n  ".join(empty)
        )
    return count_cube


def _warn_averaged_rows(lookup: Dict[str, List], rows, nodes, cube_path: str) -> None:
    """Warn when lookup rows share a node: the cube holds their mean. ``nodes`` are the rows' (Teff, logg, [M/H])."""
    at_node: Dict[Tuple[float, float, float], List[int]] = {}
    for row, node in zip(rows, nodes):
        at_node.setdefault(node, []).append(row)
    shared = [members for members in at_node.values() if len(members) > 1]
    if not shared:
        return
    columns = list(lookup)
    not_compared = {find_file_column(columns) or columns[0], find_teff_column(columns),
                    find_logg_column(columns), find_metallicity_column(columns)}
    differing = [col for col in columns if col not in not_compared
                 and any(len({lookup[col][i] for i in members}) > 1 for members in shared)]
    logger.warning(
        "%s: %d lookup rows were averaged onto %d (Teff, logg, [M/H]) node(s), so each of those "
        "nodes holds a mean spectrum. The averaged rows differ in: %s.",
        cube_path, sum(len(members) for members in shared), len(shared),
        ", ".join(differing) if differing else "no column but the file name (duplicate nodes)",
    )


# --- precompute_flux_cube and its command line ---

def precompute_flux_cube(
    model_dir:     str,
    output_file:   str,
    root_dir:      Optional[str] = None,
    override_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """Build the flux cube(s) of model_dir, a folder of .txt spectra and lookup_table.csv.

    output_file is the primary (MESA-facing) flux_cube.bin; root_dir is the SED_Tools root holding
    sed_tools.defaults; override_dict overrides the [on_collision] config (highest priority).
    Raises before writing anything if the lookup table has no Teff column, holds a non-numeric or
    non-finite Teff/logg/[M/H], or lists a spectrum that does not exist; raises without writing that
    cube if a spectrum holds no data.
    """
    model_dir   = str(model_dir)
    output_file = str(output_file)

    cfg = load_config(model_dir=Path(model_dir), root_dir=Path(root_dir) if root_dir else None,
                      override_dict=override_dict)
    print(f"On-collision strategy: {cfg.strategy}  (source: {cfg.source})")

    # The configuration a cube was built with stays beside it.
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

    groups = _group_by_extra_axes(lookup, extra_cols, file_col, teff_vals, logg_vals, meta_vals)

    # No extra axes, or every file shares the same extra-axis values: one cube.
    if len(groups) <= 1:
        _warn_averaged_rows(lookup, range(len(file_names)), zip(teff_vals, logg_vals, meta_vals), output_file)
        _populate_and_write(model_dir, file_names, teff_vals, logg_vals, meta_vals, wavelengths, output_file, usable)
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
    for label, values, fmt, unit in (("Teff ", teff_vals, ".0f", " K"), ("logg ", logg_vals, ".2f", ""),
                                     ("[M/H]", meta_vals, ".2f", "")):
        if values.size:
            print(f"  {label}: {values.min():{fmt}} – {values.max():{fmt}}{unit}")


def _main():
    parser = argparse.ArgumentParser(description="Precompute flux cube for stellar spectral grids")
    parser.add_argument("--model_dir", type=str, required=False)
    parser.add_argument("--output",    type=str, default=None)
    parser.add_argument("--root_dir",  type=str, default=None,
                        help="SED_Tools root directory (for sed_tools.defaults)")
    args = parser.parse_args()

    if not args.model_dir:
        print("No --model_dir provided. Discovering local model directories…")
        root = "data/stellar_models"
        found = set()
        for base in [root, "./" + root, "../" + root]:
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                p = os.path.join(base, name)
                if not os.path.isdir(p):
                    continue
                has_lookup = os.path.isfile(os.path.join(p, "lookup_table.csv"))
                has_txt    = any(fn.lower().endswith(".txt") for fn in os.listdir(p))
                if has_lookup or has_txt:
                    found.add(p)
        cands = sorted(found)
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


if __name__ == "__main__":
    _main()
