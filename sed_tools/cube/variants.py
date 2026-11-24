"""Grids whose extra lookup columns (alpha, vturb, ...) vary: one MESA-ready subgrid per variant (the
'split' strategy), a mean cube, or one filtered variant; subgrid names and variants_index.csv.
precompute_flux_cube in cube/build.py chooses the strategy and passes in SPLIT_THRESHOLD and its writer of one cube.
"""

import csv
import itertools
import logging
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from ..grids.lookup import (find_file_column, find_logg_column, find_metallicity_column, find_teff_column,
                            write_lookup_csv)
from .collision import _value
from .fileio import ALL_AXES_NAME, read_header, write_all_axes_header

logger = logging.getLogger(__name__)


# --- Variants and their names ---

def _group_by_extra_axes(lookup: Dict[str, List], extra_cols: List[str], file_col: str, teff_vals: np.ndarray,
                         logg_vals: np.ndarray, meta_vals: np.ndarray) -> Dict[Tuple, Dict]:
    """The lookup rows grouped by their extra-axis values: {key: {"rows", "file_names", "teff", "logg", "meta",
    "extra_vals"}}, where key is the sorted (column, value) pairs, each value as its first row spells it, and "rows"
    are row numbers in *lookup*. Values are compared as collision._value compares them ("0" and "0.0" are one)."""
    groups: Dict[Tuple, Dict] = {}
    by_value: Dict[Tuple, Dict] = {}

    for i, fname in enumerate(lookup[file_col]):
        extra_key = tuple(sorted((col, lookup[col][i]) for col in extra_cols))
        same = tuple((col, _value(text)) for col, text in extra_key)
        if same not in by_value:
            by_value[same] = groups[extra_key] = {
                "rows": [], "file_names": [], "teff": [], "logg": [], "meta": [], "extra_vals": dict(extra_key),
            }
        g = by_value[same]
        g["rows"].append(i)
        g["file_names"].append(fname)
        for column, values in (("teff", teff_vals), ("logg", logg_vals), ("meta", meta_vals)):
            g[column].append(float(values[i]))

    return groups


def _extra_key_to_filename(extra_key: Tuple) -> str:
    """Convert an extra-axis key tuple to a safe directory/filename fragment."""
    parts = []
    for col, val in extra_key:
        safe_col = col.strip().replace("/", "_").replace("[", "").replace("]", "").replace(" ", "_")
        safe_val = str(val).strip().replace(" ", "_").replace("/", "_")
        parts.append(f"{safe_col}_{safe_val}")
    return "__".join(parts) if parts else "default"


def _value_to_safe_str(val: str, decimals: int = 2) -> str:
    """A value as a file-name fragment: "0.1" -> "0p10", "-0.5" -> "m0p50", "+0.5" -> "0p50", "h-rich" -> "h_rich"."""
    val = str(val).strip()
    try:
        f = float(val)
    except ValueError:
        return re.sub(r"[^a-zA-Z0-9]+", "_", val).strip("_")
    return ("m" if f < 0 else "") + f"{abs(f):.{decimals}f}".replace(".", "p")


def _subdir_name(model_name: str, extra_key: Tuple, decimals: Dict[str, int]) -> str:
    """A variant's subgrid folder: "Husfeld", (("alpha", "0.4"), ("yhe", "0.10")) -> Husfeld_alpha_0p40_yhe_0p10,
    each numeric value with its column's ``decimals``."""
    parts = []
    for col, val in extra_key:
        safe_col = re.sub(r"[^a-zA-Z0-9]+", "_", col.strip()).strip("_")
        parts.append(f"{safe_col}_{_value_to_safe_str(val, decimals[col])}")
    return f"{model_name}_" + "_".join(parts) if parts else model_name


def _subdir_names(model_name: str, keys) -> Dict[Tuple, str]:
    """Each variant's subgrid folder, {key: name} (:func:`_subdir_name`). A numeric axis is written with two decimals,
    or with as many more as keep its values apart (0.001 and 0.004: 0p001 and 0p004). ValueError, before anything is
    written, if two variants would still share a folder (text values that differ only in punctuation)."""
    decimals = {}
    for col in {col for key in keys for col, _ in key}:
        numbers = {n for key in keys for c, v in key if c == col for n in [_value(v)] if isinstance(n, float)}
        decimals[col] = 2
        while len({_value_to_safe_str(n, decimals[col]) for n in numbers}) < len(numbers):
            decimals[col] += 1
    names = {key: _subdir_name(model_name, key, decimals) for key in keys}
    shared = sorted(name for name in set(names.values()) if list(names.values()).count(name) > 1)
    if shared:
        raise ValueError(f"Variants with different extra-axis values would share the subgrid folder(s) "
                         f"{', '.join(shared)}; no flux cube was written. Make each axis's values differ in more "
                         "than punctuation.")
    return names


def _filter_lookup_by_files(lookup: Dict[str, List], file_names: List[str], file_col: str) -> Dict[str, List]:
    """Return a copy of lookup restricted to rows whose file_name is in file_names."""
    file_set = set(file_names)
    indices = [i for i, f in enumerate(lookup[file_col]) if f in file_set]
    return {col: [lookup[col][i] for i in indices] for col in lookup}


def _print_split_announcement(groups: Dict, extra_cols: List[str], names: Dict[Tuple, str], threshold: int) -> None:
    """Print the 'Extra physical axis detected' message, naming only the axes that vary between the variants: a
    column with one value throughout (lh and vtur in Kurucz) is no variant axis, though subgrid ``names`` carry it."""
    n_combos = len(groups)
    varying = [col for col in extra_cols if len({dict(key)[col] for key in groups}) > 1]
    axis_label = ", ".join(varying)

    print(f"\nExtra physical axis detected: {axis_label}")
    print("\nMESA-compatible cubes can only vary over:")
    print("  Teff, logg, metallicity")
    print(f"\nThis grid contains {n_combos} {axis_label} variant(s):")
    for key, g in sorted(groups.items()):
        vals_str = "  ".join(f"{col} = {val}" for col, val in key if col in varying)
        print(f"  {vals_str:<40}  {len(g['file_names'])} spectra")

    if n_combos <= threshold:
        print("\nWriting separate MESA-ready subgrids:")
        for key in sorted(groups):
            print(f"  {names[key]}/")
        print("\nThe parent directory will retain the full source grid.")
        print("Use one of the subdirectories as the spectral grid in MESA Colors.")
    else:
        print(f"\n{n_combos} combinations exceed the auto-split threshold ({threshold}).")


def _variant_rows(variants_index) -> Tuple[List[str], List[Dict[str, str]]]:
    """The columns of ``variants_index.csv`` and its rows, {column: value}."""
    with open(variants_index, newline="", encoding="utf-8") as fh:
        fields = [name.lstrip("#").strip() for name in fh.readline().strip().split(",")]
        return fields, list(csv.DictReader(fh, fieldnames=fields))


def _variant_paths(variants_index) -> List[str]:
    """The folder of each variant ``variants_index.csv`` lists, relative to the grid's folder."""
    return [row["path"] for row in _variant_rows(variants_index)[1]]


# --- The all-axes cube of a split grid ---

def _extra_axes(fields: List[str], rows: List[Dict[str, str]]) -> List[Tuple[str, List, Dict[Any, str]]]:
    """The extra axes of a split grid: each column of variants_index.csv (in its order) whose values differ between
    the variants, as (column, its values sorted, {value: its text}); values as collision._value reads them, so a
    column is numeric when every value reads as a number. A column with one value throughout is no axis."""
    axes = []
    for col in fields:
        if col in ("variant_name", "n_spectra", "path"):
            continue
        texts: Dict[Any, str] = {}
        for row in rows:
            texts.setdefault(_value(row[col]), row[col].strip())
        if len(texts) > 1:
            numeric = all(isinstance(v, float) for v in texts)
            axes.append((col, sorted(texts, key=None if numeric else str), texts))
    return axes


def move_aside_top_cube(model_dir: str) -> None:
    """A split grid's top folder holds no flux_cube.bin (MESA would read one there): one an earlier version or
    download left (the all-axes cube under its old name, or a variant's 3-axis copy) is renamed to
    flux_cube.bin.old, or .old2, .old3, ... when that is taken; nothing is deleted."""
    top = os.path.join(model_dir, "flux_cube.bin")
    if not os.path.lexists(top):
        return
    n = 1
    while os.path.lexists(aside := top + (".old" if n == 1 else f".old{n}")):
        n += 1
    os.replace(top, aside)
    print(f"Moved {top} aside to {aside}: a split grid's top folder holds no flux_cube.bin")


def write_all_axes_cube(model_dir: str, part_bytes: int = 256 * 2**20) -> str:
    """The all-axes flux_cube_nd.bin of a split grid (format in cube/fileio.py), assembled from its variant cubes; no
    spectrum is read. Its Teff, log g and [M/H] axes are the union of the variants', its extra axes the
    variants_index.csv columns that vary; each variant is placed at its own values, zeros fill the rest, so the cube
    equals each variant cube at that variant's nodes. A text axis stores its values as labels. Written a part of
    wavelengths at a time (about part_bytes), to flux_cube_nd.bin.tmp and then renamed over model_dir/flux_cube_nd.bin;
    a top-level flux_cube.bin is moved aside (:func:`move_aside_top_cube`). ValueError, before anything is written,
    when the variants' wavelengths differ."""
    fields, rows = _variant_rows(os.path.join(model_dir, "variants_index.csv"))
    extra = _extra_axes(fields, rows)
    cubes = [os.path.join(model_dir, row["path"], "flux_cube.bin") for row in rows]
    headers = [read_header(path) for path in cubes]
    for path, header in zip(cubes, headers):
        if header.all_axes:
            raise ValueError(f"{path} is an all-axes cube, not a variant's")
        if not np.array_equal(header.wavelengths, headers[0].wavelengths):
            raise ValueError(f"The variant cubes {cubes[0]} and {path} have different wavelengths, so no all-axes "
                             "cube can hold both; rebuild the grid")
    shared = [np.unique(np.concatenate([h.axes[k] for h in headers])) for k in range(3)]
    wavelengths = headers[0].wavelengths
    labels = {3 + k: [texts[v] for v in values] for k, (_, values, texts) in enumerate(extra)
              if not isinstance(values[0], float)}
    axes = [*shared, *(np.array(values if 3 + k not in labels else range(len(values)), dtype=float)
                       for k, (_, values, _) in enumerate(extra))]
    block_of = {tuple(values.index(_value(row[col])) for col, values, *_ in extra): (path, header)
                for row, path, header in zip(rows, cubes, headers)}

    nt, nl, nm, nw = (*map(len, shared), len(wavelengths))
    step = max(1, part_bytes // (8 * nt * nl * nm))
    target = os.path.join(model_dir, ALL_AXES_NAME)
    move_aside_top_cube(model_dir)
    with open(target + ".tmp", "wb") as fh:
        write_all_axes_header(fh, ["teff", "logg", "meta", *(col for col, *_ in extra)], axes, labels, wavelengths)
        for slow_first in itertools.product(*(range(len(values)) for _, values, *_ in reversed(extra))):
            path, header = block_of.get(tuple(reversed(slow_first)), (None, None))
            flux = None if path is None else np.memmap(path, dtype=np.float64, mode="r", offset=header.offset,
                                                        shape=header.shape)
            at = None if path is None else np.ix_(*(np.searchsorted(shared[k], header.axes[k]) for k in (2, 1, 0)))
            for w0 in range(0, nw, step):
                part = np.zeros((min(step, nw - w0), nm, nl, nt), dtype="<f8")
                if flux is not None:
                    part[(slice(None), *at)] = flux[w0:w0 + step]
                part.tofile(fh)
            del flux
    os.replace(target + ".tmp", target)
    print(f"All-axes flux cube: {target} (Teff, log g, [M/H], {', '.join(col for col, *_ in extra)}; "
          f"{len(block_of)} of {int(np.prod([len(v) for _, v, *_ in extra]))} combinations filled)")
    return target


def ensure_all_axes_cube(model_dir: str) -> None:
    """Write a split grid's all-axes cube when it has none, or still has a flux_cube.bin at its top (an earlier
    version's); a grid that is not split, or is complete, is left as it is."""
    if os.path.isfile(os.path.join(model_dir, "variants_index.csv")) and (
            not os.path.isfile(os.path.join(model_dir, ALL_AXES_NAME))
            or os.path.lexists(os.path.join(model_dir, "flux_cube.bin"))):
        write_all_axes_cube(model_dir)


def _remove_variants(folder: str) -> None:
    """Remove an earlier split from *folder*: the subgrid folders its variants_index.csv lists, its all-axes cube,
    then the index."""
    index = os.path.join(folder, "variants_index.csv")
    if not os.path.isfile(index):
        return
    for path in _variant_paths(index):
        subgrid = os.path.join(folder, path)
        if os.path.dirname(os.path.normpath(subgrid)) == os.path.normpath(folder) and os.path.isdir(subgrid):
            shutil.rmtree(subgrid)
    if os.path.isfile(os.path.join(folder, ALL_AXES_NAME)):
        os.remove(os.path.join(folder, ALL_AXES_NAME))
    os.remove(index)


def _link_spectra(subdir: str, file_names: List[str]) -> None:
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


def _parameter_arrays(group: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Teff, logg and [M/H] of a group's rows."""
    return np.array(group["teff"]), np.array(group["logg"]), np.array(group["meta"])


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


def _print_summary(output_file, teff_vals, logg_vals, meta_vals):
    print(f"\nFlux cube: {output_file}")
    for label, values, fmt, unit in (("Teff ", teff_vals, ".0f", " K"), ("logg ", logg_vals, ".2f", ""),
                                     ("[M/H]", meta_vals, ".2f", "")):
        print(f"  {label}: {values.min():{fmt}} – {values.max():{fmt}}{unit}")


# --- The strategies ---

def _confirm_large_split(model_dir: str, n_groups: int, threshold: int) -> bool:
    """Ask before splitting more than *threshold* variants; True to go ahead."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"{model_dir}: {n_groups} extra-axis combinations exceed the auto-split "
            f"threshold ({threshold}); no flux cube was built. Run interactively "
            "to confirm the split, or set strategy = \"filter\" or \"mean\" under "
            "[on_collision] in mesa_config.toml."
        )
    print("\nOptions:")
    print(f"  s) Split all {n_groups} combinations into subgrid directories")
    print("  a) Abort — no flux cube built")
    return input("  Choice [s/a]: ").strip().lower() == "s"


def _split_into_subgrids(model_dir: str, groups: Dict, extra_cols: List[str], lookup: Dict[str, List], file_col: str,
                         write_cube, threshold: int) -> None:
    """Strategy 'split': one MESA-ready subgrid folder per variant (its lookup_table.csv, links to the parent's
    spectra, its flux_cube.bin), then variants_index.csv, and lookup_table_full.csv as the record of the full grid.
    The parent's flux_cube_nd.bin is then the all-axes cube of every variant (:func:`write_all_axes_cube`), which MESA
    does not read: MESA reads a variant's folder. Above *threshold* variants it asks first."""
    names = _subdir_names(os.path.basename(os.path.abspath(model_dir)), list(groups))
    _print_split_announcement(groups, extra_cols, names, threshold)

    if len(groups) > threshold and not _confirm_large_split(model_dir, len(groups), threshold):
        print("Aborted. No flux cube built.")
        return

    index_entries: List[Dict[str, Any]] = []

    for key, g in sorted(groups.items()):
        name = names[key]
        subdir = os.path.join(model_dir, name)
        os.makedirs(subdir, exist_ok=True)

        print(f"\n  [{name}]  {len(g['file_names'])} spectra")

        filtered = _filter_lookup_by_files(lookup, g["file_names"], file_col)
        write_lookup_csv(filtered, os.path.join(subdir, "lookup_table.csv"))
        _link_spectra(subdir, g["file_names"])

        cube_path = os.path.join(subdir, "flux_cube.bin")
        _warn_averaged_rows(lookup, g["rows"], zip(g["teff"], g["logg"], g["meta"]), cube_path)
        write_cube(g, cube_path, label=name)
        _print_summary(cube_path, *_parameter_arrays(g))
        index_entries.append({"variant_name": name, "n_spectra": len(g["file_names"]), "path": name,
                              **g["extra_vals"]})

    index_path = os.path.join(model_dir, "variants_index.csv")
    write_lookup_csv(index_entries, index_path, columns=["variant_name"] + extra_cols + ["n_spectra", "path"])
    print(f"Variants index: {index_path}")

    src_lookup = os.path.join(model_dir, "lookup_table.csv")
    full_lookup = os.path.join(model_dir, "lookup_table_full.csv")
    if os.path.exists(src_lookup) and not os.path.exists(full_lookup):
        shutil.copy2(src_lookup, full_lookup)
        print("Full-grid lookup preserved: lookup_table_full.csv")
    write_all_axes_cube(model_dir)
    print("\nDone. The parent directory is the source collection; its flux_cube_nd.bin holds every variant on all axes.")
    print("Point MESA Colors to one of the subgrid directories listed above.")


def _build_mean_cube(groups: Dict, output_file: str, write_cube) -> None:
    """Strategy 'mean': one cube averaging every variant at each node."""
    print(
        "\nWARNING: strategy='mean' collapses extra axes by averaging. "
        "This is physically unsafe.\n"
        "Consider strategy='split' (the default) to get per-variant MESA-ready subgrids."
    )
    every_row: Dict[str, List] = {"file_names": [], "teff": [], "logg": [], "meta": []}
    for g in groups.values():
        for column, values in every_row.items():
            values.extend(g[column])

    print(f"Building mean cube (no subgrids) → {output_file}")
    write_cube(every_row, output_file, label="mean")


def _build_filtered_cube(cfg, groups: Dict, extra_cols: List[str], lookup: Dict[str, List], output_file: str,
                         write_cube) -> None:
    """Strategy 'filter': one cube from the single variant the config selects."""
    extra_unique = {col: sorted(set(lookup[col])) for col in extra_cols}
    resolved, errors = cfg.resolve_filter_values(extra_unique)
    if errors:
        raise ValueError("Filter config errors:\n" + "\n".join(f"  - {e}" for e in errors))

    print(f"Filter: {resolved}")

    def selected(key: Tuple) -> bool:  # values compared as the variants were grouped ("0" and "0.0" agree)
        values = dict(key)
        return all(_value(values[col]) == _value(val) for col, val in resolved.items())

    matched_group = next((g for key, g in groups.items() if selected(key)), None)

    if matched_group is None:
        available_variants = "\n".join(f"  {dict(k)}" for k in sorted(groups.keys()))
        raise ValueError(f"No variant matches filter {resolved}.\nAvailable variants:\n{available_variants}")

    print(f"Matched variant: {matched_group['extra_vals']} "
          f"({len(matched_group['file_names'])} spectra)")
    _warn_averaged_rows(lookup, matched_group["rows"],
                        zip(matched_group["teff"], matched_group["logg"], matched_group["meta"]), output_file)
    write_cube(matched_group, output_file)
