"""Grids whose extra lookup columns (alpha, vturb, ...) vary: one MESA-ready subgrid per variant (the
'split' strategy), a mean cube, or one filtered variant; subgrid names and variants_index.csv.
precompute_flux_cube in cube/build.py chooses the strategy; build.py writes each cube.
"""

import logging
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from . import build  # build imports this module too, so its names are looked up when called
from ..grids._lookup_io import write_lookup_csv

logger = logging.getLogger("sed_tools.precompute_flux_cube")


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

    Returns {extra_key_tuple: {"rows": [...], "file_names": [...], "teff": [...], ...}},
    where "rows" are the row numbers in *lookup* and extra_key_tuple is a tuple
    of (col_name, value) pairs sorted by col_name for deterministic ordering.
    """
    groups: Dict[Tuple, Dict] = {}

    for i, fname in enumerate(lookup[file_col]):
        extra_key = tuple(
            sorted((col, lookup[col][i]) for col in extra_cols)
        )
        if extra_key not in groups:
            groups[extra_key] = {
                "rows": [], "file_names": [], "teff": [], "logg": [], "meta": [],
                "extra_vals": dict(extra_key),
            }
        g = groups[extra_key]
        g["rows"].append(i)
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
    write_lookup_csv(entries, path, columns=["variant_name"] + extra_cols + ["n_spectra", "path"])
    print(f"Variants index: {path}")


def _print_split_announcement(
    groups: Dict,
    extra_cols: List[str],
    model_name: str,
    threshold: int,
) -> None:
    """Print the structured 'Extra physical axis detected' message.

    Only the axes that vary between the variants are named: a column with
    one value throughout (lh and vtur in Kurucz) is no variant axis.  The
    subgrid directory names still carry every extra column.
    """
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
        build._warn_averaged_rows(lookup, g["rows"], zip(g["teff"], g["logg"], g["meta"]), cube_path)
        build._populate_and_write(
            model_dir, g["file_names"],
            teff_arr, logg_arr, meta_arr,
            wavelengths, cube_path, usable_ram,
            label=name,
        )
        build._print_summary(cube_path, teff_arr, logg_arr, meta_arr)

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
    """Ask before splitting more than SPLIT_THRESHOLD (build.py) variants; True to go ahead."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"{model_dir}: {n_groups} extra-axis combinations exceed the auto-split "
            f"threshold ({build.SPLIT_THRESHOLD}); no flux cube was built. Run interactively "
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
    _print_split_announcement(groups, extra_cols, model_name, build.SPLIT_THRESHOLD)

    if len(groups) > build.SPLIT_THRESHOLD and not _confirm_large_split(model_dir, len(groups)):
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
    build._populate_and_write(
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
    build._warn_averaged_rows(lookup, matched_group["rows"],
                        zip(matched_group["teff"], matched_group["logg"], matched_group["meta"]), output_file)
    build._populate_and_write(
        model_dir,
        matched_group["file_names"],
        np.array(matched_group["teff"]),
        np.array(matched_group["logg"]),
        np.array(matched_group["meta"]),
        wavelengths, output_file, usable_ram,
    )
