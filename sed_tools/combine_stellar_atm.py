#!/usr/bin/env python3
"""
Combine multiple stellar spectral grids into a unified flux cube.

Grids found by :func:`find_stellar_models` leave out earlier combine
outputs, so combining "all" grids never folds a previous combination's
copies of the same spectra back in. A grid counts as a combine output if
its lookup table has the ``source_model`` and ``original_file`` columns,
which :func:`save_combined_data` writes and nothing else in SED_Tools does
(this also finds outputs of earlier versions), or if its folder has the
default output name ``combined_models``. A combine output named
explicitly (``SED.combine(catalogs=[...])``) is still combined.
"""

import argparse
import logging
import os
import shutil
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ._resample import resample_to_grid
from .cube.fileio import write_header
from .ui_utils import banner
from .spectrum_io import read_text_spectrum

logger = logging.getLogger(__name__)

# scipy is only in the [full] extra, so the functions that need it import it
# themselves: this module, and `import sed_tools`, must work without it.


# The lookup columns only save_combined_data writes, and the default output name.
_COMBINED_COLUMNS = {"source_model", "original_file"}
_DEFAULT_COMBINED_NAME = "combined_models"


def find_stellar_models(base_dir=None):
    """``(name, path)`` of every grid in base_dir with a lookup table, except earlier
    combine outputs (see the module docstring), which are named on stdout."""
    if base_dir is None:
        from .models import STELLAR_DIR_DEFAULT
        base_dir = str(STELLAR_DIR_DEFAULT)

    model_dirs = []
    combined = []
    for item, item_path in _grids_with_lookup(base_dir):
        if _is_combined_output(item, os.path.join(item_path, "lookup_table.csv")):
            combined.append(item)
        else:
            model_dirs.append((item, item_path))
    if combined:
        print(f"Leaving out {', '.join(sorted(combined))}: written by an earlier combine. "
              "To combine one again, name it explicitly.")
    return model_dirs


def _grids_with_lookup(base_dir):
    """``(name, path)`` of every grid in base_dir with a lookup table, combine outputs included."""
    grids = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "lookup_table.csv")):
            grids.append((item, item_path))
    return grids


def _is_combined_output(name, lookup_file):
    """True for a grid an earlier combine wrote (see the module docstring)."""
    if name == _DEFAULT_COMBINED_NAME:
        return True
    try:
        with open(lookup_file) as f:
            header = f.readline()
    except (OSError, UnicodeDecodeError):
        return False  # offered as a grid; loading it reports the problem
    return _COMBINED_COLUMNS <= {c.strip() for c in header.strip().lstrip("#").split(",")}


def _count_spectra(lookup_file):
    """Rows of a lookup table as load_model_data reads them: after the header, not blank or '#'."""
    with open(lookup_file) as f:
        next(f, None)
        return sum(1 for line in f if line.strip() and not line.startswith("#"))


def select_models_interactive(model_dirs):
    """Present spectral grid options and ask until the answer is 'all', valid grid numbers, or q/b.

    The grids are returned in the order typed; q or b (back) returns [].
    """
    print("\nAvailable stellar spectral grids:")
    print("-" * 50)
    for idx, (name, path) in enumerate(model_dirs, start=1):
        lookup_file = os.path.join(path, "lookup_table.csv")
        try:
            n_models = _count_spectra(lookup_file)
        except (OSError, UnicodeDecodeError):
            n_models = "?"  # only the menu count; loading the grid reports the problem
        print(f"{idx}. {name} ({n_models} model spectra)")

    print("\nEnter spectral grid numbers (comma-separated) or 'all'; q or b to go back:")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "all":
            return model_dirs
        if user_input.lower() in ("q", "quit", "exit", "b", "back"):  # as in the pickers
            return []
        tokens = [x.strip() for x in user_input.split(",")]
        valid = all(t.isdecimal() and 1 <= int(t) <= len(model_dirs) for t in tokens)
        if valid and len(set(map(int, tokens))) == len(tokens):
            return [model_dirs[int(t) - 1] for t in tokens]
        print(f"Invalid selection {user_input!r}: enter distinct numbers from 1 to "
              f"{len(model_dirs)}, comma-separated, or 'all'.")


def load_model_data(model_path):
    """Load lookup table and extract parameter information."""
    lookup_file = os.path.join(model_path, "lookup_table.csv")

    with open(lookup_file, "r") as f:
        header = f.readline().strip().lstrip("#").strip()
        columns = [col.strip() for col in header.split(",")]

    df = pd.read_csv(lookup_file, comment="#", names=columns, skiprows=1)

    # Find columns (case-insensitive)
    file_col = columns[0]
    teff_col = logg_col = meta_col = None

    for col in columns:
        col_lower = col.lower()
        if "teff" in col_lower:
            teff_col = col
        elif "logg" in col_lower or "log(g)" in col_lower:
            logg_col = col
        elif "meta" in col_lower or "feh" in col_lower or "m/h" in col_lower:
            meta_col = col

    return {
        "files": df[file_col].values,
        "teff": df[teff_col].values if teff_col else np.zeros(len(df)),
        "logg": df[logg_col].values if logg_col else np.zeros(len(df)),
        "meta": df[meta_col].values if meta_col else np.zeros(len(df)),
        "model_dir": model_path,
    }


def load_sed(filepath):
    """Load wavelength and flux from a spectrum file, skipping comment lines."""
    return read_text_spectrum(filepath)


def prepare_sed(filepath, teff):
    """Load a spectrum. Spectra are expected to already be in standard units
    (wavelength in Angstroms, flux in erg/cm²/s/Å) via spectra_cleaner.py.
    No renormalization is applied here.

    Raises OSError if the file cannot be read, and ValueError if it gives
    fewer than two readable points.
    """
    wl, flux = load_sed(filepath)
    if wl.size < 2:
        raise ValueError(f"{filepath}: fewer than 2 readable wavelength/flux rows "
                         "(empty or not a spectrum)")
    return wl, flux


def create_unified_grid(all_models_data):
    """Create unified parameter grids from all spectral grids."""
    all_teff = np.concatenate([d["teff"] for d in all_models_data])
    all_logg = np.concatenate([d["logg"] for d in all_models_data])
    all_meta = np.concatenate([d["meta"] for d in all_models_data])

    # Get unique sorted values with tolerance cleaning
    def clean_grid(values, tol):
        grid = np.unique(np.sort(values))
        if len(grid) <= 1:
            return grid
        cleaned = [grid[0]]
        for val in grid[1:]:
            if abs(val - cleaned[-1]) > tol:
                cleaned.append(val)
        return np.array(cleaned)

    teff_grid = clean_grid(all_teff, tol=1.0)
    logg_grid = clean_grid(all_logg, tol=0.01)
    meta_grid = clean_grid(all_meta, tol=0.01)

    return teff_grid, logg_grid, meta_grid


def create_common_wavelength_grid(all_models_data, sample_size=20):
    """Create a common wavelength grid by sampling model spectra."""
    print("\nAnalyzing wavelength coverage...")

    min_wave = float("inf")
    max_wave = 0
    resolutions = []

    for model_data in all_models_data:
        n_sample = min(sample_size, len(model_data["files"]))
        indices = np.random.choice(len(model_data["files"]), n_sample, replace=False)

        for idx in indices:
            filepath = os.path.join(model_data["model_dir"], model_data["files"][idx])
            try:
                wavelengths, _ = load_sed(filepath)
            except OSError:
                # Every spectrum is read again when the cube is built, and that
                # pass raises one error listing all unreadable files.
                continue
            if len(wavelengths) <= 10:
                continue
            mask = (wavelengths >= 3000) & (wavelengths <= 25000)
            if mask.sum() <= 10:
                continue
            wl_subset = wavelengths[mask]
            min_wave = min(min_wave, wl_subset.min())
            max_wave = max(max_wave, wl_subset.max())
            resolutions.append(np.median(np.diff(wl_subset)))

    if not resolutions:
        names = ", ".join(os.path.basename(d["model_dir"]) for d in all_models_data)
        raise ValueError(
            f"No sampled spectrum from {names} has more than 10 points between 3000 and "
            "25000 Å, so no common wavelength grid can be built. Check that the spectra "
            "are readable and in Angstrom (run the spectra cleaner on each grid)."
        )
    typical_resolution = np.median(resolutions)
    grid_resolution = max(50.0, typical_resolution * 2)
    n_points = min(int((max_wave - min_wave) / grid_resolution) + 1, 5000)
    wavelength_grid = np.linspace(min_wave, max_wave, n_points)

    print(f"  Range: {min_wave:.0f} - {max_wave:.0f} Å")
    print(f"  Points: {len(wavelength_grid)}, Resolution: {grid_resolution:.1f} Å")

    return wavelength_grid


def identify_reference_model(all_models_data):
    """Identify reference spectral grid (prefer Kurucz)."""
    for i, model_data in enumerate(all_models_data):
        if "kurucz" in os.path.basename(model_data["model_dir"]).lower():
            print(f"Reference spectral grid: {os.path.basename(model_data['model_dir'])}")
            return i
    print(f"Reference spectral grid: {os.path.basename(all_models_data[0]['model_dir'])}")
    return 0


def compute_normalization_factors(target_data, reference_data, wavelength_grid):
    """Compute normalization factors to match target SEDs to reference spectral grid.

    Each target spectrum is scaled to its nearest reference spectrum in
    (Teff, log g, [M/H]) over their common wavelength range. Where no
    comparison is possible the factor is 1 (native scale kept) and a warning
    says how many spectra that affected. Raises ValueError listing every
    target spectrum that cannot be read or has non-finite parameters.
    """
    target_name = os.path.basename(target_data["model_dir"])
    ref_name = os.path.basename(reference_data["model_dir"])
    print(f"  Normalizing to: {ref_name}")

    if len(reference_data["files"]) == 0:
        logger.warning("Reference grid %s has no spectra; %s keeps its native flux scale.",
                       ref_name, target_name)
        return np.ones(len(target_data["files"]))

    # Build reference parameter array
    ref_params = np.column_stack([
        reference_data["teff"],
        reference_data["logg"],
        reference_data["meta"]
    ])
    valid_mask = np.isfinite(ref_params).all(axis=1)
    ref_params = ref_params[valid_mask]
    ref_files = np.array(reference_data["files"])[valid_mask]
    ref_teff = np.array(reference_data["teff"])[valid_mask]

    if len(ref_params) == 0:
        # build_combined_flux_cube then stops, listing the reference grid's non-finite rows.
        return np.ones(len(target_data["files"]))

    # Normalize parameters for distance calculation
    ranges = ref_params.max(axis=0) - ref_params.min(axis=0)
    ranges[ranges == 0] = 1.0
    ref_params_norm = (ref_params - ref_params.min(axis=0)) / ranges

    from scipy.spatial import KDTree

    tree = KDTree(ref_params_norm)
    norm_factors = []
    failures = []
    kept_native = Counter()

    for file, teff, logg, meta in zip(
        target_data["files"], target_data["teff"],
        target_data["logg"], target_data["meta"]
    ):
        file_target = os.path.join(target_data["model_dir"], file)
        if not all(np.isfinite([teff, logg, meta])):
            failures.append(f"{file_target}: non-finite parameters "
                            f"(teff={teff}, logg={logg}, meta={meta})")
            continue

        # Normalize query point
        query = np.array([teff, logg, meta])
        query_norm = (query - ref_params.min(axis=0)) / ranges
        _, idx = tree.query(query_norm)
        file_ref = os.path.join(reference_data["model_dir"], ref_files[idx])

        try:
            factor, reason = _normalization_factor(file_target, teff, file_ref, ref_teff[idx],
                                                   wavelength_grid)
        except (OSError, ValueError) as exc:  # unreadable target or reference spectrum
            failures.append(str(exc))
            continue
        if reason:
            kept_native[reason] += 1
        elif not (0.01 < factor < 100):
            print(f"  Warning: large normalization factor {factor:.3e} for {file} "
                  f"(teff={teff:.0f}, logg={logg:.2f}, [M/H]={meta:.2f}) — check unit consistency")
        norm_factors.append(factor)

    if failures:
        raise ValueError(
            f"Cannot normalise {len(failures)} spectra of {target_name} to {ref_name}; "
            "fix them or remove their rows from the lookup table:\n  " + "\n  ".join(failures)
        )
    for reason, count in kept_native.items():
        logger.warning("%d spectra of %s keep their native flux scale (not normalised to %s): %s.",
                       count, target_name, ref_name, reason)
    return np.array(norm_factors)


def _normalization_factor(file_target, teff, file_ref, ref_teff, wavelength_grid):
    """Ratio of reference to target integrated flux over their common wavelength range.

    Returns (factor, None), or (1.0, reason) when the two cannot be compared.
    """
    wl_tgt, flux_tgt = prepare_sed(file_target, teff)
    wl_ref, flux_ref = prepare_sed(file_ref, ref_teff)

    # Find common wavelength range
    wl_min = max(wl_tgt.min(), wl_ref.min(), wavelength_grid.min())
    wl_max = min(wl_tgt.max(), wl_ref.max(), wavelength_grid.max())
    mask = (wavelength_grid >= wl_min) & (wavelength_grid <= wl_max)
    if mask.sum() < 100:
        return 1.0, "fewer than 100 common wavelength points"

    from scipy.interpolate import interp1d

    wl_common = wavelength_grid[mask]
    interp_tgt = interp1d(wl_tgt, flux_tgt, bounds_error=False, fill_value=0)
    interp_ref = interp1d(wl_ref, flux_ref, bounds_error=False, fill_value=0)

    F_tgt = np.trapezoid(interp_tgt(wl_common), wl_common)
    F_ref = np.trapezoid(interp_ref(wl_common), wl_common)
    if not F_tgt > 0:
        return 1.0, "no positive target flux over the common range"
    return F_ref / F_tgt, None


def build_combined_flux_cube(all_models_data, teff_grid, logg_grid, meta_grid, wavelength_grid):
    """Build the combined flux cube from all spectral grids.

    Raises ValueError, before anything is written, listing every spectrum that
    cannot be read or has non-finite parameters.
    """
    n_teff, n_logg, n_meta, n_lambda = len(teff_grid), len(logg_grid), len(meta_grid), len(wavelength_grid)

    flux_cube = np.zeros((n_teff, n_logg, n_meta, n_lambda))
    filled_map = np.zeros((n_teff, n_logg, n_meta), dtype=bool)
    source_map = np.full((n_teff, n_logg, n_meta), -1, dtype=int)

    print(f"\nBuilding flux cube: {flux_cube.shape}")
    print(f"Memory: {flux_cube.nbytes / (1024**2):.1f} MB")

    # Get reference model and normalization factors
    ref_idx = identify_reference_model(all_models_data)
    ref_data = all_models_data[ref_idx]

    norm_factors = {}
    for model_idx, model_data in enumerate(all_models_data):
        if model_idx == ref_idx:
            norm_factors[model_idx] = np.ones(len(model_data["files"]))
        else:
            norm_factors[model_idx] = compute_normalization_factors(
                model_data, ref_data, wavelength_grid
            )

    # Fill the cube; later grids overwrite earlier ones at shared nodes
    failures = []
    for model_idx, model_data in enumerate(all_models_data):
        model_name = os.path.basename(model_data["model_dir"])
        print(f"\nProcessing {model_name}...")

        for i, (file, teff, logg, meta) in enumerate(tqdm(
            zip(model_data["files"], model_data["teff"],
                model_data["logg"], model_data["meta"]),
            total=len(model_data["files"]), desc=f"Spectral grid {model_idx + 1}"
        )):
            filepath = os.path.join(model_data["model_dir"], file)
            if not all(np.isfinite([teff, logg, meta])):
                failures.append(f"{filepath}: non-finite parameters "
                                f"(teff={teff}, logg={logg}, meta={meta})")
                continue
            try:
                wl, fl = prepare_sed(filepath, teff)
            except (OSError, ValueError) as exc:
                failures.append(str(exc))
                continue
            fl *= norm_factors[model_idx][i]

            node = (int(np.argmin(np.abs(teff_grid - teff))),
                    int(np.argmin(np.abs(logg_grid - logg))),
                    int(np.argmin(np.abs(meta_grid - meta))))
            flux_cube[node] = resample_to_grid(wl, fl, teff, wavelength_grid)
            filled_map[node] = True
            source_map[node] = model_idx

    if failures:
        raise ValueError(
            f"Cannot place {len(failures)} spectra in the combined cube; nothing was written. "
            "Fix them or remove their rows from the lookup table:\n  " + "\n  ".join(failures)
        )

    # Fill gaps with nearest neighbor
    empty_count = np.sum(~filled_map)
    if empty_count > 0:
        print(f"\nFilling {empty_count} empty grid points...")
        _fill_gaps(flux_cube, filled_map, source_map, teff_grid, logg_grid, meta_grid)

    return flux_cube, source_map


def _fill_gaps(flux_cube, filled_map, source_map, teff_grid, logg_grid, meta_grid):
    """Fill empty grid points using nearest neighbor interpolation."""
    # Normalize grids
    def norm(grid):
        r = grid.max() - grid.min()
        return (grid - grid.min()) / r if r > 0 else np.zeros_like(grid)

    teff_norm = norm(teff_grid)
    logg_norm = norm(logg_grid)
    meta_norm = norm(meta_grid)

    filled_indices = np.where(filled_map)
    filled_points = np.column_stack([
        teff_norm[filled_indices[0]],
        logg_norm[filled_indices[1]],
        meta_norm[filled_indices[2]]
    ])
    from scipy.spatial import KDTree

    tree = KDTree(filled_points)

    for i_t in range(len(teff_grid)):
        for i_g in range(len(logg_grid)):
            for i_m in range(len(meta_grid)):
                if not filled_map[i_t, i_g, i_m]:
                    query = [teff_norm[i_t], logg_norm[i_g], meta_norm[i_m]]
                    _, idx = tree.query(query)
                    src = (filled_indices[0][idx], filled_indices[1][idx], filled_indices[2][idx])
                    flux_cube[i_t, i_g, i_m, :] = flux_cube[src]
                    source_map[i_t, i_g, i_m] = source_map[src]


def save_combined_data(output_dir, teff_grid, logg_grid, meta_grid,
                       wavelength_grid, flux_cube, all_models_data):
    """Save the combined data and create unified lookup table.

    A spectrum already present in output_dir under its new name is not copied
    again. Raises OSError after writing the lookup table if any copy failed.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save binary flux cube (transposed for Fortran compatibility)
    binary_file = os.path.join(output_dir, "flux_cube.bin")
    with open(binary_file, "wb") as f:
        write_header(f, teff_grid, logg_grid, meta_grid, wavelength_grid)
        # Transpose to (wavelength, meta, logg, teff) for Fortran column-major order
        flux_cube.transpose(3, 2, 1, 0).astype(np.float64).tofile(f)

    print(f"\nSaved flux cube: {binary_file}")

    # Copy SED files and build lookup table
    print("\nCopying SED files...")
    lookup_data = []
    file_counter = 0
    copy_failures = []

    for model_data in tqdm(all_models_data, desc="Copying"):
        model_name = os.path.basename(model_data["model_dir"])
        for orig_file, teff, logg, meta in zip(
            model_data["files"], model_data["teff"],
            model_data["logg"], model_data["meta"]
        ):
            new_filename = f"{model_name}_{file_counter:06d}.txt"
            lookup_data.append({
                "file_name": new_filename,
                "teff": teff,
                "logg": logg,
                "meta": meta,
                "source_model": model_name,
                "original_file": orig_file,
            })

            src_path = os.path.join(model_data["model_dir"], orig_file)
            dst_path = os.path.join(output_dir, new_filename)
            if not os.path.exists(dst_path):
                try:
                    shutil.copy2(src_path, dst_path)
                except OSError as exc:
                    copy_failures.append(f"{src_path}: {exc}")

            file_counter += 1

    # Save lookup table
    lookup_df = pd.DataFrame(lookup_data)
    lookup_file = os.path.join(output_dir, "lookup_table.csv")
    with open(lookup_file, "w") as f:
        f.write("#file_name,teff,logg,meta,source_model,original_file\n")
        lookup_df.to_csv(f, index=False, header=False)

    print(f"Saved lookup table: {lookup_file}")
    print(f"Total model spectra: {len(lookup_df)}")

    if copy_failures:
        raise OSError(
            f"Could not copy {len(copy_failures)} spectra into {output_dir}; its lookup table "
            "lists them but the files are missing. Fix the cause and run the combine again "
            "(copies already made are kept):\n  " + "\n  ".join(copy_failures)
        )
    return lookup_df




def _nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


def _plot_meta_index(meta_grid):
    """[M/H] index to plot: the one nearest 0 if the grid spans 0, else the middle."""
    if meta_grid.min() <= 0.0 <= meta_grid.max():
        return _nearest_idx(meta_grid, 0.0)
    return len(meta_grid) // 2


def _sed_sample_points(teff_grid, logg_grid, meta_grid, meta_idx, flux_cube, source_map,
                       model_names):
    """Up to 15 nodes (5 Teff x 3 log g quantiles) at meta_idx that have positive flux."""
    teff_samples = np.unique(
        teff_grid[np.round(np.linspace(0, len(teff_grid) - 1, 5)).astype(int)]
    )
    logg_samples = np.unique(
        logg_grid[np.round(np.linspace(0, len(logg_grid) - 1, 3)).astype(int)]
    )

    sample_points = []
    for t_val in teff_samples:
        for g_val in logg_samples:
            ti = _nearest_idx(teff_grid, t_val)
            gi = _nearest_idx(logg_grid, g_val)
            flux = flux_cube[ti, gi, meta_idx, :]
            if not np.any(flux > 0):
                continue
            src_idx = int(source_map[ti, gi, meta_idx])
            src_name = model_names[src_idx] if 0 <= src_idx < len(model_names) else "unknown"
            sample_points.append((
                teff_grid[ti], logg_grid[gi], meta_grid[meta_idx],
                flux, src_name, src_idx
            ))
    return sample_points


def plot_sed_samples(teff_grid, logg_grid, meta_grid, wavelength_grid,
                     flux_cube, source_map, all_models_data, output_dir):
    """
    Plot representative SED samples from different regions of the combined
    parameter space, colour-coded by source spectral grid.

    Saves sed_samples.png in output_dir.
    """
    model_names = [os.path.basename(d["model_dir"]) for d in all_models_data]
    cmap_colors = plt.cm.tab10(np.linspace(0, 1, max(len(model_names), 1)))

    meta_idx = _plot_meta_index(meta_grid)
    sample_points = _sed_sample_points(teff_grid, logg_grid, meta_grid, meta_idx,
                                       flux_cube, source_map, model_names)
    if not sample_points:
        print("  Warning: no valid SED samples found — skipping sed_samples.png")
        return

    # Cap at 9 panels (3×3)
    if len(sample_points) > 9:
        indices = np.round(np.linspace(0, len(sample_points) - 1, 9)).astype(int)
        sample_points = [sample_points[i] for i in indices]

    n = len(sample_points)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (teff, logg, meta, flux, src_name, src_idx) in enumerate(sample_points):
        ax = axes[i]
        color = cmap_colors[src_idx % len(cmap_colors)]
        valid = flux > 0
        if valid.sum() > 1:
            ax.loglog(wavelength_grid[valid], flux[valid],
                      color=color, linewidth=0.8, label=src_name)
        ax.set_title(
            f"Teff={teff:.0f} K  log g={logg:.2f}  [M/H]={meta:.2f}\n"
            f"source: {src_name}",
            fontsize=8
        )
        ax.set_xlabel("Wavelength (Å)", fontsize=7)
        ax.set_ylabel("Flux (erg/cm²/s/Å)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"SED Samples Across Combined Parameter Space  ([M/H] ≈ {meta_grid[meta_idx]:.2f})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "sed_samples.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_file}")



def visualize_parameter_space(teff_grid, logg_grid, meta_grid, source_map,
                              all_models_data, output_dir,
                              wavelength_grid=None, flux_cube=None):
    """Create parameter space visualization."""
    print("\nCreating visualization...")

    model_names = [os.path.basename(d["model_dir"]) for d in all_models_data]
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    fig = plt.figure(figsize=(15, 10))

    # Teff vs logg
    ax1 = fig.add_subplot(2, 2, 1)
    for idx, (name, data) in enumerate(zip(model_names, all_models_data)):
        ax1.scatter(data["teff"], data["logg"], c=[colors[idx]], label=name, alpha=0.6, s=10)
    ax1.set_xlabel("Teff (K)")
    ax1.set_ylabel("log g")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # Teff vs [M/H]
    ax2 = fig.add_subplot(2, 2, 2)
    for idx, (name, data) in enumerate(zip(model_names, all_models_data)):
        ax2.scatter(data["teff"], data["meta"], c=[colors[idx]], label=name, alpha=0.6, s=10)
    ax2.set_xlabel("Teff (K)")
    ax2.set_ylabel("[M/H]")
    ax2.grid(True, alpha=0.3)

    # logg vs [M/H]
    ax3 = fig.add_subplot(2, 2, 3)
    for idx, (name, data) in enumerate(zip(model_names, all_models_data)):
        ax3.scatter(data["logg"], data["meta"], c=[colors[idx]], label=name, alpha=0.6, s=10)
    ax3.set_xlabel("log g")
    ax3.set_ylabel("[M/H]")
    ax3.grid(True, alpha=0.3)

    # Model density heatmap
    ax4 = fig.add_subplot(2, 2, 4)
    density = np.zeros((len(teff_grid), len(logg_grid)))
    for i in range(len(teff_grid)):
        for j in range(len(logg_grid)):
            density[i, j] = len(np.unique(source_map[i, j, :][source_map[i, j, :] >= 0]))

    im = ax4.imshow(density.T, origin="lower", aspect="auto",
                    extent=[teff_grid.min(), teff_grid.max(),
                           logg_grid.min(), logg_grid.max()], cmap="YlOrRd")
    ax4.set_xlabel("Teff (K)")
    ax4.set_ylabel("log g")
    ax4.set_title("Model spectrum density per grid point")
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "parameter_space.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {plot_file}")

    _print_combination_summary(teff_grid, logg_grid, meta_grid, source_map, model_names)

    if wavelength_grid is not None and flux_cube is not None:
        plot_sed_samples(teff_grid, logg_grid, meta_grid, wavelength_grid,
                         flux_cube, source_map, all_models_data, output_dir)


def _print_combination_summary(teff_grid, logg_grid, meta_grid, source_map, model_names):
    """Print the combined grid's extent and each source grid's share of its nodes."""
    banner("SUMMARY")
    print(f"Grid: {len(teff_grid)} × {len(logg_grid)} × {len(meta_grid)} = "
          f"{len(teff_grid) * len(logg_grid) * len(meta_grid):,} points")
    print(f"Teff: {teff_grid.min():.0f} - {teff_grid.max():.0f} K")
    print(f"logg: {logg_grid.min():.2f} - {logg_grid.max():.2f}")
    print(f"[M/H]: {meta_grid.min():.2f} - {meta_grid.max():.2f}")
    print("\nContributions:")
    for idx, name in enumerate(model_names):
        n = np.sum(source_map == idx)
        print(f"  {name}: {n:,} ({100 * n / source_map.size:.1f}%)")



def main():
    parser = argparse.ArgumentParser(description="Combine stellar spectral grids")
    parser.add_argument("--data_dir", default="../data/stellar_models/",
                       help="Base directory containing spectral grids")
    parser.add_argument("--output", default="combined_models",
                       help="Output directory name")
    parser.add_argument("--interactive", action="store_true", default=True,
                       help="Interactive spectral grid selection")
    args = parser.parse_args()

    model_dirs = find_stellar_models(args.data_dir)
    if not model_dirs:
        print(f"No spectral grids found in {args.data_dir}")
        return

    selected = select_models_interactive(model_dirs) if args.interactive else model_dirs
    if not selected:
        print("No spectral grids selected.")
        return

    print(f"\nSelected {len(selected)} spectral grids:")
    for name, _ in selected:
        print(f"  - {name}")

    # Load data
    print("\nLoading spectral grid data...")
    all_models_data = [load_model_data(path) for _, path in selected]

    # Create grids
    print("\nCreating parameter grids...")
    teff_grid, logg_grid, meta_grid = create_unified_grid(all_models_data)
    wavelength_grid = create_common_wavelength_grid(all_models_data)

    # Build cube
    flux_cube, source_map = build_combined_flux_cube(
        all_models_data, teff_grid, logg_grid, meta_grid, wavelength_grid
    )

    # Save
    output_dir = os.path.join(args.data_dir, args.output)
    save_combined_data(output_dir, teff_grid, logg_grid, meta_grid,
                      wavelength_grid, flux_cube, all_models_data)


    visualize_parameter_space(teff_grid, logg_grid, meta_grid, source_map,
                             all_models_data, output_dir,
                             wavelength_grid=wavelength_grid, flux_cube=flux_cube)

    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
