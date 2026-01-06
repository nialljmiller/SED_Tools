#!/usr/bin/env python3
"""
Combine multiple stellar atmosphere models into a unified flux cube.
"""

import argparse
import os
import shutil
import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from tqdm import tqdm

SIGMA = 5.670374419e-5  # Stefan-Boltzmann constant (erg s-1 cm-2 K-4)


def renorm_to_sigmaT4(wl, flux, Teff):
    """Renormalize flux so bolometric integral equals σT⁴."""
    Fbol_model = simps(flux, wl)
    Fbol_target = SIGMA * Teff**4
    return flux * (Fbol_target / Fbol_model)


def find_stellar_models(base_dir="../data/stellar_models/"):
    """Find all model directories containing lookup tables."""
    model_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        lookup_file = os.path.join(item_path, "lookup_table.csv")
        if os.path.isdir(item_path) and os.path.exists(lookup_file):
            model_dirs.append((item, item_path))
    return model_dirs


def select_models_interactive(model_dirs):
    """Present model options to user and get selection."""
    print("\nAvailable stellar atmosphere models:")
    print("-" * 50)
    for idx, (name, path) in enumerate(model_dirs, start=1):
        lookup_file = os.path.join(path, "lookup_table.csv")
        try:
            df = pd.read_csv(lookup_file, comment="#")
            n_models = len(df)
        except Exception:
            n_models = "?"
        print(f"{idx}. {name} ({n_models} models)")

    print("\nEnter model numbers (comma-separated) or 'all':")
    user_input = input("> ").strip()

    if user_input.lower() == "all":
        return model_dirs

    try:
        indices = [int(x.strip()) - 1 for x in user_input.split(",")]
        return [model_dirs[i] for i in indices if 0 <= i < len(model_dirs)]
    except (ValueError, IndexError):
        print("Invalid input. Using all models.")
        return model_dirs


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
    """Load wavelength and flux from a spectrum file."""
    return np.loadtxt(filepath, unpack=True)


def prepare_sed(filepath, teff):
    """Load and renormalize a spectrum."""
    wl, flux = load_sed(filepath)
    flux = renorm_to_sigmaT4(wl, flux, teff)
    return wl, flux


def create_unified_grid(all_models_data):
    """Create unified parameter grids from all models."""
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
    """Create a common wavelength grid by sampling models."""
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
                if len(wavelengths) > 10:
                    mask = (wavelengths >= 3000) & (wavelengths <= 25000)
                    if mask.sum() > 10:
                        wl_subset = wavelengths[mask]
                        min_wave = min(min_wave, wl_subset.min())
                        max_wave = max(max_wave, wl_subset.max())
                        resolutions.append(np.median(np.diff(wl_subset)))
            except Exception:
                continue

    typical_resolution = np.median(resolutions) if resolutions else 50.0
    grid_resolution = max(50.0, typical_resolution * 2)
    n_points = min(int((max_wave - min_wave) / grid_resolution) + 1, 5000)
    wavelength_grid = np.linspace(min_wave, max_wave, n_points)

    print(f"  Range: {min_wave:.0f} - {max_wave:.0f} Å")
    print(f"  Points: {len(wavelength_grid)}, Resolution: {grid_resolution:.1f} Å")

    return wavelength_grid


def identify_reference_model(all_models_data):
    """Identify reference model (prefer Kurucz)."""
    for i, model_data in enumerate(all_models_data):
        if "kurucz" in os.path.basename(model_data["model_dir"]).lower():
            print(f"Reference model: {os.path.basename(model_data['model_dir'])}")
            return i
    print(f"Reference model: {os.path.basename(all_models_data[0]['model_dir'])}")
    return 0


def compute_normalization_factors(target_data, reference_data, wavelength_grid):
    """Compute normalization factors to match target SEDs to reference model."""
    print(f"  Normalizing to: {os.path.basename(reference_data['model_dir'])}")

    if len(reference_data["files"]) == 0:
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
        return np.ones(len(target_data["files"]))

    # Normalize parameters for distance calculation
    ranges = ref_params.max(axis=0) - ref_params.min(axis=0)
    ranges[ranges == 0] = 1.0
    ref_params_norm = (ref_params - ref_params.min(axis=0)) / ranges

    tree = KDTree(ref_params_norm)
    norm_factors = []

    for i, (file, teff, logg, meta) in enumerate(zip(
        target_data["files"], target_data["teff"],
        target_data["logg"], target_data["meta"]
    )):
        if not all(np.isfinite([teff, logg, meta])):
            norm_factors.append(1.0)
            continue

        # Normalize query point
        query = np.array([teff, logg, meta])
        query_norm = (query - ref_params.min(axis=0)) / ranges

        _, idx = tree.query(query_norm)

        # Load and compare SEDs
        file_target = os.path.join(target_data["model_dir"], file)
        file_ref = os.path.join(reference_data["model_dir"], ref_files[idx])

        try:
            wl_tgt, flux_tgt = prepare_sed(file_target, teff)
            wl_ref, flux_ref = prepare_sed(file_ref, ref_teff[idx])

            # Find common wavelength range
            wl_min = max(wl_tgt.min(), wl_ref.min(), wavelength_grid.min())
            wl_max = min(wl_tgt.max(), wl_ref.max(), wavelength_grid.max())
            mask = (wavelength_grid >= wl_min) & (wavelength_grid <= wl_max)

            if mask.sum() < 100:
                norm_factors.append(1.0)
                continue

            wl_common = wavelength_grid[mask]
            interp_tgt = interp1d(wl_tgt, flux_tgt, bounds_error=False, fill_value=0)
            interp_ref = interp1d(wl_ref, flux_ref, bounds_error=False, fill_value=0)

            F_tgt = np.trapz(interp_tgt(wl_common), wl_common)
            F_ref = np.trapz(interp_ref(wl_common), wl_common)

            factor = F_ref / F_tgt if F_tgt > 0 else 1.0
            norm_factors.append(factor if 0.01 < factor < 100 else 1.0)

        except Exception:
            norm_factors.append(1.0)

    return np.array(norm_factors)


def build_combined_flux_cube(all_models_data, teff_grid, logg_grid, meta_grid, wavelength_grid):
    """Build the combined flux cube from all models."""
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

    # Fill the cube
    for model_idx, model_data in enumerate(all_models_data):
        model_name = os.path.basename(model_data["model_dir"])
        print(f"\nProcessing {model_name}...")

        for i, (file, teff, logg, meta) in enumerate(tqdm(
            zip(model_data["files"], model_data["teff"],
                model_data["logg"], model_data["meta"]),
            total=len(model_data["files"]), desc=f"Model {model_idx + 1}"
        )):
            i_teff = np.clip(np.searchsorted(teff_grid, teff), 0, n_teff - 1)
            i_logg = np.clip(np.searchsorted(logg_grid, logg), 0, n_logg - 1)
            i_meta = np.clip(np.searchsorted(meta_grid, meta), 0, n_meta - 1)

            filepath = os.path.join(model_data["model_dir"], file)
            try:
                wl, fl = prepare_sed(filepath, teff)
                fl *= norm_factors[model_idx][i]

                # Interpolate in log space
                log_fl = np.log10(np.maximum(fl, 1e-50))
                log_interp = np.interp(wavelength_grid, wl, log_fl,
                                       left=log_fl[0], right=log_fl[-1])

                flux_cube[i_teff, i_logg, i_meta, :] = 10**log_interp
                filled_map[i_teff, i_logg, i_meta] = True
                source_map[i_teff, i_logg, i_meta] = model_idx
            except Exception:
                continue

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
    """Save the combined data and create unified lookup table."""
    os.makedirs(output_dir, exist_ok=True)

    # Save binary flux cube (transposed for Fortran compatibility)
    binary_file = os.path.join(output_dir, "flux_cube.bin")
    with open(binary_file, "wb") as f:
        f.write(struct.pack("4i", len(teff_grid), len(logg_grid),
                           len(meta_grid), len(wavelength_grid)))
        teff_grid.astype(np.float64).tofile(f)
        logg_grid.astype(np.float64).tofile(f)
        meta_grid.astype(np.float64).tofile(f)
        wavelength_grid.astype(np.float64).tofile(f)
        # Transpose to (wavelength, meta, logg, teff) for Fortran column-major order
        flux_cube.transpose(3, 2, 1, 0).astype(np.float64).tofile(f)

    print(f"\nSaved flux cube: {binary_file}")

    # Copy SED files and build lookup table
    print("\nCopying SED files...")
    lookup_data = []
    file_counter = 0

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
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"\nWarning: Could not copy {src_path}: {e}")

            file_counter += 1

    # Save lookup table
    lookup_df = pd.DataFrame(lookup_data)
    lookup_file = os.path.join(output_dir, "lookup_table.csv")
    with open(lookup_file, "w") as f:
        f.write("#file_name,teff,logg,meta,source_model,original_file\n")
        lookup_df.to_csv(f, index=False, header=False)

    print(f"Saved lookup table: {lookup_file}")
    print(f"Total models: {len(lookup_df)}")

    return lookup_df


def visualize_parameter_space(teff_grid, logg_grid, meta_grid, source_map,
                              all_models_data, output_dir):
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
    ax4.set_title("Model density per grid point")
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "parameter_space.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {plot_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
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
    parser = argparse.ArgumentParser(description="Combine stellar atmosphere models")
    parser.add_argument("--data_dir", default="../data/stellar_models/",
                       help="Base directory containing stellar models")
    parser.add_argument("--output", default="combined_models",
                       help="Output directory name")
    parser.add_argument("--interactive", action="store_true", default=True,
                       help="Interactive model selection")
    args = parser.parse_args()

    model_dirs = find_stellar_models(args.data_dir)
    if not model_dirs:
        print(f"No stellar models found in {args.data_dir}")
        return

    selected = select_models_interactive(model_dirs) if args.interactive else model_dirs
    if not selected:
        print("No models selected.")
        return

    print(f"\nSelected {len(selected)} models:")
    for name, _ in selected:
        print(f"  - {name}")

    # Load data
    print("\nLoading model data...")
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

    # Visualize
    visualize_parameter_space(teff_grid, logg_grid, meta_grid, source_map,
                             all_models_data, output_dir)

    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()