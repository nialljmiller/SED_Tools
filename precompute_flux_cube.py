#!/usr/bin/env python3
import argparse
import os
import struct

import numpy as np
from tqdm import tqdm


def load_sed(filepath, index):
    """Load a spectral energy distribution file."""
    wavelengths = []
    fluxes = []

    with open(filepath, "r") as f:
        # Skip header lines (comments)
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            try:
                values = line.split()
                if len(values) >= 2:
                    wavelengths.append(float(values[0]))
                    fluxes.append(float(values[1]))
            except (ValueError, IndexError):
                pass  # Skip any malformed lines

    return np.array(wavelengths), np.array(fluxes)


def load_lookup_table(lookup_file):
    """Load the lookup table with model parameters."""
    file_names = []
    teff_values = []
    logg_values = []
    meta_values = []

    with open(lookup_file, "r") as f:
        # Skip header line
        header = f.readline().strip().split(",")

        # Find column indices
        file_col = 0  # Assume first column is filename
        teff_col = None
        logg_col = None
        meta_col = None

        for i, col in enumerate(header):
            col = col.strip().lower()
            if col == "teff":
                teff_col = i
            elif col == "logg":
                logg_col = i
            elif col in ("meta", "metallicity", "[m/h]", "feh", "[fe/h]"):
                meta_col = i
            elif col in ("file_name", "filename", "file"):
                file_col = i

        # Read data rows
        for line in f:
            if line.strip():
                values = line.strip().split(",")
                if len(values) <= max(
                    file_col, teff_col or 0, logg_col or 0, meta_col or 0
                ):
                    continue  # Skip lines that don't have enough values

                file_names.append(values[file_col].strip())

                try:
                    teff = float(values[teff_col]) if teff_col is not None else 0.0
                    logg = float(values[logg_col]) if logg_col is not None else 0.0
                    meta = float(values[meta_col]) if meta_col is not None else 0.0

                    teff_values.append(teff)
                    logg_values.append(logg)
                    meta_values.append(meta)
                except (ValueError, IndexError):
                    # Skip rows with invalid values
                    file_names.pop()  # Remove the added filename

    print(
        f"Column indices found - teff: {teff_col}, logg: {logg_col}, meta: {meta_col}"
    )
    return (
        file_names,
        np.array(teff_values),
        np.array(logg_values),
        np.array(meta_values),
    )


def build_grids(teff_values, logg_values, meta_values):
    """Build unique sorted grids for Teff, logg, and metallicity."""
    teff_grid = np.unique(teff_values)
    logg_grid = np.unique(logg_values)
    meta_grid = np.unique(meta_values)

    print(f"Unique counts - Teff: {len(teff_grid)}, logg: {len(logg_grid)}, meta: {len(meta_grid)}")
    return teff_grid, logg_grid, meta_grid


def initialize_flux_cube(teff_grid, logg_grid, meta_grid, wavelengths):
    """Initialize the 4D flux cube (teff, logg, meta, wavelength)."""
    shape = (len(teff_grid), len(logg_grid), len(meta_grid), len(wavelengths))
    print(f"Initializing flux cube with shape: {shape}")
    return np.zeros(shape, dtype=np.float64)


def populate_flux_cube(
    model_dir, file_names, teff_values, logg_values, meta_values, teff_grid, logg_grid, meta_grid, wavelengths, flux_cube
):
    """Populate the 4D flux cube with flux values from SED files."""
    teff_to_idx = {val: idx for idx, val in enumerate(teff_grid)}
    logg_to_idx = {val: idx for idx, val in enumerate(logg_grid)}
    meta_to_idx = {val: idx for idx, val in enumerate(meta_grid)}

    for i, file_name in enumerate(tqdm(file_names, desc="Populating flux cube")):
        file_path = os.path.join(model_dir, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        teff_idx = teff_to_idx.get(teff_values[i], None)
        logg_idx = logg_to_idx.get(logg_values[i], None)
        meta_idx = meta_to_idx.get(meta_values[i], None)

        if teff_idx is None or logg_idx is None or meta_idx is None:
            print(f"Warning: Invalid indices for {file_name}, skipping...")
            continue

        file_wavelengths, file_fluxes = load_sed(file_path, i)

        # Interpolate fluxes to match common wavelengths
        try:
            interpolated_fluxes = np.interp(wavelengths, file_wavelengths, file_fluxes)
        except Exception as e:
            print(f"Interpolation failed for {file_name}: {e}")
            continue

        # Store the interpolated fluxes in the flux cube
        flux_cube[teff_idx, logg_idx, meta_idx, :] = interpolated_fluxes

    return flux_cube


def compute_common_wavelengths(model_dir, file_names, sample=10_000):
    """
    Compute a common wavelength grid by sampling files and taking the union
    of available wavelengths, then sorting and (optionally) thinning.
    """
    wl = []
    step = max(1, len(file_names) // max(1, min(len(file_names), sample)))
    for i, file_name in enumerate(file_names[::step]):
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            continue
        w, _ = load_sed(file_path, i)
        if w.size:
            wl.append(w)
    if not wl:
        raise RuntimeError("Could not determine a wavelength grid from any SEDs.")
    grid = np.unique(np.concatenate(wl))
    return grid


def write_flux_cube_binary(output_file, teff_grid, logg_grid, meta_grid, wavelengths, flux_cube):
    """Write flux cube to a binary file with header metadata."""
    print(f"Writing binary flux cube: {output_file}")
    with open(output_file, "wb") as f:
        # Header: number of teff, logg, meta, wavelength points
        f.write(
            struct.pack(
                "4i", len(teff_grid), len(logg_grid), len(meta_grid), len(wavelengths)
            )
        )

        # Write grid arrays
        teff_grid.astype(np.float64).tofile(f)
        logg_grid.astype(np.float64).tofile(f)
        meta_grid.astype(np.float64).tofile(f)
        wavelengths.astype(np.float64).tofile(f)

        # FIXED: Transpose to match Fortran's column-major order expectations
        # This swaps the dimension order to (wavelength, meta, logg, teff)
        t_flux = np.transpose(flux_cube, (3, 2, 1, 0))
        t_flux.astype(np.float64).tofile(f)
    print("Binary flux cube writing completed.")


def precompute_flux_cube(model_dir, output_file):
    """
    Main function to precompute the flux cube from model files in a directory.
    """
    # Infer lookup table path
    lookup_file = os.path.join(model_dir, "lookup_table.csv")
    if not os.path.isfile(lookup_file):
        # allow fallback to any file that looks like a lookup
        cand = [p for p in os.listdir(model_dir) if p.lower().startswith("lookup_table") and p.lower().endswith(".csv")]
        if cand:
            lookup_file = os.path.join(model_dir, cand[0])
        else:
            raise FileNotFoundError(f"lookup_table.csv not found in {model_dir}")

    file_names, teff_values, logg_values, meta_values = load_lookup_table(lookup_file)
    if len(file_names) == 0:
        raise RuntimeError("No entries in lookup table.")

    # Build unique grids
    teff_grid, logg_grid, meta_grid = build_grids(teff_values, logg_values, meta_values)

    # Common wavelength grid
    wavelengths = compute_common_wavelengths(model_dir, file_names)

    # Initialize and populate flux cube
    flux_cube = initialize_flux_cube(teff_grid, logg_grid, meta_grid, wavelengths)
    flux_cube = populate_flux_cube(
        model_dir,
        file_names,
        teff_values,
        logg_values,
        meta_values,
        teff_grid,
        logg_grid,
        meta_grid,
        wavelengths,
        flux_cube,
    )

    # Write to binary file
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    write_flux_cube_binary(output_file, teff_grid, logg_grid, meta_grid, wavelengths, flux_cube)

    print("Flux cube precomputation completed successfully.")
    print(f"Output file: {output_file}")
    print(f"Teff range: {teff_grid[0]} to {teff_grid[-1]}")
    print(f"logg range: {logg_grid[0]} to {logg_grid[-1]}")
    print(f"Metallicity range: {meta_grid[0]} to {meta_grid[-1]}")
    print(f"Flux cube shape: {flux_cube.shape}")
    print(f"Flux cube size: {flux_cube.nbytes / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    # Interactive-friendly CLI wrapper
    import sys
    parser = argparse.ArgumentParser(
        description="Precompute flux cube for stellar atmosphere models"
    )
    parser.add_argument("--model_dir", type=str, required=False,
                        help="Directory containing stellar model files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output binary file (default: <model_dir>/flux_cube.bin)")
    args = parser.parse_args()

    def discover_model_dirs(root="data/stellar_models"):
        cands = []
        for base in [root, "./"+root, "../"+root]:
            if os.path.isdir(base):
                for name in sorted(os.listdir(base)):
                    p = os.path.join(base, name)
                    if not os.path.isdir(p):
                        continue
                    has_lookup = os.path.isfile(os.path.join(p, "lookup_table.csv"))
                    has_txt = any(fn.lower().endswith(".txt") for fn in os.listdir(p))
                    has_h5 = any(fn.lower().endswith(".h5") for fn in os.listdir(p))
                    if has_lookup or has_txt or has_h5:
                        cands.append(p)
        # also include current dir if it looks like a model folder
        cur = os.getcwd()
        try:
            has_lookup = os.path.isfile(os.path.join(cur, "lookup_table.csv"))
            has_txt = any(fn.lower().endswith(".txt") for fn in os.listdir(cur))
            has_h5 = any(fn.lower().endswith(".h5") for fn in os.listdir(cur))
            if has_lookup or has_txt or has_h5:
                cands.append(cur)
        except Exception:
            pass
        return sorted(set(cands))

    if not args.model_dir:
        print("No --model_dir provided. Discovering local model directories…")
        cands = discover_model_dirs()
        if not cands:
            sys.exit("Could not find any model directories under data/stellar_models/*")
        for i,p in enumerate(cands,1):
            print(f"{i}. {p}")
        sel = input("Select a model directory by number: ").strip()
        if not sel.isdigit() or not (1 <= int(sel) <= len(cands)):
            sys.exit("Invalid selection.")
        args.model_dir = cands[int(sel)-1]

    if not args.output:
        args.output = os.path.join(args.model_dir, "flux_cube.bin")
        ans = input(f"Output file [{args.output}]: ").strip()
        if ans:
            args.output = ans

    precompute_flux_cube(args.model_dir, args.output)
