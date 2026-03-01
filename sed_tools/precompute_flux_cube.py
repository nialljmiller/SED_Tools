#!/usr/bin/env python3
import argparse
import os
import struct
import time

import numpy as np
from tqdm import tqdm

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


def load_sed(filepath):
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


def initialize_flux_cube(teff_grid, logg_grid, meta_grid, wavelengths, tmp_path=None):
    """Allocate the 4D flux cube — in RAM if it fits, disk-backed memmap if not."""
    shape    = (len(teff_grid), len(logg_grid), len(meta_grid), len(wavelengths))
    size_gib = np.prod(shape) * 8 / (1024 ** 3)
    if tmp_path:
        print(f"Initializing flux cube with shape: {shape} ({size_gib:.1f} GiB, disk-backed memmap)")
        return np.memmap(tmp_path, dtype=np.float64, mode="w+", shape=shape)
    else:
        print(f"Initializing flux cube with shape: {shape} ({size_gib:.1f} GiB, in RAM)")
        return np.zeros(shape, dtype=np.float64)


def populate_flux_cube(
    model_dir, file_names, teff_values, logg_values, meta_values, teff_grid, logg_grid, meta_grid, wavelengths, flux_cube
):
    """Populate the 4D flux cube with flux values from SED files."""
    teff_to_idx = {val: idx for idx, val in enumerate(teff_grid)}
    logg_to_idx = {val: idx for idx, val in enumerate(logg_grid)}
    meta_to_idx = {val: idx for idx, val in enumerate(meta_grid)}

    n       = len(file_names)
    skipped = 0
    t_start = time.time()

    for i, file_name in enumerate(file_names):
        file_path = os.path.join(model_dir, file_name)

        if not os.path.exists(file_path):
            print(f"\nWarning: File not found: {file_path}")
            skipped += 1
            continue

        teff_idx = teff_to_idx.get(teff_values[i])
        logg_idx = logg_to_idx.get(logg_values[i])
        meta_idx = meta_to_idx.get(meta_values[i])

        if teff_idx is None or logg_idx is None or meta_idx is None:
            print(f"\nWarning: Invalid indices for {file_name}, skipping...")
            skipped += 1
            continue

        file_wavelengths, file_fluxes = load_sed(file_path)

        try:
            interpolated_fluxes = np.interp(wavelengths, file_wavelengths, file_fluxes)
        except Exception as e:
            print(f"\nInterpolation failed for {file_name}: {e}")
            skipped += 1
            continue

        flux_cube[teff_idx, logg_idx, meta_idx, :] = interpolated_fluxes

        elapsed = time.time() - t_start
        done    = i + 1
        rate    = done / elapsed if elapsed > 0 else 1.0
        eta     = (n - done) / rate
        pct     = 100.0 * done / n
        print(
            f"\r  spectra {done}/{n} ({pct:.1f}%)  "
            f"skipped {skipped}  "
            f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s   ",
            end="", flush=True,
        )

    print(f"\n  Done — {n - skipped} spectra loaded, {skipped} skipped.")
    return flux_cube


def compute_common_wavelengths(model_dir, file_names, n_sample=5):
    """
    Compute a common wavelength grid by reading a small fixed number of files.

    Stellar atmosphere catalogs are uniform — every file shares the same
    wavelength grid — so reading 5 files is sufficient to establish it.
    Reading thousands of files (each potentially millions of lines) is
    what caused the original all-night hang.
    """
    if len(file_names) == 0:
        raise RuntimeError("No files in lookup table.")

    # Pick evenly spaced indices across the catalog, capped at n_sample
    n      = len(file_names)
    step   = max(1, n // n_sample)
    subset = [file_names[i] for i in range(0, n, step)][:n_sample]

    wl      = []
    checked = 0
    for file_name in subset:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            continue
        w, _ = load_sed(file_path)
        if w.size:
            wl.append(w)
        checked += 1

    if not wl:
        raise RuntimeError(
            f"Could not determine a wavelength grid — sampled {checked} files, "
            f"all were empty or unreadable. Check that the downloaded spectra are valid."
        )

    grid = np.unique(np.concatenate(wl))
    print(f"Wavelength grid: {len(grid)} points from {len(wl)}/{checked} sampled files "
          f"(range {grid[0]:.1f}–{grid[-1]:.1f} Å)")
    return grid


def write_flux_cube_binary(output_file, teff_grid, logg_grid, meta_grid, wavelengths, flux_cube, usable_ram_bytes=None):
    """
    Write flux cube to binary file with header metadata.

    Transposes (T,L,M,W) → (W,M,L,T) in wavelength chunks sized to available
    RAM.  If the full array fits, it is done in one shot.
    """
    nteff = len(teff_grid)
    nlogg = len(logg_grid)
    nmeta = len(meta_grid)
    nwave = len(wavelengths)
    tlm   = nteff * nlogg * nmeta

    if usable_ram_bytes is None:
        usable_ram_bytes = int(_available_ram_bytes() * 0.75)

    # ×2: hold source chunk + transposed copy simultaneously
    w_chunk  = max(1, usable_ram_bytes // (tlm * 2 * 8))
    n_chunks = (nwave + w_chunk - 1) // w_chunk
    total_gib = nwave * tlm * 8 / (1024 ** 3)

    if n_chunks == 1:
        print(f"Writing binary flux cube (single-pass, {total_gib:.1f} GiB): {output_file}")
    else:
        chunk_gib = w_chunk * tlm * 8 / (1024 ** 3)
        print(f"Writing binary flux cube ({total_gib:.1f} GiB, {n_chunks} chunks x {chunk_gib:.2f} GiB): {output_file}")
        print( "  I/O bound — expect minutes to hours for large cubes.")

    t_start = time.time()

    with open(output_file, "wb") as f:
        f.write(struct.pack("4i", nteff, nlogg, nmeta, nwave))
        teff_grid.astype(np.float64).tofile(f)
        logg_grid.astype(np.float64).tofile(f)
        meta_grid.astype(np.float64).tofile(f)
        wavelengths.astype(np.float64).tofile(f)

        for chunk_i in range(n_chunks):
            w0      = chunk_i * w_chunk
            w1      = min(w0 + w_chunk, nwave)
            chunk   = np.array(flux_cube[:, :, :, w0:w1])
            chunk_t = np.ascontiguousarray(np.transpose(chunk, (3, 2, 1, 0)))
            del chunk
            chunk_t.astype(np.float64).tofile(f)
            del chunk_t

            if n_chunks > 1:
                elapsed = time.time() - t_start
                done    = chunk_i + 1
                rate    = done / elapsed if elapsed > 0 else 1.0
                eta     = (n_chunks - done) / rate
                print(
                    f"\r  chunk {done}/{n_chunks} ({100*done/n_chunks:.1f}%)  "
                    f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s   ",
                    end="", flush=True,
                )

    if n_chunks > 1:
        print()
    print("Binary flux cube writing completed.")


def precompute_flux_cube(model_dir, output_file):
    """
    Precompute the flux cube from model files in a directory.

    Scales automatically to available RAM:
      - Cube fits in 75% of available RAM → allocated in RAM, one-shot.
      - Cube too large                    → disk-backed memmap; transpose
                                            chunked to available RAM.
    """
    lookup_file = os.path.join(model_dir, "lookup_table.csv")
    if not os.path.isfile(lookup_file):
        cand = [p for p in os.listdir(model_dir)
                if p.lower().startswith("lookup_table") and p.lower().endswith(".csv")]
        if cand:
            lookup_file = os.path.join(model_dir, cand[0])
        else:
            raise FileNotFoundError(f"lookup_table.csv not found in {model_dir}")

    file_names, teff_values, logg_values, meta_values = load_lookup_table(lookup_file)
    if len(file_names) == 0:
        raise RuntimeError("No entries in lookup table.")

    teff_grid, logg_grid, meta_grid = build_grids(teff_values, logg_values, meta_values)
    wavelengths = compute_common_wavelengths(model_dir, file_names)

    cube_bytes    = len(teff_grid) * len(logg_grid) * len(meta_grid) * len(wavelengths) * 8
    cube_gib      = cube_bytes / (1024 ** 3)
    available     = _available_ram_bytes()
    usable        = int(available * 0.75)

    print(f"Cube size:      {cube_gib:.1f} GiB")
    print(f"Available RAM:  {available / (1024**3):.1f} GiB  (using up to 75% = {usable / (1024**3):.1f} GiB)")

    tmp_path = None
    if cube_bytes <= usable:
        print("Strategy:       in-RAM  (cube fits comfortably)")
        flux_cube = initialize_flux_cube(teff_grid, logg_grid, meta_grid, wavelengths)
    else:
        print("Strategy:       disk-backed memmap  (cube exceeds available RAM)")
        out_dir   = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(out_dir, exist_ok=True)
        tmp_path  = output_file + ".memmap.tmp"
        flux_cube = initialize_flux_cube(teff_grid, logg_grid, meta_grid, wavelengths, tmp_path)

    try:
        print("\nPopulating spectra:")
        flux_cube = populate_flux_cube(
            model_dir, file_names, teff_values, logg_values, meta_values,
            teff_grid, logg_grid, meta_grid, wavelengths, flux_cube,
        )

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        write_flux_cube_binary(
            output_file, teff_grid, logg_grid, meta_grid, wavelengths, flux_cube,
            usable_ram_bytes=usable,
        )
    finally:
        del flux_cube
        if tmp_path and os.path.exists(tmp_path):
            print(f"Cleaning up temp file: {tmp_path}")
            os.remove(tmp_path)

    print("Flux cube precomputation completed successfully.")
    print(f"Output file: {output_file}")
    print(f"Teff range: {teff_grid[0]} to {teff_grid[-1]}")
    print(f"logg range: {logg_grid[0]} to {logg_grid[-1]}")
    print(f"Metallicity range: {meta_grid[0]} to {meta_grid[-1]}")
    print(f"Flux cube shape: ({len(teff_grid)}, {len(logg_grid)}, {len(meta_grid)}, {len(wavelengths)})")
    print(f"Flux cube size: {cube_gib:.2f} GiB")


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