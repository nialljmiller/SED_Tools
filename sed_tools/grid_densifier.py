"""
sed_tools/grid_densifier.py

Densify a coarse Teff grid in an existing flux_cube.bin by running
SEDGenerator over every gap Teff value across all (logg, meta) nodes.

This is a grid runner around ml_sed_generator.SEDGenerator.  It does not
implement any interpolation — it generates physically grounded SEDs from
the trained network at the requested Teff values, giving MESA's Hermite
interpolation short enough intervals that the discrete photometry jumps
disappear.

A blackbody fallback (π·B_λ scaled to the nearest real grid SED's
bolometric flux) is used when no ML model is available or when a requested
Teff falls outside the generator's training range.

Usage (CLI)
-----------
    python -m sed_tools.grid_densifier \\
        --flux-cube  data/stellar_models/tmap/flux_cube.bin \\
        --output     data/stellar_models/tmap_dense/flux_cube.bin \\
        --teff-spacing 1000 \\
        --ml-model   models/sed_generator_tmap

API
---
    from sed_tools.grid_densifier import densify_grid

    densify_grid(
        src="data/stellar_models/tmap/flux_cube.bin",
        dst="data/stellar_models/tmap_dense/flux_cube.bin",
        teff_spacing=1000,
        ml_model="models/sed_generator_tmap",
    )
"""

from __future__ import annotations

import argparse
import logging
import os
import struct
import time
from typing import Optional

from ._resample import resample_to_grid

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binary cube I/O
# ---------------------------------------------------------------------------

def _flux_offset(nt, nl, nm, nw):
    """Byte offset where the flux data starts in a flux_cube.bin."""
    return 16 + (nt + nl + nm + nw) * 8


def _read_cube_header(path):
    """Read grids and wavelengths only — does not load flux into RAM."""
    with open(path, "rb") as f:
        nt, nl, nm, nw = struct.unpack("iiii", f.read(16))
        teff = np.frombuffer(f.read(8 * nt), dtype=np.float64).copy()
        logg = np.frombuffer(f.read(8 * nl), dtype=np.float64).copy()
        meta = np.frombuffer(f.read(8 * nm), dtype=np.float64).copy()
        wl   = np.frombuffer(f.read(8 * nw), dtype=np.float64).copy()
    return nt, nl, nm, nw, teff, logg, meta, wl


def _open_flux_memmap(path, nt, nl, nm, nw, mode="r"):
    """Return a memmap view of the flux data in an existing cube file."""
    offset = _flux_offset(nt, nl, nm, nw)
    return np.memmap(path, dtype=np.float64, mode=mode,
                     offset=offset, shape=(nt, nl, nm, nw))


def _init_output_cube(path, teff, logg, meta, wl, nt_new):
    """Write header to dst and return a writable memmap over the flux region."""
    nl, nm, nw = len(logg), len(meta), len(wl)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("iiii", nt_new, nl, nm, nw))
        f.write(teff.astype(np.float64).tobytes())
        f.write(logg.astype(np.float64).tobytes())
        f.write(meta.astype(np.float64).tobytes())
        f.write(wl.astype(np.float64).tobytes())
        # Pre-allocate flux region as zeros
        f.write(b"\x00" * (nt_new * nl * nm * nw * 8))
    offset = _flux_offset(nt_new, nl, nm, nw)
    return np.memmap(path, dtype=np.float64, mode="r+",
                     offset=offset, shape=(nt_new, nl, nm, nw))


# ---------------------------------------------------------------------------
# Blackbody fallback
# ---------------------------------------------------------------------------

_H  = 6.62607015e-27
_C  = 2.99792458e10
_KB = 1.380649e-16


def _blackbody(wl_ang, teff):
    """π·B_λ(T) in erg/cm²/s/Å."""
    wl_cm = wl_ang * 1e-8
    exp = np.minimum((_H * _C) / (wl_cm * _KB * teff), 709.0)
    b = (2.0 * _H * _C**2 / wl_cm**5) / (np.exp(exp) - 1.0) / 1e8
    return np.pi * b


# ---------------------------------------------------------------------------
# Dense Teff grid
# ---------------------------------------------------------------------------

def _build_dense_teff(teff_orig, spacing):
    t_start = np.ceil(teff_orig[0] / spacing) * spacing
    uniform  = np.arange(t_start, teff_orig[-1] + spacing * 0.5, spacing)
    combined = np.unique(np.concatenate([teff_orig, uniform]))
    keep = np.ones(len(combined), dtype=bool)
    for t in teff_orig:
        dists = np.abs(combined - t)
        dists[np.argmin(dists)] = 1e9
        keep[dists < 1.0] = False
    return np.sort(combined[keep])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def densify_grid(
    src: str,
    dst: str,
    *,
    teff_spacing: float = 1000.0,
    ml_model: Optional[str] = None,
) -> None:
    """
    Parameters
    ----------
    src          : path to source flux_cube.bin
    dst          : path for output dense flux_cube.bin
    teff_spacing : target Teff spacing in K (default 1000)
    ml_model     : path to a trained SEDGenerator model directory
    """
    print(f"\n{'='*60}")
    print("SED_Tools Grid Densifier")
    print(f"{'='*60}")
    print(f"  Source  : {src}")
    print(f"  Output  : {dst}")
    print(f"  Spacing : {teff_spacing:.0f} K")
    print(f"  ML model: {ml_model or 'none (blackbody fallback)'}")

    # Read header only — no flux in RAM yet
    print("\nReading source cube header...", end=" ", flush=True)
    nt_orig, nl, nm, nw, teff_orig, logg_grid, meta_grid, wavelengths = _read_cube_header(src)
    print(f"done  ({nt_orig} × {nl} × {nm} × {nw})")
    gaps = np.diff(teff_orig)
    print(f"  Teff range : {teff_orig[0]:.0f} – {teff_orig[-1]:.0f} K")
    print(f"  Largest gap: {gaps.max():.0f} K  (mean {gaps.mean():.0f} K)")

    # Source flux as read-only memmap — pages in from disk on demand
    flux_orig = _open_flux_memmap(src, nt_orig, nl, nm, nw, mode="r")

    # Load ML model
    generator = None
    if ml_model is not None:
        from sed_tools.ml_sed_generator import SEDGenerator
        print(f"\nLoading ML model...", end=" ", flush=True)
        generator = SEDGenerator(ml_model)
        print("done")

    # Build dense Teff grid
    teff_dense = _build_dense_teff(teff_orig, teff_spacing)
    nt_new = len(teff_dense)
    print(f"\nDense grid: {nt_orig} → {nt_new} Teff points "
          f"(+{nt_new - nt_orig} synthetic)")

    # Pre-compute bb_scale one original Teff row at a time — no (nt, nw) array
    print("Pre-computing blackbody scale factors...", end=" ", flush=True)
    bb_scale = np.ones((nt_orig, nl, nm), dtype=np.float64)
    for i_t in range(nt_orig):
        bb_row = _blackbody(wavelengths, teff_orig[i_t])   # (nw,) only
        bb_bol = np.trapezoid(bb_row, wavelengths)
        if bb_bol > 0:
            for i_l in range(nl):
                for i_m in range(nm):
                    real_bol = np.trapezoid(flux_orig[i_t, i_l, i_m], wavelengths)
                    if real_bol > 0:
                        bb_scale[i_t, i_l, i_m] = real_bol / bb_bol
    print("done")

    # Initialise output cube — writes header + zero-fills on disk, returns memmap
    print(f"Initialising output cube on disk...", end=" ", flush=True)
    flux_dense = _init_output_cube(dst, teff_dense, logg_grid, meta_grid,
                                    wavelengths, nt_new)
    print("done")

    # Map original Teff indices into the dense grid and copy verbatim
    real_mask = np.zeros(nt_new, dtype=bool)
    for i_orig, t in enumerate(teff_orig):
        idx_d = int(np.argmin(np.abs(teff_dense - t)))
        if np.abs(teff_dense[idx_d] - t) < 1.0:
            flux_dense[idx_d] = flux_orig[i_orig]
            real_mask[idx_d]  = True
    flux_dense.flush()

    synthetic = np.where(~real_mask)[0]
    print(f"\nGenerating {len(synthetic)} synthetic Teff points...")
    t_start = time.time()

    for count, idx_d in enumerate(synthetic, 1):
        t_new    = teff_dense[idx_d]
        i_near   = int(np.argmin(np.abs(teff_orig - t_new)))
        in_range = teff_orig[0] <= t_new <= teff_orig[-1]

        for i_l in range(nl):
            for i_m in range(nm):
                used_ml = False

                if generator is not None and in_range:
                    try:
                        wl_ml, fl_ml = generator.generate(
                            teff=t_new,
                            logg=float(logg_grid[i_l]),
                            meta=float(meta_grid[i_m]),
                            check_bounds=False,
                        )

                        flux_dense[idx_d, i_l, i_m] = resample_to_grid(wl_ml, fl_ml, t_new, wavelengths)

                        used_ml = True
                    except Exception:
                        logger.warning("ML generation failed for Teff=%.0f logg=%.2f meta=%.2f", t_new, float(logg_grid[i_l]), float(meta_grid[i_m]), exc_info=True)

                if not used_ml:
                    bb = _blackbody(wavelengths, t_new)
                    flux_dense[idx_d, i_l, i_m] = bb * bb_scale[i_near, i_l, i_m]

        flux_dense.flush()

        elapsed = time.time() - t_start
        rate    = count / elapsed if elapsed > 0 else 1.0
        eta     = (len(synthetic) - count) / rate
        label   = "ml" if (generator is not None and in_range) else "bb"
        print(f"\r  {count}/{len(synthetic)}  T={t_new:.0f} K  "
              f"[{label}]  ETA {eta:.0f}s   ", end="", flush=True)

    print()
    del flux_dense  # close memmap
    size_mb = os.path.getsize(dst) / (1024**2)
    print(f"  Written: {nt_new} × {nl} × {nm} × {nw}  ({size_mb:.1f} MiB)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Interactive workflow
# ---------------------------------------------------------------------------

def run_interactive_workflow(base_dir: str = None):
    import glob

    if base_dir is None:
        from .models import STELLAR_DIR_DEFAULT
        base_dir = str(STELLAR_DIR_DEFAULT)

    print("\n" + "=" * 60)
    print("GRID DENSIFIER")
    print("=" * 60)

    cubes = sorted(glob.glob(os.path.join(base_dir, "*", "flux_cube.bin")))
    if not cubes:
        print(f"No flux_cube.bin files found under {base_dir}")
        return

    print("\nAvailable flux cubes:")
    for i, c in enumerate(cubes, 1):
        _, _, _, _, teff, _, _, _ = _read_cube_header(c)
        gaps = np.diff(teff)
        gap_str = f"max gap {gaps.max():.0f} K" if len(gaps) > 0 else "single point"
        print(f"  {i:2d}) {os.path.relpath(c, base_dir):<45s} "
              f"Teff {teff[0]:.0f}–{teff[-1]:.0f} K  "
              f"{gap_str}  ({len(teff)} pts)")

    sel = input("\nSelect cube: > ").strip()
    try:
        src = cubes[int(sel) - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    spacing  = float(input("Target Teff spacing in K [1000]: ").strip() or "1000")
    ml_model = input("ML model path (leave blank for blackbody only): ").strip() or None

    src_dir     = os.path.dirname(src)
    parent      = os.path.dirname(src_dir)
    default_dst = os.path.join(parent, os.path.basename(src_dir) + "_dense", "flux_cube.bin")
    dst = input(f"Output path [{default_dst}]: ").strip() or default_dst

    densify_grid(src=src, dst=dst, teff_spacing=spacing, ml_model=ml_model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Densify a flux_cube.bin Teff grid using SEDGenerator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--flux-cube",    required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--teff-spacing", type=float, default=1000.0)
    parser.add_argument("--ml-model",     default=None)
    args = parser.parse_args()

    densify_grid(
        src=args.flux_cube,
        dst=args.output,
        teff_spacing=args.teff_spacing,
        ml_model=args.ml_model,
    )


if __name__ == "__main__":
    main()
