"""Densify a flux cube's Teff grid: SEDGenerator SEDs at evenly spaced Teffs between the grid's own, so MESA's
Hermite interpolation has intervals short enough that the photometry jumps disappear. Without an ML model, and
outside the source Teff range, a blackbody (π·B_λ) scaled to the nearest real SED's bolometric flux is used.
The dense cube is written a block of wavelengths at a time (contiguous in the file), sized to half the RAM.
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from typing import Optional

import numpy as np

from .._constants import available_memory_bytes as _available_ram_bytes
from .._constants import planck_flam as _blackbody
from .build import resample_to_grid
from .fileio import flux_offset, write_header
from ..ui_utils import banner


def _read_cube_header(path):
    """Read grids and wavelengths only — does not load flux into RAM."""
    with open(path, "rb") as f:
        sizes = struct.unpack("iiii", f.read(16))
        axes = [np.frombuffer(f.read(8 * n), dtype=np.float64).copy() for n in sizes]
    return (*sizes, *axes)


# --- The dense Teff grid and its SEDs ---

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


def _blackbody_scales(flux_orig, teff_orig, wavelengths):
    """Each real node's bolometric flux over its Teff's blackbody (1 where either is <= 0), indexed [teff, logg, meta];
    computed one node at a time, so no (nt, nw) array is held."""
    nt, nl, nm = flux_orig.shape[3], flux_orig.shape[2], flux_orig.shape[1]
    bb_scale = np.ones((nt, nl, nm), dtype=np.float64)
    for i_t in range(nt):
        bb_bol = np.trapezoid(_blackbody(wavelengths, teff_orig[i_t]), wavelengths)
        if not bb_bol > 0:  # also skips NaN
            continue
        for i_l in range(nl):
            for i_m in range(nm):
                real_bol = np.trapezoid(flux_orig[:, i_m, i_l, i_t], wavelengths)
                if real_bol > 0:
                    bb_scale[i_t, i_l, i_m] = real_bol / bb_bol
    return bb_scale


def _source_rows(teff_orig, teff_dense):
    """{dense Teff index: source Teff index} for the source Teffs, which are copied verbatim."""
    rows = {}
    for i_orig, t in enumerate(teff_orig):
        idx_d = int(np.argmin(np.abs(teff_dense - t)))
        if np.abs(teff_dense[idx_d] - t) < 1.0:
            rows[idx_d] = i_orig
    return rows


def _ml_sed(generator, teff, logg, meta, wavelengths, dst):
    """SEDGenerator output at one node, resampled onto the cube's wavelengths."""
    try:
        wl_ml, fl_ml = generator.generate(teff=teff, logg=logg, meta=meta, check_bounds=False)
        return resample_to_grid(wl_ml, fl_ml, teff, wavelengths)
    except Exception as exc:
        raise RuntimeError(
            f"ML generation failed for Teff={teff:.0f} logg={logg:.2f} meta={meta:.2f}: {exc}. "
            f"{dst} is only partly written and must not be used; fix the model or re-run "
            "without ml_model to use blackbodies."
        ) from exc


def _synthetic_seds(t_new, i_near, use_ml, wl_block, logg_grid, meta_grid, bb_scale, generator, dst):
    """The SEDs at every (logg, meta) node of one synthetic Teff on wl_block, shaped (nw_block, nm, nl)."""
    if not use_ml:
        bb = _blackbody(wl_block, t_new)
        return bb[:, None, None] * bb_scale[i_near].T
    seds = np.empty((len(wl_block), len(meta_grid), len(logg_grid)))
    for i_l, logg in enumerate(logg_grid):
        for i_m, meta in enumerate(meta_grid):
            seds[:, i_m, i_l] = _ml_sed(generator, t_new, float(logg), float(meta), wl_block, dst)
    return seds


def _write_dense_flux(f, flux_orig, source_rows, teff_dense, teff_orig, logg_grid, meta_grid,
                      wavelengths, bb_scale, generator, dst):
    """Write the dense cube's flux to f a block of wavelengths at a time: source Teffs verbatim, an SED at each node
    of every other Teff (an ML node is generated once per block)."""
    nw, nm, nl, nt_orig = flux_orig.shape
    nt_new = len(teff_dense)
    synthetic = []
    for idx_d in range(nt_new):
        if idx_d not in source_rows:
            t_new = teff_dense[idx_d]
            use_ml = generator is not None and teff_orig[0] <= t_new <= teff_orig[-1]
            synthetic.append((idx_d, t_new, int(np.argmin(np.abs(teff_orig - t_new))), use_ml))

    # Wavelengths per block, so its source and dense parts fit in half the available RAM
    w_block = max(1, _available_ram_bytes() // 2 // ((nt_orig + nt_new) * nl * nm * 8))
    n_blocks = (nw + w_block - 1) // w_block
    n_ml = sum(use_ml for _, _, _, use_ml in synthetic)
    print(f"\nGenerating {len(synthetic)} synthetic Teff points ({n_ml} by ML, "
          f"{len(synthetic) - n_ml} blackbody)\n  in {n_blocks} wavelength block(s)...")
    t_start = time.time()
    for count, w0 in enumerate(range(0, nw, w_block), 1):
        w1 = min(w0 + w_block, nw)
        block = np.empty((w1 - w0, nm, nl, nt_new), dtype=np.float64)
        for idx_d, i_orig in source_rows.items():
            block[..., idx_d] = flux_orig[w0:w1, :, :, i_orig]
        for idx_d, t_new, i_near, use_ml in synthetic:
            block[..., idx_d] = _synthetic_seds(t_new, i_near, use_ml, wavelengths[w0:w1],
                                                logg_grid, meta_grid, bb_scale, generator, dst)
        block.tofile(f)

        elapsed = time.time() - t_start
        eta = (n_blocks - count) * elapsed / count
        print(f"\r  block {count}/{n_blocks}  ETA {eta:.0f}s   ", end="", flush=True)
    print()


# --- densify_grid, the interactive flow and the command line ---

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

    Raises RuntimeError if the ML model fails at any node; dst is then incomplete.
    """
    print(f"\n{'='*60}")
    print("SED_Tools Grid Densifier")
    print(f"{'='*60}")
    print(f"  Source  : {src}")
    print(f"  Output  : {dst}")
    print(f"  Spacing : {teff_spacing:.0f} K")
    print(f"  ML model: {ml_model or 'none (blackbody fallback)'}")

    print("\nReading source cube header...", end=" ", flush=True)
    nt_orig, nl, nm, nw, teff_orig, logg_grid, meta_grid, wavelengths = _read_cube_header(src)
    print(f"done  ({nt_orig} × {nl} × {nm} × {nw})")
    gaps = np.diff(teff_orig)
    print(f"  Teff range : {teff_orig[0]:.0f} – {teff_orig[-1]:.0f} K")
    print(f"  Largest gap: {gaps.max():.0f} K  (mean {gaps.mean():.0f} K)")

    # Read-only, in the file's order (nw, nm, nl, nt)
    flux_orig = np.memmap(src, dtype=np.float64, mode="r", offset=flux_offset(nt_orig, nl, nm, nw),
                          shape=(nw, nm, nl, nt_orig))

    generator = None
    if ml_model is not None:
        from sed_tools.ml.generator import SEDGenerator
        print("\nLoading ML model...", end=" ", flush=True)
        generator = SEDGenerator(ml_model)
        print("done")

    teff_dense = _build_dense_teff(teff_orig, teff_spacing)
    nt_new = len(teff_dense)
    print(f"\nDense grid: {nt_orig} → {nt_new} Teff points "
          f"(+{nt_new - nt_orig} synthetic)")

    print("Pre-computing blackbody scale factors...", end=" ", flush=True)
    bb_scale = _blackbody_scales(flux_orig, teff_orig, wavelengths)
    print("done")

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "wb") as f:
        write_header(f, teff_dense, logg_grid, meta_grid, wavelengths)
        _write_dense_flux(f, flux_orig, _source_rows(teff_orig, teff_dense), teff_dense, teff_orig,
                          logg_grid, meta_grid, wavelengths, bb_scale, generator, dst)

    size_mb = os.path.getsize(dst) / (1024**2)
    print(f"  Written: {nt_new} × {nl} × {nm} × {nw}  ({size_mb:.1f} MiB)")
    print(f"{'='*60}\n")


def run_interactive_workflow(base_dir: str = None) -> Optional[int]:
    """Choose a cube, spacing, ML model and output, then densify; 1 when a model is given without torch."""
    import glob

    from ..ui_utils import ask_number as _ask_number, refused_without

    if base_dir is None:
        from .model import STELLAR_DIR_DEFAULT
        base_dir = str(STELLAR_DIR_DEFAULT)

    banner("GRID DENSIFIER")

    cubes = sorted(glob.glob(os.path.join(base_dir, "*", "flux_cube.bin")))
    if not cubes:
        print(f"No flux_cube.bin files found under {base_dir}")
        return

    print("\nAvailable flux cubes:")
    for i, c in enumerate(cubes, 1):
        _, _, _, _, teff, _, _, _ = _read_cube_header(c)
        gaps = np.diff(teff)
        gap_str = f"max gap {gaps.max():.0f} K" if len(gaps) > 0 else "single point"
        # the cube's numbers on their own line, so a long grid name never pushes them off the terminal
        print(f"  {i:2d}) {os.path.relpath(c, base_dir)}")
        print(f"      Teff {teff[0]:.0f}–{teff[-1]:.0f} K  {gap_str}  ({len(teff)} pts)")

    sel = input("\nSelect cube: > ").strip()
    if not sel.isdecimal() or not 1 <= int(sel) <= len(cubes):
        print("Invalid selection.")
        return
    src = cubes[int(sel) - 1]

    spacing  = _ask_number("Target Teff spacing in K [1000]: ", 1000.0, positive=True)
    ml_model = input("ML model path (leave blank for blackbody only): ").strip() or None
    if ml_model is not None and refused_without("torch"):
        return 1

    src_dir     = os.path.dirname(src)
    default_dst = os.path.join(os.path.dirname(src_dir), os.path.basename(src_dir) + "_dense", "flux_cube.bin")
    dst = input(f"Output path [{default_dst}]: ").strip() or default_dst

    densify_grid(src=src, dst=dst, teff_spacing=spacing, ml_model=ml_model)


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
    densify_grid(src=args.flux_cube, dst=args.output, teff_spacing=args.teff_spacing, ml_model=args.ml_model)


if __name__ == "__main__":
    main()
