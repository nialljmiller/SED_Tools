"""
sed_tools/mesa_prepare.py

Interactive tool for selecting a specific sub-variant of a stellar atmosphere
model grid (e.g. a particular alpha-enhancement or microturbulence value) and
exporting a clean, MESA-ready folder containing only that variant's flux cube
and lookup table.

Usage (CLI):
    sed-tools mesa_prepare
    sed-tools mesa_prepare --model Kurucz2003all
    sed-tools mesa_prepare --model Kurucz2003all --output /path/to/output

Usage (API):
    from sed_tools.mesa_prepare import list_variants, export_variant

    variants = list_variants("data/stellar_models/Kurucz2003all")
    export_variant(variants[0], output_dir="data/stellar_models/Kurucz2003all_alpha0")
"""

from __future__ import annotations

import os
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VariantInfo:
    """Metadata for a single extra-axis variant of a model grid."""

    label: str                          # e.g. "alpha_0.4__lh_1.25__vtur_2"
    extra_axes: Dict[str, str]          # e.g. {"alpha": "0.4", "lh": "1.25", "vtur": "2"}
    n_spectra: int
    teff_min: float
    teff_max: float
    teff_n: int
    logg_min: float
    logg_max: float
    logg_n: int
    meta_min: float
    meta_max: float
    meta_n: int
    wavelength_min: float
    wavelength_max: float
    wavelength_n: int
    flux_cube_path: Path                # path to the variant's .bin file
    source_model_dir: Path              # original model directory (where .txt files live)
    lookup_rows: pd.DataFrame = field(repr=False)  # rows from the master lookup_table


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_variants(model_dir: str | os.PathLike) -> List[VariantInfo]:
    """
    Discover all extra-axis variants inside *model_dir*.

    Reads the master ``lookup_table.csv`` to identify extra axis columns,
    groups rows by their combination of extra-axis values, then matches each
    group to the corresponding pre-built flux cube in ``fluxcube_library/``.

    Parameters
    ----------
    model_dir:
        Path to the model directory (must contain ``lookup_table.csv`` and
        ``fluxcube_library/``).

    Returns
    -------
    List of :class:`VariantInfo`, one per unique extra-axis combination found.
    Raises ``FileNotFoundError`` if required files are missing.
    """
    model_dir = Path(model_dir)
    lookup_path = model_dir / "lookup_table.csv"
    library_dir = model_dir / "fluxcube_library"

    if not lookup_path.exists():
        raise FileNotFoundError(f"lookup_table.csv not found in {model_dir}")

    df = pd.read_csv(lookup_path, dtype=str)
    # Strip leading '#' from column names (common CSV comment convention)
    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]

    # Identify the core parameter columns and the extra-axis columns
    core_cols = {"file_name", "teff", "logg", "metallicity", "flux_unit",
                 "wavelength_unit", "units_standardized", "source",
                 "original_flux_unit", "original_wavelength_unit",
                 "conversion_confidence"}
    extra_cols = [c for c in df.columns if c not in core_cols]

    # No extra axes — single uniform grid
    if not extra_cols:
        return [_variant_from_rows(df, {}, model_dir / "flux_cube.bin", df, model_dir)]

    # No fluxcube_library — model was built without variant cubes; fall back to master cube
    if not library_dir.exists():
        master_cube = model_dir / "flux_cube.bin"
        if not master_cube.exists():
            raise FileNotFoundError(
                f"No flux_cube.bin found in {model_dir}. Run 'sed-tools rebuild' first."
            )
        axes_list = ", ".join(extra_cols)
        print(
            f"  Note: fluxcube_library/ not found. The master flux_cube.bin averages "
            f"all extra-axis variants ({axes_list})."
        )
        print("  To get per-variant cubes, run 'sed-tools rebuild' on this model.")
        return [_variant_from_rows(df, {}, master_cube, df, model_dir)]

    # Group by unique combination of extra-axis values
    variants: List[VariantInfo] = []
    grouped = df.groupby(extra_cols, sort=True)

    for key, rows in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        axes = dict(zip(extra_cols, key))
        label = "__".join(f"{k}_{v}" for k, v in axes.items())

        cube_path = _find_variant_cube(library_dir, label)
        variants.append(_variant_from_rows(rows.reset_index(drop=True), axes, cube_path, df, model_dir))

    return variants


def _find_variant_cube(library_dir: Path, label: str) -> Path:
    """Return the flux cube path for *label*, or raise if not found."""
    # Exact match
    candidate = library_dir / f"flux_cube__{label}.bin"
    if candidate.exists():
        return candidate

    # Fuzzy: stem comparison
    for path in library_dir.glob("flux_cube__*.bin"):
        if path.stem.replace("flux_cube__", "") == label:
            return path

    raise FileNotFoundError(
        f"No flux cube found in {library_dir} for variant '{label}'.\n"
        f"Expected: flux_cube__{label}.bin"
    )


def _read_cube_header(path: Path) -> dict:
    """Read grid axes from a binary flux cube header."""
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as fh:
            header = fh.read(16)
            nt, nl, nm, nw = struct.unpack("4i", header)
            teff = np.fromfile(fh, dtype=np.float64, count=nt)
            logg = np.fromfile(fh, dtype=np.float64, count=nl)
            meta = np.fromfile(fh, dtype=np.float64, count=nm)
            wave = np.fromfile(fh, dtype=np.float64, count=nw)
        return {"teff": teff, "logg": logg, "meta": meta, "wave": wave}
    except Exception:
        return {}


def _variant_from_rows(
    rows: pd.DataFrame,
    axes: Dict[str, str],
    cube_path: Path,
    _full_df: pd.DataFrame,
    model_dir: Path,
) -> VariantInfo:
    """Build a :class:`VariantInfo` from a subset of lookup-table rows."""
    label = "__".join(f"{k}_{v}" for k, v in axes.items()) if axes else "full_grid"

    # Numeric parameter ranges come from the lookup table rows
    def _range(col: str):
        if col not in rows.columns:
            return float("nan"), float("nan"), 0
        vals = pd.to_numeric(rows[col], errors="coerce").dropna()
        uniq = np.unique(vals)
        return (float(uniq[0]), float(uniq[-1]), len(uniq)) if len(uniq) else (float("nan"), float("nan"), 0)

    teff_min, teff_max, teff_n   = _range("teff")
    logg_min, logg_max, logg_n   = _range("logg")
    meta_min, meta_max, meta_n   = _range("metallicity")

    # Wavelength range and point count come from the flux cube header
    hdr = _read_cube_header(cube_path)
    if hdr:
        wave_min = float(hdr["wave"][0])
        wave_max = float(hdr["wave"][-1])
        wave_n   = len(hdr["wave"])
    else:
        wave_min = wave_max = float("nan")
        wave_n = 0

    return VariantInfo(
        label=label,
        extra_axes=axes,
        n_spectra=len(rows),
        teff_min=teff_min, teff_max=teff_max, teff_n=teff_n,
        logg_min=logg_min, logg_max=logg_max, logg_n=logg_n,
        meta_min=meta_min, meta_max=meta_max, meta_n=meta_n,
        wavelength_min=wave_min, wavelength_max=wave_max, wavelength_n=wave_n,
        flux_cube_path=cube_path,
        source_model_dir=model_dir,
        lookup_rows=rows,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_variant_table(variants: List[VariantInfo], model_name: str = "") -> None:
    """Print a formatted table of all variants to stdout."""
    title = f"Sub-variants of {model_name}" if model_name else "Available sub-variants"
    print(f"\n{title}")
    print("=" * 90)

    # Column widths
    col_idx   = 4
    col_label = max(30, max(len(v.label) for v in variants) + 2)
    col_n     = 8
    col_param = 22

    header = (
        f"{'#':>{col_idx}}  "
        f"{'Variant':<{col_label}}  "
        f"{'Spectra':>{col_n}}  "
        f"{'Teff (K)':<{col_param}}"
        f"{'logg':<{col_param}}"
        f"{'[M/H]':<{col_param}}"
        f"{'Wavelength (Å)'}"
    )
    print(header)
    print("-" * 90)

    for i, v in enumerate(variants, start=1):
        def fmt_range(lo, hi, n):
            if np.isnan(lo):
                return "n/a"
            return f"{lo:.0f} – {hi:.0f}  ({n}pt)"

        def fmt_range_f(lo, hi, n):
            if np.isnan(lo):
                return "n/a"
            return f"{lo:.2f} – {hi:.2f}  ({n}pt)"

        teff_s = fmt_range(v.teff_min, v.teff_max, v.teff_n)
        logg_s = fmt_range_f(v.logg_min, v.logg_max, v.logg_n)
        meta_s = fmt_range_f(v.meta_min, v.meta_max, v.meta_n)
        wave_s = (
            f"{v.wavelength_min:.0f} – {v.wavelength_max:.0f}  ({v.wavelength_n}pt)"
            if not np.isnan(v.wavelength_min) else "n/a"
        )

        # Extra axes inline annotation
        axes_str = "  ".join(f"{k}={val}" for k, val in v.extra_axes.items())

        print(
            f"[{i:>{col_idx - 2}}]  "
            f"{axes_str if axes_str else '(no extra axes)':<{col_label}}  "
            f"{v.n_spectra:>{col_n}}  "
            f"{teff_s:<{col_param}}"
            f"{logg_s:<{col_param}}"
            f"{meta_s:<{col_param}}"
            f"{wave_s}"
        )

    print("=" * 90)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_variant(
    variant: VariantInfo,
    output_dir: str | os.PathLike,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write a clean MESA-ready folder for *variant*.

    Produces the same file structure as a normally downloaded/rebuilt model:

    * One ``.txt`` spectrum file per SED (copied from the source model dir)
    * ``lookup_table.csv`` — filtered to only this variant's spectra
    * ``flux_cube.bin``    — copied from the variant's pre-built cube

    Parameters
    ----------
    variant:
        A :class:`VariantInfo` returned by :func:`list_variants`.
    output_dir:
        Destination directory.  Created if it does not exist.
    overwrite:
        If ``False`` (default) and *output_dir* already exists, raise
        ``FileExistsError``.

    Returns
    -------
    Path to the created output directory.
    """
    out = Path(output_dir)
    src = variant.source_model_dir

    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {out}\n"
            f"Pass overwrite=True or choose a different name."
        )

    out.mkdir(parents=True, exist_ok=True)

    # 1. Copy .txt spectrum files for this variant
    file_col = next(
        (c for c in variant.lookup_rows.columns if c.lower() == "file_name"), None
    )
    if file_col is None:
        raise KeyError("lookup_rows has no 'file_name' column — cannot locate spectra.")

    filenames = variant.lookup_rows[file_col].dropna().tolist()
    print(f"  Copying {len(filenames)} spectrum files...")
    n_copied = 0
    n_missing = 0
    for fname in filenames:
        src_file = src / fname
        if not src_file.exists():
            n_missing += 1
            continue
        shutil.copy2(src_file, out / fname)
        n_copied += 1

    if n_missing:
        print(f"  Warning: {n_missing} spectrum file(s) not found in {src} — skipped.")
    print(f"  Copied {n_copied}/{len(filenames)} spectrum files.")

    # 2. Filtered lookup table
    dest_lookup = out / "lookup_table.csv"
    variant.lookup_rows.to_csv(dest_lookup, index=False)
    print(f"  Written lookup_table.csv ({len(variant.lookup_rows)} rows)")

    # 3. Flux cube
    dest_cube = out / "flux_cube.bin"
    shutil.copy2(variant.flux_cube_path, dest_cube)
    print(f"  Copied flux_cube.bin ({dest_cube.stat().st_size / 1024**2:.1f} MiB)")

    # 4. Summary
    print(f"\n  Done. MESA-ready folder: {out}")
    print(f"    {n_copied} spectra  |  lookup_table.csv  |  flux_cube.bin")
    print(f"\n  To use in MESA:")
    print(f"    stellar_atm = '{out}/'")

    return out


# ---------------------------------------------------------------------------
# Interactive CLI flow
# ---------------------------------------------------------------------------

def run_interactive(
    base_dir: str = "data/stellar_models",
    model: Optional[str] = None,
    output: Optional[str] = None,
) -> None:
    """
    Interactive MESA-prepare workflow.

    Prompts the user to pick a model (if not supplied), lists its variants,
    lets the user select one, and exports it to a clean folder.
    """
    base_path = Path(base_dir)

    # --- Step 1: pick model ---
    if model is None:
        available = sorted(
            d.name for d in base_path.iterdir()
            if d.is_dir() and (d / "lookup_table.csv").exists()
        )
        if not available:
            print(f"No models found in {base_path}. Run 'sed-tools spectra' first.")
            return

        print("\nAvailable models:")
        for i, name in enumerate(available, 1):
            print(f"  [{i}] {name}")
        raw = input("\nSelect model number (or type name): ").strip()

        if raw.isdigit() and 1 <= int(raw) <= len(available):
            model = available[int(raw) - 1]
        elif raw in available:
            model = raw
        else:
            print(f"Invalid selection: {raw}")
            return

    model_dir = base_path / model

    # --- Step 2: discover variants ---
    print(f"\nScanning {model_dir} for sub-variants...")
    try:
        variants = list_variants(model_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    if len(variants) == 1 and not variants[0].extra_axes:
        print("This model has no extra-axis variants — it is a single uniform grid.")
        print("Nothing to select; the existing flux_cube.bin already covers the full grid.")
        return

    print_variant_table(variants, model_name=model)

    # --- Step 3: user selects a variant ---
    raw = input("\nSelect variant number: ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(variants)):
        print(f"Invalid selection: {raw}")
        return
    chosen = variants[int(raw) - 1]

    print(f"\nSelected: {chosen.label}")
    for k, v in chosen.extra_axes.items():
        print(f"  {k} = {v}")

    # --- Step 4: determine output directory ---
    if output is None:
        default_name = f"{model}__{chosen.label}"
        raw_out = input(
            f"\nOutput directory [{base_path / default_name}]: "
        ).strip()
        output = raw_out if raw_out else str(base_path / default_name)

    # --- Step 5: export ---
    print(f"\nExporting variant to {output}...")
    try:
        export_variant(chosen, output_dir=output)
    except FileExistsError as exc:
        print(f"Error: {exc}")
        raw_ow = input("Overwrite? [y/N]: ").strip().lower()
        if raw_ow == "y":
            export_variant(chosen, output_dir=output, overwrite=True)
        else:
            print("Aborted.")