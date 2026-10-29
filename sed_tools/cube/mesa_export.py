"""Export one variant of a spectral grid (e.g. one alpha-enhancement or microturbulence value) as a clean
MESA-ready folder: list_variants, print_variant_table, export_variant, and the interactive flow behind
``sed-tools mesa_prepare [--grid NAME] [--output DIR]``. list_variants reads the subgrids cube/variants.py
writes, and the layout of earlier versions.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .collision import discover_extra_axes
from .model import STELLAR_DIR_DEFAULT, _read_flux_cube_header

# Lookup columns that are not variant axes in grids split into fluxcube_library/
# by earlier SED_Tools versions; the cube file names there use every other column.
_LEGACY_CORE_COLUMNS = {"file_name", "teff", "logg", "metallicity", "flux_unit",
                        "wavelength_unit", "units_standardized", "source",
                        "original_flux_unit", "original_wavelength_unit",
                        "conversion_confidence"}


@dataclass
class VariantInfo:
    """Metadata for a single extra-axis variant of a spectral grid."""

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


def list_variants(model_dir: str | os.PathLike) -> List[VariantInfo]:
    """
    Discover all extra-axis variants inside *model_dir*.

    Where the flux cubes come from, in order:

    1. ``variants_index.csv``: ``sed-tools rebuild`` split a grid whose
       extra axes (alpha, microturbulence, ...) vary into one subgrid per
       variant, ``model_dir/<path>/`` with its own ``flux_cube.bin`` and
       ``lookup_table.csv``. Variants are named by the axes that vary.
    2. ``fluxcube_library/flux_cube__<label>.bin``, as earlier SED_Tools
       versions wrote it.
    3. ``model_dir/flux_cube.bin`` as one variant (``full_grid``): a uniform
       grid, whose extra axes, if any, have a single value; or a grid built
       with the 'mean' or 'filter' collision strategy, which is noted.

    Parameters
    ----------
    model_dir:
        Path to the spectral grid directory (must contain ``lookup_table.csv``).

    Returns
    -------
    List of :class:`VariantInfo`, one per variant.
    Raises ``FileNotFoundError`` if required files are missing.
    """
    model_dir = Path(model_dir)
    lookup_path = model_dir / "lookup_table.csv"
    library_dir = model_dir / "fluxcube_library"

    if not lookup_path.exists():
        raise FileNotFoundError(f"lookup_table.csv not found in {model_dir}")
    df = _read_csv_as_text(lookup_path)

    if (model_dir / "variants_index.csv").exists():
        return _split_variants(model_dir)

    legacy_cols = [c for c in df.columns if c not in _LEGACY_CORE_COLUMNS]
    if library_dir.exists() and legacy_cols:
        return _library_variants(df, legacy_cols, library_dir, model_dir)

    master_cube = model_dir / "flux_cube.bin"
    axes = discover_extra_axes(list(df.columns))
    varying = _varying_columns(df, axes)
    if varying:
        if not master_cube.exists():
            raise FileNotFoundError(
                f"No flux_cube.bin or variants_index.csv found in {model_dir}. Run 'sed-tools rebuild' first."
            )
        n_variants = len(df[varying].drop_duplicates())
        print(
            f"  Note: {model_dir.name} has {n_variants} variants over {', '.join(varying)}, but no "
            "per-variant cubes (no variants_index.csv). Its flux_cube.bin was built with the 'mean' "
            "or 'filter' collision strategy, so it averages the variants or holds only one of them."
        )
        print("  Rebuild with the default 'split' strategy to get one MESA-ready subgrid per variant.")
    return [_variant_from_rows(df, {}, master_cube, model_dir)]


def _read_csv_as_text(path: Path) -> pd.DataFrame:
    """A lookup or index table as strings, with '#' and spaces stripped from the column names."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
    return df


def _varying_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """The *columns* that hold more than one value."""
    return [c for c in columns if df[c].nunique(dropna=False) > 1]


def _split_variants(model_dir: Path) -> List[VariantInfo]:
    """One variant per subgrid listed in ``variants_index.csv`` (see :func:`list_variants`)."""
    index_path = model_dir / "variants_index.csv"
    index = _read_csv_as_text(index_path)
    axes = [c for c in index.columns if c not in ("variant_name", "n_spectra", "path")]
    varying = _varying_columns(index, axes)

    variants: List[VariantInfo] = []
    for _, entry in index.iterrows():
        subgrid = model_dir / entry["path"]
        cube_path = subgrid / "flux_cube.bin"
        if not cube_path.is_file():
            raise FileNotFoundError(
                f"{index_path} lists the subgrid {entry['path']}, but {cube_path} does not exist. "
                f"Run 'sed-tools rebuild' on {model_dir} again."
            )
        rows = _read_csv_as_text(subgrid / "lookup_table.csv")
        # The spectra are the parent's; the subgrid holds links to them.
        variants.append(_variant_from_rows(rows, {a: entry[a] for a in varying}, cube_path, model_dir))
    return variants


def _library_variants(df: pd.DataFrame, extra_cols: List[str], library_dir: Path,
                      model_dir: Path) -> List[VariantInfo]:
    """One variant per cube in ``fluxcube_library/`` (see :func:`list_variants`)."""
    variants: List[VariantInfo] = []
    grouped = df.groupby(extra_cols, sort=True)

    for key, rows in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        axes = dict(zip(extra_cols, key))
        label = "__".join(f"{k}_{v}" for k, v in axes.items())

        cube_path = _find_variant_cube(library_dir, label)
        variants.append(_variant_from_rows(rows.reset_index(drop=True), axes, cube_path, model_dir))

    return variants


def _find_variant_cube(library_dir: Path, label: str) -> Path:
    """Return the flux cube path for *label*, or raise if not found."""
    candidate = library_dir / f"flux_cube__{label}.bin"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"No flux cube found in {library_dir} for variant '{label}'.\n"
        f"Expected: flux_cube__{label}.bin"
    )


def _variant_from_rows(
    rows: pd.DataFrame,
    axes: Dict[str, str],
    cube_path: Path,
    model_dir: Path,
) -> VariantInfo:
    """Build a :class:`VariantInfo` from a subset of lookup-table rows."""
    label = "__".join(f"{k}_{v}" for k, v in axes.items()) if axes else "full_grid"

    # Parameter ranges come from the lookup rows, the wavelengths from the cube's header.
    def _range(col: str):
        if col not in rows.columns:
            return float("nan"), float("nan"), 0
        vals = pd.to_numeric(rows[col], errors="coerce").dropna()
        uniq = np.unique(vals)
        return (float(uniq[0]), float(uniq[-1]), len(uniq)) if len(uniq) else (float("nan"), float("nan"), 0)

    teff_min, teff_max, teff_n   = _range("teff")
    logg_min, logg_max, logg_n   = _range("logg")
    meta_min, meta_max, meta_n   = _range("metallicity")

    if cube_path.exists():
        wavelengths = _read_flux_cube_header(cube_path)["wavelengths"]
        wave_min = float(wavelengths[0])
        wave_max = float(wavelengths[-1])
        wave_n   = len(wavelengths)
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


def _fmt_range(lo: float, hi: float, n: int, decimals: int) -> str:
    if np.isnan(lo):
        return "n/a"
    return f"{lo:.{decimals}f} – {hi:.{decimals}f}  ({n}pt)"


def _varying_axes(variants: List[VariantInfo]) -> List[str]:
    """The extra axes whose value differs between *variants*, in their first-seen order."""
    names = list(dict.fromkeys(k for v in variants for k in v.extra_axes))
    return [k for k in names if len({v.extra_axes.get(k) for v in variants}) > 1]


def print_variant_table(variants: List[VariantInfo], model_name: str = "") -> None:
    """Print a formatted table of all variants to stdout, naming the axes that vary."""
    title = f"Sub-variants of {model_name}" if model_name else "Available sub-variants"
    shown = _varying_axes(variants)
    if shown:
        title += f": {len(variants)} variants over {', '.join(shown)}"
    print(f"\n{title}")
    print("=" * 90)

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
        teff_s = _fmt_range(v.teff_min, v.teff_max, v.teff_n, 0)
        logg_s = _fmt_range(v.logg_min, v.logg_max, v.logg_n, 2)
        meta_s = _fmt_range(v.meta_min, v.meta_max, v.meta_n, 2)
        wave_s = _fmt_range(v.wavelength_min, v.wavelength_max, v.wavelength_n, 0)

        # Extra axes inline annotation; axes with one value throughout say nothing
        axes_str = "  ".join(f"{k}={val}" for k, val in v.extra_axes.items() if k in shown)

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


def export_variant(
    variant: VariantInfo,
    output_dir: str | os.PathLike,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write a clean MESA-ready folder for *variant*.

    Produces the same file structure as a normally downloaded/rebuilt spectral grid:

    * One ``.txt`` spectrum file per SED (copied from the source spectral grid directory)
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

    Raises ``FileNotFoundError``, before writing anything, if any of the
    variant's spectra is missing from the source directory, or its flux cube
    has not been built.
    """
    out = Path(output_dir)
    src = variant.source_model_dir

    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {out}\n"
            f"Pass overwrite=True or choose a different name."
        )

    file_col = next(
        (c for c in variant.lookup_rows.columns if c.lower() == "file_name"), None
    )
    if file_col is None:
        raise KeyError("lookup_rows has no 'file_name' column — cannot locate spectra.")

    filenames = variant.lookup_rows[file_col].dropna().tolist()
    missing = [fname for fname in filenames if not (src / fname).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} spectra of variant '{variant.label}' are missing from {src}; "
            "nothing was exported. Restore them or rebuild the grid:\n  "
            + "\n  ".join(missing)
        )
    if not variant.flux_cube_path.is_file():
        raise FileNotFoundError(
            f"The flux cube of variant '{variant.label}', {variant.flux_cube_path}, does not exist; "
            "nothing was exported. Build it with 'sed-tools rebuild'."
        )

    out.mkdir(parents=True, exist_ok=True)

    print(f"  Copying {len(filenames)} spectrum files...")
    for fname in filenames:
        shutil.copy2(src / fname, out / fname)
    print(f"  Copied {len(filenames)} spectrum files.")

    dest_lookup = out / "lookup_table.csv"
    variant.lookup_rows.to_csv(dest_lookup, index=False)
    print(f"  Written lookup_table.csv ({len(variant.lookup_rows)} rows)")

    dest_cube = out / "flux_cube.bin"
    shutil.copy2(variant.flux_cube_path, dest_cube)
    print(f"  Copied flux_cube.bin ({dest_cube.stat().st_size / 1024**2:.1f} MiB)")

    print(f"\n  Done. MESA-ready folder: {out}")
    print(f"    {len(filenames)} spectra  |  lookup_table.csv  |  flux_cube.bin")
    print("\n  To use in MESA:")
    print(f"    stellar_atm = '{out}/'")

    return out


def _choose_grid(base_path: Path) -> str:
    """Prompt for one of the spectral grids in *base_path*; exits with status 1 if none can be chosen."""
    available = sorted(
        d.name for d in base_path.iterdir()
        if d.is_dir() and (d / "lookup_table.csv").exists()
    )
    if not available:
        raise SystemExit(f"Error: no spectral grids found in {base_path}. Run 'sed-tools grids' first.")

    print("\nAvailable spectral grids:")
    for i, name in enumerate(available, 1):
        print(f"  [{i}] {name}")
    raw = input("\nSelect spectral grid number (or type name): ").strip()

    if raw.isdigit() and 1 <= int(raw) <= len(available):
        return available[int(raw) - 1]
    if raw in available:
        return raw
    raise SystemExit(f"Error: invalid spectral grid selection {raw!r}.\n"
                     f"Enter a number from 1 to {len(available)} or a grid name. Nothing was exported.")


def _choose_variant(variants: List[VariantInfo]) -> VariantInfo:
    """Prompt for one of *variants*; exits with status 1 on an invalid choice."""
    raw = input("\nSelect variant number: ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(variants)):
        raise SystemExit(f"Error: invalid variant selection {raw!r}; "
                         f"enter a number from 1 to {len(variants)}. Nothing was exported.")
    return variants[int(raw) - 1]


def run_interactive(base_dir: str = None, model=None, output=None):
    """
    Interactive MESA-prepare workflow.

    Prompts the user to pick a spectral grid (if not supplied), lists its variants,
    lets the user select one, and exports it to a clean folder. Exits with status 1
    and an error message if nothing was exported: no grid to pick, an invalid
    choice, a missing lookup table or flux cube, a grid with no variants to
    choose from, or an existing output directory the user chose not to overwrite.
    """
    base_path = Path(base_dir if base_dir is not None else STELLAR_DIR_DEFAULT)

    if model is None:
        model = _choose_grid(base_path)

    model_dir = base_path / model
    print(f"\nScanning {model_dir} for sub-variants...")
    try:
        variants = list_variants(model_dir)
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if len(variants) == 1 and not variants[0].extra_axes:
        # One cube for the whole folder needs no export: the folder is what MESA reads.
        if not variants[0].flux_cube_path.is_file():
            raise SystemExit(f"Error: {model_dir} is a single uniform grid and has no flux_cube.bin yet. "
                             "Build it with 'sed-tools rebuild'; the grid folder is then MESA-ready. "
                             "Nothing was exported.")
        print(f"{model_dir} has one flux_cube.bin for all its spectra and no variants to choose "
              "between, so it is already MESA-ready. Nothing to export.")
        return

    print_variant_table(variants, model_name=model)
    chosen = _choose_variant(variants)

    print(f"\nSelected: {chosen.label}")
    for k, v in chosen.extra_axes.items():
        print(f"  {k} = {v}")

    if output is None:
        # Axis values may hold spaces or '/' (e.g. "2 km/s"); keep the default one folder name.
        default_name = f"{model}__{re.sub(r'[^A-Za-z0-9._+-]+', '_', chosen.label)}"
        raw_out = input(
            f"\nOutput directory [{base_path / default_name}]: "
        ).strip()
        output = raw_out if raw_out else str(base_path / default_name)

    print(f"\nExporting variant to {output}...")
    try:
        export_variant(chosen, output_dir=output)
    except FileExistsError as exc:
        print(f"Error: {exc}")
        raw_ow = input("Overwrite? [y/N]: ").strip().lower()
        if raw_ow != "y":
            raise SystemExit(f"Error: {output} already exists and was not overwritten; nothing was exported.")
        export_variant(chosen, output_dir=output, overwrite=True)
