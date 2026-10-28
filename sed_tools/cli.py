#!/usr/bin/env python3
"""
SED_tools.py — one-stop launcher for spectra & filters

Pipeline per selected spectral grid:
  - SVO or MSG (Rich Townsend) spectral grid discovery
  - Download (SVO) or extract (MSG) spectra -> *.txt collection
  - Ensure lookup_table.csv
  - Build/ensure HDF5 bundle of spectra
  - Precompute flux cube file

Filters:
  - Shared NJM/original-source/SVO acquisition and component composition
  - Saves to data/filters/<Facility>/<Instrument>/<Band>.dat
"""

import argparse
import csv
import glob
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from requests import RequestException

from .config import set_data_dir, show_config
from .filter_catalogue import CatalogueError
from .filter_components import ComponentError
from .mast_spectra_grabber import MASTSpectraGrabber
from .models import FILTER_DIR_DEFAULT, STELLAR_DIR_DEFAULT
from .msg_spectra_grabber import MSGSpectraGrabber
from .njm_spectra_grabber import NJMSpectraGrabber
from .spectra_cleaner import FLUX_GUESS_WHY, FLUX_UNITS, clean_model_dir, undeclared_flux_unit
from .svo_regen_spectra_lookup import regenerate_lookup_table
from .svo_spectra_grabber import SVOSpectraGrabber
from .ui_utils import NO_DEFAULT as _NO_DEFAULT, OPTIONAL_EXTRAS, ask_number as _ask_number, missing_packages
from .ui_utils import prompt_choice
from .parsing import parse_multi_selection, parse_numeric_range
from .spectrum_io import build_h5_bundle, read_text_spectrum
from .terminal_plots import terminal_color_enabled, terminal_width

logger = logging.getLogger(__name__)

#: The errors a workflow raises for a request it cannot meet. main() and the
#: menu print their message; anything else is a bug and keeps its traceback.
EXPECTED_ERRORS = (RuntimeError, ValueError, OSError, ComponentError, CatalogueError,
                   RequestException, EOFError)

# ------------ Small Utils ------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _missing_package_error(exc: ImportError) -> Optional[str]:
    """The missing_packages line for an optional package that failed to import, else None (a bug)."""
    root = (exc.name or "").split(".")[0]
    return missing_packages([root]) if root in OPTIONAL_EXTRAS else None


def _ask_option(prompt: str, options: Sequence[str], default: Optional[str] = None) -> str:
    """Ask until the answer is one of ``options`` (any case); a blank answer gives ``default`` if there is one."""
    lowered = {o.lower(): o for o in options}
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        if raw.lower() in lowered:
            return lowered[raw.lower()]
        print(f"  Please enter one of: {', '.join(options)}.")


def _yes(prompt: str) -> bool:
    """A [y/N] question: only an answer starting with 'y' is yes."""
    return input(prompt).strip().lower().startswith("y")


def _wrap(text: str, width: Optional[int] = None, indent: str = "", hanging: str = "") -> str:
    """``text`` filled to the terminal width; URLs and hyphenated words are never split."""
    return textwrap.fill(" ".join(str(text).split()), width or terminal_width(),
                         initial_indent=indent, subsequent_indent=hanging,
                         break_on_hyphens=False, break_long_words=False)


def load_txt_spectrum(txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return read_text_spectrum(txt_path)


# ------------ HDF5 Bundling ------------

def build_h5_bundle_from_txt(model_dir: str, out_h5: str) -> None:
    """Create an HDF5 file bundling all .txt spectra."""
    if not build_h5_bundle(model_dir, out_h5):
        print(f"[H5 bundle] No .txt spectra found in {model_dir}; skipping.")
        return
    print(f"[H5 bundle] Wrote {out_h5}")


def _ask_range(prompt: str) -> Optional[Tuple[float, float]]:
    """Ask for a range like '3500,8000', '3500 8000' or '3500-8000' until one parses.

    Returns (min, max), or None for a blank answer (no cut).
    """
    while True:
        raw = input(prompt).strip()
        parsed = parse_numeric_range(raw)
        if parsed is not None or not raw:
            return parsed
        print(f"  Could not parse range from '{raw}' — need two values (e.g. '3500,8000')")


def prompt_njm_axis_cuts(
    name: str,
    grabber,  # NJMSpectraGrabber instance
    model_url: Optional[str] = None,
) -> Dict:
    """Interactively prompt the user for axis cuts on an NJM download.

    Shows the available parameter ranges from the remote lookup table,
    then asks if the user wants to cut on each axis.

    Returns a dict with keys: teff_range, logg_range, meta_range, wl_range
    (each either a (min, max) tuple or None).
    """
    cuts = {'teff_range': None, 'logg_range': None, 'meta_range': None, 'wl_range': None}

    print(f"\n  ── Axis Cuts for {name} ──")
    print(f"  You can restrict the download to a subset of the grid.")
    print(f"  Leave blank to download everything.\n")

    info = grabber.get_grid_info(name, model_url=model_url)
    if info:
        print(f"  Available grid ({info['n_spectra']} spectra):")
        if 'teff_min' in info:
            print(f"    Teff:  {info['teff_min']:.0f} – {info['teff_max']:.0f} K  ({info['teff_unique']} values)")
        if 'logg_min' in info:
            print(f"    logg:  {info['logg_min']:.2f} – {info['logg_max']:.2f}    ({info['logg_unique']} values)")
        if 'meta_min' in info:
            print(f"    [M/H]: {info['meta_min']:+.2f} – {info['meta_max']:+.2f}    ({info['meta_unique']} values)")
        print()
    else:
        print("  (Could not fetch grid info — cuts will still work if lookup_table.csv is available)\n")

    cuts['teff_range'] = _ask_range("  Teff range (e.g. '3500,8000') or blank for all: ")
    cuts['logg_range'] = _ask_range("  logg range (e.g. '3.5,5.0') or blank for all: ")
    cuts['meta_range'] = _ask_range("  [M/H] range (e.g. '-1.0,0.5') or blank for all: ")
    cuts['wl_range']   = _ask_range("  Wavelength range in Å (e.g. '3000,10000') or blank for all: ")

    if any(v is not None for v in cuts.values()):
        print(f"\n  Applied cuts:")
        if cuts['teff_range']:
            print(f"    Teff:       {cuts['teff_range'][0]:.0f} – {cuts['teff_range'][1]:.0f} K")
        if cuts['logg_range']:
            print(f"    logg:       {cuts['logg_range'][0]:.2f} – {cuts['logg_range'][1]:.2f}")
        if cuts['meta_range']:
            print(f"    [M/H]:      {cuts['meta_range'][0]:+.2f} – {cuts['meta_range'][1]:+.2f}")
        if cuts['wl_range']:
            print(f"    Wavelength: {cuts['wl_range'][0]:.1f} – {cuts['wl_range'][1]:.1f} Å")
        print()
    else:
        print("\n  No cuts — downloading full grid.\n")

    return cuts


# ------------ Workflow Runners ------------

def run_rebuild_flow(
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models: Optional[List[str]] = None,
    rebuild_h5: bool = True,
    rebuild_flux_cube: bool = True,
    *,
    flux_unit: Optional[str] = None,
) -> None:
    """Clean each chosen local grid and rebuild its lookup table, HDF5 bundle and flux cube.

    ``flux_unit`` (a ``spectra_cleaner.FLUX_UNITS`` name) is used for grids
    whose spectra do not declare their flux unit; without it sed-tools asks.
    Every grid that can be rebuilt is; then RuntimeError lists each grid that
    could not be, and why.
    """
    ensure_dir(base_dir)

    cand = _rebuildable_grids(base_dir)
    if not cand:
        raise FileNotFoundError(f"No local spectral grids found under {base_dir}")

    selected = models if models is not None else _ask_rebuild_selection(cand)
    if not selected:
        print("Nothing selected.")
        return

    failures = []
    for model_name in selected:
        print("\n" + "=" * 64)
        print(f"[rebuild] {model_name}")
        if model_name not in cand:
            print(f"[rebuild] {model_name} is not a local spectral grid under {base_dir}")
            failures.append(f"{model_name}: not a local spectral grid under {base_dir}")
            continue
        try:
            problems = _rebuild_grid(os.path.join(base_dir, model_name), model_name,
                                     rebuild_h5, rebuild_flux_cube, flux_unit)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"[rebuild] {model_name} failed: {exc}")
            problems = [f"{type(exc).__name__}: {exc}"]
        failures.extend(f"{model_name}: {problem}" for problem in problems)

    if failures:
        raise RuntimeError(f"Spectral grids not rebuilt ({len(failures)}): " + "; ".join(failures))
    print("\nRebuild complete.")


def _rebuildable_grids(base_dir: str) -> List[str]:
    """Folders under ``base_dir`` holding .txt spectra or an .h5 file."""
    cand = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p):
            continue
        has_txt = any(fn.lower().endswith(".txt") for fn in os.listdir(p))
        has_h5 = any(fn.lower().endswith(".h5") for fn in os.listdir(p))
        if has_txt or has_h5:
            cand.append(name)
    return cand


def _ask_rebuild_selection(cand: List[str]) -> List[str]:
    """Ask which grids to rebuild ('all', or numbers and ranges like 1,3-6) until the answer parses."""
    print("\nLocal spectral grids available for rebuild:")
    print("-" * 64)
    for i, m in enumerate(cand, 1):
        print(f"{i:3d}. {m}")
    print("\nSelect indices (comma / ranges like 3-6) or 'all':")
    while True:
        raw = input("> ").strip().lower()
        if raw == "all":
            return cand
        try:
            return [cand[i] for i in parse_multi_selection(raw, len(cand))]
        except ValueError as exc:
            print(f"  {exc}")


def _ask_flux_unit(grid: str, txt_files: List[str], flux_unit: Optional[str] = None) -> Optional[str]:
    """The flux unit to clean a grid with: None when its spectra declare theirs, else ``flux_unit`` or the user's choice.

    ``flux_unit`` (from --flux-unit) is used only for spectra that do not
    declare their unit; a grid that declares one keeps it, and a line says
    so.  Without ``flux_unit`` the user must type a unit: Enter alone does
    not accept the guess.  Answering 'n', or having no terminal to ask on,
    raises ValueError: the grid is not cleaned.
    """
    guess = undeclared_flux_unit(txt_files)
    if guess is None:
        if flux_unit is not None:
            print(f"[{grid}] its spectra declare their flux unit; --flux-unit {flux_unit} is not used for it")
        return None
    if flux_unit is not None:
        print(f"[{grid}] flux unit not declared; using --flux-unit {flux_unit} (best guess was {guess})")
        return flux_unit
    if not sys.stdin.isatty():
        raise ValueError(
            f"Flux unit of {grid} is not declared; best guess: '{guess}', {FLUX_UNITS[guess][2]} "
            f"({FLUX_GUESS_WHY}). Give it with --flux-unit {guess}, or another of: {', '.join(FLUX_UNITS)}. "
            "Without --flux-unit, sed-tools asks for it when run in a terminal.")
    print(f"\nFlux unit of {grid} is not declared. Accepted flux units:")
    for name, (_, _, description) in FLUX_UNITS.items():
        print(f"  {name:11s} {description}")
    print(_wrap(f"Best guess: {guess}, {FLUX_UNITS[guess][2]}; {FLUX_GUESS_WHY}.", indent="  ", hanging="    "))
    question = f"Flux unit of {grid} (type its name, e.g. {guess}), or n to leave it uncleaned: "
    while True:
        answer = input(question).strip()
        if answer in FLUX_UNITS:
            return answer
        if answer.lower() in ("n", "no"):
            raise ValueError(f"No flux unit chosen for {grid}; it was not cleaned")
        if answer.lower() in ("", "y", "yes"):
            print(f"  Type the unit's name; the guess ({guess}) is not taken without it.")
        else:
            print(f"  {answer!r} is not one of: {', '.join(FLUX_UNITS)}")


def _txt_files(model_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(model_dir, "*.txt")))


def _ask_import_flux_unit(path: str, name: Optional[str], flux_unit: Optional[str] = None) -> Optional[str]:
    """_ask_flux_unit for the spectra SED.import_grid would import from path."""
    from .api import _import_sources

    files, src_dir = _import_sources(Path(path).expanduser())
    return _ask_flux_unit(name or src_dir.name, [str(f) for f in files], flux_unit)


def _report_import_flux_unit(path: str, name: Optional[str], flux_unit: Optional[str]) -> None:
    """For ``import --dry-run``: whether the spectra declare their flux unit, and the guess when not."""
    from .api import _import_sources

    files, src_dir = _import_sources(Path(path).expanduser())
    guess = undeclared_flux_unit([str(f) for f in files])
    grid = name or src_dir.name
    if guess is None:
        print(f"[import] {grid}: the spectra declare their flux unit"
              + (f"; --flux-unit {flux_unit} would not be used" if flux_unit else ""))
    elif flux_unit:
        print(f"[import] {grid}: flux unit not declared; --flux-unit {flux_unit} would be used "
              f"(best guess: {guess})")
    else:
        print(f"[import] {grid}: flux unit not declared; best guess: {guess}, {FLUX_UNITS[guess][2]} "
              f"({FLUX_GUESS_WHY}). Importing needs --flux-unit, or an answer in a terminal.")


def _rebuild_grid(model_dir: str, model_name: str, rebuild_h5: bool, rebuild_flux_cube: bool,
                  flux_unit: Optional[str] = None) -> List[str]:
    """Clean one local grid and rebuild its products; returns what was not done (empty when complete)."""
    from .api import _build_flux_cube, _unusable_after_cleaning

    flux_unit = _ask_flux_unit(model_name, _txt_files(model_dir), flux_unit)
    summary = clean_model_dir(model_dir, try_h5_recovery=True, rebuild_lookup=True, flux_unit=flux_unit)
    print(f"[clean] {model_name}: total={summary['total']}")

    if not glob.glob(os.path.join(model_dir, "*.txt")):
        print(f"[rebuild] no spectra (.txt) present after cleaning; skipping {model_name}.")
        return ["no spectra (.txt) present after cleaning"]

    regenerate_lookup_table(model_dir)

    if rebuild_h5:
        out_h5 = os.path.join(model_dir, f"{model_name}.h5")
        build_h5_bundle_from_txt(model_dir, out_h5)

    if rebuild_flux_cube:
        _build_flux_cube(model_dir)

    unusable = _unusable_after_cleaning(summary)
    if unusable:
        return [f"{len(unusable)} spectra unusable after cleaning: {', '.join(unusable)}"]
    return []


class _ModelOption:
    """Display wrapper for a discoverable spectral grid, used by run_spectra_flow."""
    def __init__(self, name: str, sources: List[str]) -> None:
        self.name = name
        self.sources = sources
        self.label = f"{name} {''.join(f'[{s}]' for s in sources)}"


def _print_cleaning_report(summary: Dict) -> None:
    """Print the unit-detection and per-file outcome report after clean_model_dir."""
    n_total    = summary['total']
    det_stats  = summary.get('detection_stats', {})
    cat_units  = summary.get('catalog_units', {}) or {}
    det_status = det_stats.get('status', '')
    sample_size = det_stats.get('sample_size', '?')
    detected_wl   = cat_units.get('wavelength', 'unknown')
    detected_flux = cat_units.get('flux', 'unknown')

    # Unit detection summary
    if det_status == 'already_standardized':
        print(f" Method    : Sampled {sample_size}/{n_total} files")
        print(f" Result    : All samples already standardized")
        print(f" Wavelength: angstrom (Å)")
        print(f" Flux      : F_lambda (erg/s/cm²/Å)")
    elif cat_units:
        print(f" Method    : Spectral grid consensus ({sample_size}/{n_total} files sampled)")
        print(f" Wavelength: {detected_wl} ({det_stats.get('wavelength_agreement', '?')} agreement)")
        print(f" Flux      : {detected_flux} ({det_stats.get('flux_agreement', '?')} agreement)")
        print(f" Confidence: {cat_units.get('confidence', 'unknown')}")
    else:
        print(f" Status    : {det_status or 'No unit detection performed'}")
        print(f" Wavelength: {detected_wl}")
        print(f" Flux      : {detected_flux}")

    # Per-file outcome counts
    units_already_standard = (detected_wl == 'angstrom' and detected_flux == 'flam')
    n_converted = len(summary['converted'])
    n_recovered = len(summary['recovered'])
    n_skipped   = len(summary['skipped_already'])
    n_invalid   = len(summary['skipped_invalid'])
    n_index     = len(summary['skipped_index'])
    n_error     = len(summary['error'])

    # Files "converted" from units that were already Å and F_λ were only cleaned and tagged.
    n_tagged = 0
    if n_converted > 0 and units_already_standard:
        n_tagged, n_converted = n_converted, 0

    print(f" Total files: {n_total}")
    if n_converted > 0:
        print(f"Converted     : {n_converted} files")
        if detected_wl != 'angstrom':
            print(f"   λ: {detected_wl} → angstrom")
        if detected_flux != 'flam':
            print(f"   F: {detected_flux} → F_lambda")
    if n_tagged    > 0: print(f"Cleaned       : {n_tagged} files  (units already Å + F_λ)")
    if n_recovered > 0: print(f"Recovered     : {n_recovered} files  (wavelengths from HDF5)")
    if n_skipped   > 0: print(f"Already done  : {n_skipped} files  (units_standardized header)")
    if n_invalid   > 0: print(f"Invalid       : {n_invalid} files  (empty, corrupt, or λ ≤ 0)")
    if n_index     > 0: print(f"Index grids   : {n_index} files  (λ=0,1,2… no HDF5 source)")
    if n_error     > 0: print(f"Errors        : {n_error} files")

    n_usable = n_converted + n_tagged + n_recovered + n_skipped
    print()
    if   n_usable == n_total: print(f"All {n_usable} files ready for use")
    elif n_usable  > 0:       print(f"{n_usable}/{n_total} files usable")
    else:                     print("No usable files")


def _build_data_products(
    name: str,
    model_dir: str,
    src: str,
    force_bundle_h5: bool,
    build_flux_cube: bool,
) -> List[str]:
    """Build HDF5 bundle, lookup table, and flux cube for a downloaded spectral grid.

    A bundle or flux cube older than a spectrum beside it was built without
    that spectrum (a selection, before the rest was downloaded): the bundle is
    then rebuilt even without ``force_bundle_h5``, and a cube left as it is
    because ``build_flux_cube`` is False is returned as a problem.  Returns
    the problems; empty when the products are complete.
    """
    from .api import _build_flux_cube, _flux_cube_file, _older_than_a_spectrum

    # HDF5 bundle
    out_h5 = os.path.join(model_dir, f"{name}_bundle.h5" if src == "msg" else f"{name}.h5")
    if force_bundle_h5 or not os.path.exists(out_h5) or _older_than_a_spectrum(Path(out_h5), Path(model_dir)):
        build_h5_bundle_from_txt(model_dir, out_h5)
        print(f"HDF5 bundle   : {os.path.basename(out_h5)}")
    else:
        print(f"HDF5 bundle   : {os.path.basename(out_h5)} (exists)")

    # Lookup table
    regenerate_lookup_table(model_dir)
    print(f"Lookup table  : lookup_table.csv")

    # Flux cube
    if not build_flux_cube:
        cube = _flux_cube_file(Path(model_dir))
        if cube is not None and _older_than_a_spectrum(cube, Path(model_dir)):
            return [f"{cube} was built before some of the spectra beside it, so it does not include them; "
                    f"it was not rebuilt (--no-cube). Rebuild it with: sed-tools rebuild --grids {name}"]
        return []

    if _build_flux_cube(model_dir).name == "variants_index.csv":
        variants_index = os.path.join(model_dir, "variants_index.csv")
        with open(variants_index) as vf:
            header = vf.readline()
            vrows = list(csv.DictReader(vf, fieldnames=[
                c.lstrip("#").strip() for c in header.strip().split(",")
            ]))
        print(f"Flux cubes    : {len(vrows)} MESA-ready subgrid(s) created")
        for row in vrows:
            vname = row.get("variant_name", row.get("path", "?"))
            print(f"  -> {os.path.join(model_dir, vname)}/")
        print("\n  Point MESA Colors to one of the above subdirectories.")
        print(f"  See {variants_index} for the full inventory.")
    else:
        print("Flux cube     : flux_cube.bin")
    return []


def run_spectra_flow(
    source: str,
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models: Optional[List[str]] = None,
    workers: int = 5,
    force_bundle_h5: bool = True,
    build_flux_cube: bool = True,
    *,
    flux_unit: Optional[str] = None,
    teff_range: Optional[Tuple[float, float]] = None,
    logg_range: Optional[Tuple[float, float]] = None,
    meta_range: Optional[Tuple[float, float]] = None,
    wl_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Discover spectral grids, then download, clean and build each chosen one.

    ``flux_unit`` is used for grids whose spectra do not declare their flux
    unit (see run_rebuild_flow).  The ``*_range`` cuts, each ``(min, max)``,
    are passed to every source (SVO refuses them); without any, an NJM
    download asks for cuts when run in a terminal and takes the full grid
    otherwise.

    A grid that cannot be completed is reported and the rest are processed;
    then RuntimeError lists every grid that was not completed.
    """
    ranges = {"teff_range": teff_range, "logg_range": logg_range, "meta_range": meta_range, "wl_range": wl_range}
    ensure_dir(base_dir)

    src_list = _spectra_sources(source)
    grabs = _spectra_grabbers(src_list, base_dir, workers)
    model_sources, discovery_failures = _discover_spectral_grids(src_list, grabs)
    if "njm" in src_list and "njm" not in grabs:
        discovery_failures.insert(0, "njm: mirror unreachable")
    if not model_sources:
        raise RuntimeError("No spectral grids discovered" + "".join(f"; {f}" for f in discovery_failures))

    all_models = [_ModelOption(n, srcs) for n, srcs in sorted(model_sources.items())]
    if models is not None:
        chosen, failures = _requested_spectral_grids(models, all_models)
    elif not sys.stdin.isatty():
        raise ValueError(f"No terminal to choose from the {len(all_models)} spectral grids found; "
                         "name them with --grids NAME ... (or --grids all)")
    else:
        idxs = prompt_choice(all_models, label="Spectral grids", allow_back=True, multi=True)
        if idxs is None or idxs == -1:
            return
        chosen, failures = [(all_models[i].sources, all_models[i].name) for i in idxs], []

    for sources, name in chosen:
        try:
            problems = _download_and_build_grid(name, sources, grabs, base_dir, force_bundle_h5, build_flux_cube,
                                                flux_unit=flux_unit, ranges=ranges)
        except (ValueError, RequestException, OSError, RuntimeError) as exc:
            print(f"[{name}] failed: {exc}")
            problems = [f"{type(exc).__name__}: {exc}"]
        failures.extend(f"{name}: {problem}" for problem in problems)

    print("\n" + "─" * 64)
    if failures:
        raise RuntimeError(f"Spectral grids not completed ({len(failures)}): " + "; ".join(failures))
    print("Done.")


def _spectra_sources(source: str) -> List[str]:
    """Source names from --source: one name, a comma-separated list, 'both' or 'all'."""
    if source.lower() == "all":
        return ["njm", "svo", "msg", "mast"]
    if source.lower() == "both":
        return ["njm", "svo", "msg"]
    if "," in source:
        return [s.strip().lower() for s in source.split(",")]
    return [source.lower()]


def _spectra_grabbers(src_list: List[str], base_dir: str, workers: int) -> Dict:
    """A grabber per requested source; NJM is left out, with a message, when the mirror is unavailable."""
    grabs: Dict = {}
    if "njm" in src_list:
        njm = NJMSpectraGrabber(base_dir=base_dir, max_workers=workers)
        if njm.is_available():
            grabs["njm"] = njm
        else:
            others = [s for s in src_list if s != "njm"]
            print("[njm] Mirror unreachable; " + (f"using {', '.join(others)}" if others
                                                   else "no other source was requested (--source)"))
    if "svo"  in src_list: grabs["svo"]  = SVOSpectraGrabber(base_dir=base_dir, max_workers=workers)
    if "msg"  in src_list: grabs["msg"]  = MSGSpectraGrabber(base_dir=base_dir, max_workers=workers)
    if "mast" in src_list: grabs["mast"] = MASTSpectraGrabber(base_dir=base_dir, max_workers=workers)
    return grabs


def _discover_spectral_grids(src_list: List[str], grabs: Dict) -> Tuple[Dict[str, List[str]], List[str]]:
    """Each grid name with the sources listing it (in --source order), and each source that failed."""
    model_sources: Dict[str, List[str]] = {}
    failures = []
    for s in src_list:
        if s not in grabs:
            continue
        try:
            names = grabs[s].discover_models()
        except (ValueError, RequestException, OSError, RuntimeError) as exc:
            logger.warning("[%s] Spectral grid discovery failed: %s; using other sources", s, exc)
            failures.append(f"{s}: discovery failed: {exc}")
            continue
        for model_name in names:
            model_sources.setdefault(model_name, []).append(s)
    return model_sources, failures


def _requested_spectral_grids(models: List[str], all_models: List[_ModelOption]):
    """``(sources, name)`` for each name in ``models`` ('all', NAME or SOURCE:NAME), and the names not found."""
    if len(models) == 1 and models[0].lower() == "all":
        return [(opt.sources, opt.name) for opt in all_models], []
    chosen, not_found = [], []
    for m in models:
        if ":" in m:
            s, n = m.split(":", 1)
            chosen.append(([s.strip().lower()], n.strip()))
            continue
        matching = [opt for opt in all_models if opt.name == m.strip()]
        if matching:
            chosen.append((matching[0].sources, matching[0].name))
        else:
            print(f"[skip] Spectral grid '{m.strip()}' not found")
            not_found.append(f"{m.strip()}: not listed by any source")
    return chosen, not_found


def _download_from_first_source(name: str, sources: List[str], grabs: Dict, model_dir: str,
                                ranges: Optional[Dict] = None):
    """Download ``name`` from the first active source that can supply it; returns ``(source, metadata)``.

    A source that fails is reported and the next one tried, unless the failed
    attempt changed ``model_dir``: the next source's spectra would then be
    mixed with its files.  Raises RuntimeError with every source's error when
    no source succeeds.
    """
    from .api import _directory_state

    errors = []
    for src in sources:
        if src not in grabs:
            errors.append(f"{src}: not available")
            continue
        before = _directory_state(model_dir)
        meta, error = _download_from(name, src, grabs[src], model_dir, ranges)
        if error is None:
            return src, meta
        errors.append(f"{src}: {error}")
        if _directory_state(model_dir) != before:
            raise RuntimeError(f"{src} failed after changing {model_dir}, so another source's spectra would be "
                               f"mixed with its files; no further source tried. Tried: {'; '.join(errors)}")
        logger.warning("  [%s] %s — trying next source...", src, error)
    raise RuntimeError(f"no source could supply it. Tried: {'; '.join(errors)}")


def _download_from(name: str, src: str, g, model_dir: str, ranges: Optional[Dict] = None):
    """Download ``name`` from one source: ``(metadata, None)``, or ``(metadata, what went wrong)``."""
    meta = None
    try:
        meta = g.get_model_metadata(name)
        if not meta:
            return meta, "no metadata"
        print(f"  Using source: {src}")
        n_written = g.download_model_spectra(name, meta, **_download_cuts(name, src, g, meta, ranges or {}))
    except (ValueError, RequestException, OSError, RuntimeError) as exc:
        return meta, str(exc)
    print(f"Downloaded {n_written} spectra → {model_dir}")
    if not n_written and not _is_preprocessed(meta):
        return meta, "no spectra downloaded"
    return meta, None


def _download_cuts(name: str, src: str, g, meta, ranges: Dict) -> Dict:
    """The ``*_range`` cuts for one download: those given, else asked for an NJM grid in a terminal, else none."""
    if any(value is not None for value in ranges.values()):
        return dict(ranges)
    if src != "njm":
        return {}
    if not sys.stdin.isatty():
        print("  No terminal to ask for axis cuts: downloading the full grid "
              "(--teff, --logg, --meta and --wavelength cut it)")
        return {}
    return prompt_njm_axis_cuts(name, g, model_url=meta.get("model_url"))


def _is_preprocessed(meta) -> bool:
    return isinstance(meta, dict) and meta.get("pre_processed", False)


def _download_and_build_grid(name: str, sources: List[str], grabs: Dict, base_dir: str,
                             force_bundle_h5: bool, build_flux_cube: bool, *,
                             flux_unit: Optional[str] = None, ranges: Optional[Dict] = None) -> List[str]:
    """Download one grid from its first source that can supply it, clean it and build its products.

    Returns what was not done; empty when the grid is complete.
    """
    from .api import _unusable_after_cleaning

    print("\n" + "=" * 64)
    print(f"{''.join(f'[{s}]' for s in sources)} {name}")
    print("=" * 64)

    model_dir = os.path.join(base_dir, name)
    ensure_dir(model_dir)

    src, meta = _download_from_first_source(name, sources, grabs, model_dir, ranges)

    if _is_preprocessed(meta):
        print("Pre-processed data (skipping cleaning)")
        essential = ["flux_cube.bin", "lookup_table.csv"]
        missing = [f for f in essential if not os.path.exists(os.path.join(model_dir, f))]
        print(f"Missing: {', '.join(missing)}" if missing else "All essential files present")
        return [f"missing {', '.join(missing)}"] if missing else []

    flux_unit = _ask_flux_unit(name, _txt_files(model_dir), flux_unit)
    summary = clean_model_dir(model_dir, try_h5_recovery=True, rebuild_lookup=True, flux_unit=flux_unit)
    _print_cleaning_report(summary)

    if not glob.glob(os.path.join(model_dir, "*.txt")):
        print("\n  No spectra remaining after cleaning")
        return ["no spectra remaining after cleaning"]

    problems = _build_data_products(name, model_dir, src, force_bundle_h5, build_flux_cube)
    unusable = _unusable_after_cleaning(summary)
    if unusable:
        problems.append(f"{len(unusable)} spectra unusable after cleaning: {', '.join(unusable)}")
    return problems


def _print_field(name: str, text) -> None:
    """``  name : text``, wrapped to the terminal width under the text column."""
    print(_wrap(text, indent=f"  {name:<25}: ", hanging=" " * 29))


def print_filter_availability(info: dict) -> None:
    """Print what a filter set contains and what can be composed from it."""
    print(f"\n{info['facility']}/{info['instrument']}"
          + (f"  ({len(info['bands'])} bands)" if info["bands"] else ""))
    if _missing_bands(info):
        _print_field("Incomplete", _incomplete_summary(info) + "; fetch it again when NJM is reachable to complete it")
    src = f" [{info['published_source']}]" if info["published_source"] else ""
    _print_field("Published curves contain", f"{info['published'] or 'unknown'}{src}")
    if info["published"] == "mixed":
        for band, recipe in info["published_bands"].items():
            print(f"    {band}: {recipe}")
    _print_field("Separate components", _component_summary(info))
    if info["buildable"]:
        _print_field("Can be built", ", ".join(info["buildable"]))
    if info.get("variants"):
        _print_field("Variants (API only)", _variants_summary(info["variants"]))
    if info["component_status"]:
        _print_field("Note", info["component_status"])
    for note in info.get("notes") or []:
        _print_field("Note", note)
    for role, bands in info["partial_components"].items():
        print(f"    {role} only for: {', '.join(bands)}")
    review = info["component_availability"]
    _print_field("Source review", review["status"] + (f" ({review['builder']})" if review["builder"] else ""))
    if info["missing_files"]:
        _print_field("Missing files", ", ".join(info["missing_files"]))
    for error in info["errors"]:  # errors stay on one line, as elsewhere, so they can be searched
        print(f"  {'Source unavailable':<25}: {error}")
    _print_field("Atmosphere", _atmosphere_summary(info))
    if info["detector_type"]:
        _print_field("Detector", f"{info['detector_type']}-counting")
    elif info["bands"] and info.get("availability_source") in ("original", "unknown"):
        # bands from the catalogue bundled with SED_Tools: no curve has been seen yet
        _print_field("Detector", "not known until the curves are installed")
        return
    assumed = [band for band, source in info["detector_type_sources"].items()
               if source in ("assumed", "unknown")]
    if assumed:
        _print_field("Detector unverified for", ", ".join(assumed))


def _missing_bands(info: dict) -> List[str]:
    """The bands an installed set declares but does not hold, e.g. those SVO lacks when NJM was down."""
    return [band for band in info.get("expected_bands") or [] if band not in info["bands"]]


def _incomplete_summary(info: dict, room: Optional[int] = None) -> str:
    """``4 of 6 bands installed; missing Gbp_bright, Gbp_faint``, the names cut to fit ``room`` if given."""
    missing = _missing_bands(info)
    text = f"{len(info['bands'])} of {len(info['expected_bands'])} bands installed; missing "
    return text + (", ".join(missing) if room is None else _names_to_fit(missing, room - len(text)))


def _component_summary(info: dict) -> str:
    """The instrument's own separate components, e.g. ``filter, optics, qe``."""
    if info["components"]:
        return ", ".join(info["components"])
    return "available for selected bands" if info["partial_components"] else "none identified"


def _atmosphere_sites(info: dict) -> List[str]:
    """The instrument's own atmosphere sites, default first; empty when no atmosphere is offered."""
    if not info.get("atmosphere"):
        return []
    sites = list(info.get("atmosphere_sites") or [])
    default = info.get("atmosphere_site")
    if default and default not in sites:
        sites.insert(0, default)
    return sites


def _atmosphere_summary(info: dict) -> str:
    sites = _atmosphere_sites(info)
    if sites:
        default = _default_site(info, sites)
        text = f"can be added at its own site{'s' * (len(sites) > 1)}: {_site_names(sites, default)}"
        if default is None:
            text += "; no default, so choose one with --atmosphere-site"
        return text + " (in Python, Filters.atmospheres() lists the conditions)"
    if info.get("platform") in ("space", "airborne"):
        return f"not applicable ({info['platform']}-based)"
    return "not offered (no site of its own assigned)"


def _default_site(info: dict, sites: List[str]) -> Optional[str]:
    """The instrument's default atmosphere site: the one it names, else its only site."""
    return info.get("atmosphere_site") or (sites[0] if len(sites) == 1 else None)


def _site_names(sites: List[str], default: Optional[str]) -> str:
    return ", ".join(f"{s} (default)" if s == default else s for s in sites)


def _default_variant(named: dict) -> Optional[str]:
    return next((name for name, entry in named.items() if entry.get("default")), None)


def _variants_summary(variants: dict) -> str:
    """``role: a (default), b; role2: c`` for the named alternative component curves."""
    parts = []
    for role, named in variants.items():
        default = _default_variant(named)
        names = ", ".join([f"{default} (default)"] * bool(default) + [n for n in named if n != default])
        parts.append(f"{role}: {names}" + ("" if default else f" (default: the standard {role} curve)"))
    return "; ".join(parts)


def _catalogue_entry(facility: str, instrument: str, base: str, *, offline: bool) -> dict:
    """NJM's info.json and properties.json for ``filters-info``, and where info.json came from
    (FilterCatalogue.origin); errors are reported, not raised."""
    from .api import Filters

    entry = {"info": None, "properties": None, "errors": [], "origin": None}
    for kind in ("info", "properties"):
        try:
            entry[kind] = getattr(Filters, kind)(facility, instrument, filter_root=base, offline=offline)
        except (CatalogueError, OSError, ValueError) as exc:
            if str(exc) not in entry["errors"]:
                entry["errors"].append(str(exc))
    entry["origin"] = Filters._catalogue(base, offline=offline).origin(facility, instrument, "info")
    return entry


def _print_filter_set(target: str, info: dict, catalogue: dict) -> None:
    """The filters-info report: availability (Filters.available) and NJM's catalogue entry (_catalogue_entry)."""
    print_filter_availability(info)
    print_catalogue_entry(target, catalogue)


def print_catalogue_entry(target: str, entry: dict) -> None:
    """Print the NJM description, licence summary and per-band properties of an instrument."""
    info, props = entry["info"] or {}, (entry["properties"] or {}).get("bands", {})

    if info.get("description"):
        _print_field("Description", info["description"])
    origin = entry.get("origin") or {}
    if origin.get("njm_unreachable"):
        checked = origin.get("checked")
        when = time.strftime(", checked %Y-%m-%d %H:%M UTC", time.gmtime(checked)) if checked else ""
        _print_field("Catalogue", f"from the cached copy{when}; NJM could not be asked")
    licence = info.get("licence") or {}
    if licence:
        _print_field("Licence", _licence_summary(licence))
    if info.get("references"):
        _print_field("Cite", _cite_hint(target, info["references"]))
    bands = sorted(set(props) | set(info.get("bands", {})),
                   key=lambda b: (props.get(b, {}).get("WavelengthPivot") or float("inf"), b))
    if bands:
        _print_band_table(bands, props, info.get("bands", {}))
    for error in entry["errors"]:
        print(f"  {'NJM catalogue':<25}: {error}")
    if entry["errors"] and not info and not props:
        print(_wrap("so no description, licence or band properties are shown; "
                    "the availability above does not depend on them", indent=" " * 29, hanging=" " * 29))


def _cite_hint(target: str, references: list) -> str:
    """How many references NJM records for an instrument, and the API call that lists them with the licences."""
    return f"{len(references)} references; in Python: Filters.cite('{target}')"


def _first_sentence(text: str, limit: int = 200) -> str:
    """The first sentence of ``text``, cut at a word with '…' if longer than ``limit`` characters."""
    text = " ".join(str(text).split())
    match = re.match(r'(.+?[.!?]["\')\]]?)(?=\s+["\'(\[]?[A-Z0-9]|\s*$)', text)
    sentence = match.group(1) if match else text
    if len(sentence) > limit:
        sentence = sentence[:limit].rsplit(" ", 1)[0] + " …"
    return sentence


def _licence_summary(licence: dict) -> str:
    """Redistribution status, the terms' first sentence and the licence URL."""
    text = str(licence.get("redistribution", "unknown")) + "."
    if licence.get("terms"):
        text += " " + _first_sentence(licence["terms"])
    if licence.get("url"):
        text += " " + licence["url"]
    return text


def _print_band_table(bands: List[str], props: dict, band_info: dict) -> None:
    """Band, pivot, FWHM, detector and magnitude system, with descriptions wrapped to the terminal."""
    head = f"  {'Band':<16}{'Pivot (Å)':>11}{'FWHM (Å)':>11}  {'Detector':<9}{'MagSys':<8}"
    descriptions = [" ".join(str(band_info.get(b, {}).get("Description") or "").split()) for b in bands]
    room = terminal_width() - len(head)
    beside = all(len(d) <= room for d in descriptions)  # else each goes on its own lines below its row
    print((head + ("Description" if beside else "")).rstrip())
    for band, description in zip(bands, descriptions):
        p, i = props.get(band, {}), band_info.get(band, {})
        num = lambda v: f"{v:>11.1f}" if isinstance(v, (int, float)) else f"{'-':>11}"
        row = (f"  {band:<16}{num(p.get('WavelengthPivot'))}{num(p.get('FWHM'))}  "
               f"{p.get('DetectorType', '-'):<9}{i.get('MagSys') or '-':<8}")
        if beside:
            print((row + description).rstrip())
        else:
            print(row.rstrip())
            if description:
                print(_wrap(description, indent=" " * 6, hanging=" " * 6))


_UNSTATED = "unstated"


def _airmass_arg(text):
    """argparse type: an airmass, or 'unstated' for an atmosphere published without one."""
    if str(text).strip().lower() == _UNSTATED:
        return _UNSTATED
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or 'unstated', got {text!r}") from None


def _add_filter_product_options(parser, *, components=None):
    """Keep filter fetch and composition options identical."""
    parser.add_argument("--base", default=str(FILTER_DIR_DEFAULT), help="Base filter directory")
    parser.add_argument("--components", default=components,
                        help="Roles separated by commas or '+': filter, optics, qe, published")
    parser.add_argument("--bands", nargs="+", help="Compose only these bands")
    parser.add_argument("--detector-type", choices=["energy", "photon"],
                        help="Override integration weighting in a derived copy; preserves published curves")
    parser.add_argument("--airmass", type=_airmass_arg,
                        help="Add the instrument's own-site atmosphere at this airmass ('unstated' for a "
                             "team atmosphere whose source gives none)")
    parser.add_argument("--pwv", type=float, help="Atmospheric precipitable water vapour (mm)")
    parser.add_argument("--atmosphere-site",
                        help="One of the instrument's own atmosphere sites (default: its default site)")
    parser.add_argument("--allow-heavy", action="store_true", help="Permit large original-source downloads")
    parser.add_argument("--allow-partial", action="store_true", help="Use zero outside component wavelength coverage")


# ------------ Interactive filter composition ------------

def _offer_composition(instrument_dir: Path) -> None:
    """After a download, offer to build a composed product; the build's errors propagate."""
    info = _installed_availability(instrument_dir)
    print_filter_availability(info)
    if not _has_parts(info):
        return
    question = ("Build a composed version (e.g. filter only"
                + (", or with atmosphere" if _atmosphere_sites(info) else "") + ")? [y/N] ")
    if _yes(question):
        _compose_interactively(instrument_dir, info)


def _installed_availability(instrument_dir: Path) -> dict:
    """Filters.available for an installed instrument directory, without asking any source."""
    from .api import Filters

    return Filters.available(instrument_dir.parent.name, instrument_dir.name,
                             filter_root=instrument_dir.parent.parent, include_remote=False)


def _has_parts(info: dict) -> bool:
    """Whether a curve can be built from the instrument's own parts: components, an atmosphere site,
    or an original-source builder that provides them."""
    return bool(info["buildable"] or _atmosphere_sites(info)
                or info["component_availability"]["status"] == "supported")


def _compose_interactively(instrument_dir: Path, info: dict) -> None:
    """Ask for the recipe, detector convention, large downloads and atmosphere, then build the product."""
    from .api import Filters

    filter_root = instrument_dir.parent.parent
    review = info["component_availability"]
    roles = _ask_recipe(info)
    request = {
        "components": roles,
        "detector_type": _ask_detector_type(),
        "allow_heavy": bool(review.get("large_download")
                            and _yes("Allow a large original-source download if needed? [y/N] ")),
        "atmosphere": _ask_atmosphere(info, filter_root),
    }
    try:
        Filters.compose(instrument_dir, filter_root=filter_root, **request)
    except ComponentError as exc:
        _compose_the_bands_that_work(instrument_dir, request, exc)


def _ask_recipe(info: dict) -> List[str]:
    """Ask for a recipe (a listed one, or one typed as roles) until it names known roles."""
    from .filter_components import ATMOSPHERE, INSTRUMENT, parse_components

    options = [recipe.replace("+", ",") for recipe in info["buildable"]] + ["published"]
    print("Recipes (or type your own, e.g. published,optics):")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        pick = input("Choose a recipe [1]: ").strip() or "1"
        if pick.isdigit():
            if 1 <= int(pick) <= len(options):
                return list(parse_components(options[int(pick) - 1]))
            print(f"  Choose 1 to {len(options)}, or type a recipe.")
            continue
        try:
            roles = parse_components(pick)
        except ComponentError as exc:
            print(f"  {exc}; use filter, optics, qe or published.")
            continue
        if ATMOSPHERE in roles or INSTRUMENT in roles:
            print("  Use filter, optics, qe or published; the atmosphere is asked about separately.")
        elif roles:
            return list(roles)


def _ask_detector_type() -> Optional[str]:
    pick = _ask_option("Detector convention [inherit], energy or photon: ",
                       ["inherit", "energy", "photon"], default="inherit")
    return None if pick == "inherit" else pick


def _ask_atmosphere(info: dict, filter_root: Path) -> Optional[Dict]:
    """The atmosphere to fold in, chosen by number from the instrument's own sites; None for none.

    Asked only when the instrument offers an atmosphere (ground-based, with a
    site of its own).  Its sites are listed with their source, the generic
    (model-derived) ones last; Enter keeps the curve without atmosphere.
    """
    sites = _atmosphere_sites(info)
    if not sites:
        return None
    grids = _site_grids(sites, filter_root)
    sites.sort(key=lambda site: (grids.get(site) or {}).get("kind") == "derived")  # stable: default first
    _print_site_choices(sites, _default_site(info, sites), grids)
    numbers = [str(n) for n in range(len(sites) + 1)]
    pick = _ask_option("Atmosphere [0, none]: ", numbers + sites, default="0")
    if pick == "0":
        return None
    site = sites[int(pick) - 1] if pick in numbers else pick
    airmass, pwv = _ask_conditions(site, grids.get(site))
    return {"site": site, "airmass": airmass, "pwv": pwv}


def _site_grids(sites: List[str], filter_root: Path) -> Dict[str, dict]:
    """The atmosphere grids by site, cached ones refreshed from NJM's manifests (checked once a day)."""
    from .api import Filters

    try:
        grids = Filters.atmospheres(filter_root=filter_root)
    except (RequestException, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"could not list the atmosphere grids for {', '.join(sites)}: {exc}") from exc
    return grids


def _print_site_choices(sites: List[str], default: Optional[str], grids: Dict[str, dict]) -> None:
    """``0. none``, then each site numbered, with its kind where its manifest states one and its source."""
    def name(site):
        notes = [note for note in ("default site" if site == default else "", (grids.get(site) or {}).get("kind"))
                 if note]
        return site + (f" ({', '.join(notes)})" if notes else "")

    names = [name(site) for site in sites]
    width = max(len("none"), *map(len, names))
    print("Atmosphere to fold in:")
    print(f"  0. {'none':<{width}}  the curve without atmosphere")
    for number, (site, label) in enumerate(zip(sites, names), 1):
        source = (grids.get(site) or {}).get("source") or "not downloaded yet"
        print(_cut(f"  {number}. {label:<{width}}  {' '.join(source.split())}", terminal_width()))


def _ask_conditions(site: str, grid: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    """Airmass (None for 'unstated') and PWV, defaulting to the first conditions the site's grid has."""
    if grid:
        print(_wrap(_conditions_text(grid), indent=f"  {site}: ", hanging=" " * (len(site) + 4)))
    else:
        print(f"  {site}: no grid listed yet; the conditions you give are generated if the site allows it")
    if grid and grid["airmass"]:
        default = grid["airmass"][0]
    elif grid and grid.get("unstated_airmass"):
        default = _UNSTATED
    else:
        default = _NO_DEFAULT
    words = (_UNSTATED,) if not grid or grid.get("unstated_airmass") else ()
    shown = "" if default is _NO_DEFAULT else f" [{default:g}]" if isinstance(default, float) else f" [{default}]"
    airmass = _ask_number(f"Airmass{shown}: ", default=default, words=words)
    pwv = None
    if not grid:
        pwv = _ask_number("PWV mm (blank if not applicable): ", default=None)
    elif grid["pwv"]:
        pwv = _ask_number(f"PWV mm [{grid['pwv'][0]:g}]: ", default=grid["pwv"][0])
    return (None if airmass == _UNSTATED else airmass), pwv


def _conditions_text(grid: dict) -> str:
    """The airmasses and PWVs an atmosphere grid has, e.g. ``airmass 1, 1.5; pwv 2.5 mm``."""
    parts = []
    if grid.get("airmass"):
        parts.append("airmass " + ", ".join(f"{a:g}" for a in grid["airmass"]))
    if grid.get("unstated_airmass"):
        parts.append("airmass 'unstated' (as published)")
    if grid.get("pwv"):
        parts.append("pwv " + ", ".join(f"{p:g}" for p in grid["pwv"]) + " mm")
    return "; ".join(parts) or "no conditions listed"


def _compose_the_bands_that_work(instrument_dir: Path, request: Dict, error: ComponentError) -> None:
    """When composing every band failed, offer the bands that can be composed; re-raise ``error`` otherwise."""
    from .api import Filters

    working, failed = _composable_bands(instrument_dir, request)
    if not working or not failed:
        raise error
    print(f"Not every band can be composed as asked ({len(failed)} of {len(working) + len(failed)}):")
    for band, why in failed.items():
        print(_wrap(why, indent=f"  {band}: ", hanging="    "))
    if not _yes(f"Build only {', '.join(working)}? [y/N] "):
        raise error
    Filters.compose(instrument_dir, filter_root=instrument_dir.parent.parent, bands=working, **request)


def _composable_bands(instrument_dir: Path, request: Dict) -> Tuple[List[str], Dict[str, str]]:
    """Each primary band composed in memory (nothing is written): the bands that work, and why the others fail."""
    from .filter_components import InstrumentManifest, compose_band
    from .filter_library import FilterLibrary

    manifest = InstrumentManifest.load(instrument_dir)
    atmosphere_root = FilterLibrary(instrument_dir.parent.parent).atmosphere_root
    working, failed = [], {}
    for band in manifest.primary_bands():
        try:
            compose_band(instrument_dir, band, request["components"], atmosphere=request["atmosphere"],
                         atmosphere_root=atmosphere_root, manifest=manifest)
        except (ComponentError, OSError) as exc:
            failed[band] = str(exc)
        else:
            working.append(band)
    return working, failed


# ------------ Interactive filter browser ------------

def run_filters_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> int:
    """Browse available instruments; use the API for every download and product.

    A download or build that fails is reported at once, and every one that
    failed is listed again when browsing ends.  Returns the exit status:
    1 if anything failed, else 0.
    """
    from .filter_library import FilterLibrary

    library = FilterLibrary(base_dir)
    shown: set = set()
    facilities = _listed_facilities(library, shown)
    failures: List[str] = []
    download = lambda facility, instrument: _offer_download(library, facility, instrument, failures)
    while facilities:
        selected = prompt_choice(_picker_labels(facilities), "Filter Facilities", multi=True)
        if selected is None:
            break
        if len(selected) > 1:
            _download_every_instrument(library, [facilities[i] for i in selected], base_dir, failures, shown=shown)
            break
        if not _browse_instruments(library, facilities[selected[0]], download, shown=shown):
            break
    if not failures:
        return 0
    print(f"\n[filters] Failed ({len(failures)}):")
    for failure in failures:
        print(_wrap(failure, indent="  - ", hanging="    "))
    return 1


def run_filter_info_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> int:
    """Pick instruments, as when downloading, and show what ``filters-info`` shows for each.

    Returns 1 if any report was incomplete because a source could not be asked, else 0.
    """
    from .filter_library import FilterLibrary

    library = FilterLibrary(base_dir)
    shown: set = set()
    facilities = _listed_facilities(library, shown)
    incomplete: List[str] = []
    show = lambda facility, instrument: _show_filter_set(library, facility, instrument, incomplete)
    while facilities:
        index = prompt_choice(_picker_labels(facilities), "Filter Facilities")
        if index is None or not _browse_instruments(library, facilities[index], show, shown=shown):
            break
    return 1 if incomplete else 0


def run_filter_compose_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> int:
    """Pick an instrument and build a curve from its own parts, asking what is asked after a download.

    The published curves are downloaded first unless installed; an instrument
    without parts of its own is named in one line.  Errors propagate.
    """
    from .filter_library import FilterLibrary

    library = FilterLibrary(base_dir)
    picked = _pick_one_instrument(library)
    if picked is None:
        return 0
    facility, instrument, _ = _picked_names(library, *picked)
    info = library.available(facility, instrument)
    if info.get("availability_source") == "unknown":
        raise RuntimeError(f"cannot tell what {facility}/{instrument} is made of: " + "; ".join(info["errors"]))
    if not _has_parts(info):
        print(f"{facility}/{instrument} has no separate parts to build a filter curve from.")
        return 0
    directory = library.published(facility, instrument)
    info = _installed_availability(directory)
    print_filter_availability(info)
    _compose_interactively(directory, info)
    return 0


def _pick_one_instrument(library) -> Optional[Tuple[dict, dict]]:
    """The facility and instrument entries the user picks, or None if they quit."""
    shown: set = set()
    facilities = _listed_facilities(library, shown)
    picked: List[Tuple[dict, dict]] = []

    def pick(facility: dict, instrument: dict) -> bool:
        picked.append((facility, instrument))
        return False  # stop browsing

    while facilities and not picked:
        index = prompt_choice(_picker_labels(facilities), "Filter Facilities")
        if index is None or not _browse_instruments(library, facilities[index], pick, shown=shown):
            break
    return picked[0] if picked else None


def _listed_facilities(library, shown: set) -> List[dict]:
    """The facilities the library lists; a listing failure is printed once."""
    catalogue = library.facilities()
    _print_listing_errors(catalogue["errors"], shown)
    if not catalogue["entries"]:
        print("No facilities found.")
    return catalogue["entries"]


#: Sources in the order Filters.fetch tries them, and how the browser names them.
_SOURCE_ORDER = ("local", "njm", "original", "svo")
_SOURCE_NAMES = {"njm": "NJM", "original": "the original source", "svo": "SVO"}


def _fetch_key(entry: dict) -> str:
    """The spelling to download an entry with: its ``fetch_key`` when the catalogue gives one."""
    return entry.get("fetch_key") or entry["key"]


def _picker_labels(entries: List[dict]) -> List[str]:
    """The name the pickers show for each catalogue entry: the library's label, nothing else."""
    return [entry.get("label") or entry["key"] for entry in entries]


def _source_rank(source: str) -> int:
    return _SOURCE_ORDER.index(source) if source in _SOURCE_ORDER else len(_SOURCE_ORDER)


def _cut(text: str, width: int) -> str:
    """``text`` cut to ``width`` characters, ending in '…' when anything was removed."""
    return text if len(text) <= width else text[:max(width - 1, 0)] + "…"


#: What FilterLibrary.facilities() and instruments() list when NJM cannot be reached. NJM is
#: optional: the other sources are listed instead, and the mirror client has already said so.
_NJM_UNAVAILABLE = "NJM unavailable"


def _listing_failures(errors: List[str]) -> List[str]:
    """The catalogue listing errors that leave a list incomplete (an unreachable NJM does not)."""
    return [error for error in errors if error != _NJM_UNAVAILABLE]


def _print_listing_errors(errors: List[str], shown: set) -> None:
    """Print each listing failure once per browsing session."""
    for error in _listing_failures(errors):
        if error not in shown:
            shown.add(error)
            print(f"[filters] {error}")


def _download_every_instrument(library, facilities: List[dict], base_dir: str, failures: List[str], *,
                               shown: Optional[set] = None) -> None:
    """After confirming the count, download every instrument of each facility.

    An incomplete instrument list is a failure too.
    """
    shown = set() if shown is None else shown
    listings = []
    for facility in facilities:
        result = library.instruments(facility["key"])
        _print_listing_errors(result["errors"], shown)
        for error in _listing_failures(result["errors"]):
            failures.append(f"{facility['key']}: instrument list incomplete: {error}")
        listings.append((facility, result["entries"]))
    count = sum(len(entries) for _, entries in listings)
    print(f"\n{count} instrument{'s' * (count != 1)} in {len(facilities)} facilities: "
          + ", ".join(f"{f['label']} ({len(entries)})" for f, entries in listings))
    if not _yes(f"Download all {count}? [y/N] "):
        print("Nothing downloaded.")
        return
    for facility, entries in listings:
        for instrument in entries:
            _fetch_instrument(library, _fetch_key(facility), _fetch_key(instrument), failures)


def _browse_instruments(library, facility_entry: dict, act, *, shown: Optional[set] = None) -> bool:
    """Pick instruments of one facility, calling ``act(facility_entry, instrument_entry)`` for each.

    ``act`` returns whether to show the instrument list again.  Returns False if the user quit.
    """
    facility = _fetch_key(facility_entry)
    result = library.instruments(facility_entry["key"])
    _print_listing_errors(result["errors"], set() if shown is None else shown)
    instruments = result["entries"]
    if not instruments:
        print(f"No instruments found for {facility}.")
        return True
    _print_instrument_descriptions(facility, instruments)
    while True:
        index = prompt_choice(_picker_labels(instruments), f"Instruments for {facility}", allow_back=True)
        if index is None:
            return False
        if index == -1 or not act(facility_entry, instruments[index]):
            return True


def _print_instrument_descriptions(facility: str, instruments: List[dict]) -> None:
    """Each instrument of a facility, named in full, with its one-line description cut to the terminal."""
    labels = _picker_labels(instruments)
    width, name_width = terminal_width(), max(map(len, labels))
    print(f"\n{facility}: {len(labels)} instrument{'s' * (len(labels) != 1)}")
    for label, entry in zip(labels, instruments):
        description = " ".join(str(entry.get("description") or "").split()) or "no description"
        room = width - name_width - 4
        print(f"  {label:<{name_width}}  {_cut(description, room)}" if room > 1 else f"  {label}")


def _offer_download(library, facility_entry: dict, instrument_entry: dict, failures: List[str]) -> bool:
    """Describe a picked instrument, then download it unless the user declines.

    Returns whether to show the instrument list again: at once when the
    download is declined, else if the user asks for another instrument.
    """
    facility, instrument, installed = _picked_names(library, facility_entry, instrument_entry)
    _print_pick_summary(library, facility, instrument, instrument_entry, installed)
    if installed is None and input(f"Download {facility}/{instrument}? [Y/n] ").strip().lower().startswith("n"):
        return True
    _fetch_instrument(library, facility, instrument, failures, offer=True)
    return _yes("Choose another instrument? [y/N] ")


def _show_filter_set(library, facility_entry: dict, instrument_entry: dict, incomplete: List[str]) -> bool:
    """Print what ``filters-info`` prints for a picked instrument; returns whether to pick another."""
    facility, instrument, _ = _picked_names(library, facility_entry, instrument_entry)
    info = library.available(facility, instrument)
    _print_filter_set(f"{facility}/{instrument}", info,
                      _catalogue_entry(facility, instrument, str(library.root), offline=False))
    incomplete.extend(_availability_errors(info))
    return _yes("Choose another instrument? [y/N] ")


def _picked_names(library, facility_entry: dict, instrument_entry: dict) -> Tuple[str, str, Optional[Path]]:
    """The names to fetch a picked instrument with (the library's ``fetch_key``), and its installed
    directory when it holds the complete set, else None."""
    facility, instrument = _fetch_key(facility_entry), _fetch_key(instrument_entry)
    try:
        directory = library.installed_directory(facility, instrument)
    except ComponentError:  # installed under two spellings: the download step names both
        directory = None
    return facility, instrument, directory if directory is not None and library.published_complete(directory) else None


def _print_pick_summary(library, facility: str, instrument: str, entry: dict, installed: Optional[Path]) -> None:
    """Before the download question, a short version of what ``filters-info`` says about the instrument."""
    info = library.available(facility, instrument)
    catalogue = _catalogue_entry(facility, instrument, str(library.root), offline=False)
    for line in _pick_summary(f"{facility}/{instrument}", entry, info, catalogue,
                              _download_origin(library, entry, installed)):
        print(line)


#: The width of a summary line's ``  Name       : `` part.
_FIELD_WIDTH = 15


def _download_origin(library, entry: dict, installed: Optional[Path]) -> str:
    """Where a download comes from: the installed copy, or the sources the library lists in download order.

    The installed copy's full path is given when it fits the terminal beside
    the field name, else its place in the filter directory.
    """
    if installed is not None:
        text = f"installed in {installed}"
        if len(text) > terminal_width() - _FIELD_WIDTH:
            text = f"installed in the filter directory, as {installed.relative_to(library.root)}"
        return text
    sources = sorted(set(entry.get("sources") or []) - {"local"}, key=_source_rank)
    njm_down = "njm" in sources and not library.mirror.is_available()
    names = [_SOURCE_NAMES.get(source, source) for source in sources if not (source == "njm" and njm_down)]
    if not names:
        return "only the installed copy, which is incomplete" if "local" in (entry.get("sources") or []) \
            else "no source lists it"
    return ", then ".join(names) + ("; NJM is unreachable" if njm_down else "")


def _pick_summary(target: str, entry: dict, info: dict, catalogue: dict, origin: str) -> List[str]:
    """The lines of the summary: description, sources, bands, components, atmosphere and licence.

    ``info`` is Filters.available's report and ``catalogue`` _catalogue_entry's,
    as filters-info shows them; each line is cut with '…' to fit the terminal.
    """
    width = terminal_width()
    label = lambda name: f"  {name:<{_FIELD_WIDTH - 4}}: "
    field = lambda name, text: _cut(label(name) + " ".join(str(text).split()), width)
    njm = catalogue["info"] or {}
    description = entry.get("description") or _first_sentence(njm.get("description") or "")
    count = len(info["bands"])
    lines = ["", target + (f"  ({count} band{'s' * (count != 1)})" if count else "")]
    if description:
        lines.append(_cut("  " + " ".join(description.split()), width))
    lines.append(field("From", origin))
    if _missing_bands(info):
        lines.append(field("Incomplete", _incomplete_summary(info, width - _FIELD_WIDTH)))
    lines += [field("Note", note) for note in [info.get("component_status"), *(info.get("notes") or [])] if note]
    lines += [field("Unavailable", error) for error in info["errors"]]
    bands = _names_to_fit(info["bands"], width - _FIELD_WIDTH) if info["bands"] else "not known"
    lines.append(field("Bands", bands))
    lines.append(field("Components", _curve_contents(info)))
    lines.append(field("Separate", _component_summary(info)))
    if info.get("variants"):
        lines.append(field("Variants", "API only; " + _variants_summary(info["variants"])))
    sites = _atmosphere_sites(info)
    lines.append(field("Atmosphere", f"own site{'s' * (len(sites) > 1)}: "
                       + _site_names(sites, _default_site(info, sites)) if sites else _atmosphere_summary(info)))
    if njm.get("licence"):
        lines.append(field("Licence", _licence_summary(njm["licence"])))
    elif catalogue["errors"]:
        lines.append(field("Licence", f"not known ({catalogue['errors'][0]})"))
    if njm.get("references"):
        lines.append(field("Cite", _cite_hint(target, njm["references"])))
    return lines


def _names_to_fit(names: List[str], room: int) -> str:
    """``names`` joined with commas, cut after a name to fit ``room`` characters, ending '… (N more)'."""
    for shown in range(len(names), -1, -1):
        rest = len(names) - shown
        text = ", ".join(names[:shown] + ([f"… ({rest} more)"] if rest else []))
        if len(text) <= room or shown == 0:
            return text


def _curve_contents(info: dict) -> str:
    """What each published curve contains, e.g. ``filter+optics+qe in each curve``."""
    published = info["published"] or "unknown"
    if published == "mixed":
        return "differs by band: " + ", ".join(sorted(set(info["published_bands"].values())))
    return published if published == "unknown" else f"{published} in each curve"


def _fetch_instrument(library, facility: str, instrument: str, failures: List[str], *,
                      offer: bool = False) -> None:
    """Download one instrument (then offer a composed product); a failure is printed and recorded.

    This is Filters.fetch with the browsing session's library, so NJM is
    checked, and an unreachable NJM reported, once per session, not per download.
    """
    try:
        directory = library.published(facility, instrument)
    except (ValueError, RequestException, OSError, RuntimeError) as exc:
        print(f"Cannot fetch: {exc}")
        failures.append(f"{facility}/{instrument}: {exc}")
        return
    if not offer:
        return
    try:
        _offer_composition(directory)
    except (ValueError, RequestException, OSError, RuntimeError) as exc:  # ComponentError is a ValueError
        print(f"Composed product of {facility}/{instrument} not built: {exc}")
        failures.append(f"{facility}/{instrument} composed product: {exc}")


class _FilterSetOption:
    """Display wrapper for a local filter-set directory, used by run_filter_combine_flow."""
    def __init__(self, path: Path, root: Path) -> None:
        self.path = path
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        self.label = f"{rel} ({_set_curve_count(path)} filters)"


def _set_curve_count(directory: Path) -> int:
    """How many curves combine_filter_sets takes from a filter set.

    Those its MESA index file lists (every ``*.dat`` without one), less its
    manifest's ``archived_bands``; so NJM's ``_svo`` and ``_old`` copies beside
    a set are not counted.
    """
    from .filter_components import archived_bands, index_listing

    archived = archived_bands(directory)
    listed = index_listing(directory)
    return sum(1 for path in directory.glob("*.dat")
               if path.stem not in archived and (listed is None or path.name in listed))


def run_filter_combine_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> None:
    """Interactive wizard for combining local filter sets."""
    from .combine_filters import combine_filter_sets, find_filter_sets

    print("\nCombine photometric filter sets")
    print("This creates one MESA-compatible filter folder and index file")
    print("from existing .dat files.")

    chosen_base = input(f"Filter base directory [{base_dir}]: ").strip() or base_dir
    filter_sets = find_filter_sets(chosen_base)
    if not filter_sets:
        print(f"No local filter sets containing .dat files were found under {chosen_base}.")
        print("Download filters first with option 1, or run `sed-tools filters`.")
        return

    root = Path(chosen_base)
    options = [_FilterSetOption(path, root) for path in filter_sets]
    selected = prompt_choice(
        options,
        "Local filter sets to combine",
        multi=True,
        page_size=50,
        max_cols=2,
    )
    if selected is None or selected == []:
        print("No filter sets selected.")
        return
    if isinstance(selected, int):
        selected = [selected]

    selected_paths = [options[i].path for i in selected]
    print("\nSelected filter sets:")
    for path in selected_paths:
        print(f"  - {path}")

    default_name = "_".join(path.name for path in selected_paths[:3]) or "CombinedFilters"
    output = input(f"Combined instrument name [{default_name}]: ").strip() or default_name
    facility = input("Output facility label [Combined]: ").strip() or None
    instrument = input(f"Output index/instrument name [{output}]: ").strip() or None

    conflict = input("Duplicate band names: rename, overwrite, or error? [rename]: ").strip().lower() or "rename"
    while conflict not in {"rename", "overwrite", "error"}:
        conflict = input("Please enter rename, overwrite, or error [rename]: ").strip().lower() or "rename"

    print("\nAbout to create a combined filter set:")
    print(f"  Base      : {chosen_base}")
    print(f"  Output    : {output}")
    print(f"  Facility  : {facility or 'Combined'}")
    print(f"  Instrument: {instrument or output}")
    print(f"  Conflicts : {conflict}")
    confirm = input("Proceed? [Y/n] ").strip().lower()
    if confirm and not confirm.startswith("y"):
        print("Cancelled.")
        return

    out = combine_filter_sets(
        output,
        selected_paths,
        filter_root=chosen_base,
        facility=facility,
        instrument=instrument,
        on_conflict=conflict,
    )
    dat_count = len(list(out.glob("*.dat")))
    index_name = instrument or out.name
    print(f"\n[filters] Combined {dat_count} filters into {out}")
    print(f"[filters] Wrote MESA index file: {out / index_name}")
    print("Use this path as the MESA instrument/filter-set directory.")


def run_ml_generator_flow(
    base_dir: str = STELLAR_DIR_DEFAULT,
    models_dir: str = "models"
) -> Optional[int]:
    """Run ML SED Generator interactive workflow; 1 when the chosen action's packages are missing."""
    from .ml_sed_generator import run_interactive_workflow
    return run_interactive_workflow(base_dir=base_dir, models_dir=models_dir)


def run_grid_densifier_flow(base_dir: str = STELLAR_DIR_DEFAULT) -> Optional[int]:
    """Run Grid Densifier interactive workflow; 1 when an ML model is given and torch is missing."""
    from .grid_densifier import run_interactive_workflow
    return run_interactive_workflow(base_dir=base_dir)


def run_mesa_prepare_flow(
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    model: Optional[str] = None,
    output: Optional[str] = None,
) -> None:
    """
    Export a single extra-axis sub-variant of a spectral grid as a clean MESA-ready folder.

    Delegates to mesa_prepare.run_interactive(), which handles spectral grid selection,
    variant display, and output directory creation.
    """
    from .mesa_prepare import run_interactive
    run_interactive(base_dir=base_dir, model=model, output=output)


def run_combine_flow(
    base_dir: str = STELLAR_DIR_DEFAULT,
    output_name: str = "combined_models",
    interactive: bool = True,
    *,
    models: Optional[List[str]] = None,
) -> None:
    """
    Combine multiple stellar spectral grids into unified omni grid.

    This creates a single flux cube spanning the parameter space of all
    selected spectral grids.

    Args:
        base_dir: Base directory containing stellar_models/ subdirectories
        output_name: Name for the combined spectral grid directory
        interactive: If True, prompt user to select spectral grids
        models: Names of the grids to combine, instead of choosing them.
            Without it and without prompting, every local grid except the
            output of an earlier combine is used.

    A grid is never combined into itself: ValueError when ``output_name`` is
    one of the inputs.
    """
    from .api import _build_combined_grid
    from .combine_stellar_atm import _grids_with_lookup, find_stellar_models, select_models_interactive

    ensure_dir(base_dir)

    # The menu offers no earlier combine outputs; --grids may name one, and
    # --non-interactive says which it leaves out.
    model_dirs = find_stellar_models(base_dir) if interactive and models is None else _grids_with_lookup(base_dir)
    if not model_dirs:
        raise FileNotFoundError(f"No spectral grids found in {base_dir}")

    if models is not None:
        selected_models = _named_grids(models, model_dirs, base_dir)
    elif interactive:
        selected_models = select_models_interactive(model_dirs)
    else:
        selected_models = _grids_to_combine_by_default(model_dirs, output_name)
    if not selected_models:
        print("No spectral grids selected.")
        return

    output_name = _combined_grid_name(output_name, [name for name, _ in selected_models], interactive)
    print("\nSpectral grid will be saved as:", output_name, "\n")

    print(f"\nSelected {len(selected_models)} spectral grids to combine:")
    for name, _ in selected_models:
        print(f"- {name}")

    output_dir = os.path.join(base_dir, output_name)
    _build_combined_grid(selected_models, output_dir)

    print(f"\nSuccessfully combined {len(selected_models)} stellar spectral grids!")
    print(f"Output saved to: {output_dir}")
    print("You can now use this combined spectral grid in MESA by setting:")
    print(f"stellar_atm = '{output_dir}/'")


def _combined_grid_name(output_name: str, inputs: List[str], interactive: bool) -> str:
    """The output grid's name, asked for when interactive; never one of the ``inputs``."""
    if interactive:
        output_name = input(f"What should the combined spectral grid be called? [{output_name}] ").strip() or output_name
        while output_name in inputs:
            output_name = (input(f"{output_name} is one of the grids being combined; choose another name: ").strip()
                           or output_name)
    elif output_name in inputs:
        raise ValueError(f"The output grid {output_name} is also one of the grids to combine; "
                         "choose another --output or leave it out of --grids")
    return output_name


def _named_grids(models: List[str], model_dirs: List[Tuple[str, str]], base_dir: str) -> List[Tuple[str, str]]:
    """The ``(name, path)`` of each grid in ``models``, in that order; ValueError names any not found."""
    found = dict(model_dirs)
    missing = [m for m in models if m not in found]
    if missing:
        raise ValueError(f"No spectral grid with a lookup table named {', '.join(missing)} in {base_dir}; "
                         f"available: {', '.join(sorted(found))}")
    return [(m, found[m]) for m in models]


def _grids_to_combine_by_default(model_dirs: List[Tuple[str, str]], output_name: str) -> List[Tuple[str, str]]:
    """Every grid except earlier combined outputs and ``output_name``; says which were left out."""
    from .combine_stellar_atm import _is_combined_output

    keep = []
    for name, path in model_dirs:
        if name == output_name:
            print(f"Leaving out {name}: it is the output grid")
        elif _is_combined_output(name, os.path.join(path, "lookup_table.csv")):
            print(f"Leaving out {name}: an earlier combined grid (name it with --grids to include it)")
        else:
            keep.append((name, path))
    return keep


def run_ml_completer_flow(
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models_dir: str = "models"
) -> Optional[int]:
    """
    Launch the interactive ML SED Completer workflow.
    
    This is a thin wrapper that delegates to ml_sed_completer.run_interactive_workflow().
    All training and inference logic lives in the ml_sed_completer module.
    Returns 1, saying what to install, when the chosen action's packages are missing.
    """
    from .ml_sed_completer import run_interactive_workflow
    return run_interactive_workflow(base_dir, models_dir)

def run_config_flow() -> None:
    show_config()
    print("\nEnter new data directory path, or press Enter (or q) to keep current:")
    raw = input("> ").strip()
    # q and b leave the setting alone, as in the pickers; a directory with that name is ./q
    if not raw or raw.lower() in ("q", "b"):
        return
    # set_data_dir handles the "move existing data?" prompt itself.
    set_data_dir(raw, interactive=True)
    print("Restart sed-tools for the change to take effect.")


def _discover_local_grids(base_dir: str) -> List[str]:
    """Local spectral grid folders that have a lookup table or at least one .txt."""
    base_dir = str(base_dir)
    out = []
    if not os.path.isdir(base_dir):
        return out
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, "lookup_table.csv")) or any(
            fn.lower().endswith(".txt") for fn in os.listdir(p)
        ):
            out.append(name)
    return out


def run_coverage_flow(base_dir: str = str(STELLAR_DIR_DEFAULT)) -> None:
    """Interactive parameter-space coverage report for a local grid."""
    from .api import SED

    cands = _discover_local_grids(base_dir)
    if not cands:
        print(f"No local grids found under {base_dir}")
        return
    idx = prompt_choice(cands, label="Local grids", allow_back=True)
    if idx is None or idx == -1:
        return
    SED.coverage(cands[idx], model_root=base_dir, plot=True)


def run_import_flow(base_dir: str = str(STELLAR_DIR_DEFAULT)) -> None:
    """Interactive ingest of a local grid into the pipeline."""
    from .api import SED

    path = input("Path to your grid directory (or a .txt file): ").strip()
    if not path:
        print("No path given.")
        return
    name = input("Spectral grid name [blank = source folder name]: ").strip() or None
    mv = input("Move files instead of copying? [y/N] ").strip().lower().startswith("y")
    flux_unit = _ask_import_flux_unit(path, name)
    SED.import_grid(path, name=name, model_root=base_dir, move=mv, flux_unit=flux_unit)


def menu() -> str:
    use_color = terminal_color_enabled("auto")
    BOLD = "\x1b[1m" if use_color else ""
    DIM = "\x1b[2m" if use_color else ""
    CYAN = "\x1b[36m" if use_color else ""
    YELL = "\x1b[33m" if use_color else ""
    RED = "\x1b[31m" if use_color else ""
    RESET = "\x1b[0m" if use_color else ""

    print(BOLD + "=" * 50 + RESET)

    print("\n" + YELL + "-- Filters " + "-" * 39 + RESET)
    print(f"  {CYAN}1){RESET} Download Filters (NJM / SVO)")
    print(f"  {CYAN}2){RESET} Combine filter sets")
    print(f"  {CYAN}3){RESET} Filter set information")
    print(f"  {CYAN}4){RESET} Build a filter curve from its parts (filter, optics, camera, atmosphere)")

    print("\n" + YELL + "-- Spectral grids " + "-" * 32 + RESET)
    print(f"  {CYAN}5){RESET} Download spectral grids (NJM / SVO / MSG / MAST)")
    print(f"  {CYAN}6){RESET} Rebuild (lookup + HDF5 + flux cube)")
    print(f"  {CYAN}7){RESET} Combine grids into omni grid")

    print("\n" + YELL + "-- ML " + "-" * 44 + RESET)
    print(f"  {CYAN}8){RESET} ML SED Completer (train/extend incomplete SEDs)")
    print(f"  {CYAN}9){RESET} ML SED Generator (generate SEDs from parameters)")
    print(f" {CYAN}10){RESET} Grid Densifier (fill coarse Teff gaps)")

    print("\n" + YELL + "-- Tools " + "-" * 41 + RESET)
    print(f" {CYAN}11){RESET} Coverage (parameter-space summary + plot)")
    print(f" {CYAN}12){RESET} Import a local spectral grid")
    print(f" {CYAN}13){RESET} Prepare a spectral grid for MESA")
    print(f" {CYAN}14){RESET} Config (data directory)")

    print("\n" + DIM + "-" * 50 + RESET)
    print(f"  {RED}0/q){RESET} Quit")
    print(BOLD + "=" * 50 + RESET)

    choice = input("> ").strip()
    mapping = {
        "1": "filters",
        "2": "filters_combine",
        "3": "filter_info",
        "4": "filter_compose",
        "5": "grids",
        "6": "rebuild",
        "7": "combine",
        "8": "ml_completer",
        "9": "ml_generator",
        "10": "grid_densifier",
        "11": "coverage",
        "12": "import",
        "13": "mesa_prepare",
        "14": "config",
        "q": "quit",
        "Q": "quit",
        "quit": "quit",
        "QUIT": "quit",
        "Quit": "quit",
        "0": "quit",
    }
    return mapping.get(choice, "")

# ------------ Command-line parser ------------

def _build_parser() -> argparse.ArgumentParser:
    """The ``sed-tools`` parser; subcommands appear in --help in the order added here.

    Each subcommand's parser is its ``command_parser`` default, so its handler
    reports a usage error with that subcommand's usage line.
    """
    parser = argparse.ArgumentParser(description="SED Tools — spectral grids and photometric filters")
    sub = parser.add_subparsers(dest="cmd", required=False)
    _add_grid_download_parser(sub)
    _add_filter_download_parsers(sub)
    _add_filter_catalogue_parsers(sub)
    _add_filter_component_parsers(sub)
    _add_grid_build_parsers(sub)
    _add_grid_tool_parsers(sub)
    for command_parser in sub.choices.values():
        command_parser.set_defaults(command_parser=command_parser)
    return parser


def _add_flux_unit_option(parser) -> None:
    parser.add_argument("--flux-unit", choices=list(FLUX_UNITS),
                        help="Flux unit of spectra whose headers do not declare one (otherwise sed-tools "
                             "asks in a terminal, and stops without one); spectra that declare theirs keep it")


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {text!r}")
    return value


def _add_grid_download_parser(sub) -> None:
    sp = sub.add_parser("grids", aliases=["spectra"], help="Download and build spectral grids")
    sp.add_argument("--source",
                    choices=["njm", "svo", "msg", "mast", "both", "all"],
                    default="all",
                    help="Which provider(s) to use. NJM mirror is checked first when available.")
    sp.add_argument("--grids", "--models", dest="models", nargs="*", default=None,
                    help="Spectral grid names to process, or 'all'")
    sp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Output directory for spectral grids")
    sp.add_argument("--workers", type=int, default=5, help="Parallel workers")
    sp.add_argument("--no-h5", action="store_true", help="Do not force-create HDF5 bundle")
    sp.add_argument("--no-cube", action="store_true", help="Skip flux cube build")
    _add_flux_unit_option(sp)
    cut = ("; NJM, MSG and MAST cut downloads, SVO refuses cuts. Without any cut option an NJM "
           "download asks for cuts in a terminal and takes the full grid otherwise")
    sp.add_argument("--teff", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Download only spectra with Teff in this range (K)" + cut)
    sp.add_argument("--logg", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Download only spectra with log g in this range")
    sp.add_argument("--meta", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Download only spectra with [M/H] in this range")
    sp.add_argument("--wavelength", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Keep only this wavelength range (Å)")


def _add_filter_download_parsers(sub) -> None:
    fp = sub.add_parser("filters", help="Download instrument passbands or browse interactively")
    fp.add_argument("target", nargs="?", help="Facility/Instrument; omit to browse")
    _add_filter_product_options(fp)

    fcp = sub.add_parser("filters-combine", help="Combine filter sets into one MESA-compatible instrument")
    fcp.add_argument("output", help="Output name or Facility/Instrument path for the combined filter set")
    fcp.add_argument("inputs", nargs="+", help="Filter-set directories or .dat files to combine")
    fcp.add_argument("--base", default=str(FILTER_DIR_DEFAULT),
                     help="Base filter directory")
    fcp.add_argument("--facility", default=None, help="Output facility label (default: Combined)")
    fcp.add_argument("--instrument", default=None, help="Output instrument/index-file name")
    fcp.add_argument("--on-conflict", choices=["rename", "overwrite", "error"], default="rename",
                     help="How duplicate filter filenames are handled")

    fdp = sub.add_parser("filters-detector",
                         help="Tag filter files as photon- or energy-counting (DetectorType header)")
    fdp.add_argument("path", nargs="?", default=str(FILTER_DIR_DEFAULT),
                     help="Filter root, instrument directory or single .dat file")
    fdp.add_argument("--facility", default=None,
                     help="SVO facility name when PATH is a single instrument directory or file")
    fdp.add_argument("--force", action="store_true",
                     help="Re-resolve files that are already tagged; a result never replaces a type "
                          "from a stronger source (original > inferred > svo > svo-curve-match > assumed)")
    fdp.add_argument("--dry-run", action="store_true",
                     help="Report without modifying files")


def _add_filter_catalogue_parsers(sub) -> None:
    fip = sub.add_parser("filters-info",
                         help="Show what a filter set contains and which components can be requested")
    fip.add_argument("target", help="Facility/Instrument, e.g. SLOAN/SDSS")
    fip.add_argument("--base", default=str(FILTER_DIR_DEFAULT), help="Base filter directory")

    fip.add_argument("--offline", action="store_true", help="Inspect local data without contacting sources")
    fip.add_argument("--json", action="store_true", help="Emit availability as JSON")


def _add_filter_component_parsers(sub) -> None:
    from .component_sources import AUTOMATIC_BUILDERS, BUILDERS

    fcop = sub.add_parser("filters-compose", help="Compose optical-path components")
    fcop.add_argument("target", help="Facility/Instrument or installed instrument directory")
    _add_filter_product_options(fcop, components="filter")

    finv = sub.add_parser("filters-inventory", help="Inventory all SVO, NJM, local and registered instruments")
    finv.add_argument("--base", default=str(FILTER_DIR_DEFAULT), help="Base filter directory")
    finv.add_argument("--facility", help="Limit discovery to this facility")
    finv.add_argument("--offline", action="store_true", help="List local and registered instruments only")
    finv.add_argument("--json", action="store_true", help="Emit inventory and source errors as JSON")

    fbp = sub.add_parser("filters-build-components",
                         help="(Mirror curation) populate components/atmospheres from a published source")
    fbp.add_argument("source", help=f"Component source: {', '.join(sorted(BUILDERS))}; or 'all' for "
                                    f"those that need no input ({', '.join(AUTOMATIC_BUILDERS)})")
    fbp.add_argument("--base", default=str(FILTER_DIR_DEFAULT), help="Base filter directory")
    fbp.add_argument("--raw", default=None, help="Raw input directory (source specific)")
    fbp.add_argument("--instrument", help="Instrument to build, for a source that holds several "
                                          "(e.g. tabular, surveys, space); a source that needs one says so")
    fbp.add_argument("--sites", nargs="+", default=None, help="SkyCalc or team atmosphere sites")
    fbp.add_argument("--archive", default=None,
                     help="Keep every raw download here with its URL, time and SHA-256 (skycalc, team_atmospheres)")
    fbp.add_argument("--airmass", nargs="+", type=float, default=None, help="SkyCalc airmass grid")
    fbp.add_argument("--pwv", nargs="+", type=float, default=None, help="SkyCalc PWV grid (mm)")


# densify accepts --ml-gap-threshold only at this default, since densify_grid has no such threshold.
_ML_GAP_THRESHOLD_DEFAULT = 5000.0


def _add_grid_build_parsers(sub) -> None:
    rp = sub.add_parser("rebuild", help="Rebuild lookup table + HDF5 + flux cube")
    rp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base spectral grid dir")
    rp.add_argument("--grids", "--models", dest="models", nargs="*", default=None,
                    help="Names of local spectral grids to rebuild")
    rp.add_argument("--no-h5", action="store_true", help="Skip HDF5 bundle")
    rp.add_argument("--no-cube", action="store_true", help="Skip flux cube")
    _add_flux_unit_option(rp)

    cp = sub.add_parser("combine", help="Combine multiple grids into omni grid")
    cp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base spectral grid directory")
    cp.add_argument("--output", default="combined_models",
                    help="Name for the combined spectral grid")
    cp.add_argument("--non-interactive", action="store_true",
                    help="Combine without prompting: the --grids given, else every local spectral grid "
                         "except earlier combined ones")
    cp.add_argument("--grids", "--models", dest="models", nargs="+", default=None,
                    help="Names of the local spectral grids to combine (no prompts)")

    mp = sub.add_parser("ml_completer", help="Train/use ML SED completer")
    mp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base spectral grid directory")
    mp.add_argument("--models-dir", default="models",
                    help="Directory for trained ML models")

    gp = sub.add_parser("ml_generator", help="Train/use ML SED generator")
    gp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base spectral grid directory")
    gp.add_argument("--models-dir", default="models",
                    help="Directory for trained ML models")
    gp.add_argument("--auto", action="store_true",
                    help="Run optimized auto-training with hyperparameter search")
    gp.add_argument("--library", help="Library path for auto-training")
    gp.add_argument("--output", help="Output directory for auto-training")
    gp.add_argument("--trials", type=_positive_int, default=None,
                    help="With --auto: number of hyperparameter search trials (default 20)")
    gp.add_argument("--epochs", type=_positive_int, default=None,
                    help="With --auto: maximum epochs of the final training (default 100)")

    p_densify = sub.add_parser("densify", help="Fill coarse Teff gaps in a flux_cube.bin")
    p_densify.add_argument("--flux-cube", required=True, help="The flux_cube.bin to densify")
    p_densify.add_argument("--output", required=True, help="Where to write the densified flux cube")
    p_densify.add_argument("--teff-spacing", type=float, default=1000.0, help="Target Teff spacing in K")
    p_densify.add_argument("--method", default="auto",
                           choices=["auto", "interp", "ml", "blackbody"],
                           help="auto: fill gaps inside the grid's Teff range with --ml-model if given, and "
                                "elsewhere with a blackbody scaled to the nearest spectrum; ml: the same, "
                                "--ml-model required; blackbody: blackbodies only; interp: not implemented, refused")
    p_densify.add_argument("--ml-model", default=None,
                           help="Trained ML generator for gaps inside the grid's Teff range")
    p_densify.add_argument("--ml-gap-threshold", type=float, default=_ML_GAP_THRESHOLD_DEFAULT,
                           help="Not implemented: refused unless left at its default")
    p_densify.add_argument("--no-lookup", action="store_true",
                           help="No effect: densify writes only the flux cube")


def _add_grid_tool_parsers(sub) -> None:
    pp = sub.add_parser("mesa_prepare",
        help="Export a spectral grid variant as a clean MESA-ready folder")
    pp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base spectral grid directory")
    pp.add_argument("--grid", "--model", dest="model", default=None,
                    help="Spectral grid name (e.g. Kurucz2003all); prompted if omitted")
    pp.add_argument("--output", default=None,
                    help="Output directory for the exported variant; prompted if omitted")

    cfg_p = sub.add_parser("config", help="Show or set the data directory")
    cfg_p.add_argument("--set", metavar="PATH", default=None,
                       help="Set data directory to PATH")
    cfg_p.add_argument("--move", action="store_true",
                       help="With --set: move existing data to the new path")

    covp = sub.add_parser("coverage",
        help="Report parameter-space coverage of a local grid")
    covp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                      help="Base spectral grid dir")
    covp.add_argument("--grids", "--models", dest="models", nargs="*", default=None,
                      help="Spectral grid folder name(s); default: every local grid")
    covp.add_argument("--no-plot", action="store_true",
                      help="Skip the coverage plot")
    covp.add_argument("--out", default=None,
                      help="Output path for the plot PNG (single spectral grid only)")

    imp = sub.add_parser("import",
        help="Ingest a local grid of .txt spectra into the pipeline")
    imp.add_argument("--path", required=True,
                     help="Directory of .txt spectra (or a single .txt file)")
    imp.add_argument("--name", default=None,
                     help="Spectral grid name (default: source folder name)")
    imp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                     help="Base spectral grid dir")
    imp.add_argument("--move", action="store_true",
                     help="Move files instead of copying")
    imp.add_argument("--no-h5", action="store_true", help="Skip HDF5 bundle")
    imp.add_argument("--no-cube", action="store_true", help="Skip flux cube")
    imp.add_argument("--dry-run", action="store_true",
                     help="Report parseable headers and whether the flux unit is declared, without importing")
    _add_flux_unit_option(imp)


# ------------ Command handlers ------------
# Each takes the parsed arguments and its subcommand's parser (for .error)
# and returns the process exit status: 0 done, 1 failed, 2 usage error.

def _range_option(args, name: str, parser) -> Optional[Tuple[float, float]]:
    """A ``--NAME LOW HIGH`` cut as ``(low, high)``, or None when not given."""
    values = getattr(args, name)
    if values is None:
        return None
    low, high = values
    if low > high:
        parser.error(f"--{name} LOW HIGH: {low:g} is above {high:g}")
    return low, high


def _cmd_grids(args, parser):
    run_spectra_flow(
        source=args.source,
        base_dir=args.base,
        models=args.models,
        workers=args.workers,
        force_bundle_h5=not args.no_h5,
        build_flux_cube=not args.no_cube,
        flux_unit=args.flux_unit,
        teff_range=_range_option(args, "teff", parser),
        logg_range=_range_option(args, "logg", parser),
        meta_range=_range_option(args, "meta", parser),
        wl_range=_range_option(args, "wavelength", parser),
    )
    return 0


def _facility_instrument(target: str, parser) -> List[str]:
    parts = target.strip("/").split("/")
    if len(parts) != 2:
        parser.error("target must be Facility/Instrument")
    return parts


def _atmosphere_request(args, parser) -> Optional[Dict]:
    """The ``atmosphere`` argument of Filters.fetch/compose, from --airmass, --pwv and --atmosphere-site."""
    if args.pwv is not None and args.airmass is None:
        parser.error("--pwv needs --airmass")
    if args.atmosphere_site and args.airmass is None:
        parser.error("--atmosphere-site needs --airmass (a number, or 'unstated')")
    if "atmosphere" in re.split(r"[\s,+]+", (args.components or "").lower()) and args.airmass is None:
        parser.error("--components atmosphere needs --airmass (a number, or 'unstated')")
    if args.airmass is None:
        return None
    atmosphere = {"airmass": None if args.airmass == _UNSTATED else args.airmass, "pwv": args.pwv}
    if args.atmosphere_site:
        atmosphere["site"] = args.atmosphere_site
    return atmosphere


def _cmd_filters(args, parser):
    """``filters`` and ``filters-compose``: download (or browse) an instrument, optionally composing it."""
    from .api import Filters

    if not args.target:
        if any((args.components, args.bands, args.detector_type, args.airmass is not None,
                args.pwv is not None, args.atmosphere_site, args.allow_heavy, args.allow_partial)):
            parser.error("component options require a Facility/Instrument target")
        return run_filters_flow(base_dir=args.base)
    options = dict(filter_root=args.base, components=args.components, atmosphere=_atmosphere_request(args, parser),
                   bands=args.bands, detector_type=args.detector_type,
                   allow_heavy=args.allow_heavy, allow_partial=args.allow_partial)
    try:
        if args.cmd == "filters-compose":
            Filters.compose(args.target, **options)
        else:
            print(Filters.fetch(*_facility_instrument(args.target, parser), **options))
    except (ComponentError, RequestException, OSError, RuntimeError) as exc:
        print(f"[filters] {_atmosphere_error(exc, args) if options['atmosphere'] else exc}")
        return 1
    return 0


def _atmosphere_error(exc: Exception, args) -> str:
    """The error of a failed composition with an atmosphere, in command-line terms where they help.

    An instrument with several sites and no default needs --atmosphere-site;
    any other error is followed by the conditions its site has.
    """
    from .api import Filters
    from .filter_library import FilterLibrary

    directory = FilterLibrary(args.base).directory(args.target)
    try:
        info = Filters.available(directory.parent.name, directory.name, filter_root=directory.parent.parent,
                                 include_remote=False)
        grids = Filters.atmospheres(filter_root=directory.parent.parent, include_remote=False)
    except (ComponentError, OSError, RuntimeError) as lookup_error:
        return f"{exc} (its atmosphere sites could not be listed: {lookup_error})"
    sites = _atmosphere_sites(info)
    default = _default_site(info, sites)
    if not args.atmosphere_site and len(sites) > 1 and default is None:
        return (f"{directory.parent.name}/{directory.name} has several atmosphere sites "
                f"({', '.join(sites)}) and no default; choose one with --atmosphere-site")
    site = args.atmosphere_site or default
    if site in grids:
        return f"{exc}\n[filters] {site} has: {_conditions_text(grids[site])}"
    return str(exc)


def _cmd_filters_combine(args, parser):
    from .combine_filters import combine_filter_sets

    out = combine_filter_sets(
        args.output,
        args.inputs,
        filter_root=args.base,
        facility=args.facility,
        instrument=args.instrument,
        on_conflict=args.on_conflict,
    )
    print(f"[filters] Combined filter set written to {out}")
    return 0


def _cmd_filters_detector(args, parser):
    from collections import Counter

    from .api import Filters

    # Each file is printed as it is tagged, so on failure only the error remains to be said.
    try:
        rows = Filters.tag_detector_types(
            args.path, facility=args.facility, force=args.force, dry_run=args.dry_run
        )
    except (RequestException, RuntimeError, ValueError) as exc:  # SVOUnreachableError is a RequestException
        print(f"\n[filters] {exc}")
        return 1
    counts = Counter((r["detector_type"], r["source"]) for r in rows)
    changed = sum(r["changed"] for r in rows)
    verb = "would change" if args.dry_run else "changed"
    migrated = sum(r["migrated"] for r in rows)
    moved = "would move" if args.dry_run else "moved"
    print(f"\n[filters] {len(rows)} files, {changed} {verb}, {migrated} {moved} 0.3.0 '#' metadata to the column row")
    for (dtype, source), n in sorted(counts.items()):
        print(f"  {dtype:6s} [{source}]: {n}")
    kept = [r for r in rows if r["kept"]]
    if kept:
        differ = sum(r["resolved_type"] != r["detector_type"] for r in kept)
        print(f"[filters] {len(kept)} kept their DetectorType: the new resolution does not outrank "
              f"their DetectorTypeSource and was not written ({differ} of them resolve to the other type)")
        for r in kept:
            print(f"  {r['path']}: {r['note']}")
    return 0


def _cmd_filters_info(args, parser):
    from .api import Filters

    target = _facility_instrument(args.target, parser)
    info = Filters.available(*target, filter_root=args.base, include_remote=not args.offline)
    catalogue = _catalogue_entry(*target, args.base, offline=args.offline)
    if args.json:
        print(json.dumps(dict(info, catalogue=catalogue), indent=2))
    else:
        _print_filter_set("/".join(target), info, catalogue)
    # NJM's catalogue (description, licence, band properties) is an extra: without it the
    # availability report still stands, so only an incomplete availability report fails.
    return 1 if _availability_errors(info) else 0


def _availability_errors(info: dict) -> List[str]:
    """The failures that leave an availability report incomplete.

    A source in ``fallback_from`` (NJM, say, when SVO answered instead) leaves nothing out.
    """
    fallback = set(info.get("fallback_from") or ())
    return [message for source, messages in (info.get("sources_failed") or {}).items()
            if source not in fallback for message in messages]


def _cmd_filters_inventory(args, parser):
    from .api import Filters

    result = Filters.inventory(filter_root=args.base, include_remote=not args.offline, facility=args.facility)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for row in result["instruments"]:
            status = row["component_availability"]["status"]
            print(f"{row['facility']}/{row['instrument']}: {status} [{', '.join(row['sources'])}]")
        for error in result["errors"]:
            print(f"[filters] {error}")
    return 1 if result["errors"] else 0


def _cmd_filters_build_components(args, parser):
    from .component_sources import BUILDERS, get_builder

    if args.source == "all":
        return _build_all_component_sources(args.base, args.sites)
    if args.source.lower() not in BUILDERS:
        parser.error(f"unknown source {args.source!r}; choose one of: {', '.join(sorted(BUILDERS))}, or all")
    builder = get_builder(args.source)
    needs = inspect.signature(builder).parameters.get("instrument")
    if not args.instrument and needs is not None and needs.default is inspect.Parameter.empty:
        parser.error(f"{args.source} builds several instruments; choose one with --instrument")
    if args.source.lower() == "tabular":
        from .component_sources.tabular import SOURCES

        if args.instrument not in SOURCES:
            parser.error(f"tabular has no instrument {args.instrument!r}; choose one of: {', '.join(SOURCES)}")
    kwargs = {}
    if args.raw:
        kwargs["raw"] = args.raw
    if args.instrument:
        kwargs["instrument"] = args.instrument
    if args.sites:
        kwargs["sites"] = args.sites
    if args.airmass:
        kwargs["airmasses"] = args.airmass
    if args.pwv:
        kwargs["pwvs"] = args.pwv
    if args.archive:
        kwargs["archive_dir"] = args.archive
    print(f"[filters] Built {builder(args.base, **kwargs)}")
    return 0


def _build_all_component_sources(base: str, sites: Optional[List[str]]) -> int:
    """Run every automatic builder; report each failure and return 1 if any failed."""
    from .component_sources import AUTOMATIC_BUILDERS, get_builder

    failed = []
    for name in AUTOMATIC_BUILDERS:
        kw = {"sites": sites or ["paranal", "lasilla"]} if name == "skycalc" else {}
        print(f"[filters] === {name} ===")
        try:
            get_builder(name)(base, **kw)
        except (ValueError, RequestException, OSError, RuntimeError) as exc:
            failed.append(f"{name}: {exc}")
    for line in failed:
        print(f"[filters] FAILED {line}")
    return 1 if failed else 0


def _cmd_rebuild(args, parser):
    run_rebuild_flow(
        base_dir=args.base,
        models=args.models,
        rebuild_h5=not args.no_h5,
        rebuild_flux_cube=not args.no_cube,
        flux_unit=args.flux_unit,
    )
    return 0


def _cmd_combine(args, parser):
    run_combine_flow(
        base_dir=args.base,
        output_name=args.output,
        interactive=not (args.non_interactive or args.models),
        models=args.models,
    )
    return 0


def _cmd_ml_completer(args, parser):
    return run_ml_completer_flow(
        base_dir=args.base,
        models_dir=args.models_dir
    ) or 0


def _cmd_ml_generator(args, parser):
    limits = {key: value for key, value in (("n_trials", args.trials), ("epochs", args.epochs)) if value}
    if not args.auto:
        if limits:
            parser.error("--trials and --epochs apply to --auto training only")
        return run_ml_generator_flow(
            base_dir=args.base,
            models_dir=args.models_dir
        ) or 0
    if not args.library or not args.output:
        parser.error("--auto needs --library and --output")
    if importlib.util.find_spec("optuna") is None:
        print("Error: --auto training needs the optuna package, which is not installed (pip install 'sed-tools[ml-tune]').")
        return 1
    missing = missing_packages(("torch", "sklearn"))
    if missing:
        print(missing, file=sys.stderr)
        return 1
    from .ml import auto_train_generator
    auto_train_generator(library=args.library, output=args.output, **limits)
    return 0


def _cmd_densify(args, parser):
    """densify_grid fills each Teff gap with --ml-model inside the source grid's Teff range, and
    with a blackbody scaled to the nearest real spectrum otherwise; options it cannot honour are refused."""
    from .grid_densifier import densify_grid

    if args.method == "interp":
        parser.error("--method interp is not implemented: densify fills gaps with --ml-model or blackbodies")
    if args.method == "ml" and not args.ml_model:
        parser.error("--method ml needs --ml-model")
    if args.method == "blackbody" and args.ml_model:
        parser.error("--method blackbody does not use --ml-model; leave one of them out")
    if args.ml_gap_threshold != _ML_GAP_THRESHOLD_DEFAULT:
        parser.error("--ml-gap-threshold is not implemented: with --ml-model, every gap inside the source "
                     "grid's Teff range is filled by the model")
    # --no-lookup needs nothing: densify_grid writes only the flux cube.
    densify_grid(
        src=args.flux_cube,
        dst=args.output,
        teff_spacing=args.teff_spacing,
        ml_model=args.ml_model,
    )
    return 0


def _cmd_mesa_prepare(args, parser):
    run_mesa_prepare_flow(
        base_dir=args.base,
        model=args.model,
        output=args.output,
    )
    return 0


def _cmd_config(args, parser):
    if args.set:
        set_data_dir(args.set, move=args.move if args.move else None)
    else:
        show_config()
    return 0


def _cmd_coverage(args, parser):
    from .api import SED

    names = args.models
    if not names:
        names = _discover_local_grids(args.base)
        if not names:
            print(f"No local grids found under {args.base}")
            return 1
    failed = []
    for nm in names:
        try:
            SED.coverage(
                nm,
                model_root=args.base,
                plot=not args.no_plot,
                out_path=args.out if len(names) == 1 else None,
            )
        except (ValueError, OSError, RuntimeError) as exc:
            failed.append(f"{nm}: {exc}")
    for line in failed:
        print(f"[coverage] FAILED {line}")
    return 1 if failed else 0


def _cmd_import(args, parser):
    from .api import SED

    if args.dry_run:
        flux_unit = None
    else:
        flux_unit = _ask_import_flux_unit(args.path, args.name, args.flux_unit)
    SED.import_grid(
        args.path,
        name=args.name,
        model_root=args.base,
        move=args.move,
        build_h5=not args.no_h5,
        build_cube=not args.no_cube,
        dry_run=args.dry_run,
        flux_unit=flux_unit,
    )
    if args.dry_run:
        _report_import_flux_unit(args.path, args.name, args.flux_unit)
    return 0


_COMMANDS = {
    "grids": _cmd_grids,
    "spectra": _cmd_grids,
    "filters": _cmd_filters,
    "filters-compose": _cmd_filters,
    "filters-combine": _cmd_filters_combine,
    "filters-detector": _cmd_filters_detector,
    "filters-info": _cmd_filters_info,
    "filters-inventory": _cmd_filters_inventory,
    "filters-build-components": _cmd_filters_build_components,
    "rebuild": _cmd_rebuild,
    "combine": _cmd_combine,
    "ml_completer": _cmd_ml_completer,
    "ml_generator": _cmd_ml_generator,
    "densify": _cmd_densify,
    "mesa_prepare": _cmd_mesa_prepare,
    "config": _cmd_config,
    "coverage": _cmd_coverage,
    "import": _cmd_import,
}


# ------------ Interactive mode ------------

def _print_banner() -> None:
    use_color = terminal_color_enabled("auto")
    BOLD = "\x1b[1m" if use_color else ""
    RED = "\x1b[31m" if use_color else ""
    BLUE = "\x1b[34m" if use_color else ""
    GREEN = "\x1b[32m" if use_color else ""
    RESET = "\x1b[0m" if use_color else ""

    banner = (
            "▄▖▄▖▄   ▄▖    ▜   ",
            "▚ ▙▖▌▌  ▐ ▛▌▛▌▐ ▛▘",
            "▄▌▙▖▙▘▄▖▐ ▙▌▙▌▐▖▄▌",
                                  )

    print()
    print("\n" + BOLD + "=" * 50 + RESET)

    banner_colors = [RED, GREEN, BLUE]
    for i, line in enumerate(banner):
        print(f"{BOLD}{banner_colors[i]}{line}{RESET}")


def _run_menu() -> int:
    """Show the menu and run the chosen workflow until the user quits.

    A workflow that fails or is interrupted says why, and the menu is shown
    again.  Returns 1 if any workflow failed during the session, else 0.
    """
    _print_banner()
    flows = {
        "grids": lambda: run_spectra_flow(source="all"),
        "filters": run_filters_flow,
        "filters_combine": run_filter_combine_flow,
        "filter_info": run_filter_info_flow,
        "filter_compose": run_filter_compose_flow,
        "rebuild": run_rebuild_flow,
        "combine": run_combine_flow,
        "ml_completer": run_ml_completer_flow,
        "ml_generator": run_ml_generator_flow,
        "grid_densifier": run_grid_densifier_flow,
        "mesa_prepare": run_mesa_prepare_flow,
        "config": run_config_flow,
        "coverage": run_coverage_flow,
        "import": run_import_flow,
    }
    failed = False
    while True:
        try:
            choice = menu()
        except (EOFError, KeyboardInterrupt):  # input closed, or Ctrl-C at the menu: quit
            print()
            return int(failed)
        if choice == "quit":
            return int(failed)
        if choice not in flows:
            print("Unknown option: enter a number from the menu, or q to quit.")
            continue
        if not _run_menu_flow(flows[choice]):
            failed = True


def _run_menu_flow(flow) -> bool:
    """Run one menu workflow; say why if it fails, and return whether it succeeded."""
    try:
        status = flow()
    except EXPECTED_ERRORS as exc:
        _print_error(exc)
        return False
    except ImportError as exc:  # an optional package a step needs (scipy, torch, ...): say what to install
        missing = _missing_package_error(exc)
        if missing is None:
            raise
        print(missing, file=sys.stderr)
        return False
    except KeyboardInterrupt:
        print("\nInterrupted; back to the menu.")
        return True
    except SystemExit as exc:  # mesa_prepare stops with SystemExit(message)
        if exc.code in (None, 0):
            return True
        print(exc.code if isinstance(exc.code, str) else f"Stopped with status {exc.code}.", file=sys.stderr)
        return False
    return status in (None, 0)


def _print_error(exc: BaseException) -> None:
    """``sed-tools: error: <message>`` on stderr; with SED_TOOLS_DEBUG=1 the traceback first."""
    if os.environ.get("SED_TOOLS_DEBUG") == "1":
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    if isinstance(exc, EOFError):
        message = ("input ended before a question was answered; run it in a terminal, "
                   "or see the command's --help for options that give the answers")
    else:
        message = str(exc) or type(exc).__name__
    print(f"sed-tools: error: {message}", file=sys.stderr)


# ------------ Main CLI ------------

def main():
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.cmd is None:
            return _run_menu()
        return _COMMANDS[args.cmd](args, args.command_parser)
    except EXPECTED_ERRORS as exc:
        _print_error(exc)
        return 1
    except ImportError as exc:  # an optional package a step needs (scipy, torch, ...): say what to install
        missing = _missing_package_error(exc)
        if missing is None:
            raise
        print(missing, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nsed-tools: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
