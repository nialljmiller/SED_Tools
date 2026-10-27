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
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from requests import RequestException

from .config import set_data_dir, show_config
from .mast_spectra_grabber import MASTSpectraGrabber
from .models import FILTER_DIR_DEFAULT, STELLAR_DIR_DEFAULT
from .msg_spectra_grabber import MSGSpectraGrabber
from .njm_spectra_grabber import NJMSpectraGrabber
from .spectra_cleaner import clean_model_dir
from .svo_regen_spectra_lookup import regenerate_lookup_table
from .svo_spectra_grabber import SVOSpectraGrabber
from .ui_utils import prompt_choice  # public, no leading underscore
from .parsing import parse_multi_selection, parse_numeric_range
from .spectrum_io import build_h5_bundle, read_text_spectrum
from .terminal_plots import terminal_color_enabled

logger = logging.getLogger(__name__)
# ------------ Small Utils ------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    rebuild_flux_cube: bool = True
) -> None:
    """Clean each chosen local grid and rebuild its lookup table, HDF5 bundle and flux cube.

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
                                     rebuild_h5, rebuild_flux_cube)
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


def _rebuild_grid(model_dir: str, model_name: str, rebuild_h5: bool, rebuild_flux_cube: bool) -> List[str]:
    """Clean one local grid and rebuild its products; returns what was not done (empty when complete)."""
    from .api import _build_flux_cube, _unusable_after_cleaning

    summary = clean_model_dir(model_dir, try_h5_recovery=True, rebuild_lookup=True)
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
) -> None:
    """Build HDF5 bundle, lookup table, and flux cube for a downloaded spectral grid."""
    # HDF5 bundle
    out_h5 = os.path.join(model_dir, f"{name}_bundle.h5" if src == "msg" else f"{name}.h5")
    if force_bundle_h5 or not os.path.exists(out_h5):
        build_h5_bundle_from_txt(model_dir, out_h5)
        print(f"HDF5 bundle   : {os.path.basename(out_h5)}")
    else:
        print(f"HDF5 bundle   : {os.path.basename(out_h5)} (exists)")

    # Lookup table
    regenerate_lookup_table(model_dir)
    print(f"Lookup table  : lookup_table.csv")

    # Flux cube
    if not build_flux_cube:
        return
    from .api import _build_flux_cube

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


def run_spectra_flow(
    source: str,
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models: Optional[List[str]] = None,
    workers: int = 5,
    force_bundle_h5: bool = True,
    build_flux_cube: bool = True,
) -> None:
    """Discover spectral grids, then download, clean and build each chosen one.

    A grid that cannot be completed is reported and the rest are processed;
    then RuntimeError lists every grid that was not completed.
    """
    ensure_dir(base_dir)

    src_list = _spectra_sources(source)
    grabs = _spectra_grabbers(src_list, base_dir, workers)
    model_sources, discovery_failures = _discover_spectral_grids(src_list, grabs)
    if not model_sources:
        raise RuntimeError("No spectral grids discovered" + "".join(f"; {f}" for f in discovery_failures))

    all_models = [_ModelOption(n, srcs) for n, srcs in sorted(model_sources.items())]
    if models is not None:
        chosen, failures = _requested_spectral_grids(models, all_models)
    else:
        idxs = prompt_choice(all_models, label="Spectral grids", allow_back=True, multi=True)
        if idxs is None or idxs == -1:
            return
        chosen, failures = [(all_models[i].sources, all_models[i].name) for i in idxs], []

    for sources, name in chosen:
        try:
            problems = _download_and_build_grid(name, sources, grabs, base_dir, force_bundle_h5, build_flux_cube)
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
            print("[njm] Mirror unavailable, using other sources")
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


def _download_from_first_source(name: str, sources: List[str], grabs: Dict, model_dir: str):
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
        meta, error = _download_from(name, src, grabs[src], model_dir)
        if error is None:
            return src, meta
        errors.append(f"{src}: {error}")
        if _directory_state(model_dir) != before:
            raise RuntimeError(f"{src} failed after changing {model_dir}, so another source's spectra would be "
                               f"mixed with its files; no further source tried. Tried: {'; '.join(errors)}")
        logger.warning("  [%s] %s — trying next source...", src, error)
    raise RuntimeError(f"no source could supply it. Tried: {'; '.join(errors)}")


def _download_from(name: str, src: str, g, model_dir: str):
    """Download ``name`` from one source: ``(metadata, None)``, or ``(metadata, what went wrong)``."""
    meta = None
    try:
        meta = g.get_model_metadata(name)
        if not meta:
            return meta, "no metadata"
        print(f"  Using source: {src}")
        njm_cuts = prompt_njm_axis_cuts(name, g, model_url=meta.get("model_url")) if src == "njm" else {}
        n_written = g.download_model_spectra(
            name, meta,
            teff_range=njm_cuts.get('teff_range'),
            logg_range=njm_cuts.get('logg_range'),
            meta_range=njm_cuts.get('meta_range'),
            wl_range=njm_cuts.get('wl_range'),
        )
    except (ValueError, RequestException, OSError, RuntimeError) as exc:
        return meta, str(exc)
    print(f"Downloaded {n_written} spectra → {model_dir}")
    if not n_written and not _is_preprocessed(meta):
        return meta, "no spectra downloaded"
    return meta, None


def _is_preprocessed(meta) -> bool:
    return isinstance(meta, dict) and meta.get("pre_processed", False)


def _download_and_build_grid(name: str, sources: List[str], grabs: Dict, base_dir: str,
                             force_bundle_h5: bool, build_flux_cube: bool) -> List[str]:
    """Download one grid from its first source that can supply it, clean it and build its products.

    Returns what was not done; empty when the grid is complete.
    """
    from .api import _unusable_after_cleaning

    print("\n" + "=" * 64)
    print(f"{''.join(f'[{s}]' for s in sources)} {name}")
    print("=" * 64)

    model_dir = os.path.join(base_dir, name)
    ensure_dir(model_dir)

    src, meta = _download_from_first_source(name, sources, grabs, model_dir)

    if _is_preprocessed(meta):
        print("Pre-processed data (skipping cleaning)")
        essential = ["flux_cube.bin", "lookup_table.csv"]
        missing = [f for f in essential if not os.path.exists(os.path.join(model_dir, f))]
        print(f"Missing: {', '.join(missing)}" if missing else "All essential files present")
        return [f"missing {', '.join(missing)}"] if missing else []

    summary = clean_model_dir(model_dir, try_h5_recovery=True, rebuild_lookup=True)
    _print_cleaning_report(summary)

    if not glob.glob(os.path.join(model_dir, "*.txt")):
        print("\n  No spectra remaining after cleaning")
        return ["no spectra remaining after cleaning"]

    _build_data_products(name, model_dir, src, force_bundle_h5, build_flux_cube)
    unusable = _unusable_after_cleaning(summary)
    if unusable:
        return [f"{len(unusable)} spectra unusable after cleaning: {', '.join(unusable)}"]
    return []


def print_filter_availability(info: dict) -> None:
    """Print what a filter set contains and what can be composed from it."""
    print(f"\n{info['facility']}/{info['instrument']}"
          + (f"  ({len(info['bands'])} bands)" if info["bands"] else ""))
    src = f" [{info['published_source']}]" if info["published_source"] else ""
    print(f"  Published curves contain : {info['published'] or 'unknown'}{src}")
    if info["published"] == "mixed":
        for band, recipe in info["published_bands"].items():
            print(f"    {band}: {recipe}")
    comps = info["components"]
    component_summary = (', '.join(comps) if comps else
                         'available for selected bands' if info['partial_components'] else 'none identified')
    print(f"  Separate components      : {component_summary}")
    if info["buildable"]:
        print(f"  Can be built             : {', '.join(info['buildable'])}")
    if info["component_status"]:
        print(f"  Note                     : {info['component_status']}")
    for role, bands in info["partial_components"].items():
        print(f"    {role} only for: {', '.join(bands)}")
    review = info["component_availability"]
    print(f"  Source review            : {review['status']}"
          + (f" ({review['builder']})" if review["builder"] else ""))
    if info["missing_files"]:
        print(f"  Missing files            : {', '.join(info['missing_files'])}")
    for error in info["errors"]:
        print(f"  Source unavailable       : {error}")
    if info["atmosphere"]:
        print(f"  Atmosphere               : can be added (own site: {info['atmosphere_site']})")
    elif info["platform"] in ("space", "airborne"):
        print(f"  Atmosphere               : not applicable ({info['platform']}-based)")
    else:
        print("  Atmosphere               : not offered (no site of its own assigned)")
    if info["detector_type"]:
        print(f"  Detector                 : {info['detector_type']}-counting")
    assumed = [band for band, source in info["detector_type_sources"].items()
               if source in ("assumed", "unknown")]
    if assumed:
        print(f"  Detector unverified for  : {', '.join(assumed)}")


SEARCH_COLUMNS = ["facility", "instrument", "band", "WavelengthPivot", "FWHM",
                  "WavelengthMin", "WavelengthMax", "DetectorType"]


def _catalogue_entry(facility: str, instrument: str, base: str, *, offline: bool) -> dict:
    """NJM's info.json and properties.json for ``filters-info``; errors are reported, not raised."""
    from .api import Filters
    from .filter_catalogue import CatalogueError

    entry = {"info": None, "properties": None, "errors": []}
    for kind in ("info", "properties"):
        try:
            entry[kind] = getattr(Filters, kind)(facility, instrument, filter_root=base, offline=offline)
        except (CatalogueError, OSError, ValueError) as exc:
            if str(exc) not in entry["errors"]:
                entry["errors"].append(str(exc))
    return entry


def print_catalogue_entry(target: str, entry: dict) -> None:
    """Print the NJM description, licence and per-band properties of an instrument."""
    import textwrap

    info, props = entry["info"] or {}, (entry["properties"] or {}).get("bands", {})
    width, indent = 100, " " * 29

    def field(name, text):
        print(textwrap.fill(" ".join(str(text).split()), width, initial_indent=f"  {name:<25}: ",
                            subsequent_indent=indent))

    if info.get("description"):
        field("Description", info["description"])
    licence = info.get("licence") or {}
    if licence:
        field("Licence", f"{licence.get('redistribution', 'unknown')}"
              + (f". {licence['terms']}" if licence.get("terms") else "")
              + (f" {licence['url']}" if licence.get("url") else ""))
    if info.get("references"):
        field("Cite", f"{len(info['references'])} references; sed-tools filters-cite {target}")
    bands = sorted(set(props) | set(info.get("bands", {})),
                   key=lambda b: (props.get(b, {}).get("WavelengthPivot") or float("inf"), b))
    if bands:
        print(f"  {'Band':<16}{'Pivot (Å)':>11}{'FWHM (Å)':>11}  {'Detector':<9}{'MagSys':<8}Description")
        for band in bands:
            p, i = props.get(band, {}), info.get("bands", {}).get(band, {})
            num = lambda v: f"{v:>11.1f}" if isinstance(v, (int, float)) else f"{'-':>11}"
            print(f"  {band:<16}{num(p.get('WavelengthPivot'))}{num(p.get('FWHM'))}  "
                  f"{p.get('DetectorType', '-'):<9}{i.get('MagSys') or '-':<8}{i.get('Description') or ''}")
    for error in entry["errors"]:
        print(f"  NJM catalogue            : {error}")


def _bound(text: str) -> Optional[float]:
    """A range end from the command line: a number, or '-'/'' for an open end."""
    return None if text.strip().lower() in ("", "-", "none", "inf", "-inf") else float(text)


def _where(items: Optional[List[str]]) -> Dict[str, object]:
    """``KEY=LOW:HIGH`` (either end may be empty) or ``KEY=VALUE`` criteria for filters-search."""
    criteria: Dict[str, object] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--where expects KEY=VALUE or KEY=LOW:HIGH, got {item!r}")
        if ":" in value:
            lo, _, hi = value.partition(":")
            criteria[key] = (_bound(lo), _bound(hi))
        else:
            try:
                criteria[key] = float(value)
            except ValueError:
                criteria[key] = value
    return criteria


_UNSTATED = "unstated"


def _airmass_arg(text):
    """An airmass, or 'unstated' for an atmosphere published without one."""
    return _UNSTATED if str(text).strip().lower() == _UNSTATED else float(text)


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


def _offer_composition(instrument_dir: Path) -> None:
    """After a download, offer to build a composed product; the build's errors propagate."""
    from .api import Filters

    info = Filters.available(instrument_dir.parent.name, instrument_dir.name,
                             filter_root=instrument_dir.parent.parent, include_remote=False)
    print_filter_availability(info)
    options = [recipe.replace("+", ",") for recipe in info["buildable"]]
    options.append("published")
    can_atm = info["atmosphere"]
    review = info["component_availability"]
    if not info["buildable"] and not can_atm and review["status"] != "supported":
        return
    if not input("Build a composed version (e.g. filter only, or with atmosphere)? [y/N] ").strip().lower().startswith("y"):
        return
    print("Variants of this instrument's own throughput (or enter a custom recipe):")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    pick = input("Choose a variant [1]: ").strip() or "1"
    chosen = options[int(pick) - 1] if pick.isdigit() and 0 < int(pick) <= len(options) else pick
    detector = input("Detector convention [inherit], energy or photon: ").strip().lower() or None
    allow_heavy = bool(review["large_download"] and
                       input("Allow a large original-source download if needed? [y/N] ").strip().lower().startswith("y"))
    atmosphere = None
    sites = info.get("atmosphere_sites") or ([info["atmosphere_site"]] if info["atmosphere_site"] else [])
    if can_atm and sites and input(
            f"Fold in the atmosphere ({', '.join(sites)})? [y/N] ").strip().lower().startswith("y"):
        site = info["atmosphere_site"] or sites[0]
        if len(sites) > 1:
            site = input(f"Site [{site}]: ").strip() or site
        try:
            grid = Filters.atmospheres(filter_root=instrument_dir.parent.parent).get(site)
        except (RequestException, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"could not list the atmosphere grids for {site}: {exc}") from exc
        if grid:
            print(f"  airmass {grid['airmass']}" + (f", pwv {grid['pwv']} mm" if grid["pwv"] else "")
                  + (" or 'unstated' (as published)" if grid.get("unstated_airmass") else ""))
        try:
            airmass = _airmass_arg(input("Airmass [1.2]: ").strip() or 1.2)
            pwv_txt = input("PWV mm (blank if not applicable): ").strip() if not grid or grid["pwv"] else ""
            atmosphere = {"site": site, "airmass": None if airmass == _UNSTATED else airmass,
                          "pwv": float(pwv_txt) if pwv_txt else None}
        except ValueError as exc:
            raise ValueError(f"Airmass and PWV must be numbers ({exc})") from exc
    Filters.compose(instrument_dir, [c.strip() for c in chosen.split(",") if c.strip()],
                    atmosphere=atmosphere, filter_root=instrument_dir.parent.parent,
                    detector_type=detector, allow_heavy=allow_heavy)


def run_filters_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> None:
    """Browse available instruments; use the API for every download and product.

    A download or build that fails is reported at once; when browsing ends,
    RuntimeError lists every one that failed.
    """
    from .filter_library import FilterLibrary

    library = FilterLibrary(base_dir)
    catalogue = library.facilities()
    for error in catalogue["errors"]:
        print(f"[filters] {error}")
    facilities = catalogue["entries"]
    if not facilities:
        print("No facilities found.")
        return

    failures: List[str] = []
    while True:
        selected = prompt_choice([f["label"] for f in facilities], "Filter Facilities", multi=True)
        if selected is None:
            break
        if len(selected) > 1:
            _download_every_instrument(library, [facilities[i]["key"] for i in selected], base_dir, failures)
            break
        if not _browse_instruments(library, facilities[selected[0]]["key"], base_dir, failures):
            break
    if failures:
        raise RuntimeError(f"Filter downloads or builds failed ({len(failures)}): " + "; ".join(failures))


def _download_every_instrument(library, facilities: List[str], base_dir: str, failures: List[str]) -> None:
    """Download every instrument of each facility; an incomplete instrument list is a failure too."""
    for facility in facilities:
        result = library.instruments(facility)
        for error in result["errors"]:
            print(f"[filters] {error}")
            failures.append(f"{facility}: instrument list incomplete: {error}")
        for instrument in result["entries"]:
            _fetch_instrument(facility, instrument["key"], base_dir, failures)


def _browse_instruments(library, facility: str, base_dir: str, failures: List[str]) -> bool:
    """Pick instruments of one facility to download; returns False if the user quit."""
    result = library.instruments(facility)
    for error in result["errors"]:
        print(f"[filters] {error}")
    instruments = result["entries"]
    if not instruments:
        print(f"No instruments found for {facility}.")
        return True
    while True:
        index = prompt_choice([i["label"] for i in instruments], f"Instruments for {facility}", allow_back=True)
        if index is None:
            return False
        if index == -1:
            return True
        instrument = instruments[index]["key"]
        if not input(f"Download {facility}/{instrument}? [Y/n] ").strip().lower().startswith("n"):
            _fetch_instrument(facility, instrument, base_dir, failures, offer=True)
        if not input("Download another instrument? [y/N] ").strip().lower().startswith("y"):
            return True


def _fetch_instrument(facility: str, instrument: str, base_dir: str, failures: List[str], *,
                      offer: bool = False) -> None:
    """Download one instrument (then offer a composed product); a failure is printed and recorded."""
    from .api import Filters

    try:
        directory = Filters.fetch(facility, instrument, filter_root=base_dir)
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
        self.label = f"{rel} ({len(list(path.glob('*.dat')))} filters)"


def run_filter_combine_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> None:
    """Interactive wizard for combining local filter sets."""
    from .combine_filters import combine_filter_sets, find_filter_sets

    print("\nCombine photometric filter sets")
    print("This creates one MESA-compatible filter folder and index file from existing .dat files.")

    chosen_base = input(f"Filter base directory [{base_dir}]: ").strip() or base_dir
    filter_sets = find_filter_sets(chosen_base)
    if not filter_sets:
        print(f"No local filter sets containing .dat files were found under {chosen_base}.")
        print("Download filters first with option 2, or run `sed-tools filters`.")
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
) -> None:
    """Run ML SED Generator interactive workflow."""
    from .ml_sed_generator import run_interactive_workflow
    run_interactive_workflow(base_dir=base_dir, models_dir=models_dir)


def run_grid_densifier_flow(base_dir: str = STELLAR_DIR_DEFAULT) -> None:
    """Run Grid Densifier interactive workflow."""
    from .grid_densifier import run_interactive_workflow
    run_interactive_workflow(base_dir=base_dir)


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
    interactive: bool = True
) -> None:
    """
    Combine multiple stellar spectral grids into unified omni grid.

    This creates a single flux cube spanning the parameter space of all
    selected spectral grids.

    Args:
        base_dir: Base directory containing stellar_models/ subdirectories
        output_name: Name for the combined spectral grid directory
        interactive: If True, prompt user to select spectral grids
    """
    from .api import _build_combined_grid
    from .combine_stellar_atm import find_stellar_models, select_models_interactive

    ensure_dir(base_dir)

    model_dirs = find_stellar_models(base_dir)
    if not model_dirs:
        raise FileNotFoundError(f"No spectral grids found in {base_dir}")

    selected_models = select_models_interactive(model_dirs) if interactive else model_dirs
    if not selected_models:
        print("No spectral grids selected.")
        return

    if interactive:
        output_name = input(f"What should the combined spectral grid be called? [{output_name}]") or output_name

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


def run_ml_completer_flow(
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models_dir: str = "models"
) -> None:
    """
    Launch the interactive ML SED Completer workflow.
    
    This is a thin wrapper that delegates to ml_sed_completer.run_interactive_workflow().
    All training and inference logic lives in the ml_sed_completer module.
    """
    from .ml_sed_completer import run_interactive_workflow
    run_interactive_workflow(base_dir, models_dir)

def run_config_flow() -> None:
    show_config()
    print("\nEnter new data directory path, or press Enter to keep current:")
    raw = input("> ").strip()
    if not raw:
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
    SED.import_grid(path, name=name, model_root=base_dir, move=mv)


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

    print("\n" + YELL + "-- Spectral grids " + "-" * 32 + RESET)
    print(f"  {CYAN}3){RESET} Download spectral grids (NJM / SVO / MSG / MAST)")
    print(f"  {CYAN}4){RESET} Rebuild (lookup + HDF5 + flux cube)")
    print(f"  {CYAN}5){RESET} Combine grids into omni grid")

    print("\n" + YELL + "-- ML " + "-" * 44 + RESET)
    print(f"  {CYAN}6){RESET} ML SED Completer (train/extend incomplete SEDs)")
    print(f"  {CYAN}7){RESET} ML SED Generator (generate SEDs from parameters)")
    print(f"  {CYAN}8){RESET} Grid Densifier (fill coarse Teff gaps)")

    print("\n" + YELL + "-- Tools " + "-" * 41 + RESET)
    print(f"  {CYAN}9){RESET} Coverage (parameter-space summary + plot)")
    print(f" {CYAN}10){RESET} Import a local spectral grid")
    print(f" {CYAN}11){RESET} Prepare a spectral grid for MESA")
    print(f" {CYAN}12){RESET} Config ")

    print("\n" + DIM + "-" * 50 + RESET)
    print(f"  {RED}0/q){RESET} Quit")
    print(BOLD + "=" * 50 + RESET)

    choice = input("> ").strip()
    mapping = {
        "1": "filters",
        "2": "filters_combine",
        "3": "grids",
        "4": "rebuild",
        "5": "combine",
        "6": "ml_completer",
        "7": "ml_generator",
        "8": "grid_densifier",
        "9": "coverage",
        "10": "import",
        "11": "mesa_prepare",
        "12": "config",
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
    """The ``sed-tools`` parser; subcommands appear in --help in the order added here."""
    parser = argparse.ArgumentParser(description="SED Tools — spectral grids and photometric filters")
    sub = parser.add_subparsers(dest="cmd", required=False)
    _add_grid_download_parser(sub)
    _add_filter_download_parsers(sub)
    _add_filter_catalogue_parsers(sub)
    _add_filter_component_parsers(sub)
    _add_grid_build_parsers(sub)
    _add_grid_tool_parsers(sub)
    return parser


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

    fsp = sub.add_parser("filters-search",
                         help="Search NJM's filter catalogue by name, wavelength and properties (no SVO)")
    fsp.add_argument("text", nargs="?", help="Case-insensitive substring of Facility/Instrument/Band")
    fsp.add_argument("--facility", help="Facility name contains this")
    fsp.add_argument("--instrument", help="Instrument name contains this")
    fsp.add_argument("--band", help="Band name contains this")
    fsp.add_argument("--wavelength", nargs="+", type=_bound, metavar="Å",
                     help="One value: bands covering it; two: bands overlapping that range")
    fsp.add_argument("--pivot", nargs=2, type=_bound, metavar=("LOW", "HIGH"),
                     help="Pivot wavelength range in Å ('-' leaves an end open)")
    fsp.add_argument("--fwhm", nargs=2, type=_bound, metavar=("LOW", "HIGH"), help="FWHM range in Å")
    fsp.add_argument("--detector-type", choices=["energy", "photon"])
    fsp.add_argument("--where", action="append", metavar="KEY=VALUE|KEY=LOW:HIGH",
                     help="Any properties.json column, e.g. ABminusVega=:0.5 or ColorMethod=true; repeatable")
    fsp.add_argument("--columns", help="Comma-separated columns to show (default: %s)" % ",".join(SEARCH_COLUMNS))
    fsp.add_argument("--names-only", action="store_true", help="Match names in index.json only; fetch no properties")
    fsp.add_argument("--format", choices=["table", "csv", "json"], default="table")
    fsp.add_argument("--base", default=str(FILTER_DIR_DEFAULT), help="Base filter directory (cache beside it)")
    fsp.add_argument("--refresh", action="store_true", help="Ask NJM again instead of using the cache")
    fsp.add_argument("--offline", action="store_true", help="Use the cached catalogue only")

    fcip = sub.add_parser("filters-cite",
                          help="Print references (BibTeX), acknowledgements and licences for instruments")
    fcip.add_argument("targets", nargs="*",
                      help="Facility/Instrument or Facility/Instrument/Band; default: every installed instrument")
    fcip.add_argument("--format", choices=["text", "bibtex", "json"], default="text")
    fcip.add_argument("--base", default=str(FILTER_DIR_DEFAULT), help="Base filter directory")
    fcip.add_argument("--refresh", action="store_true", help="Use NJM's current info.json, not installed copies")
    fcip.add_argument("--offline", action="store_true", help="Use installed and cached info.json only")


def _add_filter_component_parsers(sub) -> None:
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
    fbp.add_argument("source", help="Component source (lsst, hst, jwst, hsc, decam, vista, ztf, tabular, skycalc, "
                                    "team_atmospheres, argus) or 'all'")
    fbp.add_argument("--base", default=str(FILTER_DIR_DEFAULT), help="Base filter directory")
    fbp.add_argument("--raw", default=None, help="Raw input directory (source specific)")
    fbp.add_argument("--instrument", help="Instrument for a shared tabular source (HAWKI, NACO, SPHERE, GSAOI or NIFS)")
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

    cp = sub.add_parser("combine", help="Combine multiple grids into omni grid")
    cp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base spectral grid directory")
    cp.add_argument("--output", default="combined_models",
                    help="Name for the combined spectral grid")
    cp.add_argument("--non-interactive", action="store_true",
                    help="Combine all spectral grids without prompting")

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

    p_densify = sub.add_parser("densify", help="Fill coarse Teff gaps in a flux_cube.bin")
    p_densify.add_argument("--flux-cube", required=True)
    p_densify.add_argument("--output", required=True)
    p_densify.add_argument("--teff-spacing", type=float, default=1000.0)
    p_densify.add_argument("--method", default="auto",
                            choices=["auto", "interp", "ml", "blackbody"])
    p_densify.add_argument("--ml-model", default=None)
    p_densify.add_argument("--ml-gap-threshold", type=float, default=_ML_GAP_THRESHOLD_DEFAULT)
    p_densify.add_argument("--no-lookup", action="store_true")


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
                      help="Spectral grid folder name(s); prompted if omitted")
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
                     help="Report parseable headers without importing")


# ------------ Command handlers ------------
# Each takes the parsed arguments and the top-level parser (for parser.error)
# and returns the process exit status.

def _cmd_grids(args, parser):
    run_spectra_flow(
        source=args.source,
        base_dir=args.base,
        models=args.models,
        workers=args.workers,
        force_bundle_h5=not args.no_h5,
        build_flux_cube=not args.no_cube
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
    if args.airmass is None:
        return None
    atmosphere = {"airmass": None if args.airmass == _UNSTATED else args.airmass, "pwv": args.pwv}
    if args.atmosphere_site:
        atmosphere["site"] = args.atmosphere_site
    return atmosphere


def _cmd_filters(args, parser):
    """``filters`` and ``filters-compose``: download (or browse) an instrument, optionally composing it."""
    from .api import Filters
    from .filter_components import ComponentError

    if not args.target:
        if any((args.components, args.bands, args.detector_type, args.airmass is not None,
                args.pwv is not None, args.atmosphere_site, args.allow_heavy, args.allow_partial)):
            parser.error("component options require a Facility/Instrument target")
        run_filters_flow(base_dir=args.base)
        return 0
    options = dict(filter_root=args.base, components=args.components, atmosphere=_atmosphere_request(args, parser),
                   bands=args.bands, detector_type=args.detector_type,
                   allow_heavy=args.allow_heavy, allow_partial=args.allow_partial)
    try:
        if args.cmd == "filters-compose":
            Filters.compose(args.target, **options)
        else:
            print(Filters.fetch(*_facility_instrument(args.target, parser), **options))
    except (ComponentError, RequestException, OSError, RuntimeError) as exc:
        print(f"[filters] {exc}")
        return 1
    return 0


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
        print_filter_availability(info)
        print_catalogue_entry("/".join(target), catalogue)
    return 1 if info.get("errors") or catalogue["errors"] else 0


def _cmd_filters_search(args, parser):
    from .api import Filters
    from .filter_catalogue import CatalogueError

    wavelength = args.wavelength
    if wavelength is not None and (len(wavelength) > 2 or wavelength == [None]):
        parser.error("--wavelength takes one value or a LOW HIGH range")
    try:
        result = Filters.search(
            args.text, facility=args.facility, instrument=args.instrument, band=args.band,
            wavelength=(wavelength[0] if len(wavelength) == 1 else tuple(wavelength)) if wavelength else None,
            pivot=tuple(args.pivot) if args.pivot else None, fwhm=tuple(args.fwhm) if args.fwhm else None,
            detector_type=args.detector_type, where=_where(args.where), properties=not args.names_only,
            filter_root=args.base, refresh=args.refresh, offline=args.offline)
    except (CatalogueError, ValueError) as exc:
        print(f"[filters] {exc}")
        return 1
    columns = args.columns.split(",") if args.columns else SEARCH_COLUMNS
    unknown = [c for c in columns if c not in result.columns and c not in SEARCH_COLUMNS]
    if unknown:
        parser.error(f"unknown columns: {', '.join(unknown)}")
    _print_search_result(result, [c for c in columns if c in result.columns], args.format)
    # Gaps (instruments NJM serves no properties for) are printed but are not failures.
    return 1 if result.attrs.get("errors") else 0


def _print_search_result(result, columns: List[str], fmt: str) -> None:
    shown = result[columns]
    if fmt == "csv":
        print(shown.to_csv(index=False), end="")
    elif fmt == "json":
        print(json.dumps({"bands": json.loads(shown.to_json(orient="records")),
                          "units": result.attrs.get("units", {}),
                          "errors": result.attrs.get("errors", []),
                          "gaps": result.attrs.get("gaps", [])}, indent=2))
    elif shown.empty:
        print("No bands match.")
    else:
        print(shown.to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.6g}"))
        n_inst = len(result[["facility", "instrument"]].drop_duplicates())
        print(f"\n{len(result)} band{'s' * (len(result) != 1)} in "
              f"{n_inst} instrument{'s' * (n_inst != 1)}")
    if fmt != "json":
        for message in result.attrs.get("gaps", []) + result.attrs.get("errors", []):
            print(f"[filters] {message}")


def _cmd_filters_cite(args, parser):
    from .api import Filters
    from .filter_catalogue import format_bibtex, format_citations

    try:
        result = Filters.citations(*args.targets, filter_root=args.base,
                                   refresh=args.refresh, offline=args.offline)
    except ValueError as exc:
        parser.error(str(exc))
    if args.format == "json":
        print(json.dumps(result, indent=2))
    elif args.format == "bibtex":
        print(format_bibtex(result), end="")
        for error in result["errors"]:
            print(f"[filters] {error}", file=sys.stderr)
    else:
        print(format_citations(result), end="")
    return 1 if result["errors"] else 0


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
    from .component_sources import get_builder

    if args.source == "all":
        return _build_all_component_sources(args.base, args.sites)
    builder = get_builder(args.source)
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
        rebuild_flux_cube=not args.no_cube
    )
    return 0


def _cmd_combine(args, parser):
    run_combine_flow(
        base_dir=args.base,
        output_name=args.output,
        interactive=not args.non_interactive
    )
    return 0


def _cmd_ml_completer(args, parser):
    run_ml_completer_flow(
        base_dir=args.base,
        models_dir=args.models_dir
    )
    return 0


def _cmd_ml_generator(args, parser):
    if not args.auto:
        run_ml_generator_flow(
            base_dir=args.base,
            models_dir=args.models_dir
        )
        return 0
    if not args.library or not args.output:
        print("Error: --library and --output are required for --auto training.")
        return 1
    if importlib.util.find_spec("optuna") is None:
        print("Error: --auto training needs the optuna package, which is not installed (pip install 'sed-tools[ml-tune]').")
        return 1
    from .ml import auto_train_generator
    auto_train_generator(library=args.library, output=args.output)
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
            print(f"[coverage] {nm}: {exc}")
            failed.append(f"{nm}: {exc}")
    for line in failed:
        print(f"[coverage] FAILED {line}")
    return 1 if failed else 0


def _cmd_import(args, parser):
    from .api import SED

    SED.import_grid(
        args.path,
        name=args.name,
        model_root=args.base,
        move=args.move,
        build_h5=not args.no_h5,
        build_cube=not args.no_cube,
        dry_run=args.dry_run,
    )
    return 0


_COMMANDS = {
    "grids": _cmd_grids,
    "spectra": _cmd_grids,
    "filters": _cmd_filters,
    "filters-compose": _cmd_filters,
    "filters-combine": _cmd_filters_combine,
    "filters-detector": _cmd_filters_detector,
    "filters-info": _cmd_filters_info,
    "filters-search": _cmd_filters_search,
    "filters-cite": _cmd_filters_cite,
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


def _run_menu():
    """Show the menu and run the chosen workflow until the user quits."""
    _print_banner()
    flows = {
        "grids": lambda: run_spectra_flow(source="all"),
        "filters": run_filters_flow,
        "filters_combine": run_filter_combine_flow,
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
    while True:
        flow = flows.get(menu())
        if flow is None:
            sys.exit(0)
        flow()


# ------------ Main CLI ------------

def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd is None:
        return _run_menu()
    return _COMMANDS[args.cmd](args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
