#!/usr/bin/env python3
"""
SED_tools.py — one-stop launcher for spectra & filters

Pipeline per selected model:
  - SVO or MSG (Rich Townsend) model discovery
  - Download (SVO) or extract (MSG) spectra -> *.txt collection
  - Ensure lookup_table.csv
  - Build/ensure HDF5 bundle of spectra
  - Precompute flux cube file

Filters (SVO only):
  - Interactive substring filter over Facility/Instrument/Band
  - Saves to data/filters/<Facility>/<Instrument>/<Band>.dat
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import set_data_dir, show_config
from .mast_spectra_grabber import MASTSpectraGrabber
from .models import FILTER_DIR_DEFAULT, STELLAR_DIR_DEFAULT
from .msg_spectra_grabber import MSGSpectraGrabber
from .njm_spectra_grabber import NJMSpectraGrabber
from .precompute_flux_cube import precompute_flux_cube
from .spectra_cleaner import clean_model_dir
from .svo_regen_spectra_lookup import regenerate_lookup_table
from .svo_spectra_grabber import SVOSpectraGrabber
from .ui_utils import prompt_choice  # public, no leading underscore
from .parsing import parse_multi_selection, parse_numeric_range
from .spectrum_io import build_h5_bundle, list_text_spectra, read_text_spectrum
from .terminal_plots import terminal_color_enabled
# ------------ Small Utils ------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_txt_spectra(model_dir: str) -> List[str]:
    return [path.name for path in list_text_spectra(model_dir)]


def load_txt_spectrum(txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return read_text_spectrum(txt_path)


# ------------ HDF5 Bundling ------------

def build_h5_bundle_from_txt(model_dir: str, out_h5: str) -> None:
    """Create an HDF5 file bundling all .txt spectra."""
    if not build_h5_bundle(model_dir, out_h5):
        print(f"[H5 bundle] No .txt spectra found in {model_dir}; skipping.")
        return
    print(f"[H5 bundle] Wrote {out_h5}")


def _parse_range(raw: str) -> Optional[Tuple[float, float]]:
    """Parse a user-entered range string like '3500,8000' or '3500 8000' or '3500-8000'.
    
    Returns (min, max) tuple or None if empty/invalid.
    """
    parsed = parse_numeric_range(raw)
    if parsed is not None or not raw.strip():
        return parsed
    print(f"  Could not parse range from '{raw}' — need two values (e.g. '3500,8000')")
    return None


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

    cuts['teff_range'] = _parse_range(input("  Teff range (e.g. '3500,8000') or blank for all: ").strip())
    cuts['logg_range'] = _parse_range(input("  logg range (e.g. '3.5,5.0') or blank for all: ").strip())
    cuts['meta_range'] = _parse_range(input("  [M/H] range (e.g. '-1.0,0.5') or blank for all: ").strip())
    cuts['wl_range']   = _parse_range(input("  Wavelength range in Å (e.g. '3000,10000') or blank for all: ").strip())

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
    ensure_dir(base_dir)

    cand = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p):
            continue
        has_txt = any(fn.lower().endswith(".txt") for fn in os.listdir(p))
        has_h5 = any(fn.lower().endswith(".h5") for fn in os.listdir(p))
        if has_txt or has_h5:
            cand.append(name)

    if not cand:
        print(f"No local models found under {base_dir}")
        return

    if models is None:
        print("\nLocal models available for rebuild:")
        print("-" * 64)
        for i, m in enumerate(cand, 1):
            print(f"{i:3d}. {m}")
        print("\nSelect indices (comma / ranges like 3-6) or 'all':")
        raw = input("> ").strip().lower()
        if raw == "all":
            selected = cand
        else:
            idxs = []
            for token in raw.split(","):
                token = token.strip()
                if "-" in token:
                    a, b = token.split("-", 1)
                    idxs += list(range(int(a), int(b)+1))
                else:
                    if token:
                        idxs.append(int(token))
            selected = [cand[i-1] for i in idxs if 1 <= i <= len(cand)]
    else:
        selected = models

    if not selected:
        print("Nothing selected.")
        return

    for model_name in selected:
        print("\n" + "=" * 64)
        print(f"[rebuild] {model_name}")
        model_dir = os.path.join(base_dir, model_name)

        summary = clean_model_dir(model_dir, try_h5_recovery=True, rebuild_lookup=True)
        print(f"[clean] {model_name}: total={summary['total']}")

        txts = glob.glob(os.path.join(model_dir, "*.txt"))
        if not txts:
            print(f"[rebuild] no spectra (.txt) present after cleaning; skipping {model_name}.")
            continue

        regenerate_lookup_table(model_dir)

        if rebuild_h5:
            out_h5 = os.path.join(model_dir, f"{model_name}.h5")
            build_h5_bundle_from_txt(model_dir, out_h5)

        if rebuild_flux_cube:
            out_flux = os.path.join(model_dir, "flux_cube.bin")
            precompute_flux_cube(model_dir, out_flux)

    print("\nRebuild complete.")











class _ModelOption:
    """Display wrapper for a discoverable model, used by run_spectra_flow."""
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
        print(f" Method    : Catalog consensus ({sample_size}/{n_total} files sampled)")
        print(f" Wavelength: {detected_wl} ({det_stats.get('wavelength_agreement', '?')} agreement)")
        print(f" Flux      : {detected_flux} ({det_stats.get('flux_agreement', '?')} agreement)")
        print(f" Confidence: {cat_units.get('confidence', 'unknown')}")
    else:
        print(f" Status    : {det_status or 'No unit detection performed'}")
        print(f" Wavelength: {detected_wl}")
        print(f" Flux      : {detected_flux}")

    # Per-file outcome counts
    units_already_standard = (detected_wl == 'angstrom' and detected_flux == 'flam')
    n_converted = len(summary.get('converted', []))
    n_tagged    = len(summary.get('tagged', []))
    n_recovered = len(summary.get('recovered', []))
    n_skipped   = len(summary.get('skipped_already', []) or summary.get('skipped', []))
    n_invalid   = len(summary.get('skipped_invalid', []) or summary.get('invalid', []))
    n_index     = len(summary.get('skipped_index', []))
    n_error     = len(summary.get('error', []))

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
    """Build HDF5 bundle, lookup table, and flux cube for a downloaded model."""
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
    flux_cube_path = os.path.join(model_dir, "flux_cube.bin")
    precompute_flux_cube(model_dir, flux_cube_path)
    variants_index = os.path.join(model_dir, "variants_index.csv")
    if os.path.exists(variants_index):
        with open(variants_index) as vf:
            header = vf.readline()
            vrows = list(csv.DictReader(vf, fieldnames=[
                c.lstrip("#").strip() for c in header.strip().split(",")
            ]))
        print(f"Flux cubes    : {len(vrows)} MESA-ready subgrid(s) created")
        for row in vrows:
            vname = row.get("variant_name", row.get("path", "?"))
            print(f"  -> {os.path.join(model_dir, vname)}/")
        print(f"\n  Point MESA Colors to one of the above subdirectories.")
        print(f"  See {variants_index} for the full inventory.")
    elif os.path.exists(flux_cube_path):
        print(f"Flux cube     : flux_cube.bin")
    else:
        print(f"Flux cube     : not built")


def run_spectra_flow(
    source: str,
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models: Optional[List[str]] = None,
    workers: int = 5,
    force_bundle_h5: bool = True,
    build_flux_cube: bool = True,
) -> None:
    ensure_dir(base_dir)

    # Parse source list
    if source.lower() == "all":
        src_list = ["njm", "svo", "msg", "mast"]
    elif source.lower() == "both":
        src_list = ["njm", "svo", "msg"]
    elif "," in source:
        src_list = [s.strip().lower() for s in source.split(",")]
    else:
        src_list = [source.lower()]

    # Initialize grabbers (NJM first so availability is checked before discovery)
    grabs: Dict = {}
    if "njm" in src_list:
        grabs["njm"] = NJMSpectraGrabber(base_dir=base_dir, max_workers=workers)
        if not grabs["njm"].is_available():
            print("[njm] Mirror unavailable, using other sources")
            del grabs["njm"]
            src_list.remove("njm")
    if "svo"  in src_list: grabs["svo"]  = SVOSpectraGrabber(base_dir=base_dir, max_workers=workers)
    if "msg"  in src_list: grabs["msg"]  = MSGSpectraGrabber(base_dir=base_dir, max_workers=workers)
    if "mast" in src_list: grabs["mast"] = MASTSpectraGrabber(base_dir=base_dir, max_workers=workers)

    # Discover models from every active source
    model_sources: Dict[str, List[str]] = {}
    for s in src_list:
        if s in grabs:
            for model_name in grabs[s].discover_models():
                model_sources.setdefault(model_name, []).append(s)
    if not model_sources:
        print("No models discovered.")
        return

    all_models = [_ModelOption(n, srcs) for n, srcs in sorted(model_sources.items())]

    # Resolve which models to process
    if models is not None:
        if len(models) == 1 and models[0].lower() == "all":
            chosen = [(opt.sources, opt.name) for opt in all_models]
        else:
            chosen = []
            for m in models:
                if ":" in m:
                    s, n = m.split(":", 1)
                    chosen.append(([s.strip().lower()], n.strip()))
                else:
                    matching = [opt for opt in all_models if opt.name == m.strip()]
                    if not matching:
                        print(f"[skip] Model '{m.strip()}' not found")
                    else:
                        chosen.append((matching[0].sources, matching[0].name))
    else:
        idxs = prompt_choice(all_models, label="Spectral models", allow_back=True, multi=True)
        if idxs is None or idxs == -1:
            return
        if isinstance(idxs, int):
            idxs = [idxs]
        chosen = [(all_models[i].sources, all_models[i].name) for i in idxs]

    for sources, name in chosen:
        print("\n" + "=" * 64)
        print(f"{''.join(f'[{s}]' for s in sources)} {name}")
        print("=" * 64)

        model_dir = os.path.join(base_dir, name)
        ensure_dir(model_dir)

        # Try sources in priority order until one supplies metadata
        meta = g = src = None
        for try_src in sources:
            g = grabs.get(try_src)
            if not g:
                continue
            meta = g.get_model_metadata(name)
            if meta:
                src = try_src
                break
            print(f"  [{try_src}] No metadata — trying next source...")
        if not (meta and g and src):
            print("No metadata available from any source")
            continue

        print(f"  Using source: {src}")
        is_preprocessed = isinstance(meta, dict) and meta.get("pre_processed", False)

        njm_cuts: Dict = {}
        if src == "njm":
            njm_cuts = prompt_njm_axis_cuts(name, g, model_url=meta.get("model_url"))

        n_written = g.download_model_spectra(
            name, meta,
            teff_range=njm_cuts.get('teff_range'),
            logg_range=njm_cuts.get('logg_range'),
            meta_range=njm_cuts.get('meta_range'),
            wl_range=njm_cuts.get('wl_range'),
        )
        print(f"Downloaded {n_written} spectra → {model_dir}")

        if is_preprocessed:
            print("Pre-processed data (skipping cleaning)")
            essential = ["flux_cube.bin", "lookup_table.csv"]
            missing = [f for f in essential if not os.path.exists(os.path.join(model_dir, f))]
            print(f"Missing: {', '.join(missing)}" if missing else "All essential files present")
            continue

        summary = clean_model_dir(model_dir, try_h5_recovery=True, rebuild_lookup=True)
        _print_cleaning_report(summary)

        if not glob.glob(os.path.join(model_dir, "*.txt")):
            print("\n  No spectra remaining after cleaning")
            continue

        _build_data_products(name, model_dir, src, force_bundle_h5, build_flux_cube)

    print("\n" + "─" * 64)
    print("Done.")



def run_filters_flow(base_dir: str = str(FILTER_DIR_DEFAULT)) -> None:
    """
    Interactive filter downloader with automatic NJMSVO fallback.
    
    Shows full SVO catalog, user selects what they want, then automatically
    downloads from NJM mirror if available, falling back to SVO if not.
    """
    ensure_dir(base_dir)
    
    from .njm_filter_grabber import NJMFilterGrabber
    from .svo_filter_grabber import SVOFilterBrowser

    # Initialize both sources
    svo = SVOFilterBrowser(base_dir=base_dir)
    njm = NJMFilterGrabber(base_dir=base_dir)
    njm_available = njm.is_available()
    
    # Browse SVO catalog (authoritative source)
    print("\nBrowsing SVO Filter Profile Service...")
    
    facilities = svo.list_facilities()
    if not facilities:
        print("No facilities found.")
        return
    
    # Facility selection loop
    while True:
        bulk_spec = input(
            "Bulk telescope mode: enter facility IDs (e.g. 1,3-5) to download ALL instruments, "
            "or press Enter for single-facility mode: "
        ).strip()

        if bulk_spec:
            try:
                facility_indexes = parse_multi_selection(bulk_spec, len(facilities))
            except ValueError as exc:
                print(f"Invalid selection: {exc}")
                continue
            if not facility_indexes:
                print("No facilities selected.")
                continue

            selected_facilities = [facilities[i] for i in facility_indexes]
            confirm = input(
                f"Download ALL instruments for {len(selected_facilities)} selected facilities? [Y/n] "
            ).strip().lower()
            if confirm and not confirm.startswith('y'):
                continue

            for facility in selected_facilities:
                instruments = svo.list_instruments(facility.key)

                if not instruments:
                    print(f"No instruments found for {facility.label}.")
                    continue

                print(f"\n{facility.label}: {len(instruments)} instruments")
                for instrument in instruments:
                    filters = svo.list_filters(facility.key, instrument.key)

                    if not filters:
                        print(f"  [skip] {instrument.label}: no filters found")
                        continue

                    downloaded = False
                    if njm_available:
                        njm_facilities = njm.discover_facilities()
                        if facility.key in njm_facilities:
                            njm_instruments = njm.discover_instruments(facility.key)
                            if instrument.key in njm_instruments:
                                print(f"  [njm] Downloading {instrument.label}...")
                                count = njm.download_filters(facility.key, instrument.key)
                                if count > 0:
                                    print(f"  [njm] Downloaded {count} filters")
                                    downloaded = True

                    if not downloaded:
                        print(f"  [svo] Downloading {instrument.label} ({len(filters)} filters)...")
                        svo.download_filters(filters)

            again = input("\nBulk download another set of facilities? [y/N] ").strip().lower()
            if again.startswith('y'):
                continue
            return

        fac_idx = prompt_choice(facilities, "Filter Facilities")
        if fac_idx is None:
            return

        facility = facilities[fac_idx]
        instruments = svo.list_instruments(facility.key)

        if not instruments:
            print(f"No instruments found for {facility.label}.")
            continue

        # Instrument selection loop
        while True:
            inst_idx = prompt_choice(
                instruments,
                f"Instruments for {facility.label}",
                allow_back=True
            )

            if inst_idx is None:
                return
            if inst_idx == -1:
                break  # Back to facilities

            instrument = instruments[inst_idx]
            filters = svo.list_filters(facility.key, instrument.key)

            if not filters:
                print(f"No filters found for {instrument.label}.")
                continue

            # Show selection
            print(f"\n{facility.label} / {instrument.label}")
            print(f"Found {len(filters)} filters")

            confirm = input("Download all filters? [Y/n] ").strip().lower()
            if confirm and not confirm.startswith('y'):
                continue

            # Smart download: Try NJM first, fall back to SVO
            downloaded = False

            if njm_available:
                # Check if NJM has this facility/instrument
                njm_facilities = njm.discover_facilities()
                if facility.key in njm_facilities:
                    njm_instruments = njm.discover_instruments(facility.key)
                    if instrument.key in njm_instruments:
                        # NJM has it - download from there
                        print(f"\n[njm] Downloading from mirror...")
                        count = njm.download_filters(facility.key, instrument.key)
                        if count > 0:
                            print(f"[njm]  Downloaded {count} filters")
                            downloaded = True

            # Fall back to SVO if NJM didn't work
            if not downloaded:
                print(f"\n[svo] Downloading from SVO...")
                svo.download_filters(filters)
                print(f"[svo]  Downloaded {len(filters)} filters")

            again = input("\nDownload another instrument? [y/N] ").strip().lower()
            if not again.startswith('y'):
                break


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
    Export a single extra-axis sub-variant of a model as a clean MESA-ready folder.

    Delegates to mesa_prepare.run_interactive(), which handles model selection,
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
    Combine multiple stellar atmosphere grids into unified omni grid.

    This creates a single flux cube spanning the parameter space of all
    selected models.

    Args:
        base_dir: Base directory containing stellar_models/ subdirectories
        output_name: Name for the output combined model directory
        interactive: If True, prompt user to select models
    """
    from .combine_stellar_atm import (build_combined_flux_cube,
                                      create_common_wavelength_grid,
                                      create_unified_grid, find_stellar_models,
                                      load_model_data, save_combined_data,
                                      select_models_interactive,
                                      visualize_parameter_space)

    ensure_dir(base_dir)

    model_dirs = find_stellar_models(base_dir)
    if not model_dirs:
        print(f"No stellar models found in {base_dir}")
        return

    selected_models = select_models_interactive(model_dirs) if interactive else model_dirs
    if not selected_models:
        print("No models selected.")
        return


    if interactive:
        output_name = input(f"What should the combined model be called? [{output_name}]") or output_name

    print("\nModel will be saved as:", output_name, "\n")


    print(f"\nSelected {len(selected_models)} models to combine:")
    for name, _ in selected_models:
        print(f"- {name}")

    print("\nLoading model data...")
    all_models_data = []
    for name, path in selected_models:
        print(f"Loading {name}...")
        all_models_data.append(load_model_data(path))

    print("\nCreating unified parameter grids...")
    teff_grid, logg_grid, meta_grid = create_unified_grid(all_models_data)
    wavelength_grid = create_common_wavelength_grid(all_models_data)

    flux_cube, source_map = build_combined_flux_cube(
        all_models_data, teff_grid, logg_grid, meta_grid, wavelength_grid
    )

    output_dir = os.path.join(base_dir, output_name)
    save_combined_data(
        output_dir,
        teff_grid,
        logg_grid,
        meta_grid,
        wavelength_grid,
        flux_cube,
        all_models_data,
    )

    visualize_parameter_space(
        teff_grid, logg_grid, meta_grid, source_map, all_models_data, output_dir,
        wavelength_grid=wavelength_grid, flux_cube=flux_cube
    )
    
    print(f"\nSuccessfully combined {len(selected_models)} stellar atmosphere models!")
    print(f"Output saved to: {output_dir}")
    print("You can now use this combined model in MESA by setting:")
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
    """Local model folders that have a lookup table or at least one .txt."""
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
    name = input("Model name [blank = source folder name]: ").strip() or None
    mv = input("Move files instead of copying? [y/N] ").strip().lower().startswith("y")
    SED.import_grid(path, name=name, model_root=base_dir, move=mv)



def menu() -> str:
    use_color = terminal_color_enabled("auto")
    BOLD = "\x1b[1m" if use_color else ""
    DIM = "\x1b[2m" if use_color else ""
    CYAN = "\x1b[36m" if use_color else ""
    YELL = "\x1b[33m" if use_color else ""
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


    print(BOLD + "=" * 50 + RESET)

    print("\n" + YELL + "-- Filters " + "-" * 39 + RESET)
    print(f"  {CYAN}1){RESET} Download Filters (NJM / SVO)")
    print(f"  {CYAN}2){RESET} Combine filter sets")

    print("\n" + YELL + "-- Spectra (Stellar Atmosphere Tables)" + "-" * 12 + RESET)
    print(f"  {CYAN}3){RESET} Download Spectra (NJM / SVO / MSG / MAST)")
    print(f"  {CYAN}4){RESET} Rebuild (lookup + HDF5 + flux cube)")
    print(f"  {CYAN}5){RESET} Combine grids into omni grid")

    print("\n" + YELL + "-- ML " + "-" * 44 + RESET)
    print(f"  {CYAN}6){RESET} ML SED Completer (train/extend incomplete SEDs)")
    print(f"  {CYAN}7){RESET} ML SED Generator (generate SEDs from parameters)")
    print(f"  {CYAN}8){RESET} Grid Densifier (fill coarse Teff gaps)")

    print("\n" + YELL + "-- Tools " + "-" * 41 + RESET)
    print(f"  {CYAN}9){RESET} Coverage (parameter-space summary + plot)")
    print(f" {CYAN}10){RESET} Load (Import a local stellar atm grid)")
    print(f" {CYAN}11){RESET} MESA Prepare (Prepare stellar atm for MESA and SED_Tools)")
    print(f" {CYAN}12){RESET} Config ")

    print("\n" + DIM + "-" * 50 + RESET)
    print(f"  {RED}0/q){RESET} Quit")
    print(BOLD + "=" * 50 + RESET)

    choice = input("> ").strip()
    mapping = {
        "1": "filters",
        "2": "filters_combine",
        "3": "spectra",
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

# ------------ Main CLI ------------

def main():
    parser = argparse.ArgumentParser(description="SED Tools — spectra & filters")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # spectra
    sp = sub.add_parser("spectra", help="Download/build spectra products")
    sp.add_argument("--source",
                    choices=["njm", "svo", "msg", "mast", "both", "all"],
                    default="all",
                    help="Which provider(s) to use. NJM mirror is checked first when available.")
    sp.add_argument("--models", nargs="*", default=None,
                    help="Model names to process.")
    sp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Output base for spectra")
    sp.add_argument("--workers", type=int, default=5, help="Parallel workers")
    sp.add_argument("--no-h5", action="store_true", help="Do not force-create HDF5 bundle")
    sp.add_argument("--no-cube", action="store_true", help="Skip flux cube build")

    # filters
    fp = sub.add_parser("filters", help="Download SVO filters")
    fp.add_argument("--base", default=str(FILTER_DIR_DEFAULT),
                    help="Output base for filters")

    fcp = sub.add_parser("filters-combine", help="Combine filter sets into one MESA-compatible instrument")
    fcp.add_argument("output", help="Output name or Facility/Instrument path for the combined filter set")
    fcp.add_argument("inputs", nargs="+", help="Filter-set directories or .dat files to combine")
    fcp.add_argument("--base", default=str(FILTER_DIR_DEFAULT),
                     help="Base filter directory")
    fcp.add_argument("--facility", default=None, help="Output facility label (default: Combined)")
    fcp.add_argument("--instrument", default=None, help="Output instrument/index-file name")
    fcp.add_argument("--on-conflict", choices=["rename", "overwrite", "error"], default="rename",
                     help="How duplicate filter filenames are handled")

    # rebuild
    rp = sub.add_parser("rebuild", help="Rebuild lookup table + HDF5 + flux cube")
    rp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base models dir")
    rp.add_argument("--models", nargs="*", default=None,
                    help="Names of local model folders to rebuild")
    rp.add_argument("--no-h5", action="store_true", help="Skip HDF5 bundle")
    rp.add_argument("--no-cube", action="store_true", help="Skip flux cube")

    # combine
    cp = sub.add_parser("combine", help="Combine multiple grids into omni grid")
    cp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base models directory")
    cp.add_argument("--output", default="combined_models",
                    help="Name for output combined model")
    cp.add_argument("--non-interactive", action="store_true",
                    help="Combine all models without prompting")

    # ml_completer
    mp = sub.add_parser("ml_completer", help="Train/use ML SED completer")
    mp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base models directory")
    mp.add_argument("--models-dir", default="models",
                    help="Directory for trained ML models")

    # ml_generator
    gp = sub.add_parser("ml_generator", help="Train/use ML SED generator")
    gp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base models directory")
    gp.add_argument("--models-dir", default="models",
                    help="Directory for trained ML models")
    gp.add_argument("--auto", action="store_true",
                    help="Run optimized auto-training with hyperparameter search")
    gp.add_argument("--library", help="Library path for auto-training")
    gp.add_argument("--output", help="Output directory for auto-training")

    # densify
    p_densify = sub.add_parser("densify", help="Fill coarse Teff gaps in a flux_cube.bin")
    p_densify.add_argument("--flux-cube", required=True)
    p_densify.add_argument("--output", required=True)
    p_densify.add_argument("--teff-spacing", type=float, default=1000.0)
    p_densify.add_argument("--method", default="auto",
                            choices=["auto", "interp", "ml", "blackbody"])
    p_densify.add_argument("--ml-model", default=None)
    p_densify.add_argument("--ml-gap-threshold", type=float, default=5000.0)
    p_densify.add_argument("--no-lookup", action="store_true")

    # mesa_prepare
    pp = sub.add_parser("mesa_prepare",
        help="Export a single sub-variant of a model as a clean MESA-ready folder")
    pp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                    help="Base models directory")
    pp.add_argument("--model", default=None,
                    help="Model name (e.g. Kurucz2003all); prompted if omitted")
    pp.add_argument("--output", default=None,
                    help="Output directory for the exported variant; prompted if omitted")

    # config
    cfg_p = sub.add_parser("config", help="Show or set the data directory")
    cfg_p.add_argument("--set", metavar="PATH", default=None,
                       help="Set data directory to PATH")
    cfg_p.add_argument("--move", action="store_true",
                       help="With --set: move existing data to the new path")

    # coverage
    covp = sub.add_parser("coverage",
        help="Report parameter-space coverage of a local grid")
    covp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                      help="Base models dir")
    covp.add_argument("--models", nargs="*", default=None,
                      help="Model folder name(s); prompted if omitted")
    covp.add_argument("--no-plot", action="store_true",
                      help="Skip the coverage plot")
    covp.add_argument("--out", default=None,
                      help="Output path for the plot PNG (single model only)")

    # import
    imp = sub.add_parser("import",
        help="Ingest a local grid of .txt spectra into the pipeline")
    imp.add_argument("--path", required=True,
                     help="Directory of .txt spectra (or a single .txt file)")
    imp.add_argument("--name", default=None,
                     help="Model name (default: source folder name)")
    imp.add_argument("--base", default=str(STELLAR_DIR_DEFAULT),
                     help="Base models dir")
    imp.add_argument("--move", action="store_true",
                     help="Move files instead of copying")
    imp.add_argument("--no-h5", action="store_true", help="Skip HDF5 bundle")
    imp.add_argument("--no-cube", action="store_true", help="Skip flux cube")
    imp.add_argument("--dry-run", action="store_true",
                     help="Report parseable headers without importing")

    args = parser.parse_args()

    if args.cmd == "spectra":
        run_spectra_flow(
            source=args.source,
            base_dir=args.base,
            models=args.models,
            workers=args.workers,
            force_bundle_h5=not args.no_h5,
            build_flux_cube=not args.no_cube
        )
    elif args.cmd == "filters":
        run_filters_flow(base_dir=args.base)
    elif args.cmd == "filters-combine":
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
    elif args.cmd == "rebuild":
        run_rebuild_flow(
            base_dir=args.base,
            models=args.models,
            rebuild_h5=not args.no_h5,
            rebuild_flux_cube=not args.no_cube
        )
    elif args.cmd == "combine":
        run_combine_flow(
            base_dir=args.base,
            output_name=args.output,
            interactive=not args.non_interactive
        )
    elif args.cmd == "ml_completer":
        run_ml_completer_flow(
            base_dir=args.base,
            models_dir=args.models_dir
        )
    elif args.cmd == "ml_generator":
        if args.auto:
            if not args.library or not args.output:
                print("Error: --library and --output are required for --auto training.")
                sys.exit(1)
            from .ml import auto_train_generator
            auto_train_generator(library=args.library, output=args.output)
        else:
            run_ml_generator_flow(
                base_dir=args.base,
                models_dir=args.models_dir
            )
    elif args.cmd == "densify":
        from .grid_densifier import densify_grid
        densify_grid(
            src=args.flux_cube,
            dst=args.output,
            teff_spacing=args.teff_spacing,
            method=args.method,
            ml_model=args.ml_model,
            ml_gap_threshold=args.ml_gap_threshold,
            write_lookup=not args.no_lookup,
        )
    elif args.cmd == "mesa_prepare":
        run_mesa_prepare_flow(
            base_dir=args.base,
            model=args.model,
            output=args.output,
        )
    elif args.cmd == "config":
        if args.set:
            set_data_dir(args.set, move=args.move if args.move else None)
        else:
            show_config()
    elif args.cmd == "coverage":
        from .api import SED
        names = args.models
        if not names:
            names = _discover_local_grids(args.base)
            if not names:
                print(f"No local grids found under {args.base}")
                return
        for nm in names:
            SED.coverage(
                nm,
                model_root=args.base,
                plot=not args.no_plot,
                out_path=args.out if len(names) == 1 else None,
            )
    elif args.cmd == "import":
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
    else:
        # Interactive mode
        while True:
            choice = menu()
            if choice == "spectra":
                run_spectra_flow(source="all")
            elif choice == "filters":
                run_filters_flow()
            elif choice == "filters_combine":
                run_filter_combine_flow()
            elif choice == "rebuild":
                run_rebuild_flow()
            elif choice == "combine":
                run_combine_flow()
            elif choice == "ml_completer":
                run_ml_completer_flow()
            elif choice == "ml_generator":
                run_ml_generator_flow()
            elif choice == "grid_densifier":
                run_grid_densifier_flow()
            elif choice == "mesa_prepare":
                run_mesa_prepare_flow()
            elif choice == "config":
                run_config_flow()
            elif choice == "coverage":
                run_coverage_flow()
            elif choice == "import":
                run_import_flow()
            else:
                sys.exit(0)


if __name__ == "__main__":
    main()
