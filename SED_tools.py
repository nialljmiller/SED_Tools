#!/usr/bin/env python3
"""
SED_tools.py — one-stop launcher for spectra & filters

Pipeline per selected model:
  - SVO or MSG (Rich Townsend) model discovery
  - Download (SVO) or extract (MSG) spectra -> *.txt collection
  - Ensure lookup_table.csv
  - Build/ensure HDF5 bundle of spectra (for SVO we create one; for MSG we keep the original .h5 + still write a bundle for consistency)
  - Precompute flux cube file

Filters (SVO only):
  - Interactive substring filter over Facility/Instrument/Band (or 'all')
  - Saves to ../data/filters/<Facility>/<Instrument>/<Band>.dat

Defaults:
  STELLAR_DIR = ../data/stellar_models/
  FILTER_DIR  = ../data/filters/

Dependencies (already used in your codebase):
  requests, bs4, h5py, numpy, tqdm, pandas, astroquery, astropy
"""

import argparse
import os
import sys
import re
from typing import List, Dict, Any, Tuple

# local modules you already have
from svo_spectra_grabber import SVOSpectraGrabber         # SVO spectra → .txt + lookup_table.csv
from msg_spectra_grabber import MSGSpectraGrabber         # MSG (Townsend) .h5 → .txt + lookup_table.csv
from precompute_flux_cube import precompute_flux_cube     # builds flux cube from lookup + .txt
from svo_regen_spectra_lookup import regenerate_lookup_table
from mast_spectra_grabber import MASTSpectraGrabber
import h5py
import numpy as np

STELLAR_DIR_DEFAULT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data/stellar_models"))
FILTER_DIR_DEFAULT  = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data/filters"))

# ------------ small utils ------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def list_txt_spectra(model_dir: str) -> List[str]:
    return sorted([f for f in os.listdir(model_dir) if f.lower().endswith(".txt")])

def parse_txt_metadata(txt_path: str) -> Dict[str, str]:
    """Parse '# key = value' header lines; return dict with raw values."""
    meta = {}
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("#") and "=" in s:
                key, val = s.lstrip("#").split("=", 1)
                meta[key.strip()] = val.strip()
            elif not s.startswith("#"):
                break
    return meta

def load_txt_spectrum(txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    wl, fl = [], []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                try:
                    wl.append(float(parts[0]))
                    fl.append(float(parts[1]))
                except ValueError:
                    continue
    return np.asarray(wl, dtype=float), np.asarray(fl, dtype=float)

def numeric_from(meta: Dict[str, str], key_candidates: List[str], default: float = np.nan) -> float:
    """Extract first numeric token from any of the candidate keys (case-insensitive)."""
    lower = {k.lower(): v for k, v in meta.items()}
    for ck in key_candidates:
        if ck.lower() in lower:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lower[ck.lower()])
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    pass
    return default



def regen_lookup_from_txt(model_dir: str) -> str:
    """Rebuild lookup_table.csv by parsing headers from all .txt spectra."""
    txt_files = list_txt_spectra(model_dir)
    if not txt_files:
        raise RuntimeError(f"No .txt spectra found in {model_dir}")

    rows = []
    keys = set(["file_name"])
    for fname in txt_files:
        path = os.path.join(model_dir, fname)
        meta = parse_txt_metadata(path)
        meta["file_name"] = fname
        rows.append(meta)
        keys.update(meta.keys())

    header = ["file_name"] + sorted([k for k in keys if k != "file_name"])
    out_csv = os.path.join(model_dir, "lookup_table.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("#" + ",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in header) + "\n")
    print(f"[rebuild] lookup_table.csv -> {out_csv}")
    return out_csv

def run_rebuild_flow(base_dir: str = STELLAR_DIR_DEFAULT,
                     models: List[str] = None,
                     rebuild_h5: bool = True,
                     rebuild_flux_cube: bool = True) -> None:
    """
    Rebuild lookup_table.csv (+ optional HDF5 bundle and flux cube)
    for existing local model directories.
    """
    ensure_dir(base_dir)

    # discover local model dirs by presence of .txt or .h5
    cand = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p):
            continue
        has_txt = any(fn.lower().endswith(".txt") for fn in os.listdir(p))
        has_h5  = any(fn.lower().endswith(".h5") for fn in os.listdir(p))
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
                    idxs.append(int(token))
            selected = [cand[i-1] for i in idxs if 1 <= i <= len(cand)]
    else:
        selected = models

    if not selected:
        print("Nothing selected.")
        return

    # try to import your dedicated regen (if present)
    regen = None
    try:
        from svo_regen_spectra_lookup import regenerate_lookup_table as regen  # type: ignore
    except Exception:
        regen = None

    for model_name in selected:
        print("\n" + "="*64)
        print(f"[rebuild] {model_name}")
        model_dir = os.path.join(base_dir, model_name)

        # 1) lookup table
        try:
            if regen:
                regen(model_dir)  # your helper
                print("[rebuild] lookup_table.csv via svo_regen_spectra_lookup")
            else:
                regen_lookup_from_txt(model_dir)  # fallback
        except Exception as e:
            print(f"[rebuild] lookup failed: {e}")

        # 2) HDF5 bundle (from .txt), ensure it exists
        if rebuild_h5:
            out_h5 = os.path.join(model_dir, f"{model_name}.h5")
            try:
                build_h5_bundle_from_txt(model_dir, out_h5)
            except Exception as e:
                print(f"[rebuild] h5 bundle failed: {e}")

        # 3) flux cube
        if rebuild_flux_cube:
            out_flux = os.path.join(model_dir, "flux_cube.bin")
            try:
                precompute_flux_cube(model_dir, out_flux)
            except Exception as e:
                print(f"[rebuild] flux cube failed: {e}")

    print("\nRebuild complete.")


# ------------ HDF5 bundling ------------

def build_h5_bundle_from_txt(model_dir: str, out_h5: str) -> None:
    """
    Create an HDF5 file bundling all .txt spectra under groups:
      /spectra/<filename>/lambda (float64)
      /spectra/<filename>/flux   (float64)
    And store key metadata as HDF5 attributes on each group:
      attrs: teff, logg, feh (if found), plus all raw header pairs in attrs["raw:<key>"]
    """
    txt_files = list_txt_spectra(model_dir)
    if not txt_files:
        print(f"[H5 bundle] No .txt spectra found in {model_dir}; skipping.")
        return

    ensure_dir(os.path.dirname(out_h5))
    with h5py.File(out_h5, "w") as h5:
        spectra_grp = h5.create_group("spectra")
        for fname in txt_files:
            path = os.path.join(model_dir, fname)
            try:
                wl, fl = load_txt_spectrum(path)
                if wl.size == 0 or fl.size == 0:
                    print(f"[H5 bundle] Empty or invalid spectrum: {fname}")
                    continue
                g = spectra_grp.create_group(fname)
                g.create_dataset("lambda", data=wl, dtype="f8")
                g.create_dataset("flux",   data=fl, dtype="f8")

                meta = parse_txt_metadata(path)
                # canonical numeric attrs
                teff = numeric_from(meta, ["Teff", "teff", "T_eff"])
                logg = numeric_from(meta, ["logg", "Logg", "log_g"])
                feh  = numeric_from(meta, ["FeH", "feh", "metallicity", "[Fe/H]", "meta"])
                if not np.isnan(teff): g.attrs["teff"] = teff
                if not np.isnan(logg): g.attrs["logg"] = logg
                if not np.isnan(feh):  g.attrs["feh"]  = feh
                # stash raw header lines so nothing is lost
                for k, v in meta.items():
                    g.attrs[f"raw:{k}"] = v
            except Exception as e:
                print(f"[H5 bundle] Error on {fname}: {e}")

    print(f"[H5 bundle] Wrote {out_h5}")

# ------------ spectra orchestration ------------

def run_spectra_flow(source: str,
                     base_dir: str = STELLAR_DIR_DEFAULT,
                     models: List[str] = None,
                     workers: int = 5,
                     force_bundle_h5: bool = True,
                     build_flux_cube: bool = True) -> None:
    """
    source: 'svo', 'msg', or 'both'
    models: optional explicit model names for the given source;
            if None → interactive chooser unified across sources
    """
    ensure_dir(base_dir)

    # Instantiate grabbers
    svo = SVOSpectraGrabber(base_dir=base_dir, max_workers=workers)   # uses your SVO module
    msg = MSGSpectraGrabber(base_dir=base_dir, max_workers=workers)   # uses your MSG module

    # Discover model menus
    choices = []
    if source in ("svo", "both", "all"):
        try:
            mlist = svo.discover_models()
            choices += [("svo", m) for m in mlist]
        except Exception as e:
            print(f"[warn] SVO discovery failed: {e}")
    if source in ("msg", "both", "all"):
        try:
            mlist = msg.discover_models()
            choices += [("msg", m) for m in mlist]
        except Exception as e:
            print(f"[warn] MSG discovery failed: {e}")

    if source in ("mast","all"):
        try:
            mast = MASTSpectraGrabber(base_dir=base_dir, max_workers=workers)
            mlist = mast.discover_models()
            choices += [("mast", m) for m in mlist]
        except Exception as e:
            print(f"[warn] MAST discovery failed: {e}")


    if not choices:
        print("No models discovered. Exiting.")
        return

    if models is None:
        # unified interactive list
        print("\nAvailable models:")
        print("-" * 64)
        for i, (src, mname) in enumerate(choices, 1):
            print(f"{i:3d}. {mname:<30s}  [{src}]")
        print("\nSelect indices (comma, ranges like 3-6 ok) or 'all':")
        raw = input("> ").strip().lower()
        if raw == "all":
            selected = choices
        else:
            idxs = []
            for token in raw.split(","):
                token = token.strip()
                if "-" in token:
                    a, b = token.split("-", 1)
                    idxs += list(range(int(a), int(b)+1))
                else:
                    idxs.append(int(token))
            selected = [choices[i-1] for i in idxs if 1 <= i <= len(choices)]
    else:
        # models explicitly provided → assume all are for the single 'source' unless the user
        # passed "src:model" forms; support both styles
        selected = []
        for m in models:
            if ":" in m:
                src, name = m.split(":", 1)
                selected.append((src.strip().lower(), name.strip()))
            else:
                selected.append((source, m))

    # Process
    for src, model_name in selected:
        print("\n" + "="*64)
        print(f"[{src}] Processing model: {model_name}")
        model_dir = os.path.join(base_dir, model_name)
        ensure_dir(model_dir)

        if src == "svo":
            # 1) discover spectra list and download
            spectra_info = svo.get_model_metadata(model_name)
            if not spectra_info:
                print(f"[SVO] No spectra found for {model_name}; skipping.")
                continue
            n = svo.download_model_spectra(model_name, spectra_info)
            print(f"[SVO] Downloaded {n} spectra into {model_dir}")
            # 2) HDF5 bundle (since SVO doesn't ship one)
            out_h5 = os.path.join(model_dir, f"{model_name}.h5")
            if force_bundle_h5 or (not os.path.exists(out_h5)):
                build_h5_bundle_from_txt(model_dir, out_h5)

        elif src == "msg":
            # 1) locate HDF5, extract to txt + lookup (your class handles both)
            spectra_info = msg.get_model_metadata(model_name)
            if not spectra_info:
                print(f"[MSG] No spectra metadata for {model_name}; skipping.")
                continue
            n = msg.download_model_spectra(model_name, spectra_info)
            print(f"[MSG] Extracted {n} spectra into {model_dir}")
            # ensure we also create a bundle of per-spectrum groups for symmetry
            out_h5_extra = os.path.join(model_dir, f"{model_name}_bundle.h5")
            if force_bundle_h5 and not os.path.exists(out_h5_extra):
                build_h5_bundle_from_txt(model_dir, out_h5_extra)

        elif src == "mast":
            spectra_info = mast.get_model_metadata(model_name)
            if not spectra_info:
                print(f"[MAST] No metadata for {model_name}; skipping.")
                continue
            n = mast.download_model_spectra(model_name, spectra_info)
            print(f"[MAST] Wrote {n} spectra into {model_dir}")
            out_h5 = os.path.join(model_dir, f"{model_name}.h5")
            if force_bundle_h5 and not os.path.exists(out_h5):
                build_h5_bundle_from_txt(model_dir, out_h5)



        # 3) lookup table should already be present from the grabbers.
        lookup_csv = os.path.join(model_dir, "lookup_table.csv")
        if not os.path.exists(lookup_csv):
            print(f"[warn] lookup_table.csv missing in {model_dir}.")
            # could regenerate from text headers if needed here.

        # 4) Flux cube
        if build_flux_cube:
            # place it alongside the model; choose a consistent name
            out_flux = os.path.join(model_dir, "flux_cube.bin")
            try:
                precompute_flux_cube(model_dir, out_flux)
            except Exception as e:
                print(f"[flux-cube] Failed for {model_name}: {e}")

    print("\nAll requested models processed.")

# ------------ filters (SVO only) ------------

def run_filters_flow(base_dir: str = FILTER_DIR_DEFAULT) -> None:
    """
    Simple SVO filters downloader with interactive substring filters.
    Saves to base_dir/<Facility>/<Instrument>/<Band>.dat
    """
    ensure_dir(base_dir)
    try:
        # import lazily so users who only want spectra don't need astroquery installed
        from astropy import units as u
        from astropy.table import unique, vstack
        from astroquery.svo_fps import SvoFps
    except Exception as e:
        print("This feature needs astropy & astroquery installed:", e)
        return

    SvoFps.TIMEOUT = 300

    print("\nFilter search (SVO): leave blank to match everything.")
    fac = input("Substring for Facility (e.g. 'Gaia', '2MASS', 'HST'): ").strip()
    ins = input("Substring for Instrument (e.g. 'WFC3', 'IRAC'): ").strip()
    bnd = input("Substring for Band (e.g. 'G', 'J', 'r'): ").strip()

    print("Choose wavelength span (Angstrom):")
    try:
        wmin = float(input("  min (default 100): ") or "100")
        wmax = float(input("  max (default 1e8): ") or "1e8")
    except ValueError:
        wmin, wmax = 100.0, 1e8

    try:
        idx = SvoFps.get_filter_index(wavelength_eff_min=wmin * u.AA,
                                      wavelength_eff_max=wmax * u.AA)
    except Exception as e:
        print("Failed to query SVO:", e)
        return

    # pandas-ish filtering via astropy table
    df = idx.to_pandas()
    if fac:
        df = df[df["Facility"].str.contains(fac, case=False, na=False)]
    if ins:
        df = df[df["Instrument"].str.contains(ins, case=False, na=False)]
    if bnd:
        df = df[df["Band"].astype(str).str.contains(bnd, case=False, na=False)]

    if df.empty:
        print("No filters matched.")
        return

    print(f"\nMatched {len(df)} filters. Download? (y/n, default y)")
    if (input("> ").strip().lower() or "y").startswith("y"):
        # drop duplicates on filterID
        df = df.drop_duplicates(subset="filterID").reset_index(drop=True)
        for i, row in df.iterrows():
            fid = row["filterID"]
            facility = (row.get("Facility") or "UnknownFacility")
            instrument = (row.get("Instrument") or "UnknownInstrument")
            band = (row.get("Band") or fid.split(".")[-1] or "unknown")

            # sanitize paths
            clean = lambda s: "".join(ch if ch.isalnum() or ch in (" ",".","_") else "_" for ch in str(s)).strip()
            fac_dir = clean(facility)
            ins_dir = clean(instrument)
            fname   = clean(band) + ".dat"

            out_dir = os.path.join(base_dir, fac_dir, ins_dir)
            ensure_dir(out_dir)
            out_path = os.path.join(out_dir, fname)
            try:
                t = SvoFps.get_transmission_data(fid)
                if t is not None and len(t) > 0:
                    t.write(out_path, format="ascii.csv", overwrite=True)
                    print(f"[saved] {out_path}")
                else:
                    print(f"[skip] no data for {fid}")
            except Exception as e:
                print(f"[error] {fid}: {e}")

    print("\nFilter flow complete.")

# ------------ CLI ------------

def main():
    p = argparse.ArgumentParser(description="SED Tools — spectra & filters")
    sub = p.add_subparsers(dest="cmd", required=False)

    # spectra
    sp = sub.add_parser("spectra", help="Download/build spectra products")
    # spectra subparser:
    sp.add_argument("--source", choices=["svo","msg","mast","both","all"], default="all",
                    help="Which provider(s) to use. 'both'=SVO+MSG, 'all'=SVO+MSG+MAST")

    sp.add_argument("--models", nargs="*", default=None,
                    help="Model names to process. For mixed sources, allow 'src:model' entries.")
    sp.add_argument("--base", default=STELLAR_DIR_DEFAULT,
                    help=f"Output base for spectra (default {STELLAR_DIR_DEFAULT})")
    sp.add_argument("--workers", type=int, default=5, help="Parallel workers for grabs/extracts (default 5)")
    sp.add_argument("--no-h5", action="store_true", help="Do not force-create HDF5 bundle from .txt")
    sp.add_argument("--no-cube", action="store_true", help="Skip flux cube build")

    # filters
    fp = sub.add_parser("filters", help="Download SVO filters (interactive substring matching)")
    fp.add_argument("--base", default=FILTER_DIR_DEFAULT,
                    help=f"Output base for filters (default {FILTER_DIR_DEFAULT})")

    # after filters subparser
    rp = sub.add_parser("rebuild", help="Rebuild lookup table + HDF5 + flux cube for local models")
    rp.add_argument("--base", default=STELLAR_DIR_DEFAULT,
                    help=f"Base models dir (default {STELLAR_DIR_DEFAULT})")
    rp.add_argument("--models", nargs="*", default=None,
                    help="Names of local model folders to rebuild (default: interactive)")
    rp.add_argument("--no-h5", action="store_true", help="Skip rebuilding the HDF5 bundle")
    rp.add_argument("--no-cube", action="store_true", help="Skip rebuilding the flux cube")



    # default (no subcommand) → interactive spectra chooser then offer filters
    args = p.parse_args()

    if args.cmd == "spectra":
        run_spectra_flow(source=args.source,
                         base_dir=args.base,
                         models=args.models,
                         workers=args.workers,
                         force_bundle_h5=not args.no_h5,
                         build_flux_cube=not args.no_cube)
    elif args.cmd == "filters":
        run_filters_flow(base_dir=args.base)

    elif args.cmd == "rebuild":
        run_rebuild_flow(base_dir=args.base,
                         models=args.models,
                         rebuild_h5=not args.no_h5,
                         rebuild_flux_cube=not args.no_cube)

    else:
        print("  1) Spectra (SVO / MSG / MAST)")
        print("  2) Filters (SVO)")
        print("  3) Rebuild (lookup + HDF5 + flux cube)")
        choice = (input("> ").strip() or "1")
        if choice == "2":
            run_filters_flow(base_dir=FILTER_DIR_DEFAULT)
        elif choice == "3":
            run_rebuild_flow(base_dir=STELLAR_DIR_DEFAULT)
        else:
            run_spectra_flow(source="all",
                             base_dir=STELLAR_DIR_DEFAULT,
                             models=None,
                             workers=5,
                             force_bundle_h5=True,
                             build_flux_cube=True)

if __name__ == "__main__":
    main()
