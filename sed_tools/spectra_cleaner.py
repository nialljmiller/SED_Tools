#!/usr/bin/env python3
"""
spectra_cleaner.py — make spectra MESA-safe and scientifically sane

What it does (per *.txt in a model folder):
  • Drops non-numeric and comment lines (keeps lines starting with '#')
  • Removes NaN/inf rows and any wavelength <= 0
  • Sorts by wavelength, deduplicates, enforces strictly increasing λ
  • Detects "index grid" (λ ≈ 0..N-1). If found:
      - If header has '# source = MSG HDF5' and '# spec_group = <path>'
        and the original HDF5 with that group is present, we
        reconstruct the true wavelength from the HDF5 and rewrite the file.
      - Otherwise, we mark the file as BAD and skip it (do not rewrite).
  • Optionally writes a .bak before changing files.
  • Rebuilds lookup_table.csv from headers of files that still exist.

Return value: a summary dict with counts and lists of fixed/skipped files.
"""

from __future__ import annotations
import os, re, glob, io
import numpy as np
import h5py

# ------------ basic IO ------------

_NUM = re.compile(r'[,\s]+')


def _parse_header(path: str) -> dict:
    meta = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if "=" in s:
                    k, v = s.lstrip("#").split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            break
    return meta


def _read_xy_body(path: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = _NUM.split(s)
            if len(parts) < 2:
                continue
            try:
                xx = float(parts[0]); yy = float(parts[1])
            except ValueError:
                continue
            x.append(xx); y.append(yy)
    return np.asarray(x, float), np.asarray(y, float)


def _write_xy_with_header(path: str, header: str, wl: np.ndarray, fl: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(header.rstrip() + "\n")
        for a, b in zip(wl, fl):
            f.write(f"{a:.6f} {b:.8e}\n")


def _read_full_header_text(path: str) -> str:
    buf = io.StringIO()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip().startswith("#"):
                break
            buf.write(line)
    return buf.getvalue()

# ------------ HDF5 wavelength recovery (generic, works for MSG .h5) ------------

def _dataset_if_exists(g, names):
    for n in names:
        if n in g and isinstance(g[n], h5py.Dataset):
            return g[n]
    return None


def _read_scalar_anywhere(grp, keys):
    for k in keys:
        if k in grp.attrs:
            try:
                return float(np.array(grp.attrs[k]).ravel()[0])
            except Exception:
                pass
    d = _dataset_if_exists(grp, keys)
    if d is not None:
        try:
            arr = np.array(d[()]).ravel()
            if arr.size:
                return float(arr[0])
        except Exception:
            pass
    return None


def _iter_datasets_recursive(g, prefix=""):
    for k, v in g.items():
        p = f"{prefix}/{k}" if prefix else k
        if isinstance(v, h5py.Dataset):
            yield p, v
        elif isinstance(v, h5py.Group):
            yield from _iter_datasets_recursive(v, p)


def _recover_wavelengths_from_group(spec_g, expected_len=None):
    # 1) direct dataset
    KEYS = ("lambda","wavelength","wave","wl","wavelength_A")
    ds = _dataset_if_exists(spec_g, KEYS)
    if ds is not None:
        wl = np.array(ds[()]).astype(float).squeeze()
        if wl.size > 1 and np.all(np.diff(wl) > 0):
            return wl

    # 2) /range/x
    if "range" in spec_g and isinstance(spec_g["range"], h5py.Group):
        rg = spec_g["range"]
        if "x" in rg and isinstance(rg["x"], h5py.Dataset):
            wl = np.array(rg["x"][()]).astype(float).ravel()
            if wl.size > 1 and np.all(np.diff(wl) > 0):
                return wl

        # 3) concatenated segments under 'ranges'
        seg_root = rg.get("ranges", rg)
        seg_names = [k for k in seg_root.keys() if k.startswith("ranges")]
        def seg_key(name):
            m = re.search(r"\[(\d+)\]", name)
            return int(m.group(1)) if m else 0
        seg_names.sort(key=seg_key)
        segs = []
        for sn in seg_names:
            sgrp = seg_root[sn]
            if not isinstance(sgrp, h5py.Group):
                continue
            xds = _dataset_if_exists(sgrp, ("x",)+KEYS)
            if xds is not None:
                x = np.array(xds[()]).astype(float).ravel()
                if x.size:
                    segs.append(x); continue
            # parametric
            start = _read_scalar_anywhere(sgrp, ("start","min","lmin","lambda_min","lam_min"))
            stop  = _read_scalar_anywhere(sgrp, ("stop","max","lmax","lambda_max","lam_max"))
            step  = _read_scalar_anywhere(sgrp, ("dlam","dl","step","delta"))
            npts  = _read_scalar_anywhere(sgrp, ("n","N","len","size"))
            if start is not None and stop is not None and step is not None:
                segs.append(np.arange(start, stop + 0.5*step, step, dtype=float))
            elif start is not None and step is not None and npts is not None:
                segs.append(start + step*np.arange(int(round(npts)), dtype=float))
            elif start is not None and stop is not None and npts is not None:
                n = int(round(npts))
                if n >= 2:
                    segs.append(np.linspace(start, stop, n, dtype=float))
        if segs:
            wl = np.concatenate(segs)
            if wl.size > 1 and np.all(np.diff(wl) >= 0):
                return wl

    # 4) deep search for any 1-D monotonic dataset matching expected_len
    if expected_len and expected_len > 1:
        for _, d in _iter_datasets_recursive(spec_g):
            try:
                arr = np.array(d[()]).astype(float).ravel()
            except Exception:
                continue
            if arr.ndim == 1 and arr.size == expected_len and np.all(np.diff(arr) > 0):
                return arr

    # give up
    return None


def _try_h5_recover(model_dir: str, spec_group: str, expected_len: int) -> np.ndarray | None:
    # try any .h5 in the folder which contains the spec_group
    for h5name in sorted(glob.glob(os.path.join(model_dir, "*.h5"))):
        try:
            with h5py.File(h5name, "r") as f:
                if spec_group in f:
                    wl = _recover_wavelengths_from_group(f[spec_group], expected_len)
                    if wl is not None and wl.size >= 2 and wl[0] > 0 and np.all(np.diff(wl) > 0):
                        return wl
        except Exception:
            continue
    return None

# ------------ cleaning core ------------

def _is_index_grid(wl: np.ndarray) -> bool:
    """Detect λ ≈ 0..N-1 (within tiny tolerance)."""
    if wl.size < 4:
        return False
    if wl[0] != 0:
        return False
    # strong test: all integers and max == N-1
    if np.allclose(wl, np.arange(wl.size), atol=1e-12, rtol=0.0):
        return True
    # fallback: uniform step ~1 and near integer sequence
    step = np.median(np.diff(wl))
    if not np.allclose(step, 1.0, atol=1e-6, rtol=0.0):
        return False
    resid = wl - np.round(wl)
    return (np.max(np.abs(resid)) < 1e-6) and int(np.round(wl[-1])) == wl.size - 1


def _clean_one_file(path: str, try_h5_recovery: bool, make_backup: bool) -> tuple[str, str]:
    """
    Returns (status, detail)
      status in {"ok","fixed","recovered","skipped","deleted"}
    """
    header = _read_full_header_text(path)
    meta = _parse_header(path)
    wl, fl = _read_xy_body(path)

    if wl.size == 0 or fl.size == 0:
        return ("deleted", "empty body")

    # remove NaN/inf and λ<=0
    good = np.isfinite(wl) & np.isfinite(fl) & (wl > 0)
    wl, fl = wl[good], fl[good]
    if wl.size < 2:
        return ("skipped", "no positive-λ points")

    # sort + strictly increasing
    order = np.argsort(wl)
    wl, fl = wl[order], fl[order]
    m = np.diff(wl) > 0
    if m.size:
        keep = np.hstack([True, m])
        wl, fl = wl[keep], fl[keep]

    # index-grid detection → recover from HDF5 if possible
    if _is_index_grid(wl):
        source = (meta.get("source") or "").lower()
        spec_group = meta.get("spec_group")
        if try_h5_recovery and ("msg hdf5" in source) and spec_group:
            model_dir = os.path.dirname(path)
            wl2 = _try_h5_recover(model_dir, spec_group, expected_len=fl.size)
            if wl2 is not None:
                wl = wl2[:fl.size]
                status = "recovered"
            else:
                return ("skipped", "index grid; no HDF5 λ found")
        else:
            return ("skipped", "index grid")

    # write back (with optional .bak)
    if make_backup:
        try:
            os.replace(path, path + ".bak")
        except Exception:
            pass
        # re-write header (keep as-is)
        _write_xy_with_header(path, header, wl, fl)
    else:
        _write_xy_with_header(path, header, wl, fl)

    return ("recovered" if '_is_index_grid' in locals() and _is_index_grid(wl) else "fixed", f"{wl[0]:.3f}..{wl[-1]:.3f} Å")


def _rebuild_lookup(model_dir: str) -> str:
    """Rebuild lookup_table.csv from headers of existing .txt files."""
    txts = sorted(glob.glob(os.path.join(model_dir, "*.txt")))
    if not txts:
        return ""
    # harvest keys
    rows = []
    keys = {"file_name"}
    for p in txts:
        meta = _parse_header(p)
        meta["file_name"] = os.path.basename(p)
        rows.append(meta)
        keys.update(meta.keys())
    header = ["file_name"] + sorted(k for k in keys if k != "file_name")
    out_csv = os.path.join(model_dir, "lookup_table.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in header) + "\n")
    return out_csv

# ------------ public API ------------

def clean_model_dir(model_dir: str,
                    try_h5_recovery: bool = True,
                    backup: bool = True,
                    rebuild_lookup: bool = True) -> dict:
    """
    Clean all spectra in model_dir in-place.

    Returns a summary:
      {
        'total': int,
        'fixed': [...files...],
        'recovered': [...files...],
        'skipped': [...files...],
        'deleted': [...files...],
        'lookup_updated': bool,
        'lookup_path': '...'
      }
    """
    txts = sorted(glob.glob(os.path.join(model_dir, "*.txt")))
    summary = {
        "total": len(txts),
        "fixed": [], "recovered": [], "skipped": [], "deleted": [],
        "lookup_updated": False, "lookup_path": ""
    }
    if not txts:
        return summary

    for p in txts:
        status, info = _clean_one_file(p, try_h5_recovery, backup)
        if status in summary:
            summary[status].append(os.path.basename(p))
        # If we skipped due to index grid, leave the original file untouched.
        # (Optional: you could delete those here, but safer to keep for inspection.)

    if rebuild_lookup:
        # Drop rows for files no longer present, and include newly recovered headers.
        lp = _rebuild_lookup(model_dir)
        if lp:
            summary["lookup_updated"] = True
            summary["lookup_path"] = lp

    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Clean spectra in a model directory")
    ap.add_argument("model_dir", help="Folder containing *.txt spectra")
    ap.add_argument("--no-h5", dest="try_h5", action="store_false", help="Do not attempt HDF5 recovery")
    ap.add_argument("--no-backup", dest="backup", action="store_false", help="Do not write .bak files")
    ap.add_argument("--no-lookup", dest="lookup", action="store_false", help="Do not rebuild lookup_table.csv")
    args = ap.parse_args()
    s = clean_model_dir(args.model_dir, try_h5_recovery=args.try_h5, backup=args.backup, rebuild_lookup=args.lookup)
    print(s)
