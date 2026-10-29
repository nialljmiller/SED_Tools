#!/usr/bin/env python3
"""MSGSpectraGrabber: spectra extracted from Townsend's MSG HDF5 grids, with Teff, logg and [M/H] from the grid axes."""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

logger = logging.getLogger("sed_tools.msg_spectra_grabber")

import h5py
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

from ._http import DISCOVERY_TIMEOUT, discovery_failure, mount_discovery, retrying_session
from .hdf5_wavelengths import _is_physical_wavelength_array as _is_physical_wavelength, _iter_datasets
from .lookup import write_lookup_csv

_SEG_IDX = re.compile(r"\[(\d+)\]")


def _bracket_number(name):
    """The number in a name's first '[n]' (as in 'specints[12]'), or 0; for sorting groups numerically."""
    m = _SEG_IDX.search(name)
    return int(m.group(1)) if m else 0


# Known axis-role overrides per model (0-based axis indices in vgrid/axes[i]/x)
AXIS_ROLE_OVERRIDES = {
    # C3K grids: axes[1]=Teff, axes[2]=[M/H], axes[3]=alpha, axes[4]=logg
    # → zero-based: teff=0, meta=1, logg=3
    "sg-C3K": {"teff": 0, "meta": 1, "logg": 3},
}

def _dataset_if_exists(g, names):
    return next((g[n] for n in names if n in g and isinstance(g[n], h5py.Dataset)), None)

def _recover_wavelengths(f, spec_g, expected_len=None):
    # 1) direct dataset within spectrum group
    WAVE_KEYS = ("lambda","wavelength","wave","wl","wavelength_A")
    ds = _dataset_if_exists(spec_g, WAVE_KEYS)
    if ds is not None:
        wl = np.array(ds[()]).astype(float).squeeze()
        if wl.size > 1 and np.all(np.diff(wl) > 0) and _is_physical_wavelength(wl):
            return wl

    # 2) range/x within spectrum group
    if "range" in spec_g and isinstance(spec_g["range"], h5py.Group):
        rg = spec_g["range"]
        if "x" in rg and isinstance(rg["x"], h5py.Dataset):
            wl = np.array(rg["x"][()]).astype(float).ravel()
            if wl.size > 1 and np.all(np.diff(wl) > 0) and _is_physical_wavelength(wl):
                return wl

        # 3) concatenated segments
        seg_root = rg.get("ranges", rg)
        seg_names = sorted((k for k in seg_root.keys() if k.startswith("ranges")), key=_bracket_number)
        segs=[]
        def num(grp, keys):
            """First of *keys* (attribute, then dataset) that holds a number, else None."""
            for k in keys:
                if k in grp.attrs:
                    try: return float(np.array(grp.attrs[k]).ravel()[0])
                    except (TypeError, ValueError, IndexError): pass  # not a number: try the next name
            d = _dataset_if_exists(grp, keys)
            if d is not None:
                arr = np.array(d[()]).ravel()
                if arr.size:
                    try: return float(arr[0])
                    except (TypeError, ValueError): pass  # not a number: no value under these names
            return None
        for sn in seg_names:
            sgrp = seg_root[sn]
            if not isinstance(sgrp, h5py.Group): continue
            xds = _dataset_if_exists(sgrp, ("x",)+WAVE_KEYS)
            if xds is not None:
                x = np.array(xds[()]).astype(float).ravel()
                if x.size: segs.append(x); continue
            start = num(sgrp, ("start","min","lmin","lambda_min","lam_min"))
            stop  = num(sgrp, ("stop","max","lmax","lambda_max","lam_max"))
            step  = num(sgrp, ("dlam","dl","step","delta"))
            npts  = num(sgrp, ("n","N","len","size"))
            if start is not None and stop is not None and step is not None:
                segs.append(np.arange(start, stop+0.5*step, step, dtype=float))
            elif start is not None and step is not None and npts is not None:
                segs.append(start + step*np.arange(int(round(npts)), dtype=float))
            elif start is not None and stop is not None and npts is not None and int(round(npts)) >= 2:
                segs.append(np.linspace(start, stop, int(round(npts)), dtype=float))
            else:
                # Skipping the segment would leave a gap that _extract_one hides by truncating
                # the flux, so look for the wavelengths in the layouts below instead
                segs = []
                break
        if segs:
            wl = np.concatenate(segs)
            if wl.size > 1 and np.all(np.diff(wl) >= 0) and _is_physical_wavelength(wl):
                return wl

    # 4) Top-level wavelength arrays in the HDF5 root (MSG often stores here)
    for key in WAVE_KEYS:
        if key in f and isinstance(f[key], h5py.Dataset):
            wl = np.array(f[key][()]).astype(float).ravel()
            if wl.size > 1 and np.all(np.diff(wl) > 0) and _is_physical_wavelength(wl):
                if expected_len is None or abs(wl.size - expected_len) <= 2:
                    return wl[:expected_len] if expected_len and expected_len < wl.size else wl

    # 5) Fallback: search entire file for a physically plausible wavelength array
    #    Only accept arrays that pass physical wavelength validation.
    if expected_len and expected_len > 1:
        for d in _iter_datasets(f):
            try:
                arr = np.array(d[()]).astype(float).ravel()
            except (TypeError, ValueError):
                continue  # not numeric, so not a wavelength grid
            if arr.size == expected_len and np.all(np.diff(arr) > 0) and _is_physical_wavelength(arr):
                return arr

    raise RuntimeError("Unable to recover wavelength grid")


def _pick_flux(spec_g):
    FLUX_PREFS = ("flux","specific_intensity","intensity","F","H","I","c")
    ds = _dataset_if_exists(spec_g, FLUX_PREFS)
    if ds is not None:
        return np.array(ds[()]).astype(float).squeeze()
    # single 1D/(N,1) candidate
    cands = [k for k, d in spec_g.items()
             if isinstance(d, h5py.Dataset) and (len(d.shape) == 1 or (len(d.shape) == 2 and d.shape[1] == 1))]
    if len(cands)==1:
        return np.array(spec_g[cands[0]][()]).astype(float).squeeze()
    raise RuntimeError("No suitable flux dataset")

def _guess_roles_from_axes(axes):
    # Basic heuristics if no override: pick max-span as Teff; logg in [-1.5,6.8]; meta in [-3.5,1.5]
    role = {}
    if not axes:
        return role
    # Teff: largest max
    role["teff"] = int(np.argmax([np.nanmax(a) for a in axes]))
    # meta: span in [-3.5,+1.5]
    cand_meta = [j for j,a in enumerate(axes) if -3.5 <= float(np.nanmin(a)) <= 1.5 and -3.5 <= float(np.nanmax(a)) <= 1.5]
    if cand_meta: role["meta"] = cand_meta[0]
    # logg: span in [-1.5,6.8], not meta
    cand_logg = [j for j,a in enumerate(axes) if -1.5 <= float(np.nanmin(a)) <= 6.8 and -1.5 <= float(np.nanmax(a)) <= 6.8 and j != role.get("meta")]
    if cand_logg: role["logg"] = cand_logg[0]
    return role

def _role_override_for_model(model_name):
    return next((mapping.copy() for key, mapping in AXIS_ROLE_OVERRIDES.items() if key in model_name), {})

def _in_range(val, rng):
    """Return True if val is within rng (lo, hi), or if rng is None/val is nan."""
    if rng is None:
        return True
    if not np.isfinite(val):
        return True  # can't filter what we don't know
    lo, hi = rng
    return lo <= val <= hi

class MSGSpectraGrabber:
    def __init__(self, base_dir="../data/stellar_models/", max_workers=6):
        self.base_dir = base_dir
        self.max_workers = max_workers
        os.makedirs(base_dir, exist_ok=True)

        self.session = retrying_session("SED_Tools (MSG)", Retry(
            total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 429]), separate_adapters=True)

        self.index_url = "http://user.astro.wisc.edu/~townsend/static.php?ref=msg-grids"
        mount_discovery(self.session, self.index_url)
        self.model_urls = {}
        self.model_axes = {}   # model -> dict(axes=list, shape=tuple, vlin=array, roles=dict)

    def discover_models(self):
        """Return the MSG grids on the index page; [] after a warning if it cannot be read."""
        print("Discovering spectral grids from MSG...")
        try:
            r = self.session.get(self.index_url, timeout=DISCOVERY_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning("MSG: grid index %s %s; MSG spectral grids are not offered",
                           self.index_url, discovery_failure(e))
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        urls = {}
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/msg/grids/" in href and href.endswith(".h5"):
                urls[os.path.basename(href).removesuffix(".h5")] = urljoin(self.index_url, href)
        self.model_urls = urls
        if not self.model_urls:
            logger.warning("MSG: no grids linked from %s; MSG spectral grids are not offered", self.index_url)
        return sorted(self.model_urls.keys())

    def _load_axes_and_vlin(self, f: h5py.File, model_name: str):
        axes=[]
        i=0
        while f"vgrid/axes[{i}]" in f:
            ax = f[f"vgrid/axes[{i}]"]
            if "x" in ax and isinstance(ax["x"], h5py.Dataset):
                axes.append(np.array(ax["x"][()], dtype=float).ravel())
            i+=1
        shape = tuple(len(a) for a in axes)
        vlin = None
        if "vgrid/v_lin_seq" in f and isinstance(f["vgrid/v_lin_seq"], h5py.Dataset):
            vlin = np.array(f["vgrid/v_lin_seq"][()], dtype=int).ravel()
            prod = int(np.prod(shape)) if shape else 0
            if prod>0 and vlin.size>0:
                if int(vlin.max()) == prod:  # 1-based
                    vlin = vlin - 1
                vlin = np.clip(vlin, 0, max(prod-1, 0))
        roles = _role_override_for_model(model_name) or _guess_roles_from_axes(axes)
        return {"axes": axes, "shape": shape, "vlin": vlin, "roles": roles}

    def get_model_metadata(self, model_name):
        """Download the grid's HDF5 file if needed and list its spectra.

        Raises RuntimeError if the grid was not found by discover_models() on
        this grabber, or the file cannot be downloaded, read, or holds no
        spectra.
        """
        print(f"  Fetching metadata for {model_name}...")
        out_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        h5_path = os.path.join(out_dir, f"{model_name}.h5")

        url = self.model_urls.get(model_name)
        if not url:
            raise RuntimeError(f"MSG grid {model_name} is unknown to this grabber; "
                               "call discover_models() first and check the name")

        if not os.path.exists(h5_path):
            print(f"    Downloading {model_name}.h5 ...")
            try:
                with self.session.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(h5_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1<<20):
                            if not chunk: break
                            f.write(chunk)
            except requests.RequestException as e:
                raise RuntimeError(f"MSG: could not download {url} to {h5_path}: {e}. "
                                   f"Delete any partial {h5_path} before retrying") from e

        try:
            f = h5py.File(h5_path, "r")
        except OSError as e:
            raise RuntimeError(f"MSG: could not read {h5_path}: {e}. If it is a partial "
                               "download, delete it and rerun to download it again") from e

        with f:
            spec_root = next((k for k in f.keys() if k.startswith("specsource")), None)
            if spec_root is None:
                raise RuntimeError(f"MSG: no 'specsource' group in {h5_path}")

            names = sorted((k for k in f[spec_root].keys() if k.startswith("specints[")), key=_bracket_number)

            self.model_axes[model_name] = self._load_axes_and_vlin(f, model_name)

            # order_index: the index into vlin, same order as the groups
            spectra = [{"fid": i + 1, "gname": f"{spec_root}/{nm}", "order_index": i} for i, nm in enumerate(names)]
        if not spectra:
            raise RuntimeError(f"MSG: {h5_path} holds no spectra ('specints[...]' groups)")
        print(f"    Found {len(spectra)} spectra for {model_name}")
        return spectra

    def _params_from_index(self, model_name, order_index):
        axinfo = self.model_axes.get(model_name, {})
        axes = axinfo.get("axes", [])
        shape = axinfo.get("shape", ())
        roles = axinfo.get("roles", {})
        vlin  = axinfo.get("vlin", None)

        if vlin is None or not shape or order_index >= len(vlin):
            return float("nan"), float("nan"), float("nan")

        flat = int(vlin[order_index])

        def vals_from_coords(coords):
            out = {}
            for key, j in roles.items():
                if j >= len(axes):
                    raise RuntimeError(f"MSG {model_name}: {key} is mapped to axis {j}, "
                                       f"but the grid has {len(axes)} axes")
                out[key] = float(axes[j][coords[j]])
            return out

        # try both unravel orders, pick the one yielding plausible values
        # (vlin was clipped into the grid when loaded, so both orders give valid coordinates)
        vc = vals_from_coords(np.unravel_index(flat, shape, order="C"))
        vf = vals_from_coords(np.unravel_index(flat, shape, order="F"))

        def plausible(v):  # a missing, NaN or infinite value fails its bounds
            return all(lo <= v.get(key, np.nan) <= hi
                       for key, lo, hi in (("teff", 1200, 150000), ("logg", -1.5, 6.8), ("meta", -5.0, 2.0)))

        if plausible(vc):
            best = vc
        elif plausible(vf):
            best = vf
        else:
            # take whichever has more finite fields
            def score(v): return sum(np.isfinite([v.get("teff",np.nan), v.get("logg",np.nan), v.get("meta",np.nan)]))
            best = vc if score(vc) >= score(vf) else vf
        return best.get("teff",np.nan), best.get("logg",np.nan), best.get("meta",np.nan)

    def _extract_one(self, h5_path, gpath, model_name, order_index):
        with h5py.File(h5_path, "r") as f:
            spec_g = f[gpath]
            fx = _pick_flux(spec_g)  # a float array, squeezed: an (N, 1) dataset is already 1-D

            # (N, 2) [wl, flux] pairs, when the first column rises: the flux is the second column (the
            # wavelengths are still recovered below); otherwise the first column is taken as the flux
            if fx.ndim == 2 and fx.shape[1] == 2:
                first = fx[:, 0]
                fx = fx[:, 1] if first.size > 1 and np.all(np.diff(first) > 0) else first
            # anything higher-dim -> force ravel (last resort)
            elif fx.ndim > 1:
                fx = fx.ravel()

            wl = _recover_wavelengths(f, spec_g, expected_len=len(fx))
            n = min(len(wl), len(fx))
            wl = wl[:n].astype(float); fx = fx[:n].astype(float)
        teff, logg, meta = self._params_from_index(model_name, order_index)
        return wl, fx, teff, logg, meta

    def download_model_spectra(self, model_name, spectra_info,
                               teff_range=None, logg_range=None,
                               meta_range=None, wl_range=None):
        """Extract the listed spectra from the grid's HDF5 file to .txt files.

        Spectra outside a requested range are left out. Returns the number of
        spectra written or already on disk. If any spectrum cannot be
        extracted, the rest are still written, lookup_table.csv is not, and
        RuntimeError lists each failure.
        """
        out_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        h5_path = os.path.join(out_dir, f"{model_name}.h5")

        if not spectra_info:
            print(f"  No spectra to extract for {model_name}.")
            return 0

        print(f"Extracting {len(spectra_info)} spectra for {model_name}...")

        rows=[]; failures=[]
        ranges = {"teff": teff_range, "logg": logg_range, "meta": meta_range, "wl": wl_range}

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self._extract_to_text, model_name, h5_path, out_dir, spec, ranges): spec
                    for spec in spectra_info}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Extracting {model_name}"):
                status, fname, detail = fut.result()
                if status == "failed":
                    failures.append(f"{fname}: {detail}")
                elif status != "excluded":
                    row = {"file_name": fname}
                    if detail: row.update(detail)
                    rows.append(row)

        if failures:
            raise RuntimeError(
                f"MSG: {len(failures)} of {len(spectra_info)} spectra of {model_name} could not be "
                f"extracted, so lookup_table.csv was not written; the others are in {out_dir}:\n  "
                + "\n  ".join(sorted(failures)))

        if rows:
            self._write_lookup(out_dir, rows)
            print(f"  Successfully extracted {len(rows)} spectra")
        else:
            print(f"  No spectra extracted for {model_name}")
        return len(rows)

    def _extract_to_text(self, model_name, h5_path, out_dir, spec, ranges):
        """Write one spectrum to <out_dir>/<model>_fid<n>.txt: (status, file name, detail).

        status is 'cached', 'excluded' (outside a requested range), 'ok' (detail: its parameters) or 'failed' (why).
        """
        fid = int(spec["fid"])
        gpath = spec["gname"]
        idx = int(spec["order_index"])
        fname = f"{model_name}_fid{fid}.txt"
        fpath = os.path.join(out_dir, fname)

        if os.path.exists(fpath) and os.path.getsize(fpath) > 1024:
            return ("cached", fname, None)

        try:
            wl, fx, teff, logg, meta = self._extract_one(h5_path, gpath, model_name, idx)
        except (KeyError, RuntimeError, OSError, ValueError) as e:
            return ("failed", fname, str(e))

        if not all(_in_range(value, ranges[key]) for key, value in (("teff", teff), ("logg", logg), ("meta", meta))):
            return ("excluded", fname, None)

        if ranges["wl"] is not None:
            wl_lo, wl_hi = ranges["wl"]
            mask = (wl >= wl_lo) & (wl <= wl_hi)
            wl, fx = wl[mask], fx[mask]
            if wl.size == 0:
                return ("excluded", fname, None)

        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("# source = MSG HDF5\n")
            fh.write(f"# spec_group = {gpath}\n")
            if np.isfinite(teff): fh.write(f"# teff = {teff}\n")
            if np.isfinite(logg): fh.write(f"# logg = {logg}\n")
            if np.isfinite(meta): fh.write(f"# meta = {meta}\n")
            fh.write("# wavelength_unit = Angstrom\n")
            fh.write("# flux_unit = erg/cm2/s/A\n")
            fh.write("# columns = wavelength_A flux\n")
            for x, y in zip(wl, fx):
                # Ensure x and y are Python scalars to avoid TypeError in f-strings
                fh.write(f"{float(x):.6f} {float(y):.8e}\n")

        return ("ok", fname, {"teff":teff,"logg":logg,"meta":meta})

    def _write_lookup(self, out_dir, rows):
        path = os.path.join(out_dir, "lookup_table.csv")
        write_lookup_csv(rows, path, columns=["file_name", "teff", "logg", "meta"], comment=False, restval=float("nan"))
        print(f"    Lookup table saved: {path}")

    # optional standalone
    def discover_and_run(self, selected_models=None):
        avail = self.discover_models()
        if not avail:
            print("No MSG spectral grids found."); return
        selected = selected_models or avail
        total=0; failed=[]
        for name in selected:
            print(f"\n[msg] Processing spectral grid: {name}")
            try:
                meta = self.get_model_metadata(name)
                total += self.download_model_spectra(name, meta)
            except RuntimeError as e:  # finish the other grids, then report
                failed.append(f"{name}: {e}")
        print(f"\n[MSG] Done. Extracted {total} spectra total.")
        if failed:
            raise RuntimeError(f"MSG: {len(failed)} of {len(selected)} grid(s) failed:\n  " + "\n  ".join(failed))

def main():
    import argparse
    p = argparse.ArgumentParser(description="Extract MSG stellar spectra to ASCII with correct params")
    p.add_argument("--output", default="../data/stellar_models/")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--models", nargs="*")
    a = p.parse_args()
    g = MSGSpectraGrabber(base_dir=a.output, max_workers=a.workers)
    g.discover_and_run(selected_models=a.models)

if __name__ == "__main__":
    main()