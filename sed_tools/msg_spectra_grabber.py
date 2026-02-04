#!/usr/bin/env python3
# msg_spectra_grabber.py — MSG extractor with correct (Teff, logg, [M/H]) for C3K and friends

import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import h5py
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

_SEG_IDX = re.compile(r"\[(\d+)\]")

# Known axis-role overrides per model (0-based axis indices in vgrid/axes[i]/x)
AXIS_ROLE_OVERRIDES = {
    # C3K grids: axes[1]=Teff, axes[2]=[M/H], axes[3]=alpha, axes[4]=logg
    # → zero-based: teff=0, meta=1, logg=3
    "sg-C3K": {"teff": 0, "meta": 1, "logg": 3},
}

def _dataset_if_exists(g, names):
    for n in names:
        if n in g and isinstance(g[n], h5py.Dataset):
            return g[n]
    return None

def _iter_datasets_recursive(g, prefix=""):
    for k, v in g.items():
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(v, h5py.Dataset):
            yield (path, v)
        elif isinstance(v, h5py.Group):
            yield from _iter_datasets_recursive(v, path)

def _recover_wavelengths(spec_g, expected_len=None):
    # 1) direct dataset
    WAVE_KEYS = ("lambda","wavelength","wave","wl","wavelength_A")
    ds = _dataset_if_exists(spec_g, WAVE_KEYS)
    if ds is not None:
        wl = np.array(ds[()]).astype(float).squeeze()
        if wl.size > 1 and np.all(np.diff(wl) > 0):
            return wl

    # 2) range/x
    if "range" in spec_g and isinstance(spec_g["range"], h5py.Group):
        rg = spec_g["range"]
        if "x" in rg and isinstance(rg["x"], h5py.Dataset):
            wl = np.array(rg["x"][()]).astype(float).ravel()
            if wl.size > 1 and np.all(np.diff(wl) > 0):
                return wl

        # 3) concatenated segments
        seg_root = rg.get("ranges", rg)
        seg_names = [k for k in seg_root.keys() if k.startswith("ranges")]
        def seg_key(name):
            m = _SEG_IDX.search(name); return int(m.group(1)) if m else 0
        seg_names.sort(key=seg_key)
        segs=[]
        def num(grp, keys):
            for k in keys:
                if k in grp.attrs:
                    try: return float(np.array(grp.attrs[k]).ravel()[0])
                    except Exception: pass
            d = _dataset_if_exists(grp, keys)
            if d is not None:
                arr = np.array(d[()]).ravel()
                if arr.size:
                    try: return float(arr[0])
                    except Exception: pass
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
            elif start is not None and stop is not None and npts is not None:
                n = int(round(npts))
                if n >= 2: segs.append(np.linspace(start, stop, n, dtype=float))
        if segs:
            wl = np.concatenate(segs)
            if wl.size > 1 and np.all(np.diff(wl) >= 0):
                return wl

    # 4) any monotonic 1-D dataset matching expected_len
    if expected_len and expected_len > 1:
        for path, d in _iter_datasets_recursive(spec_g):
            try:
                arr = np.array(d[()]).astype(float).ravel()
            except Exception:
                continue
            if arr.ndim == 1 and arr.size == expected_len and np.all(np.diff(arr) > 0):
                return arr

    # 5) last resort
    if expected_len and expected_len > 1:
        return np.arange(int(expected_len), dtype=float)
    raise RuntimeError("Unable to recover wavelength grid")

def _pick_flux(spec_g):
    FLUX_PREFS = ("flux","specific_intensity","intensity","F","H","I","c")
    ds = _dataset_if_exists(spec_g, FLUX_PREFS)
    if ds is not None:
        return np.array(ds[()]).astype(float).squeeze()
    # single 1D/(N,1) candidate
    cands=[]
    for k,d in spec_g.items():
        if isinstance(d,h5py.Dataset):
            shp=d.shape
            if len(shp)==1 or (len(shp)==2 and shp[1]==1):
                cands.append(k)
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
    for key, mapping in AXIS_ROLE_OVERRIDES.items():
        if key in model_name:
            return mapping.copy()
    return {}

class MSGSpectraGrabber:
    def __init__(self, base_dir="../data/stellar_models/", max_workers=6):
        self.base_dir = base_dir
        self.max_workers = max_workers
        os.makedirs(base_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SED_Tools (MSG)"} )
        retry = Retry(total=5, backoff_factor=1, status_forcelist=[500,502,503,504,429])
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

        self.index_url = "http://user.astro.wisc.edu/~townsend/static.php?ref=msg-grids"
        self.model_urls = {}
        self.model_axes = {}   # model -> dict(axes=list, shape=tuple, vlin=array, roles=dict)

    def discover_models(self):
        print("Discovering available models from MSG grids...")
        try:
            r = self.session.get(self.index_url, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching MSG index: {e}")
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        models = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/msg/grids/" in href and href.endswith(".h5"):
                name = os.path.basename(href).rstrip(".h5")
                models.append({"name": name, "url": urljoin(self.index_url, href)})
        self.model_urls = {m["name"]: m["url"] for m in models}
        return sorted(self.model_urls.keys())

    def _load_axes_and_vlin(self, f: h5py.File, model_name: str):
        axes=[]
        i=1
        while f"vgrid/axes[{i}]" in f:
            ax = f[f"vgrid/axes[{i}]"]
            if "x" in ax and isinstance(ax["x"], h5py.Dataset):
                axes.append(np.array(ax["x"][()], dtype=float).ravel())
            i+=1
        shape = tuple(len(a) for a in axes)
        # v_lin_seq
        vlin = None
        if "vgrid/v_lin_seq" in f and isinstance(f["vgrid/v_lin_seq"], h5py.Dataset):
            vlin = np.array(f["vgrid/v_lin_seq"][()], dtype=int).ravel()
            prod = int(np.prod(shape)) if shape else 0
            if prod>0 and vlin.size>0:
                # 1-based?
                vmax = int(vlin.max())
                if vmax == prod:
                    vlin = vlin - 1
                # clamp
                vlin = np.clip(vlin, 0, max(prod-1, 0))
        # role mapping
        roles = _role_override_for_model(model_name)
        if not roles:
            roles = _guess_roles_from_axes(axes)
        return {"axes": axes, "shape": shape, "vlin": vlin, "roles": roles}

    def get_model_metadata(self, model_name):
        print(f"  Fetching metadata for {model_name}...")
        out_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        h5_path = os.path.join(out_dir, f"{model_name}.h5")

        url = self.model_urls.get(model_name)
        if not url:
            print(f"    No URL for {model_name}")
            return []

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
                print(f"    Error downloading HDF5: {e}")
                return []

        spectra=[]
        try:
            with h5py.File(h5_path,"r") as f:
                # specsource root
                spec_root=None
                for k in f.keys():
                    if k.startswith("specsource"):
                        spec_root=k; break
                if spec_root is None:
                    print("    No 'specsource' group in HDF5"); return []

                src=f[spec_root]
                names=[k for k in src.keys() if k.startswith("specints[")]
                def keynum(s):
                    m=_SEG_IDX.search(s); return int(m.group(1)) if m else 0
                names.sort(key=keynum)

                # load axes and mapping
                axinfo = self._load_axes_and_vlin(f, model_name)
                self.model_axes[model_name] = axinfo

                for i, nm in enumerate(names):
                    spectra.append({
                        "fid": i+1,
                        "gname": f"{spec_root}/{nm}",
                        "order_index": i  # index into vlin, same order as groups
                    })
            print(f"    Found {len(spectra)} spectra for {model_name}")
            return spectra
        except Exception as e:
            print(f"    Error reading HDF5: {e}")
            return []

    def _params_from_index(self, model_name, order_index):
        axinfo = self.model_axes.get(model_name, {})
        axes = axinfo.get("axes", [])
        shape = axinfo.get("shape", ())
        roles = axinfo.get("roles", {})
        vlin  = axinfo.get("vlin", None)

        teff = logg = meta = float("nan")
        if vlin is None or not shape or order_index >= len(vlin):
            return teff, logg, meta

        flat = int(vlin[order_index])

        # try both unravel orders, pick the one yielding plausible values
        coordsC = None; coordsF = None
        try: coordsC = np.unravel_index(flat, shape, order="C")
        except Exception: pass
        try: coordsF = np.unravel_index(flat, shape, order="F")
        except Exception: pass

        def vals_from_coords(coords):
            if coords is None: return None
            out = {}
            for key, j in roles.items():
                try:
                    out[key] = float(axes[j][coords[j]])
                except Exception:
                    out[key] = float("nan")
            return out

        vc = vals_from_coords(coordsC)
        vf = vals_from_coords(coordsF)

        def plausible(v):
            if v is None: return False
            t = v.get("teff", float("nan"))
            g = v.get("logg", float("nan"))
            m = v.get("meta", float("nan"))
            ok_t = np.isfinite(t) and (1200 <= t <= 150000)
            ok_g = np.isfinite(g) and (-1.5 <= g <= 6.8)
            ok_m = np.isfinite(m) and (-5.0 <= m <= 2.0)
            return ok_t and ok_g and ok_m

        if plausible(vc):
            teff, logg, meta = vc["teff"], vc["logg"], vc["meta"]
        elif plausible(vf):
            teff, logg, meta = vf["teff"], vf["logg"], vf["meta"]
        else:
            # take whichever has more finite fields
            def score(v): return sum(np.isfinite([v.get("teff",np.nan), v.get("logg",np.nan), v.get("meta",np.nan)])) if v else -1
            if score(vc) >= score(vf) and vc:
                teff, logg, meta = vc.get("teff",np.nan), vc.get("logg",np.nan), vc.get("meta",np.nan)
            elif vf:
                teff, logg, meta = vf.get("teff",np.nan), vf.get("logg",np.nan), vf.get("meta",np.nan)

        return teff, logg, meta

    def _extract_one(self, h5_path, gpath, model_name, order_index):
        with h5py.File(h5_path, "r") as f:
            if gpath not in f: raise KeyError(f"missing {gpath}")
            spec_g = f[gpath]
            fx = _pick_flux(spec_g)

            # Normalize fx to something we can write: 1-D flux vector
            fx = np.array(fx)

            # Case 1: dataset is (N, 1) -> flatten
            if fx.ndim == 2 and fx.shape[1] == 1:
                fx = fx[:, 0]

            # Case 2: dataset is (N, 2) and looks like [wl, flux] pairs
            # Use it directly rather than guessing wavelengths elsewhere.
            elif fx.ndim == 2 and fx.shape[1] == 2:
                wl_from_fx = fx[:, 0].astype(float)
                fl_from_fx = fx[:, 1].astype(float)

                # sanity: wl monotonic increasing-ish and length reasonable
                if wl_from_fx.size > 1 and np.all(np.diff(wl_from_fx) > 0):
                    wl = wl_from_fx
                    fx = fl_from_fx
                else:
                    # fallback: take first column as flux if it doesn't look like wavelength
                    fx = fx[:, 0]

            # Case 3: anything higher-dim -> force ravel (last resort)
            elif fx.ndim > 1:
                fx = fx.ravel()



            
            wl = _recover_wavelengths(spec_g, expected_len=len(fx))
            n = min(len(wl), len(fx))
            wl = wl[:n].astype(float); fx = fx[:n].astype(float)
        teff, logg, meta = self._params_from_index(model_name, order_index)
        return wl, fx, teff, logg, meta

    def download_model_spectra(self, model_name, spectra_info):
        out_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        h5_path = os.path.join(out_dir, f"{model_name}.h5")

        if not spectra_info:
            print(f"  No spectra to extract for {model_name}.")
            return 0

        print(f"Extracting {len(spectra_info)} spectra for {model_name}...")

        rows=[]; ok=0

        def task(spec):
            fid = int(spec["fid"])
            gpath = spec["gname"]
            idx = int(spec["order_index"])
            fname = f"{model_name}_fid{fid}.txt"
            fpath = os.path.join(out_dir, fname)

            if os.path.exists(fpath) and os.path.getsize(fpath) > 1024:
                return (fname, None, True)

            try:
                wl, fx, teff, logg, meta = self._extract_one(h5_path, gpath, model_name, idx)
            except Exception as e:
                return (fname, {"error": str(e)}, False)

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

            return (fname, {"teff":teff,"logg":logg,"meta":meta}, True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(task, spec): spec for spec in spectra_info}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Extracting {model_name}"):
                fname, params, okflag = fut.result()
                if okflag:
                    row = {"file_name": fname}
                    if params: row.update(params)
                    rows.append(row); ok += 1

        if rows:
            self._write_lookup(out_dir, rows)
            print(f"  Successfully extracted {ok} spectra")
        else:
            print(f"  No spectra extracted for {model_name}")
        return ok

    def _write_lookup(self, out_dir, rows):
        header = ["file_name","teff","logg","meta"]
        with open(os.path.join(out_dir, "lookup_table.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                out = {k: r.get(k, float("nan")) for k in header}
                w.writerow(out)
        print(f"    Lookup table saved: {os.path.join(out_dir,'lookup_table.csv')}")

    # optional standalone
    def discover_and_run(self, selected_models=None):
        avail = self.discover_models()
        if not avail:
            print("No MSG models found."); return
        selected = selected_models or avail
        total=0
        for name in selected:
            print(f"\n[msg] Processing model: {name}")
            meta = self.get_model_metadata(name)
            if not meta: continue
            total += self.download_model_spectra(name, meta)
        print(f"\n[MSG] Done. Extracted {total} spectra total.")

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
