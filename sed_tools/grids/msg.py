#!/usr/bin/env python3
"""MSGSpectraGrabber: spectra extracted from Townsend's MSG HDF5 grids, with Teff, logg and [M/H] from the grid axes."""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import h5py
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

from ._http import DISCOVERY_TIMEOUT, discovery_failure, mount_discovery, retrying_session
from .hdf5_wavelengths import _is_physical_wavelength_array as _is_physical_wavelength, _search_all_datasets
from .lookup import find_logg_column, find_metallicity_column, find_teff_column, write_lookup_csv

logger = logging.getLogger("sed_tools.msg_spectra_grabber")

_SEG_IDX = re.compile(r"\[(\d+)\]")


def _bracket_number(name):
    """The number in a name's first '[n]' (as in 'specints[12]'), or 0; for sorting groups numerically."""
    m = _SEG_IDX.search(name)
    return int(m.group(1)) if m else 0


def _dataset_if_exists(g, names):
    return next((g[n] for n in names if n in g and isinstance(g[n], h5py.Dataset)), None)


def _number(grp, keys):
    """First of *keys* (attribute, then dataset) that holds a number, else None."""
    for k in keys:
        if k in grp.attrs:
            try:
                return float(np.array(grp.attrs[k]).ravel()[0])
            except (TypeError, ValueError, IndexError):
                pass  # not a number: try the next name
    d = _dataset_if_exists(grp, keys)
    if d is not None:
        arr = np.array(d[()]).ravel()
        if arr.size:
            try:
                return float(arr[0])
            except (TypeError, ValueError):
                pass  # not a number: no value under these names
    return None


def _recover_wavelengths(f, spec_g, expected_len):
    # 1) direct dataset within spectrum group
    WAVE_KEYS = ("lambda", "wavelength", "wave", "wl", "wavelength_A")
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
        segs = []
        for sn in seg_names:
            sgrp = seg_root[sn]
            if not isinstance(sgrp, h5py.Group):
                continue
            xds = _dataset_if_exists(sgrp, ("x",) + WAVE_KEYS)
            if xds is not None:
                x = np.array(xds[()]).astype(float).ravel()
                if x.size:
                    segs.append(x)
                    continue
            start = _number(sgrp, ("start", "min", "lmin", "lambda_min", "lam_min"))
            stop = _number(sgrp, ("stop", "max", "lmax", "lambda_max", "lam_max"))
            step = _number(sgrp, ("dlam", "dl", "step", "delta"))
            npts = _number(sgrp, ("n", "N", "len", "size"))
            if start is not None and stop is not None and step is not None:
                segs.append(np.arange(start, stop + 0.5 * step, step, dtype=float))
            elif start is not None and step is not None and npts is not None:
                segs.append(start + step * np.arange(int(round(npts)), dtype=float))
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
                if abs(wl.size - expected_len) <= 2:
                    return wl[:expected_len] if expected_len and expected_len < wl.size else wl

    # 5) Fallback: any rising, physically plausible dataset of exactly the expected length
    if expected_len > 1:
        wl = _search_all_datasets(f, expected_len)
        if wl is not None:
            return wl
    raise RuntimeError("Unable to recover wavelength grid")


def _pick_flux(spec_g):
    FLUX_PREFS = ("flux", "specific_intensity", "intensity", "F", "H", "I", "c")
    ds = _dataset_if_exists(spec_g, FLUX_PREFS)
    if ds is not None:
        return np.array(ds[()]).astype(float).squeeze()
    # single 1D/(N,1) candidate
    cands = [k for k, d in spec_g.items()
             if isinstance(d, h5py.Dataset) and (len(d.shape) == 1 or (len(d.shape) == 2 and d.shape[1] == 1))]
    if len(cands) == 1:
        return np.array(spec_g[cands[0]][()]).astype(float).squeeze()
    raise RuntimeError("No suitable flux dataset")


def _axis_label(axis_group):
    """The label an MSG axis group carries ('Teff', 'log(g)', '[Fe/H]', '[alpha/Fe]', ...), or ''."""
    label = axis_group.attrs.get("label", b"")
    return (label.decode() if isinstance(label, bytes) else str(label)).strip()


# MSG's vgrid group, as MSG itself reads it (src/vgrid/vgrid_io_sm.fypp, github.com/rhdtownsend/msg):
# - revision 1 names its axes 'axes(r)', revisions 2 and 3 'axes[r]', for r = 1, ..., rank;
# - revisions 1 and 2 mark the vertices that hold a spectrum in a 'mask' over all vertices, revision 3 lists their
#   linear indices in 'v_lin_seq'; the n-th of them holds spectrum specints[n] (specgrid_io_sm.fypp);
# - a linear index is 1-based, with the first axis varying fastest (indexer_index_sm.fypp): Fortran order.
_AXIS_GROUP = {1: "axes({})", 2: "axes[{}]", 3: "axes[{}]"}
_ROLES = (("teff", find_teff_column), ("logg", find_logg_column), ("meta", find_metallicity_column))


def _vertex_parameters(vgrid):
    """The axis labels of an MSG vgrid group, and {role: value at each vertex holding a spectrum, in spectrum order}
    for each role (teff, logg, meta) whose axis label reads as that lookup column (find_teff_column and the rest).

    Raises KeyError or ValueError if the group is not laid out as MSG writes it.
    """
    revision = int(vgrid.attrs["REVISION"])
    if revision not in _AXIS_GROUP:
        raise ValueError(f"vgrid revision {revision}; MSG has written revisions 1 to 3")
    groups = [vgrid[_AXIS_GROUP[revision].format(r)] for r in range(1, int(vgrid.attrs["rank"]) + 1)]
    axes = [np.asarray(g["x"][()], dtype=float).ravel() for g in groups]
    linear = vgrid["v_lin_seq"][()] if revision == 3 else np.flatnonzero(vgrid["mask"][()]) + 1
    vertex = np.unravel_index(np.asarray(linear, dtype=np.int64).ravel() - 1, [a.size for a in axes], order="F")
    labels = [_axis_label(g) for g in groups]
    index = {label: j for j, label in enumerate(labels) if label}
    found = {role: find(index) for role, find in _ROLES}
    return labels, {role: axes[index[label]][vertex[index[label]]] for role, label in found.items() if label}


def _check_parameters(h5_path, numbers, labels, values):
    """Raise RuntimeError unless the spectra specints[1..n] (*numbers*: their numbers) each have a finite Teff and
    logg, and a finite value of a metallicity axis if the grid has one (*values*, from _vertex_parameters)."""
    for role in ("teff", "logg"):
        if role not in values:
            raise RuntimeError(f"MSG: none of the axes of {h5_path} ({', '.join(map(repr, labels))}) reads as "
                               f"{role}, so its spectra have no {role}")
    n = len(values["teff"])
    if numbers != list(range(1, n + 1)):
        raise RuntimeError(f"MSG: the vgrid of {h5_path} places {n} spectra but it holds {len(numbers)} "
                           f"(specints[{numbers[0]}] to [{numbers[-1]}]), so they cannot be matched to parameters")
    lacking = [f"{np.count_nonzero(~np.isfinite(v))} of {n} spectra have no finite {role}"
               for role, v in values.items() if not np.all(np.isfinite(v))]
    if lacking:
        raise RuntimeError(f"MSG: in {h5_path}, " + "; ".join(lacking))


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

    def get_model_metadata(self, model_name):
        """Download the grid's HDF5 file if needed and list its spectra, each with its Teff, logg and meta.

        Raises RuntimeError if the grid was not found by discover_models() on
        this grabber, or the file cannot be downloaded, read, or holds no
        spectra, or its spectra lack a finite Teff or logg, or a finite value
        of the metallicity axis the grid has.
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
            if not names:
                raise RuntimeError(f"MSG: {h5_path} holds no spectra ('specints[...]' groups)")
            try:
                labels, values = _vertex_parameters(f["vgrid"])
            except (KeyError, ValueError) as e:
                raise RuntimeError(f"MSG: {h5_path} has no vgrid group laid out as MSG writes it ({e}), so its "
                                   "spectra have no parameters") from e
        _check_parameters(h5_path, [_bracket_number(nm) for nm in names], labels, values)
        print(f"    Found {len(names)} spectra for {model_name}")
        return [{"fid": n, "gname": f"{spec_root}/{nm}", **{role: float(v[n - 1]) for role, v in values.items()}}
                for n, nm in enumerate(names, 1)]

    def _extract_one(self, h5_path, gpath):
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
            return wl[:n].astype(float), fx[:n].astype(float)

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

        rows = []
        failures = []
        ranges = {"teff": teff_range, "logg": logg_range, "meta": meta_range, "wl": wl_range}

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self._extract_to_text, model_name, h5_path, out_dir, spec, ranges): spec
                    for spec in spectra_info}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Extracting {model_name}"):
                status, fname, detail = fut.result()
                if status == "failed":
                    failures.append(f"{fname}: {detail}")
                elif status != "excluded":
                    rows.append({"file_name": fname, **(detail or {})})

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

        status is 'cached' or 'ok' (detail: its parameters), 'excluded' (outside a requested range) or 'failed' (why).
        """
        fid = int(spec["fid"])
        gpath = spec["gname"]
        params = {key: spec[key] for key in ("teff", "logg", "meta") if key in spec}  # from get_model_metadata
        fname = f"{model_name}_fid{fid}.txt"
        fpath = os.path.join(out_dir, fname)

        if os.path.exists(fpath) and os.path.getsize(fpath) > 1024:
            return ("cached", fname, params)

        if not all(_in_range(params.get(key, np.nan), ranges[key]) for key in ("teff", "logg", "meta")):
            return ("excluded", fname, None)

        try:
            wl, fx = self._extract_one(h5_path, gpath)
        except (KeyError, RuntimeError, OSError, ValueError) as e:
            return ("failed", fname, str(e))

        if ranges["wl"] is not None:
            wl_lo, wl_hi = ranges["wl"]
            mask = (wl >= wl_lo) & (wl <= wl_hi)
            wl, fx = wl[mask], fx[mask]
            if wl.size == 0:
                return ("excluded", fname, None)

        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write("# source = MSG HDF5\n")
            fh.write(f"# spec_group = {gpath}\n")
            for key, value in params.items():
                fh.write(f"# {key} = {value}\n")
            fh.write("# wavelength_unit = Angstrom\n")
            fh.write("# flux_unit = erg/cm2/s/A\n")
            fh.write("# columns = wavelength_A flux\n")
            for x, y in zip(wl, fx):
                fh.write(f"{float(x):.6f} {float(y):.8e}\n")

        return ("ok", fname, params)

    def _write_lookup(self, out_dir, rows):
        path = os.path.join(out_dir, "lookup_table.csv")
        # In file-name order, as the cleaner writes its tables: the extractions finish in any order
        write_lookup_csv(sorted(rows, key=lambda row: row["file_name"]), path,
                         columns=["file_name", "teff", "logg", "meta"], comment=False, restval=float("nan"))
        print(f"    Lookup table saved: {path}")
