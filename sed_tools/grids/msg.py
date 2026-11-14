#!/usr/bin/env python3
"""MSGSpectraGrabber: spectra extracted from Townsend's MSG HDF5 grids, each with its parameters from the grid axes."""

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
from .lookup import find_logg_column, find_metallicity_column, find_teff_column, write_lookup_csv
from .spectrum_io import write_atomically

logger = logging.getLogger(__name__)

_SEG_IDX = re.compile(r"\[(\d+)\]")


def _bracket_number(name):
    """The number in a name's first '[n]' (as in 'specints[12]'), or 0; for sorting groups numerically."""
    m = _SEG_IDX.search(name)
    return int(m.group(1)) if m else 0


def _text(value):
    """An HDF5 string attribute as str."""
    return (value.decode() if isinstance(value, bytes) else str(value)).strip()


def _axis_label(axis_group):
    """The label an MSG axis group carries ('Teff', 'log(g)', '[Fe/H]', '[alpha/Fe]', ...), or ''."""
    return _text(axis_group.attrs.get("label", b""))


# MSG's spectra, limb_specint_t groups (github.com/rhdtownsend/msg at fe7c45f2; paths below are under src/):
# - 'range' holds the wavelength points in Angstrom, unpacked as MSG unpacks each range type: lin_range_t
#   x_0 + dx*(i-1), log_range_t exp(logx_0 + dlogx*(i-1)), tab_range_t its dataset 'x', comp_range_t its
#   sub-ranges ranges[1..n_ranges] one after another (range/*/*_range_operate_sm.fypp, unpack);
# - the points are bin edges: 'c' has one row per bin, n_lam - 1 rows for n_lam points
#   (specint/limb/limb_specint_construct_sm.fypp:48), and the intensity is constant over each bin
#   [lam(i), lam(i+1)) (docs: user-guide/msg-fundamentals.rst:117-122; interp_f_ in limb_specint_interp_sm.fypp).
#   Each bin is written at its centre, as MSG's own walkthrough places binned flux (python-walkthrough.ipynb:
#   "lam_c defines the central wavelength of each bin");
# - a row holds the limb-darkening law's coefficients of the specific intensity, in erg/cm2/s/A/sr, and the flux
#   over the bin is their dot product with the law's flux basis, 2 pi times these
#   (limb/limb_basis_sm.fypp:266-305, eval_flux_basis).
_FLUX_BASIS = {"CONST": (1 / 2,), "LINEAR": (1 / 2, -1 / 6), "SQRT": (1 / 2, -1 / 10, -1 / 6),
               "QUAD": (1 / 2, -1 / 6, -1 / 4), "CLARET": (1 / 2, -1 / 10, -1 / 6, -3 / 14, -1 / 4)}
#: A spectrum may hold negative fluxes, kept as MSG gives them, only while their integrated size is below this
#: fraction of the spectrum's integrated absolute flux. Some model spectra carry an isolated negative bin from
#: their source's own data.
_NEGATIVE_FLUX_LIMIT = 1e-3


def _negative_fraction(wl, fx):
    """The integrated size of a spectrum's negative fluxes over its integrated absolute flux, on its wavelengths."""
    return float(np.trapezoid(np.abs(np.minimum(fx, 0.0)), wl) / np.trapezoid(np.abs(fx), wl))


def _range_points(rg):
    """The wavelength points (Angstrom) of an MSG range group."""
    kind = _text(rg.attrs["TYPE"])
    if kind == "lin_range_t":
        return float(rg.attrs["x_0"]) + float(rg.attrs["dx"]) * np.arange(int(rg.attrs["n"]))
    if kind == "log_range_t":
        return np.exp(float(rg.attrs["logx_0"]) + float(rg.attrs["dlogx"]) * np.arange(int(rg.attrs["n"])))
    if kind == "tab_range_t":
        return np.asarray(rg["x"][()], dtype=float).ravel()
    if kind == "comp_range_t":
        return np.concatenate([_range_points(rg[f"ranges[{j}]"]) for j in range(1, int(rg.attrs["n_ranges"]) + 1)])
    raise ValueError(f"wavelength range type {kind!r}; MSG writes lin_range_t, log_range_t, tab_range_t and "
                     "comp_range_t")


def _spectrum(spec_g):
    """Bin-centre wavelengths (Angstrom) and flux (erg/cm2/s/A) of an MSG limb_specint group."""
    edges = _range_points(spec_g["range"])
    law = _text(spec_g["limb"].attrs["law"])
    if law not in _FLUX_BASIS:
        raise ValueError(f"limb-darkening law {law!r}; MSG has {', '.join(_FLUX_BASIS)}")
    basis = 2 * np.pi * np.array(_FLUX_BASIS[law])
    c = np.asarray(spec_g["c"][()], dtype=float)
    if c.shape != (edges.size - 1, basis.size):
        raise ValueError(f"c has shape {c.shape}, not ({edges.size - 1}, {basis.size}): one row per bin between "
                         f"the {edges.size} wavelength points, one column per {law} coefficient")
    return 0.5 * (edges[:-1] + edges[1:]), c @ basis


# MSG's vgrid group, as MSG itself reads it (src/vgrid/vgrid_io_sm.fypp, github.com/rhdtownsend/msg):
# - revision 1 names its axes 'axes(r)', revisions 2 and 3 'axes[r]', for r = 1, ..., rank;
# - revisions 1 and 2 mark the vertices that hold a spectrum in a 'mask' over all vertices, revision 3 lists their
#   linear indices in 'v_lin_seq'; the n-th of them holds spectrum specints[n] (specgrid_io_sm.fypp);
# - a linear index is 1-based, with the first axis varying fastest (indexer_index_sm.fypp): Fortran order.
_AXIS_GROUP = {1: "axes({})", 2: "axes[{}]", 3: "axes[{}]"}
_ROLES = (("teff", find_teff_column), ("logg", find_logg_column), ("meta", find_metallicity_column))


def _vertex_parameters(vgrid):
    """The axis labels of an MSG vgrid group, and {parameter: value at each vertex holding a spectrum, in spectrum
    order}: teff, logg and meta for the axes whose labels read as those lookup columns (find_teff_column and the
    rest), then every other axis under its own label ('[alpha/Fe]', 'C/O', 'log(xi)', ...).

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
    if "" in labels:
        raise ValueError(f"axis {labels.index('') + 1} has no label")
    index = {label: j for j, label in enumerate(labels)}
    roles = {role: index[label] for role, find in _ROLES if (label := find(index)) is not None}
    others = {label: j for label, j in index.items() if j not in roles.values()}
    return labels, {name: axes[j][vertex[j]] for name, j in {**roles, **others}.items()}


def _check_parameters(h5_path, numbers, labels, values):
    """Raise RuntimeError unless the spectra specints[1..n] (*numbers*: their numbers) each have a finite Teff and
    logg, and a finite value of every other axis the grid has (*values*, from _vertex_parameters)."""
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
        """Download the grid's HDF5 file if needed and list its spectra, each with its parameters: teff, logg,
        meta where the grid has a metallicity axis, and every other axis under its label.

        Raises RuntimeError if the grid was not found by discover_models() on
        this grabber, or the file cannot be downloaded, read, or holds no
        spectra, or its spectra lack a finite Teff or logg, or a finite value
        of any other axis the grid has.
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

            def download(part):
                with self.session.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(part, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1<<20):
                            f.write(chunk)
            try:
                write_atomically(h5_path, download)
            except requests.RequestException as e:
                raise RuntimeError(f"MSG: could not download {url} to {h5_path}: {e}") from e

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
        return [{"fid": n, "gname": f"{spec_root}/{nm}", "params": {key: float(v[n - 1]) for key, v in values.items()}}
                for n, nm in enumerate(names, 1)]

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
                f"MSG: {len(failures)} of {len(spectra_info)} spectra of {h5_path} could not be "
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
        params = spec["params"]  # from get_model_metadata
        fname = f"{model_name}_fid{fid}.txt"
        fpath = os.path.join(out_dir, fname)

        # Written whole or not at all (write_atomically); a smaller file is a partial one an earlier version left
        if os.path.exists(fpath) and os.path.getsize(fpath) > 1024:
            return ("cached", fname, params)

        if not all(_in_range(params.get(key, np.nan), ranges[key]) for key in ("teff", "logg", "meta")):
            return ("excluded", fname, None)

        try:
            with h5py.File(h5_path, "r") as f:
                wl, fx = _spectrum(f[gpath])
        except (KeyError, OSError, ValueError) as e:
            return ("failed", fname, f"{gpath}: {e}")
        # Fluxes may be 0: MSG's cool models are exactly 0 in the far ultraviolet (BT-Settl, Goettingen-HiRes)
        checks = (("wavelengths are not finite and positive", np.isfinite(wl) & (wl > 0)),
                  ("fluxes are not finite", np.isfinite(fx)))
        bad = [f"{np.count_nonzero(~ok)} of {ok.size} {what}" for what, ok in checks if not ok.all()]
        bad += [] if np.any(fx > 0) else ["no flux is positive"]
        negative = _negative_fraction(wl, fx) if not bad and np.any(fx < 0) else None
        if negative is not None and not negative < _NEGATIVE_FLUX_LIMIT:
            bad.append(f"its negative fluxes are {negative:.3g} of its absolute flux (limit {_NEGATIVE_FLUX_LIMIT:g})")
        if bad:
            return ("failed", fname, f"{gpath}: " + "; ".join(bad))

        if ranges["wl"] is not None:
            wl_lo, wl_hi = ranges["wl"]
            mask = (wl >= wl_lo) & (wl <= wl_hi)
            wl, fx = wl[mask], fx[mask]
            if wl.size == 0:
                return ("excluded", fname, None)

        def write(part):
            with open(part, "w", encoding="utf-8") as fh:
                fh.write("# source = MSG HDF5\n")
                fh.write(f"# spec_group = {gpath}\n")
                for key, value in params.items():
                    fh.write(f"# {key} = {value}\n")
                if negative is not None:
                    fh.write(f"# negative_flux_fraction = {negative:.3e}\n")
                fh.write("# wavelength_unit = Angstrom\n")
                fh.write("# flux_unit = erg/cm2/s/A\n")
                fh.write("# columns = wavelength_A flux\n")
                for x, y in zip(wl, fx):
                    fh.write(f"{float(x):.6f} {float(y):.8e}\n")
        write_atomically(fpath, write)

        return ("ok", fname, params)

    def _write_lookup(self, out_dir, rows):
        path = os.path.join(out_dir, "lookup_table.csv")
        # In file-name order, as the cleaner writes its tables: the extractions finish in any order
        columns = ["file_name", *dict.fromkeys(key for row in rows for key in row if key != "file_name")]
        write_lookup_csv(sorted(rows, key=lambda row: row["file_name"]), path,
                         columns=columns, comment=False, restval=float("nan"))
        print(f"    Lookup table saved: {path}")
