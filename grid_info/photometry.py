"""Synthetic photometry of NJM's stellar grids in NJM's filter instruments (server-only).

For each grid, ``points.json`` lists its spectra (parameters, variant, bolometric magnitude, wavelength range) and,
per variant, the rectilinear node layout with its holes. For each grid and instrument, ``<Facility>/<Instrument>.json``
gives every spectrum's band-averaged surface flux as an ST magnitude, rounded to 1e-4 mag, with each band's AB and Vega
offsets, so the AB, Vega and ST magnitudes and the bolometric corrections follow by addition (see DEFINITIONS).

Band averages are SED_Tools' ``band_average_flux_lambda``: the spectrum interpolated onto the curve's own samples (zero
outside its range), weighted by ``filter_weight`` (T, or T*lambda for photon counting) and integrated by the trapezoidal
rule. That is linear in the spectrum, so it is written here as one sparse matrix per distinct wavelength grid among a
grid's spectra; one product then applies every band to every spectrum. ``check`` asserts that the product equals
``band_average_flux_lambda`` and ``_band_magnitude`` on random (spectrum, band) pairs before anything is written.

Usage: python -m grid_info.photometry DATA_DIR OUT_DIR [--grids G ...] [--instruments Facility/Instrument ...]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from grid_info.check_grids import BOLO_FAIL, BOLO_WARN
from sed_tools.filters.properties import ST_FLAM, filter_properties, load_vega
from sed_tools.photometry import (AB_ZERO_FLUX, SPEED_OF_LIGHT_ANG_PER_S, Spectrum, _band_magnitude,
                                  band_average_flux_lambda, filter_weight, integrate_flux, load_filter_curve)

SCHEMA = 1
SIGMA_SB = 5.670374419e-5                      # erg s^-1 cm^-2 K^-4 (CODATA 2018, exact in SI)
L0_IAU = 3.0128e35                              # erg/s: M_bol = 0 (IAU 2015 Resolution B2)
PC_CM = 3.0856775814913673e18
F0_BOL = L0_IAU / (4 * math.pi * (10 * PC_CM) ** 2)   # bolometric flux at 10 pc of M_bol = 0, 2.518e-5 erg/s/cm2
DECIMALS = 4                                    # 1e-4 mag
CHECK_PAIRS = 300                               # random (spectrum, band) pairs checked against the reference
CHECK_TOL = 1e-6                                # mag

DEFINITIONS = {
    "m_st": "ST magnitude of the band-averaged surface flux F (erg/s/cm2/A): -2.5 log10(F / 3.631e-9); "
            "null where F <= 0 (an empty point or a band wholly outside the spectrum)",
    "flux": "F = 3.631e-9 * 10**(-0.4 * m_st), the band average of sed_tools.photometry.band_average_flux_lambda",
    "AB": "m_AB = m_st + AB_minus_ST (AB_minus_ST from the band's pivot wavelength, exact)",
    "Vega": "m_Vega = m_st + Vega_minus_ST, with Vega = 0 in every band (CALSPEC alpha_lyr_mod_004 as "
            "data/vega/vega_flam.csv; no V_Vega = 0.03 offset, so Vega-system BCs are 0.03 above that convention's); "
            "Vega_minus_ST is null where the Vega spectrum does not cover the band, as in properties.json",
    "m_bol": "surface bolometric magnitude -2.5 log10(sigma Teff^4 / f0), f0 = L0 / (4 pi (10 pc)^2), "
             "L0 = 3.0128e35 erg/s (IAU 2015 B2, M_bol,sun = 4.74)",
    "BC": "BC_X = m_bol - m_X in system X (AB, Vega or ST): M_bol - M_X with the radius cancelled",
    "outside": "fraction of the band's weight integral (trapezoidal, on the curve's samples) lying outside the "
               "spectrum's tabulated wavelengths, where photometry counts zero flux; null when 0 for every point",
    "fbol_ratio": "integral of the tabulated spectrum / (sigma Teff^4): the guard's flux-conservation check",
    "fbol_flag": f"'fail' where fbol_ratio is outside {list(BOLO_FAIL)}, 'warn' outside {list(BOLO_WARN)} "
                 "(grid_info.check_grids' thresholds), else null",
}


def git_revision() -> str:
    here = Path(__file__).resolve().parent
    try:
        return subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def value(v):
    """A parameter value as a float when it reads as one, else a stripped string."""
    if isinstance(v, bytes):
        v = v.decode()
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v).strip()


def json_number(x, decimals=DECIMALS):
    return round(float(x), decimals) if np.isfinite(x) else None


# ---------------------------------------------------------------------------------------------------- grids

class Grid:
    """A published grid's spectra, read from its .h5 bundle and grouped by wavelength grid."""

    def __init__(self, data_dir: Path, name: str):
        self.name = name
        self.dir = data_dir / "stellar_models" / name
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.axes = list(self.manifest.get("axes") or {})
        self.extra = [k for k, v in (self.manifest.get("extra_axes") or {}).items() if len(v) > 1]
        self.h5 = self.dir / f"{name}.h5"
        cols = {k: [] for k in ["file", *self.axes, *self.extra]}
        groups = {}                               # wavelength bytes -> (lambda, [point index])
        with h5py.File(self.h5, "r") as h5:
            for i, key in enumerate(sorted(h5["spectra"])):
                item = h5["spectra"][key]
                attrs = item.attrs
                cols["file"].append(key)
                for axis in self.axes:
                    attr = "feh" if axis == "metallicity" else axis
                    if attr not in attrs:
                        raise ValueError(f"{name}/{key} has no {attr!r} attribute, an axis of its manifest")
                    cols[axis].append(float(attrs[attr]))
                for axis in self.extra:
                    if f"raw:{axis}" not in attrs:
                        raise ValueError(f"{name}/{key} has no 'raw:{axis}' attribute, an extra axis of its manifest")
                    cols[axis].append(value(attrs[f"raw:{axis}"]))
                lam = item["lambda"][()]
                if not np.all(np.diff(lam) > 0):
                    raise ValueError(f"{name}/{key}: wavelengths are not strictly increasing")
                groups.setdefault(lam.tobytes(), (lam, []))[1].append(i)
        self.points = cols
        self.groups = list(groups.values())
        self.n = len(cols["file"])

    def fluxes(self, indices):
        with h5py.File(self.h5, "r") as h5:
            spectra = h5["spectra"]
            return np.stack([spectra[self.points["file"][i]]["flux"][()] for i in indices])

    def variants(self):
        """Per combination of the extra axes: its manifest name, values, node axes and node -> point index."""
        named = {tuple(value(v.get(a)) for a in self.extra): v["variant_name"]
                 for v in self.manifest.get("variants") or []}
        keys = list(zip(*(self.points[a] for a in self.extra))) if self.extra else [()] * self.n
        out = []
        for combo in sorted(set(keys), key=str):
            members = [i for i, k in enumerate(keys) if k == combo]
            axes = {a: sorted({self.points[a][i] for i in members}) for a in self.axes}
            shape = [len(v) for v in axes.values()]
            nodes = np.full(shape, -1, dtype=int)
            for i in members:
                where = tuple(axes[a].index(self.points[a][i]) for a in self.axes)
                if nodes[where] >= 0:
                    raise ValueError(f"{self.name}: {self.points['file'][nodes[where]]} and {self.points['file'][i]} "
                                     f"share the node {dict(zip(self.axes, (self.points[a][i] for a in self.axes)))}")
                nodes[where] = i
            out.append({"name": named.get(combo), "params": dict(zip(self.extra, combo)), "axes": axes,
                        "nodes": [None if j < 0 else int(j) for j in nodes.ravel()]})
        return out


# ---------------------------------------------------------------------------------------------------- bands

class Bands:
    """Every band of the chosen instruments: curves, zero points, and the samples of the band operator."""

    def __init__(self, data_dir: Path, instruments, vega):
        self.instruments, self.curves, self.meta, self.errors = {}, [], [], {}
        for inst in instruments:
            paths = sorted((data_dir / "filters" / inst).glob("*.dat"))
            if not paths:
                raise ValueError(f"filters/{inst} has no .dat curves")
            self.instruments[inst] = []
            for path in paths:
                curve = load_filter_curve(str(path))
                try:
                    props = filter_properties(curve.wavelength, curve.transmission, curve.detector_type, vega)
                except ValueError as exc:      # unusable curve: recorded in the table, never skipped silently
                    self.errors[(inst, path.stem)] = str(exc)
                    continue
                self.instruments[inst].append((path.stem, len(self.curves)))
                self.curves.append(curve)
                pivot, zp_vega = props["WavelengthPivot"], props["ZeroPointVega_flam"]
                self.meta.append({
                    "detector_type": curve.detector_type,
                    "pivot": pivot,
                    "AB_minus_ST": -2.5 * math.log10(pivot ** 2 / SPEED_OF_LIGHT_ANG_PER_S * ST_FLAM / AB_ZERO_FLUX),
                    "Vega_minus_ST": 2.5 * math.log10(zp_vega / ST_FLAM) if zp_vega else None,
                    "zp_vega": zp_vega,
                })
        # The operator's samples: per curve sample, its band, wavelength and weight * trapezoid coefficient / norm.
        rows, xs, cws = [], [], []
        for b, c in enumerate(self.curves):
            x = c.wavelength
            w = filter_weight(x, c.transmission, c.detector_type)
            coef = np.empty_like(x)
            coef[0], coef[-1] = (x[1] - x[0]) / 2, (x[-1] - x[-2]) / 2
            coef[1:-1] = (x[2:] - x[:-2]) / 2       # np.trapezoid(y, x) == coef @ y
            rows.append(np.full(x.size, b))
            xs.append(x)
            cws.append(coef * w / np.trapezoid(w, x))
        self.rows, self.x, self.cw = np.concatenate(rows), np.concatenate(xs), np.concatenate(cws)
        self.n = len(self.curves)

    def operator(self, lam):
        """Sparse A (bands x len(lam)): A @ flux is band_average_flux_lambda(lam, flux, curve) for every band."""
        n = lam.size
        inside = (self.x >= lam[0]) & (self.x <= lam[-1])     # np.interp(..., left=0, right=0)
        x, cw, rows = self.x[inside], self.cw[inside], self.rows[inside]
        i = np.clip(np.searchsorted(lam, x, side="right") - 1, 0, n - 2)
        t = (x - lam[i]) / (lam[i + 1] - lam[i])
        return csr_matrix((np.concatenate([cw * (1 - t), cw * t]),
                           (np.concatenate([rows, rows]), np.concatenate([i, i + 1]))), shape=(self.n, n))


# ---------------------------------------------------------------------------------------------------- photometry

def photometry(grid: Grid, bands: Bands, vega):
    """Band fluxes (points x bands), outside fractions (points x bands), and per-point bolometric quantities."""
    flux = np.empty((grid.n, bands.n))
    outside = np.empty((grid.n, bands.n))
    fbol = np.empty(grid.n)
    lam_range = np.empty((grid.n, 2))
    for lam, members in grid.groups:
        a = bands.operator(lam)
        spectra = grid.fluxes(members)
        flux[members] = (a @ spectra.T).T
        outside[members] = 1 - a @ np.ones(lam.size)
        fbol[members] = [integrate_flux(lam, s) for s in spectra]
        lam_range[members] = lam[0], lam[-1]
    vega_flux = bands.operator(vega[0]) @ vega[1]
    for b, m in enumerate(bands.meta):            # the operator reproduces the published Vega zero points
        if m["zp_vega"] is not None and abs(vega_flux[b] / m["zp_vega"] - 1) > 1e-9:
            raise AssertionError(f"band {b}: operator Vega flux {vega_flux[b]} != zero point {m['zp_vega']}")
    return flux, outside, fbol, lam_range


def check(grid: Grid, bands: Bands, flux, vega, rng):
    """Compare random (spectrum, band) pairs with band_average_flux_lambda and _band_magnitude; return the worst."""
    vega_spectrum = Spectrum(vega[0], vega[1], {})
    worst = {"flux_rel": 0.0, "AB": 0.0, "Vega": 0.0}
    pairs = list(zip(rng.integers(grid.n, size=CHECK_PAIRS), rng.integers(bands.n, size=CHECK_PAIRS)))
    with h5py.File(grid.h5, "r") as h5:
        for p, b in pairs:
            item = h5["spectra"][grid.points["file"][p]]
            lam, f = item["lambda"][()], item["flux"][()]
            curve, meta = bands.curves[b], bands.meta[b]
            ref = band_average_flux_lambda(lam, f, curve)
            if ref == 0 and flux[p, b] == 0:
                continue
            worst["flux_rel"] = max(worst["flux_rel"], abs(flux[p, b] / ref - 1))
            if ref <= 0:
                continue
            m_st = -2.5 * math.log10(flux[p, b] / ST_FLAM)
            worst["AB"] = max(worst["AB"], abs(_band_magnitude(lam, f, curve, "AB", None)[2]
                                               - (m_st + meta["AB_minus_ST"])))
            if meta["Vega_minus_ST"] is not None:
                worst["Vega"] = max(worst["Vega"], abs(_band_magnitude(lam, f, curve, "Vega", lambda: vega_spectrum)[2]
                                                       - (m_st + meta["Vega_minus_ST"])))
    if max(worst["AB"], worst["Vega"], worst["flux_rel"]) > CHECK_TOL:
        raise AssertionError(f"{grid.name}: matrix photometry differs from sed_tools.photometry: {worst}")
    return worst


# ---------------------------------------------------------------------------------------------------- output

def write_json(path: Path, doc) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, separators=(",", ":"), allow_nan=False) + "\n"
    path.write_text(text)
    return len(text)


def write_grid(out: Path, grid: Grid, bands: Bands, flux, outside, fbol, lam_range, stamp):
    teff = np.asarray(grid.points["teff"])
    m_bol = -2.5 * np.log10(SIGMA_SB * teff ** 4 / F0_BOL)
    empty = ~(fbol > 0)
    points = {k: v for k, v in grid.points.items()}
    ratio = fbol / (SIGMA_SB * teff ** 4)
    points.update({"m_bol": [json_number(x) for x in m_bol],
                   "fbol_ratio": [json_number(x, 5) for x in ratio],
                   "fbol_flag": [None if BOLO_WARN[0] <= r <= BOLO_WARN[1] else
                                 "warn" if BOLO_FAIL[0] <= r <= BOLO_FAIL[1] else "fail" for r in ratio],
                   "empty": [bool(e) for e in empty],
                   "lambda_min": lam_range[:, 0].tolist(), "lambda_max": lam_range[:, 1].tolist()})
    sizes = {"points.json": write_json(out / grid.name / "points.json", {
        "schema": SCHEMA, "grid": grid.name, **stamp, "manifest_generated": grid.manifest.get("generated"),
        "default_variant": grid.manifest.get("default_variant"), "axes": grid.axes, "extra_axes": grid.extra,
        "definitions": DEFINITIONS, "points": points, "variants": grid.variants()})}
    with np.errstate(divide="ignore", invalid="ignore"):
        m_st = -2.5 * np.log10(flux / ST_FLAM)
    for inst, members in bands.instruments.items():
        table = {}
        for band, b in members:
            meta = bands.meta[b]
            col = np.round(outside[:, b], DECIMALS)
            table[band] = {"detector_type": meta["detector_type"], "pivot": round(meta["pivot"], 3),
                           "AB_minus_ST": json_number(meta["AB_minus_ST"], 6),
                           "Vega_minus_ST": None if meta["Vega_minus_ST"] is None else
                           json_number(meta["Vega_minus_ST"], 6),
                           "m_st": [json_number(x) for x in m_st[:, b]],
                           "outside": [float(x) for x in col] if col.any() else None}
        for (i, band), why in bands.errors.items():
            if i == inst:
                table[band] = {"error": why}
        sizes[f"{inst}.json"] = write_json(out / grid.name / f"{inst}.json", {
            "schema": SCHEMA, "grid": grid.name, "instrument": inst, **stamp, "points": "points.json",
            "bands": dict(sorted(table.items()))})
    return sizes


def write_index(out: Path):
    """index.json of a tables folder: grid -> its points file and instrument tables, instrument -> grids."""
    grids, instruments = {}, {}
    for points in sorted(out.glob("*/points.json")):
        doc = json.loads(points.read_text())
        name = points.parent.name
        tables = sorted(str(p.relative_to(points.parent).with_suffix("")) for p in points.parent.glob("*/*.json"))
        grids[name] = {"points": f"{name}/points.json", "computed": doc["computed"], "code": doc["code"],
                       "spectra": len(doc["points"]["file"]), "instruments": tables}
        for inst in tables:
            instruments.setdefault(inst, []).append(name)
    return write_json(out / "index.json", {
        "schema": SCHEMA, "table": "<grid>/<Facility>/<Instrument>.json", "definitions": DEFINITIONS,
        "grids": grids, "instruments": dict(sorted(instruments.items()))})


def all_instruments(data_dir: Path):
    return sorted({str(p.parent.relative_to(data_dir / "filters")) for p in (data_dir / "filters").glob("*/*/*.dat")})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--grids", nargs="+", required=True)
    ap.add_argument("--instruments", nargs="+", help="Facility/Instrument folders (default: all)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    instruments = args.instruments or all_instruments(args.data_dir)
    stamp = {"computed": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "code": f"SED_Tools server grid_info/photometry.py @ {git_revision()}"}
    vega = load_vega(args.data_dir / "vega" / "vega_flam.csv")
    t0 = time.time()
    bands = Bands(args.data_dir, instruments, vega)
    print(f"{bands.n} bands in {len(instruments)} instruments ({bands.x.size} curve samples), "
          f"{len(bands.errors)} unusable, {time.time() - t0:.1f} s", flush=True)
    report = {}
    for name in args.grids:
        t = time.time()
        grid = Grid(args.data_dir, name)
        t_read = time.time() - t
        flux, outside, fbol, lam_range = photometry(grid, bands, vega)
        t_phot = time.time() - t - t_read
        worst = check(grid, bands, flux, vega, np.random.default_rng(args.seed))
        sizes = write_grid(args.out_dir, grid, bands, flux, outside, fbol, lam_range, stamp)
        report[name] = {"spectra": grid.n, "wavelength_grids": len(grid.groups), "bands": bands.n,
                        "seconds": {"index": round(t_read, 1), "photometry": round(t_phot, 1),
                                    "total": round(time.time() - t, 1)},
                        "check_worst": worst, "bytes": sum(sizes.values()),
                        "largest": max(sizes.items(), key=lambda kv: kv[1])}
        print(json.dumps({name: report[name]}), flush=True)
    write_index(args.out_dir)
    print(json.dumps({**stamp, "instruments": len(instruments), "report": report}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
