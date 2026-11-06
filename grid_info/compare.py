"""Where two stellar grids agree: Delta BC per band at the same (Teff, logg, [M/H]), from grid_info.photometry tables.

Grid A's points are the comparison points. Grid B's band fluxes are taken at a coinciding node, or else interpolated
multilinearly, in flux, inside one of B's rectilinear cells whose corners all hold spectra (never across a hole, never
outside B's axes). Only bands that cover both spectra fully (no weight outside either's wavelengths) are compared.
[M/H] is compared at its nominal value: each grid's own solar mixture is part of what is compared.

At equal Teff the bolometric magnitudes cancel, so Delta BC = BC_A - BC_B = m_B - m_A (any one photometric system).

Per instrument it writes <A>__<B>/<Facility>/<Instrument>.json: per band Delta BC at each compared A point (null where
not compared), how B was obtained (0 coinciding node, 1 interpolated), and summary statistics. summary.json adds, for
the --summary bands, the worst |Delta BC| over those bands at each A point and its agreement class, and the leave-one-out
interpolation error of B along Teff (B's own spectra predicted from their Teff neighbours), the noise floor of the
interpolated comparisons.

Usage: python -m grid_info.compare TABLES OUT A[:variant] B[:variant] --instruments F/I ... [--summary F/I:b1,b2 ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

THRESHOLDS = (0.01, 0.03, 0.1)
COINCIDE = 1e-6


def as_value(text):
    try:
        return float(text)
    except ValueError:
        return text


class Table:
    """One grid's photometry tables, restricted to one variant."""

    def __init__(self, tables: Path, spec: str):
        name, _, variant = spec.partition(":")
        self.dir = tables / name
        self.doc = json.loads((self.dir / "points.json").read_text())
        variants = self.doc["variants"]
        if "=" in variant:     # by values, e.g. bt-cond:alpha=0 (variants without a manifest name)
            want = {k: as_value(w) for k, w in (kv.split("=") for kv in variant.split(","))}
            chosen = [v for v in variants if all(v["params"].get(k) == w for k, w in want.items())]
        elif variant:
            chosen = [v for v in variants if v["name"] == variant]
        elif len(variants) == 1:
            chosen = variants
        else:
            chosen = [v for v in variants if v["name"] and v["name"] == self.doc["default_variant"]]
        if len(chosen) != 1:
            raise ValueError(f"{name}: choose one variant, as {name}:<variant name> or {name}:<axis>=<value>,...; "
                             f"it has {[v['name'] or v['params'] for v in variants]}")
        self.name, self.variant = name, chosen[0]
        self.label = name if len(variants) == 1 else f"{name}:{chosen[0]['name'] or ','.join(f'{k}={v}' for k, v in chosen[0]['params'].items())}"
        self.axes = list(self.variant["axes"])
        self.grid_axes = [np.asarray(v, dtype=float) for v in self.variant["axes"].values()]
        self.nodes = np.array([-1 if j is None else j for j in self.variant["nodes"]]).reshape(
            [len(v) for v in self.grid_axes])
        self.members = self.nodes[self.nodes >= 0]
        self.params = {a: np.asarray(self.doc["points"][a], dtype=float) for a in self.axes}
        self.empty = np.asarray(self.doc["points"]["empty"])
        self._cache = {}

    def instrument(self, inst):
        if inst not in self._cache:
            self._cache[inst] = json.loads((self.dir / f"{inst}.json").read_text())["bands"]
        return self._cache[inst]

    def band(self, inst, band):
        """(flux relative to ST, usable mask) per point: usable = a positive flux and the band fully covered."""
        entry = self.instrument(inst)[band]
        if "error" in entry:
            return None, None
        m = np.array([np.nan if x is None else x for x in entry["m_st"]])
        outside = np.zeros_like(m) if entry["outside"] is None else np.asarray(entry["outside"], dtype=float)
        return 10 ** (-0.4 * m), np.isfinite(m) & (outside == 0) & ~self.empty


def corners(table: Table, x):
    """(point indices, weights) to interpolate table at parameters x, or None outside its axes or next to a hole."""
    per_axis = []
    for ax, v in zip(table.grid_axes, x):
        if not ax[0] - COINCIDE <= v <= ax[-1] + COINCIDE:
            return None
        hit = np.flatnonzero(np.abs(ax - v) <= COINCIDE)
        if hit.size:
            per_axis.append([(int(hit[0]), 1.0)])
            continue
        i = int(np.searchsorted(ax, v)) - 1
        t = (v - ax[i]) / (ax[i + 1] - ax[i])
        per_axis.append([(i, 1 - t), (i + 1, t)])
    idx, wts = [], []
    for combo in itertools.product(*per_axis):
        j = table.nodes[tuple(c[0] for c in combo)]
        if j < 0:
            return None
        idx.append(j)
        wts.append(np.prod([c[1] for c in combo]))
    return np.array(idx), np.array(wts)


def stats(d, a: Table, pts):
    ok = np.isfinite(d)
    if not ok.any():
        return None
    k = int(np.nanargmax(np.abs(d)))
    return {"n": int(ok.sum()), "median_abs": round(float(np.median(np.abs(d[ok]))), 4),
            "p95_abs": round(float(np.percentile(np.abs(d[ok]), 95)), 4),
            "worst": {"dBC": round(float(d[k]), 4), **{ax: float(a.params[ax][pts[k]]) for ax in a.axes}}}


def compare(tables: Path, out: Path, spec_a: str, spec_b: str, instruments, summary):
    a, b = Table(tables, spec_a), Table(tables, spec_b)
    missing = [ax for ax in b.axes if ax not in a.axes]
    if missing:
        raise ValueError(f"{b.label} has axes {missing} that {a.label} lacks: put the grid with more axes first")
    pts = a.members
    stencils = [corners(b, [a.params[ax][p] for ax in b.axes]) for p in pts]
    mode = np.array([-1 if s is None else (0 if len(s[0]) == 1 else 1) for s in stencils])
    pair = out / f"{a.label}__{b.label}".replace(":", "~")
    worst_summary = np.zeros(pts.size)
    report = {"A": a.label, "B": b.label, "points_A": int(pts.size),
              "B_at_A_points": {"coinciding": int((mode == 0).sum()), "interpolated": int((mode == 1).sum()),
                                "not_covered": int((mode < 0).sum())}, "bands": {}}
    for inst in instruments:
        out_bands = {}
        for band in a.instrument(inst):
            fa, ua = a.band(inst, band)
            fb, ub = b.band(inst, band)
            if fa is None or fb is None:
                continue
            d = np.full(pts.size, np.nan)
            for k, (p, s) in enumerate(zip(pts, stencils)):
                if s is None or not ua[p] or not ub[s[0]].all():
                    continue
                d[k] = 2.5 * np.log10(fa[p] / (s[1] @ fb[s[0]]))   # BC_A - BC_B = m_B - m_A
            out_bands[band] = {"dBC": [None if not np.isfinite(x) else round(float(x), 4) for x in d],
                               "coinciding": stats(np.where(mode == 0, d, np.nan), a, pts),
                               "interpolated": stats(np.where(mode == 1, d, np.nan), a, pts)}
            report["bands"][f"{inst}:{band}"] = {k: out_bands[band][k] for k in ("coinciding", "interpolated")}
            if band in summary.get(inst, ()):
                both = np.isfinite(d)
                worst_summary[both] = np.maximum(worst_summary[both], np.abs(d[both]))
        path = pair / f"{inst}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"A": a.label, "B": b.label, "instrument": inst,
                                    "points": {ax: a.params[ax][pts].tolist() for ax in a.axes},
                                    "B_mode": mode.tolist(), "bands": out_bands}, separators=(",", ":")) + "\n")
    if not summary:
        report["summary"] = None
        (pair / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
        return report
    # The map keeps a point only where every summary band was compared there.
    complete = np.ones(pts.size, dtype=bool)
    for inst, names in summary.items():
        for band in names:
            entry = report["bands"].get(f"{inst}:{band}")
            if entry is None:
                raise ValueError(f"summary band {inst}:{band} was not compared: add {inst} to --instruments")
            fa, ua = a.band(inst, band)
            fb, ub = b.band(inst, band)
            for k, (p, s) in enumerate(zip(pts, stencils)):
                if s is None or not ua[p] or not ub[s[0]].all():
                    complete[k] = False
    cls = np.digitize(worst_summary, THRESHOLDS, right=True)
    report["summary"] = {
        "bands": {k: list(v) for k, v in summary.items()},
        "classes": ["<=0.01", "<=0.03", "<=0.1", ">0.1"],
        "counts": {mode_name: np.bincount(cls[complete & (mode == m)], minlength=4).tolist()
                   for mode_name, m in (("coinciding", 0), ("interpolated", 1))},
        "map": [{**{ax: float(a.params[ax][p]) for ax in a.axes}, "worst_abs_dBC": round(float(w), 4),
                 "class": int(c), "B": "node" if m == 0 else "interpolated"}
                for p, w, c, m, ok in zip(pts, worst_summary, cls, mode, complete) if ok],
        "noise_floor_B_teff_loo": loo_teff(b, summary),
    }
    (pair / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def loo_teff(t: Table, summary):
    """|m interpolated from the Teff neighbours - m| at B's interior nodes, per summary band: median and 95th pct."""
    out = {}
    ti = t.axes.index("teff")
    for inst, names in summary.items():
        for band in names:
            f, u = t.band(inst, band)
            if f is None:
                continue
            errs = []
            for where in np.ndindex(*t.nodes.shape):
                i = where[ti]
                if i == 0 or i == t.nodes.shape[ti] - 1:
                    continue
                lo, hi = list(where), list(where)
                lo[ti], hi[ti] = i - 1, i + 1
                j, jl, jh = t.nodes[where], t.nodes[tuple(lo)], t.nodes[tuple(hi)]
                if min(j, jl, jh) < 0 or not (u[j] and u[jl] and u[jh]):
                    continue
                ax = t.grid_axes[ti]
                w = (ax[i] - ax[i - 1]) / (ax[i + 1] - ax[i - 1])
                errs.append(2.5 * np.log10(f[j] / ((1 - w) * f[jl] + w * f[jh])))
            if errs:
                e = np.abs(errs)
                out[f"{inst}:{band}"] = {"n": len(e), "median_abs": round(float(np.median(e)), 4),
                                         "p95_abs": round(float(np.percentile(e, 95)), 4)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tables", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--instruments", nargs="+", required=True)
    ap.add_argument("--summary", nargs="*", default=[], help="Facility/Instrument:band1,band2 ...")
    args = ap.parse_args(argv)
    summary = {s.split(":")[0]: s.split(":")[1].split(",") for s in args.summary}
    report = compare(args.tables, args.out, args.a, args.b, args.instruments, summary)
    print(json.dumps({k: report[k] for k in ("A", "B", "points_A", "B_at_A_points")}),
          json.dumps(report["summary"] and report["summary"]["counts"]))


if __name__ == "__main__":
    sys.exit(main())
