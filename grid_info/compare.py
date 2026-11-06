"""Where two stellar grids agree: Delta BC per band at the same (Teff, logg, [M/H]), from grid_info.photometry tables.

Grid A's points are the comparison points. Grid B's band fluxes are taken at a coinciding node, or else interpolated
multilinearly, in flux, inside one of B's rectilinear cells whose corners all hold spectra (never across a hole, never
outside B's axes). Only bands that cover both spectra fully (no weight outside either's wavelengths) are compared.
[M/H] is compared at its nominal value: each grid's own solar mixture is part of what is compared.

At equal Teff the bolometric magnitudes cancel, so Delta BC = BC_A - BC_B = m_B - m_A (any one photometric system).

It writes <A>__<B>/summary.json: per band (all instruments both grids have tables for) the statistics of Delta BC
at coinciding and at interpolated points; for the --summary bands, the worst |Delta BC| over those bands at each A point
and its agreement class; and the leave-one-out interpolation error of B along Teff (B's own spectra predicted from
their Teff neighbours), the noise floor of the interpolated comparisons. For the --detail instruments it also writes
<A>__<B>/<Facility>/<Instrument>.json with Delta BC at every A point. A pair with no overlap writes nothing.

Usage: python -m grid_info.compare TABLES OUT A[:variant] B[:variant] [--instruments F/I ...] [--detail F/I ...]
       [--summary F/I:b1,b2 ...]
A[:variant]: a variant by name, or grid:axis=value,... where a variant axis (alpha=0) picks the variant and a node
axis (metallicity=0) takes that slice of the grid and drops the axis (to compare grids whose metallicity axes are
different quantities only where they correspond).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

THRESHOLDS = (0.01, 0.03, 0.1)
CLASSES = ["<=0.01", "<=0.03", "<=0.1", ">0.1"]
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
        want = {k: as_value(w) for k, w in (kv.split("=") for kv in variant.split(","))} if "=" in variant else {}
        # Values of node axes (metallicity=0) take a slice of the grid; values of variant axes (alpha=0) pick a variant.
        slices = {k: w for k, w in want.items() if k in self.doc["axes"]}
        params = {k: w for k, w in want.items() if k not in slices}
        if params:
            chosen = [v for v in variants if all(v["params"].get(k) == w for k, w in params.items())]
        elif variant and not want:
            chosen = [v for v in variants if v["name"] == variant]
        elif len(variants) == 1:
            chosen = variants
        else:
            chosen = [v for v in variants if v["name"] and v["name"] == self.doc["default_variant"]]
        if len(chosen) != 1:
            raise ValueError(f"{name}: choose one variant, as {name}:<variant name> or {name}:<axis>=<value>,...; "
                             f"it has {[v['name'] or v['params'] for v in variants]}")
        self.name, self.variant = name, chosen[0]
        label = chosen[0]["name"] or ",".join(f"{k}={v}" for k, v in chosen[0]["params"].items())
        self.label = name if len(variants) == 1 else f"{name}:{label}"
        axes = {k: list(v) for k, v in self.variant["axes"].items()}
        nodes = np.array([-1 if j is None else j for j in self.variant["nodes"]]).reshape([len(v) for v in axes.values()])
        for axis, w in slices.items():           # keep one value of the axis and drop the axis
            if w not in axes[axis]:
                raise ValueError(f"{name}: {axis} = {w} is not a node value ({axes[axis]})")
            nodes = np.take(nodes, axes[axis].index(w), axis=list(axes).index(axis))
            del axes[axis]
        if slices:
            self.label += ":" + ",".join(f"{k}={w:g}" for k, w in slices.items())
        self.axes = list(axes)
        self.grid_axes = [np.asarray(v, dtype=float) for v in axes.values()]
        self.nodes = nodes
        self.members = self.nodes[self.nodes >= 0]
        self.params = {a: np.asarray(self.doc["points"][a], dtype=float) for a in self.axes}
        self.empty = np.asarray(self.doc["points"]["empty"])
        self.instruments = {str(p.relative_to(self.dir).with_suffix("")) for p in self.dir.glob("*/*.json")}

    def instrument(self, inst):
        return json.loads((self.dir / f"{inst}.json").read_text())["bands"]

    def band(self, entry):
        """(flux relative to ST, usable mask) per point: usable = a positive flux and the band fully covered."""
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
    return idx, wts


def stats(d, a: Table, pts):
    ok = np.isfinite(d)
    if not ok.any():
        return None
    k = int(np.nanargmax(np.abs(d)))
    return {"n": int(ok.sum()), "median_abs": round(float(np.median(np.abs(d[ok]))), 4),
            "p95_abs": round(float(np.percentile(np.abs(d[ok]), 95)), 4),
            "worst": {"dBC": round(float(d[k]), 4), **{ax: float(a.params[ax][pts[k]]) for ax in a.axes}}}


def compare(tables: Path, out: Path, spec_a: str, spec_b: str, instruments=None, detail=(), summary=None):
    a, b = Table(tables, spec_a), Table(tables, spec_b)
    missing = [ax for ax in b.axes if ax not in a.axes]
    if missing:
        raise ValueError(f"{b.label} has axes {missing} that {a.label} lacks: put the grid with more axes first")
    summary = summary or {}
    instruments = sorted(instruments or (a.instruments & b.instruments))
    for inst in [*detail, *summary]:
        if inst not in instruments:
            raise ValueError(f"{inst} is not among the compared instruments")
    pts = a.members
    stencils = [corners(b, [a.params[ax][p] for ax in b.axes]) for p in pts]
    covered = np.array([s is not None for s in stencils])
    if not covered.any():
        print(f"no overlap: {a.label} and {b.label}")
        return None
    # Stencils as fixed-width arrays; padding has weight 0 and does not count.
    width = 2 ** len(b.axes)
    idx = np.zeros((pts.size, width), dtype=int)
    wts = np.zeros((pts.size, width))
    for k, s in enumerate(stencils):
        if s is not None:
            idx[k, :len(s[0])], wts[k, :len(s[1])] = s
    mode = np.where(~covered, -1, np.where((wts > 0).sum(1) == 1, 0, 1))
    pair = out / f"{a.label}__{b.label}".replace(":", "~")
    worst = np.zeros(pts.size)
    complete = np.ones(pts.size, dtype=bool)
    seen = set()
    report = {"A": a.label, "B": b.label, "points_A": int(pts.size),
              "B_at_A_points": {"coinciding": int((mode == 0).sum()), "interpolated": int((mode == 1).sum()),
                                "not_covered": int((mode < 0).sum())}, "instruments": len(instruments), "bands": {}}
    for inst in instruments:
        bands_a, bands_b = a.instrument(inst), b.instrument(inst)
        detailed = {}
        for band, entry in bands_a.items():
            if "error" in entry or "error" in bands_b.get(band, {"error": ""}):
                continue
            fa, ua = a.band(entry)
            fb, ub = b.band(bands_b[band])
            ok = covered & ua[pts] & np.all(ub[idx] | (wts == 0), axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(ok, 2.5 * np.log10(fa[pts] / (wts * fb[idx]).sum(1)), np.nan)   # m_B - m_A
            entry_stats = {"coinciding": stats(np.where(mode == 0, d, np.nan), a, pts),
                           "interpolated": stats(np.where(mode == 1, d, np.nan), a, pts)}
            report["bands"][f"{inst}:{band}"] = entry_stats
            if inst in detail:
                detailed[band] = {"dBC": [None if not np.isfinite(x) else round(float(x), 4) for x in d],
                                  **entry_stats}
            if band in summary.get(inst, ()):
                seen.add(f"{inst}:{band}")
                complete &= ok
                worst = np.where(ok, np.maximum(worst, np.abs(np.nan_to_num(d))), worst)
        if inst in detail:
            path = pair / f"{inst}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"A": a.label, "B": b.label, "instrument": inst,
                                        "points": {ax: a.params[ax][pts].tolist() for ax in a.axes},
                                        "B_mode": mode.tolist(), "bands": detailed}, separators=(",", ":")) + "\n")
    unseen = sorted({f"{i}:{b_}" for i, names in summary.items() for b_ in names} - seen)
    if unseen:
        raise ValueError(f"summary bands not compared (no such band, or an unusable curve): {unseen}")
    report["summary"] = None if not summary else summarise(a, b, pts, worst, complete, mode, summary)
    pair.mkdir(parents=True, exist_ok=True)
    (pair / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def summarise(a, b, pts, worst, complete, mode, summary):
    cls = np.digitize(worst, THRESHOLDS, right=True)
    return {
        "bands": {k: list(v) for k, v in summary.items()},
        "classes": CLASSES,
        "counts": {name: np.bincount(cls[complete & (mode == m)], minlength=4).tolist()
                   for name, m in (("coinciding", 0), ("interpolated", 1))},
        "map": [{**{ax: float(a.params[ax][p]) for ax in a.axes}, "worst_abs_dBC": round(float(w), 4),
                 "class": int(c), "B": "node" if m == 0 else "interpolated"}
                for p, w, c, m, ok in zip(pts, worst, cls, mode, complete) if ok],
        "noise_floor_B_teff_loo": loo_teff(b, summary),
    }


def loo_teff(t: Table, summary):
    """|m interpolated from the Teff neighbours - m| at B's interior nodes, per summary band: median and 95th pct."""
    out = {}
    ti = t.axes.index("teff")
    ax = t.grid_axes[ti]
    if ax.size < 3:
        return out
    centre = np.moveaxis(t.nodes, ti, 0)
    j, jl, jh = centre[1:-1], centre[:-2], centre[2:]
    w = ((ax[1:-1] - ax[:-2]) / (ax[2:] - ax[:-2])).reshape((-1,) + (1,) * (t.nodes.ndim - 1))
    w = np.broadcast_to(w, j.shape)
    filled = (j >= 0) & (jl >= 0) & (jh >= 0)
    for inst, names in summary.items():
        bands = t.instrument(inst)
        for band in names:
            f, u = t.band(bands[band])
            ok = filled & u[j] & u[jl] & u[jh]
            e = np.abs(2.5 * np.log10(f[j[ok]] / ((1 - w[ok]) * f[jl[ok]] + w[ok] * f[jh[ok]])))
            if e.size:
                out[f"{inst}:{band}"] = {"n": int(e.size), "median_abs": round(float(np.median(e)), 4),
                                         "p95_abs": round(float(np.percentile(e, 95)), 4)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tables", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--instruments", nargs="+", help="Facility/Instrument (default: all both grids have)")
    ap.add_argument("--detail", nargs="*", default=[], help="instruments that get per-point Delta BC files")
    ap.add_argument("--summary", nargs="*", default=[], help="Facility/Instrument:band1,band2 ...")
    args = ap.parse_args(argv)
    summary = {s.split(":")[0]: s.split(":")[1].split(",") for s in args.summary}
    report = compare(args.tables, args.out, args.a, args.b, args.instruments, args.detail, summary)
    if report:
        print(json.dumps({k: report[k] for k in ("A", "B", "points_A", "B_at_A_points", "instruments")}),
              json.dumps(report["summary"] and report["summary"]["counts"]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
