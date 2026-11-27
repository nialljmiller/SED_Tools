"""Repair the negative fluxes of a stellar grid's spectra by one generic rule (Nill, 27 Nov 2026).

An emergent flux cannot be negative. Each negative sample falls in one of two classes:
  numerical zero  |F| <= EPS * max|F| of the same spectrum, with EPS the float64 machine epsilon (2.2e-16): the value
                  is below the spectrum's own float64 resolution and indistinguishable from 0 in any sum over the
                  spectrum. It is set to exactly 0.
  significant     any larger negative. Its window is the contiguous run of samples around it that are negative or
                  differ by more than WINDOW_TOL from the reference, the linear interpolation at the same wavelength
                  between the model's two nearest neighbours along the repair axis. The window is replaced by the
                  reference. The repair axis is the first of --axes along which the model has a neighbour on each
                  side at the same values of the other axes, on the same wavelength grid, and with no significant
                  negative of its own. A model with no such axis is held (reported, not changed).
The original files are copied to backups/<UTC>-spectrum-repairs/ first, and every changed sample (file, wavelength,
published value, new value, class, neighbours and weights) is written to source_evidence/grids/<grid>/
spectrum_repairs.json. Only the flux column of the changed rows is rewritten; headers and other rows are kept as they
are. The grid's .h5 bundle, cubes and records are rebuilt afterwards by the caller (build_h5_bundle, grid_info.publish).

Usage (on NJM, from the server checkout, under the NJM lock):
  python3 -m grid_info.spectrum_repairs GRID --axes yhe logg teff [--dry-run]
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPS = float(np.finfo(np.float64).eps)
WINDOW_TOL = 0.01


def read_lookup(grid):
    rows = list(csv.reader(open(grid / "lookup_table.csv")))
    head = [h.strip().lstrip("#").strip() for h in rows[0]]
    return [dict(zip(head, (x.strip() for x in r))) for r in rows[1:] if r]


def read_spectrum(path):
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1]


def significant(flux):
    """Indices of negatives above the numerical-zero threshold, and of those at or below it."""
    scale = EPS * np.max(np.abs(flux))
    neg = np.flatnonzero(flux < 0)
    return neg[np.abs(flux[neg]) > scale], neg[np.abs(flux[neg]) <= scale]


def plan(grid, axes):
    """Every change the rule makes to the grid: {file: {"zero": [...], "windows": [...]}}, and the held models."""
    rows = read_lookup(grid)
    params = {r["file_name"]: tuple(float(r[a]) for a in axes) for r in rows}
    by_params = {v: k for k, v in params.items()}
    spectra = {f: read_spectrum(grid / f) for f in params}
    bad = {f: significant(fl)[0].size > 0 for f, (_, fl) in spectra.items()}
    changes, held = {}, []
    for f, (lam, flux) in sorted(spectra.items()):
        sig, tiny = significant(flux)
        entry = {}
        if tiny.size:
            entry["zero"] = [{"index": int(k), "wavelength": float(lam[k]), "published": float(flux[k]), "new": 0.0}
                             for k in tiny]
        if sig.size:
            ref = reference(f, params, by_params, spectra, bad, axes)
            if ref is None:
                held.append({"file": f, "params": dict(zip(axes, params[f])), "negatives": int(sig.size)})
            else:
                entry["windows"] = windows(lam, flux, sig, ref)
        if entry:
            changes[f] = entry
    return changes, held


def reference(f, params, by_params, spectra, bad, axes):
    """The interpolation between the model's two nearest valid neighbours along the first axis that has them."""
    lam = spectra[f][0]
    p = params[f]
    for i, axis in enumerate(axes):
        line = sorted(v for v in by_params if all(v[j] == p[j] for j in range(len(axes)) if j != i))
        lower = [v for v in line if v[i] < p[i] and usable(by_params[v], lam, spectra, bad)]
        upper = [v for v in line if v[i] > p[i] and usable(by_params[v], lam, spectra, bad)]
        if lower and upper:
            lo, hi = max(lower, key=lambda v: v[i]), min(upper, key=lambda v: v[i])
            w = (p[i] - lo[i]) / (hi[i] - lo[i])
            flux = (1 - w) * spectra[by_params[lo]][1] + w * spectra[by_params[hi]][1]
            return {"axis": axis, "value": p[i], "lower": [by_params[lo], lo[i], 1 - w],
                    "upper": [by_params[hi], hi[i], w], "flux": flux}
    return None


def usable(f, lam, spectra, bad):
    return not bad[f] and spectra[f][0].shape == lam.shape and np.array_equal(spectra[f][0], lam)


def windows(lam, flux, sig, ref):
    est = ref["flux"]
    off = lambda k: flux[k] < 0 or est[k] <= 0 or abs(flux[k] / est[k] - 1) > WINDOW_TOL
    out = []
    for k in sig:
        if out and out[-1][0] <= k <= out[-1][1]:
            continue
        a = b = int(k)
        while a > 0 and off(a - 1):
            a -= 1
        while b < lam.size - 1 and off(b + 1):
            b += 1
        out.append((a, b))
    return [{"axis": ref["axis"], "value": ref["value"],
             "neighbours": [{"file": n[0], "value": n[1], "weight": round(n[2], 12)} for n in (ref["lower"], ref["upper"])],
             "from": float(lam[a]), "to": float(lam[b]), "samples": b - a + 1,
             "changed": [{"index": k, "wavelength": float(lam[k]), "published": float(flux[k]), "new": float(est[k])}
                         for k in range(a, b + 1)]} for a, b in out]


def rewrite(path, new_values):
    """Rewrite only the flux token of the data rows whose index is in new_values."""
    lines, k = path.read_text().splitlines(keepends=True), -1
    for n, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        k += 1
        if k in new_values:
            wl = line.split()[0]
            lines[n] = f"{wl} {new_values[k]:.8e}\n"
    tmp = path.with_name(path.name + ".repair-tmp")
    tmp.write_text("".join(lines))
    shutil.copymode(path, tmp)
    os.replace(tmp, path)   # a new file: a hard link elsewhere keeps the published copy


def git_revision():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("grid")
    ap.add_argument("--axes", nargs="+", required=True, help="repair axes in order of preference")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    grid = args.data / "stellar_models" / args.grid
    changes, held = plan(grid, args.axes)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"grid": args.grid, "applied": None if args.dry_run else stamp,
              "code": f"SED_Tools server grid_info/spectrum_repairs.py @ {git_revision()}",
              "rule": {"numerical_zero": f"negative with |F| <= {EPS:.6e} * max|F| of the spectrum: set to 0",
                       "significant": f"window of samples negative or off by more than {WINDOW_TOL:.0%} from the "
                                      f"interpolation between the nearest valid neighbours along the first of "
                                      f"{args.axes} that has them: replaced by that interpolation"},
              "files": changes, "held": held}
    summary = {"files": len(changes),
               "zeroed": sum(len(c.get("zero", [])) for c in changes.values()),
               "windows": [(f, w["from"], w["to"], w["samples"], w["axis"]) for f, c in changes.items()
                           for w in c.get("windows", [])],
               "held": held}
    print(json.dumps(summary, indent=1))
    if args.dry_run:
        return
    backup = ROOT / "backups" / f"{stamp}-spectrum-repairs" / grid.relative_to(ROOT)
    backup.mkdir(parents=True, exist_ok=True)
    for f, c in changes.items():
        shutil.copy2(grid / f, backup / f)
        new = {z["index"]: z["new"] for z in c.get("zero", [])}
        new.update({s["index"]: s["new"] for w in c.get("windows", []) for s in w["changed"]})
        rewrite(grid / f, new)
    evidence = ROOT / "source_evidence" / "grids" / args.grid / "spectrum_repairs.json"
    evidence.write_text(json.dumps(record, indent=1) + "\n")
    print(f"backed up {len(changes)} files to {backup}; record {evidence}")


if __name__ == "__main__":
    main()
