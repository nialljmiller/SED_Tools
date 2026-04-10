#!/usr/bin/env python3
"""
audit_pipeline.py

Audits every installed stellar atmosphere model for:
  1. Unit standardization in raw .txt files
  2. Flux cube consistency with raw files (raw vs cube max relative error)
  3. Physical plausibility (bolometric integral vs sigma*T^4)
  4. Negative flux counts
  5. Wien peak position sanity
  6. Wavelength coverage

Run from SED_Tools root:
    python audit_pipeline.py

Output:
    audit_report.txt        — full per-model results
    audit_summary.png       — bolometric ratio bar chart per model
    wl_coverage.png         — wavelength range horizontal bars per model
    sed_comparison_hot.png  — overlaid SEDs near a hot Teff across all models
    sed_comparison_cold.png — overlaid SEDs near a cold Teff across all models
"""

import os
import sys
import struct
import traceback
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
SIGMA   = 5.670374419e-5   # erg s-1 cm-2 K-4
H       = 6.62607015e-27   # erg s
C_CGS   = 2.99792458e10    # cm/s
HC_ANG  = H * C_CGS * 1e8  # erg·Å (hc with λ in Å)
K_B     = 1.380649e-16     # erg/K
FLOOR   = 1e-300
N_SAMPLE = 30              # raw files to sample per model
BOL_TOL  = 0.30            # 30% tolerance on bolometric ratio
WIEN_TOL = 0.40            # 40% tolerance on Wien peak position ratio
RAW_CUBE_TOL = 0.10        # 10% max relative error between raw file and cube node

# Models where sigma*T^4 bolometric check is not applicable.
# grams: distance-calibrated AGB dust shells — teff is central star but flux
#        is whole system at standard distance, so sigma*T^4 is not applicable.
# bbody: analytic blackbody — integral vs sigma*T^4 comparison is circular.
BOL_CHECK_EXEMPT = {"grams_cgrid", "grams_ogrid", "bbody"}

# Models where Wien peak position check is not applicable.
# grams: dust shells have no photospheric Wien peak.
# Husfeld/tmap2: very hot NLTE grids can have strong line blanketing and
# model-tail behaviour that shifts the apparent F_lambda peak away from the
# ideal blackbody Wien location; this check is too strict for these families.
WIEN_CHECK_EXEMPT = {"grams_cgrid", "grams_ogrid", "Husfeld", "tmap2"}

# Below this Teff the continuum is dominated by molecular bands; the smoothed
# flux peak does not reliably locate the true Wien peak.
WIEN_BAND_DOMINATED_TEFF_MAX = 1500.0  # K

# Minimum fraction of non-zero cube nodes required to trust the cube bol check.
CUBE_MIN_NONZERO_FRAC = 0.5
# ---------------------------------------------------------------------------


def planck_flam(wl_ang, teff):
    """π*B_λ in erg/cm²/s/Å. wl_ang in Å, teff in K."""
    l   = wl_ang * 1e-8
    exp = np.minimum(HC_ANG / (wl_ang * K_B * teff), 700.0)
    return np.pi * (2 * H * C_CGS**2 / l**5) / (np.exp(exp) - 1.0) * 1e-8


def wien_peak_ang(teff):
    """Wien peak wavelength in Å."""
    return 2.897771955e7 / teff


def expected_bol_in_range(teff, wl_min, wl_max, n=2000):
    """Expected bolometric flux within [wl_min, wl_max] Å for blackbody at teff."""
    wl = np.linspace(max(wl_min, 1.0), wl_max, n)
    bb = planck_flam(wl, teff)
    bol_range = float(np.trapezoid(bb, wl))
    bol_total = SIGMA * teff**4
    fraction  = bol_range / bol_total if bol_total > 0 else 1.0
    return bol_range, fraction


def read_cube_header(cube_path):
    with open(cube_path, "rb") as f:
        nt, nl, nm, nw = struct.unpack("4i", f.read(16))
        teff = np.frombuffer(f.read(8*nt), dtype=np.float64)
        logg = np.frombuffer(f.read(8*nl), dtype=np.float64)
        meta = np.frombuffer(f.read(8*nm), dtype=np.float64)
        wl   = np.frombuffer(f.read(8*nw), dtype=np.float64)
    return teff, logg, meta, wl


def read_cube_node(cube_path, teff_grid, logg_grid, meta_grid, wl_grid,
                   i_t, i_l, i_m):
    nt, nl, nm, nw = len(teff_grid), len(logg_grid), len(meta_grid), len(wl_grid)
    header_bytes = 16 + 8*(nt + nl + nm + nw)
    # File is written (W,M,L,T) layout
    fl = np.empty(nw, dtype=np.float64)
    with open(cube_path, "rb") as f:
        for w in range(nw):
            offset = header_bytes + (w*(nm*nl*nt) + i_m*(nl*nt) + i_l*nt + i_t) * 8
            f.seek(offset)
            fl[w] = struct.unpack("d", f.read(8))[0]
    return fl


def parse_header_flags(filepath):
    flags = {
        "units_standardized": False,
        "wavelength_unit": None,
        "flux_unit": None,
        "teff": None,
        "logg": None,
        "meta": None,
    }
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip("#").strip().lower()
                v = v.strip()
                if k == "units_standardized":
                    flags["units_standardized"] = v.lower() in ("true", "1", "yes")
                elif k == "wavelength_unit":
                    flags["wavelength_unit"] = v
                elif k in ("flux_unit", "flux_units"):
                    flags["flux_unit"] = v
                elif k == "teff":
                    try:
                        flags["teff"] = float(v)
                    except ValueError:
                        pass
                elif k in ("logg", "log_g", "log g"):
                    try:
                        flags["logg"] = float(v)
                    except ValueError:
                        pass
                elif k in ("meta", "m/h", "[m/h]", "metallicity", "feh", "[fe/h]"):
                    try:
                        flags["meta"] = float(v)
                    except ValueError:
                        pass
    except Exception:
        pass
    return flags


def audit_model(model_dir):
    """Audit one model directory. Returns a result dict."""
    name = model_dir.name
    result = {
        "name": name,
        "model_dir": str(model_dir),
        "n_files": 0,
        "n_sampled": 0,
        "all_standardized": True,
        "wrong_wl_unit": [],
        "wrong_fl_unit": [],
        "n_negative_raw": 0,
        "bol_ratios_raw": [],
        "bol_coverage_fracs": [],
        "cube_exists": False,
        "library_exists": False,
        "cube_size_gib": 0.0,
        "cube_grid": None,
        "cube_bol_ratios": [],
        "cube_neg_counts": [],
        "raw_vs_cube_max_err": None,
        "raw_vs_cube_teff": None,
        "sampled_files": [],
        "wl_range": None,
        "wien_deviations": [],
        "n_teff_headers": 0,
        # Collision detection
        "n_lookup_entries": 0,
        "n_unique_nodes": 0,
        "n_collision_nodes": 0,
        "max_files_per_node": 0,
        "collision_examples": [],
        "errors": [],
    }

    # --- collision check via lookup_table.csv ---
    lookup_path = model_dir / "lookup_table.csv"
    if lookup_path.exists():
        try:
            import csv as _csv
            node_map = {}
            with open(lookup_path, encoding="utf-8", errors="ignore") as f:
                header_line = f.readline().lstrip("#").strip()
                cols = [c.strip().lower() for c in header_line.split(",")]
                file_col = next((i for i, c in enumerate(cols) if c in ("file_name", "filename", "file")), 0)
                teff_col = next((i for i, c in enumerate(cols) if "teff" in c), None)
                logg_col = next((i for i, c in enumerate(cols) if c in ("logg", "log_g")), None)
                meta_col = next((i for i, c in enumerate(cols) if c in ("meta", "metallicity", "feh", "[fe/h]", "[m/h]", "m/h")), None)
                for row in _csv.reader(f):
                    if not row or row[0].startswith("#"):
                        continue
                    try:
                        fname = row[file_col].strip() if file_col < len(row) else ""
                        teff = round(float(row[teff_col]), 2) if teff_col is not None and teff_col < len(row) else None
                        logg = round(float(row[logg_col]), 4) if logg_col is not None and logg_col < len(row) else None
                        meta = round(float(row[meta_col]), 4) if meta_col is not None and meta_col < len(row) else None
                        node_map.setdefault((teff, logg, meta), []).append(fname)
                    except (ValueError, IndexError):
                        continue
            result["n_lookup_entries"]  = sum(len(v) for v in node_map.values())
            result["n_unique_nodes"]    = len(node_map)
            collisions = {k: v for k, v in node_map.items() if len(v) > 1}
            result["n_collision_nodes"] = len(collisions)
            result["max_files_per_node"] = max((len(v) for v in node_map.values()), default=0)
            for key, files in sorted(collisions.items(), key=lambda x: -len(x[1]))[:3]:
                result["collision_examples"].append({
                    "node": key, "n_files": len(files), "examples": files[:4],
                })
        except Exception as e:
            result["errors"].append(f"Lookup collision check failed: {e}")

    # --- raw files ---
    txts = sorted(model_dir.glob("*.txt"))
    result["n_files"] = len(txts)
    if not txts:
        result["errors"].append("No .txt files found")
        return result

    step = max(1, len(txts) // N_SAMPLE)
    samples = txts[::step][:N_SAMPLE]
    result["n_sampled"] = len(samples)

    all_wl_mins = []
    all_wl_maxs = []

    for s in samples:
        flags = parse_header_flags(s)

        if not flags["units_standardized"]:
            result["all_standardized"] = False

        wlu = (flags["wavelength_unit"] or "").lower().replace(" ", "")
        if wlu and "angstrom" not in wlu and wlu not in ("a", "aa", "ang"):
            result["wrong_wl_unit"].append(f"{s.name}: {wlu}")

        flu = (flags["flux_unit"] or "").lower().replace(" ", "")
        ok_fl = any(x in flu for x in ("erg/cm2/s/a", "erg/cm²/s/å",
                                        "erg/cm2/s/ang", "flam"))
        if flu and not ok_fl:
            result["wrong_fl_unit"].append(f"{s.name}: {flu}")

        teff = flags["teff"]
        if teff and teff > 0:
            result["n_teff_headers"] += 1

        # Load and check
        try:
            raw = np.loadtxt(s, comments="#")
            if raw.ndim != 2 or raw.shape[1] < 2:
                continue
            wl, fl = raw[:, 0], raw[:, 1]

            all_wl_mins.append(float(wl.min()))
            all_wl_maxs.append(float(wl.max()))

            n_neg = int(np.sum(fl < 0))
            result["n_negative_raw"] += n_neg

            if teff and teff > 0:
                # Store for SED comparison plots and raw vs cube comparison
                result["sampled_files"].append((teff, flags["logg"], flags["meta"], s))

                if len(wl) > 10 and name not in BOL_CHECK_EXEMPT:
                    bol = float(np.trapezoid(np.maximum(fl, 0), wl))
                    expected_in_range, frac = expected_bol_in_range(
                        teff, float(wl.min()), float(wl.max())
                    )
                    result["bol_coverage_fracs"].append(frac)
                    if frac >= 0.05 and expected_in_range > 0:
                        result["bol_ratios_raw"].append(bol / expected_in_range)

                # Wien peak check — skip for exempt models and band-dominated Teffs
                wl_wien = wien_peak_ang(teff)
                fl_pos  = np.maximum(fl, 0)
                wien_exempt      = name in WIEN_CHECK_EXEMPT or teff < WIEN_BAND_DOMINATED_TEFF_MAX
                wien_in_range    = float(wl.min()) < wl_wien < float(wl.max())
                if not wien_exempt and fl_pos.max() > 0 and wien_in_range:
                    window    = max(1, len(wl) // 50)
                    fl_smooth = np.convolve(fl_pos, np.ones(window)/window, mode="same")
                    actual_peak_wl = float(wl[np.argmax(fl_smooth)])
                    result["wien_deviations"].append(actual_peak_wl / wl_wien)
                elif not wien_exempt and not wien_in_range and fl_pos.max() > 0:
                    result["wien_peak_oor_count"] = result.get("wien_peak_oor_count", 0) + 1

        except Exception as e:
            result["errors"].append(f"Load error {s.name}: {e}")

    if all_wl_mins:
        result["wl_range"] = (min(all_wl_mins), max(all_wl_maxs))

    # --- flux cube ---
    cube_path = model_dir / "flux_cube.bin"
    if not cube_path.exists():
        result["errors"].append("flux_cube.bin missing")
        return result

    result["cube_exists"] = True
    result["cube_size_gib"] = cube_path.stat().st_size / 1e9
    result["library_exists"] = (model_dir / "fluxcube_library").is_dir()

    try:
        teff_g, logg_g, meta_g, wl_g = read_cube_header(cube_path)
        result["cube_grid"] = {
            "nt": len(teff_g), "nl": len(logg_g),
            "nm": len(meta_g), "nw": len(wl_g),
            "teff": (float(teff_g[0]), float(teff_g[-1])),
            "logg": (float(logg_g[0]), float(logg_g[-1])),
            "meta": (float(meta_g[0]), float(meta_g[-1])),
            "wl":   (float(wl_g[0]),   float(wl_g[-1])),
        }
    except Exception as e:
        result["errors"].append(f"Cube header read failed: {e}")
        return result

    # --- raw vs cube comparison ---
    # For every sampled file, confirm that its (teff, logg, meta) snaps cleanly onto a
    # real cube grid point (within half the minimum grid spacing on each axis) before
    # comparing.  All clean-snapping files are used so the result is a distribution
    # rather than a single point, giving meaningful median and max across the grid.

    def _half_step(grid):
        if len(grid) < 2:
            return np.inf
        return float(np.min(np.diff(grid))) / 2.0

    tol_t = _half_step(teff_g)
    tol_l = _half_step(logg_g)
    tol_m = _half_step(meta_g)

    result["raw_vs_cube_snap_checked"] = 0
    result["raw_vs_cube_snap_skipped"] = 0
    # Per-file relative error summaries across wavelengths, for all clean-snapping files
    result["raw_vs_cube_node_max_errs"] = []

    # Deduplicate by cube node index so we don't read the same node multiple times
    # (many raw files share the same grid point).
    seen_nodes = set()

    for teff_raw, logg_raw, meta_raw, fpath in result["sampled_files"]:
        result["raw_vs_cube_snap_checked"] += 1

        i_t = int(np.argmin(np.abs(teff_g - teff_raw)))
        if abs(teff_g[i_t] - teff_raw) > tol_t:
            result["raw_vs_cube_snap_skipped"] += 1
            continue

        if logg_raw is not None:
            i_l = int(np.argmin(np.abs(logg_g - logg_raw)))
            if abs(logg_g[i_l] - logg_raw) > tol_l:
                result["raw_vs_cube_snap_skipped"] += 1
                continue
        else:
            i_l = len(logg_g) // 2

        if meta_raw is not None:
            i_m = int(np.argmin(np.abs(meta_g - meta_raw)))
            if abs(meta_g[i_m] - meta_raw) > tol_m:
                result["raw_vs_cube_snap_skipped"] += 1
                continue
        else:
            i_m = len(meta_g) // 2

        node_key = (i_t, i_l, i_m)
        if node_key in seen_nodes:
            # Already compared this cube node with a different raw file; skip duplicate.
            result["raw_vs_cube_snap_skipped"] += 1
            continue
        seen_nodes.add(node_key)

        try:
            raw = np.loadtxt(fpath, comments="#")
            if raw.ndim != 2 or raw.shape[1] < 2:
                continue
            wl_raw, fl_raw = raw[:, 0], raw[:, 1]
            fl_cube = read_cube_node(cube_path, teff_g, logg_g, meta_g, wl_g,
                                     i_t, i_l, i_m)
            if fl_cube.max() <= 0:
                result["raw_vs_cube_snap_skipped"] += 1
                seen_nodes.discard(node_key)
                continue
            wl_lo = max(float(wl_raw.min()), float(wl_g.min()))
            wl_hi = min(float(wl_raw.max()), float(wl_g.max()))
            if wl_hi <= wl_lo:
                continue
            mask_raw = (wl_raw >= wl_lo) & (wl_raw <= wl_hi)
            wl_r = wl_raw[mask_raw]
            fl_r = fl_raw[mask_raw]
            fl_c = np.interp(wl_r, wl_g, fl_cube)
            thresh = fl_r.max() * 0.01
            valid = fl_r > thresh
            if valid.sum() < 10:
                continue
            rel_err = np.abs(fl_r[valid] - fl_c[valid]) / (fl_r[valid] + FLOOR)
            result["raw_vs_cube_node_max_errs"].append(
                (float(teff_g[i_t]), float(logg_g[i_l]), float(meta_g[i_m]),
                 float(rel_err.max()), float(np.median(rel_err)),
                 float(np.percentile(rel_err, 99.0)))
            )
        except Exception as e:
            result["errors"].append(f"Raw vs cube comparison failed "
                                    f"(Teff={teff_g[i_t]:.0f}): {e}")

    # Summarise across all compared nodes
    if result["raw_vs_cube_node_max_errs"]:
        all_node_maxes  = [x[3] for x in result["raw_vs_cube_node_max_errs"]]
        all_node_medians = [x[4] for x in result["raw_vs_cube_node_max_errs"]]
        all_node_p99 = [x[5] for x in result["raw_vs_cube_node_max_errs"]]
        worst_idx = int(np.argmax(all_node_maxes))
        worst = result["raw_vs_cube_node_max_errs"][worst_idx]
        result["raw_vs_cube_max_err"]     = worst[3]
        result["raw_vs_cube_teff"]        = worst[0]
        result["raw_vs_cube_logg"]        = worst[1]
        result["raw_vs_cube_meta"]        = worst[2]
        result["raw_vs_cube_median_err"]  = float(np.median(all_node_maxes))
        result["raw_vs_cube_p99_err"]     = float(np.median(all_node_p99))
        result["raw_vs_cube_n_nodes"]     = len(all_node_maxes)
        # Use p99 per-node error as pass/fail criterion to avoid failing models on
        # isolated spikes (e.g., interpolation near very sharp spectral lines).
        result["raw_vs_cube_n_bad"]       = sum(1 for e in all_node_p99 if e > RAW_CUBE_TOL)

    # --- sample cube nodes ---
    n_cube_check = min(5, len(teff_g))
    t_indices = np.linspace(0, len(teff_g)-1, n_cube_check, dtype=int)
    i_l = len(logg_g) // 2
    i_m = len(meta_g) // 2

    for i_t in t_indices:
        teff = float(teff_g[i_t])
        try:
            fl_cube = read_cube_node(cube_path, teff_g, logg_g, meta_g, wl_g,
                                     i_t, i_l, i_m)
            n_neg = int(np.sum(fl_cube < 0))
            result["cube_neg_counts"].append(n_neg)

            bol = float(np.trapezoid(np.maximum(fl_cube, 0), wl_g))
            if teff > 0 and name not in BOL_CHECK_EXEMPT:
                expected_in_range, frac = expected_bol_in_range(
                    teff, float(wl_g[0]), float(wl_g[-1])
                )
                if frac >= 0.05 and expected_in_range > 0:
                    result["cube_bol_ratios"].append(bol / expected_in_range)
        except Exception as e:
            result["errors"].append(f"Cube node read failed Teff={teff:.0f}: {e}")

    return result


def format_report(results):
    lines = []
    lines.append("=" * 70)
    lines.append("SED_Tools Pipeline Audit Report")
    lines.append("=" * 70)

    pass_count = 0
    fail_count = 0

    for r in results:
        lines.append(f"\n{'─'*70}")
        lines.append(f"Model: {r['name']}")
        lines.append(f"  Files       : {r['n_files']} total, {r['n_sampled']} sampled "
                     f"({r['n_teff_headers']} with valid Teff headers)")

        # Collision report
        n_entries = r.get("n_lookup_entries", 0)
        n_nodes   = r.get("n_unique_nodes", 0)
        n_coll    = r.get("n_collision_nodes", 0)
        max_fpn   = r.get("max_files_per_node", 0)
        if n_entries > 0:
            lines.append(f"  Lookup      : {n_entries} entries -> {n_nodes} unique nodes  "
                         f"(max {max_fpn} files/node)")
            if n_coll > 0:
                if r.get("library_exists"):
                    lines.append(f"  Note: {n_coll} collision nodes — handled (fluxcube_library/ present, mean cube built)")
                    for ex in r.get("collision_examples", [])[:3]:
                        t, g, m = ex["node"]
                        lines.append(f"       Teff={t} logg={g} [M/H]={m} - "
                                     f"{ex['n_files']} files: {', '.join(ex['examples'])}")
                else:
                    lines.append(f"  *** {n_coll} NODES HAVE MULTIPLE FILES (last-write-wins in cube)")
                    for ex in r.get("collision_examples", []):
                        t, g, m = ex["node"]
                        lines.append(f"       Teff={t} logg={g} [M/H]={m} - "
                                     f"{ex['n_files']} files: {', '.join(ex['examples'])}")
                    ok = False

        ok = True

        if not r["all_standardized"]:
            lines.append("  *** UNITS NOT STANDARDIZED — run spectra_cleaner first")
            ok = False
        else:
            lines.append("  Units flag  : OK (units_standardized = True)")

        if r["wrong_wl_unit"]:
            lines.append(f"  *** WRONG WL UNIT in {len(r['wrong_wl_unit'])} files:")
            for x in r["wrong_wl_unit"][:3]:
                lines.append(f"       {x}")
            ok = False
        if r["wrong_fl_unit"]:
            lines.append(f"  *** WRONG FL UNIT in {len(r['wrong_fl_unit'])} files:")
            for x in r["wrong_fl_unit"][:3]:
                lines.append(f"       {x}")
            ok = False

        if r["wl_range"]:
            lines.append(f"  WL range    : {r['wl_range'][0]:.1f} – {r['wl_range'][1]:.1f} Å  "
                         f"(span = {r['wl_range'][1] - r['wl_range'][0]:.0f} Å)")

        if r["n_negative_raw"] > 0:
            lines.append(f"  Negative raw flux points: {r['n_negative_raw']} (across sampled files)")

        # Wien peak check
        if r["name"] in WIEN_CHECK_EXEMPT:
            lines.append("  Wien peak   : EXEMPT (non-photospheric model)")
        elif r["wien_deviations"]:
            wd = r["wien_deviations"]
            med_wd = float(np.median(wd))
            lines.append(f"  Wien peak   : median ratio={med_wd:.3f}  "
                         f"min={min(wd):.3f}  max={max(wd):.3f}  "
                         f"(n={len(wd)}, ideal=1.0)")
            if not (1 - WIEN_TOL < med_wd < 1 + WIEN_TOL):
                lines.append(f"  *** WIEN PEAK RATIO OUT OF RANGE [{1-WIEN_TOL:.1f}, {1+WIEN_TOL:.1f}] "
                             f"— possible unit error or truncated wl range")
                ok = False
        elif r.get("wien_peak_oor_count", 0) > 0:
            lines.append(f"  Wien peak   : skipped — peak wavelength below model's wl range "
                         f"({r['wien_peak_oor_count']} files; Teff too hot for coverage)")
        elif r.get("n_teff_headers", 0) > 0:
            lines.append(f"  Wien peak   : skipped — all spectra below "
                         f"{WIEN_BAND_DOMINATED_TEFF_MAX:.0f} K (band-dominated continuum)")

        # Bolometric check (raw)
        if r["name"] in BOL_CHECK_EXEMPT:
            lines.append("  Bolometric check: EXEMPT (distance-calibrated model, σT⁴ not applicable)")
        elif r["bol_ratios_raw"]:
            ratios = r["bol_ratios_raw"]
            med  = float(np.median(ratios))
            std  = float(np.std(ratios))
            frac = float(np.median(r.get("bol_coverage_fracs", [1.0])))
            lines.append(f"  Bol ratio (raw) : median={med:.4f}  std={std:.4f}  "
                         f"min={min(ratios):.4f}  max={max(ratios):.4f}  "
                         f"(wl coverage {frac*100:.0f}% of σT⁴)")
            if not (1 - BOL_TOL < med < 1 + BOL_TOL):
                lines.append(f"  *** BOL RATIO MEDIAN OUT OF RANGE [{1-BOL_TOL:.1f}, {1+BOL_TOL:.1f}]")
                ok = False
        elif r.get("bol_coverage_fracs"):
            frac = float(np.median(r["bol_coverage_fracs"]))
            lines.append(f"  Bolometric check skipped: wl coverage {frac*100:.1f}% of σT⁴ (<5% threshold)")

        if not r["cube_exists"]:
            lines.append("  *** flux_cube.bin MISSING")
            ok = False
        else:
            g = r["cube_grid"]
            lines.append(f"  Cube grid   : {g['nt']}×{g['nl']}×{g['nm']}×{g['nw']}  "
                         f"({r['cube_size_gib']:.3f} GiB)")
            lines.append(f"  Cube Teff   : {g['teff'][0]:.0f} – {g['teff'][1]:.0f} K")
            lines.append(f"  Cube log g  : {g['logg'][0]:.2f} – {g['logg'][1]:.2f}")
            lines.append(f"  Cube [M/H]  : {g['meta'][0]:.2f} – {g['meta'][1]:.2f}")
            lines.append(f"  Cube wl     : {g['wl'][0]:.1f} – {g['wl'][1]:.1f} Å")

            # Raw vs cube comparison
            n_checked = r.get("raw_vs_cube_snap_checked", 0)
            n_skipped = r.get("raw_vs_cube_snap_skipped", 0)
            n_nodes   = r.get("raw_vs_cube_n_nodes", 0)
            n_bad     = r.get("raw_vs_cube_n_bad", 0)
            if r["raw_vs_cube_max_err"] is not None:
                lines.append(
                    f"  Raw vs cube : {n_nodes} nodes compared  "
                    f"({n_checked - n_skipped} clean snaps / {n_checked} sampled)  "
                    f"{n_bad} nodes exceed {RAW_CUBE_TOL*100:.0f}% threshold"
                )
                lines.append(
                    f"               median node-max-err={r.get('raw_vs_cube_median_err', float('nan')):.4f}  "
                    f"median node-p99-err={r.get('raw_vs_cube_p99_err', float('nan')):.4f}  "
                    f"worst={r['raw_vs_cube_max_err']:.4f}  "
                    f"(node: Teff={r['raw_vs_cube_teff']:.0f} K  "
                    f"log g={r.get('raw_vs_cube_logg', float('nan')):.2f}  "
                    f"[M/H]={r.get('raw_vs_cube_meta', float('nan')):.2f})"
                )
                if n_bad > 0:
                    if r.get("library_exists"):
                        lines.append(f"  Note: RAW VS CUBE: {n_bad}/{n_nodes} nodes exceed "
                                     f"{RAW_CUBE_TOL*100:.0f}% (expected — mean cube, variants in fluxcube_library/)")
                    else:
                        lines.append(f"  *** RAW VS CUBE: {n_bad}/{n_nodes} NODES EXCEED {RAW_CUBE_TOL*100:.0f}% ERROR")
                        ok = False
            else:
                if n_skipped == n_checked and n_checked > 0:
                    lines.append(f"  Raw vs cube : skipped — all {n_checked} sampled files "
                                 f"land off-grid (gap-filled nodes only in sample)")
                else:
                    lines.append("  Raw vs cube : comparison skipped (no suitable node found)")

            if r["cube_neg_counts"]:
                total_neg = sum(r["cube_neg_counts"])
                lines.append(f"  Cube neg pts: {total_neg} across {len(r['cube_neg_counts'])} sampled nodes")
                if total_neg > 100:
                    lines.append("  *** HIGH NEGATIVE COUNT IN CUBE")
                    ok = False

            if r["cube_bol_ratios"]:
                ratios = r["cube_bol_ratios"]
                nonzero = [x for x in ratios if x > 0.01]
                n_zero = len(ratios) - len(nonzero)
                med = float(np.median(ratios))
                std = float(np.std(ratios)) if len(ratios) > 1 else 0.0
                lines.append(f"  Bol ratio (cube): median={med:.4f}  std={std:.4f}  "
                             f"min={min(ratios):.4f}  max={max(ratios):.4f}")
                if n_zero:
                    lines.append(f"  Note: {n_zero} zero-flux cube nodes (zero-filled grid edges, expected)")
                nonzero_frac = len(nonzero) / len(ratios) if ratios else 0
                if nonzero_frac < CUBE_MIN_NONZERO_FRAC:
                    lines.append(f"  Note: only {len(nonzero)}/{len(ratios)} sampled nodes populated "
                                 f"— sparse grid, cube bol check skipped")
                elif not (1 - BOL_TOL < med < 1 + BOL_TOL):
                    lines.append(f"  *** CUBE BOL RATIO MEDIAN OUT OF RANGE [{1-BOL_TOL:.1f}, {1+BOL_TOL:.1f}]")
                    ok = False
            elif r["name"] not in BOL_CHECK_EXEMPT and r["cube_exists"]:
                lines.append("  Bol ratio (cube): skipped (wl coverage <5% of σT⁴ for all nodes)")

        for e in r["errors"]:
            lines.append(f"  *** ERROR: {e}")
            ok = False

        status = "PASS" if ok else "FAIL"
        lines.append(f"  Status: {status}")
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    lines.append(f"\n{'='*70}")
    lines.append(f"SUMMARY: {pass_count} PASS / {fail_count} FAIL / {pass_count+fail_count} total")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _find_cube_for_file(model_dir, fname, teff, logg, meta):
    """
    Return the path to the best cube for this specific file.
    If a fluxcube_library/index.csv exists, find the variant whose
    extra-axis values match this file in the lookup table.
    Falls back to flux_cube.bin (mean cube) if anything is unclear.
    """
    import csv as _csv
    model_dir = Path(model_dir)
    library_dir = model_dir / "fluxcube_library"
    mean_cube   = model_dir / "flux_cube.bin"

    if not library_dir.is_dir():
        return mean_cube

    index_path = library_dir / "index.csv"
    if not index_path.exists():
        return mean_cube

    # Read lookup table to find this file's extra-axis values
    lookup_path = model_dir / "lookup_table.csv"
    if not lookup_path.exists():
        return mean_cube

    try:
        with open(lookup_path, encoding="utf-8", errors="ignore") as f:
            header = f.readline().lstrip("#").strip()
            cols   = [c.strip() for c in header.split(",")]
            reader = _csv.reader(f)
            file_row = None
            for row in reader:
                if row and row[0].strip().lstrip("#") == fname:
                    file_row = {cols[i]: row[i].strip() for i in range(min(len(cols), len(row)))}
                    break
        if file_row is None:
            return mean_cube

        # Load library index
        with open(index_path, encoding="utf-8", errors="ignore") as f:
            header2 = f.readline().lstrip("#").strip()
            icols   = [c.strip() for c in header2.split(",")]
            ireader = _csv.reader(f)
            index_rows = [
                {icols[i]: row[i].strip() for i in range(min(len(icols), len(row)))}
                for row in ireader if row and not row[0].strip().startswith("#")
            ]

        extra_cols = [c for c in icols if c not in (
            "cube_file", "n_spectra",
            "teff_min", "teff_max", "logg_min", "logg_max", "meta_min", "meta_max"
        )]

        # Find index row whose extra-axis values match this file
        for irow in index_rows:
            match = all(
                str(file_row.get(col, "")).strip().lower() ==
                str(irow.get(col, "")).strip().lower()
                for col in extra_cols
            )
            if match:
                variant_path = library_dir / irow["cube_file"]
                if variant_path.exists():
                    return variant_path

    except Exception:
        pass

    return mean_cube


def _extract_cube_sed(cube_path, teff, logg, meta):
    """
    Extract a single SED from a flux cube at the nearest (teff, logg, meta) node.
    Returns (wl_array, fl_array) or (None, None) on failure.
    """
    try:
        teff_g, logg_g, meta_g, wl_g = read_cube_header(cube_path)
        i_t = int(np.argmin(np.abs(teff_g - teff)))
        i_l = int(np.argmin(np.abs(logg_g - logg)))
        i_m = int(np.argmin(np.abs(meta_g - meta)))
        fl  = read_cube_node(cube_path, teff_g, logg_g, meta_g, wl_g, i_t, i_l, i_m)
        return wl_g, fl
    except Exception:
        return None, None


def make_summary_plot(results, out_path):
    """Bolometric ratio bar chart (raw vs cube) per model."""
    names, raw_meds, cube_meds = [], [], []
    raw_stds, cube_stds = [], []
    for r in results:
        if r["bol_ratios_raw"] or r["cube_bol_ratios"]:
            names.append(r["name"])
            raw_meds.append(float(np.median(r["bol_ratios_raw"])) if r["bol_ratios_raw"] else np.nan)
            cube_meds.append(float(np.median(r["cube_bol_ratios"])) if r["cube_bol_ratios"] else np.nan)
            raw_stds.append(float(np.std(r["bol_ratios_raw"])) if len(r["bol_ratios_raw"]) > 1 else 0.0)
            cube_stds.append(float(np.std(r["cube_bol_ratios"])) if len(r["cube_bol_ratios"]) > 1 else 0.0)

    if not names:
        print("No bolometric data to plot in summary.")
        return

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names)*0.9), 5))
    w = 0.35
    ax.bar(x - w/2, raw_meds,  w, yerr=raw_stds,  label="Raw files",
           color="steelblue", alpha=0.8, capsize=4, error_kw={"elinewidth": 1})
    ax.bar(x + w/2, cube_meds, w, yerr=cube_stds, label="Flux cube",
           color="tomato",    alpha=0.8, capsize=4, error_kw={"elinewidth": 1})

    ax.axhline(1.0,          color="black", lw=1.5, linestyle="--", label="Ideal (ratio=1)")
    ax.axhline(1 + BOL_TOL,  color="gray",  lw=0.8, linestyle=":")
    ax.axhline(1 - BOL_TOL,  color="gray",  lw=0.8, linestyle=":",
               label=f"±{int(BOL_TOL*100)}% tolerance")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Median bolometric ratio (actual / expected in wl range)")
    ax.set_title("SED_Tools Pipeline Audit — Bolometric Normalisation by Model")
    ax.legend(fontsize=8)
    all_vals = [v for v in raw_meds + cube_meds if np.isfinite(v)]
    ax.set_ylim(0, max(2.0, max(all_vals, default=2.0) * 1.15))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


def make_wl_coverage_plot(results, out_path):
    """Horizontal bar chart showing wavelength range per model."""
    # Only include models with actual data ranges
    data = [(r["name"], r["wl_range"][0], r["wl_range"][1])
            for r in results if r["wl_range"]]
    if not data:
        print("No wavelength range data to plot.")
        return

    # Sort by wl_min
    data.sort(key=lambda x: x[1])
    names  = [d[0] for d in data]
    wl_min = np.array([d[1] for d in data])
    wl_max = np.array([d[2] for d in data])
    spans  = wl_max - wl_min

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.38)))
    y = np.arange(len(names))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(names)))

    ax.barh(y, spans, left=wl_min, color=colors, alpha=0.8, height=0.65)

    # Annotate with range
    for i, (lo, hi) in enumerate(zip(wl_min, wl_max)):
        ax.text(hi + 200, i, f"{lo:.0f}–{hi:.0f} Å", va="center", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_title("SED_Tools — Wavelength Coverage per Model (from sampled raw files)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e4:.1f}μm" if v >= 1e4
                                                         else f"{v:.0f}Å"))
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


def make_sed_comparison_plot(results, target_teff, label, out_path):
    """
    For each model find the sampled file closest to target_teff, load it,
    normalize to peak flux, and overplot. Adds blackbody reference.
    Only models that actually cover (or nearly cover) the Wien peak are plotted.
    """
    # Determine a sensible wavelength window centred on the Wien peak
    wl_wien = wien_peak_ang(target_teff)
    # Show ±1.5 decades around the peak, but clamp to 200–2e5 Å
    x_lo = max(200.0,   wl_wien / 30.0)
    x_hi = min(2e5,     wl_wien * 30.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap   = plt.cm.tab20
    n_mod  = sum(1 for r in results if r["sampled_files"])
    colors = [cmap(i / max(n_mod, 1)) for i in range(n_mod)]

    plotted = 0
    for r, color in zip(results, colors):
        if not r["sampled_files"]:
            continue
        teffs = np.array([t for t, _, _, _ in r["sampled_files"]])
        idx = int(np.argmin(np.abs(teffs - target_teff)))
        best_teff, _lg, _mt, best_path = r["sampled_files"][idx]
        delta_teff = abs(best_teff - target_teff)

        try:
            raw = np.loadtxt(best_path, comments="#")
            if raw.ndim != 2 or raw.shape[1] < 2:
                continue
            wl, fl = raw[:, 0], raw[:, 1]
            fl = np.maximum(fl, 0)
            peak = fl.max()
            if peak <= 0:
                continue
            ax.plot(wl, fl / peak, color=color, alpha=0.85, lw=1.0,
                    label=f"{r['name']}  (Teff={best_teff:.0f} K, Δ={delta_teff:.0f} K)")

            # Cube SED — use variant cube if available so match is exact
            fname = Path(best_path).name
            cube_path = _find_cube_for_file(r["model_dir"], fname, best_teff, _lg, _mt)
            wl_c, fl_c = _extract_cube_sed(cube_path, best_teff, _lg, _mt)
            if wl_c is not None and fl_c.max() > 0:
                ax.plot(wl_c, fl_c / fl_c.max(), color=color, alpha=0.45,
                        lw=1.3, linestyle="--")

            plotted += 1
        except Exception:
            continue

    if plotted == 0:
        print(f"No data to plot for {label} SED comparison.")
        plt.close()
        return

    # Blackbody reference
    wl_bb = np.geomspace(max(x_lo, 100.0), x_hi, 4000)
    bb    = planck_flam(wl_bb, target_teff)
    ax.plot(wl_bb, bb / bb.max(), "k--", lw=2.0, zorder=10,
            label=f"Blackbody {target_teff:.0f} K")

    # Wien peak marker
    ax.axvline(wl_wien, color="black", lw=0.7, linestyle=":", alpha=0.5)
    ax.text(wl_wien * 1.02, 1.02, f"Wien\n{wl_wien:.0f} Å",
            transform=ax.get_xaxis_transform(), fontsize=7, va="top")

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Wavelength (Å)  [log scale]")
    ax.set_ylabel("Normalized flux  (F / F_peak)")
    ax.set_title(f"SED Comparison — {label} models near Teff = {target_teff:.0f} K")

    # Nicer x tick labels: show both Å and μm
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: (f"{v/1e4:.2g} μm" if v >= 1e4 else f"{v:.0f} Å")
    ))
    ax.legend(fontsize=7, ncol=2, loc="upper left" if wl_wien > 5000 else "upper right",
              framealpha=0.7)
    ax.grid(True, alpha=0.2, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved: {out_path}")



def make_sed_bb_overlay_plot(results, out_path):
    """
    One panel per model SED (picked as the median-Teff sampled file),
    each overlaid with a blackbody at that exact Teff.  All pairs are
    normalised to their own peak so they sit on top of each other cleanly.
    Uses a single shared wavelength window spanning all models.
    """
    # Gather one (wl, fl, teff, name) per model
    entries = []
    for r in results:
        if not r["sampled_files"]:
            continue
        teffs = [t for t, _, _, _ in r["sampled_files"]]
        median_teff = float(np.median(teffs))
        idx = int(np.argmin(np.abs(np.array(teffs) - median_teff)))
        best_teff, best_logg, best_meta, best_path = r["sampled_files"][idx]
        try:
            raw = np.loadtxt(best_path, comments="#")
            if raw.ndim != 2 or raw.shape[1] < 2:
                continue
            wl, fl = raw[:, 0], raw[:, 1]
            fl = np.maximum(fl, 0)
            if fl.max() <= 0:
                continue
            entries.append((r["name"], r["model_dir"], best_teff, best_logg, best_meta, best_path, wl, fl))
        except Exception:
            continue

    if not entries:
        print("No data for SED+blackbody overlay plot.")
        return

    # Shared wavelength window: union of all model ranges, clamped reasonably
    all_lo = [wl.min() for _, _, _, _, _, _, wl, _ in entries]
    all_hi = [wl.max() for _, _, _, _, _, _, wl, _ in entries]
    x_lo = max(min(all_lo), 100.0)
    x_hi = min(max(all_hi), 1e7)

    fig, ax = plt.subplots(figsize=(13, 6))
    cmap   = plt.cm.tab20
    colors = [cmap(i / max(len(entries), 1)) for i in range(len(entries))]

    for (name, model_dir, teff, logg, meta, fpath, wl, fl), color in zip(entries, colors):
        peak = fl.max()
        ax.plot(wl, fl / peak, color=color, alpha=0.85, lw=1.0,
                label=f"{name}  (Teff={teff:.0f} K)")

        # Cube SED — use variant cube so match is exact
        fname    = Path(fpath).name
        cube_path = _find_cube_for_file(model_dir, fname, teff, logg, meta)
        wl_c, fl_c = _extract_cube_sed(cube_path, teff, logg, meta)
        if wl_c is not None and fl_c.max() > 0:
            ax.plot(wl_c, fl_c / fl_c.max(), color=color, alpha=0.45,
                    lw=1.3, linestyle="--", label="_nolegend_")

        # Blackbody at same Teff, same normalisation
        wl_bb = np.geomspace(max(wl.min(), 10.0), wl.max(), 3000)
        bb    = planck_flam(wl_bb, teff)
        bb_peak = bb.max()
        if bb_peak > 0:
            ax.plot(wl_bb, bb / bb_peak, color=color, alpha=0.30,
                    lw=1.2, linestyle=":")

    # Legend: model names on right, style guide on left
    from matplotlib.lines import Line2D
    ax.add_artist(ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.7))
    proxies = [
        Line2D([0], [0], color="grey", lw=1.0, alpha=0.85,  label="Raw SED (solid)"),
        Line2D([0], [0], color="grey", lw=1.3, linestyle="--", alpha=0.55, label="Cube SED (dashed)"),
        Line2D([0], [0], color="grey", lw=1.2, linestyle=":",  alpha=0.45, label="Blackbody (dotted)"),
    ]
    ax.legend(handles=proxies, fontsize=7, loc="upper left", framealpha=0.7)

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Wavelength (Å)  [log scale]")
    ax.set_ylabel("Normalized flux  (F / F_peak)")
    ax.set_title("SED vs Blackbody — one representative spectrum per model, normalized to peak")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: (f"{v/1e4:.2g} μm" if v >= 1e4 else f"{v:.0f} Å")
    ))
    ax.grid(True, alpha=0.2, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sed_tools.models import STELLAR_DIR_DEFAULT

    stellar_dir = Path(os.environ.get("SED_STELLAR_DIR", STELLAR_DIR_DEFAULT))
    print(f"Auditing models in: {stellar_dir}")

    model_dirs = sorted(
        d for d in stellar_dir.iterdir()
        if d.is_dir() and any(d.glob("*.txt"))
    )

    if not model_dirs:
        sys.exit(f"No model directories with .txt files found under {stellar_dir}")

    print(f"Found {len(model_dirs)} models\n")

    results = []
    for d in model_dirs:
        print(f"  Auditing {d.name}...", end=" ", flush=True)
        try:
            r = audit_model(d)
            results.append(r)
            status = "PASS" if not r["errors"] and r["all_standardized"] else "issues"
            print(status)
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()

    report = format_report(results)
    print("\n" + report)

    report_path = "audit_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nSaved: {report_path}")

    make_summary_plot(results, "audit_summary.png")
    make_wl_coverage_plot(results, "wl_coverage.png")

    # -----------------------------------------------------------------------
    # Determine hot / cold Teff targets from the actual data across all models
    # -----------------------------------------------------------------------
    all_teffs = []
    for r in results:
        all_teffs.extend(t for t, _, _, _ in r["sampled_files"])

    if all_teffs:
        all_teffs_sorted = sorted(all_teffs)
        n = len(all_teffs_sorted)

        # "cold": 10th percentile, "hot": 90th percentile
        cold_teff = float(all_teffs_sorted[max(0, int(n * 0.10))])
        hot_teff  = float(all_teffs_sorted[min(n-1, int(n * 0.90))])

        # Round to nearest 500 K so the title looks clean
        cold_teff = round(cold_teff / 500) * 500
        hot_teff  = round(hot_teff  / 500) * 500

        print(f"\nSED comparison targets: cold={cold_teff:.0f} K, hot={hot_teff:.0f} K")
        make_sed_comparison_plot(results, hot_teff,  "Hot",  "sed_comparison_hot.png")
        make_sed_comparison_plot(results, cold_teff, "Cold", "sed_comparison_cold.png")
        make_sed_bb_overlay_plot(results, "sed_bb_overlay.png")
    else:
        print("\nWarning: no Teff headers found in any sampled files — skipping SED comparison plots.")
