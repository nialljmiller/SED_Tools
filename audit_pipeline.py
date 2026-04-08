#!/usr/bin/env python3
"""
audit_pipeline.py

Audits every installed stellar atmosphere model for:
  1. Unit standardization in raw .txt files
  2. Flux cube consistency with raw files
  3. Physical plausibility (bolometric integral vs sigma*T^4)
  4. Negative flux counts

Run from SED_Tools root:
    python audit_pipeline.py

Output:
    audit_report.txt  — full per-model results
    audit_summary.png — visual summary
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

# ---------------------------------------------------------------------------
SIGMA   = 5.670374419e-5   # erg s-1 cm-2 K-4
H       = 6.62607015e-27   # erg s
C_CGS   = 2.99792458e10    # cm/s
HC_ANG  = H * C_CGS * 1e8  # erg·Å (hc with λ in Å)
K_B     = 1.380649e-16     # erg/K
FLOOR   = 1e-300
N_SAMPLE = 20              # raw files to sample per model
BOL_TOL  = 0.30            # 30% tolerance on bolometric ratio

# Models where sigma*T^4 bolometric check is not applicable.
# grams: distance-calibrated AGB dust shell models — teff is the central
#        star temperature but flux is the whole system at a standard distance.
BOL_CHECK_EXEMPT = {"grams_cgrid", "grams_ogrid"}

# Minimum fraction of non-zero cube nodes required to trust the cube bol check.
# Sparse grids (hres: 1650 spectra / 3740 nodes = 44%) have many zero-filled
# nodes; if most sampled nodes are zero the median is meaningless.
CUBE_MIN_NONZERO_FRAC = 0.5
# ---------------------------------------------------------------------------


def planck_flam(wl_ang, teff):
    """π*B_λ in erg/cm²/s/Å. wl_ang in Å, teff in K."""
    l   = wl_ang * 1e-8                                       # Å → cm
    exp = np.minimum(HC_ANG / (wl_ang * K_B * teff), 700.0)  # hc/λkT (λ in Å)
    return np.pi * (2 * H * C_CGS**2 / l**5) / (np.exp(exp) - 1.0) * 1e-8


def expected_bol_in_range(teff, wl_min, wl_max, n=2000):
    """Expected bolometric flux within [wl_min, wl_max] Å for blackbody at teff.
    Returns (expected_in_range, fraction_of_sigmaT4)."""
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
    # File is written (W,M,L,T) — read accordingly
    # element offset for [i_t, i_l, i_m] across wavelength axis:
    # In (W,M,L,T) layout: element[w, i_m, i_l, i_t] = w*(nm*nl*nt) + i_m*(nl*nt) + i_l*nt + i_t
    # We want all W for fixed (i_t,i_l,i_m), so we read strided
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
    except Exception:
        pass
    return flags


def audit_model(model_dir):
    """Audit one model directory. Returns a result dict."""
    name = model_dir.name
    result = {
        "name": name,
        "n_files": 0,
        "n_sampled": 0,
        "all_standardized": True,
        "wrong_wl_unit": [],
        "wrong_fl_unit": [],
        "n_negative_raw": 0,
        "bol_ratios_raw": [],
        "cube_exists": False,
        "cube_size_gib": 0.0,
        "cube_grid": None,
        "cube_bol_ratios": [],
        "cube_neg_counts": [],
        "raw_vs_cube_max_err": [],
        "errors": [],
    }

    # --- raw files ---
    txts = sorted(model_dir.glob("*.txt"))
    result["n_files"] = len(txts)
    if not txts:
        result["errors"].append("No .txt files found")
        return result

    step = max(1, len(txts) // N_SAMPLE)
    samples = txts[::step][:N_SAMPLE]
    result["n_sampled"] = len(samples)

    for s in samples:
        flags = parse_header_flags(s)

        if not flags["units_standardized"]:
            result["all_standardized"] = False

        wlu = (flags["wavelength_unit"] or "").lower().replace(" ", "")
        if wlu and "angstrom" not in wlu and wlu not in ("a", "aa", "ang"):
            result["wrong_wl_unit"].append(f"{s.name}: {wlu}")

        flu = (flags["flux_unit"] or "").lower().replace(" ", "")
        # Accept erg/cm2/s/a and variants
        ok_fl = any(x in flu for x in ("erg/cm2/s/a", "erg/cm²/s/å",
                                        "erg/cm2/s/ang", "flam"))
        if flu and not ok_fl:
            result["wrong_fl_unit"].append(f"{s.name}: {flu}")

        # Load and check
        try:
            raw = np.loadtxt(s, comments="#")
            if raw.ndim != 2 or raw.shape[1] < 2:
                continue
            wl, fl = raw[:, 0], raw[:, 1]
            n_neg = int(np.sum(fl < 0))
            result["n_negative_raw"] += n_neg

            teff = flags["teff"]
            if teff and teff > 0 and len(wl) > 10 and name not in BOL_CHECK_EXEMPT:
                bol = float(np.trapezoid(np.maximum(fl, 0), wl))
                expected_in_range, frac = expected_bol_in_range(
                    teff, float(wl.min()), float(wl.max())
                )
                result.setdefault("bol_coverage_fracs", []).append(frac)
                if frac >= 0.05 and expected_in_range > 0:
                    result["bol_ratios_raw"].append(bol / expected_in_range)
        except Exception as e:
            result["errors"].append(f"Load error {s.name}: {e}")

    # --- flux cube ---
    cube_path = model_dir / "flux_cube.bin"
    if not cube_path.exists():
        result["errors"].append("flux_cube.bin missing")
        return result

    result["cube_exists"] = True
    result["cube_size_gib"] = cube_path.stat().st_size / 1e9

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

    # Sample a few cube nodes
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
        lines.append(f"  Files       : {r['n_files']} total, {r['n_sampled']} sampled")

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

        if r["n_negative_raw"] > 0:
            lines.append(f"  Negative raw flux points: {r['n_negative_raw']} (across sampled files)")

        if r["name"] in BOL_CHECK_EXEMPT:
            lines.append("  Bolometric check: EXEMPT (distance-calibrated model, σT⁴ not applicable)")
        elif r["bol_ratios_raw"]:
            ratios = r["bol_ratios_raw"]
            med = float(np.median(ratios))
            frac = float(np.median(r.get("bol_coverage_fracs", [1.0])))
            lines.append(f"  Bolometric ratio (raw)  : median={med:.4f}  "
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
            lines.append(f"  Cube wl     : {g['wl'][0]:.1f} – {g['wl'][1]:.1f} Å")

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
                lines.append(f"  Bol ratio (cube): median={med:.4f}  "
                             f"min={min(ratios):.4f}  max={max(ratios):.4f}")
                if n_zero:
                    lines.append(f"  Note: {n_zero} zero-flux cube nodes (zero-filled grid edges, expected)")
                # Only flag if enough non-zero nodes AND median is bad
                nonzero_frac = len(nonzero) / len(ratios) if ratios else 0
                if nonzero_frac < CUBE_MIN_NONZERO_FRAC:
                    lines.append(f"  Note: only {len(nonzero)}/{len(ratios)} sampled nodes populated "
                                 f"— sparse grid, cube bol check skipped")
                elif not (1 - BOL_TOL < med < 1 + BOL_TOL):
                    lines.append(f"  *** CUBE BOL RATIO MEDIAN OUT OF RANGE [{1-BOL_TOL:.1f}, {1+BOL_TOL:.1f}]")
                    ok = False
            elif r["name"] not in BOL_CHECK_EXEMPT and r["cube_exists"]:
                lines.append("  Bolometric check (cube): skipped (wl coverage <5% of σT⁴ for all nodes)")

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


def make_summary_plot(results, out_path):
    # Collect bolometric ratios per model
    names, raw_meds, cube_meds = [], [], []
    for r in results:
        if r["bol_ratios_raw"] or r["cube_bol_ratios"]:
            names.append(r["name"])
            raw_meds.append(float(np.median(r["bol_ratios_raw"])) if r["bol_ratios_raw"] else np.nan)
            cube_meds.append(float(np.median(r["cube_bol_ratios"])) if r["cube_bol_ratios"] else np.nan)

    if not names:
        return

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names)*0.8), 5))
    w = 0.35
    bars_raw  = ax.bar(x - w/2, raw_meds,  w, label="Raw files", color="steelblue", alpha=0.8)
    bars_cube = ax.bar(x + w/2, cube_meds, w, label="Flux cube",  color="tomato",    alpha=0.8)

    ax.axhline(1.0, color="black", lw=1.5, linestyle="--", label="Ideal (ratio=1)")
    ax.axhline(1 + BOL_TOL, color="gray", lw=0.8, linestyle=":")
    ax.axhline(1 - BOL_TOL, color="gray", lw=0.8, linestyle=":", label=f"±{int(BOL_TOL*100)}% tolerance")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Median bolometric ratio (actual / σT⁴)")
    ax.set_title("SED_Tools Pipeline Audit — Bolometric Normalisation by Model")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(2.0, max(filter(np.isfinite, raw_meds + cube_meds), default=2.0) * 1.1))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
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
