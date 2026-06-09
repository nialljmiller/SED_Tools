"""
sed_tools/grid_coverage.py

Parameter-space coverage for any local grid — downloaded or hand-built.

Reads lookup_table.csv directly (so it works on grids that have not yet had a
flux cube built), falling back to header_parser.parse_header over the .txt
spectra when no lookup table is present. Reports per-axis ranges, the unique
grid values and their spacing, node count vs. the full Teff x logg x [M/H]
product (fill fraction), and an Anderson-Darling normality statistic per axis.
Optionally writes a Teff-logg + 3D Teff/logg/[M/H] coverage plot.

Public entry point: grid_coverage(name_or_dir, base_dir=..., plot=..., out_path=...).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .header_parser import parse_header

_AXES = ("teff", "logg", "metallicity")


# ---------------------------------------------------------------------------
# Node table
# ---------------------------------------------------------------------------

def _resolve_dir(name_or_dir: Union[str, Path], base_dir: Optional[Path]) -> Path:
    """Resolve a grid identifier to a model directory."""
    p = Path(name_or_dir).expanduser()
    if p.is_dir():
        return p
    if base_dir is not None:
        cand = Path(base_dir).expanduser() / str(name_or_dir)
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"Grid '{name_or_dir}' not found (looked for a directory"
        + (f" under {base_dir}" if base_dir is not None else "")
        + ")."
    )


def read_grid_nodes(model_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Return a DataFrame with columns teff, logg, metallicity (floats, nan where
    unknown) for every spectrum in *model_dir*.

    Prefers lookup_table.csv; falls back to parsing .txt headers directly.
    Raises if the directory has neither a lookup table nor any .txt spectra.
    """
    model_dir = Path(model_dir)
    lookup = model_dir / "lookup_table.csv"

    if lookup.exists():
        df = pd.read_csv(lookup)
        # Normalise column names (handles the '#'-prefixed header and case).
        df.columns = [c.lstrip("#").strip().lower() for c in df.columns]

        teff_col = next((c for c in df.columns if "teff" in c), None)
        logg_col = next((c for c in df.columns if "logg" in c or "log(g)" in c), None)
        meta_col = next(
            (c for c in df.columns if "meta" in c or "feh" in c or "m/h" in c), None
        )

        out = pd.DataFrame()
        out["teff"] = pd.to_numeric(df[teff_col], errors="coerce") if teff_col else np.nan
        out["logg"] = pd.to_numeric(df[logg_col], errors="coerce") if logg_col else np.nan
        out["metallicity"] = (
            pd.to_numeric(df[meta_col], errors="coerce") if meta_col else np.nan
        )
        return out

    txts = sorted(model_dir.glob("*.txt"))
    if not txts:
        raise RuntimeError(
            f"No lookup_table.csv and no .txt spectra in {model_dir}; "
            "nothing to report coverage for."
        )
    rows = []
    for f in txts:
        h = parse_header(str(f))
        rows.append(
            {
                "teff": float(h.get("teff", float("nan"))),
                "logg": float(h.get("logg", float("nan"))),
                "metallicity": float(h.get("metallicity", float("nan"))),
            }
        )
    return pd.DataFrame(rows, columns=list(_AXES))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _anderson_darling(x: np.ndarray) -> Optional[Dict[str, float]]:
    """Anderson-Darling test for normality. None if too few distinct values
    or scipy is unavailable. Works across scipy versions (old critical-value
    API and the newer p-value API)."""
    import warnings

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if np.unique(x).size < 8:
        return None
    try:
        from scipy.stats import anderson
    except ImportError:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = anderson(x, dist="norm")
    stat = float(res.statistic)
    if getattr(res, "critical_values", None) is not None:
        sl = [float(s) for s in res.significance_level]
        i5 = sl.index(5.0) if 5.0 in sl else len(sl) // 2
        crit5 = float(res.critical_values[i5])
        return {
            "statistic": stat,
            "crit_5pct": crit5,
            "normal_at_5pct": bool(stat < crit5),
        }
    # Newer scipy: only a p-value is provided.
    p = float(getattr(res, "pvalue", float("nan")))
    return {
        "statistic": stat,
        "pvalue": p,
        "normal_at_5pct": (bool(p > 0.05) if p == p else None),
    }


def coverage_summary(model_dir: Union[str, Path]) -> Dict:
    """Compute a coverage summary dict for one grid."""
    model_dir = Path(model_dir)
    df = read_grid_nodes(model_dir)
    n_rows = len(df)

    axes: Dict[str, Dict] = {}
    n_unique = {}
    for ax in _AXES:
        col = df[ax].to_numpy(dtype=float)
        finite = col[np.isfinite(col)]
        uniq = np.unique(finite)
        n_unique[ax] = int(uniq.size)
        spacing = np.diff(uniq) if uniq.size > 1 else np.array([])
        axes[ax] = {
            "min": float(uniq.min()) if uniq.size else float("nan"),
            "max": float(uniq.max()) if uniq.size else float("nan"),
            "n_unique": int(uniq.size),
            "n_missing": int(np.count_nonzero(~np.isfinite(col))),
            "unique": uniq,
            "spacing_min": float(spacing.min()) if spacing.size else float("nan"),
            "spacing_max": float(spacing.max()) if spacing.size else float("nan"),
            "spacing_median": float(np.median(spacing)) if spacing.size else float("nan"),
            "anderson_darling": _anderson_darling(finite),
        }

    product = 1
    for ax in _AXES:
        product *= max(n_unique[ax], 1)
    complete = df.dropna(subset=list(_AXES))
    n_distinct_nodes = int(complete.drop_duplicates(subset=list(_AXES)).shape[0])
    fill = (n_distinct_nodes / product) if product else float("nan")

    return {
        "name": model_dir.name,
        "model_dir": str(model_dir),
        "n_spectra": n_rows,
        "n_distinct_nodes": n_distinct_nodes,
        "full_product": int(product),
        "fill_fraction": fill,
        "axes": axes,
        "_df": df,
    }


def _fmt_unique(uniq: np.ndarray, limit: int = 12) -> str:
    if uniq.size == 0:
        return "(none)"
    vals = [f"{v:g}" for v in uniq]
    if uniq.size <= limit:
        return ", ".join(vals)
    head = ", ".join(vals[: limit - 2])
    return f"{head}, ..., {vals[-2]}, {vals[-1]}"


def print_coverage(summary: Dict) -> None:
    """Pretty-print a coverage summary to stdout."""
    print("\n" + "=" * 64)
    print(f"Coverage: {summary['name']}")
    print("=" * 64)
    print(f"  Spectra            : {summary['n_spectra']}")
    print(f"  Distinct (T,g,M) nodes : {summary['n_distinct_nodes']}")
    print(
        f"  Full grid product  : {summary['full_product']}  "
        f"(fill {100.0 * summary['fill_fraction']:.1f}%)"
    )

    labels = {"teff": "Teff [K]", "logg": "log g", "metallicity": "[M/H]"}
    for ax in _AXES:
        a = summary["axes"][ax]
        print(f"\n  {labels[ax]}")
        if a["n_unique"] == 0:
            print("    no finite values")
            continue
        print(f"    range   : {a['min']:g} -> {a['max']:g}")
        print(f"    values  : {a['n_unique']} unique  [{_fmt_unique(a['unique'])}]")
        if not math.isnan(a["spacing_median"]):
            print(
                f"    spacing : min {a['spacing_min']:g}  "
                f"median {a['spacing_median']:g}  max {a['spacing_max']:g}"
            )
        if a["n_missing"]:
            print(f"    missing : {a['n_missing']} spectra have no value on this axis")
        ad = a["anderson_darling"]
        if ad is not None:
            if ad.get("normal_at_5pct") is None:
                verdict = "indeterminate"
            else:
                verdict = "consistent with normal" if ad["normal_at_5pct"] else "non-normal"
            if "crit_5pct" in ad:
                print(
                    f"    A-D     : {ad['statistic']:.3f} "
                    f"(5% crit {ad['crit_5pct']:.3f}; {verdict})"
                )
            else:
                print(
                    f"    A-D     : {ad['statistic']:.3f} "
                    f"(p={ad['pvalue']:.3f}; {verdict})"
                )
        else:
            print("    A-D     : skipped (need >=8 distinct values, scipy required)")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_coverage(
    df: pd.DataFrame, name: str, out_path: Union[str, Path]
) -> Path:
    """Write a Teff-logg + 3D Teff/logg/[M/H] coverage figure to out_path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    sub = df.dropna(subset=list(_AXES))
    teff = sub["teff"].to_numpy()
    logg = sub["logg"].to_numpy()
    meta = sub["metallicity"].to_numpy()

    fig = plt.figure(figsize=(13, 5.5))

    ax1 = fig.add_subplot(1, 2, 1)
    sc = ax1.scatter(teff, logg, c=meta, cmap="viridis", s=16, edgecolors="none")
    ax1.set_xlabel("Teff [K]")
    ax1.set_ylabel("log g")
    if logg.size:
        ax1.invert_yaxis()
    ax1.set_title(f"{name}: Teff-log g")
    fig.colorbar(sc, ax=ax1, label="[M/H]")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(teff, logg, meta, c=meta, cmap="viridis", s=12, edgecolors="none")
    ax2.set_xlabel("Teff [K]")
    ax2.set_ylabel("log g")
    ax2.set_zlabel("[M/H]")
    ax2.set_title("3D coverage")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def grid_coverage(
    name_or_dir: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    plot: bool = True,
    out_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Print a coverage summary for a grid and (optionally) write a coverage plot.

    Parameters
    ----------
    name_or_dir : grid name (resolved under base_dir) or a path to a model dir.
    base_dir    : base stellar_models directory used to resolve a bare name.
    plot        : whether to write the coverage figure.
    out_path    : plot output path. Defaults to <model_dir>/coverage.png.

    Returns the summary dict (with the node DataFrame under key '_df').
    """
    base = Path(base_dir).expanduser() if base_dir is not None else None
    model_dir = _resolve_dir(name_or_dir, base)
    summary = coverage_summary(model_dir)
    print_coverage(summary)

    if plot:
        target = Path(out_path) if out_path else (model_dir / "coverage.png")
        written = plot_coverage(summary["_df"], summary["name"], target)
        print(f"\n  Plot written: {written}")
        summary["plot_path"] = str(written)

    return summary
