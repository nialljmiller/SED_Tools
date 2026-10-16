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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .header_parser import parse_header
from .spectrum_io import read_text_spectrum

_AXES = ("teff", "logg", "metallicity")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_PATH_COLUMNS = (
    "filename", "file_name", "file", "path", "spectrum", "spectrum_file",
    "sed", "sed_file", "model_file",
)


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

        out = pd.DataFrame(index=df.index)
        out["teff"] = pd.to_numeric(df[teff_col], errors="coerce") if teff_col else np.nan
        out["logg"] = pd.to_numeric(df[logg_col], errors="coerce") if logg_col else np.nan
        out["metallicity"] = (
            pd.to_numeric(df[meta_col], errors="coerce") if meta_col else np.nan
        )
        path_col = next((c for c in _PATH_COLUMNS if c in df.columns), None)
        if path_col:
            filenames = df[path_col].where(df[path_col].notna(), "").astype(str)
            out["filename"] = filenames
            out["path"] = filenames.map(
                lambda value: str(
                    (Path(value).expanduser() if Path(value).expanduser().is_absolute()
                     else model_dir / value).resolve()
                ) if value else ""
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
                "filename": f.name,
                "path": str(f.resolve()),
            }
        )
    return pd.DataFrame(rows, columns=list(_AXES) + ["filename", "path"])


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
# Terminal coverage plots
# ---------------------------------------------------------------------------

def _representative_nodes(df: pd.DataFrame) -> List[Dict]:
    """Select deterministic corners plus center in normalized parameter space."""
    usable = df[np.isfinite(pd.to_numeric(df["teff"], errors="coerce"))].copy()
    if usable.empty:
        return []
    for axis in _AXES:
        usable[axis] = pd.to_numeric(usable[axis], errors="coerce")

    teff = usable["teff"].to_numpy(dtype=float)
    logg = usable["logg"].to_numpy(dtype=float)
    meta = usable["metallicity"].to_numpy(dtype=float)
    finite_logg = logg[np.isfinite(logg)]
    finite_meta = meta[np.isfinite(meta)]
    tmin, tmax, tmed = float(np.min(teff)), float(np.max(teff)), float(np.median(teff))

    if np.unique(finite_logg).size <= 1:
        targets = [
            ("A", "cool edge", {"teff": tmin}),
            ("B", "center", {"teff": tmed}),
            ("C", "hot edge", {"teff": tmax}),
        ]
    else:
        gmin, gmax = float(np.min(finite_logg)), float(np.max(finite_logg))
        mmed = float(np.median(finite_meta)) if finite_meta.size else float("nan")
        targets = [
            ("A", "cool/low-g corner", {"teff": tmin, "logg": gmin, "metallicity": mmed}),
            ("B", "cool/high-g corner", {"teff": tmin, "logg": gmax, "metallicity": mmed}),
            ("C", "hot/low-g corner", {"teff": tmax, "logg": gmin, "metallicity": mmed}),
            ("D", "hot/high-g corner", {"teff": tmax, "logg": gmax, "metallicity": mmed}),
            ("E", "center", {"teff": tmed, "logg": float(np.median(finite_logg)), "metallicity": mmed}),
        ]

    selected: List[Dict] = []
    used = set()
    for letter, description, target in targets:
        distance = np.zeros(len(usable), dtype=float)
        valid_target = np.ones(len(usable), dtype=bool)
        dimensions = 0
        for axis, wanted in target.items():
            values = usable[axis].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if not np.isfinite(wanted) or finite.size == 0 or np.max(finite) == np.min(finite):
                continue
            valid_target &= np.isfinite(values)
            distance += ((values - wanted) / (np.max(finite) - np.min(finite))) ** 2
            dimensions += 1
        if not dimensions or not np.any(valid_target):
            continue
        distance[~valid_target] = np.inf
        pos = int(np.argmin(distance))
        original_index = usable.index[pos]
        identity = str(usable.iloc[pos].get("path", "")) or str(original_index)
        if identity in used:
            continue
        used.add(identity)
        selected.append({
            "letter": letter, "description": description,
            "index": original_index, "row": usable.iloc[pos],
        })
    return selected


def _load_txt_spectrum(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Read the first two numeric columns from a text spectrum."""
    return read_text_spectrum(path)


def _node_parameters(item: Dict) -> str:
    """Return a compact parameter label suitable for a half-width plot."""
    row = item["row"]
    parts = [f"T={row['teff']:g}"]
    if np.isfinite(row["logg"]):
        parts.append(f"g={row['logg']:g}")
    if np.isfinite(row["metallicity"]):
        parts.append(f"M/H={row['metallicity']:g}")
    return " ".join(parts)


def _plot_rows(plots: List[str], columns: int = 2, gap: int = 3) -> str:
    """Arrange terminal plots in rows without coupling the renderers to layout."""
    rendered_rows = []
    for start in range(0, len(plots), columns):
        blocks = [plot.splitlines() for plot in plots[start:start + columns]]
        widths = [
            max((_visible_width(line) for line in block), default=0)
            for block in blocks
        ]
        row_height = max((len(block) for block in blocks), default=0)
        lines = []
        for line_number in range(row_height):
            parts = [
                _visible_ljust(
                    block[line_number] if line_number < len(block) else "", width
                )
                for block, width in zip(blocks, widths)
            ]
            lines.append((" " * gap).join(parts).rstrip())
        rendered_rows.append("\n".join(lines))
    return "\n\n".join(rendered_rows)


def _visible_width(text: str) -> int:
    """Measure terminal text without counting ANSI color control sequences."""
    return len(_ANSI_ESCAPE.sub("", text))


def _visible_ljust(text: str, width: int) -> str:
    return text + " " * max(0, width - _visible_width(text))


def print_terminal_coverage(summary: Dict, *, color: str = "auto") -> None:
    """Print parameter projections and representative normalized SEDs."""
    from .terminal_plots import lineplot, scatter2d, terminal_width

    df = summary["_df"]
    selected = _representative_nodes(df)
    palette = ["cyan", "yellow", "green", "magenta", "blue", "red", "white"]
    labels = {df.index.get_loc(item["index"]): item["letter"] for item in selected}
    colors = {
        df.index.get_loc(item["index"]): palette[i % len(palette)]
        for i, item in enumerate(selected)
    }
    projections = [
        ("teff", "logg", "Teff [K]", "log g", True),
        ("teff", "metallicity", "Teff [K]", "[M/H]", False),
        ("logg", "metallicity", "log g", "[M/H]", False),
    ]
    # Each panel occupies roughly half an 80-column terminal and half the
    # renderer's original height, allowing two useful plots per row.
    plot_width = max(20, min(42, (terminal_width() - 30) // 2))
    parameter_plots = []
    for xaxis, yaxis, xlabel, ylabel, invert in projections:
        parameter_plots.append(scatter2d(
            df[xaxis], df[yaxis],
            title=f"{xlabel} vs {ylabel}",
            xlabel=xlabel, ylabel=ylabel, invert_y=invert,
            point_labels=labels, point_colors=colors, color=color,
            width=plot_width, height=9,
        ))
    print(f"\n{summary['name']}: parameter coverage\n")
    print(_plot_rows(parameter_plots))

    if "path" not in df.columns or not any(str(value).strip() for value in df["path"]):
        print("\nTerminal SED plot skipped: lookup_table.csv does not identify spectrum files.")
        return

    sed_plots = []
    for i, item in enumerate(selected):
        path = Path(str(item["row"].get("path", "")))
        if not path.is_file():
            continue
        try:
            wavelength, flux = _load_txt_spectrum(path)
        except OSError:
            continue
        good = np.isfinite(wavelength) & np.isfinite(flux) & (flux > 0)
        wavelength, flux = wavelength[good], flux[good]
        if wavelength.size < 2:
            continue
        normalized = np.log10(flux / np.max(flux))
        series = [{
            "x": wavelength, "y": normalized,
            "label": _node_parameters(item), "char": item["letter"],
            "color": palette[i % len(palette)],
        }]
        sed_plots.append(lineplot(
            series, title=f"SED {item['letter']}: {item['description']}",
            xlabel="Wavelength", ylabel="log10(Fλ / max Fλ)", color=color,
            width=plot_width, height=8,
        ))
    if sed_plots:
        print("\nRepresentative SEDs (individually normalized)\n")
        print(_plot_rows(sed_plots))
    else:
        print("\nTerminal SED plot skipped: no readable representative spectra.")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_coverage(
    df: pd.DataFrame, name: str, out_path: Union[str, Path]
) -> Path:
    """Write a 2x2 coverage figure with all 2D pairings plus a 3D view."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    sub = df.dropna(subset=list(_AXES))
    teff = sub["teff"].to_numpy()
    logg = sub["logg"].to_numpy()
    meta = sub["metallicity"].to_numpy()

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # 1) Teff vs log g, coloured by [M/H]
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(teff, logg, c=meta, cmap="viridis", s=18, edgecolors="none")
    ax1.set_xlabel("Teff [K]")
    ax1.set_ylabel("log g")
    ax1.set_title(f"{name}: Teff-log g")
    if logg.size:
        ax1.invert_yaxis()
    fig.colorbar(sc1, ax=ax1, label="[M/H]")

    # 2) Teff vs [M/H], coloured by log g
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(teff, meta, c=logg, cmap="viridis", s=18, edgecolors="none")
    ax2.set_xlabel("Teff [K]")
    ax2.set_ylabel("[M/H]")
    ax2.set_title("Teff-[M/H]")
    fig.colorbar(sc2, ax=ax2, label="log g")

    # 3) log g vs [M/H], coloured by Teff
    ax3 = fig.add_subplot(gs[1, 0])
    sc3 = ax3.scatter(logg, meta, c=teff, cmap="viridis", s=18, edgecolors="none")
    ax3.set_xlabel("log g")
    ax3.set_ylabel("[M/H]")
    ax3.set_title("log g-[M/H]")
    fig.colorbar(sc3, ax=ax3, label="Teff [K]")

    # 4) 3D coverage
    ax4 = fig.add_subplot(gs[1, 1], projection="3d")
    ax4.scatter(teff, logg, meta, c=meta, cmap="viridis", s=12, edgecolors="none")
    ax4.set_xlabel("Teff [K]")
    ax4.set_ylabel("log g")
    ax4.set_zlabel("[M/H]")
    ax4.set_title("3D coverage")

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

    from .config import get_ui_setting
    from .terminal_plots import terminal_plots_enabled

    plot_mode = get_ui_setting("terminal_plots")
    color_mode = get_ui_setting("terminal_color")
    if terminal_plots_enabled(plot_mode):
        print_terminal_coverage(summary, color=color_mode)

    if plot:
        target = Path(out_path) if out_path else (model_dir / "coverage.png")
        written = plot_coverage(summary["_df"], summary["name"], target)
        print(f"\n  Plot written: {written}")
        summary["plot_path"] = str(written)

    return summary
