"""Draw filter curves in their display colours (:mod:`sed_tools.filter_colors`).

Matplotlib is imported only when a plot is drawn.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .filter_colors import display_colors

Curve = Tuple[Sequence[float], Sequence[float]]
NORMALIZATIONS = (None, False, True, "peak", "area")


def _normalized(lam: np.ndarray, t: np.ndarray, normalize) -> np.ndarray:
    if normalize in (None, False):
        return t
    if normalize in (True, "peak"):
        peak = t.max()
        return t / peak if peak > 0 else t
    if normalize == "area":
        area = np.trapezoid(t, lam)
        return t / area if area > 0 else t
    raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}")


def plot_curves(
    curves: Mapping[str, Curve],
    *,
    colors: Optional[Mapping[str, str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    normalize: Union[None, bool, str] = None,
    vega: Optional[Curve] = None,
    ax=None,
    legend: bool = True,
    log_wavelength: Optional[bool] = None,
    title: Optional[str] = None,
):
    """Plot ``{key: (wavelength Å, transmission)}`` and return the matplotlib Axes.

    ``colors`` defaults to :func:`~sed_tools.filter_colors.display_colors` of the
    curves given.  ``normalize`` is ``None`` (as tabulated), ``"peak"`` (or
    ``True``: maximum 1) or ``"area"`` (∫T dλ = 1, per Å).  ``vega`` is
    ``(wavelength Å, f_λ)``; it is drawn scaled to the tallest curve over the
    plotted range, on the same axis.  ``log_wavelength`` defaults to a log axis
    when the curves span more than a factor of ten in wavelength.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - matplotlib is a dependency
        raise ImportError("Filter plots need matplotlib: pip install matplotlib") from exc

    if not curves:
        raise ValueError("no curves to plot")
    data = {}
    for key, (wavelength, transmission) in curves.items():
        lam = np.asarray(wavelength, dtype=float)
        t = np.clip(np.nan_to_num(np.asarray(transmission, dtype=float)), 0.0, None)
        order = np.argsort(lam)
        data[key] = (lam[order], _normalized(lam[order], t[order], normalize))
    colors = dict(colors or {})
    missing = [k for k in data if k not in colors]
    if missing:
        computed = display_colors({k: curves[k] for k in missing})
        colors.update({k: computed[k]["DisplayColor"] for k in missing})

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for key in sorted(data, key=lambda k: _mean_wavelength(*data[k])):
        lam, t = data[key]
        label = (labels or {}).get(key, key)
        ax.plot(lam, t, color=colors[key], lw=1.5, label=label)
        ax.fill_between(lam, t, color=colors[key], alpha=0.12, lw=0)

    lo = min(lam[0] for lam, _ in data.values())
    hi = max(lam[-1] for lam, _ in data.values())
    top = max(float(t.max()) for _, t in data.values())
    if vega is not None:
        _draw_vega(ax, vega, lo, hi, top)

    if log_wavelength is None:
        log_wavelength = hi / max(lo, 1e-30) > 10
    if log_wavelength:
        _log_wavelength_axis(ax)
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel({None: "Transmission", False: "Transmission", True: "Transmission (peak = 1)",
                   "peak": "Transmission (peak = 1)", "area": "Transmission / ∫T dλ (Å$^{-1}$)"}[normalize])
    ax.grid(True, color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, loc="left")
    if legend and (len(data) > 1 or vega is not None):
        n = len(data) + (vega is not None)
        ax.legend(fontsize="small", frameon=False, ncol=max(1, int(np.ceil(n / 12))),
                  loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return ax


def _mean_wavelength(lam: np.ndarray, t: np.ndarray) -> float:
    """Transmission-weighted mean wavelength, which orders the curves (and the legend)."""
    area = np.trapezoid(t, lam)
    return float(np.trapezoid(lam * t, lam) / area) if area > 0 else float(lam.mean())


def _draw_vega(ax, vega: Curve, lo: float, hi: float, top: float) -> None:
    """Vega's f_λ over the plotted range, scaled so its peak matches the tallest curve."""
    vl, vf = (np.asarray(a, dtype=float) for a in vega)
    inside = (vl >= lo) & (vl <= hi)
    if inside.any() and vf[inside].max() > 0:
        ax.plot(vl[inside], vf[inside] * top / vf[inside].max(), color="0.45", lw=1.0, ls="--",
                label="Vega f$_\\lambda$ (scaled)")


def _log_wavelength_axis(ax) -> None:
    """A log wavelength axis labelled in plain numbers: 2000, 5000, 10000 rather than 2×10³."""
    from matplotlib.ticker import FuncFormatter

    def plain(value, _):  # label 1, 2 and 5 per decade
        if value <= 0:
            return ""
        leading = round(value / 10 ** np.floor(np.log10(value)), 6)
        return f"{value:g}" if leading in (1.0, 2.0, 5.0) else ""

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(plain))
    ax.xaxis.set_minor_formatter(FuncFormatter(plain))
