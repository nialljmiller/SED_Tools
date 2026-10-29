"""Display colours for filter curves (the colour a band really has, or an ordered false colour), and plots in them.

A band with at least half of its throughput (∫ T dλ) between 3800 and 7500 Å gets its
true colour: the CIE 1931 2° tristimulus values of CIE standard illuminant D65 seen
through the curve, converted to sRGB (IEC 61966-2-1) and shown at full brightness.
Narrow bands are integrated on the curve's own samples; colours outside the sRGB
gamut are clipped to it.  A flat curve is white.

Every other band gets a false colour from its wavelength order within the instrument:

- with visible bands present, infrared bands continue redder than the reddest of them
  and ultraviolet bands bluer than the bluest;
- with none, the bands span a rainbow shifted toward their regime: infrared runs
  from yellow-green to crimson, ultraviolet from purple to sky blue.

The CIE tables in ``sed_tools/cie`` are the CIE's published files, unmodified:
CIE_xyz_1931_2deg.csv (doi:10.25039/CIE.DS.xvudnb9b) and CIE_std_illum_D65.csv
(doi:10.25039/CIE.DS.hjfjmt59).

:func:`plot_curves` draws curves in these colours; matplotlib is imported only when a plot is drawn.
"""

from __future__ import annotations

import colorsys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

CIE_DIR = Path(__file__).resolve().parents[1] / "cie"
VISIBLE_ANGSTROM = (3800.0, 7500.0)
TRUE_COLOR_FRACTION = 0.5

# sRGB (IEC 61966-2-1): linear RGB from CIE XYZ, D65 white.
XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])

# False-colour hue ranges in degrees (0 red, 120 green, 240 blue), listed from the
# band nearest the visible to the farthest; hues below 0 wrap round to crimson.
INFRARED_RAINBOW = (100.0, -25.0)
ULTRAVIOLET_RAINBOW = (195.0, 290.0)

Curve = Tuple[Sequence[float], Sequence[float]]
NORMALIZATIONS = (None, False, True, "peak", "area")


@lru_cache(maxsize=1)
def _cie():
    """(wavelength Å, x̄, ȳ, z̄, D65) on the CIE tables' common 1 nm grid."""
    cmf = np.loadtxt(CIE_DIR / "CIE_xyz_1931_2deg.csv", delimiter=",")
    d65 = np.loadtxt(CIE_DIR / "CIE_std_illum_D65.csv", delimiter=",")
    illuminant = np.interp(cmf[:, 0], d65[:, 0], d65[:, 1])
    return cmf[:, 0] * 10.0, cmf[:, 1], cmf[:, 2], cmf[:, 3], illuminant


def _arrays(wavelength, transmission):
    lam = np.asarray(wavelength, dtype=float)
    t = np.clip(np.nan_to_num(np.asarray(transmission, dtype=float)), 0.0, None)
    order = np.argsort(lam)
    return lam[order], t[order]


def _mean_wavelength(lam: np.ndarray, t: np.ndarray) -> float:
    """Transmission-weighted mean wavelength: it orders the false colours, the plotted curves and the legend."""
    area = np.trapezoid(t, lam)
    return float(np.trapezoid(lam * t, lam) / area) if area > 0 else float(lam.mean())


def visible_fraction(wavelength: Sequence[float], transmission: Sequence[float]) -> float:
    """Share of ∫ T dλ between 3800 and 7500 Å (wavelength in Å)."""
    lam, t = _arrays(wavelength, transmission)
    total = float(np.trapezoid(t, lam))
    if total <= 0:
        return 0.0
    lo, hi = max(lam[0], VISIBLE_ANGSTROM[0]), min(lam[-1], VISIBLE_ANGSTROM[1])
    if lo >= hi:
        return 0.0
    grid = np.unique(np.concatenate([lam[(lam > lo) & (lam < hi)], [lo, hi]]))
    return float(np.trapezoid(np.interp(grid, lam, t), grid) / total)


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(255 * min(max(v, 0.0), 1.0))):02x}" for v in rgb)


def true_color(wavelength: Sequence[float], transmission: Sequence[float]) -> str:
    """sRGB hex colour of daylight (D65) seen through the curve (wavelength in Å)."""
    lam, t = _arrays(wavelength, transmission)
    cie_lam, xbar, ybar, zbar, d65 = _cie()
    # Integrate over the curve's own span on both grids, so a 1 nm filter is not missed.
    lo, hi = max(lam[0], cie_lam[0]), min(lam[-1], cie_lam[-1])
    if lo >= hi:
        return "#000000"
    grid = np.union1d(cie_lam[(cie_lam > lo) & (cie_lam < hi)], lam[(lam >= lo) & (lam <= hi)])
    light = np.interp(grid, lam, t) * np.interp(grid, cie_lam, d65)
    xyz = np.array([np.trapezoid(light * np.interp(grid, cie_lam, cmf), grid) for cmf in (xbar, ybar, zbar)])
    if xyz[1] <= 0:
        return "#000000"
    rgb = np.clip(XYZ_TO_SRGB @ (xyz / xyz[1]), 0.0, None)  # clip to the sRGB gamut
    if rgb.max() <= 0:
        return "#000000"
    rgb = rgb / rgb.max()
    encoded = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return _hex(encoded)


def _hue(color: str) -> float:
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 360.0 * colorsys.rgb_to_hls(r, g, b)[0]


def _hls(hue: float, lightness: float, saturation: float = 0.85) -> str:
    return _hex(colorsys.hls_to_rgb((hue % 360.0) / 360.0, lightness, saturation))


def _spread(n: int, start: float, end: float):
    return [start] if n == 1 else list(np.linspace(start, end, n))


def display_colors(curves: Mapping[str, Curve]) -> Dict[str, Dict[str, Union[str, float]]]:
    """``{band: {DisplayColor, ColorMethod, VisibleFraction}}`` for one instrument's curves."""
    info, mean = {}, {}
    for band, (wavelength, transmission) in curves.items():
        lam, t = _arrays(wavelength, transmission)
        mean[band] = _mean_wavelength(lam, t)
        fraction = visible_fraction(lam, t)
        info[band] = {"VisibleFraction": round(fraction, 4)}
        if fraction >= TRUE_COLOR_FRACTION:
            info[band].update(DisplayColor=true_color(lam, t), ColorMethod="true")

    visible = [b for b in info if "DisplayColor" in info[b]]
    middle = sum(VISIBLE_ANGSTROM) / 2
    infrared = sorted((b for b in info if b not in visible and mean[b] > middle), key=mean.get)
    ultraviolet = sorted((b for b in info if b not in visible and mean[b] <= middle), key=mean.get, reverse=True)

    if visible:
        # Continue past the instrument's own reddest and bluest true colours,
        # darkening outward so that long runs of bands stay distinguishable.
        def wrapped(b):
            h = _hue(info[b]["DisplayColor"])
            return h - 360.0 if h > 300.0 else h
        reddest = min(15.0, wrapped(max(visible, key=mean.get))) - 10.0
        bluest = max(255.0, _hue(info[min(visible, key=mean.get)]["DisplayColor"])) + 10.0
        ranges = {"ir": (reddest, reddest - 35.0), "uv": (bluest, min(bluest + 35.0, 300.0))}
        lightness = (0.55, 0.35)
    else:
        ranges = {"ir": INFRARED_RAINBOW, "uv": ULTRAVIOLET_RAINBOW}
        lightness = (0.55, 0.55)
    for key, bands in (("ir", infrared), ("uv", ultraviolet)):
        if not visible and len(bands) == 1:
            hues = [0.0 if key == "ir" else 275.0]
        else:
            hues = _spread(len(bands), *ranges[key])
        lights = _spread(len(bands), *lightness)
        for band, hue, light in zip(bands, hues, lights):
            info[band].update(DisplayColor=_hls(hue, light), ColorMethod="false")
    return info


def instrument_colors(curve_paths: Sequence[Union[str, Path]]) -> Dict[str, Dict[str, Union[str, float]]]:
    """:func:`display_colors` for curve files, read as SED_Tools reads them (Å).

    Every curve must be readable: false colours follow the order of all the
    instrument's bands, so leaving one out would change the others.
    """
    from ..photometry import load_filter_curve

    curves = {}
    for path in curve_paths:
        try:
            curve = load_filter_curve(str(path))
        except ValueError as exc:
            raise ValueError(f"{path}: cannot read the curve to colour it: {exc}") from exc
        curves[Path(path).stem] = (curve.wavelength, curve.transmission)
    return display_colors(curves)


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
    import matplotlib.pyplot as plt

    if not curves:
        raise ValueError("no curves to plot")
    data = {}
    for key, (wavelength, transmission) in curves.items():
        lam, t = _arrays(wavelength, transmission)
        data[key] = (lam, _normalized(lam, t, normalize))
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
