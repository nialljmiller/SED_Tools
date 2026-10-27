"""Display colours for filter curves: the colour a band really has, or an ordered false colour.

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
"""

from __future__ import annotations

import colorsys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np

CIE_DIR = Path(__file__).resolve().parent / "cie"
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
        area = float(np.trapezoid(t, lam))
        mean[band] = float(np.trapezoid(lam * t, lam) / area) if area > 0 else float(np.mean(lam))
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
    from .flux_cube_tool import load_filter_curve

    curves = {}
    for path in curve_paths:
        try:
            curve = load_filter_curve(str(path))
        except ValueError as exc:
            raise ValueError(f"{path}: cannot read the curve to colour it: {exc}") from exc
        curves[Path(path).stem] = (curve.wavelength, curve.transmission)
    return display_colors(curves)
