"""Small dependency-free terminal renderers used by coverage diagnostics."""

from __future__ import annotations

import locale
import os
import shutil
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np

_COLORS = {
    "black": "30", "red": "31", "green": "32", "yellow": "33",
    "blue": "34", "magenta": "35", "cyan": "36", "white": "37",
}


def _normalize_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    return value if value in {"auto", "always", "never"} else "auto"


def terminal_plots_enabled(mode: str = "auto") -> bool:
    mode = _normalize_mode(mode)
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(sys.stdout.isatty() and os.environ.get("TERM", "").lower() not in {"", "dumb"})


def terminal_color_enabled(mode: str = "auto") -> bool:
    mode = _normalize_mode(mode)
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(
        sys.stdout.isatty()
        and os.environ.get("TERM", "").lower() not in {"", "dumb"}
        and "NO_COLOR" not in os.environ
        and os.environ.get("CLICOLOR") != "0"
    )


def terminal_unicode_enabled() -> bool:
    encoding = sys.stdout.encoding or locale.getpreferredencoding(False) or ""
    normalized = encoding.lower().replace("-", "").replace("_", "")
    return "utf8" in normalized


def terminal_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size(fallback=(default, 24)).columns
    except OSError:
        return default


def _paint(char: str, color: Optional[str], enabled: bool) -> str:
    code = _COLORS.get(str(color).lower()) if color else None
    return f"\x1b[{code}m{char}\x1b[0m" if enabled and code else char


def _limits(values: np.ndarray):
    lo, hi = float(np.min(values)), float(np.max(values))
    if lo == hi:
        pad = abs(lo) * 0.05 or 0.5
        lo, hi = lo - pad, hi + pad
    return lo, hi


def scatter2d(
    x, y, *, title: str, xlabel: str, ylabel: str,
    width: Optional[int] = None, height: int = 18, invert_y: bool = False,
    point_labels: Optional[Dict[int, str]] = None,
    point_colors: Optional[Dict[int, str]] = None, color: str = "auto",
) -> str:
    """Render a compact labelled scatter plot and return it as text."""
    xv, yv = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    n = min(xv.size, yv.size)
    xv, yv = xv[:n], yv[:n]
    finite = np.isfinite(xv) & np.isfinite(yv)
    indices = np.nonzero(finite)[0]
    xv, yv = xv[finite], yv[finite]
    if not xv.size:
        return f"{title}\n(no finite data)\n{xlabel} / {ylabel}"

    canvas_w = max(20, min(width or terminal_width() - 14, 120))
    canvas_h = max(6, int(height))
    xlo, xhi = _limits(xv)
    ylo, yhi = _limits(yv)
    unicode_ok = terminal_unicode_enabled()
    dot, vert, horiz, corner = ("·", "│", "─", "└") if unicode_ok else (".", "|", "-", "+")
    canvas = [[dot if False else " " for _ in range(canvas_w)] for _ in range(canvas_h)]
    cell_colors = [[None for _ in range(canvas_w)] for _ in range(canvas_h)]
    labels = point_labels or {}
    colors = point_colors or {}

    for original, px, py in zip(indices, xv, yv):
        col = int(round((px - xlo) / (xhi - xlo) * (canvas_w - 1)))
        frac = (py - ylo) / (yhi - ylo)
        row = int(round(frac * (canvas_h - 1))) if invert_y else int(round((1.0 - frac) * (canvas_h - 1)))
        marker = labels.get(int(original), dot)
        if int(original) in labels or canvas[row][col] == " ":
            canvas[row][col] = str(marker)[:1]
            cell_colors[row][col] = colors.get(int(original))

    use_color = terminal_color_enabled(color)
    ytop, ybottom = (ylo, yhi) if invert_y else (yhi, ylo)
    lines = [title, "", f"{ylabel}  {ytop:g}"]
    for row in range(canvas_h):
        rendered = "".join(
            _paint(canvas[row][col], cell_colors[row][col], use_color)
            for col in range(canvas_w)
        )
        lines.append(f" {' ' * len(ylabel)} {vert}{rendered}")
    lines.append(f" {ybottom:g} {corner}{horiz * canvas_w}")
    lines.append(f"       {xlo:g}{' ' * max(1, canvas_w - len(f'{xlo:g}') - len(f'{xhi:g}'))}{xhi:g}")
    lines.append(f"       {xlabel.center(canvas_w)}")
    return "\n".join(lines)


def lineplot(
    series: Iterable[dict], *, title: str, xlabel: str, ylabel: str,
    width: Optional[int] = None, height: int = 16, color: str = "auto",
) -> str:
    """Render multiple numeric series on one character canvas."""
    clean: List[dict] = []
    for item in series:
        x = np.asarray(item.get("x", []), dtype=float).ravel()
        y = np.asarray(item.get("y", []), dtype=float).ravel()
        n = min(x.size, y.size)
        good = np.isfinite(x[:n]) & np.isfinite(y[:n])
        x, y = x[:n][good], y[:n][good]
        if x.size >= 2:
            order = np.argsort(x)
            clean.append({**item, "x": x[order], "y": y[order]})
    if not clean:
        return f"{title}\n(no finite series)\n{xlabel} / {ylabel}"

    canvas_w = max(20, min(width or terminal_width() - 14, 120))
    canvas_h = max(6, int(height))
    xlo = min(float(np.min(s["x"])) for s in clean)
    xhi = max(float(np.max(s["x"])) for s in clean)
    if xlo == xhi:
        xhi = xlo + 1.0
    samples = np.linspace(xlo, xhi, canvas_w)
    interpolated = []
    for item in clean:
        vals = np.interp(samples, item["x"], item["y"], left=np.nan, right=np.nan)
        vals[(samples < item["x"][0]) | (samples > item["x"][-1])] = np.nan
        interpolated.append(vals)
    all_y = np.concatenate([v[np.isfinite(v)] for v in interpolated])
    ylo, yhi = _limits(all_y)
    canvas = [[" " for _ in range(canvas_w)] for _ in range(canvas_h)]
    cell_colors = [[None for _ in range(canvas_w)] for _ in range(canvas_h)]
    for item, vals in zip(clean, interpolated):
        char = str(item.get("char", "*"))[:1]
        for col, value in enumerate(vals):
            if np.isfinite(value):
                row = int(round((1.0 - (value - ylo) / (yhi - ylo)) * (canvas_h - 1)))
                canvas[row][col] = char
                cell_colors[row][col] = item.get("color")

    unicode_ok = terminal_unicode_enabled()
    vert, horiz, corner = ("│", "─", "└") if unicode_ok else ("|", "-", "+")
    use_color = terminal_color_enabled(color)
    lines = [title, f"{ylabel}  {yhi:g}"]
    for row in range(canvas_h):
        rendered = "".join(_paint(canvas[row][c], cell_colors[row][c], use_color) for c in range(canvas_w))
        lines.append(f" {vert}{rendered}")
    lines.append(f" {ylo:g} {corner}{horiz * canvas_w}")
    lines.append(f"     {xlo:g}{' ' * max(1, canvas_w - len(f'{xlo:g}') - len(f'{xhi:g}'))}{xhi:g}")
    lines.append(f"     {xlabel.center(canvas_w)}")
    lines.append("")
    for item in clean:
        marker = _paint(str(item.get("char", "*"))[:1], item.get("color"), use_color)
        lines.append(f"  {marker}  {item.get('label', '')}")
    return "\n".join(lines)
