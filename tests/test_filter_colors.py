"""Display colours: true colours for visible bands, ordered false colours for the rest."""

import colorsys

import numpy as np
import pytest

from sed_tools.filter_colors import display_colors, true_color, visible_fraction


def box(lo, hi, step=10.0):
    """A top-hat curve from lo to hi Angstrom with zero edges."""
    lam = np.arange(lo - step, hi + 2 * step, step)
    return lam, ((lam >= lo) & (lam <= hi)).astype(float)


def hue(color):
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 360.0 * colorsys.rgb_to_hls(r, g, b)[0]


def test_all_ultraviolet_set_runs_purple_to_sky_blue():
    colors = display_colors({"FUV": box(1350, 1750), "NUV": box(1750, 2800)})
    assert hue(colors["NUV"]["DisplayColor"]) < hue(colors["FUV"]["DisplayColor"])
    assert 180 < hue(colors["NUV"]["DisplayColor"]) < 210 and 280 < hue(colors["FUV"]["DisplayColor"]) < 300
