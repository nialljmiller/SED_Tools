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


def test_flat_curve_is_white_and_lines_follow_the_spectrum():
    assert true_color([3000.0, 10000.0], [1.0, 1.0]) == "#ffffff"
    line = {nm: hue(true_color([nm * 10 - 5, nm * 10, nm * 10 + 5], [0, 1, 0])) for nm in (450, 530, 580, 656)}
    assert 220 < line[450] < 270      # blue
    assert 90 < line[530] < 150       # green
    assert 30 < line[580] < 60        # amber
    assert line[656] < 10 or line[656] > 350  # H-alpha red


def test_visible_fraction_counts_only_380_to_750_nm():
    assert visible_fraction(*box(5000, 6000)) == pytest.approx(1.0)
    assert visible_fraction(*box(12000, 14000)) == 0.0
    lam, t = box(6500, 8500)
    assert visible_fraction(lam, t) == pytest.approx(0.5, abs=0.02)
    # A narrow band whose samples start inside its passband stays at most 1.
    assert visible_fraction([5000.0, 5005.0, 5010.0], [0.8, 0.9, 0.8]) == pytest.approx(1.0)


def test_visible_bands_get_true_colours_others_ordered_false_colours():
    colors = display_colors({"U": box(3300, 3900), "B": box(3900, 4900), "V": box(5000, 5900),
                             "I": box(7600, 9000), "J": box(11000, 14000), "K": box(20000, 23000)})
    assert {b for b, c in colors.items() if c["ColorMethod"] == "true"} == {"B", "V"}
    assert 200 < hue(colors["B"]["DisplayColor"]) < 260 and 60 < hue(colors["V"]["DisplayColor"]) < 120
    # Infrared continues past the reddest visible band, darkening outward.
    lightness = [colorsys.rgb_to_hls(*(int(colors[b]["DisplayColor"][i:i + 2], 16) / 255 for i in (1, 3, 5)))[1]
                 for b in ("I", "J", "K")]
    assert lightness == sorted(lightness, reverse=True)
    assert 250 < hue(colors["U"]["DisplayColor"]) < 310


def test_all_infrared_sets_span_a_red_shifted_rainbow_in_order():
    colors = display_colors({"Ks": box(20000, 23000), "J": box(11000, 14000), "H": box(15000, 18000)})
    assert all(c["ColorMethod"] == "false" for c in colors.values())
    j, h, ks = (hue(colors[b]["DisplayColor"]) for b in ("J", "H", "Ks"))
    assert 80 < j < 120 and 30 < h < 60 and (ks > 320 or ks < 10)


def test_all_ultraviolet_set_runs_purple_to_sky_blue():
    colors = display_colors({"FUV": box(1350, 1750), "NUV": box(1750, 2800)})
    assert hue(colors["NUV"]["DisplayColor"]) < hue(colors["FUV"]["DisplayColor"])
    assert 180 < hue(colors["NUV"]["DisplayColor"]) < 210 and 280 < hue(colors["FUV"]["DisplayColor"]) < 300
