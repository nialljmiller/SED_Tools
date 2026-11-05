"""Helpers shared by the CLI, cube, config and photometry tests: input() answers, a check for
lines wider than 80 columns, a lookup-table writer, a flat filter, and simulated missing packages.
"""

from __future__ import annotations

import numpy as np


def answer(monkeypatch, *replies):
    """Feed ``input()`` these replies in order; a further prompt fails the test."""
    queue = list(replies)

    def fake_input(prompt=""):
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


def write_lookup(model_dir, header, rows):
    model_dir.mkdir(parents=True)
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    (model_dir / "lookup_table.csv").write_text("\n".join(lines) + "\n")


def flat_filter(name="V"):
    from sed_tools.flux_cube_tool import FilterCurve

    wavelength = np.arange(4000.0, 6001.0, 100.0)
    return FilterCurve(name, wavelength, np.ones_like(wavelength), {})


def _wide_lines(text, *allowed):
    """Lines over 80 columns that contain none of the ``allowed`` substrings (paths)."""
    return [line for line in text.splitlines() if len(line) > 80 and not any(a in line for a in allowed)]
