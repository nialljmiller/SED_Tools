"""Every golden case must reproduce its recorded outputs byte for byte.

See tests/golden/cases.py. A failure names the first differing file and, for
numeric files, the largest difference. Do not edit tests/golden/expected/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "golden"
EXPECTED = GOLDEN / "expected"
sys.path.insert(0, str(GOLDEN))

from cases import CASES  # noqa: E402
from record import environment  # noqa: E402

RECORDED = json.loads((EXPECTED / "environment.json").read_text())


def test_sed_tools_comes_from_this_checkout():
    import sed_tools

    assert Path(sed_tools.__file__).resolve().parent == Path(__file__).resolve().parent.parent / "sed_tools"


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_reproduces_recorded_outputs(name, tmp_path):
    if environment() != RECORDED:
        pytest.skip(f"golden outputs were recorded with {RECORDED}; this is {environment()}")
    work, out = tmp_path / "work", tmp_path / "out"
    work.mkdir()
    out.mkdir()
    CASES[name](work, out)

    expected_dir = EXPECTED / name
    expected = sorted(p.relative_to(expected_dir) for p in expected_dir.rglob("*") if p.is_file())
    actual = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    assert actual == expected, f"{name}: files differ; missing {sorted(set(expected) - set(actual))}, " \
                               f"extra {sorted(set(actual) - set(expected))}"
    for path in expected:
        old, new = (expected_dir / path).read_bytes(), (out / path).read_bytes()
        if old != new:
            pytest.fail(f"{name}: {path} differs{_largest_difference(expected_dir / path, out / path)}")


def _largest_difference(old, new):
    """For numeric files, the largest absolute difference; otherwise nothing."""
    try:
        if old.suffix == ".npy":
            a, b = np.load(old), np.load(new)
        elif old.suffix in (".dat", ".txt", ".csv"):
            a = np.genfromtxt(old, delimiter=None, comments="#", skip_header=1, invalid_raise=False)
            b = np.genfromtxt(new, delimiter=None, comments="#", skip_header=1, invalid_raise=False)
        else:
            return ""
        if a.shape != b.shape:
            return f" (shape {a.shape} -> {b.shape})"
        return f" (largest difference {np.nanmax(np.abs(a - b)):.3e})"
    except ValueError:
        return ""
