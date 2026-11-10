"""Golden: every case in tests/golden/cases.py reproduces its recorded outputs byte for byte.

A failing row names the first differing file and, for numeric files, the largest difference. The cases run
only where the outputs were recorded (expected/environment.json); elsewhere the whole test skips once, after
checking that sed_tools comes from this checkout. Do not edit tests/golden/expected/.
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
from record import environment, run  # noqa: E402

RECORDED = json.loads((EXPECTED / "environment.json").read_text())


def _from_this_checkout(t):
    import sed_tools

    assert Path(sed_tools.__file__).resolve().parent == Path(__file__).resolve().parent.parent / "sed_tools"


def _files(directory):
    return sorted(p.relative_to(directory) for p in directory.rglob("*") if p.is_file())


def _reproduces(t, name):
    """The case run as record.py runs it gives the same files, byte for byte."""
    out, expected_dir = run(name, t.tmp), EXPECTED / name
    expected, actual = _files(expected_dir), _files(out)
    assert actual == expected, f"{name}: files differ; missing {sorted(set(expected) - set(actual))}, " \
                               f"extra {sorted(set(actual) - set(expected))}"
    for path in expected:
        if (expected_dir / path).read_bytes() != (out / path).read_bytes():
            pytest.fail(f"{name}: {path} differs{_largest_difference(expected_dir / path, out / path)}")


def _largest_difference(old, new):
    """For numeric files, the largest absolute difference; otherwise nothing."""
    try:
        if old.suffix == ".npy":
            a, b = np.load(old), np.load(new)
        elif old.suffix in (".dat", ".txt", ".csv"):
            a, b = (np.genfromtxt(p, delimiter=None, comments="#", skip_header=1, invalid_raise=False)
                    for p in (old, new))
        else:
            return ""
        if a.shape != b.shape:
            return f" (shape {a.shape} -> {b.shape})"
        return f" (largest difference {np.nanmax(np.abs(a - b)):.3e})"
    except ValueError:
        return ""


def test_golden(rows):
    rows({"sed_tools_comes_from_this_checkout": _from_this_checkout})
    if environment() != RECORDED:
        pytest.skip(f"golden outputs were recorded with {RECORDED}; this is {environment()}")
    rows({f"case_reproduces_recorded_outputs_{name}": name for name in sorted(CASES)}, _reproduces)
