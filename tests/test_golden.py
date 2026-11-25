"""Golden: every case in tests/golden/cases.py reproduces its recorded outputs.

Where the outputs were recorded (expected/environment.json) every file must match byte for byte. Elsewhere (CI on
Linux, other numpy or scipy versions) a file that differs must hold the same text and the same numbers to within
RTOL of each other: a flux cube's header and a .npy array's shape and type exactly, its float64 values within RTOL;
in a text file everything but the numbers exactly, each number within RTOL or within one unit of its last printed
digit. A failing row names the first differing file and the largest difference. Do not edit tests/golden/expected/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "golden"
EXPECTED = GOLDEN / "expected"
sys.path.insert(0, str(GOLDEN))

from cases import CASES  # noqa: E402
from record import environment, run  # noqa: E402

from sed_tools.cube.fileio import read_header  # noqa: E402

RECORDED = json.loads((EXPECTED / "environment.json").read_text())
#: Off the recorded environment, the largest relative difference allowed between two numbers.
RTOL = 1e-9
NUMBER = re.compile(rb"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _from_this_checkout(t):
    import sed_tools

    assert Path(sed_tools.__file__).resolve().parent == Path(__file__).resolve().parent.parent / "sed_tools"


def _files(directory):
    return sorted(p.relative_to(directory) for p in directory.rglob("*") if p.is_file())


def _reproduces(t, name):
    """The case run as record.py runs it gives the same files: byte for byte where they were recorded, the same to
    within RTOL elsewhere."""
    out, expected_dir = run(name, t.tmp), EXPECTED / name
    expected, actual = _files(expected_dir), _files(out)
    assert actual == expected, f"{name}: files differ; missing {sorted(set(expected) - set(actual))}, " \
                               f"extra {sorted(set(actual) - set(expected))}"
    exact = environment() == RECORDED
    for path in expected:
        old, new = expected_dir / path, out / path
        if old.read_bytes() != new.read_bytes() and (exact or _difference(old, new)[0] > 1):
            pytest.fail(f"{name}: {path} differs{_largest_difference(old, new)}"
                        + ("" if exact else f"; off the recorded environment, {_difference(old, new)[1]}"))


def _difference(old, new):
    """(the largest difference between the files ``old`` and ``new`` in units of what is allowed, which numbers),
    for the comparison off the recorded environment. Float64 values are allowed RTOL, printed numbers RTOL or one
    unit of their last digit; anything else (text, a header, a shape) must be identical, or the difference is
    infinite."""
    x, y = old.read_bytes(), new.read_bytes()
    if old.suffix in (".bin", ".npy"):
        start = 10 + int.from_bytes(x[8:10], "little") if old.suffix == ".npy" else read_header(old).offset
        if len(x) != len(y) or x[:start] != y[:start] or (old.suffix == ".npy" and b"'<f8'" not in x[:start]):
            return np.inf, "the header or the size differs"
        a, b = (np.frombuffer(data[start:], dtype="<f8") for data in (x, y))
        with np.errstate(divide="ignore", invalid="ignore"):
            units = np.abs(a - b) / (RTOL * np.maximum(np.abs(a), np.abs(b)))
        units = np.nan_to_num(np.where((a == b) | (np.isnan(a) & np.isnan(b)), 0.0, units), nan=np.inf)
        i = int(np.argmax(units))
        return float(units[i]), f"value {i}: {a[i]!r} recorded, {b[i]!r} now"
    if NUMBER.split(x) != NUMBER.split(y):
        return np.inf, "the text between the numbers differs"
    worst = (0.0, "")
    for u, v in zip(NUMBER.findall(x), NUMBER.findall(y)):
        a, b = float(u), float(v)
        if a != b:
            allowed = max(RTOL * max(abs(a), abs(b)), min(_last_digit(u), _last_digit(v)))
            units = abs(a - b) / allowed if allowed else np.inf
            worst = max(worst, (units, f"{u.decode()} recorded, {v.decode()} now"))
    return worst


def _last_digit(token):
    """One unit of the last printed digit of a number written with a decimal point; 0 for an integer, which must be
    identical."""
    mantissa, _, exponent = token.lower().partition(b"e")
    if b"." not in mantissa:
        return 0.0
    return 10.0 ** (int(exponent or 0) - len(mantissa.split(b".")[1]))


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
    rows({f"case_reproduces_recorded_outputs_{name}": name for name in sorted(CASES)}, _reproduces)
