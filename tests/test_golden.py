"""Golden: every case in tests/golden/cases.py reproduces its recorded outputs.

Where the outputs were recorded (expected/environment.json) every file must match byte for byte. Elsewhere (CI on
Linux, other numpy and scipy versions) every file must still match byte for byte, except the files in TOLERANT, whose
last bits differ between platforms: there a flux cube's header and a .npy file's header must match exactly and its
float64 values to within RTOL; a text file must hold the same text and its numbers within RTOL. The cases in
PYTHON_TEXT record Python's own text and run only on the recorded Python version. A failing row names the first
differing file and the largest difference. Do not edit tests/golden/expected/.
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
#: Off the recorded environment, the largest relative difference allowed between two numbers of a TOLERANT file.
#: Linux CI (Python 3.10 and 3.12, numpy 2.5.5) differed from the recording by at most 1.7e-14 (25 Nov 2026); the
#: smallest error golden must still catch, AB 3631 -> 3630 Jy, moves a magnitude by 8e-6 of itself.
RTOL = 1e-9
#: The files whose last bits differ between platforms (measured on Linux CI, 25 Nov 2026), relative to expected/.
TOLERANT = ("*.bin", "*.npy", "filter_properties/*.json", "flux_cube_tool_band_averages/*.json",
            "interpolation_and_photometry/photometry.json", "compose_ztf_qe_variants/*/*/*.dat")
#: Cases that record Python's own text (argparse help, inspect signatures), which differs between Python versions.
PYTHON_TEXT = ("public_interface",)
NUMBER = re.compile(rb"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _from_this_checkout(t):
    import sed_tools

    assert Path(sed_tools.__file__).resolve().parent == Path(__file__).resolve().parent.parent / "sed_tools"


def _files(directory):
    return sorted(p.relative_to(directory) for p in directory.rglob("*") if p.is_file())


def _reproduces(t, name):
    """The case run as record.py runs it gives the same files: byte for byte, except a TOLERANT file off the
    recorded environment, which must be the same to within RTOL."""
    exact = environment() == RECORDED
    if name in PYTHON_TEXT and _minor(environment()["python"]) != _minor(RECORDED["python"]):
        pytest.skip(f"{name} records Python {RECORDED['python']}'s own text; this is {environment()['python']}")
    out, expected_dir = run(name, t.tmp), EXPECTED / name
    expected, actual = _files(expected_dir), _files(out)
    assert actual == expected, f"{name}: files differ; missing {sorted(set(expected) - set(actual))}, " \
                               f"extra {sorted(set(actual) - set(expected))}"
    for path in expected:
        old, new = expected_dir / path, out / path
        if old.read_bytes() == new.read_bytes():
            continue
        tolerant = not exact and any((Path(name) / path).match(pattern) for pattern in TOLERANT)
        worst, where = _difference(old, new) if tolerant else (np.inf, "")
        if worst > RTOL:
            pytest.fail(f"{name}: {path} differs{_largest_difference(old, new)}"
                        + (f"; largest relative difference {worst:.1e}, {where}" if tolerant else ""))


def _minor(version):
    return version.rsplit(".", 1)[0]


def _difference(old, new):
    """(the largest relative difference between the numbers of the files ``old`` and ``new``, where). Float64
    values are compared after a header that must be identical, printed numbers between text that must be
    identical; any other difference is infinite."""
    x, y = old.read_bytes(), new.read_bytes()
    if old.suffix in (".bin", ".npy"):
        start = 10 + int.from_bytes(x[8:10], "little") if old.suffix == ".npy" else read_header(old).offset
        if len(x) != len(y) or x[:start] != y[:start] or (old.suffix == ".npy" and b"'<f8'" not in x[:start]):
            return np.inf, "the header or the size differs"
        a, b = (np.frombuffer(data[start:], dtype="<f8") for data in (x, y))
    elif NUMBER.split(x) != NUMBER.split(y):
        return np.inf, "the text between the numbers differs"
    else:
        a, b = (np.array([float(v) for v in NUMBER.findall(data)]) for data in (x, y))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(a - b) / np.maximum(np.abs(a), np.abs(b))
    relative = np.nan_to_num(np.where((a == b) | (np.isnan(a) & np.isnan(b)), 0.0, relative), nan=np.inf)
    i = int(np.argmax(relative)) if len(a) else 0
    return (float(relative[i]), f"number {i}: {a[i]!r} recorded, {b[i]!r} now") if len(a) else (0.0, "")


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
