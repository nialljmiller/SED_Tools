"""Tests for Nill's decisions of 27 October 2026: packaging (psutil, scipy)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Packaging: psutil is required; scipy (the [full] extra) is not needed to import
# ---------------------------------------------------------------------

def test_psutil_is_a_required_dependency_and_scipy_is_not():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert any(dep.startswith("psutil>=") for dep in project["dependencies"])
    assert not any(dep.startswith("scipy") for dep in project["dependencies"])
    assert any(dep.startswith("scipy") for dep in project["optional-dependencies"]["full"])


_WITHOUT_SCIPY = """
import sys
sys.modules["scipy"] = None  # every import of scipy or a scipy submodule now raises ImportError

import numpy as np
import sed_tools
import sed_tools.combine_stellar_atm as combine

assert not [m for m, mod in sys.modules.items() if m.startswith("scipy") and mod is not None]
try:
    combine._fill_gaps(np.zeros((1, 1, 1, 1)), np.ones((1, 1, 1), bool), np.zeros((1, 1, 1), int),
                       np.ones(1), np.ones(1), np.ones(1))
except ImportError as exc:
    print("needs scipy:", exc)
"""


def test_import_sed_tools_works_without_scipy():
    done = subprocess.run([sys.executable, "-c", _WITHOUT_SCIPY], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    # The grid combination itself still needs scipy, and says so when it is called.
    assert done.stdout.startswith("needs scipy: ") and "scipy" in done.stdout[len("needs scipy: "):]
