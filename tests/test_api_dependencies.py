"""Which packages sed_tools needs: psutil is not a dependency, and scipy, optuna and the other
extras are optional; importing sed_tools and measuring the RAM work without them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Packaging: psutil is not a dependency; scipy (the [full] extra) is not needed to import
# ---------------------------------------------------------------------
def test_psutil_is_not_a_dependency_and_scipy_is_not_required():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: the tomli dependency
        import tomli as tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    assert not any(dep.startswith("psutil") for dep in project["dependencies"])
    assert not any(dep.startswith("psutil") for deps in extras.values() for dep in deps)
    assert not any(dep.startswith("scipy") for dep in project["dependencies"])
    assert any(dep.startswith("scipy") for dep in extras["full"])


_WITHOUT_PSUTIL = """
import sys
sys.modules["psutil"] = None  # import psutil now raises ImportError

import sed_tools
from sed_tools._constants import available_memory_bytes

assert not [m for m, mod in sys.modules.items() if m.startswith("psutil") and mod is not None]
print(available_memory_bytes())
"""


def test_import_sed_tools_and_measure_ram_without_psutil():
    done = subprocess.run([sys.executable, "-c", _WITHOUT_PSUTIL], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    # On Linux and macOS the RAM is measured without psutil, so nothing is assumed or warned.
    assert "Could not measure available RAM" not in done.stderr
    assert int(done.stdout) > 0


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


# ---------------------------------------------------------------------
# ml_generator --auto: models SEDGenerator can load
# ---------------------------------------------------------------------
def test_optuna_is_an_optional_extra_only():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: the tomli dependency
        import tomli as tomllib

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]
    assert project["optional-dependencies"]["ml-tune"] == ["optuna>=3"]
    assert not any("optuna" in dep for dep in project["dependencies"] + project["optional-dependencies"]["full"])


# ----------------------------------------------------------------------
# ML without scipy, torch or scikit-learn: one line naming what to install, no traceback
# ----------------------------------------------------------------------
def test_import_sed_tools_needs_no_optional_package(tmp_path):
    import os
    import site
    import subprocess
    import sys
    from pathlib import Path

    optional = ("scipy", "torch", "sklearn", "optuna")
    code = ("import sys\n"
            f"blocked = {optional!r}\n"
            "if sys.argv[1] == 'blocked':\n"
            "    for name in blocked: sys.modules[name] = None\n"
            "import sed_tools, sed_tools.cli, sed_tools.ml, sed_tools.ml_sed_generator, sed_tools.grid_densifier\n"
            "import sed_tools.ml_sed_completer\n"
            "import sed_tools.combine_stellar_atm\n"
            "print(sorted(name for name in blocked if sys.modules.get(name) is not None))\n")
    import matplotlib

    # HOME keeps the user's ~/.sed_tools out of reach; the packages and matplotlib's font cache stay where they are
    env = dict(os.environ, HOME=str(tmp_path), PYTHONUSERBASE=site.getuserbase(),
               MPLCONFIGDIR=matplotlib.get_configdir())
    for mode in ("blocked", "installed"):
        done = subprocess.run([sys.executable, "-c", code, mode], cwd=Path(__file__).resolve().parents[1],
                              env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "[]"
