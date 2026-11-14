"""Packaging and version: sed_tools' version agrees with itself; psutil is no dependency; scipy, torch,
scikit-learn and optuna are optional, and sed_tools imports and measures the RAM without them."""

import os
import site
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import sed_tools

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomli is a dependency on Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = ("scipy", "torch", "sklearn", "optuna")


def _project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _deps(*extras):
    """The dependencies, or with names the dependencies plus those extras, from pyproject.toml."""
    project = _project()
    optional = project["optional-dependencies"]
    return project["dependencies"] + [d for name in extras for d in optional[name]]


def _undocumented():
    """The public modules of each subpackage the API page documents (one section per module) that have no section."""
    import pkgutil
    import re

    documented = set(re.findall(r"automodule:: (\S+)", (ROOT / "docs" / "source" / "sed_tools.rst").read_text()))
    folders = {name: ROOT / name.replace(".", "/") for name in documented if name.count(".") == 1}
    return sorted(f"{package}.{module.name}" for package, folder in folders.items() if folder.is_dir()
                  for module in pkgutil.iter_modules([str(folder)])
                  if not module.name.startswith("_") and f"{package}.{module.name}" not in documented)


#: name: (what is read, what it must be). Both versions are written by hand at release time; a checkout
#: must agree with itself and with the installed metadata. Every public module of a documented subpackage
#: has its section on the API page.
FACTS = {
    "version_matches_pyproject": (lambda: _project()["version"], sed_tools.__version__),
    "version_matches_distribution_metadata": (lambda: version("sed-tools"), sed_tools.__version__),
    "psutil_is_no_dependency_or_extra": (
        lambda: [d for d in _deps(*_project()["optional-dependencies"]) if d.startswith("psutil")], []),
    "scipy_is_not_a_dependency": (lambda: [d for d in _deps() if d.startswith("scipy")], []),
    "scipy_is_in_the_full_extra": (lambda: any(d.startswith("scipy") for d in _deps("full")), True),
    "optuna_is_the_ml_tune_extra": (lambda: _project()["optional-dependencies"]["ml-tune"], ["optuna>=3"]),
    "optuna_is_neither_a_dependency_nor_in_full": (lambda: [d for d in _deps("full") if "optuna" in d], []),
    "every_public_module_of_a_documented_subpackage_has_an_api_section": (_undocumented, []),
}

_IMPORT_ALL = """
import sys
for name in {blocked!r}:
    sys.modules[name] = None  # importing it now raises ImportError
import sed_tools, sed_tools.cli, sed_tools.ml, sed_tools.ml.generator, sed_tools.cube.densify
import sed_tools.ml.completer
import sed_tools.cube.combine
print(sorted(name for name in {optional!r} if sys.modules.get(name) is not None))
"""

#: name: (a script for a fresh interpreter, which must exit 0, and a check of what it prints).
SCRIPTS = {
    # On Linux and macOS the RAM is measured without psutil, so nothing is assumed or warned.
    "imports_and_measures_the_ram_without_psutil": ("""
import sys
sys.modules["psutil"] = None
import sed_tools
from sed_tools._constants import available_memory_bytes
assert not [m for m, mod in sys.modules.items() if m.startswith("psutil") and mod is not None]
print(available_memory_bytes())
""", lambda out, err: int(out) > 0 and "Could not measure available RAM" not in err),
    # The grid combination itself still needs scipy, and says so when it is called.
    "imports_without_scipy_and_combining_grids_says_it_needs_it": ("""
import sys
sys.modules["scipy"] = None  # every import of scipy or a scipy submodule now raises ImportError
import numpy as np
import sed_tools
import sed_tools.cube.combine as combine
assert not [m for m, mod in sys.modules.items() if m.startswith("scipy") and mod is not None]
try:
    combine._fill_gaps(np.zeros((1, 1, 1, 1)), np.ones((1, 1, 1), bool), np.zeros((1, 1, 1), int),
                       np.ones(1), np.ones(1), np.ones(1))
except ImportError as exc:
    print("needs scipy:", exc)
""", lambda out, err: out.startswith("needs scipy: ") and "scipy" in out[len("needs scipy: "):]),
    # Blocked, every module still imports; installed, importing them loads none of the optional packages.
    "imports_every_module_with_the_optional_packages_blocked": (
        _IMPORT_ALL.format(blocked=OPTIONAL, optional=OPTIONAL), lambda out, err: out.strip() == "[]"),
    "imports_every_module_without_loading_the_optional_packages": (
        _IMPORT_ALL.format(blocked=(), optional=OPTIONAL), lambda out, err: out.strip() == "[]"),
}


def _fact(t, read, expected):
    assert read() == expected


def _runs(t, script, check):
    import matplotlib

    # HOME keeps the user's ~/.sed_tools out of reach; the packages and matplotlib's font cache stay put.
    env = dict(os.environ, HOME=str(t.tmp), PYTHONUSERBASE=site.getuserbase(), MPLCONFIGDIR=matplotlib.get_configdir())
    done = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert check(done.stdout, done.stderr), (done.stdout, done.stderr)


def test_packaging(rows):
    rows(FACTS, _fact)
    rows(SCRIPTS, _runs)
