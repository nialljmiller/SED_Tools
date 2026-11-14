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
import importlib, pkgutil, sys
for name in {blocked!r}:
    sys.modules[name] = None  # importing it now raises ImportError
import sed_tools
for _, module, _ in pkgutil.walk_packages(sed_tools.__path__, "sed_tools."):
    if not module.endswith("__main__"):  # what python -m runs: importing it would run the command
        importlib.import_module(module)
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
    # Combining grids needs no scipy: empty nodes (here 5000 K at log g 4.5) are no longer filled with its KDTree.
    "combines_grids_without_scipy": ("""
import os, sys
sys.modules["scipy"] = None  # every import of scipy or a scipy submodule now raises ImportError
import numpy as np
from sed_tools.api import SED
root = os.path.join(os.path.expanduser("~"), "models")  # the row's folder
for name, teff, logg in (("a", 5000.0, 4.0), ("b", 6000.0, 4.5)):
    os.makedirs(os.path.join(root, name))
    np.savetxt(os.path.join(root, name, "s.txt"), np.column_stack([np.linspace(1e3, 1e4, 20), np.ones(20)]))
    with open(os.path.join(root, name, "lookup_table.csv"), "w") as fh:
        fh.write(f"#file_name,teff,logg,metallicity\\ns.txt,{teff},{logg},0.0\\n")
SED.combine(["a", "b"], output="ab", model_root=root, visualization=False)
print("combined:", sorted(os.listdir(os.path.join(root, "ab"))))
""", lambda out, err: "combined: ['a_000000.txt', 'b_000001.txt', 'flux_cube.bin', 'lookup_table.csv']" in out),
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
