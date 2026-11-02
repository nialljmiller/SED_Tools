import sys
from importlib.metadata import version
from pathlib import Path

import sed_tools

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomli is a dependency on Python 3.10
    import tomli as tomllib


def test_package_version_matches_distribution_metadata():
    assert sed_tools.__version__ == version("sed-tools")


def test_package_version_matches_pyproject():
    # Both are written by hand at release time; a checkout must agree with itself.
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert sed_tools.__version__ == tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
