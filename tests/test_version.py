from importlib.metadata import version

import sed_tools


def test_package_version_matches_distribution_metadata():
    assert sed_tools.__version__ == version("sed-tools")
