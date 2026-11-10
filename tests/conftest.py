"""pytest glue: the ``rows`` fixture every area test takes (support.Rows), and, until the old test files
are converted, the fixtures they use."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from support import AffineCube, Rows, affine_cube, write_flux_cube  # noqa: E402, F401 (old files import these)


@pytest.fixture
def rows(subtests, tmp_path_factory, capsys, caplog):
    """Run a table's rows as named subtests, each isolated as a test of its own."""
    return Rows(subtests, tmp_path_factory, capsys, caplog)


# ----------------------------------------------------------------------
# For the old test files; these go with the last of them.
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _private_config_file(tmp_path_factory, monkeypatch):
    """No test reads or writes the user's own ~/.sed_tools/config.toml."""
    from sed_tools import config

    config_dir = tmp_path_factory.mktemp("sed_tools_config")
    monkeypatch.setattr(config, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "_CONFIG_FILE", config_dir / "config.toml")


@pytest.fixture(autouse=True)
def _fresh_njm_state():
    """What a process knows about NJM and SVO is shared by its clients; each test starts as a new process."""
    from sed_tools import njm_filter_grabber, svo_filter_grabber

    njm_filter_grabber._STATE.reset()
    svo_filter_grabber._STATE.reset()
    yield
    njm_filter_grabber._STATE.reset()
    svo_filter_grabber._STATE.reset()


@pytest.fixture
def cube_writer():
    return write_flux_cube


@pytest.fixture
def make_affine_cube(tmp_path):
    return lambda **kwargs: affine_cube(tmp_path, **kwargs)
