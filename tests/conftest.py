"""pytest glue: the ``rows`` fixture every area test takes (support.Rows)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from support import Rows  # noqa: E402


@pytest.fixture
def rows(subtests, tmp_path_factory, capsys, caplog):
    """Run a table's rows as named subtests, each isolated as a test of its own."""
    return Rows(subtests, tmp_path_factory, capsys, caplog)
