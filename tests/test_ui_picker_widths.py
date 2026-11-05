"""The pickers at 80 columns: a long label is shown whole when it fits and cut with an
ellipsis only when the terminal is narrower, and short names keep their columns.
"""

from __future__ import annotations

from helpers_api import no_network  # noqa: F401 (an autouse fixture)
from helpers_cli import answer


# ----------------------------------------------------------------------
# Pickers: a label is cut only when the terminal is narrower than it
# ----------------------------------------------------------------------
def _picker_page(labels, columns, monkeypatch, capsys, **kwargs):
    from sed_tools.ui_utils import prompt_choice

    monkeypatch.setenv("COLUMNS", str(columns))
    monkeypatch.setenv("LINES", "40")
    answer(monkeypatch, "q")
    assert prompt_choice(labels, "Local filter sets to combine", use_color=False, **kwargs) is None
    return capsys.readouterr().out.splitlines()


def test_combine_picker_shows_a_long_product_label_whole_at_80_columns(monkeypatch, capsys):
    long = "2MASS/2MASS/products/filter+atm_ctio_2mass_amunstated (3 filters)"
    lines = _picker_page([long, "Generic/Johnson (7 filters)"], 80, monkeypatch, capsys,
                         multi=True, page_size=50, max_cols=2)
    assert any(line.strip() == f"1. {long}" for line in lines)  # was cut to '(3 filter…'
    assert max(len(line) for line in lines) <= 80


def test_picker_cuts_with_an_ellipsis_only_when_the_terminal_is_narrower(monkeypatch, capsys):
    long = "x" * 70
    lines = _picker_page([long], 60, monkeypatch, capsys)
    row = next(line for line in lines if line.lstrip().startswith("1."))
    assert row.rstrip().endswith("…") and len(row) <= 59


def test_picker_keeps_its_columns_for_short_names(monkeypatch, capsys):
    names = [f"name{i:02d}" for i in range(30)]
    lines = _picker_page(names, 80, monkeypatch, capsys)
    rows = [line for line in lines if line.lstrip()[:1].isdigit()]
    assert len(rows) == 10 and all(line.count(".") == 3 for line in rows)  # three columns, as before


# ----------------------------------------------------------------------
# Picker: the longest labels are shown in full
# ----------------------------------------------------------------------
def test_picker_shows_the_longest_label_in_full(monkeypatch, capsys):
    from sed_tools import ui_utils

    monkeypatch.setattr(ui_utils, "terminal_width", lambda default=80: 80)
    answer(monkeypatch, "1")
    assert ui_utils.prompt_choice(["alpha", "beta"], "Things", use_color=False) == 0
    out = capsys.readouterr().out
    # The longest label used to be cut 2 characters short ("alp", "bet").
    assert "   1. alpha  " in out and "   2. beta" in out
