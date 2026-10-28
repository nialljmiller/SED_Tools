"""The last should-fix items before 0.5.0 (FINAL_050.md): menu answers, widths, missing ML packages,
the detector of an instrument not yet installed, and NJM's archived copies in filters-info."""

from __future__ import annotations

import pytest


def answer(monkeypatch, *replies):
    """Feed ``input()`` these replies in order; a further prompt fails the test."""
    queue = list(replies)

    def fake_input(prompt=""):
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


# ----------------------------------------------------------------------
# Config (menu 14): q, b or Enter keep the data directory
# ----------------------------------------------------------------------
@pytest.mark.parametrize("reply", ["", "q", "Q", "b", "B"])
def test_config_menu_keeps_the_data_directory(reply, monkeypatch):
    from sed_tools import cli

    answer(monkeypatch, reply)
    changed = []
    monkeypatch.setattr(cli, "set_data_dir", lambda *a, **k: changed.append(a))
    cli.run_config_flow()
    assert changed == []  # "q" used to be saved as data_dir = "q"


def test_config_menu_takes_any_other_answer_as_a_path(monkeypatch, tmp_path):
    from sed_tools import cli

    answer(monkeypatch, str(tmp_path / "q"))
    changed = []
    monkeypatch.setattr(cli, "set_data_dir", lambda path, **k: changed.append(path))
    cli.run_config_flow()
    assert changed == [str(tmp_path / "q")]


# ----------------------------------------------------------------------
# Combine grids (menu 7): q or b goes back, as in the other pickers
# ----------------------------------------------------------------------
@pytest.mark.parametrize("reply", ["q", "b", "Q", "back"])
def test_combine_grid_selection_goes_back(reply, tmp_path, monkeypatch):
    from sed_tools.combine_stellar_atm import select_models_interactive

    grids = [(name, str(tmp_path / name)) for name in ("A", "B")]
    answer(monkeypatch, reply)
    assert select_models_interactive(grids) == []  # only Ctrl-C used to leave it


def test_combine_menu_q_returns_to_the_menu_without_failing(tmp_path, monkeypatch, capsys):
    from sed_tools import cli

    for name in ("A", "B"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\na.txt,5000,4,0\n")
    answer(monkeypatch, "q")
    assert cli._run_menu_flow(lambda: cli.run_combine_flow(base_dir=str(tmp_path))) is True
    assert "No spectral grids selected." in capsys.readouterr().out
    assert not (tmp_path / "combined_models").exists()


# ----------------------------------------------------------------------
# Lines no wider than an 80-column terminal (paths may be longer)
# ----------------------------------------------------------------------
def _wide_lines(text, *allowed):
    """Lines over 80 columns that contain none of the ``allowed`` substrings (paths)."""
    return [line for line in text.splitlines() if len(line) > 80 and not any(a in line for a in allowed)]


def test_filter_combine_intro_fits_80_columns(tmp_path, monkeypatch, capsys):
    from sed_tools.cli import run_filter_combine_flow

    answer(monkeypatch, "")
    run_filter_combine_flow(base_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert "MESA-compatible filter folder and index file" in out
    assert _wide_lines(out, str(tmp_path)) == []  # was 87 wide


def test_densifier_cube_list_fits_80_columns(tmp_path, monkeypatch, capsys):
    import numpy as np

    from conftest import write_flux_cube
    from sed_tools.grid_densifier import run_interactive_workflow

    name = "a-rather-long-spectral-grid-name_dense"
    write_flux_cube(tmp_path / name / "flux_cube.bin", [5750, 6000, 6250], [4.0], [0.0], [1000.0, 2000.0],
                    np.ones((3, 1, 1, 2)))
    answer(monkeypatch, "")
    run_interactive_workflow(base_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert f"   1) {name}/flux_cube.bin\n      Teff 5750–6250 K  max gap 250 K  (3 pts)\n" in out
    assert _wide_lines(out, str(tmp_path)) == []  # was 92 wide with a short name


def test_mesa_prepare_invalid_choice_fits_80_columns(tmp_path, monkeypatch):
    from sed_tools import mesa_prepare

    (tmp_path / "Kurucz").mkdir()
    (tmp_path / "Kurucz" / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\na.txt,5000,4,0\n")
    answer(monkeypatch, "")
    with pytest.raises(SystemExit) as excinfo:
        mesa_prepare.run_interactive(base_dir=str(tmp_path))
    message = str(excinfo.value.code)
    assert message == ("Error: invalid spectral grid selection ''.\n"
                       "Enter a number from 1 to 1 or a grid name. Nothing was exported.")
    assert _wide_lines(message) == []  # was 107 wide
