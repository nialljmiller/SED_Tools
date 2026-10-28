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
# ML without scipy, torch or scikit-learn: one line naming what to install, no traceback
# ----------------------------------------------------------------------
FULL = 'pip install "sed-tools[full]"'


def _without(monkeypatch, *modules):
    """Make ``modules`` uninstalled for this test: importing them (or a submodule another test
    imported already) fails, and find_spec finds nothing."""
    import sys

    for name in modules:
        for loaded in [key for key in sys.modules if key == name or key.startswith(name + ".")] or [name]:
            monkeypatch.setitem(sys.modules, loaded, None)


def _run_cli(monkeypatch, *argv):
    import sys

    from sed_tools import cli

    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    return cli.main()


def test_ml_completer_command_without_scipy_says_what_to_install(monkeypatch, capsys, tmp_path):
    _without(monkeypatch, "scipy", "torch", "sklearn")
    assert _run_cli(monkeypatch, "ml_completer", "--base", str(tmp_path), "--models-dir", str(tmp_path)) == 1
    err = capsys.readouterr().err
    assert err == f"Error: missing scipy, torch, scikit-learn: {FULL}\n"  # was a scipy traceback
    assert len(err.rstrip("\n")) <= 80


def test_ml_completer_names_only_what_is_missing(monkeypatch, capsys, tmp_path):
    _without(monkeypatch, "scipy")
    assert _run_cli(monkeypatch, "ml_completer", "--base", str(tmp_path), "--models-dir", str(tmp_path)) == 1
    assert capsys.readouterr().err == f"Error: missing scipy: {FULL}\n"


def test_ml_generator_command_without_torch_says_what_to_install(monkeypatch, capsys, tmp_path):
    _without(monkeypatch, "torch", "sklearn")
    assert _run_cli(monkeypatch, "ml_generator", "--base", str(tmp_path), "--models-dir", str(tmp_path)) == 1
    assert capsys.readouterr().err == f"Error: missing torch, scikit-learn: {FULL}\n"


def test_ml_generator_auto_without_torch_says_what_to_install(monkeypatch, capsys, tmp_path):
    import importlib.util

    if importlib.util.find_spec("optuna") is None:
        pytest.skip("optuna is checked first")
    _without(monkeypatch, "torch")
    assert _run_cli(monkeypatch, "ml_generator", "--auto", "--library", str(tmp_path),
                    "--output", str(tmp_path / "out")) == 1
    assert capsys.readouterr().err == f"Error: missing torch: {FULL}\n"  # was a torch traceback


def _densifier_inputs(tmp_path):
    import numpy as np

    from conftest import write_flux_cube

    write_flux_cube(tmp_path / "Kurucz" / "flux_cube.bin", [5000, 7000], [4.0], [0.0], [1000.0, 2000.0],
                    np.ones((2, 1, 1, 2)))
    model = tmp_path / "models" / "gen"
    model.mkdir(parents=True)
    return model


def test_densify_with_an_ml_model_without_torch_says_what_to_install(monkeypatch, capsys, tmp_path):
    model = _densifier_inputs(tmp_path)
    _without(monkeypatch, "torch")
    assert _run_cli(monkeypatch, "densify", "--flux-cube", str(tmp_path / "Kurucz" / "flux_cube.bin"),
                    "--output", str(tmp_path / "dense" / "flux_cube.bin"), "--ml-model", str(model)) == 1
    assert capsys.readouterr().err == f"Error: missing torch: {FULL}\n"
    assert not (tmp_path / "dense").exists()


def test_menu_ml_options_without_packages_return_to_the_menu(monkeypatch, capsys, tmp_path):
    from sed_tools import cli

    _without(monkeypatch, "scipy", "torch", "sklearn")
    model = _densifier_inputs(tmp_path)
    densifier = cli.run_grid_densifier_flow
    monkeypatch.setattr(cli, "run_grid_densifier_flow", lambda: densifier(base_dir=str(tmp_path)))
    # 8 and 9 stop before any question; 10 asks for a cube, spacing, model and output first
    answer(monkeypatch, "8", "9", "10", "1", "", str(model), str(tmp_path / "dense" / "flux_cube.bin"), "q")
    assert cli._run_menu() == 1  # each failed, so the session ends with status 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [f"Error: missing scipy, torch, scikit-learn: {FULL}",
                                         f"Error: missing torch, scikit-learn: {FULL}",
                                         f"Error: missing torch: {FULL}"]
    assert "Traceback" not in captured.out + captured.err
    assert captured.out.count("0/q) Quit") == 4  # the menu came back after each


def test_combine_without_scipy_says_what_to_install(monkeypatch, capsys):
    from sed_tools import cli

    def needs_scipy():
        from scipy.spatial import KDTree  # noqa: F401  (as combine_stellar_atm does)

    _without(monkeypatch, "scipy")
    assert cli._run_menu_flow(needs_scipy) is False
    assert capsys.readouterr().err == f"Error: missing scipy: {FULL}\n"


def test_a_missing_module_that_is_not_optional_keeps_its_traceback():
    from sed_tools import cli

    def broken():
        raise ModuleNotFoundError("No module named 'sed_tools_typo'", name="sed_tools_typo")

    with pytest.raises(ModuleNotFoundError):
        cli._run_menu_flow(broken)


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
