"""The CLI without its optional packages: models listed with any partial install, an ML
action refused only when it is chosen, what to install for torch or scipy, and tracebacks kept for
a module that is not optional; and the filter-combine intro at 80 columns.
"""

from __future__ import annotations

import pytest

from helpers_cli import _wide_lines, answer


# ----------------------------------------------------------------------
# Lines no wider than an 80-column terminal (paths may be longer)
# ----------------------------------------------------------------------
def test_filter_combine_intro_fits_80_columns(tmp_path, monkeypatch, capsys):
    from sed_tools.cli import run_filter_combine_flow

    answer(monkeypatch, "")
    run_filter_combine_flow(base_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert "MESA-compatible filter folder and index file" in out
    assert _wide_lines(out, str(tmp_path)) == []  # was 87 wide


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


#: The packages missing in each partial install: none of the three, scipy only, torch only.
INSTALLS = {"none": ("scipy", "torch", "sklearn"), "scipy-only": ("torch", "sklearn"),
            "torch-only": ("scipy", "sklearn")}


def _trained_models(tmp_path):
    """A models directory holding one completer and one generator, as list_models reads them."""
    models = tmp_path / "models"
    (models / "my_completer_model").mkdir(parents=True)
    (models / "my_completer_model" / "config.json").write_text('{"version": "1", "framework": "pytorch"}')
    (models / "my_generator_model").mkdir()
    (models / "my_generator_model" / "config.json").write_text('{"model_type": "sed_generator"}')
    return models


@pytest.mark.parametrize("install", INSTALLS)
def test_listing_models_works_with_any_partial_install(install, monkeypatch, capsys, tmp_path):
    # 0.4.1 listed models with scipy but no torch; that must keep working (and now needs neither)
    _without(monkeypatch, *INSTALLS[install])
    models = _trained_models(tmp_path)
    answer(monkeypatch, "3")
    assert _run_cli(monkeypatch, "ml_completer", "--base", str(tmp_path), "--models-dir", str(models)) == 0
    answer(monkeypatch, "4")
    assert _run_cli(monkeypatch, "ml_generator", "--base", str(tmp_path), "--models-dir", str(models)) == 0
    captured = capsys.readouterr()
    assert "my_completer_model" in captured.out and "my_generator_model" in captured.out
    assert captured.err == ""


# (command, menu choice, what each install lacks for it); None: the action is not refused
REFUSALS = [
    ("ml_completer", "1", {"none": "torch, scikit-learn", "scipy-only": "torch, scikit-learn",
                           "torch-only": "scikit-learn"}),
    ("ml_completer", "2", {"none": "scipy, torch", "scipy-only": "torch", "torch-only": "scipy"}),
    ("ml_generator", "1", {"none": "torch, scikit-learn", "scipy-only": "torch, scikit-learn",
                           "torch-only": "scikit-learn"}),
    ("ml_generator", "2", {"none": "torch", "scipy-only": "torch", "torch-only": None}),
    ("ml_generator", "3", {"none": "torch", "scipy-only": "torch", "torch-only": None}),
]


@pytest.mark.parametrize("install", INSTALLS)
@pytest.mark.parametrize("command, choice, lacking", REFUSALS)
def test_an_ml_action_is_refused_when_chosen_if_it_needs_a_missing_package(
        command, choice, lacking, install, monkeypatch, capsys, tmp_path):
    _without(monkeypatch, *INSTALLS[install])
    answer(monkeypatch, choice)  # refused before any of the action's questions (a further one fails)
    status = _run_cli(monkeypatch, command, "--base", str(tmp_path), "--models-dir", str(tmp_path / "models"))
    err = capsys.readouterr().err
    if lacking[install] is None:  # generating needs only torch: it goes on (here: no generator to choose)
        assert status == 0 and err == ""
    else:
        assert status == 1 and err == f"Error: missing {lacking[install]}: {FULL}\n"


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


def _menu_ml_flows_in(cli, tmp_path, monkeypatch):
    """Point menu options 8, 9 and 10 at ``tmp_path``: they create their grid and models folders."""
    flows = {name: getattr(cli, name) for name in ("run_ml_completer_flow", "run_ml_generator_flow")}
    for name, flow in flows.items():
        monkeypatch.setattr(cli, name, lambda flow=flow: flow(str(tmp_path), str(tmp_path / "models")))
    densifier = cli.run_grid_densifier_flow
    monkeypatch.setattr(cli, "run_grid_densifier_flow", lambda: densifier(base_dir=str(tmp_path)))


def test_menu_ml_options_open_and_refuse_only_the_action_that_needs_a_package(monkeypatch, capsys, tmp_path):
    from sed_tools import cli

    _without(monkeypatch, *INSTALLS["none"])
    model = _densifier_inputs(tmp_path)
    _menu_ml_flows_in(cli, tmp_path, monkeypatch)
    # 8 and 9 open and go back with 0, as in 0.4.1; their training is refused when chosen; 10 is refused
    # once an ML model is named, before the output question
    answer(monkeypatch, "8", "0", "9", "0", "8", "1", "9", "1", "10", "1", "", str(model), "q")
    assert cli._run_menu() == 1  # the refused actions failed, so the session ends with status 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [f"Error: missing torch, scikit-learn: {FULL}"] * 2 + [
        f"Error: missing torch: {FULL}"]
    assert "Traceback" not in captured.out + captured.err
    assert captured.out.count("0/q) Quit") == 6  # shown first, then again after each of the 5 options
    assert not (tmp_path / "Kurucz_dense").exists()


def test_menu_9_opens_and_returns_without_torch(monkeypatch, capsys, tmp_path):
    from sed_tools import cli

    _without(monkeypatch, *INSTALLS["none"])
    _menu_ml_flows_in(cli, tmp_path, monkeypatch)
    answer(monkeypatch, "9", "0", "q")
    assert cli._run_menu() == 0  # final test, check 16: as in 0.4.1
    assert capsys.readouterr().err == ""


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
