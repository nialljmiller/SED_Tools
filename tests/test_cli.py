import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_cli_importable():
    from sed_tools import cli
    assert callable(cli.main)


@pytest.mark.parametrize("command,option", [("grids", "--grids"), ("spectra", "--models"),
                                           ("grids", "--models"), ("spectra", "--grids")])
def test_spectral_grid_command_aliases_share_one_workflow(monkeypatch, command, option):
    from sed_tools import cli

    calls = []
    monkeypatch.setattr(cli, "run_spectra_flow", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["sed-tools", command, option, "Kurucz2003all"])
    cli.main()
    assert len(calls) == 1
    assert calls[0]["models"] == ["Kurucz2003all"]


def test_cli_uses_shared_prompt_choice():
    from sed_tools import cli, ui_utils

    assert cli.prompt_choice is ui_utils.prompt_choice


def test_top_level_workflows_are_cli_aliases():
    import sed_tools
    from sed_tools import cli

    assert sed_tools.run_rebuild_flow is cli.run_rebuild_flow
    assert sed_tools.run_spectra_flow is cli.run_spectra_flow
    assert sed_tools.run_combine_flow is cli.run_combine_flow
    assert sed_tools.menu is cli.menu


def test_menu_valid_choices():
    from sed_tools.cli import menu
    import io
    from unittest.mock import patch

    for key, expected in [
        ("1", "filters"), ("2", "filters_combine"), ("3", "filter_info"),
        ("4", "filter_compose"), ("5", "grids"), ("6", "rebuild"),
        ("7", "combine"), ("8", "ml_completer"), ("9", "ml_generator"),
        ("10", "grid_densifier"), ("11", "coverage"), ("12", "import"),
        ("13", "mesa_prepare"), ("14", "config"), ("0", "quit"),
    ]:
        with patch("builtins.input", return_value=key):
            assert menu() == expected


def test_menu_invalid_returns_empty():
    from sed_tools.cli import menu
    from unittest.mock import patch
    with patch("builtins.input", return_value="99"):
        assert menu() == ""


def test_cli_config_subcommand_show(capsys):
    import sys
    from unittest.mock import patch
    with patch("sys.argv", ["sed-tools", "config"]):
        from sed_tools.cli import main
        try:
            main()
        except SystemExit:
            pass
    captured = capsys.readouterr()
    assert "Data directory" in captured.out


def test_run_config_flow_no_change(capsys):
    from sed_tools.cli import run_config_flow
    from unittest.mock import patch
    with patch("builtins.input", return_value=""):
        run_config_flow()
    captured = capsys.readouterr()
    assert "Data directory" in captured.out


def test_run_filter_combine_flow_wizard(tmp_path, capsys):
    from sed_tools.cli import run_filter_combine_flow
    from unittest.mock import patch

    first = tmp_path / "GAIA" / "GAIA"
    second = tmp_path / "2MASS" / "2MASS"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "G.dat").write_text("Wavelength,Transmission\n1,1\n")
    (second / "J.dat").write_text("Wavelength,Transmission\n2,1\n")

    answers = iter([
        "",       # use default base dir passed to flow
        "1,2",    # select both local filter sets
        "Gaia2MASS",
        "",       # default facility Combined
        "",       # default instrument/output name
        "",       # default conflict mode rename
        "",       # proceed
    ])

    with patch("builtins.input", side_effect=lambda _prompt="": next(answers)):
        run_filter_combine_flow(base_dir=str(tmp_path))

    out = tmp_path / "Combined" / "Gaia2MASS"
    assert sorted(p.name for p in out.glob("*.dat")) == ["G.dat", "J.dat"]
    assert (out / "Gaia2MASS").read_text().splitlines() == ["G.dat", "J.dat"]
    assert "Combined 2 filters" in capsys.readouterr().out
