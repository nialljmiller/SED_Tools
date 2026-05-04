import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_cli_importable():
    from sed_tools import cli
    assert callable(cli.main)


def test_menu_valid_choices():
    from sed_tools.cli import menu
    import io
    from unittest.mock import patch

    for key, expected in [
        ("1", "spectra"), ("2", "filters"), ("3", "rebuild"),
        ("4", "combine"), ("5", "ml_completer"), ("6", "ml_generator"),
        ("7", "grid_densifier"), ("8", "mesa_prepare"),
        ("9", "config"), ("0", "quit"),
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
