import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_cli_config_subcommand_show(capsys):
    from unittest.mock import patch
    with patch("sys.argv", ["sed-tools", "config"]):
        from sed_tools.cli import main
        try:
            main()
        except SystemExit:
            pass
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
