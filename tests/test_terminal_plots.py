from pathlib import Path

import numpy as np


def _write_spectrum(path: Path, teff=5000, logg=4.0, metallicity=0.0):
    path.write_text(
        f"# Teff = {teff}\n# logg = {logg}\n# metallicity = {metallicity}\n"
        "1000 1.0\n2000 2.0\n3000 1.5\n",
        encoding="utf-8",
    )


def test_terminal_modes():
    from sed_tools.terminal_plots import terminal_color_enabled, terminal_plots_enabled

    assert terminal_plots_enabled("never") is False
    assert terminal_plots_enabled("always") is True
    assert terminal_color_enabled("never") is False
    assert terminal_color_enabled("always") is True


def test_scatter2d_content():
    from sed_tools.terminal_plots import scatter2d

    output = scatter2d(
        [1, 2], [3, 4], title="Coverage", xlabel="temperature",
        ylabel="gravity", point_labels={0: "A"}, color="never",
    )
    assert all(text in output for text in ("Coverage", "temperature", "gravity", "A"))


def test_lineplot_content():
    from sed_tools.terminal_plots import lineplot

    output = lineplot(
        [{"x": [1, 2, 3], "y": [0, 1, 0], "label": "sample A", "char": "A"}],
        title="SED plot", xlabel="wave", ylabel="flux", color="never",
    )
    assert "SED plot" in output
    assert "sample A" in output


def test_read_grid_nodes_fallback_has_path(tmp_path):
    from sed_tools.grid_coverage import read_grid_nodes

    spectrum = tmp_path / "model.txt"
    _write_spectrum(spectrum)
    nodes = read_grid_nodes(tmp_path)
    assert nodes.loc[0, "filename"] == "model.txt"
    assert nodes.loc[0, "path"] == str(spectrum.resolve())


def test_read_grid_nodes_lookup_has_resolved_path(tmp_path):
    from sed_tools.grid_coverage import read_grid_nodes

    spectrum = tmp_path / "model.txt"
    _write_spectrum(spectrum)
    (tmp_path / "lookup_table.csv").write_text(
        "teff,logg,metallicity,filename\n5000,4.0,0.0,model.txt\n",
        encoding="utf-8",
    )
    nodes = read_grid_nodes(tmp_path)
    assert nodes.loc[0, "path"] == str(spectrum.resolve())


def test_grid_coverage_terminal_config_modes(tmp_path, monkeypatch, capsys):
    import sed_tools.config as config
    from sed_tools.grid_coverage import grid_coverage

    _write_spectrum(tmp_path / "cool.txt", teff=4000, logg=3.0)
    _write_spectrum(tmp_path / "hot.txt", teff=8000, logg=5.0)
    monkeypatch.setattr(config, "get_ui_setting", lambda name: "never")
    summary = grid_coverage(tmp_path, plot=False)
    output = capsys.readouterr().out
    assert summary["n_spectra"] == 2
    assert f"{tmp_path.name}: parameter coverage" not in output

    monkeypatch.setattr(config, "get_ui_setting", lambda name: "always" if name == "terminal_plots" else "never")
    grid_coverage(tmp_path, plot=False)
    output = capsys.readouterr().out
    assert f"{tmp_path.name}: parameter coverage" in output
    assert "Representative SEDs (individually normalized)" in output
    assert "SED A:" in output
    assert "SED D:" in output


def test_plot_rows_places_two_plots_side_by_side():
    from sed_tools.grid_coverage import _plot_rows

    output = _plot_rows(["first\na", "second\nb", "third\nc"])
    lines = output.splitlines()
    assert "first" in lines[0] and "second" in lines[0]
    assert "third" in output


def test_plot_rows_ignores_ansi_sequences_when_aligning():
    from sed_tools.grid_coverage import _ANSI_ESCAPE, _plot_rows

    output = _plot_rows(["plain\n\x1b[31mA\x1b[0m", "next\nB"])
    lines = [_ANSI_ESCAPE.sub("", line) for line in output.splitlines()]
    assert lines[1].index("B") == lines[0].index("next")


def test_set_data_dir_preserves_nested_ui(tmp_path, monkeypatch):
    import sed_tools.config as config

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[ui]\nterminal_plots = "always"\nterminal_color = "never"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "_CONFIG_FILE", config_file)
    config.set_data_dir(str(tmp_path / "data"))
    loaded = config.load_user_config()
    assert loaded["ui"] == {"terminal_plots": "always", "terminal_color": "never"}
    assert loaded["data_dir"] == str(tmp_path / "data")
