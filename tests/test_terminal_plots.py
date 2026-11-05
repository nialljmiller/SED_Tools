from pathlib import Path

import numpy as np


def _write_spectrum(path: Path, teff=5000, logg=4.0, metallicity=0.0):
    path.write_text(
        f"# Teff = {teff}\n# logg = {logg}\n# metallicity = {metallicity}\n"
        "1000 1.0\n2000 2.0\n3000 1.5\n",
        encoding="utf-8",
    )


def test_read_grid_nodes_fallback_has_path(tmp_path):
    from sed_tools.grid_coverage import read_grid_nodes

    spectrum = tmp_path / "model.txt"
    _write_spectrum(spectrum)
    nodes = read_grid_nodes(tmp_path)
    assert nodes.loc[0, "filename"] == "model.txt"
    assert nodes.loc[0, "path"] == str(spectrum.resolve())


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
