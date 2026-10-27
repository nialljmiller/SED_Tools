"""Tests for the grids-area cleanup: every handler that now raises or warns.

Covers config, grid_densifier, grid_coverage, spectra_cleaner and
combine_stellar_atm. No test touches the network or the user's real
~/.sed_tools directory.
"""

import pytest

import sed_tools.config as config


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------

def test_load_toml_missing_file_is_empty_unless_strict(tmp_path):
    missing = tmp_path / "absent.toml"
    assert config.load_toml(missing) == {}
    with pytest.raises(FileNotFoundError):
        config.load_toml(missing, strict=True)


def test_load_toml_malformed_file_raises_naming_it(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('data_dir = "unterminated\n')
    for strict in (False, True):
        with pytest.raises(ValueError, match="bad.toml"):
            config.load_toml(bad, strict=strict)


def test_malformed_user_config_stops_data_dir_lookup(tmp_path, monkeypatch):
    # Used to fall back silently to the default data directory.
    bad = tmp_path / "config.toml"
    bad.write_text("data_dir = \n")
    monkeypatch.setattr(config, "_CONFIG_FILE", bad)
    monkeypatch.delenv("SED_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="config.toml"):
        config.get_data_dir()


def test_move_data_removes_drained_dirs_and_keeps_conflicts(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "filters" / "A").mkdir(parents=True)
    (old / "filters" / "A" / "same.dat").write_text("x")
    (old / "stellar_models" / "B").mkdir(parents=True)
    (old / "stellar_models" / "B" / "clash.txt").write_text("old")
    (new / "filters" / "A").mkdir(parents=True)
    (new / "filters" / "A" / "same.dat").write_text("x")
    (new / "stellar_models" / "B").mkdir(parents=True)
    (new / "stellar_models" / "B" / "clash.txt").write_text("new")

    conflicts = config._move_data(old, new)

    assert not (old / "filters").exists()
    assert (old / "stellar_models" / "B" / "clash.txt").read_text() == "old"
    assert (new / "stellar_models" / "B" / "clash.txt").read_text() == "new"
    assert conflicts == [(str(old / "stellar_models" / "B" / "clash.txt"),
                          str(new / "stellar_models" / "B" / "clash.txt"))]
