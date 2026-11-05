import sys
from pathlib import Path

import pytest

import sed_tools.config as config

from helpers_cli import answer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_env_var_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SED_DATA_DIR", str(tmp_path))
    # Re-import to pick up env
    from importlib import reload
    import sed_tools.config as cfg
    reload(cfg)
    assert cfg.get_data_dir() == tmp_path
    monkeypatch.delenv("SED_DATA_DIR")
    reload(cfg)


def test_data_dir_default_is_home_relative():
    from sed_tools.config import _DEFAULT_DATA_DIR
    home = Path("~").expanduser()
    assert str(_DEFAULT_DATA_DIR).startswith(str(home))


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


# ----------------------------------------------------------------------
# Config (menu 14): q, b or Enter keep the data directory
# ----------------------------------------------------------------------
@pytest.mark.parametrize("reply", ["", "q", "b"])
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
