import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_get_data_dir_returns_path():
    from sed_tools.config import get_data_dir
    d = get_data_dir()
    assert isinstance(d, Path)


def test_default_data_dir_not_in_site_packages():
    from sed_tools.config import get_data_dir
    d = get_data_dir()
    assert "site-packages" not in str(d)


def test_env_var_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SED_DATA_DIR", str(tmp_path))
    # Re-import to pick up env
    from importlib import reload
    import sed_tools.config as cfg
    reload(cfg)
    assert cfg.get_data_dir() == tmp_path
    monkeypatch.delenv("SED_DATA_DIR")
    reload(cfg)


def test_set_data_dir_writes_config(tmp_path, monkeypatch):
    import sed_tools.config as cfg
    monkeypatch.setattr(cfg, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "_CONFIG_FILE", tmp_path / "config.toml")
    cfg.set_data_dir(str(tmp_path / "mydata"))
    assert (tmp_path / "config.toml").exists()
    content = (tmp_path / "config.toml").read_text()
    assert "mydata" in content


def test_data_dir_default_is_home_relative():
    from sed_tools.config import _DEFAULT_DATA_DIR
    home = Path("~").expanduser()
    assert str(_DEFAULT_DATA_DIR).startswith(str(home))


def test_models_data_dir_not_in_site_packages():
    from sed_tools.models import DATA_DIR_DEFAULT
    assert "site-packages" not in str(DATA_DIR_DEFAULT)


def test_stellar_dir_under_data_dir():
    from sed_tools.models import DATA_DIR_DEFAULT, STELLAR_DIR_DEFAULT
    assert str(STELLAR_DIR_DEFAULT).startswith(str(DATA_DIR_DEFAULT))


def test_filter_dir_under_data_dir():
    from sed_tools.models import DATA_DIR_DEFAULT, FILTER_DIR_DEFAULT
    assert str(FILTER_DIR_DEFAULT).startswith(str(DATA_DIR_DEFAULT))


def test_collision_config_uses_shared_loader_and_merges_layers(tmp_path):
    from sed_tools.collision_config import load_config

    (tmp_path / "sed_tools.defaults").write_text(
        '[on_collision]\nstrategy = "mean"\n[ui]\nterminal_color = "never"\n'
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "mesa_config.toml").write_text(
        '[on_collision]\nstrategy = "filter"\n'
        '[on_collision.filter]\nalpha = "min"\n'
    )

    resolved = load_config(model_dir=model, root_dir=tmp_path)
    assert resolved.strategy == "filter"
    assert resolved.filter_specs == {"alpha": "min"}


def test_collision_config_api_override_is_partial(tmp_path):
    from sed_tools.collision_config import load_config

    (tmp_path / "sed_tools.defaults").write_text(
        '[on_collision]\nstrategy = "filter"\n'
        '[on_collision.filter]\nalpha = "min"\nf_sed = "2.0"\n'
    )
    resolved = load_config(
        root_dir=tmp_path,
        override_dict={"on_collision": {"filter": {"alpha": "max"}}},
    )
    assert resolved.strategy == "filter"
    assert resolved.filter_specs == {"alpha": "max", "f_sed": "2.0"}
