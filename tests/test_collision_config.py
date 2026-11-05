"""``sed_tools.collision_config``: fuzzy matching of on-collision filter specs, ``CollisionConfig``
validation, ``load_config``'s root-directory level and the default config file.
"""

import pytest

from sed_tools.collision_config import (
    CollisionConfig,
    copy_global_config_to_model,
    load_config,
    match_filter_value,
    write_default_config,
)


# ---------------------------------------------------------------------
# match_filter_value
# ---------------------------------------------------------------------

def test_match_filter_value_min_max_numeric():
    values = [0.4, 0.0, -0.2]
    assert match_filter_value("min", values, "alpha") == (-0.2, None)
    assert match_filter_value("max", values, "alpha") == (0.4, None)


def test_match_filter_value_min_max_string_fallback():
    values = ["h-rich", "he-rich"]
    matched, err = match_filter_value("min", values, "composition")
    assert err is None
    assert matched in values  # min() by str ordering; just confirm a valid pick


def test_match_filter_value_string_exact_match_case_insensitive():
    assert match_filter_value("H-RICH", ["h-rich", "he-rich"], "composition") == ("h-rich", None)


def test_match_filter_value_no_match_reports_available():
    matched, err = match_filter_value("bogus", ["h-rich", "he-rich"], "composition")
    assert matched is None
    assert "h-rich" in err and "he-rich" in err


def test_match_filter_value_empty_available_list():
    matched, err = match_filter_value("x", [], "alpha")
    assert matched is None
    assert "No values found" in err


# ---------------------------------------------------------------------
# CollisionConfig
# ---------------------------------------------------------------------

def test_collision_config_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid strategy"):
        CollisionConfig(strategy="bogus")


def test_resolve_filter_values_reports_unmatched_key():
    cfg = CollisionConfig(strategy="filter", filter_specs={"bogus_key": "0.4"})
    resolved, errors = cfg.resolve_filter_values({"alpha": [0.0, 0.4]})
    assert resolved == {}
    assert len(errors) == 1
    assert "bogus_key" in errors[0]


# ---------------------------------------------------------------------
# load_config precedence
# ---------------------------------------------------------------------

def test_load_config_root_dir_overrides_builtin(tmp_path):
    root = tmp_path / "root"
    model = tmp_path / "model"
    root.mkdir()
    model.mkdir()
    (root / "sed_tools.defaults").write_text('[on_collision]\nstrategy = "mean"\n')

    cfg = load_config(model_dir=model, root_dir=root)
    assert cfg.strategy == "mean"
    assert cfg.source == str(root / "sed_tools.defaults")


def test_write_default_config_then_copy_to_model(tmp_path):
    root = tmp_path / "root"
    model = tmp_path / "model"
    root.mkdir()
    model.mkdir()

    path = write_default_config(root)
    assert path.exists()
    assert 'strategy = "split"' in path.read_text()

    copy_global_config_to_model(root, model)
    assert (model / "sed_tools.defaults").exists()
    assert (model / "sed_tools.defaults").read_text() == path.read_text()


def test_write_default_config_does_not_overwrite_by_default(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    path = write_default_config(root)
    path.write_text("# manually edited\n")

    write_default_config(root, overwrite=False)
    assert path.read_text() == "# manually edited\n"
