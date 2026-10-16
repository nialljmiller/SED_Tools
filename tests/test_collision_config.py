"""Unit tests for ``sed_tools.collision_config``.

Covers the fuzzy key/value matching used to resolve on-collision filter
specs, ``CollisionConfig`` validation, the four-level config precedence
chain in ``load_config``, and ``discover_extra_axes`` (which determines
what counts as a "collision" in the first place).
"""

import pytest

from sed_tools.collision_config import (
    CollisionConfig,
    _find_column,
    _keys_match,
    _normalise_key,
    copy_global_config_to_model,
    discover_extra_axes,
    load_config,
    match_filter_value,
    write_default_config,
)


# ---------------------------------------------------------------------
# Fuzzy key normalisation
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "key,expected",
    [
        ("[alpha/Fe]", "alphafe"),
        ("f_sed", "fsed"),
        ("Alpha", "alpha"),
        ("v_turb", "vturb"),
    ],
)
def test_normalise_key(key, expected):
    assert _normalise_key(key) == expected


def test_keys_match_across_formatting_variants():
    assert _keys_match("Alpha", "alpha") is True
    assert _keys_match("[Alpha/Fe]", "alpha_fe") is True
    assert _keys_match("alpha", "yhe") is False


def test_find_column_matches_by_normalised_form():
    assert _find_column("Alpha", ["yhe", "ALPHA", "fsed"]) == "ALPHA"


def test_find_column_returns_none_when_absent():
    assert _find_column("nope", ["yhe", "alpha"]) is None


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


def test_match_filter_value_exact_numeric_match():
    assert match_filter_value("0.4", [0.4, 0.0, -0.2], "alpha") == (0.4, None)


def test_match_filter_value_no_exact_numeric_reports_nearest():
    matched, err = match_filter_value("0.41", [0.4, 0.0, -0.2], "alpha")
    assert matched is None
    assert "No exact match" in err
    assert "0.4" in err


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


@pytest.mark.parametrize("strategy", ["split", "all-warn", "all", "mean", "filter"])
def test_collision_config_accepts_all_valid_strategies(strategy):
    cfg = CollisionConfig(strategy=strategy)
    assert cfg.strategy == strategy


def test_resolve_filter_values_matches_fuzzy_column():
    cfg = CollisionConfig(strategy="filter", filter_specs={"Alpha": "0.4"})
    resolved, errors = cfg.resolve_filter_values({"alpha": [0.0, 0.4], "yhe": [0.1, 0.2]})
    assert resolved == {"alpha": 0.4}
    assert errors == []


def test_resolve_filter_values_reports_unmatched_key():
    cfg = CollisionConfig(strategy="filter", filter_specs={"bogus_key": "0.4"})
    resolved, errors = cfg.resolve_filter_values({"alpha": [0.0, 0.4]})
    assert resolved == {}
    assert len(errors) == 1
    assert "bogus_key" in errors[0]


# ---------------------------------------------------------------------
# load_config precedence
# ---------------------------------------------------------------------

def test_load_config_defaults_to_builtin_when_nothing_else_present(tmp_path):
    root = tmp_path / "root"
    model = tmp_path / "model"
    root.mkdir()
    model.mkdir()

    cfg = load_config(model_dir=model, root_dir=root)
    assert cfg.strategy == "all-warn"
    assert cfg.source == "packaged sed_tools.defaults"


def test_load_config_root_dir_overrides_builtin(tmp_path):
    root = tmp_path / "root"
    model = tmp_path / "model"
    root.mkdir()
    model.mkdir()
    (root / "sed_tools.defaults").write_text('[on_collision]\nstrategy = "mean"\n')

    cfg = load_config(model_dir=model, root_dir=root)
    assert cfg.strategy == "mean"
    assert cfg.source == str(root / "sed_tools.defaults")


def test_load_config_model_dir_overrides_root_dir(tmp_path):
    root = tmp_path / "root"
    model = tmp_path / "model"
    root.mkdir()
    model.mkdir()
    (root / "sed_tools.defaults").write_text('[on_collision]\nstrategy = "mean"\n')
    (model / "mesa_config.toml").write_text(
        '[on_collision]\nstrategy = "filter"\n[on_collision.filter]\nalpha = "0.4"\n'
    )

    cfg = load_config(model_dir=model, root_dir=root)
    assert cfg.strategy == "filter"
    assert cfg.filter_specs == {"alpha": "0.4"}
    assert cfg.source == str(model / "mesa_config.toml")


def test_load_config_override_dict_has_highest_priority(tmp_path):
    root = tmp_path / "root"
    model = tmp_path / "model"
    root.mkdir()
    model.mkdir()
    (root / "sed_tools.defaults").write_text('[on_collision]\nstrategy = "mean"\n')
    (model / "mesa_config.toml").write_text('[on_collision]\nstrategy = "filter"\n')

    cfg = load_config(model_dir=model, root_dir=root, override_dict={"on_collision": {"strategy": "split"}})
    assert cfg.strategy == "split"
    assert cfg.source == "Python API dict"


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


def test_copy_global_config_is_noop_when_source_missing(tmp_path):
    root = tmp_path / "root"
    model = tmp_path / "model"
    root.mkdir()
    model.mkdir()
    copy_global_config_to_model(root, model)  # no sed_tools.defaults in root
    assert not (model / "sed_tools.defaults").exists()


# ---------------------------------------------------------------------
# discover_extra_axes
# ---------------------------------------------------------------------

def test_discover_extra_axes_excludes_mesa_and_bookkeeping_columns():
    cols = [
        "file_name", "teff", "T_eff", "logg", "[Fe/H]", "z", "source",
        "units_standardized", "wavelength_unit", "vturb",
        "alpha", "f_sed", "yhe",
    ]
    assert discover_extra_axes(cols) == ["alpha", "f_sed", "yhe"]


def test_discover_extra_axes_empty_when_no_extra_columns():
    cols = ["file_name", "teff", "logg", "metallicity", "source", "units_standardized"]
    assert discover_extra_axes(cols) == []


def test_discover_extra_axes_preserves_original_column_names():
    # Original casing/text must be preserved even though matching is fuzzy.
    cols = ["file_name", "Teff", "logg", "metallicity", "[Alpha/Fe]"]
    assert discover_extra_axes(cols) == ["[Alpha/Fe]"]
