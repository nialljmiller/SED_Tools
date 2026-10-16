"""Tests for the torch-free surface of ``sed_tools.ml``.

``ml.py`` itself has no top-level torch import at all — it lazily
imports ``SEDCompleter``/``AutoTrainer`` inside each function. This
file covers the pieces that never need a trained network:
``list_models``, ``model_info``, and the untrained-state behaviour of
the ``Completer`` convenience class. ``train_completer``,
``load_completer``, ``complete_sed``, and ``auto_train_generator`` all
require torch (transitively, via ``SEDCompleter``/``ml_optimized``) and
are out of scope here.
"""

import json

import pytest

from sed_tools.ml import Completer, list_models, model_info


def test_module_imports_without_torch():
    import sed_tools.ml  # noqa: F401


# ---------------------------------------------------------------------
# list_models — delegates to SEDCompleter.list_models
# ---------------------------------------------------------------------

def test_list_models_empty_when_directory_missing(tmp_path):
    assert list_models(str(tmp_path / "nope")) == []


def test_list_models_reads_config_fields(tmp_path):
    model_dir = tmp_path / "m1"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"version": "2.0.0", "framework": "pytorch", "architecture": {"input_dim": 10}})
    )
    models = list_models(str(tmp_path))
    assert len(models) == 1
    assert models[0]["name"] == "m1"
    assert models[0]["framework"] == "pytorch"


# ---------------------------------------------------------------------
# model_info — direct config.json read
# ---------------------------------------------------------------------

def test_model_info_returns_full_config(tmp_path):
    model_dir = tmp_path / "m1"
    model_dir.mkdir()
    config = {"version": "2.0.0", "framework": "pytorch", "architecture": {"input_dim": 10}}
    (model_dir / "config.json").write_text(json.dumps(config))

    assert model_info(str(model_dir)) == config


def test_model_info_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model config not found"):
        model_info(str(tmp_path / "nonexistent"))


# ---------------------------------------------------------------------
# Completer — untrained-state construction and properties
# ---------------------------------------------------------------------

def test_completer_default_construction_is_untrained():
    c = Completer()
    assert c.is_trained is False
    assert c.model_path is None
    assert c.config == {}
    assert c.wavelength_grid is None
