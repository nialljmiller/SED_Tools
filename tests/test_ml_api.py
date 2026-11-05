"""The torch-free part of ``sed_tools.ml``: ``model_info`` and an untrained ``Completer``.
"""

import json


from sed_tools.ml import Completer, model_info


# ---------------------------------------------------------------------
# model_info — direct config.json read
# ---------------------------------------------------------------------

def test_model_info_returns_full_config(tmp_path):
    model_dir = tmp_path / "m1"
    model_dir.mkdir()
    config = {"version": "2.0.0", "framework": "pytorch", "architecture": {"input_dim": 10}}
    (model_dir / "config.json").write_text(json.dumps(config))

    assert model_info(str(model_dir)) == config


# ---------------------------------------------------------------------
# Completer — untrained-state construction and properties
# ---------------------------------------------------------------------

def test_completer_default_construction_is_untrained():
    c = Completer()
    assert c.is_trained is False
    assert c.model_path is None
    assert c.config == {}
    assert c.wavelength_grid is None
