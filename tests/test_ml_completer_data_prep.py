"""SEDCompleter.list_models without torch: directories without a config, and generators, are
left out.
"""

import json


from sed_tools.ml_sed_completer import SEDCompleter


# ---------------------------------------------------------------------
# _prepare_training_data — pure numpy sampling/masking, no torch
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# list_models — pure filesystem/JSON scan
# ---------------------------------------------------------------------

def test_list_models_skips_dirs_without_config(tmp_path):
    (tmp_path / "not_a_model").mkdir()
    assert SEDCompleter.list_models(str(tmp_path)) == []


def test_list_models_leaves_out_generators(tmp_path):
    # 0.5.0: generators were listed too, and choosing one raised KeyError.
    # Completer configs have no model_type; SEDGenerator writes 'sed_generator'.
    for name, config in [("generator", {"model_type": "sed_generator"}), ("completer", {})]:
        (tmp_path / name).mkdir()
        (tmp_path / name / SEDCompleter.CONFIG_FILE).write_text(json.dumps(config))
    assert [m["name"] for m in SEDCompleter.list_models(str(tmp_path))] == ["completer"]
