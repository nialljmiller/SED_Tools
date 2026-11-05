"""The ML completer and generator: what training and inference compute, their workflows,
extend_catalog, list_models, train_model and load_model, and their failures.

The characterisation tests pin what training and inference compute: tiny
seeded inputs, run on the CPU, with the results stored here as literals.
They were recorded before the ml modules were split up, so a refactor that
changes the numerics, the seeds or the order of random draws fails here.
``np.random.random()`` and ``torch.rand`` drawn after training catch a change
in how many random numbers training consumed. Tolerances allow for float32 and
BLAS differences between machines; a change in the random draws moves these
values by far more.

Tests that need torch skip without it; the rest are torch-free, like the
other ``test_ml_*`` files.
"""

import json

import numpy as np
import pytest

from sed_tools import ml, ml_sed_completer as mc, ml_sed_generator as mg

from helpers_ml import _fixed_completer, _fixed_generator
from helpers_ml import cpu_torch, tiny_library  # noqa: F401 (fixtures)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
def _state_sums(model):
    values = model.state_dict().values()
    return (float(sum(v.double().sum() for v in values)),
            float(sum(v.double().abs().sum() for v in values)))


def _feed_input(monkeypatch, answers):
    """Answer ``input()`` prompts in order, echoing them like a terminal."""
    feed = iter(answers)

    def fake_input(prompt=""):
        answer = next(feed)
        print(prompt + answer)
        return answer

    monkeypatch.setattr("builtins.input", fake_input)


# Grid order sampled by np.random.seed(3) from the 12-node tiny library.
_SAMPLED_PARAMS = [
    [5000.0, 4.0, 0.0], [5000.0, 4.0, -0.5], [4000.0, 4.0, 0.0], [4000.0, 4.5, -0.5],
    [6000.0, 4.5, 0.0], [5000.0, 4.5, -0.5], [5000.0, 4.5, 0.0], [4000.0, 4.0, -0.5],
    [4000.0, 4.5, 0.0], [6000.0, 4.0, 0.0], [6000.0, 4.0, -0.5], [6000.0, 4.5, -0.5],
]


# ---------------------------------------------------------------------
# Characterisation: training-data preparation (torch-free)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Characterisation: training (torch, CPU)
# ---------------------------------------------------------------------
def test_generator_train_values(cpu_torch, tiny_library, tmp_path):
    torch = cpu_torch
    generator = mg.SEDGenerator()
    history = generator.train(str(tiny_library), str(tmp_path / "model"), epochs=3,
                              batch_size=3, max_samples=12, hidden_layers=[8])

    assert history["train_loss"] == pytest.approx(
        [1.2670124769210815, 1.2379244565963745, 1.1913456122080486], rel=1e-3)
    assert history["val_loss"] == pytest.approx(
        [1.0461859703063965, 1.0439506769180298, 1.041712760925293], rel=1e-3)
    assert generator.scaler_params == pytest.approx({
        "teff_mean": 5000.0, "teff_std": 816.49658203125, "teff_min": 4000.0, "teff_max": 6000.0,
        "logg_mean": 4.25, "logg_std": 0.25, "logg_min": 4.0, "logg_max": 4.5,
        "meta_mean": -0.25, "meta_std": 0.25, "meta_min": -0.5, "meta_max": 0.0,
        "flux_log_mean": -10.267520904541016, "flux_log_std": 1.031362771987915,
    }, rel=1e-6)
    assert _state_sums(generator.model) == pytest.approx((26.80382775522594, 1610.2340406870499), rel=1e-3)
    assert np.random.random() == 0.3834415188257777
    assert torch.rand(2).tolist() == pytest.approx([0.3398873209953308, 0.19384098052978516], abs=0)

    config = json.loads((tmp_path / "model" / "config.json").read_text())
    assert sorted(config) == ["architecture", "framework", "model_type", "parameter_ranges",
                              "scaler_params", "training_params", "version", "wavelength_grid"]
    assert config["architecture"] == {"output_dim": 1000, "hidden_layers": [8], "dropout": 0.2}
    assert config["parameter_ranges"] == {"teff": [4000.0, 6000.0], "logg": [4.0, 4.5],
                                          "meta": [-0.5, 0.0]}
    assert (config["model_type"], config["version"]) == ("sed_generator", "1.0.0")
    assert len(config["training_params"]) == 9


# ---------------------------------------------------------------------
# Characterisation: inference on fixed (seeded, untrained) networks
# ---------------------------------------------------------------------
def test_completer_complete_sed_values(cpu_torch):
    completer = _fixed_completer(cpu_torch)
    wl = np.logspace(np.log10(3000.0), np.log10(20000.0), 300)
    flux = mc.planck_function(wl, 5800.0)

    wl_out, flux_out = completer.complete_sed(wl, flux, teff=5800.0, logg=4.4, meta=0.1,
                                              extension_range=(500.0, 60000.0))

    # 0.5.0: the input's own samples are kept verbatim (they used to be resampled
    # onto the grid); only points of the 1,000-point grid beyond them are added.
    grid = np.logspace(np.log10(500.0), np.log10(60000.0), 1000)
    below, above = grid < wl[0], grid > wl[-1]
    np.testing.assert_array_equal(wl_out, np.concatenate([grid[below], wl, grid[above]]))
    np.testing.assert_array_equal(flux_out[below.sum():below.sum() + len(wl)], flux)
    # The added points keep the values they had before 0.5.0.
    on_grid = np.full(grid.size, np.nan)
    on_grid[below] = flux_out[:below.sum()]
    on_grid[above] = flux_out[below.sum() + len(wl):]
    assert on_grid[[0, 150, 300, 900, 999]] == pytest.approx([
        410416.9146332419, 1687776.9083537995, 3334628.567953557, 2839454.620631518, 2065668.2893293651,
    ], rel=1e-5)


def test_generate_check_bounds_prints_each_out_of_range_parameter(cpu_torch, capsys):
    generator = _fixed_generator(cpu_torch)

    generator.generate(9000.0, 5.5, 0.6)

    out = capsys.readouterr().out
    assert "Warning: Teff=9000 outside range [4000, 7000]" in out
    assert "Warning: logg=5.50 outside range [3.50, 5.00]" in out
    assert "Warning: [M/H]=0.60 outside range [-1.00, 0.50]" in out


# ---------------------------------------------------------------------
# Characterisation: Completer.extend_catalog with a stub network
# ---------------------------------------------------------------------
class _StubCompleter:
    """Stands in for SEDCompleter: a deterministic complete_sed, no plots."""

    def __init__(self):
        self.model = object()
        self.calls = []
        self.plotted = None

    def complete_sed(self, wavelength, flux, teff, logg, meta, extension_range):
        self.calls.append((teff, logg, meta, tuple(extension_range)))
        grid = np.logspace(np.log10(extension_range[0]), np.log10(extension_range[1]), 20)
        return grid, np.interp(grid, wavelength, flux) * (1.0 + teff / 1e4)

    def plot_inference_examples(self, examples, output_dir):
        self.plotted = [example[0] for example in examples]


def _write_spectrum(path, lo, hi, scale=1.0):
    wl = np.logspace(np.log10(lo), np.log10(hi), 50)
    np.savetxt(path, np.column_stack([wl, scale * 1e-12 * (wl / 5000.0) ** -2]))


@pytest.fixture
def catalog(tmp_path):
    """Three spectra; c.txt already covers 100-100000 A, so it is copied."""
    import pandas as pd

    cat = tmp_path / "catalog"
    cat.mkdir()
    _write_spectrum(cat / "a.txt", 3000, 20000)
    _write_spectrum(cat / "b.txt", 2000, 30000, scale=2.0)
    _write_spectrum(cat / "c.txt", 100, 100000, scale=3.0)
    pd.DataFrame({"file_name": ["a.txt", "b.txt", "c.txt"], "teff": [5000, 6000, 7000],
                  "logg": [4.0, 4.5, 3.5], "meta": [0.0, -0.5, 0.2], "extra": ["x", "y", "z"]}
                 ).to_csv(cat / "lookup_table.csv", index=False)
    return cat


def _stubbed_completer():
    completer = ml.Completer()
    completer._completer = _StubCompleter()
    return completer


def test_extend_catalog_requires_a_model():
    with pytest.raises(RuntimeError, match="No model loaded"):
        ml.Completer().extend_catalog("in", "out")


# ---------------------------------------------------------------------
# Characterisation: interactive workflows (training and inference stubbed)
# ---------------------------------------------------------------------
@pytest.fixture
def stellar_dir(tmp_path, cube_writer):
    """gridA and gridB have flux cubes; gridB's lookup has a '#' header."""
    base = tmp_path / "stellar"
    wl = np.array([1000.0, 2000.0])
    for name in ("gridA", "gridB"):
        cube_writer(base / name / "flux_cube.bin", [5000.0], [4.0], [0.0], wl, np.ones((1, 1, 1, 2)))
    (base / "gridA" / "lookup_table.csv").write_text("file_name,teff\nx.txt,5000\n")
    (base / "gridB" / "lookup_table.csv").write_text("#file_name,teff\nq.txt,5000\nr.txt,6000\n")
    (base / "notagrid").mkdir()
    return base


@pytest.mark.parametrize("answers, message", [
    # 0.5.0: a bad number is asked again (it used to end the menu with "Invalid parameters")
    (["1", "1", "abc", "", "", "", "n"], "Cancelled"),
    (["9"], "Invalid choice"),
])
def test_completer_workflow_rejects_bad_answers(stellar_dir, tmp_path, monkeypatch, capsys, answers, message):
    monkeypatch.setattr(mc.SEDCompleter, "train", lambda self, **kw: pytest.fail("trained"))
    _feed_input(monkeypatch, answers)

    mc.run_interactive_workflow(str(stellar_dir), str(tmp_path / "models"))

    assert capsys.readouterr().out.rstrip().endswith(message)


def _write_model_config(directory, config):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))


def test_completer_workflow_extends_the_chosen_grid(stellar_dir, tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    _write_model_config(models / "cm1", {"version": "2.0.0", "framework": "pytorch",
                                         "architecture": {"input_dim": 400, "hidden_layers": [8],
                                                          "output_dim": 1000}})

    def fake_load(self, path):
        self.model = object()
        self.wavelength_grid = np.logspace(2, 5, 1000)

    extend_calls = []

    def fake_extend(self, input_dir, output_dir, extension_range, verbose):
        extend_calls.append((input_dir, output_dir, extension_range, verbose))
        return {"extended": 2, "failed": 0, "skipped": 1, "total": 3, "output_dir": output_dir}

    monkeypatch.setattr(mc.SEDCompleter, "load", fake_load)
    monkeypatch.setattr(ml.Completer, "extend_catalog", fake_extend)
    # 0.5.0: Enter at the two range prompts keeps the model's own range
    _feed_input(monkeypatch, ["2", "1", "1", "", "", ""])

    mc.run_interactive_workflow(str(stellar_dir), str(models))

    out = capsys.readouterr().out
    assert extend_calls == [(str(stellar_dir / "gridA"), str(stellar_dir / "gridA_ml_extended"),
                             (100.0, 100000.0), True)]
    assert "      Architecture: 400 -> [8] -> 1000" in out
    assert "  2 SEDs extended successfully" in out


def test_completer_workflow_lists_models(tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    _write_model_config(models / "cm1", {"version": "2.0.0", "framework": "pytorch",
                                         "architecture": {"input_dim": 400, "hidden_layers": [8],
                                                          "output_dim": 1000}})
    _feed_input(monkeypatch, ["3"])

    mc.run_interactive_workflow(str(tmp_path / "stellar"), str(models))

    out = capsys.readouterr().out
    assert "\n  cm1\n" in out
    assert "    Framework: pytorch, Version: 2.0.0" in out
    assert "    Architecture: 400 -> [8] -> 1000" in out


def test_generator_workflow_trains_on_the_chosen_library(stellar_dir, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(mg.SEDGenerator, "train", lambda self, *a, **kw: calls.append((a, kw)) or {})
    _feed_input(monkeypatch, ["1", "2", "3", "5", "", ""])

    mg.run_interactive_workflow(str(stellar_dir), str(tmp_path / "models"))

    assert calls == [((str(stellar_dir / "gridB"), str(tmp_path / "models" / "sed_generator_gridB")),
                      {"epochs": 3, "batch_size": 5})]
    assert "TRAINING COMPLETE!" in capsys.readouterr().out


def test_generator_workflow_lists_generators_only(tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    _write_model_config(models / "gm1", {"model_type": "sed_generator",
                                         "parameter_ranges": {"teff": [4000, 6000], "logg": [4, 4.5],
                                                              "meta": [-0.5, 0]},
                                         "architecture": {"hidden_layers": [8], "output_dim": 1000}})
    _write_model_config(models / "cm1", {"framework": "pytorch"})
    _feed_input(monkeypatch, ["4"])

    mg.run_interactive_workflow(str(tmp_path / "stellar"), str(models))

    out = capsys.readouterr().out
    assert "\n  gm1\n" in out and "cm1" not in out
    assert "    Teff: 4000 - 6000 K" in out
    assert "    Architecture: 3 → [8] → 1000" in out


def test_generator_workflow_batch_writes_one_sed_per_grid_point(cpu_torch, stellar_dir, tmp_path, monkeypatch):
    generator = _fixed_generator(cpu_torch)
    generator.save(tmp_path / "models" / "gm1", hidden_layers=[16, 8], dropout=0.2)
    _feed_input(monkeypatch, ["3", "1", "5000,5500", "4.0", "0.0,-0.5", ""])

    mg.run_interactive_workflow(str(stellar_dir), str(tmp_path / "models"))

    batch = tmp_path / "models" / "gm1" / "SED" / "batch"
    texts = sorted(p.name for p in batch.glob("*.txt"))
    assert texts == ["sed_T5000_g4.00_m+0.00.txt", "sed_T5000_g4.00_m-0.50.txt",
                     "sed_T5500_g4.00_m+0.00.txt", "sed_T5500_g4.00_m-0.50.txt"]
    loaded = mg.SEDGenerator(tmp_path / "models" / "gm1")
    wl, flux = loaded.generate(5500.0, 4.0, -0.5, check_bounds=False)
    np.testing.assert_array_equal(np.loadtxt(batch / "sed_T5500_g4.00_m-0.50.txt"),
                                  np.loadtxt([f"{w:.6e} {f:.6e}" for w, f in zip(wl, flux)]))


# =====================================================================
# Behaviour changes: handlers that now raise or warn
# =====================================================================
# --- ml_sed_completer --------------------------------------------------

@pytest.mark.parametrize("cls", [mc.SEDCompleter, mg.SEDGenerator])
def test_list_models_warns_naming_the_unreadable_config(cls, tmp_path, caplog):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "config.json").write_text("{not valid json")

    assert cls.list_models(str(tmp_path)) == []
    assert f"Skipping {tmp_path / 'broken'}: could not read its config.json" in caplog.text


def test_completer_prepare_training_data_warns_how_many_spectra_it_dropped(tmp_path, cube_writer, caplog):
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
    flux = np.full((2, 1, 1, 200), 1e-10)
    flux[0, 0, 0, :150] = np.nan          # 4000 K: only 50 usable points
    cube_writer(tmp_path / "lib" / "flux_cube.bin", [4000.0, 5000.0], [4.0], [0.0], wl, flux)

    np.random.seed(0)
    _, _, params, _ = mc.SEDCompleter()._prepare_training_data(str(tmp_path / "lib"), 2)

    assert params[:, 0].tolist() == [5000.0]
    assert ("dropped 1 of 2 sampled spectra (1 with fewer than 100 positive, finite flux points)"
            in caplog.text)


def test_completer_train_raises_before_saving_when_every_batch_is_nan(cpu_torch, tiny_library, tmp_path, monkeypatch):
    monkeypatch.setattr(mc.SEDCompleterNetwork, "__call__", lambda self, x: self.model(x) * float("nan"))

    with pytest.raises(RuntimeError, match="produced no usable epoch"):
        mc.SEDCompleter().train(str(tiny_library), str(tmp_path / "model"), epochs=2,
                                batch_size=3, max_samples=12, hidden_layers=[8])
    assert not (tmp_path / "model").exists()


def test_completer_train_rejects_non_finite_grid_parameters(cpu_torch, tmp_path, cube_writer):
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
    cube_writer(tmp_path / "lib" / "flux_cube.bin", [4000.0, np.nan], [4.0], [0.0], wl,
                np.full((2, 1, 1, 200), 1e-10))

    with pytest.raises(ValueError, match="Non-finite Teff, logg or"):
        mc.SEDCompleter().train(str(tmp_path / "lib"), str(tmp_path / "model"), epochs=1,
                                batch_size=2, max_samples=2, hidden_layers=[8])


def test_train_model_forwards_keywords_and_rejects_unknown_ones(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mc.SEDCompleter, "train", lambda self, **kw: calls.append(kw))
    monkeypatch.setattr(ml, "Completer", lambda path: ("Completer", path))

    handle = mc.train_model("grid", str(tmp_path / "m"), epochs=2, verbose=False, dropout=0.1)

    assert handle == ("Completer", str(tmp_path / "m"))
    assert calls[0]["verbose"] is False and calls[0]["dropout"] == 0.1


def test_train_model_rejects_keywords_train_does_not_take(tmp_path):
    with pytest.raises(TypeError, match="val_split"):
        mc.train_model("grid", str(tmp_path / "m"), val_split=0.2)


def test_load_model_raises_the_load_error(tmp_path):
    with pytest.raises(ValueError, match="Model path must be a directory"):
        mc.load_model(str(tmp_path / "missing"))


def test_completer_workflow_warns_about_unreadable_lookup_tables(stellar_dir, tmp_path, monkeypatch, capsys, caplog):
    (stellar_dir / "gridA" / "lookup_table.csv").write_text("")
    (stellar_dir / "gridC").mkdir()
    (stellar_dir / "gridC" / "lookup_table.csv").write_text("")
    _write_model_config(tmp_path / "models" / "cm1", {"architecture": {}})
    monkeypatch.setattr(mc.SEDCompleter, "load", lambda self, path: setattr(self, "model", object()))

    _feed_input(monkeypatch, ["1", "x"])
    mc.run_interactive_workflow(str(stellar_dir), str(tmp_path / "models"))
    assert "   1) gridA                          (? SEDs)" in capsys.readouterr().out
    assert f"Could not count the SEDs in {stellar_dir / 'gridA' / 'lookup_table.csv'}" in caplog.text

    _feed_input(monkeypatch, ["2", "1", "x"])
    mc.run_interactive_workflow(str(stellar_dir), str(tmp_path / "models"))
    out = capsys.readouterr().out
    assert "gridC" not in out and "   1) gridB" in out
    assert "Not offering gridA: could not read" in caplog.text
    assert "Not offering gridC: could not read" in caplog.text


# --- ml_sed_generator --------------------------------------------------

def test_generator_prepare_training_data_warns_how_many_spectra_it_dropped(tmp_path, cube_writer, caplog):
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
    flux = np.full((2, 1, 1, 200), 1e-10)
    flux[0, 0, 0, 5] = np.nan
    cube_writer(tmp_path / "lib" / "flux_cube.bin", [4000.0, 5000.0], [4.0], [0.0], wl, flux)

    params, _, _ = mg.SEDGenerator()._prepare_training_data(str(tmp_path / "lib"), 2)

    assert params[:, 0].tolist() == [5000.0]
    assert "dropped 1 of 2 sampled spectra (1 with non-positive or non-finite flux)" in caplog.text


def test_generator_train_raises_before_saving_when_every_batch_is_nan(cpu_torch, tiny_library, tmp_path,
                                                                      monkeypatch, capsys):
    monkeypatch.setattr(mg.SEDGeneratorNetwork, "__call__", lambda self, x: self.model(x) * float("nan"))

    with pytest.raises(RuntimeError, match="produced no usable epoch"):
        mg.SEDGenerator().train(str(tiny_library), str(tmp_path / "model"), epochs=2,
                                batch_size=3, max_samples=12, hidden_layers=[8])
    out = capsys.readouterr().out
    assert "Warning: NaN loss at epoch 1, skipping batch" in out
    assert "Epoch 1: No valid batches, stopping" in out
    assert not (tmp_path / "model").exists()


def test_generator_train_rejects_a_grid_with_no_flux_above_the_floor(cpu_torch, tmp_path, cube_writer):
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
    cube_writer(tmp_path / "lib" / "flux_cube.bin", [4000.0, 5000.0], [4.0], [0.0], wl,
                np.full((2, 1, 1, 200), 1e-30))

    with pytest.raises(ValueError, match="No valid data points for normalization"):
        mg.SEDGenerator().train(str(tmp_path / "lib"), str(tmp_path / "model"), epochs=1,
                                batch_size=2, max_samples=2, hidden_layers=[8])
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize("number", ["0", "-1"])
def test_generator_workflow_rejects_zero_and_negative_menu_numbers(stellar_dir, tmp_path, monkeypatch, capsys, number):
    # These used to index from the end of the list and pick the last entry.
    _write_model_config(tmp_path / "models" / "gm1", {"model_type": "sed_generator"})
    _feed_input(monkeypatch, ["3", number])

    mg.run_interactive_workflow(str(stellar_dir), str(tmp_path / "models"))

    assert capsys.readouterr().out.rstrip().endswith("Invalid selection")


def test_generator_workflow_warns_about_an_unreadable_lookup_table(stellar_dir, tmp_path, monkeypatch, capsys, caplog):
    (stellar_dir / "gridA" / "lookup_table.csv").write_text("")
    _feed_input(monkeypatch, ["1", "x"])

    mg.run_interactive_workflow(str(stellar_dir), str(tmp_path / "models"))

    assert "   1) gridA                          (? SEDs)" in capsys.readouterr().out
    assert f"Could not count the SEDs in {stellar_dir / 'gridA' / 'lookup_table.csv'}" in caplog.text


# --- ml.Completer.extend_catalog ---------------------------------------

def test_extend_catalog_raises_without_a_lookup_table(tmp_path):
    # It used to extend every *.txt file with Teff=5000, logg=4.5, [M/H]=0.
    (tmp_path / "cat").mkdir()
    _write_spectrum(tmp_path / "cat" / "a.txt", 3000, 20000)

    with pytest.raises(FileNotFoundError, match="lookup_table.csv not found"):
        _stubbed_completer().extend_catalog(str(tmp_path / "cat"), str(tmp_path / "out"))


@pytest.mark.parametrize("missing", ["teff", "file_name"])
def test_extend_catalog_raises_when_a_column_is_missing(catalog, tmp_path, missing):
    # Missing parameters used to default to 5000 K / 4.5 / 0.0, and a missing
    # file column matched rows to the sorted *.txt files by position.
    import pandas as pd

    df = pd.read_csv(catalog / "lookup_table.csv").drop(columns=[missing])
    df.to_csv(catalog / "lookup_table.csv", index=False)
    completer = _stubbed_completer()

    with pytest.raises(ValueError, match=r"has no column for (file|teff|logg|meta) \(one of"):
        completer.extend_catalog(str(catalog), str(tmp_path / "out"))
    assert completer._completer.calls == []


def test_extend_catalog_reads_a_hash_prefixed_header(catalog, tmp_path):
    # sed_tools writes lookup tables with a '#file_name,...' header. It used
    # to be dropped (pd.read_csv(comment='#')) and the table refused; it now
    # gives the same result as a plain header (test_extend_catalog_values).
    text = (catalog / "lookup_table.csv").read_text()
    (catalog / "lookup_table.csv").write_text("#" + text)
    completer = _stubbed_completer()
    out_dir = tmp_path / "out"

    summary = completer.extend_catalog(str(catalog), str(out_dir), extension_range=(500.0, 60000.0))

    assert summary == {"extended": 2, "failed": 0, "skipped": 1, "total": 3, "output_dir": str(out_dir)}
    assert completer._completer.calls == [(5000.0, 4.0, 0.0, (500.0, 60000.0)),
                                          (6000.0, 4.5, -0.5, (500.0, 60000.0))]
    assert (out_dir / "lookup_table.csv").read_text().splitlines()[0] == (
        "file_name,teff,logg,meta,extra,extended_by_ml,wl_min,wl_max")


def test_extend_catalog_writes_the_good_spectra_then_lists_every_failure(catalog, tmp_path):
    import pandas as pd

    np.savetxt(catalog / "one_column.txt", np.arange(5.0))
    pd.DataFrame({
        "file_name": ["a.txt", "b.txt", "c.txt", "gone.txt", "one_column.txt", None, "a.txt", "a.txt", "a.txt"],
        "teff": [5000, 6000, 7000, 5000, 5000, 5000, "hot", -10, 5000],
        "logg": [4.0, 4.5, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 12.0],
        "meta": [0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }).to_csv(catalog / "lookup_table.csv", index=False)
    completer = _stubbed_completer()
    out_dir = tmp_path / "out"

    with pytest.raises(RuntimeError) as excinfo:
        completer.extend_catalog(str(catalog), str(out_dir), verbose=False)

    message = str(excinfo.value)
    assert message.startswith(f"6 of 9 spectra in {catalog} could not be extended "
                              f"(the other 3 were written to {out_dir}):")
    assert f"row 3 (gone.txt): gone.txt not found in {catalog}" in message
    assert "row 4 (one_column.txt): expected wavelength and flux columns, got an array of shape (5,)" in message
    assert "row 5 (nan): no file name" in message
    assert "row 6 (a.txt): Teff 'hot' is not a number" in message
    assert "row 7 (a.txt): Teff -10.0 is not positive" in message
    assert "row 8 (a.txt): logg 12.0 is above 10" in message
    assert sorted(p.name for p in out_dir.iterdir()) == ["a.txt", "b.txt", "c.txt", "lookup_table.csv"]
    assert len(completer._completer.calls) == 2


@pytest.mark.parametrize("error, message", [
    pytest.param(ValueError("flux has NaNs"), r"row 0 \(a.txt\): flux has NaNs", id="lists_a_spectrum_the_model_rejects"),
    # A broad except used to count any error as one failed spectrum and go on.
    pytest.param(RuntimeError("Tensor for argument weight is on cpu but expected on mps"), "expected on mps",
                 id="stops_at_once_on_an_unexpected_error"),
])
def test_extend_catalog_raises_what_the_model_could_not_extend(catalog, tmp_path, error, message):
    completer = _stubbed_completer()

    def fail(wavelength, flux, teff, logg, meta, extension_range):
        raise error

    completer._completer.complete_sed = fail
    with pytest.raises(RuntimeError, match=message):
        completer.extend_catalog(str(catalog), str(tmp_path / "out"), verbose=False)
