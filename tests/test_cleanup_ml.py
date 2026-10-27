"""Tests for the ml cleanup.

The characterisation tests pin what training and inference compute: tiny
seeded inputs, run on the CPU, with the results stored here as literals.
They were recorded on ``cleanup/base`` before the ml modules were split up,
so a refactor that changes the numerics, the seeds or the order of random
draws fails here. ``np.random.random()`` and ``torch.rand`` drawn after
training catch a change in how many random numbers training consumed.

Tests that need torch skip without it; the rest are torch-free, like the
other ``test_ml_*`` files.
"""

import json

import numpy as np
import pytest

from sed_tools import ml
from sed_tools import ml_sed_completer as mc
from sed_tools import ml_sed_generator as mg


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def cpu_torch(monkeypatch):
    """torch on the CPU, single-threaded and seeded; matplotlib on Agg.

    Both ``.device`` properties and the network classes pick CUDA or MPS
    when available, and the two can disagree (see the cleanup ledger), so
    the tests pin everything to the CPU.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    import matplotlib
    matplotlib.use("Agg")
    import sed_tools.ml_utils as ml_utils

    monkeypatch.setattr(ml_utils, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    np.random.seed(0)
    torch.manual_seed(0)
    yield torch
    torch.set_num_threads(threads)


@pytest.fixture
def tiny_library(tmp_path, cube_writer):
    """A 3 x 2 x 2 flux cube whose spectra differ in level and slope."""
    teff = np.array([4000.0, 5000.0, 6000.0])
    logg = np.array([4.0, 4.5])
    meta = np.array([-0.5, 0.0])
    wl = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
    flux = np.zeros((3, 2, 2, 200))
    for ti in range(3):
        for li in range(2):
            for mi in range(2):
                level = 1e-10 * (1 + 0.1 * ti + 0.01 * li + 0.02 * mi)
                flux[ti, li, mi] = level * (wl / 5000.0) ** (-2 - 0.1 * ti)
    lib = tmp_path / "lib"
    cube_writer(lib / "flux_cube.bin", teff, logg, meta, wl, flux)
    return lib


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

def test_completer_prepare_training_data_values(tiny_library):
    np.random.seed(3)
    X, y, params, masks = mc.SEDCompleter()._prepare_training_data(str(tiny_library), 12)

    assert params.tolist() == _SAMPLED_PARAMS
    assert X.astype(np.float64).sum() == pytest.approx(9.305711475402932e-07, rel=1e-12)
    assert y.astype(np.float64).sum() == pytest.approx(2.7577729128764674e-06, rel=1e-12)
    assert masks.sum(axis=1).tolist() == [166.0] * 12
    assert np.random.random() == 0.765826378456161


def test_generator_prepare_training_data_values(tiny_library):
    np.random.seed(3)
    params, flux, masks = mg.SEDGenerator()._prepare_training_data(str(tiny_library), 12)

    assert params.tolist() == _SAMPLED_PARAMS
    assert flux.astype(np.float64).sum() == pytest.approx(2.7577729128243876e-06, rel=1e-12)
    assert masks.sum(axis=1).tolist() == [566.0] * 12
    assert np.random.random() == 0.20724287813818676


# ---------------------------------------------------------------------
# Characterisation: training (torch, CPU)
# ---------------------------------------------------------------------

def test_completer_train_values(cpu_torch, tiny_library, tmp_path):
    torch = cpu_torch
    completer = mc.SEDCompleter()
    history = completer.train(str(tiny_library), str(tmp_path / "model"), epochs=3,
                              batch_size=3, max_samples=12, hidden_layers=[8], patience=20)

    assert history["train_loss"] == pytest.approx(
        [1.217055320739746, 1.2087888717651367, 1.1866831382115681], rel=1e-6)
    assert history["val_loss"] == pytest.approx(
        [0.9876893758773804, 0.977878987789154, 0.9719147086143494], rel=1e-6)
    assert completer.scaler_params == pytest.approx({
        "teff_mean": 5000.0, "teff_std": 816.49658203125,
        "logg_mean": 4.25, "logg_std": 0.25,
        "meta_mean": -0.25, "meta_std": 0.25,
        "flux_log_mean": -10.225753784179688, "flux_log_std": 0.7680805921554565,
        "ext_log_mean": -10.368162155151367, "ext_log_std": 1.48019278049469,
    }, rel=1e-9)
    assert _state_sums(completer.model) == pytest.approx((37.48151246241832, 1686.1791792145195), rel=1e-5)
    assert np.random.random() == 0.9571551589530464
    assert torch.rand(2).tolist() == pytest.approx([0.41561412811279297, 0.5646539926528931], abs=0)

    config = json.loads((tmp_path / "model" / "config.json").read_text())
    assert sorted(config) == ["architecture", "framework", "regions", "scaler_params",
                              "version", "wavelength_grid"]
    assert config["architecture"] == {"input_dim": 400, "output_dim": 1000,
                                      "hidden_layers": [8], "dropout": 0.3}
    assert config["regions"] == {"known_start": 300, "known_end": 700, "known_width": 400}
    assert (config["version"], config["framework"]) == ("2.0.0", "pytorch")
    assert config["wavelength_grid"] == np.logspace(2, 5, 1000).tolist()
    assert sorted(p.name for p in (tmp_path / "model").iterdir()) == [
        "config.json", "model.pt", "prediction_examples.png", "training_plots.png"]


def test_generator_train_values(cpu_torch, tiny_library, tmp_path):
    torch = cpu_torch
    generator = mg.SEDGenerator()
    history = generator.train(str(tiny_library), str(tmp_path / "model"), epochs=3,
                              batch_size=3, max_samples=12, hidden_layers=[8])

    assert history["train_loss"] == pytest.approx(
        [1.2670124769210815, 1.2379244565963745, 1.1913456122080486], rel=1e-6)
    assert history["val_loss"] == pytest.approx(
        [1.0461859703063965, 1.0439506769180298, 1.041712760925293], rel=1e-6)
    assert generator.scaler_params == pytest.approx({
        "teff_mean": 5000.0, "teff_std": 816.49658203125, "teff_min": 4000.0, "teff_max": 6000.0,
        "logg_mean": 4.25, "logg_std": 0.25, "logg_min": 4.0, "logg_max": 4.5,
        "meta_mean": -0.25, "meta_std": 0.25, "meta_min": -0.5, "meta_max": 0.0,
        "flux_log_mean": -10.267520904541016, "flux_log_std": 1.031362771987915,
    }, rel=1e-9)
    assert _state_sums(generator.model) == pytest.approx((26.80382775522594, 1610.2340406870499), rel=1e-5)
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

def _fixed_completer(torch):
    torch.manual_seed(0)
    completer = mc.SEDCompleter()
    completer.model = mc.SEDCompleterNetwork(input_dim=400, output_dim=1000,
                                             hidden_layers=[16, 8], dropout=0.3)
    completer.wavelength_grid = completer._create_wavelength_grid()
    completer._known_start, completer._known_end = 300, 700
    completer.scaler_params = {
        "teff_mean": 5500.0, "teff_std": 1500.0, "logg_mean": 4.0, "logg_std": 0.5,
        "meta_mean": 0.0, "meta_std": 0.5, "flux_log_mean": 6.5, "flux_log_std": 1.0,
        "ext_log_mean": 6.0, "ext_log_std": 1.5,
    }
    return completer


def _fixed_generator(torch):
    torch.manual_seed(0)
    generator = mg.SEDGenerator()
    generator.model = mg.SEDGeneratorNetwork(output_dim=1000, hidden_layers=[16, 8], dropout=0.2)
    generator.wavelength_grid = generator._create_wavelength_grid()
    generator.scaler_params = {
        "teff_mean": 5500.0, "teff_std": 1500.0, "teff_min": 4000.0, "teff_max": 7000.0,
        "logg_mean": 4.0, "logg_std": 0.5, "logg_min": 3.5, "logg_max": 5.0,
        "meta_mean": 0.0, "meta_std": 0.5, "meta_min": -1.0, "meta_max": 0.5,
        "flux_log_mean": -10.0, "flux_log_std": 1.0,
    }
    generator.config = {"parameter_ranges": {"teff": [4000.0, 7000.0], "logg": [3.5, 5.0],
                                             "meta": [-1.0, 0.5]}}
    return generator


def test_completer_complete_sed_values(cpu_torch):
    completer = _fixed_completer(cpu_torch)
    wl = np.logspace(np.log10(3000.0), np.log10(20000.0), 300)
    flux = mc.planck_function(wl, 5800.0)

    wl_out, flux_out = completer.complete_sed(wl, flux, teff=5800.0, logg=4.4, meta=0.1,
                                              extension_range=(500.0, 60000.0))

    assert (wl_out[0], wl_out[-1]) == pytest.approx((500.0, 60000.0), rel=1e-14)
    assert flux_out[[0, 150, 300, 450, 600, 750, 900, 999]] == pytest.approx([
        410416.9146332419, 1687776.9083537995, 3334628.567953557, 8817909.586145839,
        4883482.30195489, 710804.2571917092, 2839454.620631518, 2065668.2893293651,
    ], rel=1e-6)
    assert flux_out.sum() == pytest.approx(3194475556.049119, rel=1e-6)
    # Inside the input's coverage the output is the input, interpolated.
    inside = (wl_out >= wl[0]) & (wl_out <= wl[-1])
    np.testing.assert_array_equal(flux_out[inside], np.interp(wl_out[inside], wl, flux))


def test_generator_generate_values(cpu_torch):
    generator = _fixed_generator(cpu_torch)

    wl_out, flux_out = generator.generate(5200.0, 4.2, -0.2)

    np.testing.assert_array_equal(wl_out, np.logspace(2, 5, 1000))
    assert flux_out.dtype == np.float32
    assert flux_out[[0, 250, 500, 750, 999]] == pytest.approx([
        7.192110101916782e-11, 7.95194882163841e-11, 1.8567157533677658e-10,
        1.7018070286312081e-10, 1.7596650525586455e-10,
    ], rel=1e-5)
    assert flux_out.astype(np.float64).sum() == pytest.approx(1.142879055664181e-07, rel=1e-5)


def test_generate_check_bounds_prints_each_out_of_range_parameter(cpu_torch, capsys):
    generator = _fixed_generator(cpu_torch)

    generator.generate(9000.0, 5.5, 0.6)

    out = capsys.readouterr().out
    assert "Warning: Teff=9000 outside range [4000, 7000]" in out
    assert "Warning: logg=5.50 outside range [3.50, 5.00]" in out
    assert "Warning: [M/H]=0.60 outside range [-1.00, 0.50]" in out


def test_generate_with_outputs_saves_the_generated_sed(cpu_torch, tmp_path):
    generator = _fixed_generator(cpu_torch)

    wl, flux = generator.generate_with_outputs(5200.0, 4.2, -0.2, str(tmp_path))

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["sed_T5200_g4.20_m-0.20.txt", "sed_T5200_g4.20_m-0.20_params.png",
                     "sed_T5200_g4.20_m-0.20_spectrum.png"]
    text = (tmp_path / "sed_T5200_g4.20_m-0.20.txt").read_text().splitlines()
    assert text[:2] == ["# wavelength_A flux_erg/s/cm2/A", "# # Teff=5200.0 logg=4.2 [M/H]=-0.2"]
    np.testing.assert_array_equal(np.loadtxt(tmp_path / "sed_T5200_g4.20_m-0.20.txt"),
                                  np.loadtxt([f"{w:.6e} {f:.6e}" for w, f in zip(wl, flux)]))


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


def test_extend_catalog_values(catalog, tmp_path):
    completer = _stubbed_completer()
    out_dir = tmp_path / "out"

    summary = completer.extend_catalog(str(catalog), str(out_dir), extension_range=(500.0, 60000.0))

    assert summary == {"extended": 2, "failed": 0, "skipped": 1, "total": 3, "output_dir": str(out_dir)}
    assert completer._completer.calls == [(5000.0, 4.0, 0.0, (500.0, 60000.0)),
                                          (6000.0, 4.5, -0.5, (500.0, 60000.0))]
    assert completer._completer.plotted == ["a.txt", "b.txt"]
    assert (out_dir / "lookup_table.csv").read_text() == (
        "file_name,teff,logg,meta,extra,extended_by_ml,wl_min,wl_max\n"
        "a.txt,5000,4.0,0.0,x,True,499.99999999999994,60000.00000000003\n"
        "b.txt,6000,4.5,-0.5,y,True,499.99999999999994,60000.00000000003\n"
        "c.txt,7000,3.5,0.2,z,False,,\n"
    )
    data_a = np.loadtxt(catalog / "a.txt")
    grid = np.logspace(np.log10(500.0), np.log10(60000.0), 20)
    expected_a = np.column_stack([grid, np.interp(grid, data_a[:, 0], data_a[:, 1]) * 1.5])
    np.testing.assert_allclose(np.loadtxt(out_dir / "a.txt"), expected_a, rtol=5e-7)
    np.testing.assert_allclose(np.loadtxt(out_dir / "c.txt"), np.loadtxt(catalog / "c.txt"), rtol=5e-7)


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


def test_completer_workflow_trains_on_the_chosen_library(stellar_dir, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(mc.SEDCompleter, "train", lambda self, **kw: calls.append(kw) or {})
    _feed_input(monkeypatch, ["1", "2", "3", "5", "", ""])

    mc.run_interactive_workflow(str(stellar_dir), str(tmp_path / "models"))

    out = capsys.readouterr().out
    assert calls == [{"library_path": str(stellar_dir / "gridB"),
                      "output_path": str(tmp_path / "models" / "sed_model_gridB"),
                      "epochs": 3, "batch_size": 5}]
    assert "   1) gridA                          (1 SEDs)" in out
    assert "TRAINING COMPLETE!" in out


@pytest.mark.parametrize("answers, message", [
    (["1", "x"], "Invalid selection"),
    (["1", "0"], "Invalid selection"),
    (["1", "9"], "Invalid selection"),
    (["1", "1", "abc", ""], "Invalid parameters"),
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

    extend_calls = []

    def fake_extend(self, input_dir, output_dir, extension_range, verbose):
        extend_calls.append((input_dir, output_dir, extension_range, verbose))
        return {"extended": 2, "failed": 0, "skipped": 1, "total": 3, "output_dir": output_dir}

    monkeypatch.setattr(mc.SEDCompleter, "load", fake_load)
    monkeypatch.setattr(ml.Completer, "extend_catalog", fake_extend)
    _feed_input(monkeypatch, ["2", "1", "1", ""])

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
