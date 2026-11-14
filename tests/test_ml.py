"""ML: the completer and generator, what their training and inference compute, their menus, extend_catalog,
list_models, and their failures; the torch-free sed_tools.ml API; MPS and the optuna auto-trainer.

The known-answer rows pin what training and inference compute on tiny seeded inputs on the CPU, recorded
before the ml modules were split up: a refactor that changes the numerics, the seeds or the order of random
draws fails them. ``np.random.random()`` and ``torch.rand`` drawn after training catch a change in how many
random numbers training consumed. The tolerances allow for float32 and BLAS differences between machines; a
change in the random draws moves these values by far more.

Every row that trains is seeded (``cpu_torch`` seeds numpy and torch; the optuna rows seed its sampler too).
Rows that need torch, scikit-learn, optuna or MPS skip without them; the rest are torch-free.
"""

import functools
import json
import logging

import numpy as np
import pytest

from sed_tools import ml
from sed_tools.ml import Completer, completer as mc, generator as mg, model_info

from support import (KURUCZ, answer, cpu_torch, fixed_completer, fixed_generator, scenarios, tiny_library,
                     write_flux_cube)

SCENARIOS, row = scenarios()

WL = np.logspace(np.log10(1000.0), np.log10(50000.0), 200)
TINY = dict(batch_size=3, max_samples=12, hidden_layers=[8])  # training on tiny_library
CM1 = {"version": "2.0.0", "framework": "pytorch",  # a completer model's config
       "architecture": {"input_dim": 400, "hidden_layers": [8], "output_dim": 1000}}


def write_config(directory, config):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))


def grid(t, teff, flux):
    """A flux cube at ``t.tmp/lib`` over ``teff`` (logg 4, [M/H] 0) and 200 wavelengths: ``flux`` is
    the (n, 1, 1, 200) array, or one value everywhere."""
    write_flux_cube(t.tmp / "lib" / "flux_cube.bin", teff, [4.0], [0.0], WL, np.broadcast_to(flux, (2, 1, 1, 200)))
    return str(t.tmp / "lib")


def kurucz(t):
    """The 12-spectrum golden Kurucz grid (3 Teff x 2 logg x 2 [M/H]) with its flux cube."""
    import shutil

    from sed_tools.cube.build import precompute_flux_cube

    library = t.tmp / "Kurucz2003all"
    shutil.copytree(KURUCZ, library)
    precompute_flux_cube(str(library), str(library / "flux_cube.bin"))
    return library


def seeded_optuna(t):
    """optuna, its sampler seeded: ml/tuning.py looks up optuna.create_study when it runs."""
    optuna = pytest.importorskip("optuna")
    t.mp.setattr(optuna, "create_study",
                 functools.partial(optuna.create_study, sampler=optuna.samplers.TPESampler(seed=1)))


def one_point_grid(directory):
    """A flux cube of one spectrum (5000 K, logg 4, [M/H] 0) at two wavelengths in ``directory``."""
    write_flux_cube(directory / "flux_cube.bin", [5000.0], [4.0], [0.0], [1000.0, 2000.0], np.ones((1, 1, 1, 2)))


def stellar_grids(t):
    """gridA and gridB have flux cubes; gridB's lookup has a '#' header."""
    base = t.tmp / "stellar"
    for name in ("gridA", "gridB"):
        one_point_grid(base / name)
    (base / "gridA" / "lookup_table.csv").write_text("file_name,teff\nx.txt,5000\n")
    (base / "gridB" / "lookup_table.csv").write_text("#file_name,teff\nq.txt,5000\nr.txt,6000\n")
    (base / "notagrid").mkdir()
    return base


def loads_a_model(t):
    """SEDCompleter.load gives a model whose grid is 100-1e5 A; returns the calls to ml.Completer.extend_catalog."""
    def fake_load(self, path):
        self.model = object()
        self.wavelength_grid = np.logspace(2, 5, 1000)

    calls = []

    def fake_extend(self, input_dir, output_dir, extension_range, verbose):
        calls.append((input_dir, output_dir, extension_range, verbose))
        return {"extended": 2, "failed": 0, "skipped": 1, "total": 3, "output_dir": output_dir}

    t.mp.setattr(mc.SEDCompleter, "load", fake_load)
    t.mp.setattr(ml.Completer, "extend_catalog", fake_extend)
    return calls


def window_completer(torch):
    """A completer whose network answers zeros, on the CPU."""
    class Constant:
        def eval(self):
            return self

        def __call__(self, x):
            return torch.zeros((1, 1000))

    completer = fixed_completer(torch)
    completer.model = Constant()
    completer._device = torch.device("cpu")
    return completer


def assert_keeps_original(wl_out, flux_out, wl, flux):
    """The output is the input verbatim with points added only beyond its ends."""
    start = int(np.searchsorted(wl_out, wl[0]))
    np.testing.assert_array_equal(wl_out[start:start + len(wl)], wl)
    np.testing.assert_array_equal(flux_out[start:start + len(wl)], flux)
    assert np.all(wl_out[:start] < wl[0])
    assert np.all(wl_out[start + len(wl):] > wl[-1])


def state_sums(model):
    values = model.state_dict().values()
    return (float(sum(v.double().sum() for v in values)), float(sum(v.double().abs().sum() for v in values)))


# --------------------------------------------------------------------- extend_catalog with a stub network
class StubCompleter:
    """Stands in for SEDCompleter: records each spectrum's parameters and range, extends to a fixed grid."""

    def __init__(self):
        self.model, self.calls, self.ranges = object(), [], []

    def complete_sed(self, wavelength, flux, teff, logg, meta, extension_range):
        self.calls.append((teff, logg, meta))
        self.ranges.append(tuple(extension_range))
        grid = np.logspace(np.log10(extension_range[0]), np.log10(extension_range[1]), 20)
        return grid, np.interp(grid, wavelength, flux) * (1.0 + teff / 1e4)

    def plot_inference_examples(self, examples, output_dir):
        pass


def stubbed_completer():
    completer = ml.Completer()
    completer._completer = StubCompleter()
    return completer


def write_spectrum(path, lo, hi, scale=1.0):
    wl = np.logspace(np.log10(lo), np.log10(hi), 50)
    np.savetxt(path, np.column_stack([wl, scale * 1e-12 * (wl / 5000.0) ** -2]))


def three_spectra(t):
    """Three spectra; c.txt already covers 100-100000 A, so it is copied."""
    import pandas as pd

    cat = t.tmp / "catalog"
    cat.mkdir()
    write_spectrum(cat / "a.txt", 3000, 20000)
    write_spectrum(cat / "b.txt", 2000, 30000, scale=2.0)
    write_spectrum(cat / "c.txt", 100, 100000, scale=3.0)
    pd.DataFrame({"file_name": ["a.txt", "b.txt", "c.txt"], "teff": [5000, 6000, 7000],
                  "logg": [4.0, 4.5, 3.5], "meta": [0.0, -0.5, 0.2], "extra": ["x", "y", "z"]}
                 ).to_csv(cat / "lookup_table.csv", index=False)
    return cat


# ======================================================================
# Tables
# ======================================================================
def list_models_warns_naming_the_unreadable_config(t, cls):
    (t.tmp / "broken").mkdir()
    (t.tmp / "broken" / "config.json").write_text("{not valid json")
    assert cls.list_models(str(t.tmp)) == []
    assert f"Skipping {t.tmp / 'broken'}: could not read its config.json" in t.caplog.text


def training_drops_a_last_batch_of_one_spectrum(t, engine):
    # 9 training spectra in batches of 8: BatchNorm used to raise "Expected more than 1 value per channel".
    cpu_torch(t)
    history = engine().train(str(tiny_library(t)), str(t.tmp / "model"), epochs=1, batch_size=8, max_samples=12,
                             hidden_layers=[8])
    assert len(history["train_loss"]) == 1


def auto_train_best(t, lib, out):
    seeded_optuna(t)
    from sed_tools.ml.tuning import AutoTrainer

    AutoTrainer().train_best(lib, out, n_trials=1, epochs=1)


def train_raises_before_saving_when_every_batch_is_nan(t, network, train):
    cpu_torch(t)
    t.mp.setattr(network, "__call__", lambda self, x: self.model(x) * float("nan"))
    with pytest.raises(RuntimeError, match="produced no usable epoch"):
        train(t, str(tiny_library(t)), str(t.tmp / "model"))
    out = t.capsys.readouterr().out
    assert "Warning: NaN loss at epoch 1, skipping batch" in out
    assert "Epoch 1: No valid batches, stopping" in out
    assert not (t.tmp / "model").exists()


def workflow_ends_on_the_answer(t, engine, answers, message):
    # 0 and -1 used to index from the end of the list and pick the last entry.
    stellar_dir = stellar_grids(t)
    write_config(t.tmp / "models" / "gm1", {"model_type": "sed_generator"})
    for cls in (mc.SEDCompleter, mg.SEDGenerator):
        t.mp.setattr(cls, "train", lambda self, **kw: pytest.fail("trained"))
    answer(t.mp, *answers, echo=True)
    engine.run_interactive_workflow(str(stellar_dir), str(t.tmp / "models"))
    assert t.capsys.readouterr().out.rstrip().endswith(message)


def extend_catalog_raises_when_a_column_is_missing(t, missing):
    # Missing parameters used to default to 5000 K / 4.5 / 0.0, and a missing
    # file column matched rows to the sorted *.txt files by position.
    import pandas as pd

    catalog = three_spectra(t)
    df = pd.read_csv(catalog / "lookup_table.csv").drop(columns=[missing])
    df.to_csv(catalog / "lookup_table.csv", index=False)
    completer = stubbed_completer()
    with pytest.raises(ValueError, match=r"has no column for (file|teff|logg|meta)\. Columns read"):
        completer.extend_catalog(str(catalog), str(t.tmp / "out"))
    assert completer._completer.calls == []


def extend_catalog_raises_what_the_model_could_not_extend(t, error, message):
    # A broad except used to count any error as one failed spectrum and go on.
    catalog = three_spectra(t)
    completer = stubbed_completer()

    def fail(wavelength, flux, teff, logg, meta, extension_range):
        raise error

    completer._completer.complete_sed = fail
    with pytest.raises(RuntimeError, match=message):
        completer.extend_catalog(str(catalog), str(t.tmp / "out"), verbose=False)


LIST_MODELS_WARNS = {f"list_models_warns_naming_the_unreadable_config_{cls.__name__}": cls
                     for cls in (mc.SEDCompleter, mg.SEDGenerator)}
LAST_BATCH_OF_ONE = {f"training_drops_a_last_batch_of_one_spectrum_{cls.__name__}": cls
                     for cls in (mc.SEDCompleter, mg.SEDGenerator)}
ALL_NAN = {
    "completer_train_raises_before_saving_when_every_batch_is_nan": (
        mc.SEDCompleterNetwork, lambda t, lib, out: mc.SEDCompleter().train(lib, out, epochs=2, **TINY)),
    "generator_train_raises_before_saving_when_every_batch_is_nan": (
        mg.SEDGeneratorNetwork, lambda t, lib, out: mg.SEDGenerator().train(lib, out, epochs=2, **TINY)),
    "auto_trainer_stops_on_a_trial_with_no_usable_epoch": (mg.SEDGeneratorNetwork, auto_train_best)}
ENDING_ANSWERS = {
    "completer_workflow_is_cancelled_by_n_at_start_training": (mc, ["1", "1", "", "", "", "n"], "Cancelled"),
    "generator_workflow_is_cancelled_by_n_at_start_training": (mg, ["1", "1", "", "", "", "n"], "Cancelled"),
    "completer_workflow_rejects_bad_answers_invalid_choice": (mc, ["9"], "Invalid choice"),
    "generator_workflow_rejects_zero_and_negative_menu_numbers_0": (mg, ["3", "0"], "Invalid selection"),
    "generator_workflow_rejects_zero_and_negative_menu_numbers_minus_1": (mg, ["3", "-1"], "Invalid selection")}
MISSING_COLUMN = {f"extend_catalog_raises_when_a_column_is_missing_{name}": name for name in ("teff", "file_name")}
NOT_EXTENDED = {
    "extend_catalog_raises_what_the_model_could_not_extend_lists_a_spectrum_the_model_rejects": (
        ValueError("flux has NaNs"), r"row 0 \(a.txt\): flux has NaNs"),
    "extend_catalog_raises_what_the_model_could_not_extend_stops_at_once_on_an_unexpected_error": (
        RuntimeError("Tensor for argument weight is on cpu but expected on mps"), "expected on mps")}


# ======================================================================
# The torch-free API, list_models and data preparation
# ======================================================================
@row
def model_info_returns_full_config(t):
    config = {"version": "2.0.0", "framework": "pytorch", "architecture": {"input_dim": 10}}
    write_config(t.tmp / "m1", config)
    assert model_info(str(t.tmp / "m1")) == config


@row
def completer_default_construction_is_untrained(t):
    c = Completer()
    assert c.is_trained is False
    assert c.model_path is None
    assert c.config == {}
    assert c.wavelength_grid is None


@row
def list_models_skips_dirs_without_config(t):
    (t.tmp / "not_a_model").mkdir()
    assert mc.SEDCompleter.list_models(str(t.tmp)) == []


@row
def list_models_leaves_out_generators(t):
    # 0.5.0: generators were listed too, and choosing one raised KeyError.
    # Completer configs have no model_type; SEDGenerator writes 'sed_generator'.
    for name, config in [("generator", {"model_type": "sed_generator"}), ("completer", {})]:
        (t.tmp / name).mkdir()
        (t.tmp / name / mc.SEDCompleter.CONFIG_FILE).write_text(json.dumps(config))
    assert [m["name"] for m in mc.SEDCompleter.list_models(str(t.tmp))] == ["completer"]


# ======================================================================
# MPS, the two networks, extend_catalog on '#'-headed tables, the optuna auto-trainer
# ======================================================================
@row
def train_complete_and_generate_run_on_mps(t):
    # Every train, complete_sed and generate used to raise "Tensor for
    # argument weight is on cpu but expected on mps" here.
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("needs torch with MPS (Apple silicon)")
    pytest.importorskip("sklearn")
    import matplotlib
    matplotlib.use("Agg")
    np.random.seed(0)
    torch.manual_seed(0)
    torch.mps.manual_seed(0)
    # The loss-curve plots use no torch, and after one epoch take seconds to save
    t.mp.setattr(mc.SEDCompleter, "_plot_training_results", lambda *args: None)
    t.mp.setattr(mg.SEDGenerator, "_plot_training", lambda *args: None)
    library = tiny_library(t)

    completer = mc.SEDCompleter()
    completer.train(str(library), str(t.tmp / "completer"), epochs=1, **TINY)
    assert completer.device.type == "mps"
    assert {p.device.type for p in completer.model.parameters()} == {"mps"}

    wl = np.logspace(np.log10(3000.0), np.log10(20000.0), 300)
    sed = mc.planck_function(wl, 5000.0)
    for model in (completer, mc.SEDCompleter(t.tmp / "completer")):
        wl_out, flux_out = model.complete_sed(wl, sed, teff=5000.0, logg=4.2, meta=-0.2,
                                              extension_range=(500.0, 60000.0))
        assert np.all(np.isfinite(flux_out)) and len(flux_out) == len(wl_out)

    generator = mg.SEDGenerator()
    generator.train(str(library), str(t.tmp / "generator"), epochs=1, **TINY)
    assert generator.device.type == "mps"
    assert {p.device.type for p in generator.model.parameters()} == {"mps"}

    for model in (generator, mg.SEDGenerator(t.tmp / "generator")):
        wl_out, flux_out = model.generate(5000.0, 4.2, -0.2)
        assert np.all(np.isfinite(flux_out)) and len(flux_out) == len(wl_out) == 1000


@row
def both_networks_keep_their_defaults_model_pt_keys_and_move_with_to(t):
    # The two networks share one base class (ml/common.Network); SEDGeneratorNetwork gained .to().
    # model.pt is the mlp's state_dict, keyed by layer index as 0.4.1 and 0.5.0 wrote it.
    torch = cpu_torch(t)
    completer = mc.SEDCompleterNetwork(input_dim=4, output_dim=5, hidden_layers=[3, 2])
    generator = mg.SEDGeneratorNetwork(output_dim=5, hidden_layers=[3, 2])
    batch_norm = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]
    keys = (["0.weight", "0.bias"] + [f"1.{k}" for k in batch_norm] + ["4.weight", "4.bias"]
            + [f"5.{k}" for k in batch_norm] + ["7.weight", "7.bias"])
    for network, n_in, dropout in ((completer, 4 + 3, 0.3), (generator, 3, 0.2)):
        assert (network.output_dim, network.hidden_layers, network.dropout) == (5, [3, 2], dropout)
        assert list(network.state_dict()) == keys
        assert tuple(network.state_dict()["0.weight"].shape) == (3, n_in)
        assert network.to(torch.device("cpu")) is network and network.device == torch.device("cpu")
        assert {p.device.type for p in network.parameters()} == {"cpu"}
    assert completer.input_dim == 4
    assert mc.SEDCompleterNetwork(4, 5).hidden_layers == [256, 128, 64]
    assert mg.SEDGeneratorNetwork(5).hidden_layers == [512, 512, 256, 256]


@row
def extend_catalog_extends_each_spectrum_with_its_own_parameters(t):
    # pd.read_csv(comment='#') used to drop the '#file_name,...' header, so the
    # first spectrum became the header and every other one was extended as
    # Teff 5000, logg 4.5, [M/H] 0; on cleanup/base it raised instead.
    import pandas as pd
    from sed_tools.grids.lookup import write_lookup_csv

    catalog = t.tmp / "catalog"
    catalog.mkdir()
    names = ["a.txt", "b.txt", "c.txt"]
    for name, (lo, hi) in zip(names, [(3000, 20000), (2000, 30000), (4000, 9000)]):
        write_spectrum(catalog / name, lo, hi)
    write_lookup_csv({"file_name": names, "teff": [4500.0, 5800.0, 7200.0],
                      "logg": [4.0, 4.4, 3.5], "metallicity": [-1.0, 0.0, 0.3]},
                     str(catalog / "lookup_table.csv"))
    assert (catalog / "lookup_table.csv").read_text().startswith("#file_name,teff,logg,metallicity\n")
    completer = stubbed_completer()
    out_dir = t.tmp / "out"

    summary = completer.extend_catalog(str(catalog), str(out_dir), extension_range=(500.0, 60000.0),
                                       verbose=False)

    assert summary == {"extended": 3, "failed": 0, "skipped": 0, "total": 3, "output_dir": str(out_dir)}
    assert completer._completer.calls == [(4500.0, 4.0, -1.0), (5800.0, 4.4, 0.0), (7200.0, 3.5, 0.3)]
    written = pd.read_csv(out_dir / "lookup_table.csv")
    assert written["file_name"].tolist() == names
    assert written[["teff", "logg", "metallicity"]].values.tolist() == [
        [4500.0, 4.0, -1.0], [5800.0, 4.4, 0.0], [7200.0, 3.5, 0.3]]
    assert sorted(p.name for p in out_dir.glob("*.txt")) == names


@row
def auto_trainer_saves_a_model_sedgenerator_and_the_api_load(t):
    # It used to save a ResidualMLP's weights as model_best.pt and overwrite
    # config.json with keys of its own, so nothing could load the model.
    cpu_torch(t)
    seeded_optuna(t)
    from sed_tools import api
    from sed_tools.ml.tuning import AutoTrainer

    kurucz_library = kurucz(t)
    output = t.tmp / "base" / "models" / "auto"
    generator, _ = AutoTrainer().train_best(str(kurucz_library), str(output), n_trials=2, epochs=2)

    assert isinstance(generator, mg.SEDGenerator)
    assert sorted(p.name for p in output.iterdir()) == [
        "config.json", "model.pt", "prediction_examples.png", "training_plots.png"]
    config = json.loads((output / "config.json").read_text())
    assert sorted(config) == ["architecture", "framework", "model_type", "parameter_ranges",
                              "scaler_params", "training_params", "version", "wavelength_grid"]
    assert (config["model_type"], config["version"], config["framework"]) == ("sed_generator", "1.0.0", "pytorch")
    # The scalers are those of SEDGenerator's own preparation of the cube (11 usable spectra)
    reference = mg.SEDGenerator()
    params, flux, masks = reference._prepare_training_data(str(kurucz_library), 10000)
    reference._normalise_training_data(params, flux, masks, str(kurucz_library))
    assert len(params) == 11
    assert config["scaler_params"] == pytest.approx(reference.scaler_params, rel=1e-6)
    assert [m["name"] for m in mg.SEDGenerator.list_models(str(output.parent))] == ["auto"]

    wl, flux_out = mg.SEDGenerator(output).generate(6000.0, 4.25, -0.25)
    assert len(wl) == len(flux_out) == 1000
    assert np.all(np.isfinite(flux_out)) and np.all(flux_out > 0)
    wl_api, flux_api = api.MLGenerator(base_dir=t.tmp / "base").load("auto").generate(
        teff=6000.0, logg=4.25, metallicity=-0.25)
    np.testing.assert_array_equal(wl_api, wl)
    np.testing.assert_array_equal(flux_api, flux_out)


# ======================================================================
# Known answers: training and inference (torch, CPU)
# ======================================================================
@row
def generator_train_values(t):
    torch = cpu_torch(t)
    generator = mg.SEDGenerator()
    history = generator.train(str(tiny_library(t)), str(t.tmp / "model"), epochs=3, **TINY)

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
    assert state_sums(generator.model) == pytest.approx((26.80382775522594, 1610.2340406870499), rel=1e-3)
    assert np.random.random() == 0.3834415188257777
    assert torch.rand(2).tolist() == pytest.approx([0.3398873209953308, 0.19384098052978516], abs=0)

    config = json.loads((t.tmp / "model" / "config.json").read_text())
    assert sorted(config) == ["architecture", "framework", "model_type", "parameter_ranges",
                              "scaler_params", "training_params", "version", "wavelength_grid"]
    assert config["architecture"] == {"output_dim": 1000, "hidden_layers": [8], "dropout": 0.2}
    assert config["parameter_ranges"] == {"teff": [4000.0, 6000.0], "logg": [4.0, 4.5],
                                          "meta": [-0.5, 0.0]}
    assert (config["model_type"], config["version"]) == ("sed_generator", "1.0.0")
    assert len(config["training_params"]) == 9


@row
def completer_complete_sed_values(t):
    completer = fixed_completer(cpu_torch(t))
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


@row
def generate_check_bounds_prints_each_out_of_range_parameter(t):
    fixed_generator(cpu_torch(t)).generate(9000.0, 5.5, 0.6)

    out = t.capsys.readouterr().out
    assert "Warning: Teff=9000 outside range [4000, 7000]" in out
    assert "Warning: logg=5.50 outside range [3.50, 5.00]" in out
    assert "Warning: [M/H]=0.60 outside range [-1.00, 0.50]" in out


# ======================================================================
# The interactive workflows (training and inference stubbed)
# ======================================================================
@row
def completer_workflow_extends_the_chosen_grid(t):
    stellar_dir = stellar_grids(t)
    models = t.tmp / "models"
    write_config(models / "cm1", CM1)
    extend_calls = loads_a_model(t)
    # 0.5.0: Enter at the two range prompts keeps the model's own range
    answer(t.mp, "2", "1", "1", "", "", "", echo=True)

    mc.run_interactive_workflow(str(stellar_dir), str(models))

    out = t.capsys.readouterr().out
    assert extend_calls == [(str(stellar_dir / "gridA"), str(stellar_dir / "gridA_ml_extended"),
                             (100.0, 100000.0), True)]
    assert "      Architecture: 400 -> [8] -> 1000" in out
    assert "  2 SEDs extended successfully" in out


@row
def completer_workflow_lists_models(t):
    models = t.tmp / "models"
    write_config(models / "cm1", CM1)
    answer(t.mp, "3", echo=True)

    mc.run_interactive_workflow(str(t.tmp / "stellar"), str(models))

    out = t.capsys.readouterr().out
    assert "\n  cm1\n" in out
    assert "    Framework: pytorch, Version: 2.0.0" in out
    assert "    Architecture: 400 -> [8] -> 1000" in out


@row
def generator_workflow_trains_on_the_chosen_library(t):
    stellar_dir = stellar_grids(t)
    calls = []
    t.mp.setattr(mg.SEDGenerator, "train", lambda self, *a, **kw: calls.append((a, kw)) or {})
    answer(t.mp, "1", "2", "3", "5", "", "", echo=True)

    mg.run_interactive_workflow(str(stellar_dir), str(t.tmp / "models"))

    assert calls == [((str(stellar_dir / "gridB"), str(t.tmp / "models" / "sed_generator_gridB")),
                      {"epochs": 3, "batch_size": 5})]
    assert "TRAINING COMPLETE!" in t.capsys.readouterr().out


@row
def generator_workflow_lists_generators_only(t):
    models = t.tmp / "models"
    write_config(models / "gm1", {"model_type": "sed_generator",
                                  "parameter_ranges": {"teff": [4000, 6000], "logg": [4, 4.5], "meta": [-0.5, 0]},
                                  "architecture": {"hidden_layers": [8], "output_dim": 1000}})
    write_config(models / "cm1", {"framework": "pytorch"})
    answer(t.mp, "4", echo=True)

    mg.run_interactive_workflow(str(t.tmp / "stellar"), str(models))

    out = t.capsys.readouterr().out
    assert "\n  gm1\n" in out and "cm1" not in out
    assert "    Teff: 4000 - 6000 K" in out
    assert "    Architecture: 3 → [8] → 1000" in out


@row
def generator_workflow_batch_writes_one_sed_per_grid_point(t):
    generator = fixed_generator(cpu_torch(t))
    stellar_dir = stellar_grids(t)
    generator.save(t.tmp / "models" / "gm1", hidden_layers=[16, 8], dropout=0.2)
    answer(t.mp, "3", "1", "5000,5500", "4.0", "0.0,-0.5", "", echo=True)

    mg.run_interactive_workflow(str(stellar_dir), str(t.tmp / "models"))

    batch = t.tmp / "models" / "gm1" / "SED" / "batch"
    texts = sorted(p.name for p in batch.glob("*.txt"))
    assert texts == ["sed_T5000_g4.00_m+0.00.txt", "sed_T5000_g4.00_m-0.50.txt",
                     "sed_T5500_g4.00_m+0.00.txt", "sed_T5500_g4.00_m-0.50.txt"]
    loaded = mg.SEDGenerator(t.tmp / "models" / "gm1")
    wl, flux = loaded.generate(5500.0, 4.0, -0.5, check_bounds=False)
    np.testing.assert_array_equal(np.loadtxt(batch / "sed_T5500_g4.00_m-0.50.txt"),
                                  np.loadtxt([f"{w:.6e} {f:.6e}" for w, f in zip(wl, flux)]))


@row
def completer_workflow_warns_about_unreadable_lookup_tables(t):
    stellar_dir = stellar_grids(t)
    (stellar_dir / "gridA" / "lookup_table.csv").write_text("")
    (stellar_dir / "gridC").mkdir()
    (stellar_dir / "gridC" / "lookup_table.csv").write_text("")
    write_config(t.tmp / "models" / "cm1", {"architecture": {}})
    t.mp.setattr(mc.SEDCompleter, "load", lambda self, path: setattr(self, "model", object()))

    answer(t.mp, "1", "x", echo=True)
    mc.run_interactive_workflow(str(stellar_dir), str(t.tmp / "models"))
    assert "   1) gridA                          (? SEDs)" in t.capsys.readouterr().out
    assert f"Could not count the SEDs in {stellar_dir / 'gridA' / 'lookup_table.csv'}" in t.caplog.text

    answer(t.mp, "2", "1", "x", echo=True)
    mc.run_interactive_workflow(str(stellar_dir), str(t.tmp / "models"))
    out = t.capsys.readouterr().out
    assert "gridC" not in out and "   1) gridB" in out
    assert "Not offering gridA: could not read" in t.caplog.text
    assert "Not offering gridC: could not read" in t.caplog.text


@row
def generator_workflow_warns_about_an_unreadable_lookup_table(t):
    stellar_dir = stellar_grids(t)
    (stellar_dir / "gridA" / "lookup_table.csv").write_text("")
    answer(t.mp, "1", "x", echo=True)

    mg.run_interactive_workflow(str(stellar_dir), str(t.tmp / "models"))

    assert "   1) gridA                          (? SEDs)" in t.capsys.readouterr().out
    assert f"Could not count the SEDs in {stellar_dir / 'gridA' / 'lookup_table.csv'}" in t.caplog.text


# ======================================================================
# Training data: what training refuses and warns about
# ======================================================================
@row
def completer_prepare_training_data_warns_how_many_spectra_it_dropped(t):
    flux = np.full((2, 1, 1, 200), 1e-10)
    flux[0, 0, 0, :150] = np.nan          # 4000 K: only 50 usable points
    lib = grid(t, [4000.0, 5000.0], flux)

    np.random.seed(0)
    _, _, params, _ = mc.SEDCompleter()._prepare_training_data(lib, 2)

    assert params[:, 0].tolist() == [5000.0]
    assert ("dropped 1 of 2 sampled spectra (1 with fewer than 100 positive, finite flux points)"
            in t.caplog.text)


@row
def generator_prepare_training_data_warns_how_many_spectra_it_dropped(t):
    flux = np.full((2, 1, 1, 200), 1e-10)
    flux[0, 0, 0, 5] = np.nan
    params, _, _ = mg.SEDGenerator()._prepare_training_data(grid(t, [4000.0, 5000.0], flux), 2)

    assert params[:, 0].tolist() == [5000.0]
    assert "dropped 1 of 2 sampled spectra (1 with non-positive or non-finite flux)" in t.caplog.text


@row
def prepare_training_data_raises_when_all_flux_invalid(t):
    with pytest.raises(ValueError, match="No valid training samples"):
        mg.SEDGenerator()._prepare_training_data(grid(t, [4000.0, 5000.0], 0.0))  # invalid: not > 0


@row
def completer_train_rejects_non_finite_grid_parameters(t):
    cpu_torch(t)
    with pytest.raises(ValueError, match="Non-finite Teff, logg or"):
        mc.SEDCompleter().train(grid(t, [4000.0, np.nan], 1e-10), str(t.tmp / "model"), epochs=1,
                                batch_size=2, max_samples=2, hidden_layers=[8])


@row
def generator_train_rejects_a_grid_with_no_flux_above_the_floor(t):
    cpu_torch(t)
    with pytest.raises(ValueError, match="No valid data points for normalization"):
        mg.SEDGenerator().train(grid(t, [4000.0, 5000.0], 1e-30), str(t.tmp / "model"), epochs=1,
                                batch_size=2, max_samples=2, hidden_layers=[8])
    assert not (t.tmp / "model").exists()


# ======================================================================
# ml.Completer.extend_catalog
# ======================================================================
@row
def extend_catalog_raises_without_a_lookup_table(t):
    # It used to extend every *.txt file with Teff=5000, logg=4.5, [M/H]=0.
    (t.tmp / "cat").mkdir()
    write_spectrum(t.tmp / "cat" / "a.txt", 3000, 20000)

    with pytest.raises(FileNotFoundError, match="lookup_table.csv"):
        stubbed_completer().extend_catalog(str(t.tmp / "cat"), str(t.tmp / "out"))


@row
def extend_catalog_reads_a_hash_prefixed_header(t):
    # sed_tools writes lookup tables with a '#file_name,...' header. It used
    # to be dropped (pd.read_csv(comment='#')) and the table refused; it now
    # gives the same result as a plain header.
    catalog = three_spectra(t)
    (catalog / "lookup_table.csv").write_text("#" + (catalog / "lookup_table.csv").read_text())
    completer = stubbed_completer()
    out_dir = t.tmp / "out"

    summary = completer.extend_catalog(str(catalog), str(out_dir), extension_range=(500.0, 60000.0))

    assert summary == {"extended": 2, "failed": 0, "skipped": 1, "total": 3, "output_dir": str(out_dir)}
    assert completer._completer.calls == [(5000.0, 4.0, 0.0), (6000.0, 4.5, -0.5)]
    assert completer._completer.ranges == [(500.0, 60000.0)] * 2
    assert (out_dir / "lookup_table.csv").read_text().splitlines()[0] == (
        "file_name,teff,logg,meta,extra,extended_by_ml,wl_min,wl_max")


@row
def extend_catalog_writes_the_good_spectra_then_lists_every_failure(t):
    import pandas as pd

    catalog = three_spectra(t)
    np.savetxt(catalog / "one_column.txt", np.arange(5.0))
    pd.DataFrame({
        "file_name": ["a.txt", "b.txt", "c.txt", "gone.txt", "one_column.txt", None, "a.txt", "a.txt", "a.txt"],
        "teff": [5000, 6000, 7000, 5000, 5000, 5000, "hot", -10, 5000],
        "logg": [4.0, 4.5, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 12.0],
        "meta": [0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }).to_csv(catalog / "lookup_table.csv", index=False)
    completer = stubbed_completer()
    out_dir = t.tmp / "out"

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


# ======================================================================
# Ranges: the completer keeps every original sample; a reversed range raises wherever it is given
# ======================================================================
@row
def extend_catalog_keeps_every_sample_of_the_kurucz_grid(t):
    # Extension used to resample every spectrum onto 1,000 points in 100 Å-1e5 Å:
    # a 147 Å-1.6e6 Å Kurucz spectrum lost its own samples and everything above 1e5 Å.
    torch = cpu_torch(t)
    library = kurucz(t)
    np.random.seed(0)
    torch.manual_seed(0)
    mc.SEDCompleter().train(str(library), str(t.tmp / "model"), epochs=1, batch_size=3, hidden_layers=[8])

    out = t.tmp / "extended"
    summary = Completer(str(t.tmp / "model")).extend_catalog(str(library), str(out), verbose=False)

    assert summary["extended"] == 12 and summary["total"] == 12
    spectra = sorted(library.glob("*.txt"))
    assert len(spectra) == 12
    for path in spectra:
        original = np.loadtxt(path)
        extended = np.loadtxt(out / path.name)
        assert_keeps_original(extended[:, 0], extended[:, 1], original[:, 0], original[:, 1])
        assert extended[0, 0] == pytest.approx(100.0)
        assert extended[-1, 0] == original[-1, 0] == 1.6e6


@row
def complete_sed_keeps_samples_outside_a_narrower_range(t):
    # The requested range ends inside the SED: nothing above it is cut.
    completer = fixed_completer(cpu_torch(t))
    wl = np.logspace(np.log10(3000.0), np.log10(2e5), 300)
    flux = mc.planck_function(wl, 5800.0)

    wl_out, flux_out = completer.complete_sed(wl, flux, 5800.0, 4.4, 0.0, extension_range=(500.0, 1e5))

    assert_keeps_original(wl_out, flux_out, wl, flux)
    assert wl_out[0] == pytest.approx(500.0) and wl_out[-1] == wl[-1] > 1e5
    assert np.all(np.diff(wl_out) > 0)


def extend_menu(t, *answers):
    """One grid and one completer model, the menu answered: the extension ranges extend_catalog was given."""
    base = t.tmp / "stellar"
    one_point_grid(base / "grid")
    (base / "grid" / "lookup_table.csv").write_text("file_name,teff\nx.txt,5000\n")
    write_config(t.tmp / "models" / "cm1", {"version": "2.0.0"})
    calls = loads_a_model(t)
    answer(t.mp, *answers)
    mc.run_interactive_workflow(str(base), str(t.tmp / "models"))
    return [call[2] for call in calls]


@row
def extend_menu_defaults_to_the_models_own_range(t):
    ranges = extend_menu(t, "2", "1", "1", "", "", "")
    assert ranges == [(100.0, 100000.0)]
    assert "the model covers 100-100000 Å" in t.capsys.readouterr().out


@row
def extend_menu_takes_the_range_typed(t):
    # The menu used to extend to 100 Å-1e5 Å whatever the grid or model.
    assert extend_menu(t, "2", "1", "1", "50", "3e6", "") == [(50.0, 3e6)]


@row
def a_reversed_range_raises_in_the_menus_and_the_api(t):
    # One rule (ui_utils.ordered_range): the extend menu asked again; complete_sed returned wavelengths out of
    # order; extend_catalog copied every spectrum as already covering the range; the axis-cut prompt swapped them.
    from sed_tools.ui_utils import ask_range

    wl = np.logspace(np.log10(3000.0), np.log10(20000.0), 50)
    with pytest.raises(ValueError, match="reversed"):
        mc.SEDCompleter().complete_sed(wl, mc.planck_function(wl, 5800.0), 5800.0, 4.4, 0.0,
                                       extension_range=(60000.0, 500.0))
    completer = stubbed_completer()
    with pytest.raises(ValueError, match="reversed"):
        completer.extend_catalog(str(three_spectra(t)), str(t.tmp / "out"), extension_range=(60000.0, 500.0))
    assert completer._completer.calls == [] and not (t.tmp / "out").exists()
    with pytest.raises(ValueError, match="reversed"):
        extend_menu(t, "2", "1", "1", "50", "40")  # asked once; extend_catalog is never reached
    answer(t.mp, "8000,3500")
    with pytest.raises(ValueError, match="reversed"):
        ask_range("Teff range: ")
    answer(t.mp, "5000,5000")
    assert ask_range("Teff range: ") == (5000.0, 5000.0)  # one value is a range, not a reversed one


@row
def completer_train_menu_takes_the_numbers_typed(t):
    base = t.tmp / "stellar"
    one_point_grid(base / "grid")
    calls = []
    t.mp.setattr(mc.SEDCompleter, "train", lambda self, **kw: calls.append(kw) or {})
    answer(t.mp, "1", "1", "7", "", "", "")
    mc.run_interactive_workflow(str(base), str(t.tmp / "models"))
    assert [(c["epochs"], c["batch_size"]) for c in calls] == [(7, 32)]


@row
def completer_list_models_leaves_out_generators_so_choosing_cannot_raise(t):
    # A generator listed by the completer menu raised KeyError when chosen.
    write_config(t.tmp / "gen", {"model_type": "sed_generator"})
    assert mc.SEDCompleter.list_models(str(t.tmp)) == []
    assert [g["name"] for g in mg.SEDGenerator.list_models(str(t.tmp))] == ["gen"]


@row
def generator_number_prompts_take_the_numbers_typed(t):
    answer(t.mp, "")
    assert mg._ask_number("Epochs [200]: ", 200, whole=True) == 200
    answer(t.mp, "-0.5")
    assert mg._ask_number("[M/H]: ") == -0.5
    answer(t.mp, "5000, 5500")
    assert mg._ask_numbers("Teff values: ") == [5000.0, 5500.0]


@row
def generator_generate_menu_takes_the_parameters_typed(t):
    calls = []
    t.mp.setattr(mg.SEDGenerator, "list_models",
                 staticmethod(lambda models_dir: [{"name": "g", "path": str(t.tmp / "g")}]))
    t.mp.setattr(mg.SEDGenerator, "load", lambda self, path: None)
    t.mp.setattr(mg.SEDGenerator, "generate_with_outputs",
                 lambda self, *args: calls.append(args) or (np.array([1.0, 2.0]), np.array([1.0, 1.0])))
    answer(t.mp, "2", "1", "5800", "4.4", "0", str(t.tmp / "out"))
    mg.run_interactive_workflow(str(t.tmp), str(t.tmp))
    assert calls == [(5800.0, 4.4, 0.0, str(t.tmp / "out"))]


@row
def window_coverage_warns_below_the_80_percent_training_required(t):
    from sed_tools.ml.completer import _warn_window_coverage

    window = np.arange(10.0)
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.ml.completer"):
        _warn_window_coverage(window, np.array([0.0, 7.0]))  # 8 of 10 points: as training required
        assert not t.caplog.records
        _warn_window_coverage(window, np.array([0.0, 6.0]))
    assert "covers 70% of the network's input window (0-9 Å); the other 30% is filled with 1e-20" in t.caplog.text
    assert "Training required at least 80% coverage" in t.caplog.text


@row
def complete_sed_warns_when_the_input_misses_most_of_the_window(t):
    completer = window_completer(pytest.importorskip("torch"))
    window = completer.wavelength_grid[300:700]
    narrow = np.logspace(np.log10(3000.0), np.log10(20000.0), 300)
    covered = np.mean((window >= narrow[0]) & (window <= narrow[-1]))
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.ml.completer"):
        completer.complete_sed(narrow, mc.planck_function(narrow, 5800.0), 5800.0, 4.4, 0.1,
                               extension_range=(500.0, 60000.0))
    assert f"covers {100 * covered:.0f}% of the network's input window" in t.caplog.text
    assert f"the other {100 * (1 - covered):.0f}% is filled with 1e-20" in t.caplog.text

    t.caplog.clear()
    wide = np.logspace(np.log10(500.0), np.log10(20000.0), 300)
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.ml.completer"):
        completer.complete_sed(wide, mc.planck_function(wide, 5800.0), 5800.0, 4.4, 0.1,
                               extension_range=(300.0, 60000.0))
    assert "input window" not in t.caplog.text


def test_ml(rows):
    rows(LIST_MODELS_WARNS, list_models_warns_naming_the_unreadable_config)
    rows(LAST_BATCH_OF_ONE, training_drops_a_last_batch_of_one_spectrum)
    rows(ALL_NAN, train_raises_before_saving_when_every_batch_is_nan)
    rows(ENDING_ANSWERS, workflow_ends_on_the_answer)
    rows(MISSING_COLUMN, extend_catalog_raises_when_a_column_is_missing)
    rows(NOT_EXTENDED, extend_catalog_raises_what_the_model_could_not_extend)
    rows(SCENARIOS)
