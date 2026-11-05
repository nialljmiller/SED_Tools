"""ML ranges and menus: extend_catalog and complete_sed keep the samples outside the
model's range, the extend, train and generate menus ask again after a bad answer, and the warnings when
an input misses most of the window.
"""

import logging
import shutil

import numpy as np
import pytest

from sed_tools import ml_sed_completer as mc

from helpers_api import no_network  # noqa: F401 (an autouse fixture)
from helpers_cube import GOLDEN_MODELS, _feed_input


@pytest.fixture
def cpu_torch(monkeypatch):
    """torch pinned to the CPU through ml_utils.get_device; matplotlib on Agg."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    import matplotlib
    matplotlib.use("Agg")
    from sed_tools import ml_utils

    monkeypatch.setattr(ml_utils, "get_device", lambda: torch.device("cpu"))
    return torch


# ---------------------------------------------------------------------
# The completer keeps every original sample when it extends a spectrum
# ---------------------------------------------------------------------
def _assert_keeps_original(wl_out, flux_out, wl, flux):
    """The output is the input verbatim with points added only beyond its ends."""
    start = int(np.searchsorted(wl_out, wl[0]))
    np.testing.assert_array_equal(wl_out[start:start + len(wl)], wl)
    np.testing.assert_array_equal(flux_out[start:start + len(wl)], flux)
    assert np.all(wl_out[:start] < wl[0])
    assert np.all(wl_out[start + len(wl):] > wl[-1])


def test_extend_catalog_keeps_every_sample_of_the_kurucz_grid(cpu_torch, tmp_path):
    # Extension used to resample every spectrum onto 1,000 points in 100 Å-1e5 Å:
    # a 147 Å-1.6e6 Å Kurucz spectrum lost its own samples and everything above 1e5 Å.
    from sed_tools.precompute_flux_cube import precompute_flux_cube

    library = tmp_path / "Kurucz2003all"
    shutil.copytree(GOLDEN_MODELS / "Kurucz2003all", library)
    precompute_flux_cube(str(library), str(library / "flux_cube.bin"))
    np.random.seed(0)
    cpu_torch.manual_seed(0)
    model = mc.train_model(str(library), str(tmp_path / "model"), epochs=1, batch_size=3,
                           hidden_layers=[8])

    out = tmp_path / "extended"
    summary = mc.extend_catalog(model, str(library), str(out))

    assert summary["extended"] == 12 and summary["total"] == 12
    spectra = sorted(library.glob("*.txt"))
    assert len(spectra) == 12
    for path in spectra:
        original = np.loadtxt(path)
        extended = np.loadtxt(out / path.name)
        _assert_keeps_original(extended[:, 0], extended[:, 1], original[:, 0], original[:, 1])
        assert extended[0, 0] == pytest.approx(100.0)
        assert extended[-1, 0] == original[-1, 0] == 1.6e6


def test_complete_sed_keeps_samples_outside_a_narrower_range(cpu_torch):
    # The requested range ends inside the SED: nothing above it is cut.
    torch = cpu_torch
    torch.manual_seed(0)
    completer = mc.SEDCompleter()
    completer.model = mc.SEDCompleterNetwork(input_dim=400, output_dim=1000, hidden_layers=[8], dropout=0.3)
    completer.wavelength_grid = completer._create_wavelength_grid()
    completer._known_start, completer._known_end = 300, 700
    completer.scaler_params = {
        "teff_mean": 5500.0, "teff_std": 1500.0, "logg_mean": 4.0, "logg_std": 0.5,
        "meta_mean": 0.0, "meta_std": 0.5, "flux_log_mean": 6.5, "flux_log_std": 1.0,
        "ext_log_mean": 6.0, "ext_log_std": 1.5,
    }
    wl = np.logspace(np.log10(3000.0), np.log10(2e5), 300)
    flux = mc.planck_function(wl, 5800.0)

    wl_out, flux_out = completer.complete_sed(wl, flux, 5800.0, 4.4, 0.0, extension_range=(500.0, 1e5))

    _assert_keeps_original(wl_out, flux_out, wl, flux)
    assert wl_out[0] == pytest.approx(500.0) and wl_out[-1] == wl[-1] > 1e5
    assert np.all(np.diff(wl_out) > 0)


@pytest.fixture
def extend_menu(tmp_path, cube_writer, monkeypatch):
    """One grid and one completer model; returns the extension ranges extend_catalog is called with."""
    import json

    from sed_tools import ml

    base = tmp_path / "stellar"
    cube_writer(base / "grid" / "flux_cube.bin", [5000.0], [4.0], [0.0], [1000.0, 2000.0],
                np.ones((1, 1, 1, 2)))
    (base / "grid" / "lookup_table.csv").write_text("file_name,teff\nx.txt,5000\n")
    models = tmp_path / "models"
    (models / "cm1").mkdir(parents=True)
    (models / "cm1" / "config.json").write_text(json.dumps({"version": "2.0.0"}))

    def fake_load(self, path):
        self.model = object()
        self.wavelength_grid = np.logspace(2, 5, 1000)

    ranges = []

    def fake_extend(self, input_dir, output_dir, extension_range, verbose):
        ranges.append(extension_range)
        return {"extended": 1, "failed": 0, "skipped": 0, "total": 1, "output_dir": output_dir}

    monkeypatch.setattr(mc.SEDCompleter, "load", fake_load)
    monkeypatch.setattr(ml.Completer, "extend_catalog", fake_extend)
    return base, models, ranges


def test_extend_menu_defaults_to_the_models_own_range(extend_menu, monkeypatch, capsys):
    base, models, ranges = extend_menu
    _feed_input(monkeypatch, ["2", "1", "1", "", "", ""])
    mc.run_interactive_workflow(str(base), str(models))
    assert ranges == [(100.0, 100000.0)]
    assert "the model covers 100-100000 Å" in capsys.readouterr().out


def test_extend_menu_asks_for_the_range_again_on_a_bad_answer(extend_menu, monkeypatch, capsys):
    # The menu used to extend to 100 Å-1e5 Å whatever the grid or model.
    base, models, ranges = extend_menu
    _feed_input(monkeypatch, ["2", "1", "1", "abc", "50", "40", "50", "3e6", ""])
    mc.run_interactive_workflow(str(base), str(models))
    out = capsys.readouterr().out
    assert ranges == [(50.0, 3e6)]
    assert "'abc' is not a positive number; try again." in out
    assert "50 Å is not below 40 Å; try again." in out


def test_completer_train_menu_asks_again_for_a_bad_number(tmp_path, cube_writer, monkeypatch, capsys):
    # "abc" at Epochs used to end the menu with "Invalid parameters".
    base = tmp_path / "stellar"
    cube_writer(base / "grid" / "flux_cube.bin", [5000.0], [4.0], [0.0], [1000.0, 2000.0],
                np.ones((1, 1, 1, 2)))
    calls = []
    monkeypatch.setattr(mc.SEDCompleter, "train", lambda self, **kw: calls.append(kw) or {})
    _feed_input(monkeypatch, ["1", "1", "abc", "0", "7", "", "", ""])
    mc.run_interactive_workflow(str(base), str(tmp_path / "models"))
    out = capsys.readouterr().out
    assert [(c["epochs"], c["batch_size"]) for c in calls] == [(7, 32)]
    assert "'abc' is not a positive whole number; try again." in out
    assert "'0' is not a positive whole number; try again." in out


def test_completer_list_models_leaves_out_generators_so_choosing_cannot_raise(tmp_path):
    # A generator listed by the completer menu raised KeyError when chosen.
    import json

    from sed_tools import ml_sed_generator as mg

    generator = tmp_path / "gen"
    generator.mkdir()
    (generator / "config.json").write_text(json.dumps({"model_type": "sed_generator"}))
    assert mc.SEDCompleter.list_models(str(tmp_path)) == []
    assert [g["name"] for g in mg.SEDGenerator.list_models(str(tmp_path))] == ["gen"]


# ---------------------------------------------------------------------
# Number prompts in the generator and densifier menus ask again
# ---------------------------------------------------------------------
def test_generator_number_prompts_ask_again(monkeypatch, capsys):
    # int(input()) / float(input()) used to raise ValueError and end the menu.
    from sed_tools import ml_sed_generator as mg

    _feed_input(monkeypatch, ["x", "1.5", "-3", ""])
    assert mg._ask_number("Epochs [200]: ", 200, whole=True, positive=True) == 200
    _feed_input(monkeypatch, ["nan", "", "-0.5"])
    assert mg._ask_number("[M/H]: ") == -0.5
    _feed_input(monkeypatch, ["5000,x", "", "5000, 5500"])
    assert mg._ask_numbers("Teff values: ") == [5000.0, 5500.0]
    out = capsys.readouterr().out
    for bad in ("'x'", "'1.5'", "'-3'"):
        assert f"{bad} is not a positive whole number; try again." in out
    assert "'nan' is not a number; try again." in out and "Please enter a number." in out
    assert "'5000,x' is not a comma-separated list of numbers; try again." in out


def test_generator_generate_menu_asks_again_for_a_bad_teff(tmp_path, monkeypatch, capsys):
    from sed_tools import ml_sed_generator as mg

    calls = []
    monkeypatch.setattr(mg.SEDGenerator, "list_models",
                        staticmethod(lambda models_dir: [{"name": "g", "path": str(tmp_path / "g")}]))
    monkeypatch.setattr(mg.SEDGenerator, "load", lambda self, path: None)
    monkeypatch.setattr(mg.SEDGenerator, "generate_with_outputs",
                        lambda self, *args: calls.append(args) or (np.array([1.0, 2.0]), np.array([1.0, 1.0])))
    _feed_input(monkeypatch, ["2", "1", "warm", "5800", "4.4", "0", str(tmp_path / "out")])
    mg.run_interactive_workflow(str(tmp_path), str(tmp_path))
    assert calls == [(5800.0, 4.4, 0.0, str(tmp_path / "out"))]
    assert "'warm' is not a positive number; try again." in capsys.readouterr().out


# ----------------------------------------------------------------------
# ML completer: an input that covers too little of the network's window
# ----------------------------------------------------------------------
def test_window_coverage_warns_below_the_80_percent_training_required(caplog):
    from sed_tools.ml_sed_completer import _warn_window_coverage

    window = np.arange(10.0)
    with caplog.at_level(logging.WARNING, logger="sed_tools.ml_sed_completer"):
        _warn_window_coverage(window, np.array([0.0, 7.0]))  # 8 of 10 points: as training required
        assert not caplog.records
        _warn_window_coverage(window, np.array([0.0, 6.0]))
    assert "covers 70% of the network's input window (0-9 Å); the other 30% is filled with 1e-20" in caplog.text
    assert "Training required at least 80% coverage" in caplog.text


def completer_with_a_constant_network(torch):
    from sed_tools import ml_sed_completer as mc

    class Constant:
        def eval(self):
            return self

        def __call__(self, x):
            return torch.zeros((1, 1000))

    completer = mc.SEDCompleter()
    completer.model = Constant()
    completer._device = torch.device("cpu")
    completer.wavelength_grid = completer._create_wavelength_grid()
    completer._known_start, completer._known_end = 300, 700
    completer.scaler_params = {
        "teff_mean": 5500.0, "teff_std": 1500.0, "logg_mean": 4.0, "logg_std": 0.5,
        "meta_mean": 0.0, "meta_std": 0.5, "flux_log_mean": 6.5, "flux_log_std": 1.0,
        "ext_log_mean": 6.0, "ext_log_std": 1.5,
    }
    return completer


def test_complete_sed_warns_when_the_input_misses_most_of_the_window(caplog):
    torch = pytest.importorskip("torch")
    from sed_tools.ml_sed_completer import planck_function

    completer = completer_with_a_constant_network(torch)
    window = completer.wavelength_grid[300:700]
    narrow = np.logspace(np.log10(3000.0), np.log10(20000.0), 300)
    covered = np.mean((window >= narrow[0]) & (window <= narrow[-1]))
    with caplog.at_level(logging.WARNING, logger="sed_tools.ml_sed_completer"):
        completer.complete_sed(narrow, planck_function(narrow, 5800.0), 5800.0, 4.4, 0.1,
                               extension_range=(500.0, 60000.0))
    assert f"covers {100 * covered:.0f}% of the network's input window" in caplog.text
    assert f"the other {100 * (1 - covered):.0f}% is filled with 1e-20" in caplog.text

    caplog.clear()
    wide = np.logspace(np.log10(500.0), np.log10(20000.0), 300)
    with caplog.at_level(logging.WARNING, logger="sed_tools.ml_sed_completer"):
        completer.complete_sed(wide, planck_function(wide, 5800.0), 5800.0, 4.4, 0.1,
                               extension_range=(300.0, 60000.0))
    assert "input window" not in caplog.text
