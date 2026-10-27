"""Tests for the ml fixes: the device on Apple silicon, extend_catalog on
'#'-headed lookup tables, and ``ml_generator --auto``.

Tests that need torch (or optuna) skip without it; the rest are torch-free.
"""

import numpy as np
import pytest

from sed_tools import ml_sed_completer as mc
from sed_tools import ml_sed_generator as mg


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


# ---------------------------------------------------------------------
# Apple silicon: the network and every tensor on get_device()
# ---------------------------------------------------------------------

def _mps_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.backends.mps.is_available()


@pytest.mark.skipif(not _mps_available(), reason="needs torch with MPS (Apple silicon)")
def test_train_complete_and_generate_run_on_mps(tiny_library, tmp_path, monkeypatch):
    # Every train, complete_sed and generate used to raise "Tensor for
    # argument weight is on cpu but expected on mps" here.
    pytest.importorskip("sklearn")
    import matplotlib
    matplotlib.use("Agg")
    # The loss-curve plots use no torch, and after one epoch take seconds to save
    monkeypatch.setattr(mc.SEDCompleter, "_plot_training_results", lambda *args: None)
    monkeypatch.setattr(mg.SEDGenerator, "_plot_training", lambda *args: None)

    completer = mc.SEDCompleter()
    completer.train(str(tiny_library), str(tmp_path / "completer"), epochs=1,
                    batch_size=3, max_samples=12, hidden_layers=[8])
    assert completer.device.type == "mps"
    assert {p.device.type for p in completer.model.parameters()} == {"mps"}

    wl = np.logspace(np.log10(3000.0), np.log10(20000.0), 300)
    sed = mc.planck_function(wl, 5000.0)
    for model in (completer, mc.SEDCompleter(tmp_path / "completer")):
        wl_out, flux_out = model.complete_sed(wl, sed, teff=5000.0, logg=4.2, meta=-0.2,
                                              extension_range=(500.0, 60000.0))
        assert np.all(np.isfinite(flux_out)) and len(flux_out) == len(wl_out)

    generator = mg.SEDGenerator()
    generator.train(str(tiny_library), str(tmp_path / "generator"), epochs=1,
                    batch_size=3, max_samples=12, hidden_layers=[8])
    assert generator.device.type == "mps"
    assert {p.device.type for p in generator.model.parameters()} == {"mps"}

    for model in (generator, mg.SEDGenerator(tmp_path / "generator")):
        wl_out, flux_out = model.generate(5000.0, 4.2, -0.2)
        assert np.all(np.isfinite(flux_out)) and len(flux_out) == len(wl_out) == 1000


# ---------------------------------------------------------------------
# extend_catalog on the lookup tables sed_tools writes ('#file_name,...')
# ---------------------------------------------------------------------

class _StubCompleter:
    """Stands in for SEDCompleter: records each call, extends to a fixed grid."""

    def __init__(self):
        self.model = object()
        self.calls = []

    def complete_sed(self, wavelength, flux, teff, logg, meta, extension_range):
        self.calls.append((teff, logg, meta))
        grid = np.logspace(np.log10(extension_range[0]), np.log10(extension_range[1]), 20)
        return grid, np.interp(grid, wavelength, flux)

    def plot_inference_examples(self, examples, output_dir):
        pass


def _write_spectrum(path, lo, hi):
    wl = np.logspace(np.log10(lo), np.log10(hi), 50)
    np.savetxt(path, np.column_stack([wl, 1e-12 * (wl / 5000.0) ** -2]))


def test_extend_catalog_extends_each_spectrum_with_its_own_parameters(tmp_path):
    # pd.read_csv(comment='#') used to drop the '#file_name,...' header, so the
    # first spectrum became the header and every other one was extended as
    # Teff 5000, logg 4.5, [M/H] 0; on cleanup/base it raised instead.
    import pandas as pd
    from sed_tools import ml
    from sed_tools._lookup_io import write_lookup_csv

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    names = ["a.txt", "b.txt", "c.txt"]
    for name, (lo, hi) in zip(names, [(3000, 20000), (2000, 30000), (4000, 9000)]):
        _write_spectrum(catalog / name, lo, hi)
    write_lookup_csv({"file_name": names, "teff": [4500.0, 5800.0, 7200.0],
                      "logg": [4.0, 4.4, 3.5], "metallicity": [-1.0, 0.0, 0.3]},
                     str(catalog / "lookup_table.csv"))
    assert (catalog / "lookup_table.csv").read_text().startswith("#file_name,teff,logg,metallicity\n")
    completer = ml.Completer()
    completer._completer = _StubCompleter()
    out_dir = tmp_path / "out"

    summary = completer.extend_catalog(str(catalog), str(out_dir), extension_range=(500.0, 60000.0),
                                       verbose=False)

    assert summary == {"extended": 3, "failed": 0, "skipped": 0, "total": 3, "output_dir": str(out_dir)}
    assert completer._completer.calls == [(4500.0, 4.0, -1.0), (5800.0, 4.4, 0.0), (7200.0, 3.5, 0.3)]
    written = pd.read_csv(out_dir / "lookup_table.csv")
    assert written["file_name"].tolist() == names
    assert written[["teff", "logg", "metallicity"]].values.tolist() == [
        [4500.0, 4.0, -1.0], [5800.0, 4.4, 0.0], [7200.0, 3.5, 0.3]]
    assert sorted(p.name for p in out_dir.glob("*.txt")) == names


def _feed_input(monkeypatch, answers):
    feed = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(feed))


def test_menus_count_every_spectrum_of_a_hash_headed_lookup(tmp_path, cube_writer, monkeypatch, capsys):
    # The menus read the lookup with comment='#' too, so they showed one SED fewer.
    from sed_tools._lookup_io import write_lookup_csv

    base = tmp_path / "stellar"
    for name in ("hashed", "plain"):
        cube_writer(base / name / "flux_cube.bin", [5000.0], [4.0], [0.0], [1000.0, 2000.0],
                    np.ones((1, 1, 1, 2)))
    write_lookup_csv({"file_name": ["a.txt", "b.txt", "c.txt"], "teff": [5000, 5500, 6000]},
                     str(base / "hashed" / "lookup_table.csv"))
    (base / "plain" / "lookup_table.csv").write_text("file_name,teff\na.txt,5000\nb.txt,5500\n")
    models = tmp_path / "models"
    (models / "cm1").mkdir(parents=True)
    (models / "cm1" / "config.json").write_text("{}")
    monkeypatch.setattr(mc.SEDCompleter, "load", lambda self, path: setattr(self, "model", object()))

    # Completer training menu, completer extend menu, generator training menu
    for module, answers in [(mc, ["1", "x"]), (mc, ["2", "1", "x"]), (mg, ["1", "x"])]:
        _feed_input(monkeypatch, answers)
        module.run_interactive_workflow(str(base), str(models))
        out = capsys.readouterr().out
        assert f"{'hashed':30s} (3 SEDs)" in out
        assert f"{'plain':30s} (2 SEDs)" in out
