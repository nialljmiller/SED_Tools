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
