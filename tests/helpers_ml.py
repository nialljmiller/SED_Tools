"""Fixtures shared by the ML tests: CPU-only, seeded torch, a tiny training library, and fixed
completer and generator networks.
"""

import numpy as np
import pytest

from sed_tools import ml_sed_completer as mc, ml_sed_generator as mg


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
