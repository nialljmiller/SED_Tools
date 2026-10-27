"""Tests for Nill's decisions of 27 October 2026: packaging (psutil, scipy) and the
densifier's wavelength blocks."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from sed_tools._constants import planck_flam
from sed_tools._resample import resample_to_grid
from sed_tools.flux_cube_tool import FluxCube

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Packaging: psutil is required; scipy (the [full] extra) is not needed to import
# ---------------------------------------------------------------------

def test_psutil_is_a_required_dependency_and_scipy_is_not():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert any(dep.startswith("psutil>=") for dep in project["dependencies"])
    assert not any(dep.startswith("scipy") for dep in project["dependencies"])
    assert any(dep.startswith("scipy") for dep in project["optional-dependencies"]["full"])


_WITHOUT_SCIPY = """
import sys
sys.modules["scipy"] = None  # every import of scipy or a scipy submodule now raises ImportError

import numpy as np
import sed_tools
import sed_tools.combine_stellar_atm as combine

assert not [m for m, mod in sys.modules.items() if m.startswith("scipy") and mod is not None]
try:
    combine._fill_gaps(np.zeros((1, 1, 1, 1)), np.ones((1, 1, 1), bool), np.zeros((1, 1, 1), int),
                       np.ones(1), np.ones(1), np.ones(1))
except ImportError as exc:
    print("needs scipy:", exc)
"""


def test_import_sed_tools_works_without_scipy():
    done = subprocess.run([sys.executable, "-c", _WITHOUT_SCIPY], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    # The grid combination itself still needs scipy, and says so when it is called.
    assert done.stdout.startswith("needs scipy: ") and "scipy" in done.stdout[len("needs scipy: "):]


# ---------------------------------------------------------------------
# Densifier: written one block of wavelengths at a time, same bytes
# ---------------------------------------------------------------------

def _uneven_source_cube(path, cube_writer):
    """3 Teff x 3 logg x 2 [M/H] x 30 wavelengths, a different spectrum at every node."""
    teff, logg, meta = [4000.0, 6000.0, 9000.0], [3.0, 4.0, 5.0], [-1.0, 0.0]
    wl = np.geomspace(1000.0, 50000.0, 30)
    rng = np.random.default_rng(3)
    flux = np.empty((3, 3, 2, 30))
    for i_t, t in enumerate(teff):
        flux[i_t] = planck_flam(wl, t) * rng.uniform(0.5, 1.5, size=(3, 2, 30))
    cube_writer(path, teff, logg, meta, wl, flux)


def _densify(monkeypatch, tmp_path, name, ram, **kwargs):
    """Densify src.bin into <name>.bin as if `ram` bytes of RAM were available; return its bytes."""
    import sed_tools.grid_densifier as gd

    monkeypatch.setattr(gd, "_available_ram_bytes", lambda: ram)
    gd.densify_grid(str(tmp_path / "src.bin"), str(tmp_path / f"{name}.bin"), teff_spacing=1000.0, **kwargs)
    return (tmp_path / f"{name}.bin").read_bytes()


# 6 dense Teffs: each wavelength holds (3 + 6) x 3 x 2 doubles = 432 bytes, of which a
# block may use half the RAM, so 6048 bytes gives blocks of 7 wavelengths: 7, 7, 7, 7, 2.
SEVEN_WAVELENGTHS = 2 * 7 * 432


def test_densified_cube_does_not_depend_on_the_block_size(tmp_path, cube_writer, monkeypatch, capsys):
    _uneven_source_cube(tmp_path / "src.bin", cube_writer)

    one_block = _densify(monkeypatch, tmp_path, "one", 2**40)
    assert "in 1 wavelength block(s)" in capsys.readouterr().out
    assert _densify(monkeypatch, tmp_path, "five", SEVEN_WAVELENGTHS) == one_block
    assert "in 5 wavelength block(s)" in capsys.readouterr().out
    assert _densify(monkeypatch, tmp_path, "thirty", 1) == one_block
    assert "in 30 wavelength block(s)" in capsys.readouterr().out

    source = FluxCube.from_file(str(tmp_path / "src.bin"))
    dense = FluxCube.from_file(str(tmp_path / "five.bin"))
    np.testing.assert_array_equal(dense.teff_grid, [4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0])
    np.testing.assert_array_equal(dense.flux[[0, 2, 5]], source.flux)


def test_densified_ml_nodes_do_not_depend_on_the_block_size(tmp_path, cube_writer, monkeypatch):
    import types

    model_wl = np.geomspace(900.0, 60000.0, 200)

    class PowerLawGenerator:
        def __init__(self, path):
            pass

        def generate(self, teff, logg, meta, check_bounds=True):
            return model_wl.copy(), 1e-3 * teff * (model_wl / 5000.0) ** (-2.0 - 0.1 * logg + meta)

    fake = types.ModuleType("sed_tools.ml_sed_generator")
    fake.SEDGenerator = PowerLawGenerator
    monkeypatch.setitem(sys.modules, "sed_tools.ml_sed_generator", fake)
    _uneven_source_cube(tmp_path / "src.bin", cube_writer)

    one_block = _densify(monkeypatch, tmp_path, "one", 2**40, ml_model="fake")
    assert _densify(monkeypatch, tmp_path, "five", SEVEN_WAVELENGTHS, ml_model="fake") == one_block

    dense = FluxCube.from_file(str(tmp_path / "five.bin"))
    for i_t, t in ((1, 5000.0), (3, 7000.0), (4, 8000.0)):
        for i_l, logg in enumerate(dense.logg_grid):
            for i_m, meta in enumerate(dense.meta_grid):
                expected = resample_to_grid(*PowerLawGenerator(None).generate(t, logg, meta), t, dense.wavelengths)
                np.testing.assert_array_equal(dense.flux[i_t, i_l, i_m], expected)
