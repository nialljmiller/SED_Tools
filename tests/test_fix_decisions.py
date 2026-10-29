"""Tests for Nill's decisions of 27 October 2026: packaging (no psutil; scipy), the
densifier's wavelength blocks, stopping on an undeclared flux unit, and magnitudes of
zero flux."""

import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from sed_tools._constants import planck_flam
from sed_tools._resample import resample_to_grid
from sed_tools.flux_cube_tool import FluxCube

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Packaging: psutil is not a dependency; scipy (the [full] extra) is not needed to import
# ---------------------------------------------------------------------

def test_psutil_is_not_a_dependency_and_scipy_is_not_required():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    assert not any(dep.startswith("psutil") for dep in project["dependencies"])
    assert not any(dep.startswith("psutil") for deps in extras.values() for dep in deps)
    assert not any(dep.startswith("scipy") for dep in project["dependencies"])
    assert any(dep.startswith("scipy") for dep in extras["full"])


_WITHOUT_PSUTIL = """
import sys
sys.modules["psutil"] = None  # import psutil now raises ImportError

import sed_tools
from sed_tools._constants import available_memory_bytes

assert not [m for m, mod in sys.modules.items() if m.startswith("psutil") and mod is not None]
print(available_memory_bytes())
"""


def test_import_sed_tools_and_measure_ram_without_psutil():
    done = subprocess.run([sys.executable, "-c", _WITHOUT_PSUTIL], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    # On Linux and macOS the RAM is measured without psutil, so nothing is assumed or warned.
    assert "Could not measure available RAM" not in done.stderr
    assert int(done.stdout) > 0


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


# ---------------------------------------------------------------------
# Undeclared flux units: stop and ask, offering the detector's best guess
# ---------------------------------------------------------------------

def _surface_flux_grid(directory, flux_header=None, teffs=(3500.0, 3600.0, 3700.0)):
    """Spectra of pi*B_lambda, a surface flux in erg/s/cm2/A, with or without a flux-unit header."""
    directory.mkdir(parents=True, exist_ok=True)
    wl = np.linspace(3000.0, 10000.0, 200)
    for i, teff in enumerate(teffs):
        lines = [f"# teff = {teff}\n", "# logg = 4.5\n", "# metallicity = 0.0\n", "# wavelength_unit = Angstrom\n"]
        if flux_header:
            lines.append(f"# flux_unit = {flux_header}\n")
        lines += [f"{w:.6f} {f:.8e}\n" for w, f in zip(wl, planck_flam(wl, teff))]
        (directory / f"s{i}.txt").write_text("".join(lines))
    return directory


def _contents(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.txt"))}


def test_clean_model_dir_stops_on_an_undeclared_flux_unit_naming_the_best_guess(tmp_path):
    from sed_tools.spectra_cleaner import clean_model_dir, undeclared_flux_unit

    grid = _surface_flux_grid(tmp_path / "Cool")
    before = _contents(grid)
    # The range detector reads these surface fluxes (1e5-1e6) as Jansky at Earth.
    assert undeclared_flux_unit(sorted(str(p) for p in grid.glob("*.txt")), 1.0) == "fnu_jy"

    with pytest.raises(ValueError) as err:
        clean_model_dir(str(grid), sample_fraction=1.0)
    message = str(err.value)
    assert f"Flux unit of {grid} is not declared; best guess: 'fnu_jy', Jansky" in message
    assert "Pass the unit as flux_unit='fnu_jy', or as another of flam, fnu, fnu_jy, flam_si, flam_si_nm, flam_si_um, normalized" in message
    assert _contents(grid) == before
    assert not (grid / "lookup_table.csv").exists()


def test_clean_model_dir_uses_the_flux_unit_it_is_given(tmp_path):
    from sed_tools.spectra_cleaner import clean_model_dir
    from sed_tools.spectrum_io import read_text_spectrum

    grid = _surface_flux_grid(tmp_path / "Cool")
    summary = clean_model_dir(str(grid), sample_fraction=1.0, flux_unit="flam")

    assert len(summary["converted"]) == 3 and summary["catalog_units"]["flux"] == "flam"
    wl, flux = read_text_spectrum(str(grid / "s0.txt"))
    np.testing.assert_allclose(flux, planck_flam(wl, 3500.0), rtol=1e-8)
    assert "# original_flux_unit = flam" in (grid / "s0.txt").read_text()


def test_declared_flux_units_clean_as_before(tmp_path):
    # A declared unit is used without asking, and gives the same files as passing it.
    from sed_tools.spectra_cleaner import clean_model_dir

    declared = _surface_flux_grid(tmp_path / "Declared", flux_header="erg/cm2/s/A")
    passed = _surface_flux_grid(tmp_path / "Passed")
    clean_model_dir(str(declared), sample_fraction=1.0)
    clean_model_dir(str(passed), sample_fraction=1.0, flux_unit="flam")
    assert _contents(declared) == _contents(passed)


def test_an_unknown_flux_unit_name_is_refused_before_anything_changes(tmp_path):
    from sed_tools.spectra_cleaner import clean_model_dir

    grid = _surface_flux_grid(tmp_path / "Cool")
    before = _contents(grid)
    with pytest.raises(ValueError, match=r"flux_unit='Jy' is not one of flam, fnu, fnu_jy, flam_si, flam_si_nm, flam_si_um, normalized"):
        clean_model_dir(str(grid), flux_unit="Jy")
    assert _contents(grid) == before


def test_a_spectrum_cleaned_on_its_own_stops_on_an_undeclared_flux_unit(tmp_path):
    from sed_tools.spectra_cleaner import clean_spectrum_file

    grid = _surface_flux_grid(tmp_path / "Cool")
    before = (grid / "s0.txt").read_bytes()
    with pytest.raises(ValueError, match=r"Flux unit of this spectrum is not declared; best guess: 'fnu_jy'"):
        clean_spectrum_file(str(grid / "s0.txt"))
    assert (grid / "s0.txt").read_bytes() == before


def test_files_cleaned_one_by_one_without_a_declared_flux_unit_are_listed(tmp_path, monkeypatch):
    # The sample held only a standardized file, so there are no grid-wide units.
    import sed_tools.spectra_cleaner as sc

    grid = _surface_flux_grid(tmp_path / "Cool")
    done = grid / "done.txt"
    done.write_text("# units_standardized = True\n4000 1.0\n5000 2.0\n")
    monkeypatch.setattr(sc.random, "sample", lambda files, n: [str(done)])

    with pytest.raises(RuntimeError, match=r"(?s)3 of 4 spectra .*Flux unit of this spectrum is not declared; "
                                           r"best guess: 'fnu_jy'.* \(3\): s0.txt, s1.txt, s2.txt"):
        sc.clean_model_dir(str(grid), sample_fraction=0.1)
    summary = sc.clean_model_dir(str(grid), sample_fraction=0.1, flux_unit="flam")
    assert summary["converted"] == ["s0.txt", "s1.txt", "s2.txt"]


class _Stdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def _answers(monkeypatch, *answers):
    """Feed input() these answers on a terminal; return the prompts it was given."""
    prompts, queue = [], list(answers)
    monkeypatch.setattr("sys.stdin", _Stdin(True))
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or queue.pop(0))
    return prompts


@pytest.mark.parametrize("answers, unit", [(["", "fnu_jy"], "fnu_jy"), (["y", "flam"], "flam"), (["flam"], "flam")])
def test_cli_offers_the_best_guess_and_takes_a_typed_unit(tmp_path, monkeypatch, capsys, answers, unit):
    from sed_tools import cli

    grid = _surface_flux_grid(tmp_path / "Cool")
    prompts = _answers(monkeypatch, *answers)
    assert cli._ask_flux_unit("Cool", sorted(str(p) for p in grid.glob("*.txt"))) == unit
    # Enter or 'y' does not take the guess: the unit must be typed.
    assert prompts == ["Flux unit of Cool (type its name, e.g. fnu_jy), or n to leave it uncleaned: "] * len(answers)
    out = capsys.readouterr().out
    assert "flam        erg/s/cm²/Å (F_λ)" in out and "fnu_jy      Jansky (F_ν)" in out
    assert ("Best guess: fnu_jy, Jansky (F_ν); guessed from the size of the flux values, "
            "assuming flux received at Earth.") in " ".join(out.split())


def test_cli_asks_again_after_an_unknown_unit_and_stops_on_no(tmp_path, monkeypatch, capsys):
    from sed_tools import cli

    grid = _surface_flux_grid(tmp_path / "Cool")
    prompts = _answers(monkeypatch, "Jy", "n")
    with pytest.raises(ValueError, match="No flux unit chosen for Cool; it was not cleaned"):
        cli._ask_flux_unit("Cool", sorted(str(p) for p in grid.glob("*.txt")))
    assert len(prompts) == 2
    assert "'Jy' is not one of: flam, fnu, fnu_jy, flam_si, flam_si_nm, flam_si_um, normalized" in capsys.readouterr().out


def test_cli_does_not_ask_when_the_flux_unit_is_declared(tmp_path, monkeypatch):
    from sed_tools import cli

    grid = _surface_flux_grid(tmp_path / "Declared", flux_header="erg/cm2/s/A")
    prompts = _answers(monkeypatch)
    assert cli._ask_flux_unit("Declared", sorted(str(p) for p in grid.glob("*.txt"))) is None
    assert prompts == []


def test_cli_rebuild_asks_once_per_grid_and_cleans_with_the_answer(tmp_path, monkeypatch):
    from sed_tools import cli
    from sed_tools.spectrum_io import read_text_spectrum

    for name in ("A", "B"):
        _surface_flux_grid(tmp_path / name)
    prompts = _answers(monkeypatch, "flam", "fnu_jy")
    cli.run_rebuild_flow(base_dir=str(tmp_path), models=["A", "B"], rebuild_h5=False, rebuild_flux_cube=False)

    assert [p.split(" (")[0] for p in prompts] == ["Flux unit of A", "Flux unit of B"]
    wl, flux = read_text_spectrum(str(tmp_path / "A" / "s0.txt"))
    np.testing.assert_allclose(flux, planck_flam(wl, 3500.0), rtol=1e-8)
    assert "# original_flux_unit = fnu" in (tmp_path / "B" / "s0.txt").read_text()


def test_cli_without_a_terminal_stops_naming_the_grid_and_the_guess(tmp_path, monkeypatch):
    from sed_tools import cli

    grid = _surface_flux_grid(tmp_path / "Cool")
    before = _contents(grid)
    monkeypatch.setattr("sys.stdin", _Stdin(False))
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("asked without a terminal"))

    with pytest.raises(RuntimeError) as err:
        cli.run_rebuild_flow(base_dir=str(tmp_path), models=["Cool"], rebuild_h5=False, rebuild_flux_cube=False)
    message = str(err.value)
    assert f"Cool: ValueError: Flux unit of {grid} is not declared; best guess: 'fnu_jy'" in message
    assert "--flux-unit fnu_jy" in message and "sed-tools asks for it when run in a terminal" in message
    assert _contents(grid) == before


def test_import_grid_stops_before_moving_files_and_imports_with_a_unit(tmp_path):
    from sed_tools.api import SED

    src = _surface_flux_grid(tmp_path / "mine")
    with pytest.raises(ValueError, match=rf"Flux unit of {src} is not declared; best guess: 'fnu_jy'"):
        SED.import_grid(src, model_root=tmp_path / "models", move=True, build_cube=False, build_h5=False)
    assert len(list(src.glob("*.txt"))) == 3
    assert not (tmp_path / "models" / "mine").exists()

    SED.import_grid(src, model_root=tmp_path / "models", move=True, build_cube=False, build_h5=False,
                    flux_unit="flam")
    assert "# original_flux_unit = flam" in (tmp_path / "models" / "mine" / "s0.txt").read_text()


def test_cli_import_asks_for_the_flux_unit_and_passes_it_on(tmp_path, monkeypatch):
    from sed_tools import cli

    src = _surface_flux_grid(tmp_path / "mine")
    calls = []
    monkeypatch.setattr("sed_tools.api.SED.import_grid", lambda *a, **k: calls.append(k))
    _answers(monkeypatch, str(src), "", "n", "flam")
    cli.run_import_flow(base_dir=str(tmp_path / "models"))
    assert calls[0]["flux_unit"] == "flam"


def test_cli_import_command_asks_on_a_terminal_and_not_for_a_dry_run(tmp_path, monkeypatch):
    from sed_tools import cli

    src = _surface_flux_grid(tmp_path / "mine")
    calls = []
    monkeypatch.setattr("sed_tools.api.SED.import_grid", lambda *a, **k: calls.append(k))
    prompts = _answers(monkeypatch, "fnu")
    for extra in ([], ["--dry-run"]):
        monkeypatch.setattr(sys, "argv", ["sed-tools", "import", "--path", str(src),
                                          "--base", str(tmp_path / "models")] + extra)
        assert cli.main() == 0
    assert [c["flux_unit"] for c in calls] == ["fnu", None]
    assert len(prompts) == 1


def test_fetch_stops_on_an_undeclared_flux_unit_and_cleans_with_the_one_given(tmp_path, monkeypatch):
    from sed_tools.api import SED

    def download(cls, catalog, sources, model_dir, workers, cuts):
        _surface_flux_grid(model_dir)
        return "svo"

    monkeypatch.setattr(SED, "_download_from_first_source", classmethod(download))
    with pytest.raises(ValueError, match=r"Flux unit of .*Cool is not declared; best guess: 'fnu_jy'"):
        SED.fetch("Cool", source="svo", model_root=tmp_path)
    sed = SED.fetch("Cool", source="svo", model_root=tmp_path, flux_unit="flam")
    assert len(sed.grid) == 3
    assert "# original_flux_unit = flam" in (tmp_path / "Cool" / "s0.txt").read_text()


# ---------------------------------------------------------------------
# Magnitudes of zero or negative band flux: the computed value, no error
# ---------------------------------------------------------------------

def _box_band(zero_point=2.0e-10):
    from sed_tools.flux_cube_tool import FilterCurve

    return FilterCurve("Band", np.array([3000.0, 4000.0, 5000.0, 6000.0]), np.array([0.0, 1.0, 1.0, 0.0]),
                       {"zp_vega": zero_point})


class _OneFilter:
    """Stands in for the SEDModel behind an EvaluatedSED: every filter resolves to one curve."""

    def __init__(self, curve):
        self.curve = curve

    def _resolve_filters(self, filters):
        return {"Band": self.curve}


def _photometry(flux_level, system, zero_point=2.0e-10):
    from sed_tools.models import EvaluatedSED

    wl = np.linspace(2000.0, 8000.0, 300)
    sed = EvaluatedSED(wl, np.full(wl.size, flux_level), {"teff": 5000.0, "logg": 4.0, "meta": 0.0},
                       _OneFilter(_box_band(zero_point)))
    return sed.photometry("Band", system=system)["Band"].magnitude


@pytest.mark.parametrize("system", ["AB", "Vega"])
def test_magnitude_of_zero_flux_is_infinite_in_both_systems(system):
    # The Vega path used to raise ValueError from math.log10(0).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _photometry(0.0, system) == math.inf
        assert _band_magnitude(0.0, system) == math.inf


@pytest.mark.parametrize("system", ["AB", "Vega"])
def test_magnitude_of_negative_flux_is_nan_in_both_systems(system):
    # The AB path used to return inf for a negative flux.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(_photometry(-1e-12, system))
        assert math.isnan(_band_magnitude(-1e-12, system))


def _band_magnitude(flux_level, system, zero_point=2.0e-10):
    from sed_tools.flux_cube_tool import _band_magnitude as band_magnitude

    wl = np.linspace(2000.0, 8000.0, 300)
    return band_magnitude(wl, np.full(wl.size, flux_level), _box_band(zero_point), system, None)[2]


@pytest.mark.parametrize("flux_level, zero_point, expected", [
    (1e-12, 0.0, -math.inf), (0.0, 0.0, math.nan), (1e-12, -2.0e-10, math.nan), (0.0, -2.0e-10, math.inf),
])
def test_vega_magnitude_with_a_zero_or_negative_zero_point_is_computed(flux_level, zero_point, expected):
    # A non-positive Vega zero point used to raise ValueError in both modules.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for magnitude in (_photometry(flux_level, "Vega", zero_point),
                          _band_magnitude(flux_level, "Vega", zero_point)):
            if math.isnan(expected):
                assert math.isnan(magnitude)
            else:
                assert magnitude == expected
