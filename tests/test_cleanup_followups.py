"""Follow-ups to the cleanup: exit statuses, re-prompting menus and warnings that change no numbers."""

import json
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import requests


class NetworkUsed(BaseException):
    """Raised past every ``except Exception`` so a network call cannot be swallowed."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise NetworkUsed("tests must not use the network")

    monkeypatch.setattr(requests.Session, "send", refuse)


def answer(monkeypatch, *replies):
    """Feed ``input()`` these replies in order; a further prompt fails the test."""
    queue = list(replies)

    def fake_input(prompt=""):
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


# ----------------------------------------------------------------------
# 1. Catalogue gaps vs fetch failures
# ----------------------------------------------------------------------
NJM = "https://njm.invalid/sed_tools"


class FakeMirror:
    """Serves ``files`` under NJM; anything else is a 404."""

    base_url = NJM

    def __init__(self, files):
        self.files = files

    def is_available(self):
        return True

    def _get(self, url, timeout=None, headers=None):
        rel = url[len(NJM) + 1:]
        if rel not in self.files:
            raise requests.HTTPError("404", response=SimpleNamespace(status_code=404, headers={}))
        return SimpleNamespace(status_code=200, content=self.files[rel].encode(), headers={})


def catalogue_files(serve_ir=True):
    """Fac/Cam is served whole; Fac/Old has no properties file (a gap); Other/IR's may be missing (a failure)."""
    def instrument(bands, properties):
        return {"filters": bands, "detector_types": {}, "properties": properties, "info": None}

    index = {"generated": "2026-10-27T00:00:00", "filters": {"facilities_detailed": {
        "Fac": {"instruments": {"Cam": instrument(["g"], "properties.json"), "Old": instrument(["U"], None)}},
        "Other": {"instruments": {"IR": instrument(["K"], "properties.json")}},
    }}}
    files = {"index.json": json.dumps(index),
             "filters/Fac/Cam/properties.json": json.dumps({"bands": {"g": {"WavelengthPivot": 4800.0}}})}
    if serve_ir:
        files["filters/Other/IR/properties.json"] = json.dumps({"bands": {"K": {"WavelengthPivot": 21600.0}}})
    return files


def test_search_reports_gaps_apart_from_fetch_failures(tmp_path):
    from sed_tools.filter_catalogue import FilterCatalogue

    catalogue = FilterCatalogue(tmp_path / "filters", mirror=FakeMirror(catalogue_files(serve_ir=False)))
    result = catalogue.search()
    assert list(result["band"]) == ["g", "U", "K"]
    assert result.attrs["gaps"] == ["Fac/Old: NJM has no properties file"]
    assert result.attrs["errors"] == ["Other/IR: NJM does not serve filters/Other/IR/properties.json"]


# ----------------------------------------------------------------------
# 2. mesa_prepare: every failure exits non-zero
# ----------------------------------------------------------------------
def write_lookup(model_dir, header, rows):
    model_dir.mkdir(parents=True)
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    (model_dir / "lookup_table.csv").write_text("\n".join(lines) + "\n")


@pytest.fixture
def variant_grid(tmp_path, cube_writer):
    """``<base>/Grid`` with two alpha variants, each with its spectra and cube; returns base."""
    base = tmp_path / "stellar"
    grid = base / "Grid"
    write_lookup(grid, ["file_name", "teff", "logg", "metallicity", "alpha"],
                 [("a.txt", 5000, 4.0, 0.0, "0.0"), ("b.txt", 5000, 4.0, 0.0, "0.4")])
    for name in ("a.txt", "b.txt"):
        (grid / name).write_text("4000 1.0\n5000 2.0\n")
    for alpha in ("0.0", "0.4"):
        cube_writer(grid / "fluxcube_library" / f"flux_cube__alpha_{alpha}.bin",
                    [5000.0], [4.0], [0.0], [4000.0, 5000.0], np.ones((1, 1, 1, 2)))
    return base


def run_mesa_prepare(base, *replies, monkeypatch, **kwargs):
    from sed_tools import mesa_prepare

    answer(monkeypatch, *replies)
    with pytest.raises(SystemExit) as excinfo:
        mesa_prepare.run_interactive(base_dir=str(base), **kwargs)
    return str(excinfo.value.code)


def test_mesa_prepare_exits_1_without_spectral_grids(tmp_path, monkeypatch):
    message = run_mesa_prepare(tmp_path, monkeypatch=monkeypatch)
    assert message.startswith("Error: no spectral grids found in")


def test_mesa_prepare_exits_1_on_an_invalid_grid_choice(variant_grid, monkeypatch):
    message = run_mesa_prepare(variant_grid, "9", monkeypatch=monkeypatch)
    assert message.startswith("Error: invalid spectral grid selection '9'")


def test_mesa_prepare_exits_1_for_a_single_grid_without_a_flux_cube(tmp_path, monkeypatch):
    # It used to print "the existing flux_cube.bin already covers the full grid"
    # and exit 0 even when no flux_cube.bin had been built.
    write_lookup(tmp_path / "Plain", ["file_name", "teff", "logg", "metallicity"], [("a.txt", 5000, 4.0, 0.0)])
    message = run_mesa_prepare(tmp_path, monkeypatch=monkeypatch, model="Plain", output=str(tmp_path / "out"))
    assert "has no flux_cube.bin" in message and "Nothing was exported" in message
    assert not (tmp_path / "out").exists()


def test_mesa_prepare_accepts_a_single_grid_that_has_its_flux_cube(tmp_path, monkeypatch, capsys, cube_writer):
    import numpy as np

    from sed_tools import mesa_prepare

    write_lookup(tmp_path / "Plain", ["file_name", "teff", "logg", "metallicity"], [("a.txt", 5000, 4.0, 0.0)])
    cube_writer(tmp_path / "Plain" / "flux_cube.bin", [5000.0], [4.0], [0.0], [1000.0, 2000.0], np.ones((1, 1, 1, 2)))
    mesa_prepare.run_interactive(base_dir=str(tmp_path), model="Plain", output=str(tmp_path / "out"))
    assert "already MESA-ready" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_mesa_prepare_exits_1_on_an_invalid_variant_choice(variant_grid, monkeypatch):
    message = run_mesa_prepare(variant_grid, "3", monkeypatch=monkeypatch, model="Grid",
                               output=str(variant_grid / "out"))
    assert message.startswith("Error: invalid variant selection '3'")
    assert not (variant_grid / "out").exists()


def test_mesa_prepare_exits_1_when_overwrite_is_declined(variant_grid, monkeypatch):
    out = variant_grid / "out"
    out.mkdir()
    (out / "keep.txt").write_text("untouched")
    message = run_mesa_prepare(variant_grid, "1", "n", monkeypatch=monkeypatch, model="Grid", output=str(out))
    assert message == f"Error: {out} already exists and was not overwritten; nothing was exported."
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_mesa_prepare_still_exports_the_chosen_variant(variant_grid, monkeypatch, capsys):
    from sed_tools import mesa_prepare

    answer(monkeypatch, "2")
    out = variant_grid / "out"
    mesa_prepare.run_interactive(base_dir=str(variant_grid), model="Grid", output=str(out))
    assert sorted(p.name for p in out.iterdir()) == ["b.txt", "flux_cube.bin", "lookup_table.csv"]


def test_sed_tools_mesa_prepare_exit_status_is_1_on_failure(tmp_path):
    import subprocess
    from pathlib import Path

    write_lookup(tmp_path / "Plain", ["file_name", "teff", "logg", "metallicity"], [("a.txt", 5000, 4.0, 0.0)])
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-m", "sed_tools.cli", "mesa_prepare", "--base", str(tmp_path),
                           "--grid", "Plain", "--output", str(tmp_path / "out")],
                          cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert done.returncode == 1 and "has no flux_cube.bin" in done.stderr


# ----------------------------------------------------------------------
# 3. Combine menu: invalid input asks again, never selects everything
# ----------------------------------------------------------------------
def test_combine_menu_asks_again_until_the_selection_is_valid(tmp_path, monkeypatch, capsys):
    from sed_tools.combine_stellar_atm import select_models_interactive

    grids = [(name, str(tmp_path / name)) for name in ("A", "B", "C")]
    # Empty input and a bad token used to combine every grid; out-of-range
    # numbers were dropped silently. A repeated grid is refused too.
    answer(monkeypatch, "", "two", "1,", "0", "4", "1,9", "2,2", "3,1")
    assert select_models_interactive(grids) == [grids[2], grids[0]]  # in the order typed
    out = capsys.readouterr().out
    for reply in ("''", "'two'", "'1,'", "'0'", "'4'", "'1,9'", "'2,2'"):
        assert f"Invalid selection {reply}: enter distinct numbers from 1 to 3, comma-separated, or 'all'." in out
    assert "Using all" not in out


def test_combine_menu_all_still_selects_every_grid(tmp_path, monkeypatch):
    from sed_tools.combine_stellar_atm import select_models_interactive

    grids = [(name, str(tmp_path / name)) for name in ("A", "B")]
    answer(monkeypatch, "ALL")
    assert select_models_interactive(grids) == grids


# ----------------------------------------------------------------------
# 4a. ML completer: an input that covers too little of the network's window
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


# ----------------------------------------------------------------------
# 4b. Band averages: a filter that extends past the SED
# ----------------------------------------------------------------------
def flat_filter(name="V"):
    from sed_tools.flux_cube_tool import FilterCurve

    wavelength = np.arange(4000.0, 6001.0, 100.0)
    return FilterCurve(name, wavelength, np.ones_like(wavelength), {})


def test_band_averages_warn_about_transmission_outside_the_sed(caplog):
    from sed_tools.flux_cube_tool import band_average_flux_lambda, band_average_flux_nu

    sed_wavelength = np.arange(3000.0, 5001.0, 100.0)
    sed_flux = np.ones_like(sed_wavelength)
    for average in (band_average_flux_lambda, band_average_flux_nu):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="sed_tools.flux_cube_tool"):
            average(sed_wavelength, sed_flux, flat_filter())
        # Transmission is 1 from 4000 to 6000 Å; 950 of its 2000 Å·T lie past 5000 Å.
        assert ("Filter V: 47.5% of its transmission lies outside the SED's wavelength range "
                "(3000-5000 Å), where the band average counts the flux as zero, so it is biased low.") in caplog.text


def test_band_averages_inside_the_sed_do_not_warn(caplog):
    from sed_tools.flux_cube_tool import FilterCurve, band_average_flux_lambda

    sed_wavelength = np.arange(3000.0, 7001.0, 100.0)
    # Zero transmission past the SED's end does not bias the average.
    wavelength = np.arange(4000.0, 8001.0, 100.0)
    curve = FilterCurve("R", wavelength, np.where(wavelength <= 6000.0, 1.0, 0.0), {})
    with caplog.at_level(logging.WARNING, logger="sed_tools.flux_cube_tool"):
        band_average_flux_lambda(sed_wavelength, np.ones_like(sed_wavelength), flat_filter())
        band_average_flux_lambda(sed_wavelength, np.ones_like(sed_wavelength), curve)
    assert not caplog.records


def test_model_photometry_warns_about_a_filter_past_the_model(make_affine_cube, caplog):
    from sed_tools.models import SEDModel

    cube = make_affine_cube(wavelengths=(3000.0, 4000.0, 5000.0))
    sed = SEDModel(name="affine", flux_cube_path=cube.path)(4500.0, 2.5, -0.3)
    with caplog.at_level(logging.WARNING, logger="sed_tools.flux_cube_tool"):
        sed.photometry(flat_filter(), system="AB")
    assert "Filter V: 47.5% of its transmission lies outside the SED's wavelength range" in caplog.text


# ----------------------------------------------------------------------
# 4c. Flux-cube builder: lookup rows averaged onto one node
# ----------------------------------------------------------------------
def write_grid(model_dir, columns, rows):
    """One flat spectrum per row and a lookup table."""
    write_lookup(model_dir, columns, rows)
    for row in rows:
        (model_dir / row[0]).write_text("".join(f"{w:.1f} 1e-10\n" for w in np.linspace(1000.0, 10000.0, 20)))
    return model_dir


def build_cube(model_dir, caplog, **override):
    import importlib

    precompute = importlib.import_module("sed_tools.precompute_flux_cube")
    with caplog.at_level(logging.WARNING, logger="sed_tools.precompute_flux_cube"):
        precompute.precompute_flux_cube(str(model_dir), str(model_dir / "flux_cube.bin"),
                                        override_dict=override or None)
    return [r.getMessage() for r in caplog.records if "lookup rows were averaged" in r.getMessage()]


def test_bosz_bookkeeping_columns_that_vary_are_named_when_averaged(tmp_path, caplog):
    # am, cm and vturb are not extra axes, so rows differing only there share a node.
    grid = write_grid(tmp_path / "BOSZ", ["file_name", "teff", "logg", "metallicity", "am", "cm", "vturb"], [
        ("a.txt", 5000, 4.0, 0.0, 0.0, 0.0, 1), ("b.txt", 5000, 4.0, 0.0, 0.4, 0.0, 2),
        ("c.txt", 6000, 4.0, 0.0, 0.0, 0.0, 1), ("d.txt", 6000, 4.0, 0.0, 0.0, 0.0, 1),
        ("e.txt", 6000, 4.5, 0.0, 0.0, 0.0, 1),
    ])
    assert build_cube(grid, caplog) == [
        f"{grid / 'flux_cube.bin'}: 4 lookup rows were averaged onto 2 (Teff, logg, [M/H]) node(s), so each "
        "of those nodes holds a mean spectrum. The averaged rows differ in: am, vturb."]


def test_duplicate_nodes_without_extra_axes_are_reported(tmp_path, caplog):
    grid = write_grid(tmp_path / "Plain", ["file_name", "teff", "logg", "metallicity"], [
        ("a.txt", 5000, 4.0, 0.0), ("b.txt", 5000.0, 4.0, 0.0), ("c.txt", 6000, 4.0, 0.0),
    ])
    (message,) = build_cube(grid, caplog)
    assert "2 lookup rows were averaged onto 1 (Teff, logg, [M/H]) node(s)" in message
    assert message.endswith("differ in: no column but the file name (duplicate nodes).")


def test_a_grid_with_one_row_per_node_does_not_warn(tmp_path, caplog):
    grid = write_grid(tmp_path / "Plain", ["file_name", "teff", "logg", "metallicity"], [
        ("a.txt", 5000, 4.0, 0.0), ("b.txt", 6000, 4.0, 0.0),
    ])
    assert build_cube(grid, caplog) == []


def test_split_and_filter_cubes_report_their_own_averaged_rows(tmp_path, caplog):
    columns = ["file_name", "teff", "logg", "metallicity", "alpha", "vturb"]
    rows = [("a.txt", 5000, 4.0, 0.0, "0.0", 1), ("b.txt", 5000, 4.0, 0.0, "0.0", 2),
            ("c.txt", 5000, 4.0, 0.0, "0.4", 1)]
    split = write_grid(tmp_path / "Split", columns, rows)
    (message,) = build_cube(split, caplog)
    assert message.startswith(f"{split / 'Split_alpha_0p00' / 'flux_cube.bin'}: 2 lookup rows were averaged")
    assert message.endswith("differ in: vturb.")

    caplog.clear()
    filtered = write_grid(tmp_path / "Filtered", columns, rows)
    strategy = {"on_collision": {"strategy": "filter", "filter": {"alpha": "0.4"}}}
    assert build_cube(filtered, caplog, **strategy) == []
    strategy["on_collision"]["filter"]["alpha"] = "0.0"
    (message,) = build_cube(filtered, caplog, **strategy)
    assert message.startswith(f"{filtered / 'flux_cube.bin'}: 2 lookup rows were averaged")


# ----------------------------------------------------------------------
# 5. Picker: the longest labels are shown in full
# ----------------------------------------------------------------------
def test_picker_shows_the_longest_label_in_full(monkeypatch, capsys):
    from sed_tools import ui_utils

    monkeypatch.setattr(ui_utils, "terminal_width", lambda default=80: 80)
    answer(monkeypatch, "1")
    assert ui_utils.prompt_choice(["alpha", "beta"], "Things", use_color=False) == 0
    out = capsys.readouterr().out
    # The longest label used to be cut 2 characters short ("alp", "bet").
    assert "   1. alpha  " in out and "   2. beta" in out
