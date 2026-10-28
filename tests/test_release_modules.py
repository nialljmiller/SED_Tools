"""Tests for the 0.5.0 release fixes in the module files (REPORT_CLI P4, P9, P16, P17, P20).

Tests that need torch skip without it; the rest are torch-free.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from sed_tools import ml_sed_completer as mc

GOLDEN_MODELS = Path(__file__).resolve().parent / "golden" / "inputs" / "stellar_models"


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
# P4: the completer keeps every original sample when it extends a spectrum
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


def _feed_input(monkeypatch, answers):
    feed = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(feed))


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
# P1 (module side): number prompts in the generator and densifier menus ask again
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
    assert "'nan' is not a number; try again." in out and "'' is not a number; try again." in out
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


def test_densifier_spacing_prompt_asks_again(tmp_path, cube_writer, monkeypatch, capsys):
    from sed_tools import grid_densifier

    cube_writer(tmp_path / "grid" / "flux_cube.bin", [5000.0, 7000.0], [4.0], [0.0], [1000.0, 2000.0],
                np.ones((2, 1, 1, 2)))
    calls = []
    monkeypatch.setattr(grid_densifier, "densify_grid", lambda **kw: calls.append(kw))
    _feed_input(monkeypatch, ["1", "fine", "-5", "250", "", ""])
    grid_densifier.run_interactive_workflow(str(tmp_path))
    assert [c["teff_spacing"] for c in calls] == [250.0]
    out = capsys.readouterr().out
    assert "'fine' is not a positive number; try again." in out
    assert "'-5' is not a positive number; try again." in out


# ---------------------------------------------------------------------
# P20: mesa_prepare lists and exports the variants 'sed-tools rebuild' writes
# ---------------------------------------------------------------------

def _kurucz_alpha(base):
    """Both golden Kurucz grids (alpha 0 and 0.4) as one grid; lh and vtur are the same throughout."""
    import pandas as pd

    from sed_tools.svo_regen_spectra_lookup import regenerate_lookup_table

    grid = base / "KuruczAlpha"
    grid.mkdir(parents=True)
    for name in ("Kurucz2003all", "Kurucz2003all__alpha_04"):
        for path in (GOLDEN_MODELS / name).glob("*.txt"):
            shutil.copy2(path, grid / path.name)
    regenerate_lookup_table(str(grid))
    lookup = pd.read_csv(grid / "lookup_table.csv", dtype=str)
    assert sorted(set(lookup[" vtur"])) == ["2 km/s"]  # a value with a space and a '/'
    return grid


@pytest.fixture
def split_grid(tmp_path):
    """KuruczAlpha after a rebuild: split into one subgrid per alpha, plus variants_index.csv."""
    from sed_tools.api import _build_flux_cube

    grid = _kurucz_alpha(tmp_path / "stellar")
    assert _build_flux_cube(grid).name == "variants_index.csv"
    assert not (grid / "flux_cube.bin").exists()
    return grid


def test_list_variants_reads_the_subgrids_rebuild_writes(split_grid):
    # Used to raise "No flux_cube.bin found in .../KuruczAlpha. Run 'sed-tools rebuild' first."
    from sed_tools.mesa_prepare import list_variants

    variants = list_variants(split_grid)

    assert [v.label for v in variants] == ["alpha_0", "alpha_0.4"]
    assert [v.extra_axes for v in variants] == [{"alpha": "0"}, {"alpha": "0.4"}]
    assert [v.n_spectra for v in variants] == [12, 12]
    sub = "KuruczAlpha_alpha_0p40_lh_1p25_vtur_2_km_s"
    assert variants[1].flux_cube_path == split_grid / sub / "flux_cube.bin"
    assert variants[1].source_model_dir == split_grid
    assert variants[1].wavelength_n == 1199
    assert set(variants[1].lookup_rows["alpha"]) == {"0.4"}


def test_export_variant_writes_a_mesa_ready_folder_from_a_split_grid(split_grid, tmp_path):
    import pandas as pd

    from sed_tools.mesa_prepare import export_variant, list_variants

    variant = list_variants(split_grid)[1]
    out = export_variant(variant, tmp_path / "export")

    spectra = sorted(out.glob("*.txt"))
    assert len(spectra) == 12 and not any(p.is_symlink() for p in spectra)
    for path in spectra:
        assert path.read_bytes() == (split_grid / path.name).read_bytes()
    sub = split_grid / "KuruczAlpha_alpha_0p40_lh_1p25_vtur_2_km_s"
    assert (out / "flux_cube.bin").read_bytes() == (sub / "flux_cube.bin").read_bytes()
    lookup = pd.read_csv(out / "lookup_table.csv", dtype=str)
    assert sorted(lookup["file_name"]) == [p.name for p in spectra]
    assert set(lookup["alpha"]) == {"0.4"}


def test_mesa_prepare_menu_exports_a_variant_of_a_split_grid(split_grid, monkeypatch, capsys):
    from sed_tools import mesa_prepare

    _feed_input(monkeypatch, ["2", ""])
    mesa_prepare.run_interactive(base_dir=str(split_grid.parent), model="KuruczAlpha")

    out = capsys.readouterr().out
    assert "Sub-variants of KuruczAlpha: 2 variants over alpha" in out
    assert "lh" not in out.split("Sub-variants")[1] and "vtur" not in out
    exported = split_grid.parent / "KuruczAlpha__alpha_0.4"
    assert len(list(exported.glob("*.txt"))) == 12 and (exported / "flux_cube.bin").is_file()


def test_list_variants_raises_for_a_subgrid_without_its_cube(split_grid):
    from sed_tools.mesa_prepare import list_variants

    sub = split_grid / "KuruczAlpha_alpha_0p00_lh_1p25_vtur_2_km_s"
    (sub / "flux_cube.bin").rename(sub / "flux_cube.bin.moved")
    with pytest.raises(FileNotFoundError, match="lists the subgrid KuruczAlpha_alpha_0p00_lh_1p25_vtur_2_km_s"):
        list_variants(split_grid)


def test_mesa_prepare_on_a_uniform_grid_says_one_thing(tmp_path, monkeypatch, capsys):
    # The golden Kurucz grid has alpha, lh and vtur columns with one value each. It used
    # to print "the master flux_cube.bin averages all extra-axis variants (alpha, lh, vtur)"
    # and then "This spectral grid has no extra-axis variants".
    from sed_tools import mesa_prepare
    from sed_tools.precompute_flux_cube import precompute_flux_cube

    grid = tmp_path / "Kurucz2003all"
    shutil.copytree(GOLDEN_MODELS / "Kurucz2003all", grid)
    precompute_flux_cube(str(grid), str(grid / "flux_cube.bin"))
    capsys.readouterr()

    mesa_prepare.run_interactive(base_dir=str(tmp_path), model="Kurucz2003all")

    out = capsys.readouterr().out
    assert "averages" not in out and "Note" not in out
    assert "has one flux_cube.bin for all its spectra and no variants to choose between" in out
    assert "already MESA-ready" in out


def test_mesa_prepare_notes_a_mean_cube_naming_only_the_varying_axes(tmp_path, monkeypatch, capsys):
    from sed_tools import mesa_prepare
    from sed_tools.precompute_flux_cube import precompute_flux_cube

    grid = _kurucz_alpha(tmp_path)
    precompute_flux_cube(str(grid), str(grid / "flux_cube.bin"),
                         override_dict={"on_collision": {"strategy": "mean"}})
    capsys.readouterr()

    (variant,) = mesa_prepare.list_variants(grid)

    out = capsys.readouterr().out
    assert "Note: KuruczAlpha has 2 variants over alpha, but no per-variant cubes" in out
    assert "lh" not in out and "vtur" not in out
    assert variant.label == "full_grid" and variant.flux_cube_path == grid / "flux_cube.bin"


def _legacy_library_grid(tmp_path, cube_writer):
    """A grid in the layout of earlier SED_Tools versions: cube names carry every extra column."""
    grid = tmp_path / "Grid"
    grid.mkdir()
    (grid / "lookup_table.csv").write_text(
        "file_name,teff,logg,metallicity,alpha,vtur\na.txt,5000,4.0,0.0,0.0,2\nb.txt,5000,4.0,0.0,0.4,2\n")
    for name in ("a.txt", "b.txt"):
        (grid / name).write_text("4000 1.0\n5000 2.0\n")
    for alpha in ("0.0", "0.4"):
        cube_writer(grid / "fluxcube_library" / f"flux_cube__alpha_{alpha}__vtur_2.bin",
                    [5000.0], [4.0], [0.0], [4000.0, 5000.0], np.ones((1, 1, 1, 2)))
    return grid


def test_list_variants_still_reads_a_fluxcube_library(tmp_path, cube_writer):
    from sed_tools.mesa_prepare import export_variant, list_variants

    variants = list_variants(_legacy_library_grid(tmp_path, cube_writer))

    assert [v.label for v in variants] == ["alpha_0.0__vtur_2", "alpha_0.4__vtur_2"]
    out = export_variant(variants[1], tmp_path / "export")
    assert sorted(p.name for p in out.iterdir()) == ["b.txt", "flux_cube.bin", "lookup_table.csv"]


def test_print_variant_table_names_only_the_axes_that_vary(tmp_path, cube_writer, capsys):
    from sed_tools.mesa_prepare import list_variants, print_variant_table

    print_variant_table(list_variants(_legacy_library_grid(tmp_path, cube_writer)), model_name="Grid")
    out = capsys.readouterr().out
    assert "Sub-variants of Grid: 2 variants over alpha" in out
    assert "alpha=0.4" in out and "vtur=" not in out


def test_export_variant_refuses_a_missing_cube_before_writing(tmp_path):
    from sed_tools.mesa_prepare import export_variant, list_variants

    grid = tmp_path / "Plain"
    grid.mkdir()
    (grid / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\na.txt,5000,4.0,0.0\n")
    (grid / "a.txt").write_text("4000 1.0\n5000 2.0\n")
    (variant,) = list_variants(grid)
    with pytest.raises(FileNotFoundError, match="does not exist; nothing was exported"):
        export_variant(variant, tmp_path / "out")
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------
# P16: combining leaves out earlier combine outputs; the menu counts every spectrum
# ---------------------------------------------------------------------

@pytest.fixture
def two_grids(tmp_path):
    base = tmp_path / "stellar"
    for name in ("Kurucz2003all", "Kurucz2003all__alpha_04"):
        shutil.copytree(GOLDEN_MODELS / name, base / name)
    return base


def test_find_stellar_models_leaves_out_earlier_combine_outputs(two_grids, capsys):
    from sed_tools.api import SED
    from sed_tools.combine_stellar_atm import find_stellar_models

    np.random.seed(0)
    SED.combine(["Kurucz2003all", "Kurucz2003all__alpha_04"], output="omni", model_root=two_grids,
                visualization=False)
    # A combine output whose lookup was rebuilt since is still found by its default name.
    shutil.copytree(GOLDEN_MODELS / "Kurucz2003all", two_grids / "combined_models")
    capsys.readouterr()

    found = find_stellar_models(str(two_grids))

    assert sorted(name for name, _ in found) == ["Kurucz2003all", "Kurucz2003all__alpha_04"]
    assert ("Leaving out combined_models, omni: written by an earlier combine. "
            "To combine one again, name it explicitly.") in capsys.readouterr().out


def test_non_interactive_combine_does_not_fold_in_the_previous_output(two_grids, capsys):
    # The second run used to combine Kurucz2003all, Kurucz2003all__alpha_04 and
    # combined_models: 48 spectra, each of the 24 twice.
    import matplotlib
    import pandas as pd

    from sed_tools.cli import run_combine_flow

    matplotlib.use("Agg")
    np.random.seed(0)
    run_combine_flow(str(two_grids), interactive=False)
    run_combine_flow(str(two_grids), output_name="combined2", interactive=False)

    out = capsys.readouterr().out
    assert "Selected 2 spectral grids to combine:" in out and "- combined_models" not in out
    lookup = pd.read_csv(two_grids / "combined2" / "lookup_table.csv")
    assert len(lookup) == 24
    assert set(lookup["source_model"]) == {"Kurucz2003all", "Kurucz2003all__alpha_04"}


def test_combine_menu_counts_every_spectrum_of_a_hash_headed_lookup(tmp_path, monkeypatch, capsys):
    # comment="#" dropped the '#file_name,...' header, so the first spectrum became the header.
    from sed_tools._lookup_io import write_lookup_csv
    from sed_tools.combine_stellar_atm import select_models_interactive

    grids = []
    for name, header in (("hashed", True), ("plain", False)):
        (tmp_path / name).mkdir()
        lookup = tmp_path / name / "lookup_table.csv"
        if header:
            write_lookup_csv({"file_name": ["a.txt", "b.txt", "c.txt"], "teff": [5000, 5500, 6000]},
                             str(lookup))
        else:
            lookup.write_text("file_name,teff\na.txt,5000\nb.txt,5500\nc.txt,6000\n")
        grids.append((name, str(tmp_path / name)))
    _feed_input(monkeypatch, ["all"])

    select_models_interactive(grids)

    out = capsys.readouterr().out
    assert "1. hashed (3 model spectra)" in out and "2. plain (3 model spectra)" in out


# ---------------------------------------------------------------------
# P9 (module side): atmosphere and coverage errors say what to choose, not Python syntax
# ---------------------------------------------------------------------

def _camera(root, *, default_site=None, sites=()):
    """Test/Camera with one band g, a Gaussian at 5000 Å, and the given atmosphere sites."""
    from sed_tools.filter_components import InstrumentManifest
    from sed_tools.filter_metadata import PHOTON, write_filter_file

    inst = root / "Test" / "Camera"
    inst.mkdir(parents=True)
    wl = np.linspace(4000.0, 6000.0, 201)
    write_filter_file(inst / "g.dat", wl, np.exp(-0.5 * ((wl - 5000) / 200) ** 2), detector_type=PHOTON,
                      source="svo", wavelength_unit="Angstrom", extra={"Components": "filter"})
    manifest = InstrumentManifest.load(inst)
    manifest.facility, manifest.instrument = "Test", "Camera"
    manifest.ground_based = True
    manifest.default_site = default_site
    manifest.atmosphere_sites = list(sites)
    manifest.outside_range = "error"
    manifest.save(inst)
    return inst


def _write_team_site(root, name, entries, **kw):
    """A team atmosphere site; entries are (airmass, pwv, (wavelengths, transmission))."""
    from sed_tools.component_sources._common import DerivedCurve, SiteAtmosphere, write_site
    from sed_tools.filter_components import TEAM

    def table(wavelength, transmission):
        return lambda session: (np.array(wavelength), np.array(transmission))

    curves = [(am, pwv, DerivedCurve("u", table(*curve), "")) for am, pwv, curve in entries]
    write_site(root, SiteAtmosphere(site=name, altitude_m=1000.0, entries=curves, source="test team",
                                    kind=TEAM, model="MODTRAN", model_version="not stated",
                                    citation="Someone 2001", **kw), None)


@pytest.mark.parametrize("site_kw, match", [
    ({}, r"^Test/Camera band g: .* covers 4500-7000 A but the band spans"),
    ({"outside_range": "error", "coverage_tolerance": 1e-4}, r"^Test/Camera band g: .* lies beyond it"),
])
def test_coverage_error_names_the_band(tmp_path, site_kw, match):
    # Used to be "am1.00_pwv1.dat covers 9000-56000 A but 70.25% of the band lies beyond it".
    from sed_tools.filter_components import FILTER, ComponentCoverageError, compose_band

    inst = _camera(tmp_path / "filters", default_site="short_team")
    _write_team_site(tmp_path / "atm", "short_team", [(1.2, None, ([4500.0, 7000.0], [0.8, 0.8]))], **site_kw)
    with pytest.raises(ComponentCoverageError, match=match):
        compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=tmp_path / "atm")


def test_several_sites_error_lists_the_sites_to_choose_from():
    # Used to be "... sites ['hopkins_2mass', 'ctio_2mass']; choose one with atmosphere={'site': ..., ...}".
    from sed_tools.filter_components import AtmosphereSpec, ComponentError, InstrumentManifest, resolve_atmosphere

    twomass = InstrumentManifest(facility="2MASS", instrument="2MASS", bands=["J"], ground_based=True)
    with pytest.raises(ComponentError) as excinfo:
        resolve_atmosphere(twomass, AtmosphereSpec(None, None))
    assert str(excinfo.value) == "2MASS/2MASS has several atmosphere sites; choose a site: hopkins_2mass, ctio_2mass"


def test_missing_conditions_error_says_what_the_site_has(tmp_path):
    # Used to be "No cached atmosphere for hopkins_2mass_am1.2; fetch or generate the requested conditions".
    from sed_tools.filter_components import AtmosphereSpec, ComponentError, atmosphere_file

    flat = ([3000.0, 7000.0], [0.9, 0.9])
    _write_team_site(tmp_path, "plexus_team", [(None, None, flat)])
    _write_team_site(tmp_path, "grid_team", [(1.0, 2.5, flat), (1.2, 2.5, flat), (1.2, 5.0, flat)])
    cases = [
        (AtmosphereSpec("plexus_team", 1.2),
         "Site plexus_team has no atmosphere for airmass 1.2; it has unstated airmass (the atmosphere as published)"),
        (AtmosphereSpec("grid_team", 1.5, 2.5),
         "Site grid_team has no atmosphere for airmass 1.5 and PWV 2.5 mm; it has airmass 1, 1.2 with PWV 2.5, 5 mm"),
        (AtmosphereSpec("nowhere", 1.2), "No atmosphere of site nowhere is cached; fetch or generate the site first"),
    ]
    for spec, message in cases:
        with pytest.raises(ComponentError) as excinfo:
            atmosphere_file(tmp_path, spec)
        assert str(excinfo.value) == message


def test_atmosphere_without_conditions_error_has_no_python_syntax(tmp_path):
    from sed_tools.filter_components import ATMOSPHERE, FILTER, ComponentError, compose_band

    inst = _camera(tmp_path / "filters", default_site="short_team")
    with pytest.raises(ComponentError) as excinfo:
        compose_band(inst, "g", [FILTER, ATMOSPHERE], atmosphere_root=tmp_path / "atm")
    assert str(excinfo.value) == ("Including the atmosphere needs the observing conditions: an airmass, "
                                  "and a PWV where the site models it")


# ---------------------------------------------------------------------
# Undeclared flux unit, coverage's A-D note, import's "Load with" hint
# ---------------------------------------------------------------------

def _undeclared_flux_grid(directory):
    """Three spectra of a surface flux in erg/s/cm2/A whose headers do not declare the flux unit."""
    from sed_tools._constants import planck_flam

    directory.mkdir(parents=True)
    wl = np.linspace(3000.0, 10000.0, 200)
    for i, teff in enumerate((3500.0, 3600.0, 3700.0)):
        lines = [f"# teff = {teff}\n", "# logg = 4.5\n", "# metallicity = 0.0\n", "# wavelength_unit = Angstrom\n"]
        lines += [f"{w:.6f} {f:.8e}\n" for w, f in zip(wl, planck_flam(wl, teff))]
        (directory / f"s{i}.txt").write_text("".join(lines))
    return directory


def test_undeclared_flux_unit_error_names_the_cli_option(tmp_path):
    from sed_tools.spectra_cleaner import check_flux_unit

    grid = _undeclared_flux_grid(tmp_path / "Cool")
    with pytest.raises(ValueError) as excinfo:
        check_flux_unit(str(grid), sorted(str(p) for p in grid.glob("*.txt")), sample_fraction=1.0)
    message = str(excinfo.value)
    assert "Pass the unit as flux_unit='fnu_jy'" in message
    assert "on the command line, pass --flux-unit fnu_jy to sed-tools grids, rebuild or import" in message


def _coverage_grid(directory, teffs):
    directory.mkdir()
    rows = "".join(f"s{i}.txt,{t},4.0,0.0\n" for i, t in enumerate(teffs))
    (directory / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\n" + rows)
    return directory


def test_coverage_says_why_the_normality_test_was_skipped(tmp_path, monkeypatch, capsys):
    # It said "scipy required" whenever the test was skipped, with scipy installed.
    import sys

    from sed_tools.grid_coverage import coverage_summary, print_coverage

    print_coverage(coverage_summary(_coverage_grid(tmp_path / "few", [5000, 5500, 6000])))
    out = capsys.readouterr().out
    assert "A-D     : skipped (needs at least 8 distinct values; this axis has 3)" in out
    assert "scipy" not in out

    monkeypatch.setitem(sys.modules, "scipy.stats", None)  # as if scipy were not installed
    print_coverage(coverage_summary(_coverage_grid(tmp_path / "many", range(4000, 9000, 500))))
    out = capsys.readouterr().out
    assert "    A-D     : skipped (needs scipy, which is not installed)" in out


def test_import_hint_names_the_model_root_it_imported_into(tmp_path, monkeypatch, capsys):
    from sed_tools.api import SED

    models = tmp_path / "models"
    SED.import_grid(_undeclared_flux_grid(tmp_path / "mine"), model_root=models, build_cube=False,
                    build_h5=False, flux_unit="flam")
    assert f"Load with: SED.local('mine', model_root={str(models)!r})" in capsys.readouterr().out

    monkeypatch.setattr(SED, "_model_root", models)
    SED.import_grid(_undeclared_flux_grid(tmp_path / "default"), build_cube=False, build_h5=False,
                    flux_unit="flam")
    assert "Load with: SED.local('default')\n" in capsys.readouterr().out
