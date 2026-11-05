"""The flux-cube menus: mesa_prepare's exit statuses and messages, the variants of split grids
(listed, exported, printed), combine and densifier prompts that ask again or go back, output width, and
the warnings about averaged rows and filters past a model.
"""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pytest

from helpers_api import no_network  # noqa: F401 (an autouse fixture)
from helpers_cli import _wide_lines, answer, flat_filter, write_lookup
from helpers_cube import GOLDEN_MODELS, _feed_input


# ----------------------------------------------------------------------
# mesa_prepare: every failure exits non-zero
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Combine menu: invalid input asks again, never selects everything
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
# Band averages: a filter that extends past the SED
# ----------------------------------------------------------------------
def test_model_photometry_warns_once_about_filters_past_the_model(make_affine_cube, caplog):
    from sed_tools.models import SEDModel

    cube = make_affine_cube(wavelengths=(3000.0, 4000.0, 5000.0))
    sed = SEDModel(name="affine", flux_cube_path=cube.path)(4500.0, 2.5, -0.3)
    with caplog.at_level(logging.WARNING, logger="sed_tools.flux_cube_tool"):
        sed.photometry(flat_filter("V"), flat_filter("B"), system="AB")
    [warning] = [r.getMessage() for r in caplog.records]
    assert warning.endswith("so these bands are biased low: V (47.5%), B (47.5%).")


# ----------------------------------------------------------------------
# Flux-cube builder: lookup rows averaged onto one node
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
# Combine grids (menu 7): q or b goes back, as in the other pickers
# ----------------------------------------------------------------------
@pytest.mark.parametrize("reply", ["q", "b", "Q", "back"])
def test_combine_grid_selection_goes_back(reply, tmp_path, monkeypatch):
    from sed_tools.combine_stellar_atm import select_models_interactive

    grids = [(name, str(tmp_path / name)) for name in ("A", "B")]
    answer(monkeypatch, reply)
    assert select_models_interactive(grids) == []  # only Ctrl-C used to leave it


def test_combine_menu_q_returns_to_the_menu_without_failing(tmp_path, monkeypatch, capsys):
    from sed_tools import cli

    for name in ("A", "B"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\na.txt,5000,4,0\n")
    answer(monkeypatch, "q")
    assert cli._run_menu_flow(lambda: cli.run_combine_flow(base_dir=str(tmp_path))) is True
    assert "No spectral grids selected." in capsys.readouterr().out
    assert not (tmp_path / "combined_models").exists()


# ----------------------------------------------------------------------
# Lines no wider than an 80-column terminal (paths may be longer)
# ----------------------------------------------------------------------
def test_densifier_cube_list_fits_80_columns(tmp_path, monkeypatch, capsys):
    import numpy as np

    from conftest import write_flux_cube
    from sed_tools.grid_densifier import run_interactive_workflow

    name = "a-rather-long-spectral-grid-name_dense"
    write_flux_cube(tmp_path / name / "flux_cube.bin", [5750, 6000, 6250], [4.0], [0.0], [1000.0, 2000.0],
                    np.ones((3, 1, 1, 2)))
    answer(monkeypatch, "")
    run_interactive_workflow(base_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert f"   1) {name}/flux_cube.bin\n      Teff 5750–6250 K  max gap 250 K  (3 pts)\n" in out
    assert _wide_lines(out, str(tmp_path)) == []  # was 92 wide with a short name


def test_mesa_prepare_invalid_choice_fits_80_columns(tmp_path, monkeypatch):
    from sed_tools import mesa_prepare

    (tmp_path / "Kurucz").mkdir()
    (tmp_path / "Kurucz" / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\na.txt,5000,4,0\n")
    answer(monkeypatch, "")
    with pytest.raises(SystemExit) as excinfo:
        mesa_prepare.run_interactive(base_dir=str(tmp_path))
    message = str(excinfo.value.code)
    assert message == ("Error: invalid spectral grid selection ''.\n"
                       "Enter a number from 1 to 1 or a grid name. Nothing was exported.")
    assert _wide_lines(message) == []  # was 107 wide


# ---------------------------------------------------------------------
# Number prompts in the generator and densifier menus ask again
# ---------------------------------------------------------------------
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
# mesa_prepare lists and exports the variants 'sed-tools rebuild' writes
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
# Combining leaves out earlier combine outputs; the menu counts every spectrum
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
# Undeclared flux unit, coverage's A-D note, import's "Load with" hint
# ---------------------------------------------------------------------
def test_mesa_prepare_takes_a_combined_grid_as_it_is(two_grids, monkeypatch, capsys):
    # source_model and original_file differ on every row of a combined grid's lookup;
    # they say where each spectrum came from and are not variants of the grid.
    from sed_tools import mesa_prepare
    from sed_tools.api import SED

    np.random.seed(0)
    SED.combine(["Kurucz2003all", "Kurucz2003all__alpha_04"], output="omni", model_root=two_grids,
                visualization=False)
    capsys.readouterr()

    mesa_prepare.run_interactive(base_dir=str(two_grids), model="omni")

    out = capsys.readouterr().out
    assert "Note" not in out and "variants over" not in out
    assert "has one flux_cube.bin for all its spectra and no variants to choose between" in out


def test_mesa_prepare_ignores_the_ml_completers_columns(tmp_path, cube_writer, capsys):
    from sed_tools import mesa_prepare

    grid = tmp_path / "Ext"
    grid.mkdir()
    (grid / "lookup_table.csv").write_text(
        "file_name,teff,logg,metallicity,extended_by_ml,wl_min,wl_max\n"
        "a.txt,5000,4.0,0.0,True,100.0,400000.0\nb.txt,6000,4.0,0.0,False,,\n")
    cube_writer(grid / "flux_cube.bin", [5000.0, 6000.0], [4.0], [0.0], [1000.0, 2000.0], np.ones((2, 1, 1, 2)))

    (variant,) = mesa_prepare.list_variants(grid)

    assert variant.label == "full_grid" and "Note" not in capsys.readouterr().out
