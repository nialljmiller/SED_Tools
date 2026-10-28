"""The last should-fix items before 0.5.0 (FINAL_050.md): menu answers, widths, missing ML packages,
the detector of an instrument not yet installed, and NJM's archived copies in filters-info."""

from __future__ import annotations

import pytest


def answer(monkeypatch, *replies):
    """Feed ``input()`` these replies in order; a further prompt fails the test."""
    queue = list(replies)

    def fake_input(prompt=""):
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


# ----------------------------------------------------------------------
# Config (menu 14): q, b or Enter keep the data directory
# ----------------------------------------------------------------------
@pytest.mark.parametrize("reply", ["", "q", "Q", "b", "B"])
def test_config_menu_keeps_the_data_directory(reply, monkeypatch):
    from sed_tools import cli

    answer(monkeypatch, reply)
    changed = []
    monkeypatch.setattr(cli, "set_data_dir", lambda *a, **k: changed.append(a))
    cli.run_config_flow()
    assert changed == []  # "q" used to be saved as data_dir = "q"


def test_config_menu_takes_any_other_answer_as_a_path(monkeypatch, tmp_path):
    from sed_tools import cli

    answer(monkeypatch, str(tmp_path / "q"))
    changed = []
    monkeypatch.setattr(cli, "set_data_dir", lambda path, **k: changed.append(path))
    cli.run_config_flow()
    assert changed == [str(tmp_path / "q")]


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
def _wide_lines(text, *allowed):
    """Lines over 80 columns that contain none of the ``allowed`` substrings (paths)."""
    return [line for line in text.splitlines() if len(line) > 80 and not any(a in line for a in allowed)]


def test_filter_combine_intro_fits_80_columns(tmp_path, monkeypatch, capsys):
    from sed_tools.cli import run_filter_combine_flow

    answer(monkeypatch, "")
    run_filter_combine_flow(base_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert "MESA-compatible filter folder and index file" in out
    assert _wide_lines(out, str(tmp_path)) == []  # was 87 wide


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


# ----------------------------------------------------------------------
# Pickers: a label is cut only when the terminal is narrower than it
# ----------------------------------------------------------------------
def _picker_page(labels, columns, monkeypatch, capsys, **kwargs):
    from sed_tools.ui_utils import prompt_choice

    monkeypatch.setenv("COLUMNS", str(columns))
    monkeypatch.setenv("LINES", "40")
    answer(monkeypatch, "q")
    assert prompt_choice(labels, "Local filter sets to combine", use_color=False, **kwargs) is None
    return capsys.readouterr().out.splitlines()


def test_combine_picker_shows_a_long_product_label_whole_at_80_columns(monkeypatch, capsys):
    long = "2MASS/2MASS/products/filter+atm_ctio_2mass_amunstated (3 filters)"
    lines = _picker_page([long, "Generic/Johnson (7 filters)"], 80, monkeypatch, capsys,
                         multi=True, page_size=50, max_cols=2)
    assert any(line.strip() == f"1. {long}" for line in lines)  # was cut to '(3 filter…'
    assert max(len(line) for line in lines) <= 80


def test_picker_cuts_with_an_ellipsis_only_when_the_terminal_is_narrower(monkeypatch, capsys):
    long = "x" * 70
    lines = _picker_page([long], 60, monkeypatch, capsys)
    row = next(line for line in lines if line.lstrip().startswith("1."))
    assert row.rstrip().endswith("…") and len(row) <= 59


def test_picker_keeps_its_columns_for_short_names(monkeypatch, capsys):
    names = [f"name{i:02d}" for i in range(30)]
    lines = _picker_page(names, 80, monkeypatch, capsys)
    rows = [line for line in lines if line.lstrip()[:1].isdigit()]
    assert len(rows) == 10 and all(line.count(".") == 3 for line in rows)  # three columns, as before


# ----------------------------------------------------------------------
# ML without scipy, torch or scikit-learn: one line naming what to install, no traceback
# ----------------------------------------------------------------------
FULL = 'pip install "sed-tools[full]"'


def _without(monkeypatch, *modules):
    """Make ``modules`` uninstalled for this test: importing them (or a submodule another test
    imported already) fails, and find_spec finds nothing."""
    import sys

    for name in modules:
        for loaded in [key for key in sys.modules if key == name or key.startswith(name + ".")] or [name]:
            monkeypatch.setitem(sys.modules, loaded, None)


def _run_cli(monkeypatch, *argv):
    import sys

    from sed_tools import cli

    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    return cli.main()


#: The packages missing in each partial install: none of the three, scipy only, torch only.
INSTALLS = {"none": ("scipy", "torch", "sklearn"), "scipy-only": ("torch", "sklearn"),
            "torch-only": ("scipy", "sklearn")}


def _trained_models(tmp_path):
    """A models directory holding one completer and one generator, as list_models reads them."""
    models = tmp_path / "models"
    (models / "my_completer_model").mkdir(parents=True)
    (models / "my_completer_model" / "config.json").write_text('{"version": "1", "framework": "pytorch"}')
    (models / "my_generator_model").mkdir()
    (models / "my_generator_model" / "config.json").write_text('{"model_type": "sed_generator"}')
    return models


@pytest.mark.parametrize("install", INSTALLS)
def test_listing_models_works_with_any_partial_install(install, monkeypatch, capsys, tmp_path):
    # 0.4.1 listed models with scipy but no torch; that must keep working (and now needs neither)
    _without(monkeypatch, *INSTALLS[install])
    models = _trained_models(tmp_path)
    answer(monkeypatch, "3")
    assert _run_cli(monkeypatch, "ml_completer", "--base", str(tmp_path), "--models-dir", str(models)) == 0
    answer(monkeypatch, "4")
    assert _run_cli(monkeypatch, "ml_generator", "--base", str(tmp_path), "--models-dir", str(models)) == 0
    captured = capsys.readouterr()
    assert "my_completer_model" in captured.out and "my_generator_model" in captured.out
    assert captured.err == ""


# (command, menu choice, what each install lacks for it); None: the action is not refused
REFUSALS = [
    ("ml_completer", "1", {"none": "torch, scikit-learn", "scipy-only": "torch, scikit-learn",
                           "torch-only": "scikit-learn"}),
    ("ml_completer", "2", {"none": "scipy, torch", "scipy-only": "torch", "torch-only": "scipy"}),
    ("ml_generator", "1", {"none": "torch, scikit-learn", "scipy-only": "torch, scikit-learn",
                           "torch-only": "scikit-learn"}),
    ("ml_generator", "2", {"none": "torch", "scipy-only": "torch", "torch-only": None}),
    ("ml_generator", "3", {"none": "torch", "scipy-only": "torch", "torch-only": None}),
]


@pytest.mark.parametrize("install", INSTALLS)
@pytest.mark.parametrize("command, choice, lacking", REFUSALS)
def test_an_ml_action_is_refused_when_chosen_if_it_needs_a_missing_package(
        command, choice, lacking, install, monkeypatch, capsys, tmp_path):
    _without(monkeypatch, *INSTALLS[install])
    answer(monkeypatch, choice)  # refused before any of the action's questions (a further one fails)
    status = _run_cli(monkeypatch, command, "--base", str(tmp_path), "--models-dir", str(tmp_path / "models"))
    err = capsys.readouterr().err
    if lacking[install] is None:  # generating needs only torch: it goes on (here: no generator to choose)
        assert status == 0 and err == ""
    else:
        assert status == 1 and err == f"Error: missing {lacking[install]}: {FULL}\n"


def test_ml_generator_auto_without_torch_says_what_to_install(monkeypatch, capsys, tmp_path):
    import importlib.util

    if importlib.util.find_spec("optuna") is None:
        pytest.skip("optuna is checked first")
    _without(monkeypatch, "torch")
    assert _run_cli(monkeypatch, "ml_generator", "--auto", "--library", str(tmp_path),
                    "--output", str(tmp_path / "out")) == 1
    assert capsys.readouterr().err == f"Error: missing torch: {FULL}\n"  # was a torch traceback


def _densifier_inputs(tmp_path):
    import numpy as np

    from conftest import write_flux_cube

    write_flux_cube(tmp_path / "Kurucz" / "flux_cube.bin", [5000, 7000], [4.0], [0.0], [1000.0, 2000.0],
                    np.ones((2, 1, 1, 2)))
    model = tmp_path / "models" / "gen"
    model.mkdir(parents=True)
    return model


def test_densify_with_an_ml_model_without_torch_says_what_to_install(monkeypatch, capsys, tmp_path):
    model = _densifier_inputs(tmp_path)
    _without(monkeypatch, "torch")
    assert _run_cli(monkeypatch, "densify", "--flux-cube", str(tmp_path / "Kurucz" / "flux_cube.bin"),
                    "--output", str(tmp_path / "dense" / "flux_cube.bin"), "--ml-model", str(model)) == 1
    assert capsys.readouterr().err == f"Error: missing torch: {FULL}\n"
    assert not (tmp_path / "dense").exists()


def _menu_ml_flows_in(cli, tmp_path, monkeypatch):
    """Point menu options 8, 9 and 10 at ``tmp_path``: they create their grid and models folders."""
    flows = {name: getattr(cli, name) for name in ("run_ml_completer_flow", "run_ml_generator_flow")}
    for name, flow in flows.items():
        monkeypatch.setattr(cli, name, lambda flow=flow: flow(str(tmp_path), str(tmp_path / "models")))
    densifier = cli.run_grid_densifier_flow
    monkeypatch.setattr(cli, "run_grid_densifier_flow", lambda: densifier(base_dir=str(tmp_path)))


def test_menu_ml_options_open_and_refuse_only_the_action_that_needs_a_package(monkeypatch, capsys, tmp_path):
    from sed_tools import cli

    _without(monkeypatch, *INSTALLS["none"])
    model = _densifier_inputs(tmp_path)
    _menu_ml_flows_in(cli, tmp_path, monkeypatch)
    # 8 and 9 open and go back with 0, as in 0.4.1; their training is refused when chosen; 10 is refused
    # once an ML model is named, before the output question
    answer(monkeypatch, "8", "0", "9", "0", "8", "1", "9", "1", "10", "1", "", str(model), "q")
    assert cli._run_menu() == 1  # the refused actions failed, so the session ends with status 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [f"Error: missing torch, scikit-learn: {FULL}"] * 2 + [
        f"Error: missing torch: {FULL}"]
    assert "Traceback" not in captured.out + captured.err
    assert captured.out.count("0/q) Quit") == 6  # shown first, then again after each of the 5 options
    assert not (tmp_path / "Kurucz_dense").exists()


def test_menu_9_opens_and_returns_without_torch(monkeypatch, capsys, tmp_path):
    from sed_tools import cli

    _without(monkeypatch, *INSTALLS["none"])
    _menu_ml_flows_in(cli, tmp_path, monkeypatch)
    answer(monkeypatch, "9", "0", "q")
    assert cli._run_menu() == 0  # final test, check 16: as in 0.4.1
    assert capsys.readouterr().err == ""


def test_combine_without_scipy_says_what_to_install(monkeypatch, capsys):
    from sed_tools import cli

    def needs_scipy():
        from scipy.spatial import KDTree  # noqa: F401  (as combine_stellar_atm does)

    _without(monkeypatch, "scipy")
    assert cli._run_menu_flow(needs_scipy) is False
    assert capsys.readouterr().err == f"Error: missing scipy: {FULL}\n"


def test_a_missing_module_that_is_not_optional_keeps_its_traceback():
    from sed_tools import cli

    def broken():
        raise ModuleNotFoundError("No module named 'sed_tools_typo'", name="sed_tools_typo")

    with pytest.raises(ModuleNotFoundError):
        cli._run_menu_flow(broken)


def test_import_sed_tools_needs_no_optional_package(tmp_path):
    import os
    import site
    import subprocess
    import sys
    from pathlib import Path

    optional = ("scipy", "torch", "sklearn", "optuna")
    code = ("import sys\n"
            f"blocked = {optional!r}\n"
            "if sys.argv[1] == 'blocked':\n"
            "    for name in blocked: sys.modules[name] = None\n"
            "import sed_tools, sed_tools.cli, sed_tools.ml, sed_tools.ml_sed_generator, sed_tools.grid_densifier\n"
            "import sed_tools.ml_sed_completer\n"
            "import sed_tools.combine_stellar_atm\n"
            "print(sorted(name for name in blocked if sys.modules.get(name) is not None))\n")
    import matplotlib

    # HOME keeps the user's ~/.sed_tools out of reach; the packages and matplotlib's font cache stay where they are
    env = dict(os.environ, HOME=str(tmp_path), PYTHONUSERBASE=site.getuserbase(),
               MPLCONFIGDIR=matplotlib.get_configdir())
    for mode in ("blocked", "installed"):
        done = subprocess.run([sys.executable, "-c", code, mode], cwd=Path(__file__).resolve().parents[1],
                              env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "[]"


# ----------------------------------------------------------------------
# filters-info before anything is installed and with no source answering:
# the detector is not known, never the energy default
# ----------------------------------------------------------------------
@pytest.fixture
def bundled_dr2(monkeypatch):
    from sed_tools import filter_library
    from sed_tools.filter_library import instrument_catalogue

    gaia = {"DR2": {"svo_instruments": ["GAIA/GAIA2"], "bands": ["BP", "G", "RP"],
                    "svo_bands": {"GAIA/GAIA2": ["G", "Gbp=BP", "Grp=RP"]}}}
    monkeypatch.setattr(filter_library, "load_instrument_catalogue",
                        lambda: instrument_catalogue({"GAIA": {"instruments": gaia}}))


@pytest.mark.parametrize("builder", [True, False])
def test_bundled_availability_leaves_the_detector_unknown(builder, tmp_path, bundled_dr2, monkeypatch, capsys):
    from types import SimpleNamespace

    from sed_tools import filter_library
    from sed_tools.cli import print_filter_availability
    from sed_tools.filter_library import FilterLibrary
    from test_release_library import svo_stub

    if not builder:
        monkeypatch.delitem(filter_library.BUILDERS_FOR, ("GAIA", "DR2"))
    library = FilterLibrary(tmp_path / "filters")
    library.mirror = SimpleNamespace(is_available=lambda: False, fetch_manifest=lambda f, i: None,
                                     discover_filters=lambda f, i: [])
    library.svo = svo_stub({}, {}, down=True)
    info = library.available("GAIA", "DR2")
    assert info["availability_source"] == ("original" if builder else "unknown")
    # GAIA/DR2's builder writes photon-counting curves; this used to say energy, "unverified"
    assert info["detector_type"] is None
    assert info["detector_type_sources"] == {"BP": "unknown", "G": "unknown", "RP": "unknown"}
    print_filter_availability(info)
    out = capsys.readouterr().out
    assert "  Detector                 : not known until the curves are installed\n" in out
    assert "energy" not in out and "unverified" not in out


# ----------------------------------------------------------------------
# filters-info: NJM's archived copies (*_svo, *_old) stay apart from the set,
# also beside an installed copy whose band names differ from NJM's
# ----------------------------------------------------------------------
HFI_INFO = {"bands": {"100": {"Description": "HFI 100"}, "100_svo": {"Description": "PLANCK HFI 100GH"},
                      "857": {"Description": "HFI 857"}, "857_svo": {"Description": "PLANCK HFI 857GHz"}}}


def _hfi_catalogue(tmp_path, *, njm_index_file=True, offline=False):
    """A catalogue client for NJM's Planck/HFI beside 0.4.1's PLANCK/HFI from SVO (bands 100GHz, 857GHz)."""
    import json
    from types import SimpleNamespace

    from sed_tools.filter_catalogue import FilterCatalogue
    from test_release_library import Served, njm_index

    installed = tmp_path / "filters" / "PLANCK" / "HFI"
    installed.mkdir(parents=True)
    for band in ("100GHz", "857GHz"):
        (installed / f"{band}.dat").write_text("Wavelength,Transmission\n1,1\n2,1\n")
    (installed / "HFI").write_text("100GHz.dat\n857GHz.dat\n")
    files = {"index.json": njm_index({"Planck": {"HFI": {
                 "filters": ["100", "100_svo", "857", "857_svo"], "properties": "properties.json",
                 "info": "info.json"}}}),
             "filters/Planck/HFI/info.json": json.dumps(HFI_INFO),
             "filters/Planck/HFI/properties.json": json.dumps(HFI_INFO)}
    if njm_index_file:
        files["filters/Planck/HFI/HFI"] = "100.dat\n857.dat\n"
    server = Served(files)
    mirror = SimpleNamespace(base_url="https://nillmill.ddns.net/sed_tools", is_available=lambda: True,
                             _get=lambda url, **kwargs: server.get(url, **kwargs))
    return FilterCatalogue(tmp_path / "filters", mirror=mirror, offline=offline)


@pytest.mark.parametrize("spelling", [("Planck", "HFI"), ("PLANCK", "HFI")])
def test_archived_copies_stay_apart_beside_an_installed_copy_under_other_band_names(spelling, tmp_path):
    catalogue = _hfi_catalogue(tmp_path)
    for kind in ("info", "properties"):
        data = getattr(catalogue, kind)(*spelling)
        assert list(data["bands"]) == ["100", "857"]  # the *_svo rows used to be listed with them
        assert list(data["archived_bands"]) == ["100_svo", "857_svo"]


def test_bundled_bands_decide_when_njm_has_no_index_file(tmp_path, monkeypatch):
    from sed_tools import filter_library
    from sed_tools.filter_library import instrument_catalogue

    monkeypatch.setattr(filter_library, "load_instrument_catalogue", lambda: instrument_catalogue(
        {"Planck": {"instruments": {"HFI": {"bands": ["100", "857"]}}}}))
    data = _hfi_catalogue(tmp_path, njm_index_file=False).info("Planck", "HFI")
    assert list(data["bands"]) == ["100", "857"] and list(data["archived_bands"]) == ["100_svo", "857_svo"]


def test_bands_no_list_names_are_left_as_they_are(tmp_path, monkeypatch):
    from sed_tools import filter_library
    from sed_tools.filter_library import instrument_catalogue

    monkeypatch.setattr(filter_library, "load_instrument_catalogue", lambda: instrument_catalogue({}))
    data = _hfi_catalogue(tmp_path, njm_index_file=False).info("Planck", "HFI")
    assert data == HFI_INFO  # no set to tell the archived copies by: nothing is hidden


def test_bands_of_an_installed_copy_under_njm_names_still_decide(tmp_path):
    catalogue = _hfi_catalogue(tmp_path)
    (tmp_path / "filters" / "PLANCK" / "HFI" / "HFI").write_text("100.dat\n")  # an installed set of one band
    data = catalogue.info("Planck", "HFI")
    assert list(data["bands"]) == ["100"] and list(data["archived_bands"]) == ["100_svo", "857", "857_svo"]
