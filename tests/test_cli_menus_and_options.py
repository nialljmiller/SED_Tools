"""The CLI's messages, menus and options: messages instead of tracebacks, --flux-unit,
variants, atmospheres, the pickers, output width, usage errors, combine and the newer options.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sed_tools import cli
from sed_tools.api import Filters, SED
from sed_tools.filter_components import ComponentError

from helpers_cli_menus import _Stdin, _availability, _summary_inputs, answer
from helpers_cli_menus import catalogue, gaia, narrow_terminal  # noqa: F401 (fixtures)


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    return cli.main()


# ----------------------------------------------------------------------
# messages, not tracebacks; the menu survives a failed workflow
# ----------------------------------------------------------------------
def test_debug_variable_shows_the_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SED_TOOLS_DEBUG", "1")
    assert run_cli(monkeypatch, "rebuild", "--base", str(tmp_path)) == 1
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err and err.endswith("No local spectral grids found under "
                                                                         f"{tmp_path}\n")


def test_main_explains_input_that_ended(monkeypatch, capsys):
    def ask():
        raise EOFError

    monkeypatch.setattr(cli, "run_rebuild_flow", lambda **kwargs: ask())
    assert run_cli(monkeypatch, "rebuild") == 1
    assert "input ended before a question was answered" in capsys.readouterr().err


@pytest.fixture
def menu_flows(monkeypatch):
    monkeypatch.setattr(cli, "_print_banner", lambda: None)
    monkeypatch.setattr(cli, "menu", lambda: _menu_answers.pop(0))


_menu_answers = []


def test_menu_reports_a_failed_workflow_and_shows_the_menu_again(menu_flows, monkeypatch, capsys):
    def rebuild():
        raise FileNotFoundError("No local spectral grids found under /x")

    ran = []
    monkeypatch.setattr(cli, "run_rebuild_flow", rebuild)
    monkeypatch.setattr(cli, "run_config_flow", lambda: ran.append("config"))
    _menu_answers[:] = ["rebuild", "config", "quit"]
    assert cli._run_menu() == 1
    assert ran == ["config"]
    assert "sed-tools: error: No local spectral grids found under /x" in capsys.readouterr().err


@pytest.mark.parametrize("exit_code,printed,status", [("Not a variant grid", "Not a variant grid\n", 1),
                                                      (None, "", 0)])
def test_menu_survives_a_workflow_that_exits(menu_flows, monkeypatch, capsys, exit_code, printed, status):
    def prepare():
        raise SystemExit(exit_code)

    monkeypatch.setattr(cli, "run_mesa_prepare_flow", prepare)
    _menu_answers[:] = ["mesa_prepare", "quit"]
    assert cli._run_menu() == status
    assert capsys.readouterr().err == printed


def test_menu_returns_after_ctrl_c_in_a_workflow(menu_flows, monkeypatch, capsys):
    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_coverage_flow", interrupted)
    _menu_answers[:] = ["coverage", "quit"]
    assert cli._run_menu() == 0
    assert "Interrupted; back to the menu." in capsys.readouterr().out


@pytest.mark.parametrize("typed", ["x"])
def test_menu_says_unknown_option_and_asks_again(monkeypatch, capsys, typed):
    monkeypatch.setattr(cli, "_print_banner", lambda: None)
    answer(monkeypatch, typed, "q")
    assert cli._run_menu() == 0
    assert capsys.readouterr().out.count("Unknown option: enter a number from the menu, or q to quit.") == 1


def test_menu_quits_when_input_ends(monkeypatch):
    monkeypatch.setattr(cli, "_print_banner", lambda: None)

    def closed(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)
    assert cli._run_menu() == 0


# ----------------------------------------------------------------------
# --flux-unit, and no default answer to the flux-unit question
# ----------------------------------------------------------------------
def _undeclared_grid(directory: Path) -> Path:
    directory.mkdir(parents=True)
    for i, teff in enumerate((3500, 3600, 3700)):
        (directory / f"s{i}.txt").write_text(f"# teff = {teff}\n# logg = 4.5\n# meta = 0.0\n"
                                            "4000 1e-3\n5000 2e-3\n6000 3e-3\n")
    return directory


def test_rebuild_uses_the_flux_unit_option_without_asking(tmp_path, monkeypatch, capsys):
    _undeclared_grid(tmp_path / "Cool")
    cleaned = []
    monkeypatch.setattr("sys.stdin", _Stdin(False))
    monkeypatch.setattr(cli.grids, "clean_model_dir", lambda d, **k: cleaned.append(k["flux_unit"]) or
                        {"total": 3, "error": [], "skipped_invalid": [], "skipped_index": []})
    monkeypatch.setattr(cli.grids, "regenerate_lookup_table", lambda d: None)
    assert run_cli(monkeypatch, "rebuild", "--base", str(tmp_path), "--grids", "Cool", "--no-h5", "--no-cube",
                   "--flux-unit", "flam") == 0
    assert cleaned == ["flam"]


# ----------------------------------------------------------------------
# variants
# ----------------------------------------------------------------------
@pytest.fixture
def composed(monkeypatch):
    calls = []
    monkeypatch.setattr(Filters, "compose", classmethod(lambda cls, *a, **k: calls.append((a, k))))
    monkeypatch.setattr(Filters, "fetch", classmethod(lambda cls, *a, **k: calls.append((a, k)) or "dir"))
    return calls


ARGUS_VARIANTS = {"filter": {"production": {"description": "Production filters", "default": True},
                             "addem": {"description": "ADDem filters"}}}


def test_availability_lists_variants_with_the_default_marked(capsys):
    cli.print_filter_availability(_availability(variants=ARGUS_VARIANTS))
    out = " ".join(capsys.readouterr().out.split())
    assert "Variants (API only) : filter: production (default), addem" in out
    assert "--variant" not in out


# ----------------------------------------------------------------------
# Atmospheres and the composition prompt
# ----------------------------------------------------------------------
def test_availability_lists_every_atmosphere_site(capsys):
    cli.print_filter_availability(_availability(atmosphere=True, atmosphere_sites=["hopkins_2mass", "ctio_2mass"]))
    out = " ".join(capsys.readouterr().out.split())
    assert ("Atmosphere : can be added at its own sites: hopkins_2mass, ctio_2mass; no default, "
            "so choose one with --atmosphere-site (in Python, Filters.atmospheres() lists the conditions)") in out
    assert "None" not in out


@pytest.mark.parametrize("grid,typed,airmass,pwv", [
    ({"airmass": [], "pwv": [], "unstated_airmass": True}, [""], None, None),
    ({"airmass": [1.0, 1.5], "pwv": [2.5, 5.0], "unstated_airmass": False}, ["1.5", "5"], 1.5, 5.0),
])
def test_airmass_defaults_to_the_first_condition_the_site_has(monkeypatch, capsys, grid, typed, airmass, pwv):
    prompts = answer(monkeypatch, *typed)
    assert cli._ask_conditions("site", grid) == (airmass, pwv)
    shown = "unstated" if not grid["airmass"] else f"{grid['airmass'][0]:g}"
    assert prompts[0] == f"Airmass [{shown}]: "


def test_composition_asks_again_after_a_bad_recipe_or_detector_type(tmp_path, monkeypatch, composed, capsys):
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: _availability()))
    answer(monkeypatch, "y", "bogus", "9", "filter,atmosphere", "published,optics", "xyz", "photon")
    cli._offer_composition(tmp_path / "Fac" / "Cam")
    out = capsys.readouterr().out
    assert "Unknown filter component: 'bogus'" in out and "Choose 1 to 2" in out
    assert "the atmosphere is asked about separately" in out and "Please enter one of: inherit, energy, photon" in out
    assert composed[0][1]["components"] == ["published", "optics"] and composed[0][1]["detector_type"] == "photon"


def test_composable_bands_compose_in_memory(tmp_path, monkeypatch):
    import sed_tools.filter_components as fc

    manifest = SimpleNamespace(primary_bands=lambda: ["J", "Z"])
    monkeypatch.setattr(fc.InstrumentManifest, "load", classmethod(lambda cls, d: manifest))

    def compose_band(directory, band, components, **kwargs):
        if band == "Z":
            raise ComponentError("covers 9000-56000 A")

    monkeypatch.setattr(fc, "compose_band", compose_band)
    working, failed = cli._composable_bands(tmp_path / "Paranal" / "VISTA",
                                            {"components": ["filter"], "atmosphere": None, "variants": None})
    assert working == ["J"] and failed == {"Z": "covers 9000-56000 A"}


def test_several_sites_without_a_default_name_the_cli_option(tmp_path, monkeypatch, capsys):
    def compose(cls, *a, **k):
        raise ComponentError("has several atmosphere sites; choose one with atmosphere={'site': ..., 'airmass': ...}")

    monkeypatch.setattr(Filters, "compose", classmethod(compose))
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: _availability(
        atmosphere=True, atmosphere_sites=["hopkins_2mass", "ctio_2mass"])))
    monkeypatch.setattr(Filters, "atmospheres", classmethod(lambda cls, *a, **k: {}))
    assert run_cli(monkeypatch, "filters-compose", "2MASS/2MASS", "--base", str(tmp_path), "--airmass",
                   "unstated") == 1
    out = capsys.readouterr().out
    assert ("2MASS/2MASS has several atmosphere sites (hopkins_2mass, ctio_2mass) and no default; "
            "choose one with --atmosphere-site") in out
    assert "atmosphere={" not in out


def test_atmosphere_component_without_airmass_is_a_usage_error(monkeypatch, capsys, composed):
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch, "filters-compose", "A/B", "--components", "filter,atmosphere")
    assert stop.value.code == 2
    assert "--components atmosphere needs --airmass" in capsys.readouterr().err


SITES = {
    "paranal": {"airmass": [1.0, 1.5], "pwv": [2.5, 5.0], "unstated_airmass": False, "kind": "derived",
                "source": "ESO SkyCalc sky model (Noll et al. 2012)"},
    "paranal_vircam": {"airmass": [1.0, 2.0], "pwv": [1.0, 3.0], "unstated_airmass": False, "kind": "team",
                       "source": "Model atmospheres distributed with ESO's VIRCAM curves"},
    "lsst_std": {"airmass": [1.0, 1.2], "pwv": [], "unstated_airmass": False,
                 "source": "Rubin fiducial atmosphere (MODTRAN, with aerosols), lsst/throughputs v1.9.2"}}


@pytest.fixture
def sites(monkeypatch):
    """Filters.atmospheres from the cache, as SITES; returns the include_remote of each call."""
    calls = []
    monkeypatch.setattr(Filters, "atmospheres", classmethod(
        lambda cls, filter_root=None, include_remote=True: calls.append(include_remote) or SITES))
    return calls


def test_pressing_enter_at_the_atmosphere_choice_folds_in_none(tmp_path, monkeypatch, composed, sites, capsys):
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: _availability(
        atmosphere=True, atmosphere_site="lsst_std", atmosphere_sites=["lsst_std"])))
    prompts = answer(monkeypatch, "y", "1", "", "")
    cli._offer_composition(tmp_path / "LSST" / "LSST")
    assert [p for p in prompts if p.startswith("Atmosphere")] == ["Atmosphere [0, none]: "]
    assert composed[0][1]["atmosphere"] is None and sites == [True]  # one listing; installed sites are checked daily
    out = capsys.readouterr().out
    assert ("Atmosphere to fold in:\n"
            "  0. none                     the curve without atmosphere\n"
            "  1. lsst_std (default site)  Rubin fiducial atmosphere (MODTRAN, with aerosols…\n"
            ) in out


# ----------------------------------------------------------------------
# The picker: 0.4.1's numbered columns of names
# ----------------------------------------------------------------------
def test_the_description_block_names_in_full_and_cuts_descriptions_at_the_edge(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "40")
    cli._print_instrument_descriptions("Cameras", [
        {"key": "PointGrey_Grasshopper2_14S5C", "description": "Point Grey Grasshopper2 14S5C colour camera"},
        {"key": "SQM", "description": ""}])
    assert capsys.readouterr().out.splitlines() == [
        "", "Cameras: 2 instruments",
        "  PointGrey_Grasshopper2_14S5C  Point G…",
        "  SQM                           no desc…"]
    cli._print_instrument_descriptions("Solo", [{"key": "A" * 40, "description": "text"}])
    assert capsys.readouterr().out.splitlines() == ["", "Solo: 1 instrument", "  " + "A" * 40]


def test_picking_an_instrument_shows_its_summary_before_the_download_question(gaia, tmp_path, monkeypatch, capsys):
    info = _availability(facility="GAIA", instrument="DR2", bands=["BP", "G", "RP"], published="filter+optics+qe",
                         components=[], buildable=[], platform="space",
                         component_status="ESA publishes only total passbands (Riello et al. 2021).")
    catalogue = {"info": {"licence": LICENCE, "references": ["a", "b"]}, "properties": None, "errors": []}
    looked_up = _summary_inputs(monkeypatch, info, catalogue)
    shown = []
    monkeypatch.setattr("builtins.input", lambda prompt="": shown.append((prompt, capsys.readouterr().out))
                        or ["1", "1", "y", "n", "q"][len(shown) - 1])
    assert cli.run_filters_flow(str(tmp_path)) == 0
    prompt, summary = shown[2]  # what was printed between picking DR2 and the download question
    assert prompt == "Download GAIA/DR2? [Y/n] "
    assert summary.splitlines() == [
        "",
        "GAIA/DR2  (3 bands)",
        "  Gaia DR2 passbands (Evans et al. 2018)",
        "  From       : NJM, then the original source",
        "  Note       : ESA publishes only total passbands (Riello et al. 2021).",
        "  Bands      : BP, G, RP",
        "  Components : filter+optics+qe in each curve",
        "  Separate   : none identified",
        "  Atmosphere : not applicable (space-based)",
        "  Licence    : attribution. MIT License Copyright (c) 2026 The University. http…",
        "  Cite       : 2 references; in Python: Filters.cite('GAIA/DR2')"]
    assert looked_up == ["GAIA/DR2"] and gaia == ["GAIA/DR2"]


def test_declining_the_download_goes_back_to_the_instrument_list(gaia, tmp_path, monkeypatch, capsys):
    _summary_inputs(monkeypatch)
    prompts = answer(monkeypatch, "1", "2", "n", "b", "q")
    assert cli.run_filters_flow(str(tmp_path)) == 0
    assert prompts == ["> ", "> ", "Download GAIA/DR2r? [Y/n] ", "> ", "> "]
    assert gaia == []
    assert capsys.readouterr().out.count("Instruments for GAIA") == 2


def test_the_summary_fits_80_columns_in_about_a_dozen_lines():
    info = _availability(bands=[f"F{n}W" for n in range(100, 700, 10)], variants=ARGUS_VARIANTS,
                         notes=["no component manifest: NJM serves only the published curves, described here "
                                "from their headers"], atmosphere=True, atmosphere_site="paranal",
                         atmosphere_sites=["paranal", "lasilla"], published="filter+instrument+atmosphere")
    catalogue = {"info": {"licence": LICENCE, "references": ["a"], "description": "Long. Text."},
                 "properties": None, "errors": []}
    lines = cli._pick_summary("Fac/Cam", {"description": "word " * 30}, info, catalogue, "SVO")
    assert all(len(line) <= 80 for line in lines) and len(lines) <= 13
    flat = "\n".join(lines)
    assert "Fac/Cam  (60 bands)" in flat and "… (" in flat and "more)" in flat
    assert "Atmosphere : own sites: paranal (default), lasilla" in flat
    assert "Variants   : API only; filter: production (default), addem" in flat
    assert "filters-cite" not in flat and "--variant" not in flat


def test_the_summary_says_where_a_download_comes_from():
    up = SimpleNamespace(mirror=SimpleNamespace(is_available=lambda: True), root=Path("/data/filters"))
    down = SimpleNamespace(mirror=SimpleNamespace(is_available=lambda: False))
    entry = {"sources": ["svo", "local", "njm", "original"]}
    assert cli._download_origin(up, entry, None) == "NJM, then the original source, then SVO"
    assert cli._download_origin(down, entry, None) == "the original source, then SVO; NJM is unreachable"
    assert cli._download_origin(up, {"sources": ["local"]}, None) == "only the installed copy, which is incomplete"
    assert cli._download_origin(up, entry, up.root / "GAIA" / "DR2") == "installed in /data/filters/GAIA/DR2"
    up.root = Path("/a/data/directory/long/enough/that/its/path/does/not/fit/beside/the/name/filters")
    origin = cli._download_origin(up, entry, up.root / "GAIA" / "DR2")
    assert origin == "installed in the filter directory, as GAIA/DR2"
    lines = cli._pick_summary("GAIA/DR2", {}, _availability(), {"info": None, "properties": None, "errors": []},
                              origin)
    assert "  From       : installed in the filter directory, as GAIA/DR2" in lines
    assert all(len(line) <= 80 for line in lines)


def test_downloads_use_the_fetch_key(catalogue, tmp_path, monkeypatch):
    answer(monkeypatch, "1", "1", "y", "n", "q")
    assert cli.run_filters_flow(str(tmp_path)) == 0
    assert catalogue == ["Gaia/DR2"]


# ----------------------------------------------------------------------
# Menu entries 13 and 14: filter set information and atmosphere models
# ----------------------------------------------------------------------
def test_menu_3_shows_filters_info_for_a_picked_instrument(gaia, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_print_banner", lambda: None)
    looked_up = _summary_inputs(monkeypatch, _availability(facility="GAIA", instrument="GAIA0"), {
        "info": {"licence": LICENCE, "references": ["a"]}, "properties": None, "errors": []})
    prompts = answer(monkeypatch, "3", "1", "4", "n", "q", "q")
    assert cli._run_menu() == 0
    out = capsys.readouterr().out
    assert "\nGAIA/GAIA0  (2 bands)\n  Published curves contain" in out
    assert "Cite                     : 1 references; in Python: Filters.cite('GAIA/GAIA0')" in out
    assert looked_up == ["GAIA/GAIA0"] and gaia == []  # nothing downloaded
    assert prompts[4] == "> "  # back at the facility picker, then the menu


@pytest.fixture
def lsst(tmp_path, monkeypatch):
    """Facility LSST with instrument LSST, whose curves download to tmp_path/LSST/LSST; returns the compose calls."""
    from sed_tools.filter_library import FilterLibrary

    monkeypatch.setattr(cli, "_print_banner", lambda: None)
    monkeypatch.setattr(FilterLibrary, "facilities", lambda self: {"entries": [
        {"key": "LSST", "label": "LSST", "sources": ["njm"]}], "errors": []})
    monkeypatch.setattr(FilterLibrary, "instruments", lambda self, facility: {"entries": [
        {"key": "LSST", "label": "LSST", "sources": ["njm"]}], "errors": []})
    monkeypatch.setattr(FilterLibrary, "published", lambda self, f, i: tmp_path / f / i)
    composed = []
    monkeypatch.setattr(Filters, "compose", classmethod(lambda cls, *a, **k: composed.append((a, k))))
    return composed


def test_menu_4_builds_a_curve_from_the_parts_of_a_picked_instrument(lsst, tmp_path, monkeypatch, capsys):
    parts = _availability(facility="LSST", instrument="LSST", buildable=["filter", "filter+optics+qe"],
                          components=["filter", "optics", "qe"])
    _summary_inputs(monkeypatch, parts)
    monkeypatch.setattr(cli.filters, "_installed_availability", lambda directory: parts)
    prompts = answer(monkeypatch, "4", "1", "1", "2", "", "q")
    assert cli._run_menu() == 0
    assert ("  3) Filter set information\n"
            "  4) Build a filter curve from its parts (filter, optics, camera, atmosphere)\n") in capsys.readouterr().out
    assert prompts[3:5] == ["Choose a recipe [1]: ", "Detector convention [inherit], energy or photon: "]
    assert lsst == [((tmp_path / "LSST" / "LSST",), {"filter_root": tmp_path, "components": ["filter", "optics", "qe"],
                                                     "detector_type": None, "allow_heavy": False, "atmosphere": None})]


def test_menu_4_says_in_one_line_when_an_instrument_has_no_parts(lsst, monkeypatch, capsys):
    _summary_inputs(monkeypatch, _availability(facility="LSST", instrument="LSST", buildable=[], components=[],
                                               component_availability={"status": "total-only", "builder": None}))
    monkeypatch.setattr(cli.filters, "_installed_availability", lambda directory: pytest.fail("nothing is downloaded"))
    answer(monkeypatch, "4", "1", "1", "q")
    assert cli._run_menu() == 0
    out = capsys.readouterr().out
    assert "\nLSST/LSST has no separate parts to build a filter curve from.\n" in out and lsst == []


# ----------------------------------------------------------------------
# Output width
# ----------------------------------------------------------------------
LICENCE = {"redistribution": "attribution",
           "terms": "MIT License Copyright (c) 2026 The University. Permission is hereby granted, free of charge.",
           "url": "https://github.com/argus-hdps/argus_sim/blob/main/LICENSE-file-with-hyphens"}


def test_band_descriptions_wrap_to_the_terminal(capsys):
    props = {"G": {"WavelengthPivot": 6217.6, "FWHM": 4548.2, "DetectorType": "photon"}}
    info = {"bands": {"G": {"MagSys": "Vega", "Description": "GAIA Gbp filter, DR2 Maíz Apellániz & Weiler "
                                                            "revision (faint, G>10.87)"}}}
    cli.print_catalogue_entry("GAIA/GAIA", {"info": info, "properties": {"bands": props}, "errors": []})
    lines = capsys.readouterr().out.splitlines()
    assert all(len(line) <= 80 for line in lines) and len(lines) > 2


# ----------------------------------------------------------------------
# Usage errors and exit codes
# ----------------------------------------------------------------------
def test_usage_errors_show_the_subcommand_usage(monkeypatch, capsys):
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch, "filters-info", "GAIA")
    err = capsys.readouterr().err
    assert stop.value.code == 2 and err.startswith("usage: sed-tools filters-info")
    assert "sed-tools filters-info: error: target must be Facility/Instrument" in err


@pytest.mark.parametrize("argv,message", [
    (["filters-compose", "A/B", "--airmass", "high"], "expected a number or 'unstated', got 'high'"),
    (["ml_generator", "--auto"], "--auto needs --library and --output"),
    (["ml_generator", "--trials", "3"], "--trials and --epochs apply to --auto training only"),
    (["ml_generator", "--auto", "--trials", "0"], "expected a whole number of at least 1, got '0'"),
    (["grids", "--teff", "8000", "3500"], "--teff LOW HIGH: 8000 is above 3500"),
])
def test_usage_errors_exit_2_in_plain_words(monkeypatch, capsys, argv, message):
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch, *argv)
    err = capsys.readouterr().err
    assert stop.value.code == 2 and message in err and "_bound" not in err


# ----------------------------------------------------------------------
# combine
# ----------------------------------------------------------------------
def _grid(root: Path, name: str, combined: bool = False) -> None:
    (root / name).mkdir(parents=True)
    columns = "#file_name,teff,logg,meta" + (",source_model,original_file" if combined else "")
    (root / name / "lookup_table.csv").write_text(columns + "\n")


@pytest.fixture
def combine(tmp_path, monkeypatch):
    _grid(tmp_path, "Kurucz")
    _grid(tmp_path, "Phoenix")
    _grid(tmp_path, "combined_models", combined=True)
    _grid(tmp_path, "older_mix", combined=True)
    built = []
    monkeypatch.setattr("sed_tools.cli.grids._build_combined_grid", lambda models, out: built.append(
        ([name for name, _ in models], Path(out).name)))
    return built


def test_non_interactive_combine_leaves_out_earlier_combined_grids(tmp_path, monkeypatch, capsys, combine):
    assert run_cli(monkeypatch, "combine", "--base", str(tmp_path), "--non-interactive") == 0
    assert [(sorted(names), out) for names, out in combine] == [(["Kurucz", "Phoenix"], "combined_models")]
    out = capsys.readouterr().out
    assert "Leaving out combined_models: it is the output grid" in out
    assert "Leaving out older_mix: an earlier combined grid (name it with --grids to include it)" in out


@pytest.mark.parametrize("argv,message", [
    (["--grids", "Nope"], "No spectral grid with a lookup table named Nope"),
    (["--grids", "Kurucz", "combined_models"], "The output grid combined_models is also one of the grids"),
])
def test_combine_refuses_unknown_grids_and_combining_into_an_input(tmp_path, monkeypatch, capsys, combine,
                                                                   argv, message):
    assert run_cli(monkeypatch, "combine", "--base", str(tmp_path), *argv) == 1
    assert message in capsys.readouterr().err and combine == []


def test_combine_prompt_has_a_space_and_refuses_an_input_name(tmp_path, monkeypatch, combine):
    monkeypatch.setattr("sed_tools.combine_stellar_atm.select_models_interactive",
                        lambda dirs: [d for d in dirs if d[0] in ("Kurucz", "Phoenix")])
    prompts = answer(monkeypatch, "Kurucz", "", "mine")
    cli.run_combine_flow(base_dir=str(tmp_path))
    assert prompts[0] == "What should the combined spectral grid be called? [combined_models] "
    assert [(sorted(names), out) for names, out in combine] == [(["Kurucz", "Phoenix"], "mine")]


# ----------------------------------------------------------------------
# Stale help and options
# ----------------------------------------------------------------------
@pytest.mark.parametrize("argv,message", [
    (["nosuch"], "unknown source 'nosuch'; choose one of: argus, cassini"),
    (["tabular"], "tabular builds several instruments; choose one with --instrument"),
    (["tabular", "--instrument", "NOPE"], "tabular has no instrument 'NOPE'; choose one of: HAWKI"),
])
def test_build_components_refuses_bad_requests_cleanly(monkeypatch, capsys, argv, message):
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch, "filters-build-components", *argv)
    assert stop.value.code == 2 and message in capsys.readouterr().err


def test_coverage_prints_each_failure_once(tmp_path, monkeypatch, capsys):
    def coverage(cls, name, **kwargs):
        raise ValueError("lookup_table.csv has no Teff column")

    monkeypatch.setattr(SED, "coverage", classmethod(coverage))
    assert run_cli(monkeypatch, "coverage", "--grids", "B", "--no-plot", "--base", str(tmp_path)) == 1
    assert capsys.readouterr().out.count("lookup_table.csv has no Teff column") == 1


# ----------------------------------------------------------------------
# Grid cuts without a terminal, and --auto limits
# ----------------------------------------------------------------------
def test_njm_cuts_are_asked_in_a_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Stdin(True))
    monkeypatch.setattr(cli.grids, "prompt_njm_axis_cuts", lambda name, g, model_url=None: {"teff_range": (1.0, 2.0)})
    assert cli._download_cuts("G", "njm", None, {"model_url": "u"}, {}) == {"teff_range": (1.0, 2.0)}


def test_auto_training_takes_trials_and_epochs(monkeypatch):
    import importlib.util

    import sed_tools.ml

    calls = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    monkeypatch.setattr(sed_tools.ml, "auto_train_generator", lambda **kwargs: calls.append(kwargs))
    assert run_cli(monkeypatch, "ml_generator", "--auto", "--library", "lib", "--output", "out",
                   "--trials", "2", "--epochs", "1") == 0
    assert run_cli(monkeypatch, "ml_generator", "--auto", "--library", "lib", "--output", "out") == 0
    assert calls == [{"library": "lib", "output": "out", "n_trials": 2, "epochs": 1},
                     {"library": "lib", "output": "out"}]


# ----------------------------------------------------------------------
# NJM unreachable: its catalogue is an extra
# ----------------------------------------------------------------------
def test_grids_without_a_terminal_or_names_says_to_use_grids(monkeypatch, tmp_path):
    class Grabber:
        def __init__(self, *args, **kwargs):
            pass

        def discover_models(self):
            return ["bt-cond"]

    monkeypatch.setattr("sed_tools.msg_spectra_grabber.MSGSpectraGrabber", Grabber)
    monkeypatch.setattr("sys.stdin", _Stdin(False))
    with pytest.raises(ValueError, match=r"No terminal to choose from the 1 spectral grids found; name them with "
                                         r"--grids NAME \.\.\. \(or --grids all\)"):
        cli.run_spectra_flow("msg", base_dir=str(tmp_path))


def test_the_picker_finds_an_instrument_installed_under_an_older_spelling(tmp_path):
    # 0.4.1 wrote Gemini/GMOS-N; NJM's name is GMOS_N. The picker must see the installed copy, not offer a second one.
    from sed_tools.filter_library import FilterLibrary

    installed = tmp_path / "Gemini" / "GMOS-N"
    installed.mkdir(parents=True)
    (installed / "g.dat").write_text("Wavelength,Transmission\n4000,0.1\n5000,0.5\n6000,0.1\n")
    (installed / "GMOS-N").write_text("g.dat\n")
    library = FilterLibrary(tmp_path)
    facility, instrument, directory = cli._picked_names(library, {"key": "Gemini"}, {"key": "GMOS_N"})
    assert (facility, instrument) == ("Gemini", "GMOS_N")
    assert directory is None or directory.resolve() == installed.resolve()
    assert library.installed_directory("Gemini", "GMOS_N").resolve() == installed.resolve()


# ----------------------------------------------------------------------
# The catalogue says where its data came from
# ----------------------------------------------------------------------
def test_filters_info_says_when_the_catalogue_is_the_cached_copy(capsys):
    cli.print_catalogue_entry("Fac/Cam", {"info": {"description": "d"}, "properties": None, "errors": [],
                                          "origin": {"source": "cache", "checked": 0.0 + 1761600000,
                                                     "njm_unreachable": True}})
    out = " ".join(capsys.readouterr().out.split())  # the field wraps to the terminal
    assert "Catalogue : from the cached copy, checked 2025-10-27 21:20 UTC; NJM could not be asked" in out


# ---------------------------------------------------------------------------
# SkyCalc: derived grids, the web form, provenance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("argv,expected", [
    (["--airmass", "unstated", "--atmosphere-site", "ctio_2mass"], {"airmass": None, "pwv": None, "site": "ctio_2mass"}),
])
def test_cli_passes_the_atmosphere_site_and_unstated_airmass(tmp_path, monkeypatch, argv, expected):
    from sed_tools import api, cli

    seen = {}
    monkeypatch.setattr(api.Filters, "compose", classmethod(lambda cls, target, **kw: seen.update(kw)))
    monkeypatch.setattr("sys.argv", ["sed-tools", "filters-compose", "2MASS/2MASS", "--base", str(tmp_path),
                                    "--components", "filter", *argv])
    assert cli.main() in (0, None)
    assert seen["atmosphere"] == expected
