"""The 0.5.0 CLI fixes: messages instead of tracebacks, --flux-unit, variants, atmospheres,
the picker, output width, usage errors, combine and the new options (REPORT_CLI P1-P19)."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sed_tools import cli
from sed_tools.api import Filters, SED
from sed_tools.filter_components import ComponentError


class _Stdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def answer(monkeypatch, *answers):
    """Feed input() these answers; return the prompts it was given. Running out is a test failure."""
    queue, prompts = list(answers), []

    def fake_input(prompt=""):
        prompts.append(prompt)
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    return cli.main()


@pytest.fixture(autouse=True)
def narrow_terminal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.delenv("SED_TOOLS_DEBUG", raising=False)


# ----------------------------------------------------------------------
# P1: messages, not tracebacks; the menu survives a failed workflow
# ----------------------------------------------------------------------
def test_main_prints_an_expected_error_without_a_traceback(tmp_path, monkeypatch, capsys):
    assert run_cli(monkeypatch, "rebuild", "--base", str(tmp_path)) == 1
    err = capsys.readouterr().err
    assert err == f"sed-tools: error: No local spectral grids found under {tmp_path}\n"


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


def test_a_bug_still_shows_its_traceback(monkeypatch):
    monkeypatch.setattr(cli, "run_rebuild_flow", lambda **kwargs: {}["missing"])
    with pytest.raises(KeyError):
        run_cli(monkeypatch, "rebuild")


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
                                                      (0, "", 0), (None, "", 0)])
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


@pytest.mark.parametrize("typed", ["99", "", "x"])
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


def test_number_prompts_ask_again(monkeypatch, capsys):
    answer(monkeypatch, "high", "", "1.5")
    assert cli._ask_number("Airmass: ") == 1.5
    out = capsys.readouterr().out
    assert "'high' is not a number; try again." in out and "Please enter a number." in out
    answer(monkeypatch, "")
    assert cli._ask_number("PWV: ", default=None) is None
    answer(monkeypatch, "UNSTATED")
    assert cli._ask_number("Airmass: ", words=("unstated",)) == "unstated"


# ----------------------------------------------------------------------
# P2/P3: --flux-unit, and no default answer to the flux-unit question
# ----------------------------------------------------------------------
def _undeclared_grid(directory: Path) -> Path:
    directory.mkdir(parents=True)
    for i, teff in enumerate((3500, 3600, 3700)):
        (directory / f"s{i}.txt").write_text(f"# teff = {teff}\n# logg = 4.5\n# meta = 0.0\n"
                                            "4000 1e-3\n5000 2e-3\n6000 3e-3\n")
    return directory


def test_flux_unit_choices_come_from_spectra_cleaner():
    from sed_tools.spectra_cleaner import FLUX_UNITS

    parser = cli._build_parser()
    for command in ("grids", "rebuild", "import"):
        action = next(a for a in parser._subparsers._group_actions[0].choices[command]._actions
                      if a.dest == "flux_unit")
        assert action.choices == list(FLUX_UNITS)


def test_rebuild_uses_the_flux_unit_option_without_asking(tmp_path, monkeypatch, capsys):
    _undeclared_grid(tmp_path / "Cool")
    cleaned = []
    monkeypatch.setattr("sys.stdin", _Stdin(False))
    monkeypatch.setattr(cli, "clean_model_dir", lambda d, **k: cleaned.append(k["flux_unit"]) or
                        {"total": 3, "error": [], "skipped_invalid": [], "skipped_index": []})
    monkeypatch.setattr(cli, "regenerate_lookup_table", lambda d: None)
    assert run_cli(monkeypatch, "rebuild", "--base", str(tmp_path), "--grids", "Cool", "--no-h5", "--no-cube",
                   "--flux-unit", "flam") == 0
    assert cleaned == ["flam"]


def test_an_unknown_flux_unit_is_a_usage_error(monkeypatch, capsys):
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch, "rebuild", "--flux-unit", "jansky")
    assert stop.value.code == 2
    assert "usage: sed-tools rebuild" in capsys.readouterr().err


def test_grids_passes_the_flux_unit_and_ranges(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_spectra_flow", lambda **kwargs: calls.append(kwargs))
    assert run_cli(monkeypatch, "grids", "--source", "njm", "--flux-unit", "fnu_jy", "--teff", "3500", "8000",
                   "--meta", "-1", "0.5", "--wavelength", "3000", "10000") == 0
    assert calls[0]["flux_unit"] == "fnu_jy"
    assert (calls[0]["teff_range"], calls[0]["logg_range"], calls[0]["meta_range"], calls[0]["wl_range"]) == (
        (3500.0, 8000.0), None, (-1.0, 0.5), (3000.0, 10000.0))


def test_import_dry_run_reports_the_undeclared_unit(tmp_path, monkeypatch, capsys):
    src = _undeclared_grid(tmp_path / "mine")
    assert run_cli(monkeypatch, "import", "--path", str(src), "--base", str(tmp_path / "models"), "--dry-run") == 0
    out = capsys.readouterr().out
    assert "[import] mine: flux unit not declared; best guess:" in out and "needs --flux-unit" in out
    assert not (tmp_path / "models" / "mine").exists()


def test_import_passes_the_flux_unit_option(tmp_path, monkeypatch):
    src = _undeclared_grid(tmp_path / "mine")
    calls = []
    monkeypatch.setattr("sys.stdin", _Stdin(False))
    monkeypatch.setattr(SED, "import_grid", classmethod(lambda cls, *a, **k: calls.append(k)))
    assert run_cli(monkeypatch, "import", "--path", str(src), "--flux-unit", "flam") == 0
    assert calls[0]["flux_unit"] == "flam"


# ----------------------------------------------------------------------
# P8: variants
# ----------------------------------------------------------------------
@pytest.fixture
def composed(monkeypatch):
    calls = []
    monkeypatch.setattr(Filters, "compose", classmethod(lambda cls, *a, **k: calls.append((a, k))))
    monkeypatch.setattr(Filters, "fetch", classmethod(lambda cls, *a, **k: calls.append((a, k)) or "dir"))
    return calls


@pytest.mark.parametrize("command", ["filters", "filters-compose"])
def test_variants_are_chosen_in_the_api_only(monkeypatch, capsys, composed, command):
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch, command, "Argus/DR280", "--variant", "filter=addem")
    assert stop.value.code == 2 and "unrecognized arguments: --variant filter=addem" in capsys.readouterr().err
    assert composed == []


ARGUS_VARIANTS = {"filter": {"production": {"description": "Production filters", "default": True},
                             "addem": {"description": "ADDem filters"}}}


def _availability(**extra):
    info = {"facility": "Fac", "instrument": "Cam", "bands": ["J", "H"], "published_source": "", "published": "",
            "published_bands": {}, "components": ["filter"], "partial_components": {}, "buildable": ["filter"],
            "component_status": "", "component_availability": {"status": "supported", "builder": None},
            "missing_files": [], "errors": [], "atmosphere": False, "atmosphere_site": None,
            "atmosphere_sites": [], "platform": "ground", "detector_type": "", "detector_type_sources": {},
            "availability_source": "local", "variants": {}}
    info.update(extra)
    return info


def test_availability_lists_variants_with_the_default_marked(capsys):
    cli.print_filter_availability(_availability(variants=ARGUS_VARIANTS))
    out = " ".join(capsys.readouterr().out.split())
    assert "Variants (API only) : filter: production (default), addem" in out
    assert "--variant" not in out


def test_composition_prompt_does_not_ask_for_variants(tmp_path, monkeypatch, composed, capsys):
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: _availability(variants=ARGUS_VARIANTS)))
    prompts = answer(monkeypatch, "y", "1", "")
    cli._offer_composition(tmp_path / "Argus" / "DR280")
    assert composed[0][1]["components"] == ["filter"] and "variants" not in composed[0][1]
    assert "Recipes (or type your own" in capsys.readouterr().out
    assert not any("variant" in prompt for prompt in prompts)


# ----------------------------------------------------------------------
# P9 and P10: atmospheres and the composition prompt
# ----------------------------------------------------------------------
def test_availability_lists_every_atmosphere_site(capsys):
    cli.print_filter_availability(_availability(atmosphere=True, atmosphere_sites=["hopkins_2mass", "ctio_2mass"]))
    out = " ".join(capsys.readouterr().out.split())
    assert ("Atmosphere : can be added at its own sites: hopkins_2mass, ctio_2mass; no default, "
            "so choose one with --atmosphere-site (in Python, Filters.atmospheres() lists the conditions)") in out
    assert "None" not in out


@pytest.mark.parametrize("grid,typed,airmass,pwv", [
    ({"airmass": [], "pwv": [], "unstated_airmass": True}, [""], None, None),
    ({"airmass": [1.0, 1.5], "pwv": [], "unstated_airmass": False}, [""], 1.0, None),
    ({"airmass": [1.0, 1.5], "pwv": [2.5, 5.0], "unstated_airmass": False}, ["", ""], 1.0, 2.5),
    ({"airmass": [1.0, 1.5], "pwv": [2.5, 5.0], "unstated_airmass": False}, ["1.5", "5"], 1.5, 5.0),
])
def test_airmass_defaults_to_the_first_condition_the_site_has(monkeypatch, capsys, grid, typed, airmass, pwv):
    prompts = answer(monkeypatch, *typed)
    assert cli._ask_conditions("site", grid) == (airmass, pwv)
    shown = "unstated" if not grid["airmass"] else f"{grid['airmass'][0]:g}"
    assert prompts[0] == f"Airmass [{shown}]: "


@pytest.mark.parametrize("platform", ["space", "ground"])
def test_composition_does_not_mention_an_atmosphere_that_is_not_offered(tmp_path, monkeypatch, composed, platform):
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: _availability(platform=platform)))
    prompts = answer(monkeypatch, "y", "1", "")
    cli._offer_composition(tmp_path / "Fac" / "Cam")
    assert prompts[0] == "Build a composed version (e.g. filter only)? [y/N] "
    assert not any("atmosphere" in p.lower() for p in prompts)
    assert composed[0][1]["atmosphere"] is None


def test_composition_asks_again_after_a_bad_recipe_or_detector_type(tmp_path, monkeypatch, composed, capsys):
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: _availability()))
    answer(monkeypatch, "y", "bogus", "9", "filter,atmosphere", "published,optics", "xyz", "photon")
    cli._offer_composition(tmp_path / "Fac" / "Cam")
    out = capsys.readouterr().out
    assert "Unknown filter component: 'bogus'" in out and "Choose 1 to 2" in out
    assert "the atmosphere is asked about separately" in out and "Please enter one of: inherit, energy, photon" in out
    assert composed[0][1]["components"] == ["published", "optics"] and composed[0][1]["detector_type"] == "photon"


def test_composition_offers_the_bands_that_work(tmp_path, monkeypatch, capsys):
    calls = []

    def compose(cls, directory, **kwargs):
        calls.append(kwargs.get("bands"))
        if kwargs.get("bands") is None:
            raise ComponentError("am1.00_pwv1.dat covers 9000-56000 A but 70% of the band lies beyond it")

    monkeypatch.setattr(Filters, "compose", classmethod(compose))
    monkeypatch.setattr(cli, "_composable_bands", lambda d, request: (["J", "H"], {"Z": "covers 9000-56000 A"}))
    answer(monkeypatch, "y")
    request = {"components": ["filter"], "variants": None, "detector_type": None, "allow_heavy": False,
               "atmosphere": {"site": "paranal", "airmass": 1.0, "pwv": 1.0}}
    cli._compose_the_bands_that_work(tmp_path / "Paranal" / "VISTA", request, ComponentError("no"))
    assert calls == [["J", "H"]]
    assert "Z: covers 9000-56000 A" in capsys.readouterr().out


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


def test_a_missing_atmosphere_lists_the_conditions_the_site_has(tmp_path, monkeypatch, capsys):
    def compose(cls, *a, **k):
        raise ComponentError("No cached atmosphere for ctio_2mass_am1.2; fetch or generate the requested conditions")

    monkeypatch.setattr(Filters, "compose", classmethod(compose))
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: _availability(
        atmosphere=True, atmosphere_sites=["hopkins_2mass", "ctio_2mass"])))
    monkeypatch.setattr(Filters, "atmospheres", classmethod(lambda cls, *a, **k: {
        "ctio_2mass": {"airmass": [], "pwv": [], "unstated_airmass": True}}))
    assert run_cli(monkeypatch, "filters-compose", "2MASS/2MASS", "--base", str(tmp_path), "--airmass", "1.2",
                   "--atmosphere-site", "ctio_2mass") == 1
    assert "[filters] ctio_2mass has: airmass 'unstated' (as published)" in capsys.readouterr().out


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


def test_an_atmosphere_is_chosen_by_number_generic_sites_last(tmp_path, monkeypatch, composed, sites, capsys):
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: _availability(
        atmosphere=True, atmosphere_site="paranal", atmosphere_sites=["paranal", "paranal_vircam"])))
    answer(monkeypatch, "y", "1", "", "2", "", "")
    cli._offer_composition(tmp_path / "Paranal" / "VISTA")
    assert composed[0][1]["atmosphere"] == {"site": "paranal", "airmass": 1.0, "pwv": 2.5}
    out = capsys.readouterr().out
    assert "  1. paranal_vircam (team)            Model atmospheres" in out
    assert "  2. paranal (default site, derived)  ESO SkyCalc sky model" in out
    assert "  paranal: airmass 1, 1.5; pwv 2.5, 5 mm" in out


def test_an_atmosphere_can_be_named_and_a_site_without_a_grid_is_said_so(tmp_path, monkeypatch, composed, sites,
                                                                            capsys):
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: _availability(
        atmosphere=True, atmosphere_sites=["hopkins_2mass", "lsst_std"])))
    answer(monkeypatch, "y", "1", "", "LSST_STD", "1.2")
    cli._offer_composition(tmp_path / "Fac" / "Cam")
    assert composed[0][1]["atmosphere"] == {"site": "lsst_std", "airmass": 1.2, "pwv": None}
    assert "  1. hopkins_2mass  not downloaded yet" in capsys.readouterr().out
    assert sites == [True]  # one listing, which includes NJM's sites not cached yet


def _summary_inputs(monkeypatch, info=None, catalogue=None):
    """Stub what the picker's summary reads (FilterLibrary.available, NJM's catalogue entry); returns the calls."""
    from sed_tools.filter_library import FilterLibrary

    calls = []
    monkeypatch.setattr(FilterLibrary, "available",
                        lambda self, f, i, remote=True: calls.append(f"{f}/{i}") or (info or _availability()))
    monkeypatch.setattr(cli, "_catalogue_entry", lambda *a, **k: catalogue or {
        "info": None, "properties": None, "errors": ["NJM does not serve it"]})
    return calls


def test_an_installed_instrument_is_not_offered_for_download(tmp_path, monkeypatch, capsys):
    from sed_tools.filter_library import FilterLibrary

    fetched, offered = [], []
    monkeypatch.setattr(FilterLibrary, "instruments", lambda self, facility: {
        "entries": [{"key": "Cam", "label": "Cam", "sources": ["local", "svo"]}], "errors": []})
    monkeypatch.setattr(FilterLibrary, "published_complete", staticmethod(lambda d: True))
    monkeypatch.setattr(FilterLibrary, "installed_directory", lambda self, f, i: tmp_path / f / i)
    monkeypatch.setattr(FilterLibrary, "published", lambda self, f, i: fetched.append((f, i)) or tmp_path)
    monkeypatch.setattr(cli, "_offer_composition", offered.append)
    _summary_inputs(monkeypatch)
    prompts = answer(monkeypatch, "1", "n")
    download = lambda facility, instrument: cli._offer_download(library, facility, instrument, [])
    library = FilterLibrary(tmp_path)
    assert cli._browse_instruments(library, {"key": "Fac", "label": "Fac"}, download)
    assert not any(p.startswith("Download") for p in prompts)
    # The path is too long for 80 columns, so its place in the filter directory is given.
    assert "  From       : installed in the filter directory, as Fac/Cam\n" in capsys.readouterr().out
    assert fetched == [("Fac", "Cam")] and offered == [tmp_path]


# ----------------------------------------------------------------------
# The picker: 0.4.1's numbered columns of names
# ----------------------------------------------------------------------
GAIA_INSTRUMENTS = [
    {"key": name, "label": name, "fetch_key": name, "sources": sources, "description": description}
    for name, sources, description in [
        ("DR2", ["njm", "original"], "Gaia DR2 passbands (Evans et al. 2018)"),
        ("DR2r", ["njm", "original"], "Gaia DR2 revised passbands (Evans et al. 2018)"),
        ("GAIA", ["njm"], "Gaia DR3 passbands + BP bright/faint (Maíz Apellániz & Weiler 2018)"),
        ("GAIA0", ["njm"], "Gaia pre-launch passbands (Jordi et al. 2010)"),
        ("Gaia2m", ["svo"], "Revised passbands for Gaia DR2.")]]


@pytest.fixture
def gaia(tmp_path, monkeypatch):
    from sed_tools.filter_library import FilterLibrary

    monkeypatch.setattr(FilterLibrary, "facilities", lambda self: {"entries": [
        {"key": "GAIA", "label": "GAIA", "fetch_key": "GAIA", "sources": ["njm", "svo"],
         "description": "Gaia mission."}], "errors": []})
    monkeypatch.setattr(FilterLibrary, "instruments", lambda self, facility: {
        "entries": GAIA_INSTRUMENTS, "errors": []})
    monkeypatch.setattr(FilterLibrary, "mirror", SimpleNamespace(is_available=lambda: True))
    fetched = []
    monkeypatch.setattr(FilterLibrary, "published", lambda self, f, i: fetched.append(f"{f}/{i}") or tmp_path)
    monkeypatch.setattr(cli, "_offer_composition", lambda directory: None)
    return fetched


def test_the_pickers_show_names_only(gaia, tmp_path, monkeypatch, capsys):
    answer(monkeypatch, "1", "q")
    assert cli.run_filters_flow(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "   1. GAIA" in out and "Gaia mission" not in out  # no facility descriptions are invented
    pickers = out.split("\nGAIA: 5 instruments\n")[0] + out.split("Instruments for GAIA")[1]
    assert not any(text in pickers for text in ("[njm", "[svo", "Tags show", "Evans et al."))
    assert cli._picker_labels(GAIA_INSTRUMENTS + [{"key": "Other"}]) == [
        "DR2", "DR2r", "GAIA", "GAIA0", "Gaia2m", "Other"]


def test_picking_a_facility_lists_its_instruments_with_their_descriptions_once(gaia, tmp_path, monkeypatch,
                                                                               capsys):
    answer(monkeypatch, "1", "/DR2", "q")
    cli.run_filters_flow(str(tmp_path))
    out = capsys.readouterr().out
    block = out.split("\nGAIA: 5 instruments\n")[1].split("\n\nInstruments for GAIA")[0]
    assert block.splitlines() == [
        "  DR2     Gaia DR2 passbands (Evans et al. 2018)",
        "  DR2r    Gaia DR2 revised passbands (Evans et al. 2018)",
        "  GAIA    Gaia DR3 passbands + BP bright/faint (Maíz Apellániz & Weiler 2018)",
        "  GAIA0   Gaia pre-launch passbands (Jordi et al. 2010)",
        "  Gaia2m  Revised passbands for Gaia DR2."]
    assert out.count("GAIA: 5 instruments") == 1  # not again when the list is filtered


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


@pytest.mark.parametrize("columns", ["80", "120"])
def test_the_instrument_picker_shows_full_names_in_columns(gaia, tmp_path, monkeypatch, capsys, columns):
    monkeypatch.setenv("COLUMNS", columns)
    answer(monkeypatch, "1", "q")
    cli.run_filters_flow(str(tmp_path))
    rows = capsys.readouterr().out.split("Instruments for GAIA\n")[1].splitlines()[1:3]
    assert re.fullmatch(r"\s+1\. DR2\s+3\. GAIA\s+5\. Gaia2m\s*", rows[0])  # 0.4.1 cut these to 'D', 'GA', 'Gaia'
    assert re.fullmatch(r"\s+2\. DR2r\s+4\. GAIA0\s*", rows[1])
    assert all(len(row) <= int(columns) for row in rows)


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


def test_picker_cuts_long_labels_with_an_ellipsis(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "40")
    answer(monkeypatch, "q")
    cli.prompt_choice(["x" * 50], "Things")
    assert "x" * 23 + "…" in capsys.readouterr().out


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    from sed_tools.filter_library import FilterLibrary

    monkeypatch.setattr(FilterLibrary, "facilities", lambda self: {"entries": [
        {"key": "GAIA", "label": "GAIA", "fetch_key": "Gaia", "sources": ["svo"]},
        {"key": "TWOMASS", "label": "2MASS", "sources": ["svo"]}], "errors": []})
    monkeypatch.setattr(FilterLibrary, "instruments", lambda self, facility: {"entries": [
        {"key": "GAIA2", "label": "GAIA2", "fetch_key": "DR2", "sources": ["svo"]},
        {"key": "Cam", "label": "Cam", "sources": ["svo"]}], "errors": []})
    fetched = []
    monkeypatch.setattr(FilterLibrary, "published", lambda self, f, i: fetched.append(f"{f}/{i}") or tmp_path)
    monkeypatch.setattr(cli, "_offer_composition", lambda directory: None)
    _summary_inputs(monkeypatch)
    return fetched


def test_downloads_use_the_fetch_key(catalogue, tmp_path, monkeypatch):
    answer(monkeypatch, "1", "1", "y", "n", "q")
    assert cli.run_filters_flow(str(tmp_path)) == 0
    assert catalogue == ["Gaia/DR2"]


def test_downloading_several_facilities_asks_first_with_the_count(catalogue, tmp_path, monkeypatch, capsys):
    prompts = answer(monkeypatch, "1,2", "n")
    assert cli.run_filters_flow(str(tmp_path)) == 0
    assert catalogue == []
    assert prompts[-1] == "Download all 4? [y/N] "
    assert "4 instruments in 2 facilities: GAIA (2), 2MASS (2)" in capsys.readouterr().out


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
    monkeypatch.setattr(cli, "_installed_availability", lambda directory: parts)
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
    monkeypatch.setattr(cli, "_installed_availability", lambda directory: pytest.fail("nothing is downloaded"))
    answer(monkeypatch, "4", "1", "1", "q")
    assert cli._run_menu() == 0
    out = capsys.readouterr().out
    assert "\nLSST/LSST has no separate parts to build a filter curve from.\n" in out and lsst == []


def test_the_combine_wizard_counts_only_the_curves_combine_takes(tmp_path):
    listed = tmp_path / "NJM" / "Cam"
    listed.mkdir(parents=True)
    for band in ("g", "r", "g_svo", "r_old"):
        (listed / f"{band}.dat").write_text("Wavelength,Transmission\n1,1\n2,1\n")
    (listed / "Cam").write_text("g.dat\nr.dat\n")
    archived = tmp_path / "NJM" / "Archived"
    archived.mkdir()
    for band in ("H", "H_svo"):
        (archived / f"{band}.dat").write_text("Wavelength,Transmission\n1,1\n2,1\n")
    (archived / "manifest.json").write_text('{"bands": ["H"], "archived_bands": {"H_svo": {}}}')
    assert cli._FilterSetOption(listed, tmp_path).label == "NJM/Cam (2 filters)"
    assert cli._FilterSetOption(archived, tmp_path).label == "NJM/Archived (1 filters)"


def test_the_removed_commands_are_usage_errors(monkeypatch, capsys):
    for command in ("filters-search", "filters-cite", "filters-atmospheres"):
        with pytest.raises(SystemExit) as stop:
            run_cli(monkeypatch, command)
        assert stop.value.code == 2 and f"invalid choice: '{command}'" in capsys.readouterr().err


# ----------------------------------------------------------------------
# P12/P13: output width
# ----------------------------------------------------------------------
LICENCE = {"redistribution": "attribution",
           "terms": "MIT License Copyright (c) 2026 The University. Permission is hereby granted, free of charge.",
           "url": "https://github.com/argus-hdps/argus_sim/blob/main/LICENSE-file-with-hyphens"}


def test_filters_info_shows_the_licence_first_sentence_and_url(capsys):
    cli.print_catalogue_entry("Argus/DR280", {"info": {"licence": LICENCE}, "properties": None, "errors": []})
    out = capsys.readouterr().out
    assert all(len(line) <= 80 for line in out.splitlines() if LICENCE["url"] not in line)
    assert LICENCE["url"] in out  # never split, although it is longer than the room left
    flat = " ".join(out.split())
    assert "attribution. MIT License Copyright (c) 2026 The University." in flat
    assert "Permission is hereby" not in flat and "filters-cite" not in flat


def test_band_descriptions_wrap_to_the_terminal(capsys):
    props = {"G": {"WavelengthPivot": 6217.6, "FWHM": 4548.2, "DetectorType": "photon"}}
    info = {"bands": {"G": {"MagSys": "Vega", "Description": "GAIA Gbp filter, DR2 Maíz Apellániz & Weiler "
                                                            "revision (faint, G>10.87)"}}}
    cli.print_catalogue_entry("GAIA/GAIA", {"info": info, "properties": {"bands": props}, "errors": []})
    lines = capsys.readouterr().out.splitlines()
    assert all(len(line) <= 80 for line in lines) and len(lines) > 2


# ----------------------------------------------------------------------
# P14: usage errors and exit codes
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
# P16: combine
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
    monkeypatch.setattr("sed_tools.api._build_combined_grid", lambda models, out: built.append(
        ([name for name, _ in models], Path(out).name)))
    return built


def test_non_interactive_combine_leaves_out_earlier_combined_grids(tmp_path, monkeypatch, capsys, combine):
    assert run_cli(monkeypatch, "combine", "--base", str(tmp_path), "--non-interactive") == 0
    assert [(sorted(names), out) for names, out in combine] == [(["Kurucz", "Phoenix"], "combined_models")]
    out = capsys.readouterr().out
    assert "Leaving out combined_models: it is the output grid" in out
    assert "Leaving out older_mix: an earlier combined grid (name it with --grids to include it)" in out


def test_combine_grids_option_takes_the_named_grids_without_prompts(tmp_path, monkeypatch, combine):
    assert run_cli(monkeypatch, "combine", "--base", str(tmp_path), "--grids", "Phoenix", "older_mix",
                   "--output", "mix2") == 0
    assert combine == [(["Phoenix", "older_mix"], "mix2")]


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


def test_filter_combine_hint_names_menu_option_1(tmp_path, monkeypatch, capsys):
    answer(monkeypatch, "")
    cli.run_filter_combine_flow(base_dir=str(tmp_path))
    assert "Download filters first with option 1" in capsys.readouterr().out


# ----------------------------------------------------------------------
# P17: stale help and options
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


def test_build_components_help_lists_every_registered_source():
    from sed_tools.component_sources import BUILDERS

    help_text = " ".join(cli._build_parser()._subparsers._group_actions[0].choices[
        "filters-build-components"].format_help().split())
    assert all(name in help_text for name in BUILDERS)


def test_coverage_prints_each_failure_once(tmp_path, monkeypatch, capsys):
    def coverage(cls, name, **kwargs):
        raise ValueError("lookup_table.csv has no Teff column")

    monkeypatch.setattr(SED, "coverage", classmethod(coverage))
    assert run_cli(monkeypatch, "coverage", "--grids", "B", "--no-plot", "--base", str(tmp_path)) == 1
    assert capsys.readouterr().out.count("lookup_table.csv has no Teff column") == 1


def test_menu_says_config_without_a_trailing_space(monkeypatch, capsys):
    answer(monkeypatch, "q")
    cli.menu()
    assert "Config (data directory)\n" in capsys.readouterr().out


# ----------------------------------------------------------------------
# P18/P19: grid cuts without a terminal, and --auto limits
# ----------------------------------------------------------------------
def test_given_cuts_go_to_any_source_without_asking(monkeypatch):
    monkeypatch.setattr(cli, "prompt_njm_axis_cuts", lambda *a, **k: pytest.fail("asked"))
    ranges = {"teff_range": (3500.0, 8000.0), "logg_range": None, "meta_range": None, "wl_range": None}
    assert cli._download_cuts("G", "msg", None, {}, ranges) == ranges
    assert cli._download_cuts("G", "njm", None, {}, ranges) == ranges


def test_njm_cuts_are_not_asked_without_a_terminal(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _Stdin(False))
    monkeypatch.setattr(cli, "prompt_njm_axis_cuts", lambda *a, **k: pytest.fail("asked"))
    none = {"teff_range": None, "logg_range": None, "meta_range": None, "wl_range": None}
    assert cli._download_cuts("G", "njm", None, {}, none) == {}
    assert "No terminal to ask for axis cuts: downloading the full grid" in capsys.readouterr().out


def test_njm_cuts_are_asked_in_a_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Stdin(True))
    monkeypatch.setattr(cli, "prompt_njm_axis_cuts", lambda name, g, model_url=None: {"teff_range": (1.0, 2.0)})
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
def test_filters_info_does_not_fail_when_only_njm_is_unreachable(monkeypatch, capsys):
    info = _availability(errors=["NJM: Connection refused"], availability_source="svo",
                         sources_failed={"njm": ["Connection refused"]}, fallback_from=["njm"])
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: info))
    monkeypatch.setattr(cli, "_catalogue_entry", lambda *a, **k: {
        "info": None, "properties": None, "errors": ["Cannot fetch index.json from NJM: Connection refused"]})
    assert run_cli(monkeypatch, "filters-info", "Fac/Cam") == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "Source unavailable : NJM: Connection refused" in out
    assert "so no description, licence or band properties are shown" in out


def test_filters_info_fails_when_no_source_answered(monkeypatch):
    info = _availability(errors=["NJM: Connection refused", "SVO: timed out"], availability_source="unknown",
                         sources_failed={"njm": ["Connection refused"], "svo": ["timed out"]}, fallback_from=[])
    monkeypatch.setattr(Filters, "available", classmethod(lambda cls, *a, **k: info))
    monkeypatch.setattr(cli, "_catalogue_entry", lambda *a, **k: {"info": None, "properties": None, "errors": []})
    assert run_cli(monkeypatch, "filters-info", "Fac/Cam") == 1


def test_browsing_prints_each_listing_error_once(tmp_path, monkeypatch, capsys):
    from sed_tools.filter_library import FilterLibrary

    monkeypatch.setattr(FilterLibrary, "facilities", lambda self: {
        "entries": [{"key": "A", "label": "A", "sources": ["svo"]}], "errors": ["SVO: timed out"]})
    monkeypatch.setattr(FilterLibrary, "instruments", lambda self, facility: {
        "entries": [{"key": "Cam", "label": "Cam", "sources": ["svo"]}], "errors": ["SVO: timed out"]})
    answer(monkeypatch, "1", "b", "1", "b", "q")
    assert cli.run_filters_flow(str(tmp_path)) == 0
    assert capsys.readouterr().out.count("[filters] SVO: timed out") == 1


def test_an_unreachable_njm_is_neither_printed_again_nor_a_failure(tmp_path, monkeypatch, capsys):
    from sed_tools.filter_library import FilterLibrary

    monkeypatch.setattr(FilterLibrary, "facilities", lambda self: {"entries": [
        {"key": "A", "label": "A", "sources": ["svo"]}, {"key": "B", "label": "B", "sources": ["svo"]}],
        "errors": ["NJM unavailable"]})
    monkeypatch.setattr(FilterLibrary, "instruments", lambda self, facility: {
        "entries": [{"key": "Cam", "label": "Cam", "sources": ["svo"]}], "errors": ["NJM unavailable"]})
    fetched = []
    monkeypatch.setattr(FilterLibrary, "published", lambda self, f, i: fetched.append(f"{f}/{i}") or tmp_path)
    answer(monkeypatch, "1,2", "y")
    assert cli.run_filters_flow(str(tmp_path)) == 0
    assert fetched == ["A/Cam", "B/Cam"]
    assert "NJM unavailable" not in capsys.readouterr().out


def test_availability_notes_are_shown(capsys):
    cli.print_filter_availability(_availability(notes=["no component manifest: NJM serves only the published "
                                                       "curves, described here from their headers"]))
    out = " ".join(capsys.readouterr().out.split())
    assert "Note : no component manifest: NJM serves only the published curves" in out


def test_an_unknown_band_is_a_clean_error(monkeypatch, capsys):
    def compose(cls, *a, **k):
        raise ComponentError("Argus/DR280 has no band 'nosuch'")

    monkeypatch.setattr(Filters, "compose", classmethod(compose))
    assert run_cli(monkeypatch, "filters-compose", "Argus/DR280", "--bands", "nosuch") == 1
    assert capsys.readouterr().out == "[filters] Argus/DR280 has no band 'nosuch'\n"


def test_grids_without_a_terminal_or_names_says_to_use_grids(monkeypatch, tmp_path):
    class Grabber:
        def __init__(self, *args, **kwargs):
            pass

        def discover_models(self):
            return ["bt-cond"]

    monkeypatch.setattr(cli, "MSGSpectraGrabber", Grabber)
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
