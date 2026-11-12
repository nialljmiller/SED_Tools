"""CLI and ``python -m`` entry points: exit statuses and messages, the menus, pickers and prompts, the CLI
without its optional packages, and every old module path's main().

One test walks the tables below; each row runs isolated (support.Rows) and no row uses the network. Rows
from ``python -m`` runs (MAINS, HELP) start in their own folder; menu and picker rows (MENUS) see an
80-column terminal.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import requests

from sed_tools import cli
from sed_tools.api import SED, Filters
from sed_tools.filter_components import ComponentError
from sed_tools.njm_filter_grabber import NJMFilterGrabber

from support import (CLEAN, CURVE, GRABBERS, KURUCZ, NJM, Session, answer, cpu_torch, fake_grabber, filters_api,
                     fixed_completer, fixed_generator, narrow_terminal, njm_index, run_cli, run_module, scenarios,
                     wide_lines, write_flux_cube, write_lookup)

# runpy warns that an old path imported earlier in the session is in sys.modules; python -m starts afresh.
pytestmark = pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS, row = scenarios()
MAINS, main_row = scenarios()
MENUS, menu_row = scenarios()


def _raise(error):
    raise error


# ----------------------------------------------------------------------
# The filter pickers' stand-ins: a terminal, availability summaries, GAIA's instruments
# ----------------------------------------------------------------------
class Stdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def availability(**extra):
    info = {"facility": "Fac", "instrument": "Cam", "bands": ["J", "H"], "published_source": "", "published": "",
            "published_bands": {}, "components": ["filter"], "partial_components": {}, "buildable": ["filter"],
            "component_status": "", "component_availability": {"status": "supported", "builder": None},
            "missing_files": [], "errors": [], "atmosphere": False, "atmosphere_site": None,
            "atmosphere_sites": [], "platform": "ground", "detector_type": "", "detector_type_sources": {},
            "availability_source": "local", "variants": {}}
    info.update(extra)
    return info


def summary_inputs(mp, info=None, catalogue=None):
    """Stub what the picker's summary reads (FilterLibrary.available, NJM's catalogue entry); return the calls."""
    from sed_tools.filter_library import FilterLibrary

    calls = []
    mp.setattr(FilterLibrary, "available",
               lambda self, f, i, remote=True: calls.append(f"{f}/{i}") or (info or availability()))
    mp.setattr(cli.filters, "_catalogue_entry", lambda *a, **k: catalogue or {
        "info": None, "properties": None, "errors": ["NJM does not serve it"]})
    return calls


GAIA_INSTRUMENTS = [
    {"key": name, "label": name, "fetch_key": name, "sources": sources, "description": description}
    for name, sources, description in [
        ("DR2", ["njm", "original"], "Gaia DR2 passbands (Evans et al. 2018)"),
        ("DR2r", ["njm", "original"], "Gaia DR2 revised passbands (Evans et al. 2018)"),
        ("GAIA", ["njm"], "Gaia DR3 passbands + BP bright/faint (Maíz Apellániz & Weiler 2018)"),
        ("GAIA0", ["njm"], "Gaia pre-launch passbands (Jordi et al. 2010)"),
        ("Gaia2m", ["svo"], "Revised passbands for Gaia DR2.")]]


def _picker_library(t, facilities, instruments):
    from sed_tools.filter_library import FilterLibrary

    t.mp.setattr(FilterLibrary, "facilities", lambda self: {"entries": facilities, "errors": []})
    t.mp.setattr(FilterLibrary, "instruments", lambda self, facility: {"entries": instruments, "errors": []})
    fetched = []
    t.mp.setattr(FilterLibrary, "published", lambda self, f, i: fetched.append(f"{f}/{i}") or t.tmp)
    t.mp.setattr(cli.filters, "_offer_composition", lambda directory: None)
    return fetched


def gaia(t):
    """The filter pickers over GAIA and its instruments, NJM up; returns the sets fetched."""
    from sed_tools.filter_library import FilterLibrary

    t.mp.setattr(FilterLibrary, "mirror", SimpleNamespace(is_available=lambda: True))
    return _picker_library(t, [{"key": "GAIA", "label": "GAIA", "fetch_key": "GAIA", "sources": ["njm", "svo"],
                                "description": "Gaia mission."}], GAIA_INSTRUMENTS)


def catalogue(t):
    """The filter pickers over GAIA and 2MASS from SVO, with the summary stubbed; returns the sets fetched."""
    fetched = _picker_library(t, [{"key": "GAIA", "label": "GAIA", "fetch_key": "Gaia", "sources": ["svo"]},
                                  {"key": "TWOMASS", "label": "2MASS", "sources": ["svo"]}],
                              [{"key": "GAIA2", "label": "GAIA2", "fetch_key": "DR2", "sources": ["svo"]},
                               {"key": "Cam", "label": "Cam", "sources": ["svo"]}])
    summary_inputs(t.mp)
    return fetched


# ----------------------------------------------------------------------
# Top-level workflows, the config subcommand and the filter-combine wizard
# ----------------------------------------------------------------------
@row
def top_level_workflows_are_cli_aliases(t):
    import sed_tools

    assert sed_tools.run_rebuild_flow is cli.run_rebuild_flow
    assert sed_tools.run_spectra_flow is cli.run_spectra_flow
    assert sed_tools.run_combine_flow is cli.run_combine_flow
    assert sed_tools.menu is cli.menu


@row
def cli_config_subcommand_show(t):
    with contextlib.suppress(SystemExit):
        run_cli(t.mp, "config")
    captured = t.capsys.readouterr()
    assert "Data directory" in captured.out


@row
def run_filter_combine_flow_wizard(t):
    for folder, band, wl in (("GAIA/GAIA", "G", 1), ("2MASS/2MASS", "J", 2)):
        (t.tmp / folder).mkdir(parents=True)
        (t.tmp / folder / f"{band}.dat").write_text(f"Wavelength,Transmission\n{wl},1\n")
    # the base dir given, both local sets, a name, then the defaults: facility, output name, conflicts, proceed
    answer(t.mp, "", "1,2", "Gaia2MASS", "", "", "", "")
    cli.run_filter_combine_flow(base_dir=str(t.tmp))
    out = t.tmp / "Combined" / "Gaia2MASS"
    assert sorted(p.name for p in out.glob("*.dat")) == ["G.dat", "J.dat"]
    assert (out / "Gaia2MASS").read_text().splitlines() == ["G.dat", "J.dat"]
    assert "Combined 2 filters" in t.capsys.readouterr().out


# ----------------------------------------------------------------------
# Failures exit non-zero or say so: filters-info, coverage, the spectra and filter flows, rebuild, densify
# ----------------------------------------------------------------------
#: name: (source errors, NJM catalogue errors, --offline, exit status). NJM's catalogue is an extra: without
#: it the availability report still stands.
FILTERS_INFO = {
    "filters_info_exit_status_0_when_the_sources_answer": ([], [], False, 0),
    "filters_info_exit_status_1_when_a_source_failed": (["SVO: no metadata for Fac/Cam.g"], [], False, 1),
    "filters_info_exit_status_0_when_njm_is_unreachable": ([], ["NJM unreachable"], False, 0),
    "filters_info_exit_status_0_offline_without_the_cached_catalogue": (
        [], ["info.json is not cached and working offline"], True, 0),
}


def _filters_info(t, source_errors, catalogue_errors, offline, expected):
    info = {"facility": "Fac", "instrument": "Cam", "bands": [], "published_source": "", "published": "",
            "published_bands": {}, "components": [], "partial_components": {}, "buildable": [],
            "component_status": "", "component_availability": {"status": "unreviewed", "builder": None},
            "missing_files": [], "errors": source_errors, "atmosphere": False, "platform": "ground",
            "detector_type": "", "detector_type_sources": {}, "availability_source": "svo",
            "sources_failed": {"svo": [e.split(": ", 1)[1] for e in source_errors]} if source_errors else {},
            "fallback_from": []}
    t.mp.setattr(filters_api(t), "available", classmethod(lambda cls, *args, **kwargs: info))
    t.mp.setattr(cli, "_catalogue_entry", lambda *args, **kwargs:
                 {"info": None, "properties": None, "errors": catalogue_errors})
    argv = ["filters-info", "Fac/Cam"] + (["--offline"] if offline else [])
    assert run_cli(t.mp, *argv) == expected


@row
def coverage_exits_1_when_there_are_no_local_grids(t):
    assert run_cli(t.mp, "coverage", "--base", str(t.tmp)) == 1
    assert "No local grids found" in t.capsys.readouterr().out


def _python_m_cli(*argv, **kwargs):
    """``python -m sed_tools.cli *argv`` in a fresh interpreter."""
    return subprocess.run([sys.executable, "-m", "sed_tools.cli", *argv], cwd=ROOT, capture_output=True, text=True,
                          **kwargs)


@row
def running_the_cli_module_passes_on_the_exit_status(t):
    done = _python_m_cli("coverage", "--base", str(t.tmp))
    assert done.returncode == 1, done.stderr


def _spectra_flow(t, **grabbers):
    """run_spectra_flow with these grabbers (the others list nothing), and cleaning and product builds faked;
    returns the grids built."""
    built = []
    for name, target in GRABBERS.items():
        t.mp.setattr(target, grabbers.get(name, fake_grabber()))
    t.mp.setattr(cli.grids, "clean_model_dir", lambda *a, **k: CLEAN)
    t.mp.setattr(cli.grids, "undeclared_flux_unit", lambda *a, **k: None)  # cleaning is faked too
    t.mp.setattr(cli.grids, "_build_data_products", lambda name, *a: built.append(name) or [])
    return built


@row
def spectra_flow_builds_what_it_can_then_lists_every_grid_not_completed(t):
    built = _spectra_flow(t, svo=fake_grabber(["Good", "NoMeta", "Empty"]),
                          msg=fake_grabber(["Good"], metadata={"m": 1}, downloaded=1),
                          mast=fake_grabber(["Empty"], metadata={"m": 1}, downloaded=0))
    with pytest.raises(RuntimeError) as excinfo:
        cli.run_spectra_flow("all", base_dir=str(t.tmp), models=["Good", "NoMeta", "Empty", "Missing"])
    message = str(excinfo.value)
    assert built == ["Good"]
    assert "NoMeta: RuntimeError: no source could supply it. Tried: svo: no metadata" in message
    assert ("Empty: RuntimeError: no source could supply it. Tried: svo: no metadata for Empty; "
            "mast: no spectra downloaded") in message
    assert "Missing: not listed by any source" in message
    assert "Good" not in message
    assert "Done." not in t.capsys.readouterr().out


@row
def spectra_flow_raises_when_no_grid_is_discovered(t):
    _spectra_flow(t, svo=fake_grabber([]))  # as a grabber that could not read its index, after its warning
    with pytest.raises(RuntimeError, match="No spectral grids discovered"):
        cli.run_spectra_flow("svo", base_dir=str(t.tmp), models=["Grid"])


@row
def axis_cut_prompt_takes_ranges_and_never_reads_a_typo_as_no_cut(t):
    grabber = SimpleNamespace(get_grid_info=lambda name, model_url=None: None)
    answer(t.mp, "3500,8000", "", "", "")
    cuts = cli.prompt_njm_axis_cuts("Grid", grabber)
    assert cuts == {"teff_range": (3500.0, 8000.0), "logg_range": None, "meta_range": None, "wl_range": None}
    answer(t.mp, "3500 to 8000")
    with pytest.raises(ValueError):  # not None, which would download the whole grid
        cli.prompt_njm_axis_cuts("Grid", grabber)


def _rebuild(t):
    """Local grids A, B and C with rebuild's cleaning and builds faked; cleaning B fails. Returns the grids
    rebuilt."""
    for name in "ABC":
        (t.tmp / name).mkdir()
        (t.tmp / name / "s.txt").write_text("1000 1\n")
    rebuilt = []

    def clean(model_dir, **kwargs):
        if model_dir.endswith("B"):
            raise ValueError("corrupt header in s.txt")
        rebuilt.append(Path(model_dir).name)
        return dict(CLEAN)

    t.mp.setattr(cli.grids, "clean_model_dir", clean)
    t.mp.setattr(cli.grids, "regenerate_lookup_table", lambda model_dir: None)
    t.mp.setattr(cli.grids, "build_h5_bundle_from_txt", lambda model_dir, out: None)
    t.mp.setattr("sed_tools.cli.grids._build_flux_cube", lambda model_dir: Path(model_dir) / "flux_cube.bin")
    return rebuilt


@row
def rebuild_rebuilds_every_grid_it_can_then_lists_the_rest(t):
    rebuilt = _rebuild(t)
    with pytest.raises(RuntimeError) as excinfo:
        cli.run_rebuild_flow(base_dir=str(t.tmp), models=["A", "B", "C", "Nope"])
    assert rebuilt == ["A", "C"]
    assert "B: ValueError: corrupt header in s.txt" in str(excinfo.value)
    assert "Nope: not a local spectral grid under" in str(excinfo.value)


@row
def rebuild_asks_which_grids(t):
    rebuilt = _rebuild(t)
    answer(t.mp, "1,3")
    cli.run_rebuild_flow(base_dir=str(t.tmp))
    assert rebuilt == ["A", "C"]


@row
def rebuild_never_writes_its_bundle_over_a_downloaded_grid_file(t):
    _rebuild(t)
    t.mp.setattr(cli.grids, "build_h5_bundle_from_txt", cli.build_h5_bundle_from_txt)  # the real bundle
    msg = t.tmp / "sg-Test"
    msg.mkdir()
    (msg / "s.txt").write_text("1000 1\n")
    with h5py.File(msg / "sg-Test.h5", "w") as fh:  # the grid file as MSG serves it: no 'spectra' group
        fh.create_group("specsource")
    downloaded = (msg / "sg-Test.h5").read_bytes()
    cli.run_rebuild_flow(base_dir=str(t.tmp), models=["sg-Test", "A", "A"])
    assert (msg / "sg-Test.h5").read_bytes() == downloaded
    with h5py.File(msg / "sg-Test_bundle.h5", "r") as fh:
        assert list(fh["spectra"]) == ["s.txt"]
    assert sorted(p.name for p in (t.tmp / "A").glob("*.h5")) == ["A.h5"]  # a bundle is rebuilt in place


@row
def rebuild_and_combine_raise_when_there_are_no_local_grids(t):
    with pytest.raises(FileNotFoundError, match="No local spectral grids found"):
        cli.run_rebuild_flow(base_dir=str(t.tmp), models=["A"])
    with pytest.raises(FileNotFoundError, match="No spectral grids found"):
        cli.run_combine_flow(base_dir=str(t.tmp), interactive=False)


def _filter_browser(t, listing_errors=None):
    """Facilities A and B with instruments Cam1 and Cam2; fetching B/Cam2 fails. Returns the sets fetched."""
    from sed_tools.filter_library import FilterLibrary

    filters_api(t)
    t.mp.setattr(FilterLibrary, "facilities", lambda self: {
        "entries": [{"key": "A", "label": "A"}, {"key": "B", "label": "B"}], "errors": []})
    t.mp.setattr(FilterLibrary, "instruments", lambda self, facility: {
        "entries": [{"key": "Cam1", "label": "Cam1"}, {"key": "Cam2", "label": "Cam2"}],
        "errors": (listing_errors or {}).get(facility, [])})
    fetched = []

    def fetch(self, facility, instrument):
        if (facility, instrument) == ("B", "Cam2"):
            raise requests.ConnectionError("SVO down")
        fetched.append(f"{facility}/{instrument}")
        return t.tmp / facility / instrument

    t.mp.setattr(FilterLibrary, "published", fetch)  # the browser downloads with its own library
    t.mp.setattr(cli.filters, "_offer_composition", lambda directory: None)
    # What the browser describes before asking to download (Filters.available, NJM's catalogue entry).
    t.mp.setattr(FilterLibrary, "available", lambda self, facility, instrument, remote=True: {
        "facility": facility, "instrument": instrument, "bands": ["g"], "published": "filter",
        "published_bands": {}, "components": [], "partial_components": {}, "component_status": None,
        "errors": [], "atmosphere": False, "platform": "ground", "variants": {}})
    t.mp.setattr(cli.filters, "_catalogue_entry", lambda *a, **k: {"info": None, "properties": None, "errors": []})
    return fetched


@row
def filter_batch_downloads_the_rest_then_lists_failures(t):
    fetched = _filter_browser(t, {"A": ["SVO A: timed out"]})
    t.mp.setattr("sys.stdin", Stdin(True))  # a terminal: the count is confirmed
    answer(t.mp, "1,2", "y")
    assert cli.run_filters_flow() == 1
    out = t.capsys.readouterr().out
    assert fetched == ["A/Cam1", "A/Cam2", "B/Cam1"]
    assert "B/Cam2: SVO down" in out
    assert "A: instrument list incomplete: SVO A: timed out" in out
    assert "Cannot fetch: SVO down" in out


@row
def filter_browser_reports_a_failed_build_when_browsing_ends(t):
    _filter_browser(t)
    t.mp.setattr(cli.filters, "_offer_composition", lambda directory: _raise(
        ValueError("Airmass and PWV must be numbers (could not convert string to float: 'x')")))
    answer(t.mp, "1", "1", "y", "y", "q")  # facility A, instrument 1, download, then quit from the instrument list
    assert cli.run_filters_flow() == 1
    out = t.capsys.readouterr().out
    assert "Composed product of A/Cam1 not built: Airmass and PWV must be numbers" in out
    assert "A/Cam1 composed product: Airmass and PWV must be numbers" in out.split("Failed (1):")[1]


@row
def filters_command_without_a_target_fails_when_a_download_failed(t):
    _filter_browser(t)
    answer(t.mp, "2", "2", "y", "n", "q")
    assert run_cli(t.mp, "filters") == 1
    assert "B/Cam2: SVO down" in t.capsys.readouterr().out.split("Failed (1):")[1]


def _paranal(t, atmospheres):
    """An instrument whose curve can be composed with Paranal's atmosphere, listed by ``atmospheres``."""
    filters = filters_api(t)
    info = {"buildable": ["filter"], "atmosphere": True, "atmosphere_site": "paranal", "atmosphere_sites": None,
            "component_availability": {"status": "supported", "large_download": False}}
    t.mp.setattr(filters, "available", classmethod(lambda cls, *a, **k: info))
    t.mp.setattr(filters, "atmospheres", classmethod(atmospheres))
    t.mp.setattr(cli.filters, "print_filter_availability", lambda info: None)
    return filters


@row
def offer_composition_folds_in_the_site_and_airmass_typed(t):
    built = []
    filters = _paranal(t, lambda cls, *a, **k: {})
    t.mp.setattr(filters, "compose",
                 classmethod(lambda cls, *a, **k: built.append(k["atmosphere"]) or t.tmp / "product"))
    answer(t.mp, "y", "1", "1", "1.2", "")  # site 1 of the numbered atmosphere choice
    cli._offer_composition(t.tmp / "Fac" / "Cam")
    assert built == [{"site": "paranal", "airmass": 1.2, "pwv": None}]


def _densify_calls(t):
    """densify_grid's calls, each checked to bind to its real signature; nothing is densified."""
    import inspect

    module = importlib.import_module("sed_tools.grid_densifier")
    signature = inspect.signature(module.densify_grid)
    calls = []

    def record(*args, **kwargs):
        signature.bind(*args, **kwargs)
        calls.append(kwargs)

    t.mp.setattr(module, "densify_grid", record)
    return calls


@row
def densify_passes_only_what_densify_grid_accepts(t):
    calls = _densify_calls(t)
    base = ["densify", "--flux-cube", "in.bin", "--output", "out.bin", "--teff-spacing", "250"]
    assert run_cli(t.mp, *base, "--no-lookup") == 0
    assert run_cli(t.mp, *base, "--method", "ml", "--ml-model", "gen") == 0
    assert run_cli(t.mp, *base, "--method", "blackbody") == 0
    assert calls == [
        {"src": "in.bin", "dst": "out.bin", "teff_spacing": 250.0, "ml_model": None},
        {"src": "in.bin", "dst": "out.bin", "teff_spacing": 250.0, "ml_model": "gen"},
        {"src": "in.bin", "dst": "out.bin", "teff_spacing": 250.0, "ml_model": None},
    ]


@row
def cleaning_report_reads_the_summary_clean_model_dir_returns(t):
    cli._print_cleaning_report(dict(CLEAN, total=4, converted=["a", "b"], recovered=["c"], skipped_already=["d"],
                                    catalog_units={"wavelength": "angstrom", "flux": "flam"},
                                    detection_stats={"status": "consensus"}))
    out = t.capsys.readouterr().out
    assert "Cleaned       : 2 files" in out and "Recovered     : 1 files" in out
    assert "Already done  : 1 files" in out
    assert "All 4 files ready for use" in out


@row
def picker_rejects_an_invalid_regex_instead_of_showing_everything(t):
    from sed_tools.ui_utils import prompt_choice

    answer(t.mp, "/beta", "//[unclosed", "1")
    assert prompt_choice(["alpha", "beta", "gamma"], "Things", max_label=20) == 0
    out = t.capsys.readouterr().out
    assert "Invalid regular expression '[unclosed'" in out
    # The substring filter stays in force: the last listing shows beta only.
    last_listing = out.rsplit("Things", 1)[1]
    assert "beta" in last_listing and "alpha" not in last_listing


@row
def ml_generator_auto_fails_at_once_without_optuna(t):
    find_spec = importlib.util.find_spec
    t.mp.setattr(importlib.util, "find_spec", lambda name, *args: None if name == "optuna" else find_spec(name, *args))
    assert run_cli(t.mp, "ml_generator", "--auto", "--library", "lib", "--output", "out") == 1
    out, err = t.capsys.readouterr()
    assert not out and len(err.splitlines()) == 1  # the one-line refusal of the other ML actions
    assert err.startswith("Error: missing ") and "optuna" in err and "ml-tune" in err
    assert "ml-tune" in err.split("pip install ")[1]


@row
def composition_says_when_the_atmosphere_list_failed(t):
    _paranal(t, lambda cls, *a, **k: _raise(RuntimeError("NJM mirror unreachable")))
    answer(t.mp, "y", "1", "", "y")
    with pytest.raises(RuntimeError, match="could not list the atmosphere grids for paranal: NJM mirror unreachable"):
        cli._offer_composition(t.tmp / "Fac" / "Cam")


@row
def sed_tools_mesa_prepare_exit_status_is_1_on_failure(t):
    write_lookup(t.tmp / "Plain", ["file_name", "teff", "logg", "metallicity"], [("a.txt", 5000, 4.0, 0.0)])
    done = _python_m_cli("mesa_prepare", "--base", str(t.tmp), "--grid", "Plain", "--output", str(t.tmp / "out"),
                         stdin=subprocess.DEVNULL)
    assert done.returncode == 1 and "has no flux_cube.bin" in done.stderr


# ----------------------------------------------------------------------
# Usage errors: exit 2 in plain words, under the subcommand's usage, before any work is done
# ----------------------------------------------------------------------
_DENSIFY = ["densify", "--flux-cube", "in.bin", "--output", "out.bin"]

#: name: (argv, what stderr says).
USAGE = {
    "densify_refuses_options_it_cannot_honour_interp": (
        [*_DENSIFY, "--method", "interp"], "--method interp is not implemented"),
    "densify_refuses_options_it_cannot_honour_ml_without_a_model": (
        [*_DENSIFY, "--method", "ml"], "--method ml needs --ml-model"),
    "densify_refuses_options_it_cannot_honour_blackbody_with_a_model": (
        [*_DENSIFY, "--method", "blackbody", "--ml-model", "gen"], "--method blackbody does not use --ml-model"),
    "densify_refuses_options_it_cannot_honour_gap_threshold": (
        [*_DENSIFY, "--ml-gap-threshold", "2000"], "--ml-gap-threshold is not implemented"),
}


def _usage_error(t, argv, message):
    narrow_terminal(t)
    densified = _densify_calls(t)
    with pytest.raises(SystemExit) as stop:
        run_cli(t.mp, *argv)
    err = t.capsys.readouterr().err
    assert stop.value.code == 2 and message in err and "_bound" not in err
    assert err.startswith(f"usage: sed-tools {argv[0]}") and densified == []


# ----------------------------------------------------------------------
# python -m <old path>: --help, a bad argument and a short run of every module with a main()
# ----------------------------------------------------------------------
def _in_its_folder(t):
    """As ``python -m`` from a shell in the row's folder, whose stellar_models/ is the default grid folder."""
    t.mp.chdir(t.tmp)
    # svo_regen_spectra_lookup without argument parsing rewrote every lookup table under it, even for --help
    t.mp.setattr("sed_tools.cube.model.STELLAR_DIR_DEFAULT", t.tmp / "stellar_models")


#: Each old path with a main(), and its help's description.
HELP = {
    "combine_filters": "Combine filter sets into one MESA-compatible instrument",
    "combine_stellar_atm": "Combine stellar spectral grids",
    "flux_cube_tool": "Inspect and interpolate a flux cube",
    "grid_densifier": "Densify a flux_cube.bin Teff grid using SEDGenerator.",
    "ml_sed_completer": "ML-powered SED Completer (PyTorch)",
    "ml_sed_generator": "ML SED Generator",
    "msg_spectra_grabber": "Extract MSG stellar spectra to ASCII with correct params",
    "njm_filter_grabber": "Download filters from NJM mirror",
    "precompute_flux_cube": "Precompute flux cube for stellar spectral grids",
    "svo_filter_grabber": "Interactively browse the SVO Filter Profile Service",
    "svo_regen_spectra_lookup":
        "Rewrite lookup_table.csv in every spectral grid folder from the headers of its spectra",
}


def _help(t, module):
    _in_its_folder(t)
    assert run_module(t.mp, module, "--help") == 0
    out = " ".join(t.capsys.readouterr().out.split())
    assert out.lower().startswith("usage:") and HELP[module] in out


def _bad_argument(t, module):
    _in_its_folder(t)
    assert run_module(t.mp, module, "--no-such-option") == 2
    captured = t.capsys.readouterr()
    assert captured.out == "" and "usage:" in captured.err and "error:" in captured.err


def _in_its_folder_run(t, scenario):
    _in_its_folder(t)
    scenario(t)


def _johnson(t, band, curve):
    (t.tmp / "Generic" / "Johnson").mkdir(parents=True)
    (t.tmp / "Generic" / "Johnson" / f"{band}.dat").write_text("Wavelength,Transmission\n" + curve)


@main_row
def combine_filters_writes_the_combined_set(t):
    _johnson(t, "B", "4000,0\n4400,1\n4800,0\n")
    assert run_module(t.mp, "combine_filters", "Optical", "Generic/Johnson", "--base", str(t.tmp)) == 0
    out = t.tmp / "Combined" / "Optical"
    assert (out / "Optical").read_text().split() == ["B.dat"]
    assert t.capsys.readouterr().out == f"[filters] Combined filter set written to {out}\n"


@main_row
def filters_combine_command_writes_the_combined_set(t):
    _johnson(t, "V", "5000,0\n5500,1\n6000,0\n")
    assert run_cli(t.mp, "filters-combine", "Mine/Set", "Generic/Johnson", "--base", str(t.tmp)) == 0
    assert (t.tmp / "Mine" / "Set" / "Set").read_text().split() == ["V.dat"]
    assert t.capsys.readouterr().out == f"[filters] Combined filter set written to {t.tmp / 'Mine' / 'Set'}\n"


@main_row
def svo_filter_grabber_ignores_arguments_after_the_folder_as_in_041(t):
    from sed_tools.filters import svo

    seen = []
    t.mp.setattr(svo, "run_interactive", seen.append)
    assert svo.main([str(t.tmp), "extra"]) == 0
    assert seen == [str(t.tmp)]


@main_row
def combine_stellar_atm_combines_the_grids_chosen(t):
    assert run_module(t.mp, "combine_stellar_atm", "--data_dir", str(t.tmp)) == 0
    assert f"No spectral grids found in {t.tmp}" in t.capsys.readouterr().out
    shutil.copytree(KURUCZ, t.tmp / KURUCZ.name)
    answer(t.mp, "q")
    assert run_module(t.mp, "combine_stellar_atm", "--data_dir", str(t.tmp)) == 0
    assert "No spectral grids selected." in t.capsys.readouterr().out
    assert not (t.tmp / "combined_models").exists()
    answer(t.mp, "1")
    assert run_module(t.mp, "combine_stellar_atm", "--data_dir", str(t.tmp), "--output", "omni") == 0
    assert (t.tmp / "omni" / "flux_cube.bin").is_file() and (t.tmp / "omni" / "lookup_table.csv").is_file()
    assert t.capsys.readouterr().out.endswith(f"Done! Output: {t.tmp / 'omni'}\n")


@main_row
def precompute_flux_cube_builds_the_cube_it_is_given(t):
    grid = shutil.copytree(KURUCZ, t.tmp / "grid")
    assert run_module(t.mp, "precompute_flux_cube", "--model_dir", str(grid), "--output", "cube.bin") == 0
    assert (t.tmp / "cube.bin").stat().st_size > 16


@main_row
def precompute_flux_cube_asks_for_the_grid_and_the_output(t):
    assert run_module(t.mp, "precompute_flux_cube") == "No model directories found."
    shutil.copytree(KURUCZ, t.tmp / "data" / "stellar_models" / KURUCZ.name)
    (t.tmp / "data" / "stellar_models" / "notes.txt").write_text("not a grid\n")
    answer(t.mp, "0")
    assert run_module(t.mp, "precompute_flux_cube") == "Invalid selection."
    # Listed once (it was listed as ./data/stellar_models/<grid> and as data/stellar_models/<grid>)
    menu = [line for line in t.capsys.readouterr().out.splitlines() if line[:1].isdigit()]
    assert menu == [f"1. data/stellar_models/{KURUCZ.name}"]
    answer(t.mp, "1", "")  # the first grid listed, and the default output beside it
    assert run_module(t.mp, "precompute_flux_cube") == 0
    assert "No --model_dir provided. Discovering local model directories" in t.capsys.readouterr().out
    assert len(list(t.tmp.glob("**/flux_cube.bin"))) == 1


@main_row
def svo_regen_spectra_lookup_rewrites_every_grids_lookup_table(t):
    stellar = t.tmp / "stellar_models"  # the default grid folder
    for name in ("A", "B"):
        (stellar / name).mkdir(parents=True)
        (stellar / name / "s.txt").write_text("# teff = 5000\n# logg = 4.5\n# metallicity = 0.0\n4000 1\n5000 2\n")
    (stellar / "A" / "A_alpha_0p40").mkdir()  # a split grid's subgrid, which crashed it until 0.6.0
    (stellar / "A" / "A_alpha_0p40" / "s.txt").symlink_to("../s.txt")
    assert run_module(t.mp, "svo_regen_spectra_lookup") == 0
    for folder in ("A", "B", "A/A_alpha_0p40"):
        assert (stellar / folder / "lookup_table.csv").read_text().splitlines()[1] == "s.txt,4.5,0.0,5000.0"
    assert t.capsys.readouterr().out.count("Processing spectral grid:") == 3


@main_row
def msg_spectra_grabber_runs_the_grids_it_is_given(t):
    from sed_tools.grids.msg import MSGSpectraGrabber

    runs = []
    t.mp.setattr(MSGSpectraGrabber, "discover_and_run", lambda self, selected_models=None:
                 runs.append((self.base_dir, self.max_workers, selected_models)))
    assert run_module(t.mp, "msg_spectra_grabber", "--output", str(t.tmp), "--workers", "2",
                      "--models", "A", "B") == 0
    assert runs == [(str(t.tmp), 2, ["A", "B"])]


@main_row
def grid_densifier_passes_its_arguments_on(t):
    calls = []
    t.mp.setattr("sed_tools.cube.densify.densify_grid", lambda **kwargs: calls.append(kwargs))
    assert run_module(t.mp, "grid_densifier", "--flux-cube", "in.bin", "--output", "out.bin",
                      "--teff-spacing", "250") == 0
    assert calls == [{"src": "in.bin", "dst": "out.bin", "teff_spacing": 250.0, "ml_model": None}]


def _njm_served(t):
    """NJM serves one facility with one camera of one band; downloads are recorded instead of made."""
    files = {"index.json": njm_index({"Fac": {"Cam": {"filters": ["g"]}}}), "filters/Fac/Cam/Cam": "g.dat\n",
             "filters/Fac/Cam/g.dat": CURVE}
    t.mp.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    t.mp.setattr("sed_tools.filters.njm.requests.Session", lambda: Session(files, base=NJM))
    calls = []
    t.mp.setattr(NJMFilterGrabber, "download_filters", lambda self, *a: calls.append(("set", *a)))
    t.mp.setattr(NJMFilterGrabber, "download_all_filters", lambda self, **k: calls.append(("all", k)))
    return calls


@main_row
def njm_filter_grabber_lists_and_downloads(t):
    calls, out = _njm_served(t), str(t.tmp / "filters")
    assert run_module(t.mp, "njm_filter_grabber", "--list", "--output", out) == 0
    assert "Fac:\n    - Cam (1 filters)" in t.capsys.readouterr().out.split("Available facilities:")[1]
    for argv in (["--facility", "Fac", "--instrument", "Cam"], ["--facility", "Fac"], []):
        assert run_module(t.mp, "njm_filter_grabber", *argv, "--output", out) == 0
    assert calls == [("set", "Fac", "Cam"), ("all", {"facility": "Fac"}), ("all", {})]


@main_row
def njm_filter_grabber_needs_a_facility_for_an_instrument(t):
    calls = _njm_served(t)
    assert run_module(t.mp, "njm_filter_grabber", "--instrument", "Cam", "--output", str(t.tmp)) == 1
    assert t.capsys.readouterr().out == "Error: --instrument requires --facility\n" and calls == []


@main_row
def njm_filter_grabber_exits_1_when_njm_is_down(t):
    t.mp.setattr(NJMFilterGrabber, "_check_availability", lambda self: False)
    assert run_module(t.mp, "njm_filter_grabber", "--list", "--output", str(t.tmp)) == 1
    assert t.capsys.readouterr().out.endswith("NJM mirror is not available\n")


@main_row
def ml_train_commands_pass_their_options_on(t):
    from sed_tools.ml.completer import SEDCompleter
    from sed_tools.ml.generator import SEDGenerator

    calls = []
    t.mp.setattr(SEDCompleter, "train", lambda self, **kwargs: calls.append(kwargs))
    t.mp.setattr(SEDGenerator, "train", lambda self, *args, **kwargs: calls.append((args, kwargs)))
    assert run_module(t.mp, "ml_sed_completer", "train", "--library", "lib", "--output", "out", "--epochs", "3",
                      "--batch-size", "4", "--lr", "0.01", "--max-samples", "9", "--patience", "2") == 0
    assert run_module(t.mp, "ml_sed_generator", "train", "--library", "lib", "--output", "out",
                      "--epochs", "5") == 0
    assert calls == [{"library_path": "lib", "output_path": "out", "epochs": 3, "batch_size": 4,
                      "learning_rate": 0.01, "max_samples": 9, "patience": 2},
                     (("lib", "out"), {"epochs": 5, "batch_size": 64})]


@main_row
def ml_sed_completer_lists_and_completes_with_a_saved_model(t):
    torch, models = cpu_torch(t), t.tmp / "models"
    assert run_module(t.mp, "ml_sed_completer", "list", "--models-dir", str(models)) == 0
    assert t.capsys.readouterr().out == f"No models found in {models}/\n"
    fixed_completer(torch).save(models / "tiny", hidden_layers=[16, 8])
    t.capsys.readouterr()
    assert run_module(t.mp, "ml_sed_completer", "list", "--models-dir", str(models)) == 0
    out = t.capsys.readouterr().out
    assert "  tiny\n" in out and "Architecture: 400 -> [16, 8] -> 1000" in out
    wl = np.logspace(np.log10(3000.0), np.log10(9000.0), 50)
    np.savetxt(t.tmp / "in.txt", np.column_stack([wl, 1e-10 * (wl / 5000.0) ** -2]))
    assert run_module(t.mp, "ml_sed_completer", "complete", "--model", str(models / "tiny"), "--input", "in.txt",
                      "--output", "out.txt", "--teff", "5500", "--logg", "4.5", "--meta", "0", "--wl-min", "1000",
                      "--wl-max", "20000") == 0
    extended = np.loadtxt(t.tmp / "out.txt")
    assert extended[0, 0] <= 1000.0 * 1.01 and extended[-1, 0] >= 20000.0 * 0.99
    assert "Saved completed SED to: out.txt" in t.capsys.readouterr().out


@main_row
def ml_sed_generator_lists_and_generates_with_a_saved_model(t):
    torch, models = cpu_torch(t), t.tmp / "models"
    fixed_generator(torch).save(models / "gen", hidden_layers=[16, 8])
    t.capsys.readouterr()
    assert run_module(t.mp, "ml_sed_generator", "list", "--models-dir", str(models)) == 0
    assert t.capsys.readouterr().out == f"gen: {models / 'gen'}\n"
    assert run_module(t.mp, "ml_sed_generator", "generate", "--model", str(models / "gen"), "--teff", "5500",
                      "--logg", "4.5", "--meta", "0", "--output", "sed.txt") == 0
    lines = (t.tmp / "sed.txt").read_text().splitlines()
    assert lines[0] == "# wavelength_A flux_erg/s/cm2/A" and "Teff=5500.0 logg=4.5 [M/H]=0.0" in lines[1]
    assert np.loadtxt(t.tmp / "sed.txt").shape == (1000, 2)
    assert t.capsys.readouterr().out.endswith("Saved: sed.txt\n")


@main_row
def building_all_components_runs_every_builder_and_lists_the_failures(t):
    built = []

    def builder(name):
        def build(base, **kwargs):
            if name == "bad":
                raise RuntimeError("source did not answer")
            built.append((name, base, kwargs))
        return build

    t.mp.setattr(cli, "AUTOMATIC_BUILDERS", ("first", "skycalc", "bad"))
    t.mp.setattr(cli, "get_builder", builder)
    assert run_cli(t.mp, "filters-build-components", "all", "--base", str(t.tmp)) == 1
    assert built == [("first", str(t.tmp), {}), ("skycalc", str(t.tmp), {"sites": ["paranal", "lasilla"]})]
    out = t.capsys.readouterr().out
    assert "[filters] === bad ===" in out and out.endswith("[filters] FAILED bad: source did not answer\n")
    built.clear()
    t.mp.setattr(cli, "AUTOMATIC_BUILDERS", ("skycalc",))
    assert run_cli(t.mp, "filters-build-components", "all", "--base", str(t.tmp), "--sites", "lapalma") == 0
    assert built == [("skycalc", str(t.tmp), {"sites": ["lapalma"]})]


@main_row
def the_coverage_menu_reports_the_grid_chosen(t):
    from sed_tools.cli.grids import run_coverage_flow

    base = t.tmp / "stellar_models"
    base.mkdir()
    run_coverage_flow(str(base))
    assert t.capsys.readouterr().out == f"No local grids found under {base}\n"
    shutil.copytree(KURUCZ, base / KURUCZ.name)
    answer(t.mp, "b")
    run_coverage_flow(str(base))
    assert not (base / KURUCZ.name / "coverage.png").exists()
    answer(t.mp, "1")
    run_coverage_flow(str(base))
    assert (base / KURUCZ.name / "coverage.png").is_file()
    out = t.capsys.readouterr().out
    assert f"Coverage: {KURUCZ.name}" in out and "range   : 5750 -> 6250" in out


# ----------------------------------------------------------------------
# Menus: messages instead of tracebacks, and the menu survives a failed workflow
# ----------------------------------------------------------------------
def _no_banner(t):
    t.mp.setattr(cli, "_print_banner", lambda: None)


def _menu_choices(t, *choices):
    """The main menu, without its banner, chosen ``choices`` in turn."""
    _no_banner(t)
    t.mp.setattr(cli, "menu", iter(choices).__next__)


@menu_row
def debug_variable_shows_the_traceback(t):
    t.mp.setenv("SED_TOOLS_DEBUG", "1")
    assert run_cli(t.mp, "rebuild", "--base", str(t.tmp)) == 1
    err = t.capsys.readouterr().err
    assert "Traceback (most recent call last)" in err and err.endswith("No local spectral grids found under "
                                                                         f"{t.tmp}\n")


@menu_row
def menu_reports_a_failed_workflow_and_shows_the_menu_again(t):
    ran = []
    _menu_choices(t, "rebuild", "config", "quit")
    t.mp.setattr(cli, "run_rebuild_flow", lambda: _raise(FileNotFoundError("No local spectral grids found under /x")))
    t.mp.setattr(cli, "run_config_flow", lambda: ran.append("config"))
    assert cli._run_menu() == 1
    assert ran == ["config"]
    assert "sed-tools: error: No local spectral grids found under /x" in t.capsys.readouterr().err


#: name: (the exit code a workflow raises SystemExit with, what the menu prints to stderr, its exit status).
MENU_EXITS = {
    "menu_survives_a_workflow_that_exits_0": (None, "", 0),
    "menu_survives_a_workflow_that_exits_with_a_message": ("Not a variant grid", "Not a variant grid\n", 1),
}


def _menu_survives_a_workflow_that_exits(t, exit_code, printed, status):
    narrow_terminal(t)
    _menu_choices(t, "mesa_prepare", "quit")
    t.mp.setattr(cli, "run_mesa_prepare_flow", lambda: _raise(SystemExit(exit_code)))
    assert cli._run_menu() == status
    assert t.capsys.readouterr().err == printed


@menu_row
def menu_returns_after_ctrl_c_in_a_workflow(t):
    _menu_choices(t, "coverage", "quit")
    t.mp.setattr(cli, "run_coverage_flow", lambda: _raise(KeyboardInterrupt))
    assert cli._run_menu() == 0
    assert "Interrupted; back to the menu." in t.capsys.readouterr().out


@menu_row
def menu_says_unknown_option_and_asks_again(t):
    _no_banner(t)
    answer(t.mp, "x", "q")
    assert cli._run_menu() == 0
    assert t.capsys.readouterr().out.count("Unknown option: enter a number from the menu, or q to quit.") == 1


@menu_row
def menu_quits_when_input_ends(t):
    _no_banner(t)
    t.mp.setattr("builtins.input", lambda prompt="": _raise(EOFError))
    assert cli._run_menu() == 0


@menu_row
def rebuild_uses_the_flux_unit_option_without_asking(t):
    (t.tmp / "Cool").mkdir()
    for i, teff in enumerate((3500, 3600, 3700)):  # a grid that does not say its flux unit
        (t.tmp / "Cool" / f"s{i}.txt").write_text(f"# teff = {teff}\n# logg = 4.5\n# meta = 0.0\n"
                                                 "4000 1e-3\n5000 2e-3\n6000 3e-3\n")
    cleaned = []
    t.mp.setattr("sys.stdin", Stdin(False))
    t.mp.setattr(cli.grids, "clean_model_dir", lambda d, **k: cleaned.append(k["flux_unit"]) or
                 {"total": 3, "error": [], "skipped_invalid": [], "skipped_index": []})
    t.mp.setattr(cli.grids, "regenerate_lookup_table", lambda d: None)
    assert run_cli(t.mp, "rebuild", "--base", str(t.tmp), "--grids", "Cool", "--no-h5", "--no-cube",
                   "--flux-unit", "flam") == 0
    assert cleaned == ["flam"]


# ----------------------------------------------------------------------
# Variants, atmospheres and the composition prompt
# ----------------------------------------------------------------------
def _composed(t):
    """Filters.compose and Filters.fetch record their calls; returns them."""
    calls = []
    t.mp.setattr(Filters, "compose", classmethod(lambda cls, *a, **k: calls.append((a, k)) or t.tmp / "product"))
    t.mp.setattr(Filters, "fetch", classmethod(lambda cls, *a, **k: calls.append((a, k)) or "dir"))
    return calls


ARGUS_VARIANTS = {"filter": {"production": {"description": "Production filters", "default": True},
                             "addem": {"description": "ADDem filters"}}}

#: name: (availability fields, what the report says, what it must not say).
AVAILABILITY = {
    "availability_lists_variants_with_the_default_marked": (
        {"variants": ARGUS_VARIANTS}, "Variants (API only) : filter: production (default), addem", "--variant"),
    "availability_lists_every_atmosphere_site": (
        {"atmosphere": True, "atmosphere_sites": ["hopkins_2mass", "ctio_2mass"]},
        "Atmosphere : can be added at its own sites: hopkins_2mass, ctio_2mass; no default, "
        "so choose one with --atmosphere-site (in Python, Filters.atmospheres() lists the conditions)", "None"),
}


def _availability_lists(t, fields, says, unsaid):
    narrow_terminal(t)
    cli.print_filter_availability(availability(**fields))
    out = " ".join(t.capsys.readouterr().out.split())
    assert says in out
    assert unsaid not in out


#: name: (the site's conditions, what is typed, the airmass and PWV chosen).
AIRMASS = {
    "airmass_defaults_to_the_first_condition_the_site_has_unstated": (
        {"airmass": [], "pwv": [], "unstated_airmass": True}, [""], None, None),
    "airmass_defaults_to_the_first_condition_the_site_has_listed": (
        {"airmass": [1.0, 1.5], "pwv": [2.5, 5.0], "unstated_airmass": False}, ["1.5", "5"], 1.5, 5.0),
}


def _airmass_defaults(t, grid, typed, airmass, pwv):
    narrow_terminal(t)
    prompts = answer(t.mp, *typed)
    assert cli._ask_conditions("site", grid) == (airmass, pwv)
    shown = "unstated" if not grid["airmass"] else f"{grid['airmass'][0]:g}"
    assert prompts[0] == f"Airmass [{shown}]: "


@menu_row
def composition_takes_a_typed_recipe(t):
    composed = _composed(t)
    t.mp.setattr(Filters, "available", classmethod(lambda cls, *a, **k: availability()))
    answer(t.mp, "y", "published,optics")
    cli._offer_composition(t.tmp / "Fac" / "Cam")
    assert composed[0][1]["components"] == ["published", "optics"] and "detector_type" not in composed[0][1]


@menu_row
def composable_bands_compose_in_memory(t):
    import sed_tools.filter_components as fc

    manifest = SimpleNamespace(primary_bands=lambda: ["J", "Z"])
    t.mp.setattr(fc.InstrumentManifest, "load", classmethod(lambda cls, d: manifest))
    t.mp.setattr(fc, "compose_band", lambda directory, band, components, **kwargs:
                 _raise(ComponentError("covers 9000-56000 A")) if band == "Z" else None)
    working, failed = cli._composable_bands(t.tmp / "Paranal" / "VISTA",
                                            {"components": ["filter"], "atmosphere": None, "variants": None})
    assert working == ["J"] and failed == {"Z": "covers 9000-56000 A"}


SITES = {
    "paranal": {"airmass": [1.0, 1.5], "pwv": [2.5, 5.0], "unstated_airmass": False, "kind": "derived",
                "source": "ESO SkyCalc sky model (Noll et al. 2012)"},
    "paranal_vircam": {"airmass": [1.0, 2.0], "pwv": [1.0, 3.0], "unstated_airmass": False, "kind": "team",
                       "source": "Model atmospheres distributed with ESO's VIRCAM curves"},
    "lsst_std": {"airmass": [1.0, 1.2], "pwv": [], "unstated_airmass": False,
                 "source": "Rubin fiducial atmosphere (MODTRAN, with aerosols), lsst/throughputs v1.9.2"}}


@menu_row
def pressing_enter_at_the_atmosphere_choice_folds_in_none(t):
    composed, listings = _composed(t), []  # listings: include_remote of each Filters.atmospheres call
    t.mp.setattr(Filters, "atmospheres", classmethod(
        lambda cls, filter_root=None, include_remote=True: listings.append(include_remote) or SITES))
    t.mp.setattr(Filters, "available", classmethod(lambda cls, *a, **k: availability(
        atmosphere=True, atmosphere_site="lsst_std", atmosphere_sites=["lsst_std"])))
    prompts = answer(t.mp, "y", "1", "")
    cli._offer_composition(t.tmp / "LSST" / "LSST")
    assert [p for p in prompts if p.startswith("Atmosphere")] == ["Atmosphere [0, none]: "]
    assert composed[0][1]["atmosphere"] is None and listings == [True]  # one listing; installed sites are checked daily
    out = t.capsys.readouterr().out
    assert ("Atmosphere to fold in:\n"
            "  0. none                     the curve without atmosphere\n"
            "  1. lsst_std (default site)  Rubin fiducial atmosphere (MODTRAN, with aerosols…\n"
            ) in out


# ----------------------------------------------------------------------
# The picker: 0.4.1's numbered columns of names, then a summary; menu entries 3 and 4
# ----------------------------------------------------------------------
LICENCE = {"redistribution": "attribution",
           "terms": "MIT License Copyright (c) 2026 The University. Permission is hereby granted, free of charge.",
           "url": "https://github.com/argus-hdps/argus_sim/blob/main/LICENSE-file-with-hyphens"}


@menu_row
def the_description_block_names_in_full_and_cuts_descriptions_at_the_edge(t):
    t.mp.setenv("COLUMNS", "40")
    cli._print_instrument_descriptions("Cameras", [
        {"key": "PointGrey_Grasshopper2_14S5C", "description": "Point Grey Grasshopper2 14S5C colour camera"},
        {"key": "SQM", "description": ""}])
    assert t.capsys.readouterr().out.splitlines() == [
        "", "Cameras: 2 instruments",
        "  PointGrey_Grasshopper2_14S5C  Point G…",
        "  SQM                           no desc…"]
    cli._print_instrument_descriptions("Solo", [{"key": "A" * 40, "description": "text"}])
    assert t.capsys.readouterr().out.splitlines() == ["", "Solo: 1 instrument", "  " + "A" * 40]


@menu_row
def picking_an_instrument_shows_its_summary_before_the_download_question(t):
    fetched = gaia(t)
    info = availability(facility="GAIA", instrument="DR2", bands=["BP", "G", "RP"], published="filter+optics+qe",
                        components=[], buildable=[], platform="space",
                        component_status="ESA publishes only total passbands (Riello et al. 2021).")
    entry = {"info": {"licence": LICENCE, "references": ["a", "b"]}, "properties": None, "errors": []}
    looked_up = summary_inputs(t.mp, info, entry)
    shown = []
    t.mp.setattr("builtins.input", lambda prompt="": shown.append((prompt, t.capsys.readouterr().out))
                 or ["1", "1", "y", "n", "q"][len(shown) - 1])
    assert cli.run_filters_flow(str(t.tmp)) == 0
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
    assert looked_up == ["GAIA/DR2"] and fetched == ["GAIA/DR2"]


@menu_row
def declining_the_download_goes_back_to_the_instrument_list(t):
    fetched = gaia(t)
    summary_inputs(t.mp)
    prompts = answer(t.mp, "1", "2", "n", "b", "q")
    assert cli.run_filters_flow(str(t.tmp)) == 0
    assert prompts == ["> ", "> ", "Download GAIA/DR2r? [Y/n] ", "> ", "> "]
    assert fetched == []
    assert t.capsys.readouterr().out.count("Instruments for GAIA") == 2


@menu_row
def the_summary_fits_80_columns_in_about_a_dozen_lines(t):
    info = availability(bands=[f"F{n}W" for n in range(100, 700, 10)], variants=ARGUS_VARIANTS,
                        notes=["no component manifest: NJM serves only the published curves, described here "
                               "from their headers"], atmosphere=True, atmosphere_site="paranal",
                        atmosphere_sites=["paranal", "lasilla"], published="filter+instrument+atmosphere")
    entry = {"info": {"licence": LICENCE, "references": ["a"], "description": "Long. Text."},
             "properties": None, "errors": []}
    lines = cli._pick_summary("Fac/Cam", {"description": "word " * 30}, info, entry, "SVO")
    assert all(len(line) <= 80 for line in lines) and len(lines) <= 13
    flat = "\n".join(lines)
    assert "Fac/Cam  (60 bands)" in flat and "… (" in flat and "more)" in flat
    assert "Atmosphere : own sites: paranal (default), lasilla" in flat
    assert "Variants   : API only; filter: production (default), addem" in flat
    assert "filters-cite" not in flat and "--variant" not in flat


@menu_row
def the_summary_says_where_a_download_comes_from(t):
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
    lines = cli._pick_summary("GAIA/DR2", {}, availability(), {"info": None, "properties": None, "errors": []},
                              origin)
    assert "  From       : installed in the filter directory, as GAIA/DR2" in lines
    assert all(len(line) <= 80 for line in lines)


@menu_row
def downloads_use_the_fetch_key(t):
    fetched = catalogue(t)
    answer(t.mp, "1", "1", "y", "n", "q")
    assert cli.run_filters_flow(str(t.tmp)) == 0
    assert fetched == ["Gaia/DR2"]


@menu_row
def menu_3_shows_filters_info_for_a_picked_instrument(t):
    fetched = gaia(t)
    _no_banner(t)
    looked_up = summary_inputs(t.mp, availability(facility="GAIA", instrument="GAIA0"), {
        "info": {"licence": LICENCE, "references": ["a"]}, "properties": None, "errors": []})
    prompts = answer(t.mp, "3", "1", "4", "n", "q", "q")
    assert cli._run_menu() == 0
    out = t.capsys.readouterr().out
    assert "\nGAIA/GAIA0  (2 bands)\n  Published curves contain" in out
    assert "Cite                     : 1 references; in Python: Filters.cite('GAIA/GAIA0')" in out
    assert looked_up == ["GAIA/GAIA0"] and fetched == []  # nothing downloaded
    assert prompts[4] == "> "  # back at the facility picker, then the menu


def _lsst(t, parts):
    """Facility LSST with instrument LSST, whose parts are ``parts`` and whose curves download to
    <tmp>/LSST/LSST; returns the compose calls."""
    from sed_tools.filter_library import FilterLibrary

    _no_banner(t)
    for listing in ("facilities", "instruments"):
        t.mp.setattr(FilterLibrary, listing, lambda self, *facility: {"entries": [
            {"key": "LSST", "label": "LSST", "sources": ["njm"]}], "errors": []})
    t.mp.setattr(FilterLibrary, "published", lambda self, f, i: t.tmp / f / i)
    composed = []
    t.mp.setattr(Filters, "compose", classmethod(lambda cls, *a, **k: composed.append((a, k)) or t.tmp / "product"))
    summary_inputs(t.mp, availability(facility="LSST", instrument="LSST", **parts))
    return composed


@menu_row
def menu_4_builds_a_curve_from_the_parts_of_a_picked_instrument(t):
    parts = {"buildable": ["filter", "filter+optics+qe"], "components": ["filter", "optics", "qe"]}
    composed = _lsst(t, parts)
    t.mp.setattr(cli.filters, "_installed_availability", lambda directory: availability(
        facility="LSST", instrument="LSST", **parts))
    prompts = answer(t.mp, "4", "1", "1", "2", "q")
    assert cli._run_menu() == 0
    assert ("  3) Filter set information\n"
            "  4) Build a filter curve from its parts (filter, optics, camera, atmosphere)\n"
            ) in t.capsys.readouterr().out
    assert prompts[3:] == ["Choose a recipe [1]: ", "> "]  # no detector question before building
    assert composed == [((t.tmp / "LSST" / "LSST",), {"filter_root": t.tmp, "components": ["filter", "optics", "qe"],
                                                       "allow_heavy": False, "atmosphere": None})]


@menu_row
def menu_4_says_in_one_line_when_an_instrument_has_no_parts(t):
    composed = _lsst(t, {"buildable": [], "components": [],
                         "component_availability": {"status": "total-only", "builder": None}})
    t.mp.setattr(cli.filters, "_installed_availability", lambda directory: pytest.fail("nothing is downloaded"))
    answer(t.mp, "4", "1", "1", "q")
    assert cli._run_menu() == 0
    out = t.capsys.readouterr().out
    assert "\nLSST/LSST has no separate parts to build a filter curve from.\n" in out and composed == []


@menu_row
def band_descriptions_wrap_to_the_terminal(t):
    props = {"G": {"WavelengthPivot": 6217.6, "FWHM": 4548.2, "DetectorType": "photon"}}
    info = {"bands": {"G": {"MagSys": "Vega", "Description": "GAIA Gbp filter, DR2 Maíz Apellániz & Weiler "
                                                            "revision (faint, G>10.87)"}}}
    cli.print_catalogue_entry("GAIA/GAIA", {"info": info, "properties": {"bands": props}, "errors": []})
    lines = t.capsys.readouterr().out.splitlines()
    assert all(len(line) <= 80 for line in lines) and len(lines) > 2


# ----------------------------------------------------------------------
# combine, coverage, grid cuts without a terminal, --auto limits, older spellings, the cached catalogue
# ----------------------------------------------------------------------
def _combine(t):
    """Grids Kurucz and Phoenix, and two combined ones; combining records (grid names, output) instead."""
    for name, combined in (("Kurucz", False), ("Phoenix", False), ("combined_models", True), ("older_mix", True)):
        (t.tmp / name).mkdir(parents=True)
        columns = "#file_name,teff,logg,meta" + (",source_model,original_file" if combined else "")
        (t.tmp / name / "lookup_table.csv").write_text(columns + "\n")
    built = []
    t.mp.setattr("sed_tools.cli.grids._build_combined_grid", lambda models, out: built.append(
        ([name for name, _ in models], Path(out).name)))
    return built


@menu_row
def non_interactive_combine_leaves_out_earlier_combined_grids(t):
    built = _combine(t)
    assert run_cli(t.mp, "combine", "--base", str(t.tmp), "--non-interactive") == 0
    assert [(sorted(names), out) for names, out in built] == [(["Kurucz", "Phoenix"], "combined_models")]
    out = t.capsys.readouterr().out
    assert "Leaving out combined_models: it is the output grid" in out
    assert "Leaving out older_mix: an earlier combined grid (name it with --grids to include it)" in out


#: name: (the grids named, what stderr says).
COMBINE_REFUSES = {
    "combine_refuses_unknown_grids_and_combining_into_an_input_unknown": (
        ["--grids", "Nope"], "No spectral grid with a lookup table named Nope"),
    "combine_refuses_unknown_grids_and_combining_into_an_input_output": (
        ["--grids", "Kurucz", "combined_models"], "The output grid combined_models is also one of the grids"),
}


def _combine_refuses(t, argv, message):
    narrow_terminal(t)
    built = _combine(t)
    assert run_cli(t.mp, "combine", "--base", str(t.tmp), *argv) == 1
    assert message in t.capsys.readouterr().err and built == []


@menu_row
def combine_prompt_has_a_space_and_refuses_an_input_name(t):
    built = _combine(t)
    t.mp.setattr("sed_tools.combine_stellar_atm.select_models_interactive",
                 lambda dirs: [d for d in dirs if d[0] in ("Kurucz", "Phoenix")])
    answer(t.mp, "Kurucz")
    with pytest.raises(ValueError, match="The output grid Kurucz is also one of the grids to combine"):
        cli.run_combine_flow(base_dir=str(t.tmp))
    prompts = answer(t.mp, "mine")
    cli.run_combine_flow(base_dir=str(t.tmp))
    assert prompts[0] == "What should the combined spectral grid be called? [combined_models] "
    assert [(sorted(names), out) for names, out in built] == [(["Kurucz", "Phoenix"], "mine")]


@menu_row
def coverage_prints_each_failure_once(t):
    t.mp.setattr(SED, "coverage", classmethod(lambda cls, name, **kwargs: _raise(
        ValueError("lookup_table.csv has no Teff column"))))
    assert run_cli(t.mp, "coverage", "--grids", "B", "--no-plot", "--base", str(t.tmp)) == 1
    assert t.capsys.readouterr().out.count("lookup_table.csv has no Teff column") == 1


@menu_row
def njm_cuts_are_asked_in_a_terminal(t):
    t.mp.setattr("sys.stdin", Stdin(True))
    t.mp.setattr(cli.grids, "prompt_njm_axis_cuts", lambda name, g, model_url=None: {"teff_range": (1.0, 2.0)})
    assert cli._download_cuts("G", "njm", None, {"model_url": "u"}, {}) == {"teff_range": (1.0, 2.0)}


@menu_row
def auto_training_takes_trials_and_epochs(t):
    import sed_tools.ml

    calls = []
    t.mp.setattr(importlib.util, "find_spec", lambda name, *a: object())
    t.mp.setattr(sed_tools.ml, "auto_train_generator", lambda **kwargs: calls.append(kwargs))
    assert run_cli(t.mp, "ml_generator", "--auto", "--library", "lib", "--output", "out",
                   "--trials", "2", "--epochs", "1") == 0
    assert run_cli(t.mp, "ml_generator", "--auto", "--library", "lib", "--output", "out") == 0
    assert calls == [{"library": "lib", "output": "out", "n_trials": 2, "epochs": 1},
                     {"library": "lib", "output": "out"}]


@menu_row
def grids_without_a_terminal_or_names_says_to_use_grids(t):
    t.mp.setattr("sed_tools.msg_spectra_grabber.MSGSpectraGrabber",
                 lambda *args, **kwargs: SimpleNamespace(discover_models=lambda: ["bt-cond"]))
    t.mp.setattr("sys.stdin", Stdin(False))
    with pytest.raises(ValueError, match=r"No terminal to choose from the 1 spectral grids found; name them with "
                                         r"--grids NAME \.\.\. \(or --grids all\)"):
        cli.run_spectra_flow("msg", base_dir=str(t.tmp))


@menu_row
def the_picker_finds_an_instrument_installed_under_an_older_spelling(t):
    # 0.4.1 wrote Gemini/GMOS-N; NJM's name is GMOS_N. The picker must see the installed copy, not offer a second one.
    from sed_tools.filter_library import FilterLibrary

    installed = t.tmp / "Gemini" / "GMOS-N"
    installed.mkdir(parents=True)
    (installed / "g.dat").write_text("Wavelength,Transmission\n4000,0.1\n5000,0.5\n6000,0.1\n")
    (installed / "GMOS-N").write_text("g.dat\n")
    library = FilterLibrary(t.tmp)
    facility, instrument, directory = cli._picked_names(library, {"key": "Gemini"}, {"key": "GMOS_N"})
    assert (facility, instrument) == ("Gemini", "GMOS_N")
    assert directory is None or directory.resolve() == installed.resolve()
    assert library.installed_directory("Gemini", "GMOS_N").resolve() == installed.resolve()


@menu_row
def filters_info_says_when_the_catalogue_is_the_cached_copy(t):
    cli.print_catalogue_entry("Fac/Cam", {"info": {"description": "d"}, "properties": None, "errors": [],
                                          "origin": {"source": "cache", "checked": 0.0 + 1761600000,
                                                     "njm_unreachable": True}})
    out = " ".join(t.capsys.readouterr().out.split())  # the field wraps to the terminal
    assert "Catalogue : from the cached copy, checked 2025-10-27 21:20 UTC; NJM could not be asked" in out


@menu_row
def cli_passes_the_atmosphere_site_and_unstated_airmass(t):
    seen = {}
    t.mp.setattr(Filters, "compose", classmethod(lambda cls, target, **kw: seen.update(kw)))
    assert run_cli(t.mp, "filters-compose", "2MASS/2MASS", "--base", str(t.tmp), "--components", "filter",
                   "--airmass", "unstated", "--atmosphere-site", "ctio_2mass") in (0, None)
    assert seen["atmosphere"] == {"airmass": None, "pwv": None, "site": "ctio_2mass"}


# ----------------------------------------------------------------------
# Without the optional packages: models listed with any partial install, an ML action refused only when it
# is chosen, one line naming what to install, and tracebacks kept for a module that is not optional
# ----------------------------------------------------------------------
@row
def filter_combine_intro_fits_80_columns(t):
    answer(t.mp, "")
    cli.run_filter_combine_flow(base_dir=str(t.tmp))
    out = t.capsys.readouterr().out
    assert "MESA-compatible filter folder and index file" in out
    assert wide_lines(out, str(t.tmp)) == []  # was 87 wide


FULL = 'pip install "sed-tools[full]"'

#: The packages missing in each partial install: none of the three, scipy only, torch only.
INSTALLS = {"none": ("scipy", "torch", "sklearn"), "scipy-only": ("torch", "sklearn"),
            "torch-only": ("scipy", "sklearn")}

#: (command, menu choice, what each install lacks for it); None: the action is not refused.
REFUSALS = [
    ("ml_completer", "1", {"none": "torch, scikit-learn", "scipy-only": "torch, scikit-learn",
                           "torch-only": "scikit-learn"}),
    ("ml_completer", "2", {"none": "scipy, torch", "scipy-only": "torch", "torch-only": "scipy"}),
    ("ml_generator", "1", {"none": "torch, scikit-learn", "scipy-only": "torch, scikit-learn",
                           "torch-only": "scikit-learn"}),
    ("ml_generator", "2", {"none": "torch", "scipy-only": "torch", "torch-only": None}),
    ("ml_generator", "3", {"none": "torch", "scipy-only": "torch", "torch-only": None}),
]


def _without(t, *modules):
    """Make ``modules`` uninstalled for this row: importing them (or a submodule another row imported
    already) fails, and find_spec finds nothing."""
    for name in modules:
        for loaded in [key for key in sys.modules if key == name or key.startswith(name + ".")] or [name]:
            t.mp.setitem(sys.modules, loaded, None)


def _listing_models_works_with_any_partial_install(t, install):
    # 0.4.1 listed models with scipy but no torch; that must keep working (and now needs neither)
    _without(t, *INSTALLS[install])
    models = t.tmp / "models"  # one completer and one generator, as list_models reads them
    (models / "my_completer_model").mkdir(parents=True)
    (models / "my_completer_model" / "config.json").write_text('{"version": "1", "framework": "pytorch"}')
    (models / "my_generator_model").mkdir()
    (models / "my_generator_model" / "config.json").write_text('{"model_type": "sed_generator"}')
    answer(t.mp, "3")
    assert run_cli(t.mp, "ml_completer", "--base", str(t.tmp), "--models-dir", str(models)) == 0
    answer(t.mp, "4")
    assert run_cli(t.mp, "ml_generator", "--base", str(t.tmp), "--models-dir", str(models)) == 0
    captured = t.capsys.readouterr()
    assert "my_completer_model" in captured.out and "my_generator_model" in captured.out
    assert captured.err == ""


def _an_ml_action_is_refused_when_chosen_if_it_needs_a_missing_package(t, command, choice, lacking, install):
    _without(t, *INSTALLS[install])
    answer(t.mp, choice)  # refused before any of the action's questions (a further one fails)
    status = run_cli(t.mp, command, "--base", str(t.tmp), "--models-dir", str(t.tmp / "models"))
    err = t.capsys.readouterr().err
    if lacking[install] is None:  # generating needs only torch: it goes on (here: no generator to choose)
        assert status == 0 and err == ""
    else:
        assert status == 1 and err == f"Error: missing {lacking[install]}: {FULL}\n"


def _densifier_inputs(t):
    write_flux_cube(t.tmp / "Kurucz" / "flux_cube.bin", [5000, 7000], [4.0], [0.0], [1000.0, 2000.0],
                    np.ones((2, 1, 1, 2)))
    model = t.tmp / "models" / "gen"
    model.mkdir(parents=True)
    return model


@row
def densify_with_an_ml_model_without_torch_says_what_to_install(t):
    model = _densifier_inputs(t)
    _without(t, "torch")
    assert run_cli(t.mp, "densify", "--flux-cube", str(t.tmp / "Kurucz" / "flux_cube.bin"),
                   "--output", str(t.tmp / "dense" / "flux_cube.bin"), "--ml-model", str(model)) == 1
    assert t.capsys.readouterr().err == f"Error: missing torch: {FULL}\n"
    assert not (t.tmp / "dense").exists()


def _menu_ml_flows_in_the_row_folder(t):
    """Point menu options 8, 9 and 10 at the row's folder: they create their grid and models folders."""
    for name in ("run_ml_completer_flow", "run_ml_generator_flow"):
        t.mp.setattr(cli, name, lambda flow=getattr(cli, name): flow(str(t.tmp), str(t.tmp / "models")))
    densifier = cli.run_grid_densifier_flow
    t.mp.setattr(cli, "run_grid_densifier_flow", lambda: densifier(base_dir=str(t.tmp)))


@row
def menu_ml_options_open_and_refuse_only_the_action_that_needs_a_package(t):
    _without(t, *INSTALLS["none"])
    model = _densifier_inputs(t)
    _menu_ml_flows_in_the_row_folder(t)
    # 8 and 9 open and go back with 0, as in 0.4.1; their training is refused when chosen; 10 is refused
    # once an ML model is named, before the output question
    answer(t.mp, "8", "0", "9", "0", "8", "1", "9", "1", "10", "1", "", str(model), "q")
    assert cli._run_menu() == 1  # the refused actions failed, so the session ends with status 1
    captured = t.capsys.readouterr()
    assert captured.err.splitlines() == [f"Error: missing torch, scikit-learn: {FULL}"] * 2 + [
        f"Error: missing torch: {FULL}"]
    assert "Traceback" not in captured.out + captured.err
    assert captured.out.count("0/q) Quit") == 6  # shown first, then again after each of the 5 options
    assert not (t.tmp / "Kurucz_dense").exists()


@row
def menu_9_opens_and_returns_without_torch(t):
    _without(t, *INSTALLS["none"])
    _menu_ml_flows_in_the_row_folder(t)
    answer(t.mp, "9", "0", "q")
    assert cli._run_menu() == 0  # final test, check 16: as in 0.4.1
    assert t.capsys.readouterr().err == ""


@row
def combine_without_scipy_says_what_to_install(t):
    def needs_scipy():
        from scipy.spatial import KDTree  # noqa: F401  (as combine_stellar_atm does)

    _without(t, "scipy")
    assert cli._run_menu_flow(needs_scipy) is False
    assert t.capsys.readouterr().err == f"Error: missing scipy: {FULL}\n"


@row
def a_missing_module_that_is_not_optional_keeps_its_traceback(t):
    with pytest.raises(ModuleNotFoundError):
        cli._run_menu_flow(lambda: _raise(ModuleNotFoundError("No module named 'sed_tools_typo'",
                                                              name="sed_tools_typo")))


# ----------------------------------------------------------------------
# The pickers at 80 columns: a label is cut only when the terminal is narrower than it
# ----------------------------------------------------------------------
def _picker_page(t, labels, columns, **kwargs):
    from sed_tools.ui_utils import prompt_choice

    t.mp.setenv("COLUMNS", str(columns))
    t.mp.setenv("LINES", "40")
    answer(t.mp, "q")
    assert prompt_choice(labels, "Local filter sets to combine", use_color=False, **kwargs) is None
    return t.capsys.readouterr().out.splitlines()


@row
def combine_picker_shows_a_long_product_label_whole_at_80_columns(t):
    long = "2MASS/2MASS/products/filter+atm_ctio_2mass_amunstated (3 filters)"
    lines = _picker_page(t, [long, "Generic/Johnson (7 filters)"], 80, multi=True, page_size=50, max_cols=2)
    assert any(line.strip() == f"1. {long}" for line in lines)  # was cut to '(3 filter…'
    assert max(len(line) for line in lines) <= 80


@row
def picker_keeps_its_columns_for_short_names(t):
    lines = _picker_page(t, [f"name{i:02d}" for i in range(30)], 80)
    rows = [line for line in lines if line.lstrip()[:1].isdigit()]
    assert len(rows) == 10 and all(line.count(".") == 3 for line in rows)  # three columns, as before


@row
def picker_shows_the_longest_label_in_full(t):
    from sed_tools import ui_utils

    t.mp.setattr(ui_utils, "terminal_width", lambda default=80: 80)
    answer(t.mp, "1")
    assert ui_utils.prompt_choice(["alpha", "beta"], "Things", use_color=False) == 0
    out = t.capsys.readouterr().out
    # The longest label used to be cut 2 characters short ("alp", "bet").
    assert "   1. alpha  " in out and "   2. beta" in out


# ----------------------------------------------------------------------
# Prompts and pickers: numbers asked once, names only, descriptions once, full names in
# columns, long labels cut with an ellipsis, and a count before downloading several
# ----------------------------------------------------------------------
@menu_row
def number_prompts_take_a_number_a_default_or_a_word(t):
    answer(t.mp, "1.5")
    assert cli._ask_number("Airmass: ") == 1.5
    answer(t.mp, "")
    assert cli._ask_number("PWV: ", default=None) is None
    answer(t.mp, "UNSTATED")
    assert cli._ask_number("Airmass: ", words=("unstated",)) == "unstated"


@menu_row
def composition_offers_the_bands_that_work(t):
    calls = []

    def compose(cls, directory, **kwargs):
        calls.append(kwargs.get("bands"))
        if kwargs.get("bands") is None:
            raise ComponentError("am1.00_pwv1.dat covers 9000-56000 A but 70% of the band lies beyond it")

    t.mp.setattr(Filters, "compose", classmethod(compose))
    t.mp.setattr(cli.filters, "_composable_bands", lambda d, request: (["J", "H"], {"Z": "covers 9000-56000 A"}))
    answer(t.mp, "y")
    request = {"components": ["filter"], "variants": None, "detector_type": None, "allow_heavy": False,
               "atmosphere": {"site": "paranal", "airmass": 1.0, "pwv": 1.0}}
    cli._compose_the_bands_that_work(t.tmp / "Paranal" / "VISTA", request, ComponentError("no"))
    assert calls == [["J", "H"]]
    assert "Z: covers 9000-56000 A" in t.capsys.readouterr().out


@menu_row
def the_pickers_show_names_only(t):
    gaia(t)
    answer(t.mp, "1", "q")
    assert cli.run_filters_flow(str(t.tmp)) == 0
    out = t.capsys.readouterr().out
    assert "   1. GAIA" in out and "Gaia mission" not in out  # no facility descriptions are invented
    pickers = out.split("\nGAIA: 5 instruments\n")[0] + out.split("Instruments for GAIA")[1]
    assert not any(text in pickers for text in ("[njm", "[svo", "Tags show", "Evans et al."))
    assert cli._picker_labels(GAIA_INSTRUMENTS + [{"key": "Other"}]) == [
        "DR2", "DR2r", "GAIA", "GAIA0", "Gaia2m", "Other"]


@menu_row
def picking_a_facility_lists_its_instruments_with_their_descriptions_once(t):
    gaia(t)
    answer(t.mp, "1", "/DR2", "q")
    cli.run_filters_flow(str(t.tmp))
    out = t.capsys.readouterr().out
    block = out.split("\nGAIA: 5 instruments\n")[1].split("\n\nInstruments for GAIA")[0]
    assert block.splitlines() == [
        "  DR2     Gaia DR2 passbands (Evans et al. 2018)",
        "  DR2r    Gaia DR2 revised passbands (Evans et al. 2018)",
        "  GAIA    Gaia DR3 passbands + BP bright/faint (Maíz Apellániz & Weiler 2018)",
        "  GAIA0   Gaia pre-launch passbands (Jordi et al. 2010)",
        "  Gaia2m  Revised passbands for Gaia DR2."]
    assert out.count("GAIA: 5 instruments") == 1  # not again when the list is filtered


@menu_row
def the_instrument_picker_shows_full_names_in_columns(t):
    gaia(t)
    answer(t.mp, "1", "q")
    cli.run_filters_flow(str(t.tmp))
    rows = t.capsys.readouterr().out.split("Instruments for GAIA\n")[1].splitlines()[1:3]
    assert re.fullmatch(r"\s+1\. DR2\s+3\. GAIA\s+5\. Gaia2m\s*", rows[0])  # 0.4.1 cut these to 'D', 'GA', 'Gaia'
    assert re.fullmatch(r"\s+2\. DR2r\s+4\. GAIA0\s*", rows[1])
    assert all(len(row) <= 80 for row in rows)


@menu_row
def picker_cuts_long_labels_with_an_ellipsis(t):
    t.mp.setenv("COLUMNS", "40")
    answer(t.mp, "q")
    cli.prompt_choice(["x" * 50], "Things")
    assert "x" * 23 + "…" in t.capsys.readouterr().out


@menu_row
def downloading_several_facilities_asks_first_with_the_count(t):
    fetched = catalogue(t)
    t.mp.setattr("sys.stdin", Stdin(True))  # only a terminal is asked; piped input downloads, as in 0.4.1
    prompts = answer(t.mp, "1,2", "n")
    assert cli.run_filters_flow(str(t.tmp)) == 0
    assert fetched == []
    assert prompts[-1] == "Download all 4? [y/N] "
    assert "4 instruments in 2 facilities: GAIA (2), 2MASS (2)" in t.capsys.readouterr().out


@menu_row
def every_number_question_is_the_same_one(t):
    from sed_tools import ml_sed_completer, ml_sed_generator, ui_utils

    assert cli._ask_number is ml_sed_generator._ask_number is ml_sed_completer._ask_number is ui_utils.ask_number


def _narrow(t, scenario):
    narrow_terminal(t)
    scenario(t)


def test_cli(rows):
    rows(SCENARIOS)
    rows(FILTERS_INFO, _filters_info)
    rows(USAGE, _usage_error)
    rows({f"{module}_help_prints_the_usage_and_exits_0": module for module in HELP}, _help)
    rows({f"{module}_a_bad_argument_prints_the_usage_and_exits_2": module for module in HELP}, _bad_argument)
    rows(MAINS, _in_its_folder_run)
    rows(MENUS, _narrow)
    rows(MENU_EXITS, _menu_survives_a_workflow_that_exits)
    rows(AVAILABILITY, _availability_lists)
    rows(AIRMASS, _airmass_defaults)
    rows(COMBINE_REFUSES, _combine_refuses)
    rows({f"listing_models_works_with_any_partial_install_{i.replace('-', '_')}": i for i in INSTALLS},
         _listing_models_works_with_any_partial_install)
    rows({f"an_ml_action_is_refused_when_chosen_if_it_needs_a_missing_package_{c}_{n}_{i.replace('-', '_')}":
          (c, n, lacking, i) for c, n, lacking in REFUSALS for i in INSTALLS},
         _an_ml_action_is_refused_when_chosen_if_it_needs_a_missing_package)
