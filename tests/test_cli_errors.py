"""Failures in the CLI are reported with a non-zero exit or a warning: filters-info, coverage, the
spectra and filter flows, rebuild, densify, mesa_prepare, and the prompts that ask again.

No test uses the network: the autouse fixture fails any real request.
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from sed_tools.api import SED

from helpers_api import CLEAN, GRABBERS, fake_grabber, tagged
from helpers_api import filters, no_network  # noqa: F401 (fixtures)
from helpers_cli import write_lookup


# ----------------------------------------------------------------------
# CLI exit status
# ----------------------------------------------------------------------
def run_cli(monkeypatch, *argv):
    import sys

    from sed_tools import cli

    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    return cli.main()


@pytest.mark.parametrize("source_errors,catalogue_errors,offline,expected", [
    ([], [], False, 0),
    (["SVO: no metadata for Fac/Cam.g"], [], False, 1),
    # NJM's catalogue is an extra: without it the availability report still stands.
    ([], ["NJM unreachable"], False, 0),
    ([], ["info.json is not cached and working offline"], True, 0),
])
def test_filters_info_exit_status(filters, monkeypatch, source_errors, catalogue_errors, offline, expected):
    from sed_tools import cli

    info = {"facility": "Fac", "instrument": "Cam", "bands": [], "published_source": "", "published": "",
            "published_bands": {}, "components": [], "partial_components": {}, "buildable": [],
            "component_status": "", "component_availability": {"status": "unreviewed", "builder": None},
            "missing_files": [], "errors": source_errors, "atmosphere": False, "platform": "ground",
            "detector_type": "", "detector_type_sources": {}, "availability_source": "svo",
            "sources_failed": {"svo": [e.split(": ", 1)[1] for e in source_errors]} if source_errors else {},
            "fallback_from": []}
    monkeypatch.setattr(filters, "available", classmethod(lambda cls, *args, **kwargs: info))
    monkeypatch.setattr(cli, "_catalogue_entry", lambda *args, **kwargs:
                        {"info": None, "properties": None, "errors": catalogue_errors})
    argv = ["filters-info", "Fac/Cam"] + (["--offline"] if offline else [])
    assert run_cli(monkeypatch, *argv) == expected


def test_coverage_exits_1_when_there_are_no_local_grids(tmp_path, monkeypatch, capsys):
    assert run_cli(monkeypatch, "coverage", "--base", str(tmp_path)) == 1
    assert "No local grids found" in capsys.readouterr().out


def test_running_the_cli_module_passes_on_the_exit_status(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-m", "sed_tools.cli", "coverage", "--base", str(tmp_path)],
                          cwd=root, capture_output=True, text=True)
    assert done.returncode == 1, done.stderr


# ----------------------------------------------------------------------
# Spectral grid download workflow
# ----------------------------------------------------------------------
@pytest.fixture
def grid_flow(monkeypatch):
    """run_spectra_flow with fake grabbers, cleaning and product builds; returns the grids built."""
    from sed_tools import cli

    built = []

    def install(cleaned=CLEAN, cube_built=True, **grabbers):
        for name, cls_name in (("njm", "NJMSpectraGrabber"), ("svo", "SVOSpectraGrabber"),
                               ("msg", "MSGSpectraGrabber"), ("mast", "MASTSpectraGrabber")):
            monkeypatch.setattr(GRABBERS[name], grabbers.get(name, fake_grabber()))
        monkeypatch.setattr(cli.grids, "clean_model_dir", lambda *a, **k: cleaned)
        monkeypatch.setattr(cli.grids, "undeclared_flux_unit", lambda *a, **k: None)  # cleaning is faked too

        def build(name, model_dir, *args):
            if not cube_built:
                raise RuntimeError(f"No flux cube was built in {model_dir}")
            built.append(name)
            return []  # what was not done, as _build_data_products returns

        monkeypatch.setattr(cli.grids, "_build_data_products", build)
        return built
    return install


def test_spectra_flow_builds_what_it_can_then_lists_every_grid_not_completed(grid_flow, tmp_path, capsys):
    from sed_tools.cli import run_spectra_flow

    built = grid_flow(svo=fake_grabber(["Good", "NoMeta", "Empty"]),
                      msg=fake_grabber(["Good"], metadata={"m": 1}, downloaded=1),
                      mast=fake_grabber(["Empty"], metadata={"m": 1}, downloaded=0))
    with pytest.raises(RuntimeError) as excinfo:
        run_spectra_flow("all", base_dir=str(tmp_path), models=["Good", "NoMeta", "Empty", "Missing"])
    message = str(excinfo.value)
    assert built == ["Good"]
    assert "NoMeta: RuntimeError: no source could supply it. Tried: svo: no metadata" in message
    assert ("Empty: RuntimeError: no source could supply it. Tried: svo: no metadata for Empty; "
            "mast: no spectra downloaded") in message
    assert "Missing: not listed by any source" in message
    assert "Good" not in message
    assert "Done." not in capsys.readouterr().out


def test_spectra_flow_carries_on_after_a_grid_whose_build_fails(grid_flow, tmp_path):
    from sed_tools.cli import run_spectra_flow

    grid_flow(cube_built=False, svo=fake_grabber(["A", "B"], metadata=[{"fid": 1}], downloaded=1))
    with pytest.raises(RuntimeError) as excinfo:
        run_spectra_flow("svo", base_dir=str(tmp_path), models=["A", "B"])
    assert "A: RuntimeError: No flux cube was built in" in str(excinfo.value)
    assert "B: RuntimeError: No flux cube was built in" in str(excinfo.value)


def test_spectra_flow_completes_quietly_when_every_grid_is_built(grid_flow, tmp_path, capsys):
    from sed_tools.cli import run_spectra_flow

    grid_flow(svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], downloaded=1))
    run_spectra_flow("svo", base_dir=str(tmp_path), models=["Grid"])
    assert capsys.readouterr().out.rstrip().endswith("Done.")


def test_spectra_flow_raises_when_no_grid_is_discovered(grid_flow, tmp_path):
    from sed_tools.cli import run_spectra_flow

    grid_flow(svo=fake_grabber([]))  # as a grabber that could not read its index, after its warning
    with pytest.raises(RuntimeError, match="No spectral grids discovered"):
        run_spectra_flow("svo", base_dir=str(tmp_path), models=["Grid"])


def test_grids_command_fails_when_a_grid_is_not_found(grid_flow, tmp_path, monkeypatch, capsys):
    grid_flow(svo=fake_grabber(["Grid"]))
    assert run_cli(monkeypatch, "grids", "--source", "svo", "--grids", "Missing", "--base", str(tmp_path)) == 1
    assert "sed-tools: error: " in capsys.readouterr().err


def test_axis_cut_prompt_asks_again_after_an_unparseable_range(monkeypatch, capsys):
    from sed_tools.cli import prompt_njm_axis_cuts

    class Grabber:
        def get_grid_info(self, name, model_url=None):
            return None

    answers = iter(["3500 to 8000", "3500,8000", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cuts = prompt_njm_axis_cuts("Grid", Grabber())
    assert cuts == {"teff_range": (3500.0, 8000.0), "logg_range": None, "meta_range": None, "wl_range": None}
    assert "Could not parse range from '3500 to 8000'" in capsys.readouterr().out


# ----------------------------------------------------------------------
# Flux cube builds and the rebuild workflow
# ----------------------------------------------------------------------
@pytest.fixture
def rebuild(tmp_path, monkeypatch):
    """Local grids A, B and C with run_rebuild_flow's cleaning and builds faked; B's cleaning fails."""
    from sed_tools import cli

    for name in ("A", "B", "C"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "s.txt").write_text("1000 1\n")
    rebuilt = []

    def clean(model_dir, **kwargs):
        if model_dir.endswith("B"):
            raise ValueError("corrupt header in s.txt")
        rebuilt.append(Path(model_dir).name)
        return dict(CLEAN)

    monkeypatch.setattr(cli.grids, "clean_model_dir", clean)
    monkeypatch.setattr(cli.grids, "regenerate_lookup_table", lambda model_dir: None)
    monkeypatch.setattr(cli.grids, "build_h5_bundle_from_txt", lambda model_dir, out: None)
    monkeypatch.setattr("sed_tools.cli.grids._build_flux_cube", lambda model_dir: Path(model_dir) / "flux_cube.bin")
    return rebuilt


def test_rebuild_rebuilds_every_grid_it_can_then_lists_the_rest(tmp_path, rebuild):
    from sed_tools.cli import run_rebuild_flow

    with pytest.raises(RuntimeError) as excinfo:
        run_rebuild_flow(base_dir=str(tmp_path), models=["A", "B", "C", "Nope"])
    assert rebuild == ["A", "C"]
    assert "B: ValueError: corrupt header in s.txt" in str(excinfo.value)
    assert "Nope: not a local spectral grid under" in str(excinfo.value)


def test_rebuild_asks_again_for_an_out_of_range_selection(tmp_path, rebuild, monkeypatch, capsys):
    from sed_tools.cli import run_rebuild_flow

    answers = iter(["7", "1,3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    run_rebuild_flow(base_dir=str(tmp_path))
    assert rebuild == ["A", "C"]
    assert "Selection out of bounds: 7" in capsys.readouterr().out


def test_rebuild_and_combine_raise_when_there_are_no_local_grids(tmp_path):
    from sed_tools.cli import run_combine_flow, run_rebuild_flow

    with pytest.raises(FileNotFoundError, match="No local spectral grids found"):
        run_rebuild_flow(base_dir=str(tmp_path), models=["A"])
    with pytest.raises(FileNotFoundError, match="No spectral grids found"):
        run_combine_flow(base_dir=str(tmp_path), interactive=False)


# ----------------------------------------------------------------------
# Interactive filter downloads
# ----------------------------------------------------------------------
@pytest.fixture
def filter_browser(tmp_path, monkeypatch, filters):
    """Facilities A and B with instruments Cam1 and Cam2; fetching B/Cam2 fails."""
    from sed_tools import cli
    from sed_tools.filter_library import FilterLibrary

    listing_errors = {}
    monkeypatch.setattr(FilterLibrary, "facilities", lambda self: {
        "entries": [{"key": "A", "label": "A"}, {"key": "B", "label": "B"}], "errors": []})
    monkeypatch.setattr(FilterLibrary, "instruments", lambda self, facility: {
        "entries": [{"key": "Cam1", "label": "Cam1"}, {"key": "Cam2", "label": "Cam2"}],
        "errors": listing_errors.get(facility, [])})
    fetched = []

    def fetch(self, facility, instrument):
        if (facility, instrument) == ("B", "Cam2"):
            raise requests.ConnectionError("SVO down")
        fetched.append(f"{facility}/{instrument}")
        return tmp_path / facility / instrument

    monkeypatch.setattr(FilterLibrary, "published", fetch)  # the browser downloads with its own library
    monkeypatch.setattr(cli.filters, "_offer_composition", lambda directory: None)
    # What the browser describes before asking to download (Filters.available, NJM's catalogue entry).
    monkeypatch.setattr(FilterLibrary, "available", lambda self, facility, instrument, remote=True: {
        "facility": facility, "instrument": instrument, "bands": ["g"], "published": "filter",
        "published_bands": {}, "components": [], "partial_components": {}, "component_status": None,
        "errors": [], "atmosphere": False, "platform": "ground", "variants": {}})
    monkeypatch.setattr(cli.filters, "_catalogue_entry", lambda *a, **k: {"info": None, "properties": None, "errors": []})
    return SimpleNamespace(fetched=fetched, listing_errors=listing_errors)


def answer(monkeypatch, *answers):
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))


def test_filter_batch_downloads_the_rest_then_lists_failures(filter_browser, monkeypatch, capsys):
    from sed_tools.cli import run_filters_flow

    filter_browser.listing_errors["A"] = ["SVO A: timed out"]
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))  # a terminal: the count is confirmed
    answer(monkeypatch, "1,2", "y")
    assert run_filters_flow() == 1
    out = capsys.readouterr().out
    assert filter_browser.fetched == ["A/Cam1", "A/Cam2", "B/Cam1"]
    assert "B/Cam2: SVO down" in out
    assert "A: instrument list incomplete: SVO A: timed out" in out
    assert "Cannot fetch: SVO down" in out


def test_filter_browser_reports_a_failed_build_when_browsing_ends(filter_browser, monkeypatch, capsys):
    from sed_tools import cli

    def fail(directory):
        raise ValueError("Airmass and PWV must be numbers (could not convert string to float: 'x')")

    monkeypatch.setattr(cli.filters, "_offer_composition", fail)
    # Facility A, instrument 1, download, then quit from the instrument list.
    answer(monkeypatch, "1", "1", "y", "y", "q")
    assert cli.run_filters_flow() == 1
    out = capsys.readouterr().out
    assert "Composed product of A/Cam1 not built: Airmass and PWV must be numbers" in out
    assert "A/Cam1 composed product: Airmass and PWV must be numbers" in out.split("Failed (1):")[1]


def test_filter_browser_ends_quietly_when_nothing_failed(filter_browser, monkeypatch):
    from sed_tools.cli import run_filters_flow

    answer(monkeypatch, "1", "2", "y", "n", "q")
    run_filters_flow()
    assert filter_browser.fetched == ["A/Cam2"]


def test_filters_command_without_a_target_fails_when_a_download_failed(filter_browser, monkeypatch, capsys):
    answer(monkeypatch, "2", "2", "y", "n", "q")
    assert run_cli(monkeypatch, "filters") == 1
    assert "B/Cam2: SVO down" in capsys.readouterr().out.split("Failed (1):")[1]


def test_offer_composition_asks_again_for_an_airmass_that_is_not_a_number(tmp_path, monkeypatch, filters, capsys):
    from sed_tools import cli

    info = {"buildable": ["filter"], "atmosphere": True, "atmosphere_site": "paranal", "atmosphere_sites": None,
            "component_availability": {"status": "supported", "large_download": False}}
    built = []
    monkeypatch.setattr(filters, "available", classmethod(lambda cls, *a, **k: info))
    monkeypatch.setattr(filters, "atmospheres", classmethod(lambda cls, *a, **k: {}))
    monkeypatch.setattr(cli.filters, "print_filter_availability", lambda info: None)
    monkeypatch.setattr(filters, "compose", classmethod(lambda cls, *a, **k: built.append(k["atmosphere"])))
    answer(monkeypatch, "y", "1", "", "1", "high", "1.2", "")  # site 1 of the numbered atmosphere choice
    cli._offer_composition(tmp_path / "Fac" / "Cam")
    assert "'high' is not a number or 'unstated'" in capsys.readouterr().out
    assert built == [{"site": "paranal", "airmass": 1.2, "pwv": None}]


# ----------------------------------------------------------------------
# densify, coverage and the cleaning report
# ----------------------------------------------------------------------
@pytest.fixture
def densify(monkeypatch):
    """Record densify_grid calls, checking each binds to the real signature."""
    import inspect

    module = importlib.import_module("sed_tools.grid_densifier")
    signature = inspect.signature(module.densify_grid)
    calls = []

    def record(*args, **kwargs):
        signature.bind(*args, **kwargs)
        calls.append(kwargs)

    monkeypatch.setattr(module, "densify_grid", record)
    return calls


def test_densify_passes_only_what_densify_grid_accepts(densify, monkeypatch):
    base = ["densify", "--flux-cube", "in.bin", "--output", "out.bin", "--teff-spacing", "250"]
    assert run_cli(monkeypatch, *base, "--no-lookup") == 0
    assert run_cli(monkeypatch, *base, "--method", "ml", "--ml-model", "gen") == 0
    assert run_cli(monkeypatch, *base, "--method", "blackbody") == 0
    assert densify == [
        {"src": "in.bin", "dst": "out.bin", "teff_spacing": 250.0, "ml_model": None},
        {"src": "in.bin", "dst": "out.bin", "teff_spacing": 250.0, "ml_model": "gen"},
        {"src": "in.bin", "dst": "out.bin", "teff_spacing": 250.0, "ml_model": None},
    ]


@pytest.mark.parametrize("options,message", [
    (["--method", "interp"], "--method interp is not implemented"),
    (["--method", "ml"], "--method ml needs --ml-model"),
    (["--method", "blackbody", "--ml-model", "gen"], "--method blackbody does not use --ml-model"),
    (["--ml-gap-threshold", "2000"], "--ml-gap-threshold is not implemented"),
])
def test_densify_refuses_options_it_cannot_honour(densify, monkeypatch, capsys, options, message):
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, "densify", "--flux-cube", "in.bin", "--output", "out.bin", *options)
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err
    assert densify == []


def test_coverage_reports_every_grid_then_exits_1_if_any_failed(tmp_path, monkeypatch, capsys):
    covered = []

    def coverage(cls, name, **kwargs):
        if name == "B":
            raise ValueError("lookup_table.csv has no Teff column")
        covered.append(name)

    monkeypatch.setattr(SED, "coverage", classmethod(coverage))
    assert run_cli(monkeypatch, "coverage", "--grids", "A", "B", "C", "--no-plot", "--base", str(tmp_path)) == 1
    assert covered == ["A", "C"]
    assert "[coverage] FAILED B: lookup_table.csv has no Teff column" in capsys.readouterr().out


def test_cleaning_report_reads_the_summary_clean_model_dir_returns(capsys):
    from sed_tools.cli import _print_cleaning_report

    summary = dict(CLEAN, total=4, converted=["a", "b"], recovered=["c"], skipped_already=["d"],
                   catalog_units={"wavelength": "angstrom", "flux": "flam"},
                   detection_stats={"status": "consensus"})
    _print_cleaning_report(summary)
    out = capsys.readouterr().out
    assert "Cleaned       : 2 files" in out and "Recovered     : 1 files" in out
    assert "Already done  : 1 files" in out
    assert "All 4 files ready for use" in out


# ----------------------------------------------------------------------
# Interactive picker
# ----------------------------------------------------------------------
def test_picker_rejects_an_invalid_regex_instead_of_showing_everything(monkeypatch, capsys):
    from sed_tools.ui_utils import prompt_choice

    answer(monkeypatch, "/beta", "//[unclosed", "1")
    assert prompt_choice(["alpha", "beta", "gamma"], "Things", max_label=20) == 0
    out = capsys.readouterr().out
    assert "Invalid regular expression '[unclosed'" in out
    # The substring filter stays in force: the last listing shows beta only.
    last_listing = out.rsplit("Things", 1)[1]
    assert "beta" in last_listing and "alpha" not in last_listing


def test_spectra_flow_tries_the_next_source_when_one_raises(grid_flow, tmp_path, caplog):
    from sed_tools.cli import run_spectra_flow

    built = grid_flow(svo=fake_grabber(["Grid"], metadata=RuntimeError("SVO: incomplete FID list")),
                      msg=fake_grabber(["Grid"], metadata={"m": 1}, downloaded=1))
    with caplog.at_level(logging.WARNING, logger="sed_tools.cli"):
        run_spectra_flow("all", base_dir=str(tmp_path), models=["Grid"])
    assert built == ["Grid"]
    assert "[svo] SVO: incomplete FID list — trying next source..." in caplog.text


def test_spectra_flow_tries_no_other_source_after_a_partial_download(grid_flow, tmp_path):
    """Only files written by the failed attempt are detected; files from earlier runs are not."""
    from sed_tools.cli import run_spectra_flow

    built = grid_flow(svo=fake_grabber(["A", "B"], metadata=[{"fid": 1}], downloaded=1,
                                       download_error=RuntimeError("2 of 5 spectra failed")),
                      msg=fake_grabber(["A", "B", "C"], metadata=RuntimeError("MSG down")),
                      mast=fake_grabber(["C"], metadata={"m": 1}, downloaded=1))
    with pytest.raises(RuntimeError) as excinfo:
        run_spectra_flow("all", base_dir=str(tmp_path), models=["A", "B", "C"])
    message = str(excinfo.value)
    assert built == ["C"]
    for name in ("A", "B"):
        assert f"{name}: RuntimeError: svo failed after changing" in message
        assert "Tried: svo: 2 of 5 spectra failed" in message
    # msg (which would fail with "MSG down") was never tried for A or B, only for C, which mast then built.
    assert "MSG down" not in message


# ----------------------------------------------------------------------
# ML wrappers
# ----------------------------------------------------------------------
def test_ml_generator_auto_fails_at_once_without_optuna(monkeypatch, capsys):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args: None if name == "optuna" else real_find_spec(name, *args))
    assert run_cli(monkeypatch, "ml_generator", "--auto", "--library", "lib", "--output", "out") == 1
    out, err = capsys.readouterr()
    assert not out and len(err.splitlines()) == 1  # the one-line refusal of the other ML actions
    assert err.startswith("Error: missing ") and "optuna" in err and "ml-tune" in err
    assert "ml-tune" in err.split("pip install ")[1]


# ----------------------------------------------------------------------
# Detector-type tagging
# ----------------------------------------------------------------------
def test_filters_detector_prints_what_was_tagged_and_what_failed(tmp_path, monkeypatch, capsys):
    import sed_tools.filter_metadata as fm

    def tag_tree(root, *, force, dry_run, progress):
        progress(tagged(Path(root) / "g.dat"))
        raise RuntimeError("1 of 2 filter files could not be tagged and were left unchanged:\n  r.dat: bad")

    monkeypatch.setattr(fm, "tag_filter_tree", tag_tree)
    assert run_cli(monkeypatch, "filters-detector", str(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "g.dat: photon [svo]" in captured.out
    assert "sed-tools: error: 1 of 2 filter files could not be tagged" in captured.err


def test_composition_says_when_the_atmosphere_list_failed(tmp_path, monkeypatch, filters):
    from sed_tools import cli

    info = {"buildable": ["filter"], "atmosphere": True, "atmosphere_site": "paranal", "atmosphere_sites": None,
            "component_availability": {"status": "supported", "large_download": False}}

    def atmospheres(cls, *args, **kwargs):
        raise RuntimeError("NJM mirror unreachable")

    monkeypatch.setattr(filters, "available", classmethod(lambda cls, *a, **k: info))
    monkeypatch.setattr(filters, "atmospheres", classmethod(atmospheres))
    monkeypatch.setattr(cli.filters, "print_filter_availability", lambda info: None)
    answer(monkeypatch, "y", "1", "", "y")
    with pytest.raises(RuntimeError, match="could not list the atmosphere grids for paranal: NJM mirror unreachable"):
        cli._offer_composition(tmp_path / "Fac" / "Cam")


# ----------------------------------------------------------------------
# mesa_prepare: every failure exits non-zero
# ----------------------------------------------------------------------
def test_sed_tools_mesa_prepare_exit_status_is_1_on_failure(tmp_path):
    import subprocess
    from pathlib import Path

    write_lookup(tmp_path / "Plain", ["file_name", "teff", "logg", "metallicity"], [("a.txt", 5000, 4.0, 0.0)])
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-m", "sed_tools.cli", "mesa_prepare", "--base", str(tmp_path),
                           "--grid", "Plain", "--output", str(tmp_path / "out")],
                          cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert done.returncode == 1 and "has no flux_cube.bin" in done.stderr
