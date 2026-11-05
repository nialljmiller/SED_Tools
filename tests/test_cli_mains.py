"""``python -m <old path>`` for every module with a main(): its --help, a bad argument, and a short run.

Each module runs in this process through ``runpy.run_module(..., alter_sys=True)``, the call ``python -m``
makes, so main() is measured and no interpreter is started per case. Nothing touches the network or writes
outside tmp_path.
"""

import runpy
import shutil
import sys

import numpy as np
import pytest
import requests

from sed_tools.njm_filter_grabber import NJMFilterGrabber

from helpers_filters import CURVE, Served, njm_index
from helpers_grids import KURUCZ
from helpers_ml import _fixed_completer, _fixed_generator, cpu_torch  # noqa: F401 (cpu_torch is a fixture)
from helpers_random import keep_random_state  # noqa: F401 (an autouse fixture)

# runpy warns that an old path imported earlier in the session is in sys.modules; python -m starts afresh.
pytestmark = pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")


class NetworkUsed(BaseException):
    """Raised past every ``except Exception`` so a network call cannot be swallowed."""


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise NetworkUsed("tests must not use the network")

    monkeypatch.setattr(requests.Session, "send", refuse)
    monkeypatch.chdir(tmp_path)
    # svo_regen_spectra_lookup without argument parsing rewrote every lookup table under it, even for --help
    monkeypatch.setattr("sed_tools.cube.model.STELLAR_DIR_DEFAULT", tmp_path / "stellar_models")


def answer(monkeypatch, *replies):
    """Feed ``input()`` these replies in order; a further prompt fails the test."""
    queue = list(replies)

    def fake_input(prompt=""):
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


def run_m(monkeypatch, module, *argv):
    """``python -m sed_tools.<module> *argv``: the exit status (0 when main returns)."""
    monkeypatch.setattr(sys, "argv", [module, *argv])
    try:
        runpy.run_module(f"sed_tools.{module}", run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        return 0 if exc.code is None else exc.code
    return 0


# ----------------------------------------------------------------------
# --help and a bad argument
# ----------------------------------------------------------------------
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
    "svo_regen_spectra_lookup": "Rewrite lookup_table.csv in every spectral grid folder from the headers of its spectra",
}
# svo_filter_grabber takes any other argument as its base directory.
ARGPARSE = sorted(set(HELP) - {"svo_filter_grabber"})


@pytest.mark.parametrize("module", sorted(HELP))
def test_help_prints_the_usage_and_exits_0(module, monkeypatch, capsys):
    assert run_m(monkeypatch, module, "--help") == 0
    out = " ".join(capsys.readouterr().out.split())
    assert out.lower().startswith("usage:") and HELP[module] in out


@pytest.mark.parametrize("module", ARGPARSE)
def test_a_bad_argument_prints_the_usage_and_exits_2(module, monkeypatch, capsys):
    assert run_m(monkeypatch, module, "--no-such-option") == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "usage:" in captured.err and "error:" in captured.err


# ----------------------------------------------------------------------
# Short runs
# ----------------------------------------------------------------------
def test_combine_filters_writes_the_combined_set(tmp_path, monkeypatch, capsys):
    (tmp_path / "Generic" / "Johnson").mkdir(parents=True)
    (tmp_path / "Generic" / "Johnson" / "B.dat").write_text("Wavelength,Transmission\n4000,0\n4400,1\n4800,0\n")
    assert run_m(monkeypatch, "combine_filters", "Optical", "Generic/Johnson", "--base", str(tmp_path)) == 0
    out = tmp_path / "Combined" / "Optical"
    assert (out / "Optical").read_text().split() == ["B.dat"]
    assert capsys.readouterr().out == f"[filters] Combined filter set written to {out}\n"


def test_combine_stellar_atm_combines_the_grids_chosen(tmp_path, monkeypatch, capsys):
    assert run_m(monkeypatch, "combine_stellar_atm", "--data_dir", str(tmp_path)) == 0
    assert f"No spectral grids found in {tmp_path}" in capsys.readouterr().out
    shutil.copytree(KURUCZ, tmp_path / KURUCZ.name)
    answer(monkeypatch, "q")
    assert run_m(monkeypatch, "combine_stellar_atm", "--data_dir", str(tmp_path)) == 0
    assert "No spectral grids selected." in capsys.readouterr().out
    assert not (tmp_path / "combined_models").exists()
    answer(monkeypatch, "1")
    assert run_m(monkeypatch, "combine_stellar_atm", "--data_dir", str(tmp_path), "--output", "omni") == 0
    assert (tmp_path / "omni" / "flux_cube.bin").is_file() and (tmp_path / "omni" / "lookup_table.csv").is_file()
    assert capsys.readouterr().out.endswith(f"Done! Output: {tmp_path / 'omni'}\n")


def test_precompute_flux_cube_builds_the_cube_it_is_given(tmp_path, monkeypatch):
    grid = shutil.copytree(KURUCZ, tmp_path / "grid")
    assert run_m(monkeypatch, "precompute_flux_cube", "--model_dir", str(grid), "--output", "cube.bin") == 0
    assert (tmp_path / "cube.bin").stat().st_size > 16


def test_precompute_flux_cube_asks_for_the_grid_and_the_output(tmp_path, monkeypatch, capsys):
    assert run_m(monkeypatch, "precompute_flux_cube") == "No model directories found."
    shutil.copytree(KURUCZ, tmp_path / "data" / "stellar_models" / KURUCZ.name)
    (tmp_path / "data" / "stellar_models" / "notes.txt").write_text("not a grid\n")
    answer(monkeypatch, "0")
    assert run_m(monkeypatch, "precompute_flux_cube") == "Invalid selection."
    # Listed once (it was listed as ./data/stellar_models/<grid> and as data/stellar_models/<grid>)
    menu = [line for line in capsys.readouterr().out.splitlines() if line[:1].isdigit()]
    assert menu == [f"1. data/stellar_models/{KURUCZ.name}"]
    answer(monkeypatch, "1", "")  # the first grid listed, and the default output beside it
    assert run_m(monkeypatch, "precompute_flux_cube") == 0
    assert "No --model_dir provided. Discovering local model directories" in capsys.readouterr().out
    assert len(list(tmp_path.glob("**/flux_cube.bin"))) == 1


def test_svo_regen_spectra_lookup_rewrites_every_grids_lookup_table(tmp_path, monkeypatch, capsys):
    stellar = tmp_path / "stellar_models"
    for name in ("A", "B"):
        (stellar / name).mkdir(parents=True)
        (stellar / name / "s.txt").write_text("# teff = 5000\n# logg = 4.5\n# metallicity = 0.0\n4000 1\n5000 2\n")
    monkeypatch.setattr("sed_tools.cube.model.STELLAR_DIR_DEFAULT", stellar)
    assert run_m(monkeypatch, "svo_regen_spectra_lookup") == 0
    for name in ("A", "B"):
        assert (stellar / name / "lookup_table.csv").read_text().splitlines()[1] == "s.txt,4.5,0.0,5000.0"
    assert capsys.readouterr().out.count("Processing spectral grid:") == 2


def test_msg_spectra_grabber_runs_the_grids_it_is_given(tmp_path, monkeypatch):
    from sed_tools.grids.msg import MSGSpectraGrabber

    runs = []
    monkeypatch.setattr(MSGSpectraGrabber, "discover_and_run", lambda self, selected_models=None:
                        runs.append((self.base_dir, self.max_workers, selected_models)))
    assert run_m(monkeypatch, "msg_spectra_grabber", "--output", str(tmp_path), "--workers", "2",
                 "--models", "A", "B") == 0
    assert runs == [(str(tmp_path), 2, ["A", "B"])]


def test_grid_densifier_passes_its_arguments_on(monkeypatch):
    calls = []
    monkeypatch.setattr("sed_tools.cube.densify.densify_grid", lambda **kwargs: calls.append(kwargs))
    assert run_m(monkeypatch, "grid_densifier", "--flux-cube", "in.bin", "--output", "out.bin",
                 "--teff-spacing", "250") == 0
    assert calls == [{"src": "in.bin", "dst": "out.bin", "teff_spacing": 250.0, "ml_model": None}]


# ----------------------------------------------------------------------
# njm_filter_grabber
# ----------------------------------------------------------------------
@pytest.fixture
def njm_served(monkeypatch):
    """NJM serves one facility with one camera of one band; downloads are recorded instead of made."""
    files = {"index.json": njm_index({"Fac": {"Cam": {"filters": ["g"]}}}), "filters/Fac/Cam/Cam": "g.dat\n",
             "filters/Fac/Cam/g.dat": CURVE}
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    monkeypatch.setattr("sed_tools.filters.njm.requests.Session", lambda: Served(files))
    calls = []
    monkeypatch.setattr(NJMFilterGrabber, "download_filters", lambda self, *a: calls.append(("set", *a)))
    monkeypatch.setattr(NJMFilterGrabber, "download_all_filters", lambda self, **k: calls.append(("all", k)))
    return calls


def test_njm_filter_grabber_lists_and_downloads(tmp_path, monkeypatch, capsys, njm_served):
    out = str(tmp_path / "filters")
    assert run_m(monkeypatch, "njm_filter_grabber", "--list", "--output", out) == 0
    assert "Fac:\n    - Cam (1 filters)" in capsys.readouterr().out.split("Available facilities:")[1]
    for argv in (["--facility", "Fac", "--instrument", "Cam"], ["--facility", "Fac"], []):
        assert run_m(monkeypatch, "njm_filter_grabber", *argv, "--output", out) == 0
    assert njm_served == [("set", "Fac", "Cam"), ("all", {"facility": "Fac"}), ("all", {})]


def test_njm_filter_grabber_needs_a_facility_for_an_instrument(tmp_path, monkeypatch, capsys, njm_served):
    assert run_m(monkeypatch, "njm_filter_grabber", "--instrument", "Cam", "--output", str(tmp_path)) == 1
    assert capsys.readouterr().out == "Error: --instrument requires --facility\n" and njm_served == []


def test_njm_filter_grabber_exits_1_when_njm_is_down(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: False)
    assert run_m(monkeypatch, "njm_filter_grabber", "--list", "--output", str(tmp_path)) == 1
    assert capsys.readouterr().out.endswith("NJM mirror is not available\n")


# ----------------------------------------------------------------------
# ml_sed_completer and ml_sed_generator
# ----------------------------------------------------------------------
def test_ml_train_commands_pass_their_options_on(monkeypatch):
    from sed_tools.ml.completer import SEDCompleter
    from sed_tools.ml.generator import SEDGenerator

    calls = []
    monkeypatch.setattr(SEDCompleter, "train", lambda self, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(SEDGenerator, "train", lambda self, *args, **kwargs: calls.append((args, kwargs)))
    assert run_m(monkeypatch, "ml_sed_completer", "train", "--library", "lib", "--output", "out", "--epochs", "3",
                 "--batch-size", "4", "--lr", "0.01", "--max-samples", "9", "--patience", "2") == 0
    assert run_m(monkeypatch, "ml_sed_generator", "train", "--library", "lib", "--output", "out",
                 "--epochs", "5") == 0
    assert calls == [{"library_path": "lib", "output_path": "out", "epochs": 3, "batch_size": 4,
                      "learning_rate": 0.01, "max_samples": 9, "patience": 2},
                     (("lib", "out"), {"epochs": 5, "batch_size": 64})]


def test_ml_sed_completer_lists_and_completes_with_a_saved_model(cpu_torch, tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    assert run_m(monkeypatch, "ml_sed_completer", "list", "--models-dir", str(models)) == 0
    assert capsys.readouterr().out == f"No models found in {models}/\n"
    _fixed_completer(cpu_torch).save(models / "tiny", hidden_layers=[16, 8])
    capsys.readouterr()
    assert run_m(monkeypatch, "ml_sed_completer", "list", "--models-dir", str(models)) == 0
    out = capsys.readouterr().out
    assert "  tiny\n" in out and "Architecture: 400 -> [16, 8] -> 1000" in out

    wl = np.logspace(np.log10(3000.0), np.log10(9000.0), 50)
    np.savetxt(tmp_path / "in.txt", np.column_stack([wl, 1e-10 * (wl / 5000.0) ** -2]))
    assert run_m(monkeypatch, "ml_sed_completer", "complete", "--model", str(models / "tiny"), "--input", "in.txt",
                 "--output", "out.txt", "--teff", "5500", "--logg", "4.5", "--meta", "0", "--wl-min", "1000",
                 "--wl-max", "20000") == 0
    extended = np.loadtxt(tmp_path / "out.txt")
    assert extended[0, 0] <= 1000.0 * 1.01 and extended[-1, 0] >= 20000.0 * 0.99
    assert "Saved completed SED to: out.txt" in capsys.readouterr().out


def test_ml_sed_generator_lists_and_generates_with_a_saved_model(cpu_torch, tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    _fixed_generator(cpu_torch).save(models / "gen", hidden_layers=[16, 8])
    capsys.readouterr()
    assert run_m(monkeypatch, "ml_sed_generator", "list", "--models-dir", str(models)) == 0
    assert capsys.readouterr().out == f"gen: {models / 'gen'}\n"
    assert run_m(monkeypatch, "ml_sed_generator", "generate", "--model", str(models / "gen"), "--teff", "5500",
                 "--logg", "4.5", "--meta", "0", "--output", "sed.txt") == 0
    lines = (tmp_path / "sed.txt").read_text().splitlines()
    assert lines[0] == "# wavelength_A flux_erg/s/cm2/A" and "Teff=5500.0 logg=4.5 [M/H]=0.0" in lines[1]
    assert np.loadtxt(tmp_path / "sed.txt").shape == (1000, 2)
    assert capsys.readouterr().out.endswith("Saved: sed.txt\n")


# ----------------------------------------------------------------------
# sed-tools commands with no other test
# ----------------------------------------------------------------------
def run_cli(monkeypatch, *argv):
    from sed_tools import cli

    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    return cli.main()


def test_filters_combine_command_writes_the_combined_set(tmp_path, monkeypatch, capsys):
    (tmp_path / "Generic" / "Johnson").mkdir(parents=True)
    (tmp_path / "Generic" / "Johnson" / "V.dat").write_text("Wavelength,Transmission\n5000,0\n5500,1\n6000,0\n")
    assert run_cli(monkeypatch, "filters-combine", "Mine/Set", "Generic/Johnson", "--base", str(tmp_path)) == 0
    assert (tmp_path / "Mine" / "Set" / "Set").read_text().split() == ["V.dat"]
    assert capsys.readouterr().out == f"[filters] Combined filter set written to {tmp_path / 'Mine' / 'Set'}\n"


def test_building_all_components_runs_every_builder_and_lists_the_failures(tmp_path, monkeypatch, capsys):
    from sed_tools import cli

    built = []

    def builder(name):
        def build(base, **kwargs):
            if name == "bad":
                raise RuntimeError("source did not answer")
            built.append((name, base, kwargs))
        return build

    monkeypatch.setattr(cli, "AUTOMATIC_BUILDERS", ("first", "skycalc", "bad"))
    monkeypatch.setattr(cli, "get_builder", builder)
    assert run_cli(monkeypatch, "filters-build-components", "all", "--base", str(tmp_path)) == 1
    assert built == [("first", str(tmp_path), {}), ("skycalc", str(tmp_path), {"sites": ["paranal", "lasilla"]})]
    out = capsys.readouterr().out
    assert "[filters] === bad ===" in out and out.endswith("[filters] FAILED bad: source did not answer\n")
    built.clear()
    monkeypatch.setattr(cli, "AUTOMATIC_BUILDERS", ("skycalc",))
    assert run_cli(monkeypatch, "filters-build-components", "all", "--base", str(tmp_path), "--sites", "lapalma") == 0
    assert built == [("skycalc", str(tmp_path), {"sites": ["lapalma"]})]


def test_the_coverage_menu_reports_the_grid_chosen(tmp_path, monkeypatch, capsys):
    from sed_tools.cli.grids import run_coverage_flow

    base = tmp_path / "stellar_models"
    base.mkdir()
    run_coverage_flow(str(base))
    assert capsys.readouterr().out == f"No local grids found under {base}\n"
    shutil.copytree(KURUCZ, base / KURUCZ.name)
    answer(monkeypatch, "b")
    run_coverage_flow(str(base))
    assert not (base / KURUCZ.name / "coverage.png").exists()
    answer(monkeypatch, "1")
    run_coverage_flow(str(base))
    assert (base / KURUCZ.name / "coverage.png").is_file()
    out = capsys.readouterr().out
    assert f"Coverage: {KURUCZ.name}" in out and "range   : 5750 -> 6250" in out
