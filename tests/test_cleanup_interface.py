"""Failures in the public API and CLI are raised, warned about or reported with a non-zero exit."""

import importlib
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from sed_tools.api import SED


class NetworkUsed(BaseException):
    """Raised past every ``except Exception`` so a network call cannot be swallowed."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise NetworkUsed("tests must not use the network")

    monkeypatch.setattr(requests.Session, "send", refuse)


# ----------------------------------------------------------------------
# Spectral grid sources
# ----------------------------------------------------------------------
GRABBERS = {
    "njm": "sed_tools.njm_spectra_grabber.NJMSpectraGrabber",
    "svo": "sed_tools.svo_spectra_grabber.SVOSpectraGrabber",
    "msg": "sed_tools.msg_spectra_grabber.MSGSpectraGrabber",
    "mast": "sed_tools.mast_spectra_grabber.MASTSpectraGrabber",
}


def fake_grabber(models=(), error=None, available=True, metadata=None, downloaded=0, download_error=None):
    """A grabber class; ``metadata`` may be an exception to raise, and ``download_error`` is
    raised after ``downloaded`` spectra have been written."""
    class Grabber:
        base_url = "https://mirror.invalid"

        def __init__(self, base_dir=None, max_workers=None):
            self.base_dir = base_dir

        def is_available(self):
            return available

        def discover_models(self):
            if error is not None:
                raise error
            return list(models)

        def get_model_metadata(self, name):
            if isinstance(metadata, Exception):
                raise metadata
            return metadata

        def download_model_spectra(self, name, meta, **cuts):
            for i in range(downloaded):
                path = Path(self.base_dir) / name / f"spectrum{i}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("1000 1\n2000 1\n")
            if download_error is not None:
                raise download_error
            return downloaded

    return Grabber


@pytest.fixture
def sources(monkeypatch):
    """Replace the four spectral grid grabbers; unspecified ones list nothing."""
    def install(**grabbers):
        for name, target in GRABBERS.items():
            monkeypatch.setattr(target, grabbers.get(name, fake_grabber()))
    return install


def test_query_warns_about_a_failing_source_and_lists_the_others(sources, caplog):
    sources(svo=fake_grabber(error=requests.ConnectionError("SVO down")), msg=fake_grabber(["MSGgrid"]))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"):
        grids = SED.query(include_local=False)
    assert [g.name for g in grids] == ["MSGgrid"]
    assert "svo spectral grid discovery failed: SVO down" in caplog.text


def test_query_warns_when_the_njm_mirror_is_unreachable(sources, caplog):
    sources(njm=fake_grabber(["NJMgrid"], available=False), mast=fake_grabber(["BOSZ"]))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"):
        grids = SED.query(include_local=False)
    assert [g.name for g in grids] == ["BOSZ"]
    assert "njm spectral grid discovery failed: NJM mirror https://mirror.invalid is unreachable" in caplog.text


def test_query_raises_when_the_requested_source_fails(sources):
    sources(svo=fake_grabber(error=requests.ConnectionError("SVO down")))
    with pytest.raises(RuntimeError, match="svo: SVO down"):
        SED.query(source="svo", include_local=False)


def test_query_raises_with_every_error_when_all_sources_fail(sources):
    down = {name: fake_grabber(error=RuntimeError(f"{name} down")) for name in GRABBERS}
    sources(**down)
    with pytest.raises(RuntimeError) as excinfo:
        SED.query(include_local=False)
    for name in GRABBERS:
        assert f"{name}: {name} down" in str(excinfo.value)


def test_query_rejects_an_unknown_source(sources):
    sources()
    with pytest.raises(ValueError, match="Unknown source: sov"):
        SED.query(source="sov")


def test_fetch_error_names_every_source_that_failed(sources, tmp_path, caplog):
    sources(njm=fake_grabber(available=False), msg=fake_grabber(error=ValueError("bad index")))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"), pytest.raises(ValueError) as excinfo:
        SED.fetch("Missing", model_root=tmp_path)
    message = str(excinfo.value)
    assert "Spectral grid 'Missing' could not be fetched" in message
    assert "njm: discovery failed: NJM mirror https://mirror.invalid is unreachable" in message
    assert "msg: discovery failed: bad index" in message
    assert "svo: no metadata for Missing" in message
    assert "msg discovery failed: bad index; trying the other sources" in caplog.text
    assert "svo: no metadata for Missing, trying next" in caplog.text


def test_fetch_falls_back_to_the_next_source_with_a_warning(sources, tmp_path, monkeypatch, caplog):
    sources(njm=fake_grabber(["Grid"], metadata=RuntimeError("NJM mirror has no files for Grid")),
            svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], download_error=RuntimeError("SVO timed out")),
            msg=fake_grabber(["Grid"], metadata={"m": 1}, downloaded=1))
    monkeypatch.setattr("sed_tools.spectra_cleaner.clean_model_dir", lambda *a, **k: dict(CLEAN))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"):
        sed = SED.fetch("Grid", model_root=tmp_path)
    assert sed.grid.source == "msg"
    assert "njm: NJM mirror has no files for Grid, trying next" in caplog.text
    assert "svo: SVO timed out, trying next" in caplog.text


def test_fetch_stops_when_a_failed_download_left_files_behind(sources, tmp_path):
    sources(svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], downloaded=2, download_error=RuntimeError("1 of 3 failed")),
            msg=fake_grabber(["Grid"], metadata={"m": 1}, downloaded=1))
    with pytest.raises(RuntimeError, match="svo failed after changing .*mixed with its files.*svo: 1 of 3 failed"):
        SED.fetch("Grid", model_root=tmp_path)
    assert sorted(p.name for p in (tmp_path / "Grid").iterdir()) == ["spectrum0.txt", "spectrum1.txt"]


def test_fetch_from_msg_reads_the_msg_index_first(tmp_path, monkeypatch):
    class MSG:
        def __init__(self, base_dir=None, max_workers=None):
            self.model_urls = {}

        def discover_models(self):
            self.model_urls = {"Grid": "https://msg.invalid/Grid.h5"}
            return ["Grid"]

        def get_model_metadata(self, name):
            if name not in self.model_urls:
                raise RuntimeError(f"MSG grid {name} is unknown to this grabber")
            return {}

    monkeypatch.setattr(GRABBERS["msg"], MSG)
    with pytest.raises(ValueError, match="msg: no metadata for Grid"):
        SED.fetch("Grid", source="msg", model_root=tmp_path)


def test_fetch_from_mast_does_not_wait_for_the_keyboard(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("SED.fetch must not prompt"))
    with pytest.raises(ValueError, match="mast: MAST \\(BOSZ\\) grids are fetched with 'sed-tools grids --source mast'"):
        SED.fetch("BOSZ-2024-r10000", source="mast", model_root=tmp_path)


def test_fetch_rejects_an_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="Unknown source: sov"):
        SED.fetch("Grid", source="sov", model_root=tmp_path)


def test_fetch_raises_when_nothing_was_downloaded(sources, tmp_path):
    sources(svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], downloaded=0))
    with pytest.raises(ValueError, match="svo: no spectra downloaded"):
        SED.fetch("Grid", source="svo", model_root=tmp_path)


# ----------------------------------------------------------------------
# Local spectral grids
# ----------------------------------------------------------------------
def write_cube_header(path, teff=(5000.0,), logg=(4.0,), meta=(0.0,), wavelengths=(1000.0, 2000.0)):
    import struct

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("4i", len(teff), len(logg), len(meta), len(wavelengths)))
        for axis in (teff, logg, meta, wavelengths):
            np.asarray(axis, dtype=np.float64).tofile(fh)
        np.zeros(len(teff) * len(logg) * len(meta) * len(wavelengths)).tofile(fh)


def test_query_raises_for_a_local_grid_whose_cube_header_is_unreadable(tmp_path, monkeypatch):
    (tmp_path / "Broken").mkdir()
    (tmp_path / "Broken" / "flux_cube.bin").write_bytes(b"abc")
    monkeypatch.setattr(SED, "_model_root", tmp_path)
    with pytest.raises(ValueError, match="flux cube header of local grid 'Broken'.*sed-tools rebuild --grids Broken"):
        SED.query(include_remote=False)


def test_query_raises_for_a_local_grid_whose_lookup_table_is_unreadable(tmp_path, monkeypatch):
    write_cube_header(tmp_path / "Grid" / "flux_cube.bin")
    (tmp_path / "Grid" / "lookup_table.csv").write_text("")
    monkeypatch.setattr(SED, "_model_root", tmp_path)
    with pytest.raises(ValueError, match="lookup_table.csv of local grid 'Grid'"):
        SED.query(include_remote=False)


def test_local_raises_for_a_listed_spectrum_it_cannot_read(tmp_path):
    grid = tmp_path / "Grid"
    write_cube_header(grid / "flux_cube.bin")
    (grid / "lookup_table.csv").write_text("file_name,teff,logg,meta\nbad.txt,5000,4.0,0.0\n")
    (grid / "bad.txt").write_text("not a spectrum\n")
    with pytest.raises(ValueError, match="Cannot read spectrum .*bad.txt listed in"):
        SED.local("Grid", model_root=tmp_path)


def spectrum(teff=5000.0, logg=4.0, metallicity=0.0, filename=None):
    import numpy as np

    from sed_tools.api import Spectrum

    return Spectrum(wavelength=np.array([1000.0, 2000.0, 3000.0]), flux=np.array([1.0, 2.0, 3.0]),
                    teff=teff, logg=logg, metallicity=metallicity, filename=filename)


def test_grid_write_raises_when_the_flux_cube_build_fails(tmp_path, monkeypatch):
    from sed_tools.api import SpectralGrid

    def fail(*args):
        raise RuntimeError("cube failed")

    # The package re-exports the function under the module's name, so patch the module itself.
    monkeypatch.setattr(importlib.import_module("sed_tools.precompute_flux_cube"), "precompute_flux_cube", fail)
    grid = SpectralGrid(name="Grid", source="local", spectra=[spectrum()])
    with pytest.raises(RuntimeError, match="cube failed"):
        grid.write(tmp_path / "out", build_h5=False)


def test_grid_write_raises_when_there_is_nothing_to_bundle(tmp_path):
    from sed_tools.api import SpectralGrid

    grid = SpectralGrid(name="Grid", source="local", spectra=[spectrum(filename="a.dat")])
    with pytest.raises(RuntimeError, match="No .txt spectra in .* to bundle into Grid.h5"):
        grid.write(tmp_path / "out", build_flux_cube=False)


def test_import_dry_run_counts_spectra_lacking_a_parameter(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    spectrum().save(src / "good.txt")
    spectrum(metallicity=float("nan")).save(src / "no_meta.txt")

    report = SED.import_grid(src, name="Grid", model_root=tmp_path / "models", dry_run=True)
    assert report["n_missing"] == 1


# ----------------------------------------------------------------------
# Filters and ML
# ----------------------------------------------------------------------
@pytest.fixture
def filters(tmp_path, monkeypatch):
    from sed_tools.api import Filters

    monkeypatch.setattr(Filters, "_filter_root", tmp_path / "filters")
    monkeypatch.setattr(Filters, "_catalogues", {})
    return Filters


def test_filter_query_warns_about_discovery_errors(filters, monkeypatch, caplog):
    from sed_tools.filter_library import FilterLibrary

    report = {"entries": [{"key": "HST", "label": "HST", "sources": ["original"]}],
              "errors": ["SVO facility list: connection refused"]}
    monkeypatch.setattr(FilterLibrary, "facilities", lambda self: report)
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"):
        rows = filters.query(include_local=False)
    assert [row["facility"] for row in rows] == ["HST"]
    assert "Filter discovery: SVO facility list: connection refused" in caplog.text


def test_bibtex_citations_raise_when_a_target_failed(filters, monkeypatch):
    from sed_tools.filter_catalogue import CatalogueError

    result = {"instruments": [], "references": [], "acknowledgements": [], "errors": ["A/B: not cached"]}
    monkeypatch.setattr(filters, "citations", classmethod(lambda cls, *targets, **kwargs: result))
    with pytest.raises(CatalogueError, match="Cannot cite every target: A/B: not cached"):
        filters.cite("A/B", format="bibtex")
    assert "[filters] A/B: not cached" in filters.cite("A/B", format="text")


def test_plot_warns_when_njm_cannot_resolve_an_instrument_name(filters, tmp_path, monkeypatch, caplog):
    from sed_tools.filter_catalogue import CatalogueError

    fetched = tmp_path / "fetched" / "Fac" / "Cam"
    fetched.mkdir(parents=True)
    (fetched / "g.dat").write_text("Wavelength,Transmission\n4000,0\n4500,1\n5000,0\n")

    class Catalogue:
        def resolve(self, facility, instrument):
            raise CatalogueError(f"NJM does not serve {facility}/{instrument}")

    monkeypatch.setattr(filters, "_fetch_published", classmethod(lambda cls, fac, inst, filter_root=None: fetched))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"):
        chosen = filters._plot_bands(("Fac/Cam",), None, Catalogue(), fetch=True)
    assert chosen == [("Fac", "Cam", "g", fetched)]
    assert "NJM does not serve Fac/Cam; fetching Fac/Cam under the name given" in caplog.text


def test_ml_generator_parameter_ranges_need_the_trained_ranges():
    from types import SimpleNamespace

    from sed_tools.api import MLGenerator

    generator = MLGenerator(base_dir=None)
    generator._generator = SimpleNamespace(config={"parameter_ranges": {
        "teff": [3000, 9000], "logg": [0, 5], "meta": [-2, 0.5]}})
    assert generator.parameter_ranges() == {"teff": (3000, 9000), "logg": (0, 5), "metallicity": (-2, 0.5)}
    generator._generator = SimpleNamespace(config={})
    with pytest.raises(KeyError, match="parameter_ranges"):
        generator.parameter_ranges()


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
CLEAN = {"total": 1, "converted": [], "skipped_already": [], "skipped_invalid": [], "skipped_index": [],
         "error": [], "recovered": [], "detection_stats": {}, "catalog_units": None}


@pytest.fixture
def grid_flow(monkeypatch):
    """run_spectra_flow with fake grabbers, cleaning and product builds; returns the grids built."""
    from sed_tools import cli

    built = []

    def install(cleaned=CLEAN, cube_built=True, **grabbers):
        for name, cls_name in (("njm", "NJMSpectraGrabber"), ("svo", "SVOSpectraGrabber"),
                               ("msg", "MSGSpectraGrabber"), ("mast", "MASTSpectraGrabber")):
            monkeypatch.setattr(cli, cls_name, grabbers.get(name, fake_grabber()))
        monkeypatch.setattr(cli, "clean_model_dir", lambda *a, **k: cleaned)
        monkeypatch.setattr(cli, "undeclared_flux_unit", lambda *a, **k: None)  # cleaning is faked too

        def build(name, model_dir, *args):
            if not cube_built:
                raise RuntimeError(f"No flux cube was built in {model_dir}")
            built.append(name)
            return []  # what was not done, as _build_data_products returns

        monkeypatch.setattr(cli, "_build_data_products", build)
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
    assert "Empty: RuntimeError: no source could supply it. Tried: svo: no metadata; mast: no spectra downloaded" in message
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


def test_spectra_flow_uses_other_sources_when_discovery_fails(grid_flow, tmp_path, caplog):
    from sed_tools.cli import run_spectra_flow

    grid_flow(svo=fake_grabber(error=requests.ConnectionError("SVO down")),
              msg=fake_grabber(["Grid"], metadata={"m": 1}, downloaded=1))
    with caplog.at_level(logging.WARNING, logger="sed_tools.cli"):
        run_spectra_flow("all", base_dir=str(tmp_path), models=["Grid"])
    assert "[svo] Spectral grid discovery failed: SVO down; using other sources" in caplog.text


def test_spectra_flow_raises_when_no_grid_is_discovered(grid_flow, tmp_path):
    from sed_tools.cli import run_spectra_flow

    grid_flow(svo=fake_grabber(error=requests.ConnectionError("SVO down")))
    with pytest.raises(RuntimeError, match="No spectral grids discovered; svo: discovery failed: SVO down"):
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
def precompute(monkeypatch):
    """Replace precompute_flux_cube with one that writes the given files (or nothing)."""
    def install(*names):
        def fake(model_dir, output_file):
            for name in names:
                (Path(model_dir) / name).write_text("built")
        monkeypatch.setattr(importlib.import_module("sed_tools.precompute_flux_cube"), "precompute_flux_cube", fake)
    return install


def test_flux_cube_helper_reports_what_was_written(tmp_path, precompute):
    from sed_tools.api import _build_flux_cube

    precompute("flux_cube.bin")
    assert _build_flux_cube(tmp_path) == tmp_path / "flux_cube.bin"
    precompute("variants_index.csv")
    assert _build_flux_cube(tmp_path) == tmp_path / "variants_index.csv"


def test_flux_cube_helper_raises_when_nothing_new_was_written(tmp_path, precompute):
    from sed_tools.api import _build_flux_cube

    (tmp_path / "flux_cube.bin").write_text("from an earlier build")
    precompute()
    with pytest.raises(RuntimeError, match="No flux cube was built in"):
        _build_flux_cube(tmp_path)


def test_grid_write_says_when_it_built_one_cube_per_variant(tmp_path, precompute, capsys):
    from sed_tools.api import SpectralGrid

    precompute("variants_index.csv")
    SpectralGrid(name="Grid", source="local", spectra=[spectrum()]).write(tmp_path / "out", build_h5=False)
    out = capsys.readouterr().out
    assert "Built one flux cube per extra-axis variant" in out and "Built flux_cube.bin" not in out


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

    monkeypatch.setattr(cli, "clean_model_dir", clean)
    monkeypatch.setattr(cli, "regenerate_lookup_table", lambda model_dir: None)
    monkeypatch.setattr(cli, "build_h5_bundle_from_txt", lambda model_dir, out: None)
    monkeypatch.setattr("sed_tools.api._build_flux_cube", lambda model_dir: Path(model_dir) / "flux_cube.bin")
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
    monkeypatch.setattr(cli, "_offer_composition", lambda directory: None)
    # What the browser describes before asking to download (Filters.available, NJM's catalogue entry).
    monkeypatch.setattr(FilterLibrary, "available", lambda self, facility, instrument, remote=True: {
        "facility": facility, "instrument": instrument, "bands": ["g"], "published": "filter",
        "published_bands": {}, "components": [], "partial_components": {}, "component_status": None,
        "errors": [], "atmosphere": False, "platform": "ground", "variants": {}})
    monkeypatch.setattr(cli, "_catalogue_entry", lambda *a, **k: {"info": None, "properties": None, "errors": []})
    return SimpleNamespace(fetched=fetched, listing_errors=listing_errors)


def answer(monkeypatch, *answers):
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))


def test_filter_batch_downloads_the_rest_then_lists_failures(filter_browser, monkeypatch, capsys):
    from sed_tools.cli import run_filters_flow

    filter_browser.listing_errors["A"] = ["SVO A: timed out"]
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

    monkeypatch.setattr(cli, "_offer_composition", fail)
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
    monkeypatch.setattr(cli, "print_filter_availability", lambda info: None)
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


def test_fetched_grid_records_the_source_it_came_from(sources, tmp_path, monkeypatch):
    sources(svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], downloaded=1))
    monkeypatch.setattr("sed_tools.spectra_cleaner.clean_model_dir", lambda *a, **k: dict(CLEAN))
    assert SED.fetch("Grid", model_root=tmp_path).grid.source == "svo"


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
@pytest.mark.parametrize("summary,error,message", [
    ({"extended": 0, "failed": 0, "skipped": 0, "total": 0}, FileNotFoundError, "No spectra to extend in"),
])
def test_ml_completer_extend_raises_on_spectra_it_could_not_extend(tmp_path, monkeypatch, summary, error, message):
    from sed_tools.api import MLCompleter

    monkeypatch.setattr("sed_tools.ml_sed_completer.extend_catalog", lambda **kwargs: dict(summary, output_dir="o"))
    completer = MLCompleter(base_dir=tmp_path)
    completer._model = object()
    with pytest.raises(error, match=message):
        completer.extend("Grid")


def test_ml_generator_writes_outputs_beside_the_loaded_model(tmp_path, monkeypatch):
    from sed_tools.api import MLGenerator

    calls = []

    class Generator:
        def __init__(self, path=None):
            self.config = {}

        def generate_with_outputs(self, **kwargs):
            calls.append(kwargs["output_dir"])

    monkeypatch.setattr("sed_tools.ml_sed_generator.SEDGenerator", Generator)
    (tmp_path / "models" / "gen").mkdir(parents=True)
    generator = MLGenerator(base_dir=tmp_path).load("gen")
    generator.generate_with_outputs(5800, 4.4, 0.0)
    assert calls == [str(tmp_path / "models" / "gen" / "SED")]


def test_ml_generator_auto_fails_at_once_without_optuna(monkeypatch, capsys):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args: None if name == "optuna" else real_find_spec(name, *args))
    assert run_cli(monkeypatch, "ml_generator", "--auto", "--library", "lib", "--output", "out") == 1
    out, err = capsys.readouterr()
    assert not out and len(err.splitlines()) == 1  # the one-line refusal of the other ML actions
    assert err.startswith("Error: missing ") and "optuna" in err and "ml-tune" in err


# ----------------------------------------------------------------------
# Detector-type tagging
# ----------------------------------------------------------------------
def tagged(path):
    return SimpleNamespace(path=path, detector_type="photon", source="svo", filter_id="F", changed=True,
                           migrated=False, note="", kept=False, resolved_type="photon", resolved_source="svo")


def test_tagging_one_facility_finishes_every_file_then_lists_failures(tmp_path, monkeypatch, filters, capsys):
    import sed_tools.filter_metadata as fm

    for band in ("g", "r", "i"):
        (tmp_path / f"{band}.dat").write_text("Wavelength,Transmission\n1,1\n")

    def tag(path, **kwargs):
        if path.stem == "r":
            raise requests.ConnectionError("SVO timed out")
        return tagged(path)

    monkeypatch.setattr(fm, "SVOMetadataClient", lambda: None)
    monkeypatch.setattr(fm, "tag_filter_file", tag)
    with pytest.raises(RuntimeError, match=r"(?s)1 of 3 filter files could not be tagged.*r\.dat: SVO timed out"):
        filters.tag_detector_types(tmp_path, facility="Fac")
    out = capsys.readouterr().out
    assert "g.dat: photon [svo]" in out and "i.dat: photon [svo]" in out


def test_tagging_stops_at_once_when_svo_is_unreachable(tmp_path, monkeypatch, filters):
    import sed_tools.filter_metadata as fm

    for band in ("g", "r"):
        (tmp_path / f"{band}.dat").write_text("Wavelength,Transmission\n1,1\n")
    calls = []

    def tag(path, **kwargs):
        calls.append(path.stem)
        raise fm.SVOUnreachableError("SVO unreachable")

    monkeypatch.setattr(fm, "SVOMetadataClient", lambda: None)
    monkeypatch.setattr(fm, "tag_filter_file", tag)
    with pytest.raises(fm.SVOUnreachableError):
        filters.tag_detector_types(tmp_path, facility="Fac")
    assert len(calls) == 1


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
    monkeypatch.setattr(cli, "print_filter_availability", lambda info: None)
    answer(monkeypatch, "y", "1", "", "y")
    with pytest.raises(RuntimeError, match="could not list the atmosphere grids for paranal: NJM mirror unreachable"):
        cli._offer_composition(tmp_path / "Fac" / "Cam")
