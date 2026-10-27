"""Failures in the public API and CLI are raised, warned about or reported with a non-zero exit."""

import importlib
import logging

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


def fake_grabber(models=(), error=None, available=True, metadata=None, downloaded=0):
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
            return metadata

        def download_model_spectra(self, name, meta, **cuts):
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
    assert "Spectral grid 'Missing' not found" in message
    assert "njm: discovery failed: NJM mirror https://mirror.invalid is unreachable" in message
    assert "msg: discovery failed: bad index" in message
    assert "svo: no metadata for Missing" in message
    assert "msg discovery failed: bad index; trying the other sources" in caplog.text
    assert "svo: no metadata returned" in caplog.text


def test_fetch_falls_back_to_the_next_source_with_a_warning(sources, tmp_path, caplog):
    sources(njm=fake_grabber(["Grid"]), svo=fake_grabber(["Grid"], metadata=[{"fid": 1}]))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"), pytest.raises(RuntimeError, match="from svo"):
        SED.fetch("Grid", model_root=tmp_path)
    assert "njm: no metadata returned, trying next" in caplog.text


def test_fetch_rejects_an_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="Unknown source: sov"):
        SED.fetch("Grid", source="sov", model_root=tmp_path)


def test_fetch_raises_when_nothing_was_downloaded(sources, tmp_path):
    sources(svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], downloaded=0))
    with pytest.raises(RuntimeError, match="No spectra of 'Grid' were downloaded from svo"):
        SED.fetch("Grid", source="svo", model_root=tmp_path)


def test_fetch_raises_when_cleaning_leaves_unusable_spectra(sources, tmp_path, monkeypatch):
    sources(svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], downloaded=3))
    summary = {"total": 3, "converted": [], "skipped_invalid": ["a.txt"], "skipped_index": [], "error": ["b.txt"]}
    monkeypatch.setattr("sed_tools.spectra_cleaner.clean_model_dir", lambda *a, **k: summary)
    with pytest.raises(RuntimeError, match="2 of 3 spectra of 'Grid' .* are unusable after cleaning"):
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


def test_import_lists_spectra_left_out_after_building_the_rest(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    spectrum().save(src / "good.txt")
    spectrum(metallicity=float("nan")).save(src / "no_meta.txt")
    summary = {"total": 2, "converted": [], "skipped_invalid": [], "skipped_index": [], "error": []}
    monkeypatch.setattr("sed_tools.spectra_cleaner.clean_model_dir", lambda *a, **k: summary)
    monkeypatch.setattr("sed_tools.svo_regen_spectra_lookup.regenerate_lookup_table", lambda *a: None)

    with pytest.raises(RuntimeError, match=r"Lacking a Teff, logg or \[M/H\] header \(1\): no_meta.txt"):
        SED.import_grid(src, name="Grid", model_root=tmp_path / "models", build_cube=False)
    assert (tmp_path / "models" / "Grid" / "Grid.h5").is_file()

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


def test_filters_cite_exits_1_when_any_target_failed(filters, monkeypatch, capsys):
    result = {"instruments": [{"instrument": "Fac/Cam", "bands": None}], "acknowledgements": [],
              "references": [{"key": "ref", "used_by": ["Fac/Cam"], "bibtex": "@misc{ref}"}],
              "errors": ["Other/IR: not cached"]}
    monkeypatch.setattr(filters, "citations", classmethod(lambda cls, *targets, **kwargs: result))
    assert run_cli(monkeypatch, "filters-cite", "Fac/Cam", "Other/IR") == 1
    assert "[filters] Other/IR: not cached" in capsys.readouterr().out


@pytest.mark.parametrize("source_errors,catalogue_errors,offline,expected", [
    ([], [], False, 0),
    (["SVO: no metadata for Fac/Cam.g"], [], False, 1),
    ([], ["NJM unreachable"], False, 1),
    ([], ["info.json is not cached and working offline"], True, 0),
])
def test_filters_info_exit_status(filters, monkeypatch, source_errors, catalogue_errors, offline, expected):
    from sed_tools import cli

    info = {"facility": "Fac", "instrument": "Cam", "bands": [], "published_source": "", "published": "",
            "published_bands": {}, "components": [], "partial_components": {}, "buildable": [],
            "component_status": "", "component_availability": {"status": "unreviewed", "builder": None},
            "missing_files": [], "errors": source_errors, "atmosphere": False, "platform": "ground",
            "detector_type": "", "detector_type_sources": {}, "availability_source": "svo"}
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
