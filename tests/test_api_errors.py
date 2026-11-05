"""Failures in the public API are raised or warned about: query, fetch, local, the grid writer and
import, the filters (query, citations, plots), tagging and the ML wrappers.

No test uses the network: the autouse fixture fails any real request.
"""

import importlib
import importlib.util
import logging
from pathlib import Path

import pytest
import requests

from sed_tools.api import SED

from helpers_api import CLEAN, GRABBERS, fake_grabber, tagged
from helpers_api import filters, no_network  # noqa: F401 (fixtures)


# ----------------------------------------------------------------------
# Spectral grid sources
# ----------------------------------------------------------------------
@pytest.fixture
def sources(monkeypatch):
    """Replace the four spectral grid grabbers; unspecified ones list nothing."""
    def install(**grabbers):
        for name, target in GRABBERS.items():
            monkeypatch.setattr(target, grabbers.get(name, fake_grabber()))
    return install


def test_query_leaves_out_an_unreachable_njm_and_raises_when_it_is_the_source_asked_for(sources, caplog):
    sources(njm=fake_grabber(["NJMgrid"], available=False), mast=fake_grabber(["BOSZ"]))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"):
        assert [g.name for g in SED.query(include_local=False)] == ["BOSZ"]
    assert not caplog.text  # the NJM client warns; the API does not again
    with pytest.raises(RuntimeError, match="NJM mirror https://mirror.invalid is unreachable"):
        SED.query(source="njm", include_local=False)


def test_query_rejects_an_unknown_source(sources):
    sources()
    with pytest.raises(ValueError, match="Unknown source: sov"):
        SED.query(source="sov")


def test_fetch_error_names_every_source_that_failed(sources, tmp_path, caplog):
    sources(njm=fake_grabber(available=False), msg=fake_grabber(["Missing"]))
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"), pytest.raises(ValueError) as excinfo:
        SED.fetch("Missing", model_root=tmp_path)
    message = str(excinfo.value)
    assert "Spectral grid 'Missing' could not be fetched" in message
    assert "njm: discovery failed: NJM mirror https://mirror.invalid is unreachable" in message
    assert "njm" not in caplog.text  # the NJM client warns; the API does not again
    assert "msg: no metadata for Missing" in message
    assert "svo: no metadata for Missing" in message
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


def test_fetch_from_mast_does_not_wait_for_the_keyboard(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("SED.fetch must not prompt"))
    with pytest.raises(ValueError, match="mast: MAST \\(BOSZ\\) grids are fetched with 'sed-tools grids --source mast'"):
        SED.fetch("BOSZ-2024-r10000", source="mast", model_root=tmp_path)


def test_fetch_rejects_an_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="Unknown source: sov"):
        SED.fetch("Grid", source="sov", model_root=tmp_path)


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


# ----------------------------------------------------------------------
# Detector-type tagging
# ----------------------------------------------------------------------
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
