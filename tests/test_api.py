"""Public API, top level and the 0.4.1/0.5.x legacy surface: query, fetch and local, the local grids, the
filters, tagging and the ML wrappers fail loudly; the 0.4.1 names and old module paths still work; the
config file and data directory. One test walks the tables below; each row runs isolated (support.Rows).
"""

import importlib
import importlib.util
import json
import logging
import math
import os
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import requests

import sed_tools
import sed_tools.config as config
import sed_tools.cube.tool as tool
import sed_tools.filter_library as old_library
import sed_tools.filters.library as library
import sed_tools.flux_cube_tool as old_tool
from sed_tools import api, ml
from sed_tools._flux import AB_ZERO_FLUX
from sed_tools.api import SED, Filters, MLGenerator, SpectralGrid, SpectralGridInfo, Spectrum
from sed_tools.collision_config import CollisionConfig
from sed_tools.ml_sed_completer import SEDCompleter, SEDCompleterNetwork
from sed_tools.ml_sed_generator import SEDGenerator
from sed_tools.models import SED as GridFinder
from sed_tools.njm_filter_grabber import _STATE, NJMFilterGrabber

from support import (CLEAN, GRABBERS, GRID, KURUCZ, NJM, STALE, Session, answer, atmosphere_site, cpu_torch,
                     fake_grabber, filters_api, njm_grids, njm_index, scenarios, tiny_library, write_flux_cube)

SCENARIOS, row = scenarios()


def _raise(error):
    raise error


def _spectrum(teff=5000.0, logg=4.0, metallicity=0.0, filename=None):
    return Spectrum(wavelength=np.array([1000.0, 2000.0, 3000.0]), flux=np.array([1.0, 2.0, 3.0]), teff=teff,
                    logg=logg, metallicity=metallicity, filename=filename)


# ----------------------------------------------------------------------
# Failures are raised or warned about: spectral grid sources, local grids, filters, tagging, ML
# ----------------------------------------------------------------------
def _sources(t, **grabbers):
    """The four spectral grid grabbers replaced; those not given list nothing."""
    for name, target in GRABBERS.items():
        t.mp.setattr(target, grabbers.get(name, fake_grabber()))


def _api_warnings(t):
    return t.caplog.at_level(logging.WARNING, logger="sed_tools.api")


@row
def query_leaves_out_an_unreachable_njm_and_raises_when_it_is_the_source_asked_for(t):
    _sources(t, njm=fake_grabber(["NJMgrid"], available=False), mast=fake_grabber(["BOSZ"]))
    with _api_warnings(t):
        assert [g.name for g in SED.query(include_local=False)] == ["BOSZ"]
    assert not t.caplog.text  # the NJM client warns; the API does not again
    with pytest.raises(RuntimeError, match="NJM mirror https://mirror.invalid is unreachable"):
        SED.query(source="njm", include_local=False)


@row
def fetch_error_names_every_source_that_failed(t):
    _sources(t, njm=fake_grabber(available=False), msg=fake_grabber(["Missing"]))
    with _api_warnings(t), pytest.raises(ValueError) as excinfo:
        SED.fetch("Missing", model_root=t.tmp)
    message = str(excinfo.value)
    assert "Spectral grid 'Missing' could not be fetched" in message
    assert "njm: discovery failed: NJM mirror https://mirror.invalid is unreachable" in message
    assert "njm" not in t.caplog.text  # the NJM client warns; the API does not again
    assert "msg: no metadata for Missing" in message
    assert "svo: no metadata for Missing" in message
    assert "svo: no metadata for Missing, trying next" in t.caplog.text


@row
def fetch_falls_back_to_the_next_source_with_a_warning(t):
    _sources(t, njm=fake_grabber(["Grid"], metadata=RuntimeError("NJM mirror has no files for Grid")),
             svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], download_error=RuntimeError("SVO timed out")),
             msg=fake_grabber(["Grid"], metadata={"m": 1}, downloaded=1))
    t.mp.setattr("sed_tools.spectra_cleaner.clean_model_dir", lambda *a, **k: dict(CLEAN))
    with _api_warnings(t):
        sed = SED.fetch("Grid", model_root=t.tmp)
    assert sed.grid.source == "msg"
    assert "njm: NJM mirror has no files for Grid, trying next" in t.caplog.text
    assert "svo: SVO timed out, trying next" in t.caplog.text


@row
def fetch_stops_when_a_failed_download_left_files_behind(t):
    _sources(t, svo=fake_grabber(["Grid"], metadata=[{"fid": 1}], downloaded=2,
                                 download_error=RuntimeError("1 of 3 failed")),
             msg=fake_grabber(["Grid"], metadata={"m": 1}, downloaded=1))
    with pytest.raises(RuntimeError, match="svo failed after changing .*mixed with its files.*svo: 1 of 3 failed"):
        SED.fetch("Grid", model_root=t.tmp)
    assert sorted(p.name for p in (t.tmp / "Grid").iterdir()) == ["spectrum0.txt", "spectrum1.txt"]


@row
def fetch_from_mast_does_not_wait_for_the_keyboard(t):
    t.mp.setattr("builtins.input", lambda prompt="": pytest.fail("SED.fetch must not prompt"))
    with pytest.raises(ValueError, match="mast: MAST \\(BOSZ\\) grids are fetched with "
                                         "'sed-tools grids --source mast'"):
        SED.fetch("BOSZ-2024-r10000", source="mast", model_root=t.tmp)


#: name: the call that names the misspelt source.
UNKNOWN_SOURCE = {
    "query_rejects_an_unknown_source": lambda t: (_sources(t), SED.query(source="sov")),
}


def _unknown_source(t, call):
    with pytest.raises(ValueError, match="Unknown source: sov"):
        call(t)


#: A flux cube of one point at 1000 and 2000 Å, its flux zero.
ONE_POINT_CUBE = struct.pack("4i", 1, 1, 1, 2) + np.array([5000.0, 4.0, 0.0, 1000.0, 2000.0, 0.0, 0.0]).tobytes()


def _local_grid(t, files):
    """Files under the model root, which SED.query then reads."""
    for name, data in files.items():
        (t.tmp / name).parent.mkdir(exist_ok=True)
        (t.tmp / name).write_bytes(data)
    t.mp.setattr(SED, "_model_root", t.tmp)


@row
def query_raises_for_a_local_grid_whose_cube_header_is_unreadable(t):
    _local_grid(t, {"Broken/flux_cube.bin": b"abc"})
    with pytest.raises(ValueError, match="flux cube header of local grid 'Broken'.*sed-tools rebuild --grids Broken"):
        SED.query(include_remote=False)


@row
def query_raises_for_a_local_grid_whose_lookup_table_is_unreadable(t):
    _local_grid(t, {"Grid/flux_cube.bin": ONE_POINT_CUBE, "Grid/lookup_table.csv": b""})
    with pytest.raises(ValueError, match="lookup_table.csv of local grid 'Grid'"):
        SED.query(include_remote=False)


@row
def local_raises_for_a_listed_spectrum_it_cannot_read(t):
    grid = t.tmp / "Grid"
    grid.mkdir()
    (grid / "flux_cube.bin").write_bytes(ONE_POINT_CUBE)
    (grid / "lookup_table.csv").write_text("file_name,teff,logg,meta\nbad.txt,5000,4.0,0.0\n")
    (grid / "bad.txt").write_text("not a spectrum\n")
    with pytest.raises(ValueError, match="Cannot read spectrum .*bad.txt listed in"):
        SED.local("Grid", model_root=t.tmp)


@row
def grid_write_raises_when_there_is_nothing_to_bundle(t):
    grid = SpectralGrid(name="Grid", source="local", spectra=[_spectrum(filename="a.dat")])
    with pytest.raises(RuntimeError, match="No .txt spectra in .* to bundle into Grid.h5"):
        grid.write(t.tmp / "out", build_flux_cube=False)


@row
def grid_write_lists_in_its_lookup_table_the_files_it_writes(t):
    grid = SpectralGrid(name="Grid", source="local", spectra=[_spectrum(), _spectrum(teff=5500.0, logg=4.5)])
    out = grid.write(t.tmp / "out", build_flux_cube=False, build_h5=False)
    listed = [line.split(",")[0] for line in (out / "lookup_table.csv").read_text().splitlines()[1:]]
    assert listed == ["teff5000_logg4.00_meta0.00.txt", "teff5500_logg4.50_meta0.00.txt"]
    assert all((out / name).is_file() for name in listed)


@row
def import_dry_run_counts_spectra_lacking_a_parameter(t):
    src = t.tmp / "src"
    src.mkdir()
    _spectrum().save(src / "good.txt")
    _spectrum(metallicity=float("nan")).save(src / "no_meta.txt")
    report = SED.import_grid(src, name="Grid", model_root=t.tmp / "models", dry_run=True)
    assert report["n_missing"] == 1


@row
def filter_query_warns_about_discovery_errors(t):
    from sed_tools.filter_library import FilterLibrary

    filters = filters_api(t)
    report = {"entries": [{"key": "HST", "label": "HST", "sources": ["original"]}],
              "errors": ["SVO facility list: connection refused"]}
    t.mp.setattr(FilterLibrary, "facilities", lambda self: report)
    with _api_warnings(t):
        rows = filters.query(include_local=False)
    assert [row["facility"] for row in rows] == ["HST"]
    assert "Filter discovery: SVO facility list: connection refused" in t.caplog.text


@row
def bibtex_citations_raise_when_a_target_failed(t):
    from sed_tools.filter_catalogue import CatalogueError

    filters = filters_api(t)
    result = {"instruments": [], "references": [], "acknowledgements": [], "errors": ["A/B: not cached"]}
    t.mp.setattr(filters, "citations", classmethod(lambda cls, *targets, **kwargs: result))
    with pytest.raises(CatalogueError, match="Cannot cite every target: A/B: not cached"):
        filters.cite("A/B", format="bibtex")
    assert "[filters] A/B: not cached" in filters.cite("A/B", format="text")


@row
def plot_warns_when_njm_cannot_resolve_an_instrument_name(t):
    from sed_tools.filter_catalogue import CatalogueError

    filters = filters_api(t)
    fetched = t.tmp / "fetched" / "Fac" / "Cam"
    fetched.mkdir(parents=True)
    (fetched / "g.dat").write_text("Wavelength,Transmission\n4000,0\n4500,1\n5000,0\n")
    catalogue = SimpleNamespace(resolve=lambda facility, instrument: _raise(
        CatalogueError(f"NJM does not serve {facility}/{instrument}")))
    t.mp.setattr(filters, "_fetch_published", classmethod(lambda cls, fac, inst, filter_root=None: fetched))
    with _api_warnings(t):
        chosen = filters._plot_bands(("Fac/Cam",), None, catalogue, fetch=True)
    assert chosen == [("Fac", "Cam", "g", fetched)]
    assert "NJM does not serve Fac/Cam; fetching Fac/Cam under the name given" in t.caplog.text


@row
def ml_generator_parameter_ranges_need_the_trained_ranges(t):
    generator = MLGenerator(base_dir=None)
    generator._generator = SimpleNamespace(config={"parameter_ranges": {
        "teff": [3000, 9000], "logg": [0, 5], "meta": [-2, 0.5]}})
    assert generator.parameter_ranges() == {"teff": (3000, 9000), "logg": (0, 5), "metallicity": (-2, 0.5)}
    generator._generator = SimpleNamespace(config={})
    with pytest.raises(KeyError, match="parameter_ranges"):
        generator.parameter_ranges()


def _precompute_writes(t, *names):
    """precompute_flux_cube replaced by one that writes these files (or nothing)."""
    def fake(model_dir, output_file):
        for name in names:
            (Path(model_dir) / name).write_text("built")

    t.mp.setattr(importlib.import_module("sed_tools.precompute_flux_cube"), "precompute_flux_cube", fake)


@row
def flux_cube_helper_raises_when_nothing_new_was_written(t):
    from sed_tools.api import _build_flux_cube

    (t.tmp / "flux_cube.bin").write_text("from an earlier build")
    _precompute_writes(t)
    with pytest.raises(RuntimeError, match="No flux cube was built in"):
        _build_flux_cube(t.tmp)


@row
def grid_write_says_when_it_built_one_cube_per_variant(t):
    _precompute_writes(t, "variants_index.csv")
    SpectralGrid(name="Grid", source="local", spectra=[_spectrum()]).write(t.tmp / "out", build_h5=False)
    out = t.capsys.readouterr().out
    assert "Built one flux cube per extra-axis variant" in out and "Built flux_cube.bin" not in out


@row
def ml_generator_writes_outputs_beside_the_loaded_model(t):
    calls = []

    class Generator:
        def __init__(self, path=None):
            self.config = {}

        def generate_with_outputs(self, **kwargs):
            calls.append(kwargs["output_dir"])

    t.mp.setattr("sed_tools.ml_sed_generator.SEDGenerator", Generator)
    (t.tmp / "models" / "gen").mkdir(parents=True)
    MLGenerator(base_dir=t.tmp).load("gen").generate_with_outputs(5800, 4.4, 0.0)
    assert calls == [str(t.tmp / "models" / "gen" / "SED")]


def tagged(path):
    return SimpleNamespace(path=path, detector_type="photon", source="svo", filter_id="F", changed=True,
                           migrated=False, note="", kept=False, resolved_type="photon", resolved_source="svo")


def _tag_with(t, bands, tag):
    """Filter files for ``bands`` in the row's folder, tagged by ``tag``; returns Filters."""
    import sed_tools.filter_metadata as fm

    for band in bands:
        (t.tmp / f"{band}.dat").write_text("Wavelength,Transmission\n1,1\n")
    t.mp.setattr(fm, "SVOMetadataClient", lambda: None)
    t.mp.setattr(fm, "tag_filter_file", tag)
    return filters_api(t)


@row
def tagging_one_facility_finishes_every_file_then_lists_failures(t):
    def tag(path, **kwargs):
        if path.stem == "r":
            raise requests.ConnectionError("SVO timed out")
        return tagged(path)

    filters = _tag_with(t, "gri", tag)
    with pytest.raises(RuntimeError, match=r"(?s)1 of 3 filter files could not be tagged.*r\.dat: SVO timed out"):
        filters.tag_detector_types(t.tmp, facility="Fac")
    out = t.capsys.readouterr().out
    assert "g.dat: photon [svo]" in out and "i.dat: photon [svo]" in out


@row
def tagging_stops_at_once_when_svo_is_unreachable(t):
    import sed_tools.filter_metadata as fm

    calls = []

    def tag(path, **kwargs):
        calls.append(path.stem)
        raise fm.SVOUnreachableError("SVO unreachable")

    filters = _tag_with(t, "gr", tag)
    with pytest.raises(fm.SVOUnreachableError):
        filters.tag_detector_types(t.tmp, facility="Fac")
    assert len(calls) == 1


# ----------------------------------------------------------------------
# Public API (0.4.1 names, most of them) that nothing else calls: Spectrum, SpectralGrid, SpectralGridInfo
# ----------------------------------------------------------------------
@row
def spectrum_aliases_and_dict(t):
    s = _spectrum(5000.0, 4.5, 0.0, "5000.0.txt")
    assert s.wl is s.wavelength and s.fl is s.flux
    assert s.to_dict() == {"wavelength": s.wavelength, "flux": s.flux, "teff": 5000.0, "logg": 4.5,
                           "metallicity": 0.0, "filename": "5000.0.txt"}


@row
def spectral_grid_axes_and_indexing(t):
    grid = SpectralGrid("G", "local", [_spectrum(*point, f"{point[0]}.txt") for point in
                                       [(6000.0, 4.0, 0.0), (5000.0, 4.5, -1.0), (5000.0, 4.0, 0.0)]])
    assert grid.teff_grid.tolist() == [5000.0, 6000.0]
    assert grid.logg_grid.tolist() == [4.0, 4.5]
    assert grid.metallicity_grid.tolist() == [-1.0, 0.0]
    assert grid[1].logg == 4.5 and grid[-1].teff == 5000.0


@row
def grid_info_covers_a_point_and_shows_its_ranges(t):
    info = SpectralGridInfo("G", "njm", teff_range=(4000.0, 6000.0), logg_range=(4.0, 5.0))
    assert info.covers(5000.0, 4.5, 3.0)  # no [M/H] range: any value is covered
    assert info.covers() and not info.covers(teff=7000.0) and not info.covers(logg=3.9)
    assert repr(info) == "SpectralGridInfo('G', source='njm', teff=(4000.0, 6000.0), logg=(4.0, 5.0))"
    assert repr(SpectralGridInfo("G", "svo", n_spectra=3)) == "SpectralGridInfo('G', source='svo', n_spectra=3)"


@row
def query_fetch_and_local_reach_the_njm_and_svo_grids(t):
    njm_grids(t)
    for source in ("svo", "msg", "mast"):
        t.mp.setattr(GRABBERS[source], fake_grabber(["SVOgrid"] if source == "svo" else []))
    t.mp.setattr(SED, "_model_root", t.tmp)
    assert [(g.name, g.source) for g in api.query(include_local=False)] == [("SVOgrid", "svo"), (GRID, "njm")]
    sed = api.fetch(GRID, source="njm", teff_min=5900, teff_max=6100)
    assert len(sed.grid) == 4 and sed.grid.teff_grid.tolist() == [6000.0]
    assert [g.name for g in api.query(source="local", include_remote=False)] == [GRID]
    local = api.local(GRID, filter_root=t.tmp / "filters")
    node = next(s for s in local.grid if (s.logg, s.metallicity) == (4.5, 0.0))
    at_node = local(6000.0, 4.5, 0.0)
    assert len(local.grid) == 4 and np.array_equal(at_node.wavelength, node.wl)
    assert np.allclose(at_node.flux, node.fl, rtol=1e-9, atol=0)


def _installed(t):
    """Cool (Teff 4000-5000) and Hot (Teff 5000-7000), at log g 4-5 and [M/H] -1-0, and a folder without a cube."""
    for name, teff in {"Cool": [4000.0, 5000.0], "Hot": [5000.0, 6000.0, 7000.0]}.items():
        write_flux_cube(t.tmp / name / "flux_cube.bin", teff, [4.0, 5.0], [-1.0, 0.0], [4000.0, 5000.0, 6000.0],
                        np.ones((len(teff), 2, 2, 3)))
    (t.tmp / "Empty").mkdir()
    return t.tmp


@row
def find_model_lists_the_grids_covering_a_point_nearest_first(t):
    installed = _installed(t)
    sed = SED(model_root=installed, filter_root=installed / "filters")
    assert [m.name for m in sed.find_model(5000, 4.5, -0.5)] == ["Cool", "Hot"]  # equally near: by name
    assert [m.name for m in sed.find_model(5000, 4.5, -0.5, limit=1)] == ["Cool"]
    [hot] = sed.find_model(6500, 4.5, -0.5)
    assert hot.as_dict() == {"name": "Hot", "flux_cube": str(installed / "Hot" / "flux_cube.bin"),
                             "teff_range": (5000.0, 7000.0), "logg_range": (4.0, 5.0),
                             "metallicity_range": (-1.0, 0.0), "distance": pytest.approx(0.75),
                             "contains_point": True, "covers_range": False}
    assert sed.find_model(8000, 4.5, -0.5) == []
    partial = GridFinder(model_root=installed).find_model(8000, 4.5, -0.5, allow_partial=True)
    assert [(m.name, m.contains_point) for m in partial] == [("Hot", False), ("Cool", False)]
    assert partial[0] < partial[1]  # ModelMatch orders by distance


@row
def model_selects_a_grid_to_interpolate(t):
    installed = _installed(t)
    sed = SED(model_root=installed, filter_root=installed / "filters")
    chosen = sed.model("Hot")
    assert sed.model() is chosen and chosen.name == "Hot"
    assert sed(6500, 4.5, -0.5).flux.tolist() == pytest.approx([1.0, 1.0, 1.0])


@row
def the_grid_finder_resolves_a_model_and_takes_z_for_metallicity(t):
    installed = _installed(t)
    finder = GridFinder(model_root=installed, filter_root=installed / "filters")
    cube, loose = installed / "Hot" / "flux_cube.bin", installed / "Loose.bin"
    loose.write_bytes(cube.read_bytes())
    # the cube, its folder, a grid's name, the name of a <name>.bin under the model root
    for given, found in ((cube, cube), (installed / "Hot", cube), ("Hot", cube), ("Loose", loose)):
        assert finder._resolve_model_path(given) == found
    with pytest.raises(FileNotFoundError, match="Could not resolve model 'Empty'"):
        finder.model("Empty")
    assert finder.model("Loose").flux_cube_path == loose and finder.model("Hot").name == "Hot"
    assert finder.model("Loose").name == finder.model(loose).name == "Loose"  # not the model root's name
    for call in ({"z": -0.5}, {"Z": -0.5}, {"metallicity": -0.5}):
        assert finder(6000, 4.5, **call).flux.tolist() == pytest.approx([1.0, 1.0, 1.0])
    with pytest.raises(TypeError, match="Unexpected keyword arguments: feh"):
        finder(6000, 4.5, feh=0.0)


@row
def coverage_reports_a_local_grid(t):
    shutil.copytree(KURUCZ, t.tmp / GRID)
    summary = SED.coverage(GRID, model_root=t.tmp, plot=False)
    assert summary["n_spectra"] == 12 and summary["axes"]["teff"]["unique"].tolist() == [5750.0, 6000.0, 6250.0]
    assert f"Coverage: {GRID}" in t.capsys.readouterr().out


# ----------------------------------------------------------------------
# MLCompleter, MLGenerator and the 0.4.1 ml API, on a tiny grid
# ----------------------------------------------------------------------
def _tiny_training(t):
    """CPU torch, the tiny library, and no training plots; returns torch."""
    torch = cpu_torch(t)
    tiny_library(t)
    for engine, names in ((SEDCompleter, ("_plot_training_results", "_plot_prediction_examples")),
                          (SEDGenerator, ("_plot_training", "_plot_predictions"))):
        for name in names:
            t.mp.setattr(engine, name, lambda self, *args, **kwargs: None)
    return torch


@row
def ml_completer_trains_and_loads_a_model(t):
    _tiny_training(t)
    completer = SED.ml_completer(model_root=t.tmp)
    assert (completer.base_dir, completer.models_dir) == (t.tmp, "models")
    assert completer.train("lib", epochs=1, batch_size=3) is completer and completer._model.is_trained
    assert [m["name"] for m in ml.Completer.available_models(str(t.tmp / "models"))] == ["lib_completer"]
    loaded = SED.ml_completer(model_root=t.tmp).load("lib_completer")
    assert repr(loaded._model) == f"<Completer(trained @ {t.tmp / 'models' / 'lib_completer'})>"


@row
def ml_generator_trains_lists_and_loads_a_model(t):
    _tiny_training(t)
    generator = SED.ml_generator(model_root=t.tmp)
    assert generator.train("lib", epochs=1, batch_size=3) is generator
    [listed] = MLGenerator.list_models(str(t.tmp / "models"))
    assert listed["name"] == "sed_generator_lib"
    assert listed["parameter_ranges"] == {"teff": [4000.0, 6000.0], "logg": [4.0, 4.5], "meta": [-0.5, 0.0]}
    loaded = SED.ml_generator(model_root=t.tmp).load("sed_generator_lib")
    wl, flux = loaded.generate(5000.0, 4.25, -0.25)
    assert wl.shape == flux.shape == (1000,) and np.isfinite(flux).all()


@row
def the_041_completer_api_loads_and_describes_a_model(t):
    _tiny_training(t)
    lib, first, second = t.tmp / "lib", t.tmp / "models" / "a", t.tmp / "models" / "b"
    for path in (first, second):
        SEDCompleter().train(str(lib), str(path), epochs=1, batch_size=3, hidden_layers=[8], verbose=False)
    loaded = ml.load_completer(str(first))
    assert isinstance(loaded, SEDCompleter) and loaded.config["architecture"]["hidden_layers"] == [8]
    assert repr(ml.Completer()) == "<Completer(untrained)>"
    assert repr(ml.Completer(str(second))) == f"<Completer(trained @ {second})>"
    assert ml.Completer().load(str(first)).model_path == str(first)
    assert sorted(m["name"] for m in ml.Completer.available_models(str(t.tmp / "models"))) == ["a", "b"]


@row
def the_completer_network_moves_to_a_device(t):
    torch = cpu_torch(t)
    network = SEDCompleterNetwork(input_dim=4, output_dim=3, hidden_layers=[2])
    assert network.to(torch.device("cpu")) is network and network.device == torch.device("cpu")


# ----------------------------------------------------------------------
# The NJM filter grabber's 0.4.1 helpers, and smaller pieces
# ----------------------------------------------------------------------
def _filter_mirror(t):
    t.mp.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    grabber = NJMFilterGrabber(base_dir=str(t.tmp / "filters"))
    grabber.session = Session({"index.json": njm_index({"Bepi_Colombo": {"SIMBIO": {}}, "Generic": {"Johnson": {}},
                                                        "NJMOnly": {}}),
                               "filters/Generic/Johnson/Johnson": "B.dat\nV.dat\n"}, base=NJM)
    return grabber


@row
def find_facility_matches_svo_names_to_njm_ones(t):
    mirror = _filter_mirror(t)
    assert mirror.find_facility("Generic") == "Generic"
    assert mirror.find_facility("Bepi-Colom") == "Bepi_Colombo"  # SVO's shortened name
    assert mirror.find_facility("Missing") is None
    assert mirror.get_exclusive_facilities(["Bepi-Colom", "generic"]) == ["NJMOnly"]


@row
def get_filter_info_describes_an_njm_instrument(t):
    mirror = _filter_mirror(t)
    assert mirror.get_filter_info("Generic", "Johnson") == {
        "facility": "Generic", "instrument": "Johnson", "filter_count": 2, "filters": ["B.dat", "V.dat"],
        "url": f"{NJM}/filters/Generic/Johnson/"}
    _STATE.reset()  # a new process, which finds NJM down
    t.mp.setattr(NJMFilterGrabber, "_check_availability", lambda self: False)
    assert NJMFilterGrabber(base_dir=mirror.base_dir).get_filter_info("Generic", "Johnson") == {}


@row
def ui_settings_come_from_the_config_file(t):
    assert config.get_ui_config() == {"terminal_plots": "auto", "terminal_color": "auto"}  # the shipped defaults
    config._CONFIG_FILE.write_text('[ui]\nterminal_plots = "Always"\nterminal_color = "never"\n')
    assert config.get_ui_setting("terminal_plots") == "always"
    assert config.get_ui_setting("unset", default="never") == "never"


@row
def list_txt_spectra_gives_the_names_of_the_top_level_spectra(t):
    for name in ("b.txt", "a.TXT", "c.dat", "sub/d.txt"):
        (t.tmp / name).parent.mkdir(exist_ok=True)
        (t.tmp / name).write_text("1 1\n")
    assert sed_tools.list_txt_spectra(str(t.tmp)) == ["a.TXT", "b.txt"]


@row
def collision_config_repr(t):
    assert repr(CollisionConfig("filter", {"alpha": "0.0"}, source="model")) == (
        "CollisionConfig(strategy='filter', filter_specs={'alpha': '0.0'}, source='model')")


@row
def an_old_module_path_lists_the_moved_modules_names(t):
    import sed_tools.filter_library as old
    import sed_tools.njm_filter_grabber as old_njm

    assert {"FilterLibrary", "get_builder"} <= set(dir(old))
    assert {"threading", "urllib3", "NJMFilterGrabber"} <= set(dir(old_njm))  # restored names too
    assert dir(old_njm) == sorted(dir(old_njm))


@row
def atmospheres_lists_the_installed_sites(t):
    atmosphere_site(t.tmp, "lsst_std", STALE, ["am1.20.dat"])
    assert Filters.atmospheres(t.tmp / "filters", include_remote=False) == {"lsst_std": {
        "altitude_m": 2647.0, "airmass": [1.2], "pwv": [], "unstated_airmass": False, "source": "Rubin",
        "kind": None, "model": None, "model_version": None, "citation": None}}


# ----------------------------------------------------------------------
# Old module paths (sed_tools.filter_library and the rest) forward to the moved modules, also for test tools
# ----------------------------------------------------------------------
def _sentinel():
    return "patched"


@row
def monkeypatch_through_an_old_path_is_undone_on_the_moved_module(t):
    original = library.get_builder
    with pytest.MonkeyPatch.context() as m:
        m.setattr(old_library, "get_builder", _sentinel)
        assert library.get_builder is _sentinel
        m.setattr(old_library, "get_builder", len)  # nested
        assert library.get_builder is len
    assert library.get_builder is original


@row
def monkeypatch_of_a_new_name_through_an_old_path_is_removed_again(t):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(old_library, "not_there_before", 1, raising=False)
        assert library.not_there_before == 1
    assert not hasattr(library, "not_there_before")


@row
def mock_patch_through_an_old_path_is_undone_on_the_moved_module(t):
    original = library.get_builder
    with mock.patch.object(old_library, "get_builder", _sentinel):
        assert library.get_builder is _sentinel
    assert library.get_builder is original
    with mock.patch("sed_tools.filter_library.get_builder", _sentinel):
        assert library.get_builder is _sentinel
    assert library.get_builder is original


@row
def reloading_an_old_path_leaves_the_moved_module_as_it_was(t):
    importlib.reload(old_tool)
    assert old_tool.__name__ == "sed_tools.flux_cube_tool"
    assert type(tool) is types.ModuleType
    assert tool.__name__ == tool.__spec__.name == "sed_tools.cube.tool"
    assert tool.__package__ == "sed_tools.cube"
    t.mp.setattr(tool.Path, "rglob", lambda self, pattern: iter(()))  # search no folders
    old_tool.discover_flux_cube_files([])  # its `from .model import` needs the moved module's own package


@row
def every_module_star_imports_and_keeps_its_recorded_names(t):
    # Older code runs `from sed_tools.svo_filter_grabber import *`: it fails outright once a module's __all__
    # names something gone, and a dropped name breaks whoever used it. tests/star_imports.py records the
    # surface in tests/fixtures/star_imports.json (its docstring gives the one command); names may be added.
    # A module whose optional dependency is not installed is skipped. Run apart, so that importing every
    # module leaves this process as it was.
    out = t.tmp / "surface.json"
    done = subprocess.run([sys.executable, str(Path(__file__).with_name("star_imports.py")), "--out", str(out)],
                          env={**os.environ, "HOME": str(t.tmp)}, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    now = json.loads(out.read_text())
    recorded = json.loads((Path(__file__).parent / "fixtures" / "star_imports.json").read_text())
    failing = {module: names for module, names in now.items() if isinstance(names, str)}
    lost = {module: "the module is gone" if module not in now else sorted(set(names) - set(now[module]))
            for module, names in recorded.items()
            if module not in now or (isinstance(now[module], list) and set(names) - set(now[module]))}
    assert not failing and not lost, f"star imports that fail: {failing}; recorded names no longer given: {lost}"


# ----------------------------------------------------------------------
# Programmatic use as 0.4.1 documented it: grid names, find_atm/find_atmospheres, interpolation, photometry
# ----------------------------------------------------------------------
DEMO_WL = np.array([4000.0, 5000.0])


def _cube(t, name, teff, logg, flux=1.0):
    """A flat 2 x 2 x 2 cube at [M/H] -1 and 0, 4000 and 5000 Å."""
    write_flux_cube(t.tmp / name / "flux_cube.bin", teff, logg, [-1.0, 0.0], DEMO_WL, np.full((2, 2, 2, 2), flux))


def _evaluated(t, flux, **roots):
    """A flat cube found with find_atmospheres, loaded with model(), and evaluated at 1500 K, log g 1.5, -0.5."""
    _cube(t, "demo", [1000.0, 2000.0], [1.0, 2.0], flux)
    sed = GridFinder(model_root=t.tmp, **roots)
    atm = sed.find_atmospheres(teff_range=(1000.0, 2000.0), logg_range=(1.0, 2.0), metallicity_range=(-1.0, 0.0))
    return sed.model(atm[0])(teff=1500.0, logg=1.5, metallicity=-0.5)


def _gaia_photometry(t):
    """Photometry of the flat cube (2.0) through a GAIA filter."""
    curve = t.tmp / "filters" / "GAIA" / "GAIA.dat"
    curve.parent.mkdir(parents=True)
    curve.write_text("# wavelength_unit = angstrom\n4000 1.0\n5000 1.0\n")
    return _evaluated(t, 2.0, filter_root=t.tmp / "filters").photometry("Gaia")


@row
def spectral_grid_names_preserve_api_compatibility(t):
    from sed_tools import Catalog, CatalogInfo, SpectralGrid, SpectralGridInfo, find_atmospheres, find_grids
    from sed_tools.api import SED

    assert SpectralGrid is Catalog
    assert SpectralGridInfo is CatalogInfo
    assert find_grids is find_atmospheres
    grid = SpectralGrid(name="example", source="local")
    sed = SED(catalog=grid)
    assert sed.grid is sed.cat is grid


@row
def spectral_grid_discovery_names(t):
    from sed_tools import SED, find_grids

    _cube(t, "demo", [4000.0, 6000.0], [3.0, 5.0])
    sed = SED(model_root=t.tmp)
    assert sed.available_grids() == sed.available_models() == ["demo"]
    matches = find_grids(teff_range=(4500.0, 5500.0), model_root=t.tmp)
    assert [match.name for match in matches] == ["demo"]


@row
def find_atm_alias(t):
    _cube(t, "demo", [4000.0, 6000.0], [3.0, 5.0])
    matches = sed_tools.find_atm(teff_range=(4500.0, 5500.0), logg_range=(3.5, 4.5),
                                 Z_range=(-0.5, 0.0), model_root=t.tmp)  # within the cube's [M/H] range [-1, 0]
    assert len(matches) >= 1


@row
def find_atmospheres_returns_model_match(t):
    _cube(t, "m1", [4000.0, 6000.0], [3.0, 5.0])
    matches = GridFinder(model_root=t.tmp).find_atmospheres(teff_range=(4500.0, 5500.0), logg_range=(3.5, 4.5),
                                                          metallicity_range=(-0.5, 0.0))
    assert len(matches) >= 1
    m = matches[0]
    assert hasattr(m, "name")
    assert hasattr(m, "contains_point")
    assert hasattr(m, "teff_range")
    assert hasattr(m, "logg_range")
    assert hasattr(m, "metallicity_range")


@row
def find_atmospheres_no_match(t):
    _cube(t, "m1", [4000.0, 5000.0], [3.0, 4.0])
    matches = GridFinder(model_root=t.tmp).find_atmospheres(teff_range=(50000.0, 60000.0), logg_range=(3.5, 4.5),
                                                          metallicity_range=(-0.5, 0.0))
    assert matches == []


@row
def interpolation_returns_correct_shape(t):
    result = _evaluated(t, 3.0)
    assert result.wavelength.shape == DEMO_WL.shape
    assert result.flux.shape == DEMO_WL.shape


@row
def interpolation_flat_field(t):
    flux_value = 2.0
    result = _evaluated(t, flux_value)
    assert np.allclose(result.flux, flux_value)


@row
def photometry_ab_magnitude_finite(t):
    phot = _gaia_photometry(t)
    assert "GAIA" in phot
    assert math.isfinite(phot["GAIA"].magnitude)
    assert phot["GAIA"].system == "AB"


@row
def photometry_magnitude_formula(t):
    g = _gaia_photometry(t)["GAIA"]
    expected = -2.5 * math.log10(g.flux_density / AB_ZERO_FLUX)
    assert math.isclose(g.magnitude, expected)


@row
def evaluated_sed_property_aliases(t):
    result = _evaluated(t, 1.0)
    assert np.array_equal(result.wl, result.wavelength)
    assert np.array_equal(result.fl, result.flux)
    assert math.isclose(result.teff, 1500.0)
    assert math.isclose(result.logg, 1.5)
    assert math.isclose(result.metallicity, -0.5)


# ----------------------------------------------------------------------
# The config file and the data directory
# ----------------------------------------------------------------------
@row
def env_var_overrides_default(t):
    from importlib import reload

    import sed_tools.config as cfg

    t.mp.setenv("SED_DATA_DIR", str(t.tmp))
    reload(cfg)  # re-import to pick up the environment
    assert cfg.get_data_dir() == t.tmp
    t.mp.delenv("SED_DATA_DIR")
    reload(cfg)


@row
def data_dir_default_is_home_relative(t):
    from sed_tools.config import _DEFAULT_DATA_DIR

    home = Path("~").expanduser()
    assert str(_DEFAULT_DATA_DIR).startswith(str(home))


@row
def load_toml_missing_file_is_empty_unless_strict(t):
    missing = t.tmp / "absent.toml"
    assert config.load_toml(missing) == {}
    with pytest.raises(FileNotFoundError):
        config.load_toml(missing, strict=True)


@row
def load_toml_malformed_file_raises_naming_it(t):
    bad = t.tmp / "bad.toml"
    bad.write_text('data_dir = "unterminated\n')
    for strict in (False, True):
        with pytest.raises(ValueError, match="bad.toml"):
            config.load_toml(bad, strict=strict)


@row
def malformed_user_config_stops_data_dir_lookup(t):
    # Used to fall back silently to the default data directory.
    bad = t.tmp / "config.toml"
    bad.write_text("data_dir = \n")
    t.mp.setattr(config, "_CONFIG_FILE", bad)
    t.mp.delenv("SED_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="config.toml"):
        config.get_data_dir()


@row
def move_data_removes_drained_dirs_and_keeps_conflicts(t):
    old, new = t.tmp / "old", t.tmp / "new"
    for root, text in ((old, "old"), (new, "new")):
        (root / "filters" / "A").mkdir(parents=True)
        (root / "filters" / "A" / "same.dat").write_text("x")
        (root / "stellar_models" / "B").mkdir(parents=True)
        (root / "stellar_models" / "B" / "clash.txt").write_text(text)
    conflicts = config._move_data(old, new)
    assert not (old / "filters").exists()
    assert (old / "stellar_models" / "B" / "clash.txt").read_text() == "old"
    assert (new / "stellar_models" / "B" / "clash.txt").read_text() == "new"
    assert conflicts == [(str(old / "stellar_models" / "B" / "clash.txt"),
                          str(new / "stellar_models" / "B" / "clash.txt"))]


#: name: the answer to the config menu's data-directory question, which keeps it ("q" used to be saved).
CONFIG_KEEPS = {"config_menu_keeps_the_data_directory_on_enter": "", "config_menu_keeps_the_data_directory_on_q": "q",
                "config_menu_keeps_the_data_directory_on_b": "b"}


def _config_menu(t, reply):
    """The config menu (14) answered ``reply``; returns the data directories it saved."""
    from sed_tools import cli

    answer(t.mp, reply)
    changed = []
    t.mp.setattr(cli, "set_data_dir", lambda path, **k: changed.append(path))
    cli.run_config_flow()
    return changed


def _config_menu_keeps(t, reply):
    changed = _config_menu(t, reply)
    assert changed == []  # "q" used to be saved as data_dir = "q"


@row
def config_menu_takes_any_other_answer_as_a_path(t):
    changed = _config_menu(t, str(t.tmp / "q"))
    assert changed == [str(t.tmp / "q")]


def test_api(rows):
    rows(UNKNOWN_SOURCE, _unknown_source)
    rows(SCENARIOS)
    rows(CONFIG_KEEPS, _config_menu_keeps)
