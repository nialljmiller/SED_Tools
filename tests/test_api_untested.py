"""Public Python API (0.4.1 names, most of them) that no other test calls.

NJM and SVO are the suite's stand-ins; the ML models are tiny and trained on the CPU. Nothing touches the
network or writes outside tmp_path.
"""

import shutil

import numpy as np
import pytest
import requests

import sed_tools
from sed_tools import api, config, ml
from sed_tools.api import SED, Filters, MLGenerator, SpectralGrid, SpectralGridInfo, Spectrum
from sed_tools.collision_config import CollisionConfig
from sed_tools.ml_sed_completer import SEDCompleter, SEDCompleterNetwork
from sed_tools.ml_sed_generator import SEDGenerator
from sed_tools.models import SED as GridFinder
from sed_tools.njm_filter_grabber import _STATE, NJMFilterGrabber
from test_cleanup_interface import GRABBERS, fake_grabber
from test_cleanup_ml import cpu_torch, tiny_library  # noqa: F401 (fixtures)
from test_release_decide import GRID, KURUCZ, njm, njm_grid  # noqa: F401 (fixtures)
from test_release_library import NJM, Served, njm_index
from test_release_njmdown import STALE, atmosphere_site


class NetworkUsed(BaseException):
    """Raised past every ``except Exception`` so a network call cannot be swallowed."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise NetworkUsed("tests must not use the network")

    monkeypatch.setattr(requests.Session, "send", refuse)


# ----------------------------------------------------------------------
# Spectrum, SpectralGrid, SpectralGridInfo
# ----------------------------------------------------------------------
def spectrum(teff, logg, meta):
    return Spectrum(np.array([4000.0, 5000.0]), np.array([1.0, 2.0]), teff, logg, meta, filename=f"{teff}.txt")


def test_spectrum_aliases_and_dict():
    s = spectrum(5000.0, 4.5, 0.0)
    assert s.wl is s.wavelength and s.fl is s.flux
    assert s.to_dict() == {"wavelength": s.wavelength, "flux": s.flux, "teff": 5000.0, "logg": 4.5,
                           "metallicity": 0.0, "filename": "5000.0.txt"}


def test_spectral_grid_axes_and_indexing():
    grid = SpectralGrid("G", "local", [spectrum(t, g, m) for t, g, m in
                                       [(6000.0, 4.0, 0.0), (5000.0, 4.5, -1.0), (5000.0, 4.0, 0.0)]])
    assert grid.teff_grid.tolist() == [5000.0, 6000.0]
    assert grid.logg_grid.tolist() == [4.0, 4.5]
    assert grid.metallicity_grid.tolist() == [-1.0, 0.0]
    assert grid[1].logg == 4.5 and grid[-1].teff == 5000.0


def test_grid_info_covers_a_point_and_shows_its_ranges():
    info = SpectralGridInfo("G", "njm", teff_range=(4000.0, 6000.0), logg_range=(4.0, 5.0))
    assert info.covers(5000.0, 4.5, 3.0)  # no [M/H] range: any value is covered
    assert info.covers() and not info.covers(teff=7000.0) and not info.covers(logg=3.9)
    assert repr(info) == "SpectralGridInfo('G', source='njm', teff=(4000.0, 6000.0), logg=(4.0, 5.0))"
    assert repr(SpectralGridInfo("G", "svo", n_spectra=3)) == "SpectralGridInfo('G', source='svo', n_spectra=3)"


# ----------------------------------------------------------------------
# query, fetch and local, from the NJM and SVO stand-ins
# ----------------------------------------------------------------------
def test_query_fetch_and_local_reach_the_njm_and_svo_grids(njm, tmp_path, monkeypatch):
    for source in ("svo", "msg", "mast"):
        monkeypatch.setattr(GRABBERS[source], fake_grabber(["SVOgrid"] if source == "svo" else []))
    monkeypatch.setattr(SED, "_model_root", tmp_path)
    assert [(g.name, g.source) for g in api.query(include_local=False)] == [("SVOgrid", "svo"), (GRID, "njm")]

    sed = api.fetch(GRID, source="njm", teff_min=5900, teff_max=6100)
    assert len(sed.grid) == 4 and sed.grid.teff_grid.tolist() == [6000.0]
    assert [g.name for g in api.query(source="local", include_remote=False)] == [GRID]
    local = api.local(GRID, filter_root=tmp_path / "filters")
    node = next(s for s in local.grid if (s.logg, s.metallicity) == (4.5, 0.0))
    at_node = local(6000.0, 4.5, 0.0)
    assert len(local.grid) == 4 and np.array_equal(at_node.wavelength, node.wl)
    assert np.allclose(at_node.flux, node.fl, rtol=1e-9, atol=0)


# ----------------------------------------------------------------------
# SED.find_model, SED.model and SED.coverage
# ----------------------------------------------------------------------
@pytest.fixture
def installed(tmp_path, cube_writer):
    """Cool (Teff 4000-5000) and Hot (Teff 5000-7000), at log g 4-5 and [M/H] -1-0, and a folder without a cube."""
    for name, teff in {"Cool": [4000.0, 5000.0], "Hot": [5000.0, 6000.0, 7000.0]}.items():
        flux = np.ones((len(teff), 2, 2, 3))
        cube_writer(tmp_path / name / "flux_cube.bin", teff, [4.0, 5.0], [-1.0, 0.0], [4000.0, 5000.0, 6000.0], flux)
    (tmp_path / "Empty").mkdir()
    return tmp_path


def test_find_model_lists_the_grids_covering_a_point_nearest_first(installed):
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


def test_model_selects_a_grid_to_interpolate(installed):
    sed = SED(model_root=installed, filter_root=installed / "filters")
    with pytest.raises(RuntimeError, match="No model loaded"):
        sed.model()
    chosen = sed.model("Hot")
    assert sed.model() is chosen and chosen.name == "Hot"
    assert sed(6500, 4.5, -0.5).flux.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_the_grid_finder_resolves_a_model_and_takes_z_for_metallicity(installed):
    finder = GridFinder(model_root=installed, filter_root=installed / "filters")
    with pytest.raises(RuntimeError, match="No atmosphere selected"):
        finder(5000, 4.5, 0.0)
    cube, loose = installed / "Hot" / "flux_cube.bin", installed / "Loose.bin"
    loose.write_bytes(cube.read_bytes())
    # the cube, its folder, a grid's name, the name of a <name>.bin under the model root
    for given, found in ((cube, cube), (installed / "Hot", cube), ("Hot", cube), ("Loose", loose)):
        assert finder._resolve_model_path(given) == found
    with pytest.raises(FileNotFoundError, match="Could not resolve model 'Empty'"):
        finder.model("Empty")
    assert finder.model("Loose").flux_cube_path == loose and finder.model("Hot").name == "Hot"
    for call in ({"z": -0.5}, {"Z": -0.5}, {"metallicity": -0.5}):
        assert finder(6000, 4.5, **call).flux.tolist() == pytest.approx([1.0, 1.0, 1.0])
    with pytest.raises(TypeError, match="Unexpected keyword arguments: feh"):
        finder(6000, 4.5, feh=0.0)
    with pytest.raises(TypeError, match="Metallicity value missing"):
        finder(6000, 4.5)


def test_coverage_reports_a_local_grid(tmp_path, capsys):
    shutil.copytree(KURUCZ, tmp_path / GRID)
    summary = SED.coverage(GRID, model_root=tmp_path, plot=False)
    assert summary["n_spectra"] == 12 and summary["axes"]["teff"]["unique"].tolist() == [5750.0, 6000.0, 6250.0]
    assert f"Coverage: {GRID}" in capsys.readouterr().out


# ----------------------------------------------------------------------
# MLCompleter, MLGenerator and the 0.4.1 ml API, on a tiny grid
# ----------------------------------------------------------------------
@pytest.fixture
def no_training_plots(monkeypatch):
    for engine, names in ((SEDCompleter, ("_plot_training_results", "_plot_prediction_examples")),
                          (SEDGenerator, ("_plot_training", "_plot_predictions"))):
        for name in names:
            monkeypatch.setattr(engine, name, lambda self, *args, **kwargs: None)


def test_ml_completer_trains_and_loads_a_model(cpu_torch, tiny_library, no_training_plots, tmp_path):
    completer = SED.ml_completer(model_root=tmp_path)
    assert (completer.base_dir, completer.models_dir) == (tmp_path, "models")
    with pytest.raises(FileNotFoundError, match="No flux_cube.bin found in"):
        completer.train("missing")
    assert completer.train("lib", epochs=1, batch_size=3) is completer and completer._model.is_trained
    assert [m["name"] for m in ml.Completer.available_models(str(tmp_path / "models"))] == ["lib_completer"]
    loaded = SED.ml_completer(model_root=tmp_path).load("lib_completer")
    assert repr(loaded._model) == f"<Completer(trained @ {tmp_path / 'models' / 'lib_completer'})>"


def test_ml_generator_trains_lists_and_loads_a_model(cpu_torch, tiny_library, no_training_plots, tmp_path):
    generator = SED.ml_generator(model_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="No flux_cube.bin found in"):
        generator.train("missing")
    assert generator.train("lib", epochs=1, batch_size=3) is generator
    [listed] = MLGenerator.list_models(str(tmp_path / "models"))
    assert listed["name"] == "sed_generator_lib"
    assert listed["parameter_ranges"] == {"teff": [4000.0, 6000.0], "logg": [4.0, 4.5], "meta": [-0.5, 0.0]}
    loaded = SED.ml_generator(model_root=tmp_path).load("sed_generator_lib")
    wl, flux = loaded.generate(5000.0, 4.25, -0.25)
    assert wl.shape == flux.shape == (1000,) and np.isfinite(flux).all()
    with pytest.raises(FileNotFoundError, match="Model not found"):
        loaded.load("missing")


def test_the_041_completer_api_trains_loads_and_describes_a_model(cpu_torch, tiny_library, no_training_plots,
                                                                   tmp_path):
    first, second = tmp_path / "models" / "a", tmp_path / "models" / "b"
    trained = ml.train_completer(str(tiny_library), str(first), epochs=1, batch_size=3, hidden_layers=[8],
                                 verbose=False)
    assert isinstance(trained, SEDCompleter) and trained.config["architecture"]["hidden_layers"] == [8]
    assert ml.load_completer(str(first)).model is not None

    completer = ml.Completer()
    assert repr(completer) == "<Completer(untrained)>"
    assert completer.train(str(tiny_library), str(second), epochs=1, batch_size=3, hidden_layers=[8]) is completer
    assert repr(completer) == f"<Completer(trained @ {second})>"
    assert ml.Completer().load(str(first)).model_path == str(first)
    assert [m["name"] for m in ml.Completer.available_models(str(tmp_path / "models"))] == ["a", "b"]


def test_the_completer_network_moves_to_a_device(cpu_torch):
    network = SEDCompleterNetwork(input_dim=4, output_dim=3, hidden_layers=[2])
    assert network.to(cpu_torch.device("cpu")) is network and network.device == cpu_torch.device("cpu")


# ----------------------------------------------------------------------
# The NJM filter grabber's 0.4.1 helpers
# ----------------------------------------------------------------------
@pytest.fixture
def filter_mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    grabber = NJMFilterGrabber(base_dir=str(tmp_path / "filters"))
    grabber.session = Served({"index.json": njm_index({"Bepi_Colombo": {"SIMBIO": {}}, "Generic": {"Johnson": {}},
                                                       "NJMOnly": {}}),
                              "filters/Generic/Johnson/Johnson": "B.dat\nV.dat\n"})
    return grabber


def test_find_facility_matches_svo_names_to_njm_ones(filter_mirror):
    assert filter_mirror.find_facility("Generic") == "Generic"
    assert filter_mirror.find_facility("Bepi-Colom") == "Bepi_Colombo"  # SVO's shortened name
    assert filter_mirror.find_facility("Missing") is None
    assert filter_mirror.get_exclusive_facilities(["Bepi-Colom", "generic"]) == ["NJMOnly"]


def test_get_filter_info_describes_an_njm_instrument(filter_mirror, monkeypatch):
    assert filter_mirror.get_filter_info("Generic", "Johnson") == {
        "facility": "Generic", "instrument": "Johnson", "filter_count": 2, "filters": ["B.dat", "V.dat"],
        "url": f"{NJM}/filters/Generic/Johnson/"}
    _STATE.reset()  # a new process, which finds NJM down
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: False)
    assert NJMFilterGrabber(base_dir=filter_mirror.base_dir).get_filter_info("Generic", "Johnson") == {}


# ----------------------------------------------------------------------
# Smaller pieces
# ----------------------------------------------------------------------
def test_ui_settings_come_from_the_config_file():
    assert config.get_ui_config() == {"terminal_plots": "auto", "terminal_color": "auto"}  # the shipped defaults
    config._CONFIG_FILE.write_text('[ui]\nterminal_plots = "Always"\nterminal_color = "sometimes"\n')
    assert config.get_ui_setting("terminal_plots") == "always"
    with pytest.warns(UserWarning, match="Invalid ui.terminal_color value 'sometimes'; using 'auto'."):
        assert config.get_ui_setting("terminal_color") == "auto"
    assert config.get_ui_setting("unset", default="never") == "never"
    config._CONFIG_FILE.write_text('ui = "plain"\n')
    assert config.get_ui_config() == {}


def test_list_txt_spectra_gives_the_names_of_the_top_level_spectra(tmp_path):
    for name in ("b.txt", "a.TXT", "c.dat", "sub/d.txt"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("1 1\n")
    assert sed_tools.list_txt_spectra(str(tmp_path)) == ["a.TXT", "b.txt"]


def test_collision_config_repr():
    assert repr(CollisionConfig("filter", {"alpha": "0.0"}, source="model")) == (
        "CollisionConfig(strategy='filter', filter_specs={'alpha': '0.0'}, source='model')")


def test_an_old_module_path_lists_the_moved_modules_names():
    import sed_tools.filter_library as old
    import sed_tools.njm_filter_grabber as old_njm

    assert {"FilterLibrary", "get_builder"} <= set(dir(old))
    assert {"threading", "urllib3", "NJMFilterGrabber"} <= set(dir(old_njm))  # restored names too
    assert dir(old_njm) == sorted(dir(old_njm))


def test_atmospheres_lists_the_installed_sites(tmp_path):
    atmosphere_site(tmp_path, "lsst_std", STALE, ["am1.20.dat"])
    assert Filters.atmospheres(tmp_path / "filters", include_remote=False) == {"lsst_std": {
        "altitude_m": 2647.0, "airmass": [1.2], "pwv": [], "unstated_airmass": False, "source": "Rubin",
        "kind": None, "model": None, "model_version": None, "citation": None}}
