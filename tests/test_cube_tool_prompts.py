"""The prompts of the flux cube tool (python -m sed_tools.flux_cube_tool), answered through ``input()``.

The searches look only under tmp_path: the working directory, the data directory and the hints are all there.
"""

import numpy as np
import pytest

from sed_tools import flux_cube_tool as tool
from sed_tools.photometry import PC_TO_M, RSUN_TO_M


def answer(monkeypatch, *replies):
    """Feed ``input()`` these replies in order; a further prompt fails the test."""
    queue = list(replies)

    def fake_input(prompt=""):
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


@pytest.fixture(autouse=True)
def searched_only_here(tmp_path, monkeypatch):
    """The cube and Vega searches see an empty working directory and data directory, and no environment hints."""
    import matplotlib

    matplotlib.use("Agg")
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    monkeypatch.setattr("sed_tools.cube.model.DATA_DIR_DEFAULT", tmp_path / "data")
    for variable in ("SED_FLUX_CUBE", "SED_FLUX_CUBE_DIR", "SED_VEGA_SPECTRUM"):
        monkeypatch.delenv(variable, raising=False)


WL = np.linspace(3000.0, 9000.0, 61)


@pytest.fixture
def grid(tmp_path, cube_writer):
    """grids/g/flux_cube.bin (flat spectra at two Teff and two log g), with filters B and V in grids/filters/Sys."""
    flux = np.ones((2, 2, 1, WL.size)) * np.array([1.0, 2.0])[:, None, None, None]
    cube_writer(tmp_path / "grids" / "g" / "flux_cube.bin", [5000.0, 6000.0], [4.0, 4.5], [0.0], WL, flux)
    system = tmp_path / "grids" / "filters" / "Sys"
    system.mkdir(parents=True)
    for band, (lo, hi) in {"B": (4000, 5000), "V": (5000, 6000)}.items():
        (system / f"{band}.dat").write_text(f"Wavelength,Transmission\n{lo - 100},0\n{lo},1\n{hi},1\n{hi + 100},0\n")
    return tmp_path / "grids" / "g"


def write_vega(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([WL, np.full(WL.size, 3.0)]))
    return path


# ----------------------------------------------------------------------
# Distance, radius and saving
# ----------------------------------------------------------------------
def test_distance_and_radius_take_a_unit_or_the_default_one(monkeypatch):
    answer(monkeypatch, "10 pc", "3.1e17 m", "2", "1 Rsun", "7e8 m", "2 R_sun")
    assert [tool.prompt_distance_m() for _ in range(3)] == pytest.approx([10 * PC_TO_M, 3.1e17, 2 * PC_TO_M])
    assert [tool.prompt_radius_m() for _ in range(3)] == pytest.approx([RSUN_TO_M, 7e8, 2 * RSUN_TO_M])


def test_prompt_length_scales_only_its_own_units(monkeypatch):
    answer(monkeypatch, "4 au", "4 AU")
    assert tool._prompt_length("? ", "m", ("au",), 1.5e11) == pytest.approx(6e11)  # the unit is case-blind
    answer(monkeypatch, "4 m")
    assert tool._prompt_length("? ", "au", ("au",), 1.5e11) == 4.0


def test_save_sed_writes_two_columns_and_makes_its_folder(tmp_path):
    path = tmp_path / "out" / "sed.txt"
    tool.save_sed(str(path), WL, 2 * WL)
    assert path.read_text().splitlines()[0] == "# wavelength flux"
    assert np.loadtxt(path).tolist() == np.column_stack([WL, 2 * WL]).tolist()


def test_infer_filter_directories_looks_beside_and_above_the_cube(grid, tmp_path):
    (grid / "filters").mkdir()
    found = tool.infer_filter_directories(str(grid / "flux_cube.bin"))
    assert list(dict.fromkeys(path for path in found if tmp_path in path.parents)) == [
        (grid / "filters").resolve(), (tmp_path / "grids" / "filters").resolve()]


# ----------------------------------------------------------------------
# Which flux cube
# ----------------------------------------------------------------------
def test_one_flux_cube_found_is_used_without_asking(grid, capsys):
    assert tool.prompt_for_flux_cube_path([grid]) == (grid / "flux_cube.bin").resolve()
    assert capsys.readouterr().out == f"Using flux cube: {(grid / 'flux_cube.bin').resolve()}\n"


def test_several_flux_cubes_are_offered_by_number(grid, tmp_path, monkeypatch, capsys):
    other = tmp_path / "grids" / "h" / "flux_cube.bin"
    other.parent.mkdir()
    other.write_bytes((grid / "flux_cube.bin").read_bytes())
    answer(monkeypatch, "", "3", "2")  # Enter lists them again; 3 is not listed
    assert tool.prompt_for_flux_cube_path([tmp_path / "grids"]) == other.resolve()
    out = capsys.readouterr().out
    assert out.count("Discovered flux cube files:") == 3
    assert "Invalid selection. Please choose one of the listed numbers." in out
    answer(monkeypatch, str(grid))  # or a path: a folder holding one cube
    assert tool.prompt_for_flux_cube_path([tmp_path / "grids"]) == grid / "flux_cube.bin"


def test_with_no_flux_cube_found_a_path_is_asked_for_until_quit(grid, monkeypatch, capsys):
    answer(monkeypatch, "", "missing", str(grid / "flux_cube.bin"))
    assert tool.prompt_for_flux_cube_path() == grid / "flux_cube.bin"
    out = capsys.readouterr().out
    assert "Please provide a path to a flux cube file." in out
    assert "Flux cube path 'missing' does not exist" in out
    answer(monkeypatch, "q")
    with pytest.raises(SystemExit, match="No flux cube selected; aborting."):
        tool.prompt_for_flux_cube_path()


# ----------------------------------------------------------------------
# Which Vega spectrum
# ----------------------------------------------------------------------
def test_a_vega_spectrum_given_or_hinted_is_used(grid, tmp_path, monkeypatch):
    cube = grid / "flux_cube.bin"
    vega = write_vega(tmp_path / "ref" / "my_vega.dat")
    assert tool.resolve_vega_spectrum_path(str(vega), cube) == vega.resolve()
    with pytest.raises(FileNotFoundError, match="Vega spectrum 'nope.dat' does not exist"):
        tool.resolve_vega_spectrum_path("nope.dat", cube)
    assert tool.resolve_vega_spectrum_path(None, cube, [vega]) == vega.resolve()
    assert tool.resolve_vega_spectrum_path(None, cube, [vega.parent]) == vega.resolve()  # a folder is searched
    band = vega.parent / "B.dat"
    band.write_text("4000 1\n")
    assert tool.resolve_vega_spectrum_path(None, cube, [band]) == vega.resolve()  # and a filter file's folder
    monkeypatch.setenv("SED_VEGA_SPECTRUM", str(vega))
    assert tool.resolve_vega_spectrum_path(None, cube) == vega.resolve()


def test_vega_spectra_near_the_cube_are_found_and_offered_by_number(grid, monkeypatch, capsys):
    cube = grid / "flux_cube.bin"
    with pytest.raises(FileNotFoundError, match="Could not locate a Vega reference spectrum."):
        tool.resolve_vega_spectrum_path(None, cube)
    first = write_vega(grid / "vega.dat")
    assert tool.resolve_vega_spectrum_path(None, cube) == first.resolve()
    second = write_vega(grid.parent / "alpha_lyr_vega.txt")
    answer(monkeypatch, "x", "3", "2")  # listed nearest first
    assert tool.resolve_vega_spectrum_path(None, cube) == second.resolve()
    out = capsys.readouterr().out
    assert "Discovered potential Vega spectra:" in out
    assert out.count("Please select one of the listed Vega spectra.") == 2


# ----------------------------------------------------------------------
# The whole tool
# ----------------------------------------------------------------------
def test_tool_scales_and_measures_vega_magnitudes_with_the_cube_it_finds(grid, tmp_path, monkeypatch, capsys):
    write_vega(grid / "vega.dat")
    answer(monkeypatch, "y", "10 pc", "1 Rsun")  # scale for distance; the cube, filters and Vega are found
    status = tool.main(["--model-dir", str(grid), "--teff", "5000", "--logg", "4.0", "--metallicity", "0",
                        "--photometric-system", "Vega", "--plot", str(tmp_path / "sed.png"),
                        "--save-sed", str(tmp_path / "sed.txt")])
    assert status == 0
    out = capsys.readouterr().out
    scale = (RSUN_TO_M / (10 * PC_TO_M)) ** 2
    assert f"Using flux cube: {(grid / 'flux_cube.bin').resolve()}" in out
    assert f"[scaling] Applied geometric dilution: (R/d)^2 = {scale:.3e}" in out
    assert out.count("Using Vega spectrum:") == 1  # read once for both filters
    assert "Photometric system: Vega" in out and "(system: Sys)" in out
    mag = -2.5 * np.log10(scale / 3.0)  # a flat SED of flux 1, scaled, against a flat Vega of flux 3
    for band in ("Sys/B", "Sys/V"):
        line = next(line for line in out.splitlines() if line.strip().startswith(band))
        assert float(line.split("mag=")[1]) == pytest.approx(mag, abs=1e-3)
    assert np.loadtxt(tmp_path / "sed.txt")[:, 1] == pytest.approx(np.full(WL.size, scale))
    assert (tmp_path / "sed.png").is_file()


def test_tool_stops_when_vega_magnitudes_have_no_vega_spectrum(grid, tmp_path, monkeypatch):
    answer(monkeypatch, "n")
    with pytest.raises(SystemExit, match="Could not locate a Vega reference spectrum. Provide a Vega spectrum "
                                         "via --vega-sed or place one near the flux cube."):
        tool.main(["--flux-cube", str(grid / "flux_cube.bin"), "--teff", "5000", "--logg", "4.0",
                   "--metallicity", "0", "--photometric-system", "Vega", "--plot", str(tmp_path / "sed.png")])
