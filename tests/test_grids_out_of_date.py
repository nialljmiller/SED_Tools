"""Whether a grid's flux cube and HDF5 bundle are out of date is read from their contents,
not from file times: a grid copied without its times is complete, a selection followed by the rest is
rebuilt, and a cube whose empty grid points later get spectra is rebuilt with them.

Every source is mocked; nothing here touches the network.
"""

import os
import shutil
import time

import numpy as np

from sed_tools import cli
from sed_tools.api import SED, _bundle_shortfall, _flux_cube_shortfall
from sed_tools.models import _read_flux_cube_header
from sed_tools.precompute_flux_cube import spectra_not_in_cube

from helpers_grids import GRID, KURUCZ, copy_without_times, digests, run_cli
from helpers_grids import njm, njm_grid  # noqa: F401 (fixtures)


def teff_axis(folder):
    return list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"])


# ----------------------------------------------------------------------
# Out of date by content, not by file times
# ----------------------------------------------------------------------
def test_a_whole_grid_copied_without_its_times_is_cut_as_it_stands(njm, tmp_path, capsys):
    SED.fetch(GRID, source="njm", model_root=tmp_path / "a")
    folder = copy_without_times(tmp_path / "a" / GRID, tmp_path / "b" / GRID)
    assert _flux_cube_shortfall(folder) is None and _bundle_shortfall(folder / f"{GRID}.h5", folder) is None
    before = digests(folder)
    njm.requested.clear()
    capsys.readouterr()

    sed = SED.fetch(GRID, source="njm", model_root=tmp_path / "b", teff_min=5900, teff_max=6100)

    assert digests(folder) == before  # no lookup table, bundle or cube rewritten
    assert njm.spectra_requested() == [] and len(sed.grid) == 4
    out = capsys.readouterr().out
    assert "[fetch] 4 already present, 0 downloaded" in out
    assert "nothing was downloaded or rewritten" in " ".join(out.split())


def test_grids_no_cube_on_a_grid_copied_without_its_times_exits_0(njm, tmp_path, monkeypatch, capsys):
    SED.fetch(GRID, source="njm", model_root=tmp_path / "a")
    folder = copy_without_times(tmp_path / "a" / GRID, tmp_path / "b" / GRID)
    cube = (folder / "flux_cube.bin").read_bytes()

    # The re-check's command, which 0.4.1 runs with exit 0 on such a copy
    assert run_cli(monkeypatch, "grids", "--source", "njm", "--grids", GRID, "--no-cube",
                   "--base", str(tmp_path / "b")) == 0
    assert (folder / "flux_cube.bin").read_bytes() == cube
    assert run_cli(monkeypatch, "grids", "--source", "njm", "--grids", GRID, "--no-cube", "--no-h5",
                   "--base", str(tmp_path / "b")) == 0
    assert "HDF5 bundle   : Kurucz2003all.h5 (exists)" in capsys.readouterr().out


def test_a_selection_then_the_rest_is_rebuilt_whatever_the_file_times(njm, tmp_path, capsys):
    SED.fetch(GRID, source="njm", model_root=tmp_path, teff_min=5900, teff_max=6100)
    folder = tmp_path / GRID
    future = time.time() + 3600  # the cube now looks newer than anything downloaded later
    for name in ("flux_cube.bin", f"{GRID}.h5"):
        os.utime(folder / name, (future, future))
    capsys.readouterr()

    sed = SED.fetch(GRID, source="njm", model_root=tmp_path)

    assert len(sed.grid) == 12 and teff_axis(folder) == [5750.0, 6000.0, 6250.0]
    assert spectra_not_in_cube(folder / "flux_cube.bin", folder / "lookup_table.csv")[1] == []
    out = capsys.readouterr().out
    assert "[fetch] 4 already present, 8 downloaded" in out
    assert "[fetch] flux_cube.bin does not include 8 of the 12 spectra beside it:" in out


def test_a_whole_grid_and_a_selection_alone_are_complete(njm, tmp_path):
    SED.fetch(GRID, source="njm", model_root=tmp_path / "whole")
    SED.fetch(GRID, source="njm", model_root=tmp_path / "cut", teff_min=5900, teff_max=6100)
    for folder in (tmp_path / "whole" / GRID, tmp_path / "cut" / GRID):
        assert _flux_cube_shortfall(folder) is None
        assert _bundle_shortfall(folder / f"{GRID}.h5", folder) is None


def test_a_bundle_that_lacks_spectra_says_how_many(njm, tmp_path):
    SED.fetch(GRID, source="njm", model_root=tmp_path, teff_min=5900, teff_max=6100)
    folder = tmp_path / GRID
    shutil.copyfile(KURUCZ / "Kurucz2003all_fid16515.txt", folder / "Kurucz2003all_fid16515.txt")
    assert _bundle_shortfall(folder / f"{GRID}.h5", folder) == "holds 4 of the 5 spectra beside it"


def test_an_unreadable_cube_is_out_of_date(njm, tmp_path):
    SED.fetch(GRID, source="njm", model_root=tmp_path)
    folder = tmp_path / GRID
    cube = folder / "flux_cube.bin"
    cube.write_bytes(cube.read_bytes()[:1000])  # truncated
    assert _flux_cube_shortfall(folder).startswith("cannot be compared with the spectra beside it")


# ----------------------------------------------------------------------
# Holes in a cube: zero, then filled when their spectra arrive
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Messages and folders
# ----------------------------------------------------------------------
def test_the_rebuild_hint_names_base_only_when_the_grid_is_elsewhere(monkeypatch, tmp_path):
    monkeypatch.setattr("sed_tools.grids.pipeline.STELLAR_DIR_DEFAULT", tmp_path / "default")
    assert cli._rebuild_command(GRID, str(tmp_path / "default" / GRID)) == f"sed-tools rebuild --grids {GRID}"
    elsewhere = tmp_path / "my grids"
    assert (cli._rebuild_command(GRID, str(elsewhere / GRID))
            == f"sed-tools rebuild --grids {GRID} --base '{elsewhere}'")


def test_bounds_no_spectrum_meets_leave_no_folder(njm, tmp_path):
    sed = SED.fetch(GRID, source="njm", model_root=tmp_path, teff_min=9000)
    assert len(sed.grid) == 0 and not (tmp_path / GRID).exists()

    (tmp_path / GRID).mkdir()  # a folder the call did not make stays, empty or not
    SED.fetch(GRID, source="njm", model_root=tmp_path, teff_min=9000)
    assert (tmp_path / GRID).is_dir()
