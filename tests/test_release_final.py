"""The last fixes before 0.5.0, after the re-check of 3c7a416.

1. Whether a grid's flux cube and HDF5 bundle are out of date is read from
   their contents, not from file times: a grid copied without its times is
   complete, and a selection followed by the rest is still rebuilt.
2. A cube whose empty grid points later get spectra is rebuilt with them.
3. Messages: the rebuild hint names --base, the incomplete-set warning and
   the densifier fit 80 columns, ``[fetch]`` counts what it downloaded,
   ``ml_generator --auto`` refuses in one line; bounds no spectrum meets leave
   no folder; ``filters-info`` describes curves installed under other band
   names; a set NJM completes gets a manifest describing every band.

Every source is mocked; nothing here touches the network.
"""

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from sed_tools import cli
from sed_tools.api import SED, _build_flux_cube, _bundle_shortfall, _flux_cube_shortfall
from sed_tools.filter_library import FilterLibrary
from sed_tools.models import _read_flux_cube_header
from sed_tools.precompute_flux_cube import spectra_not_in_cube
from sed_tools.svo_regen_spectra_lookup import regenerate_lookup_table
from test_release_decide import GRID, KURUCZ, digests, njm, njm_grid, run_cli  # noqa: F401 (fixtures)
from test_release_library import CURVE, Served, njm_index
from test_release_njmdown import GAIA, NJM_DOWN, bundled, svo_browser  # noqa: F401 (bundled is a fixture)
from test_release_njmdown import njm as njm_filters  # noqa: F401 (a fixture)

ALPHA_04 = KURUCZ.parent / f"{GRID}__alpha_04"


def copy_without_times(src, dst):
    """Copy a folder as ``cp -R`` or ``scp`` without ``-p`` do, then make every spectrum newer than the
    cube, bundle and lookup table, as a copy in the wrong order leaves them."""
    shutil.copytree(src, dst, copy_function=shutil.copyfile, symlinks=True)
    now = time.time()
    for path in Path(dst).rglob("*"):
        if path.is_file() and not path.is_symlink():
            stamp = now + 100 if path.suffix == ".txt" else now
            os.utime(path, (stamp, stamp))
    return Path(dst)


def teff_axis(folder):
    return list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"])


# ----------------------------------------------------------------------
# 1. Out of date by content, not by file times
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


def variants_grid(folder):
    """The test grid at [alpha/M] 0 and 0.4 in one folder: built as one flux cube per variant."""
    folder.mkdir(parents=True)
    for source in (KURUCZ, ALPHA_04):
        for spectrum in source.glob("*.txt"):
            shutil.copy2(spectrum, folder / spectrum.name)
    regenerate_lookup_table(str(folder))
    assert _build_flux_cube(folder).name == "variants_index.csv"
    return folder


def test_a_variants_grid_is_compared_variant_by_variant(tmp_path):
    folder = variants_grid(tmp_path / "a" / GRID)
    assert _flux_cube_shortfall(folder) is None
    assert _flux_cube_shortfall(copy_without_times(folder, tmp_path / "b" / GRID)) is None

    # One more spectrum, at a Teff no variant's cube has
    text = (KURUCZ / "Kurucz2003all_fid16515.txt").read_text().replace("# teff = 5750.0", "# teff = 6500.0")
    (folder / "Kurucz2003all_fid99999.txt").write_text(text)
    regenerate_lookup_table(str(folder))
    assert _flux_cube_shortfall(folder) == "does not include 1 of the 25 spectra beside it"


def test_a_spectrum_zero_at_every_sampled_wavelength_is_still_found(tmp_path, cube_writer):
    """A spectrum covering only part of the cube's wavelengths is zero elsewhere (resample_to_grid)."""
    wl = np.linspace(1000.0, 20000.0, 20)  # the sampled planes are 2, 4, 6, 8, 11, 13, 15, 17
    flux = np.zeros((3, 1, 1, 20))
    flux[0, 0, 0] = 1.0                    # covers every wavelength
    flux[1, 0, 0, [0, 19]] = 1.0           # only at wavelengths no sampled plane has
    cube_writer(tmp_path / "flux_cube.bin", [5000.0, 6000.0, 7000.0], [4.0], [0.0], wl, flux)
    (tmp_path / "lookup_table.csv").write_text(
        "#file_name,teff,logg,metallicity\na.txt,5000,4.0,0.0\nb.txt,6000,4.0,0.0\nc.txt,7000,4.0,0.0\n")
    names, missing = spectra_not_in_cube(tmp_path / "flux_cube.bin", tmp_path / "lookup_table.csv")
    assert names == ["a.txt", "b.txt", "c.txt"] and missing == ["c.txt"]  # 7000 K: zero throughout


def test_an_unreadable_cube_is_out_of_date(njm, tmp_path):
    SED.fetch(GRID, source="njm", model_root=tmp_path)
    folder = tmp_path / GRID
    cube = folder / "flux_cube.bin"
    cube.write_bytes(cube.read_bytes()[:1000])  # truncated
    assert _flux_cube_shortfall(folder).startswith("cannot be compared with the spectra beside it")


# ----------------------------------------------------------------------
# 2. Holes in a cube: zero, then filled when their spectra arrive
# ----------------------------------------------------------------------
def test_disjoint_selections_leave_empty_points_and_a_later_fill_is_rebuilt(njm, tmp_path, capsys):
    folder = tmp_path / GRID
    SED.fetch(GRID, source="njm", model_root=tmp_path, teff_max=5800, logg_max=4.2)
    SED.fetch(GRID, source="njm", model_root=tmp_path, teff_min=6200, logg_min=4.3)
    # The empty points are zero, the others hold their spectra
    nw = len(_read_flux_cube_header(folder / "flux_cube.bin")["wavelengths"])
    cube = np.fromfile(folder / "flux_cube.bin", dtype=np.float64, offset=16 + 8 * (2 + 2 + 2 + nw))
    flux = cube.reshape(nw, 2, 2, 2)  # (wavelength, [M/H], logg, Teff)
    assert not flux[:, :, 1, 0].any() and not flux[:, :, 0, 1].any()
    assert flux[:, :, 0, 0].any(axis=0).all() and flux[:, :, 1, 1].any(axis=0).all()

    # Spectra on exactly those empty points: every one is on the cube's axes, yet the cube lacks them
    capsys.readouterr()
    SED.fetch(GRID, source="njm", model_root=tmp_path, teff_max=5800, logg_min=4.3)
    assert "[fetch] flux_cube.bin does not include 2 of the 6 spectra beside it:" in capsys.readouterr().out
    assert spectra_not_in_cube(folder / "flux_cube.bin", folder / "lookup_table.csv")[1] == []


# ----------------------------------------------------------------------
# 3. Messages and folders
# ----------------------------------------------------------------------
def test_the_rebuild_hint_names_base_only_when_the_grid_is_elsewhere(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "STELLAR_DIR_DEFAULT", tmp_path / "default")
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


def test_the_densifier_line_fits_80_columns(tmp_path, cube_writer, capsys):
    from sed_tools.grid_densifier import densify_grid

    wl = np.linspace(3000.0, 9000.0, 7)
    cube_writer(tmp_path / "src.bin", [4000.0, 8000.0], [4.0], [0.0], wl, np.ones((2, 1, 1, 7)))
    densify_grid(str(tmp_path / "src.bin"), str(tmp_path / "dense.bin"), teff_spacing=1000.0)
    lines = capsys.readouterr().out.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("Generating"))
    assert lines[at:at + 2] == ["Generating 3 synthetic Teff points (0 by ML, 3 blackbody)",
                                "  in 1 wavelength block(s)..."]


def test_filters_info_describes_curves_installed_under_other_band_names(tmp_path):
    """0.4.1's PLANCK/HFI from SVO (100GHz ...) is what NJM keeps as 100_svo ...: its records describe it."""
    from types import SimpleNamespace

    from sed_tools.filter_catalogue import FilterCatalogue

    installed = tmp_path / "filters" / "PLANCK" / "HFI"
    installed.mkdir(parents=True)
    for band in ("100GHz", "857GHz"):
        (installed / f"{band}.dat").write_text(f"Wavelength,Transmission,FilterID=PLANCK/HFI.{band}\n1,1\n2,1\n")
    (installed / "HFI").write_text("100GHz.dat\n857GHz.dat\n")
    info = {"bands": {"100": {"Description": "HFI 100", "curve": {"FilterID": "Planck/HFI.100"}},
                      "100_svo": {"Description": "PLANCK HFI 100GH", "curve": {"FilterID": "PLANCK/HFI.100GHz"}},
                      "857": {"Description": "HFI 857", "curve": {"FilterID": "Planck/HFI.857"}},
                      "857_svo": {"Description": "PLANCK HFI 857GHz", "curve": {"FilterID": "PLANCK/HFI.857GHz"}}}}
    properties = {"bands": {"100": {"WavelengthPivot": 30013450.0}, "100_svo": {"WavelengthPivot": 30035700.0},
                            "857": {"WavelengthPivot": 3525232.0}, "857_svo": {"WavelengthPivot": 3527564.0}}}
    server = Served({"index.json": njm_index({"Planck": {"HFI": {
                         "filters": ["100", "100_svo", "857", "857_svo"], "properties": "properties.json",
                         "info": "info.json"}}}),
                     "filters/Planck/HFI/HFI": "100.dat\n857.dat\n",
                     "filters/Planck/HFI/info.json": json.dumps(info),
                     "filters/Planck/HFI/properties.json": json.dumps(properties)})
    mirror = SimpleNamespace(base_url="https://nillmill.ddns.net/sed_tools", is_available=lambda: True,
                             _get=lambda url, **kwargs: server.get(url, **kwargs))
    catalogue = FilterCatalogue(tmp_path / "filters", mirror=mirror)

    described = catalogue.info("PLANCK", "HFI")
    assert {band: record["Description"] for band, record in described["bands"].items()} == {
        "100GHz": "PLANCK HFI 100GH", "857GHz": "PLANCK HFI 857GHz"}
    assert sorted(described["archived_bands"]) == ["100", "857"]  # NJM's own curves, not installed
    pivots = catalogue.properties("PLANCK", "HFI")["bands"]
    assert {band: record["WavelengthPivot"] for band, record in pivots.items()} == {
        "100GHz": 30035700.0, "857GHz": 3527564.0}


def test_a_set_njm_completes_gets_the_manifest_of_a_fresh_download(tmp_path, bundled, njm_filters):
    bundled({"GAIA": GAIA})
    bands = GAIA["GAIA"]["bands"]
    curve = CURVE.replace("original\n", "original;Components=filter+instrument;ComponentsSource=original\n")
    files = {"index.json": njm_index({"GAIA": {"GAIA": {"filters": bands, "size_bytes": 1}}}),
             "filters/GAIA/GAIA/GAIA": "".join(f"{b}.dat\n" for b in bands)}
    files.update({f"filters/GAIA/GAIA/{b}.dat": curve for b in bands})

    root = tmp_path / "filters"
    library = FilterLibrary(root)
    library.mirror = NJM_DOWN
    library.svo = svo_browser(root, {("GAIA", "GAIA3"): ["G", "Gbp", "Grp", "Grvs"]})
    library.published("GAIA", "GAIA")  # 4 of 6 bands, with a warning
    library = FilterLibrary(root)
    library.mirror = njm_filters(root, files)
    library.published("GAIA", "GAIA")

    fresh = FilterLibrary(tmp_path / "fresh")
    fresh.mirror = njm_filters(tmp_path / "fresh", files)
    fresh.published("GAIA", "GAIA")
    completed = json.loads((root / "GAIA" / "GAIA" / "manifest.json").read_text())
    assert sorted(completed["detector_types"]) == sorted(completed["published_bands"]) == bands
    assert completed == json.loads((tmp_path / "fresh" / "GAIA" / "GAIA" / "manifest.json").read_text())
