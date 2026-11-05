"""SED.fetch and SED.query with Teff, logg or metallicity bounds: only the spectra within
them are downloaded, a bound of 0 counts, and the selection's lookup table, HDF5 bundle and flux cube
are built as ``sed-tools grids --teff ...`` does.

Every source is mocked; nothing here touches the network.
"""

import logging
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from sed_tools.api import SED, _parameter_cuts
from sed_tools.models import _read_flux_cube_header

from helpers_grids import GRID, digests, run_cli
from helpers_grids import njm, njm_grid, no_network  # noqa: F401 (fixtures)


# ----------------------------------------------------------------------
# SED.fetch with bounds downloads only the spectra within them
# ----------------------------------------------------------------------
#: Teff 5900-6100 selects the four spectra at 6000 K.
AT_6000 = ["Kurucz2003all_fid16526.txt", "Kurucz2003all_fid16527.txt",
           "Kurucz2003all_fid18906.txt", "Kurucz2003all_fid18907.txt"]


def same_grid(a, b):
    """Whether two SpectralGrids hold the same spectra, in the same order, with the same values."""
    assert (a.name, a.source, len(a)) == (b.name, b.source, len(b))
    for x, y in zip(a, b):
        assert (x.filename, x.teff, x.logg, x.metallicity) == (y.filename, y.teff, y.logg, y.metallicity)
        assert np.array_equal(x.wavelength, y.wavelength) and np.array_equal(x.flux, y.flux)
    return True


def test_bounds_become_the_grabbers_ranges_and_a_bound_of_0_counts():
    assert _parameter_cuts(None, None, None, None, None, None) == {}
    assert _parameter_cuts(0, None, None, 4.5, None, 0.0) == {"teff_range": (0.0, math.inf),
                                                             "logg_range": (-math.inf, 4.5),
                                                             "meta_range": (-math.inf, 0.0)}


def test_a_selection_returns_what_the_whole_download_did_and_downloads_only_it(njm, tmp_path):
    whole = SED.fetch(GRID, source="njm", model_root=tmp_path / "whole").grid.filter(teff_min=5900, teff_max=6100)
    njm.requested.clear()

    sed = SED.fetch(GRID, source="njm", model_root=tmp_path / "cut", teff_min=5900, teff_max=6100)

    assert [s.filename for s in sed.grid] == AT_6000 and same_grid(sed.grid, whole)
    assert sorted(njm.spectra_requested()) == AT_6000
    assert not [rel for rel in njm.requested if rel.endswith((".bin", ".h5"))]  # NJM's whole-grid products
    folder = tmp_path / "cut" / GRID
    assert sorted(p.name for p in folder.glob("*.txt")) == AT_6000
    for name in AT_6000:  # byte for byte the copies a whole download makes
        assert (folder / name).read_bytes() == (tmp_path / "whole" / GRID / name).read_bytes()
    lookup = (folder / "lookup_table.csv").read_text().splitlines()
    assert sorted(row.split(",")[0] for row in lookup[1:]) == AT_6000
    with h5py.File(folder / f"{GRID}.h5", "r") as bundle:
        assert sorted(bundle["spectra"]) == AT_6000
    assert list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"]) == [6000.0]


def test_sed_local_then_uses_the_selection(njm, tmp_path):
    fetched = SED.fetch(GRID, source="njm", model_root=tmp_path, teff_min=5900, teff_max=6100).grid
    local = SED.local(GRID, model_root=tmp_path)
    assert [s.filename for s in local.grid] == AT_6000
    assert [(s.teff, s.logg, s.metallicity) for s in local.grid] == [(s.teff, s.logg, s.metallicity) for s in fetched]
    assert local.parameter_ranges()["teff"] == (6000.0, 6000.0)
    assert np.all(np.isfinite(local(6000, 4.25, -0.25).flux))


def test_a_later_fetch_completes_the_grid_and_rebuilds_from_every_spectrum(njm, tmp_path):
    SED.fetch(GRID, source="njm", model_root=tmp_path, teff_min=5900, teff_max=6100)
    folder = tmp_path / GRID

    wider = SED.fetch(GRID, source="njm", model_root=tmp_path, teff_min=5700, teff_max=6100)
    assert len(wider.grid) == 8 and len(list(folder.glob("*.txt"))) == 8
    assert list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"]) == [5750.0, 6000.0]

    njm.requested.clear()
    sed = SED.fetch(GRID, source="njm", model_root=tmp_path)
    assert len(sed.grid) == 12 and len(njm.spectra_requested()) == 4  # only the rest
    assert list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"]) == [5750.0, 6000.0, 6250.0]
    with h5py.File(folder / f"{GRID}.h5", "r") as bundle:
        assert len(bundle["spectra"]) == 12
    assert len(SED.local(GRID, model_root=tmp_path).grid) == 12


def test_a_whole_grid_already_installed_is_cut_as_it_stands(njm, tmp_path):
    whole = SED.fetch(GRID, source="njm", model_root=tmp_path).grid
    folder = tmp_path / GRID
    before = digests(folder)
    njm.requested.clear()

    sed = SED.fetch(GRID, source="njm", model_root=tmp_path, teff_min=5900, teff_max=6100)

    assert digests(folder) == before  # no file, lookup table or cube rewritten or removed
    assert njm.spectra_requested() == [] and not [r for r in njm.requested if r.endswith((".bin", ".h5"))]
    assert same_grid(sed.grid, whole.filter(teff_min=5900, teff_max=6100))


def test_bounds_no_spectrum_meets_return_an_empty_grid_and_download_nothing(njm, tmp_path):
    sed = SED.fetch(GRID, source="njm", model_root=tmp_path, teff_min=9000)
    assert len(sed.grid) == 0 and sed.grid.source == "njm"
    assert njm.spectra_requested() == [] and not list((tmp_path / GRID).glob("*"))


def test_svo_cannot_select_so_the_whole_grid_is_downloaded_then_cut(tmp_path, monkeypatch, caplog):
    class SVO:
        def __init__(self, base_dir=None, max_workers=None):
            self.base_dir = Path(base_dir)

        def get_model_metadata(self, name):
            return [{"fid": n} for n in (1, 2, 3)]

        def download_model_spectra(self, name, spectra_info, **ranges):
            if any(r is not None for r in ranges.values()):
                raise ValueError("SVO downloads cannot be cut")  # as the real grabber
            for n, teff in zip((1, 2, 3), (5000, 6000, 7000)):
                path = self.base_dir / name / f"{name}_fid{n}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# teff = {teff}\n# logg = 4.0\n# metallicity = 0.0\n# units_standardized = True\n"
                                "1000 1.0\n2000 2.0\n")
            return 3

    monkeypatch.setattr("sed_tools.svo_spectra_grabber.SVOSpectraGrabber", SVO)
    with caplog.at_level(logging.WARNING, logger="sed_tools.api"):
        sed = SED.fetch("Grid", source="svo", model_root=tmp_path, teff_min=5500, teff_max=6500)
    assert [r.getMessage() for r in caplog.records] == [
        "[fetch] svo cannot select spectra by Teff, logg or metallicity, so the whole of Grid is downloaded, then cut"]
    assert [s.teff for s in sed.grid] == [6000.0]
    assert len(list((tmp_path / "Grid").glob("*.txt"))) == 3 and not (tmp_path / "Grid" / "flux_cube.bin").exists()


def test_grids_after_a_selection_rebuilds_or_says_why_it_did_not(njm, tmp_path, monkeypatch, capsys):
    base = str(tmp_path)
    assert run_cli(monkeypatch, "grids", "--grids", GRID, "--source", "njm", "--base", base,
                   "--teff", "5900", "6100") == 0
    folder = tmp_path / GRID
    assert list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"]) == [6000.0]

    # The rest, without a cube build: the selection's cube is not taken for the whole grid's.
    assert run_cli(monkeypatch, "grids", "--grids", GRID, "--source", "njm", "--base", base,
                   "--no-h5", "--no-cube") == 1
    err = " ".join(capsys.readouterr().err.split())
    assert "flux_cube.bin does not include 8 of the 12 spectra beside it" in err
    assert f"Rebuild it with: sed-tools rebuild --grids {GRID} --base {base}" in err
    with h5py.File(folder / f"{GRID}.h5", "r") as bundle:  # rebuilt although --no-h5: it lacked the rest
        assert len(bundle["spectra"]) == 12

    assert run_cli(monkeypatch, "grids", "--grids", GRID, "--source", "njm", "--base", base) == 0
    assert list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"]) == [5750.0, 6000.0, 6250.0]


# ---------------------------------------------------------------- 7. zero bounds
# logg and [M/H] ranges chosen so that a bound of 0 keeps some grids and drops others.
LOCAL_GRIDS = {
    "Negative": {"logg": [-1.0, -0.5], "meta": [-2.0, -1.0]},
    "Positive": {"logg": [0.5, 1.0], "meta": [0.25, 0.5]},
    "Straddle": {"logg": [-0.5, 0.5], "meta": [-0.5, 0.5]},
}


@pytest.fixture
def local_grids(tmp_path, monkeypatch, cube_writer):
    for name, axes in LOCAL_GRIDS.items():
        teff = [4000.0, 6000.0]
        flux = np.ones((len(teff), len(axes["logg"]), len(axes["meta"]), 2))
        cube_writer(tmp_path / name / "flux_cube.bin", teff, axes["logg"], axes["meta"], [4000.0, 5000.0], flux)
    monkeypatch.setattr(SED, "_model_root", tmp_path)
    return tmp_path


@pytest.mark.parametrize("bound, expected", [
    ({"teff_min": 0}, {"Negative", "Positive", "Straddle"}),
    ({"teff_max": 0}, set()),
    ({"logg_min": 0.0}, {"Positive", "Straddle"}),
    ({"logg_max": 0.0}, {"Negative", "Straddle"}),
    ({"metallicity_min": 0.0}, {"Positive", "Straddle"}),
    ({"metallicity_max": 0.0}, {"Negative", "Straddle"}),
    ({"metallicity_max": 0}, {"Negative", "Straddle"}),
    ({"logg_min": 0, "metallicity_max": 0.0}, {"Straddle"}),
])
def test_query_applies_a_bound_of_zero(local_grids, bound, expected):
    found = SED.query(source="local", include_remote=False, **bound)
    assert {info.name for info in found} == expected


@pytest.fixture
def downloaded_grid(tmp_path, monkeypatch):
    """A grid already on disk; SED.fetch's download step is replaced by a no-op."""
    grid = tmp_path / "Tiny"
    grid.mkdir()
    rows = ["file_name,teff,logg,metallicity"]
    for logg in (-0.5, 0.0, 0.5):
        for meta in (-0.5, 0.0, 0.5):
            name = f"g{logg:+.1f}_m{meta:+.1f}.txt"
            (grid / name).write_text("4000 1.0\n5000 2.0\n")
            rows.append(f"{name},5000,{logg},{meta}")
    (grid / "lookup_table.csv").write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(SED, "_download_from", classmethod(lambda cls, src, catalog, model_dir, workers, cuts: None))
    return tmp_path


@pytest.mark.parametrize("bound, keep", [
    ({"teff_min": 0}, lambda logg, meta: True),
    ({"teff_max": 0}, lambda logg, meta: False),
    ({"logg_min": 0.0}, lambda logg, meta: logg >= 0),
    ({"logg_max": 0.0}, lambda logg, meta: logg <= 0),
    ({"metallicity_min": 0.0}, lambda logg, meta: meta >= 0),
    ({"metallicity_max": 0.0}, lambda logg, meta: meta <= 0),
    ({"metallicity_max": 0}, lambda logg, meta: meta <= 0),
])
def test_fetch_applies_a_bound_of_zero(downloaded_grid, bound, keep):
    # From SVO, which cannot select spectra, the whole grid is downloaded and the bounds applied after.
    sed = SED.fetch("Tiny", source="svo", clean=False, model_root=downloaded_grid, **bound)

    expected = {(logg, meta) for logg in (-0.5, 0.0, 0.5) for meta in (-0.5, 0.0, 0.5) if keep(logg, meta)}
    assert {(s.logg, s.metallicity) for s in sed.grid} == expected
