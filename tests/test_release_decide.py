"""Nill's decisions of 28 October for 0.5.0.

1. With NJM down, a product of NJM's that SVO holds only in part is installed,
   with one warning, and the call succeeds; the set is shown as incomplete.
2. SED.fetch with Teff, logg or metallicity bounds downloads only the spectra
   within them, and builds the selection's lookup table, HDF5 bundle and flux
   cube, as ``sed-tools grids --teff ...`` does.

Every source is mocked; nothing here touches the network.
"""

import hashlib
import logging
import math
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from sed_tools import cli
from sed_tools.api import SED, Filters, _parameter_cuts
from sed_tools.filter_components import ComponentError
from sed_tools.filter_library import FilterLibrary
from sed_tools.models import _read_flux_cube_header
from test_release_njmdown import GAIA, NJM_DOWN, bundled, svo_browser  # noqa: F401 (bundled is a fixture)


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    return cli.main()


# ----------------------------------------------------------------------
# 1. A partial set from SVO, NJM down: installed, warned about, a success
# ----------------------------------------------------------------------
@pytest.fixture
def gaia_from_svo(tmp_path, bundled, monkeypatch):
    """Every FilterLibrary sees NJM down and an SVO holding 4 of GAIA/GAIA's 6 bands; returns the filter root."""
    bundled({"GAIA": GAIA})
    root = tmp_path / "filters"
    browser = svo_browser(root, {("GAIA", "GAIA3"): ["G", "Gbp", "Grp", "Grvs"]})
    monkeypatch.setattr(FilterLibrary, "mirror", NJM_DOWN)
    monkeypatch.setattr(FilterLibrary, "svo", property(lambda self: browser))
    monkeypatch.setenv("COLUMNS", "80")
    return root


def incomplete_warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and "is incomplete" in r.getMessage()]


def test_filters_fetch_returns_the_partial_set_with_one_warning(gaia_from_svo, caplog):
    with caplog.at_level(logging.WARNING):
        directory = Filters.fetch("GAIA", "GAIA", filter_root=gaia_from_svo)
    assert directory == gaia_from_svo / "GAIA" / "GAIA"
    assert len(incomplete_warnings(caplog)) == 1  # one line for the set, not one per band
    assert "missing Gbp_bright, Gbp_faint" in incomplete_warnings(caplog)[0].getMessage()

    info = Filters.available("GAIA", "GAIA", filter_root=gaia_from_svo, include_remote=False)
    assert info["bands"] == ["G", "Gbp", "Grp", "Grvs"]
    assert info["expected_bands"] == GAIA["GAIA"]["bands"]  # how a script tells the set is incomplete
    assert info["missing_files"] == ["Gbp_bright.dat", "Gbp_faint.dat"]


def test_the_filters_command_exits_0_and_warns_once(gaia_from_svo, monkeypatch, capsys, caplog):
    with caplog.at_level(logging.WARNING):
        assert run_cli(monkeypatch, "filters", "GAIA/GAIA", "--base", str(gaia_from_svo)) == 0
    assert capsys.readouterr().out.rstrip().endswith(str(gaia_from_svo / "GAIA" / "GAIA"))
    [warning] = incomplete_warnings(caplog)
    assert "Fetch again when NJM is reachable to complete it" in warning.getMessage()


def test_filters_info_and_the_picker_summary_say_the_set_is_incomplete(gaia_from_svo, monkeypatch, capsys):
    Filters.fetch("GAIA", "GAIA", filter_root=gaia_from_svo)
    capsys.readouterr()
    assert run_cli(monkeypatch, "filters-info", "GAIA/GAIA", "--base", str(gaia_from_svo), "--offline") == 0
    out = " ".join(capsys.readouterr().out.split())  # wrapped at 80 columns
    assert ("Incomplete : 4 of 6 bands installed; missing Gbp_bright, Gbp_faint; fetch it again when NJM is "
            "reachable to complete it") in out

    info = Filters.available("GAIA", "GAIA", filter_root=gaia_from_svo, include_remote=False)
    lines = cli._pick_summary("GAIA/GAIA", {}, info, {"info": None, "properties": None, "errors": []}, "SVO")
    assert "  Incomplete : 4 of 6 bands installed; missing Gbp_bright, Gbp_faint" in lines
    assert all(len(line) <= 80 for line in lines)


def test_a_complete_set_has_no_incomplete_line(tmp_path, monkeypatch, capsys):
    from test_release_library import write_set

    root = tmp_path / "filters"
    write_set(root / "Fac" / "Cam", ["B", "V"])
    assert run_cli(monkeypatch, "filters-info", "Fac/Cam", "--base", str(root), "--offline") == 0
    assert "Incomplete" not in capsys.readouterr().out


def test_an_svo_curve_that_could_not_be_saved_still_fails(gaia_from_svo, monkeypatch):
    def refuse(self, row, out_path):
        if row.band == "Grvs":
            raise ValueError("SVO returned no transmission data")
        original(self, row, out_path)

    from sed_tools.svo_filter_grabber import SVOFilterBrowser

    original = SVOFilterBrowser._save_filter
    monkeypatch.setattr(SVOFilterBrowser, "_save_filter", refuse)
    with pytest.raises(ComponentError) as raised:
        Filters.fetch("GAIA", "GAIA", filter_root=gaia_from_svo)
    message = str(raised.value)
    assert "GAIA/GAIA is only partly installed" in message and "3 of NJM's 6 bands came from SVO's GAIA/GAIA3" in message
    assert "GAIA/GAIA3.Grvs: SVO returned no transmission data" in message


# ----------------------------------------------------------------------
# 2. SED.fetch with bounds downloads only the spectra within them
# ----------------------------------------------------------------------
GRID = "Kurucz2003all"
NJM_URL = "https://nillmill.ddns.net/sed_tools"
#: The golden Kurucz test grid: Teff 5750, 6000, 6250; logg 4.0, 4.5; [M/H] -0.5, 0.0.
KURUCZ = Path(__file__).parent / "golden" / "inputs" / "stellar_models" / GRID
#: Teff 5900-6100 selects the four spectra at 6000 K.
AT_6000 = ["Kurucz2003all_fid16526.txt", "Kurucz2003all_fid16527.txt",
           "Kurucz2003all_fid18906.txt", "Kurucz2003all_fid18907.txt"]


class Reply:
    def __init__(self, status, body=b""):
        self.status_code, self.content, self.headers = status, body, {}
        self.text = body.decode("utf-8", errors="replace")

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        import json
        return json.loads(self.text)

    def iter_content(self, chunk_size=1):
        yield self.content

    def close(self):
        pass


class FakeNJM:
    """A stand-in for NJM's web server: ``root`` holds index.json and stellar_models/<grid>/ with its files."""

    def __init__(self, root):
        self.root, self.requested, self.headers, self.verify = Path(root), [], {}, True

    def __call__(self):  # requests.Session()
        return self

    def get(self, url, **kwargs):
        rel = url.split("?")[0][len(NJM_URL) + 1:]
        self.requested.append(rel)
        path = self.root / rel
        if rel.endswith("/") and path.is_dir():
            links = "".join(f'<a href="{p.name}">{p.name}</a>\n' for p in sorted(path.iterdir()))
            return Reply(200, f"<html><body>{links}</body></html>".encode())
        return Reply(200, path.read_bytes()) if path.is_file() else Reply(404)

    def head(self, url, **kwargs):
        return Reply(200 if (self.root / url[len(NJM_URL) + 1:]).exists() else 404)

    def spectra_requested(self):
        return [Path(rel).name for rel in self.requested if rel.endswith(".txt")]


@pytest.fixture(scope="module")
def njm_grid(tmp_path_factory):
    """NJM's copy of the Kurucz test grid, with the whole grid's flux cube and HDF5 bundle beside it."""
    from sed_tools.api import _build_flux_cube
    from sed_tools.spectrum_io import build_h5_bundle

    root = tmp_path_factory.mktemp("njm")
    served = root / "stellar_models" / GRID
    shutil.copytree(KURUCZ, served)
    build_h5_bundle(served, served / f"{GRID}.h5")
    _build_flux_cube(served)
    (root / "index.json").write_text('{"models": ["%s"]}' % GRID)
    return root


@pytest.fixture
def njm(njm_grid, monkeypatch):
    server = FakeNJM(njm_grid)
    monkeypatch.setattr("sed_tools.njm_spectra_grabber.requests.Session", server)
    return server


def digests(directory):
    """Each file's bytes and modification time."""
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(Path(directory).iterdir()) if p.is_file()}


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
    err = capsys.readouterr().err
    assert "flux_cube.bin was built before some of the spectra beside it" in " ".join(err.split())
    with h5py.File(folder / f"{GRID}.h5", "r") as bundle:  # rebuilt although --no-h5: it lacked the rest
        assert len(bundle["spectra"]) == 12

    assert run_cli(monkeypatch, "grids", "--grids", GRID, "--source", "njm", "--base", base) == 0
    assert list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"]) == [5750.0, 6000.0, 6250.0]
