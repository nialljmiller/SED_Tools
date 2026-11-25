"""Spectral grids: cleaning and flux units; the sources' downloads and failures (NJM, SVO, MSG, MAST); flux cubes
and bundles out of date by their contents; selections by Teff, logg and [M/H]; and the spectrum readers.

No row uses the network: every source gets a fake session.
"""

import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import requests

import sed_tools.config as config
from sed_tools import cli
from sed_tools._constants import SIGMA, planck_flam
from sed_tools.api import SED, _bundle_shortfall, _flux_cube_shortfall
from sed_tools.cube.build import spectra_not_in_cube
from sed_tools.cube.model import _read_flux_cube_header
from sed_tools.filters import njm as filters_njm
from sed_tools.grids import mast, msg, njm, pipeline, svo
from sed_tools.grids.clean import clean_model_dir, clean_spectrum_file, rebuild_lookup_table
from sed_tools.grids.lookup import regenerate_lookup_table
from sed_tools.grids.spectrum_io import read_text_spectrum
from sed_tools.grids.units import FLUX_UNITS, detect_units_from_header, validate_converted_spectrum

from support import (CURVE, GRID, KURUCZ, NJM, Resp, Session, _reset_sources, answer, copy_without_times, njm_grids,
                     run_cli, scenarios, write_flux_cube)

SCENARIOS, row = scenarios()


def _header(teff, logg=4.0, units="# wavelength_unit = Angstrom\n# flux_unit = erg/cm2/s/A\n"):
    return f"# teff = {teff}\n# logg = {logg}\n# metallicity = 0.0\n{units}"


def _write(path, header, wl, flux, row="{:.6f} {:.8e}\n"):
    """A text spectrum: the header lines, then wavelength and flux."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "".join(row.format(w, f) for w, f in zip(wl, flux)))


def _contents(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.txt"))}


# ----------------------------------------------------------------------
# Cleaning: coverage, wavelengths recovered from HDF5, clean_model_dir's failures, the bolometric check
# ----------------------------------------------------------------------
NM_FLAM = "# wavelength_unit = nm\n# flux_unit = erg/s/cm2/A\n"


def _spectrum(path, teff=5000.0, units=NM_FLAM, wl=np.linspace(400.0, 700.0, 30)):
    _write(path, "# source = testgrid\n" + _header(teff, units=units), wl, np.full(len(wl), 1e-14))


def _is_standardized(path):
    return "units_standardized = True" in path.read_text()


@row
def terminal_coverage_reports_missing_representative_spectra(t):
    """A representative spectrum missing from disk used to be dropped silently."""
    from sed_tools.grids.coverage import grid_coverage

    (t.tmp / "cool.txt").write_text("# teff = 4000\n# logg = 3.0\n1000 1.0\n2000 2.0\n")
    (t.tmp / "lookup_table.csv").write_text(
        "teff,logg,metallicity,filename\n4000,3.0,0.0,cool.txt\n8000,5.0,0.0,gone.txt\n")
    t.mp.setattr(config, "get_ui_setting", lambda name: "always" if name == "terminal_plots" else "never")
    grid_coverage(t.tmp, plot=False)
    output = t.capsys.readouterr().out
    assert "Representative SEDs" in output
    assert "spectrum file not found" in output
    assert "gone.txt" in output
    # One Teff and one log g: no corner to mark; and no file column, so no SED to plot
    (t.tmp / "flat").mkdir()
    (t.tmp / "flat" / "lookup_table.csv").write_text("teff,logg,metallicity\n5000,4.0,0.0\n5000,4.0,0.0\n")
    grid_coverage(t.tmp / "flat", plot=False)
    assert ("Terminal SED plot skipped: lookup_table.csv does not identify spectrum files."
            in t.capsys.readouterr().out)
    # Representative spectra with fewer than two positive fluxes are listed, not plotted
    for name, teff in (("a.txt", 4000), ("b.txt", 5000)):
        (t.tmp / "dim" / name).parent.mkdir(exist_ok=True)
        (t.tmp / "dim" / name).write_text(f"# teff = {teff}\n# logg = 3.0\n1000 0.0\n2000 1.0\n")
    grid_coverage(t.tmp / "dim", plot=False)
    output = t.capsys.readouterr().out
    assert "Terminal SED plot skipped: no readable representative spectra." in output
    assert f"skipped SED A: fewer than 2 positive flux points in {t.tmp / 'dim' / 'a.txt'}" in output
    assert f"skipped SED C: fewer than 2 positive flux points in {t.tmp / 'dim' / 'b.txt'}" in output
    # No spectrum with a Teff: nothing is representative
    (t.tmp / "unknown").mkdir()
    (t.tmp / "unknown" / "lookup_table.csv").write_text("file_name,teff\nx.txt,\n")
    grid_coverage(t.tmp / "unknown", plot=False)
    assert "Terminal SED plot skipped: no readable representative spectra." in t.capsys.readouterr().out


@row
def recovery_warns_about_unreadable_h5(t):
    """An unreadable .h5 used to be skipped with a DEBUG message only."""
    from sed_tools.grids.clean import recover_wavelengths_from_hdf5

    (t.tmp / "broken.h5").write_bytes(b"not an hdf5 file")
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.grids.hdf5_wavelengths"):
        assert recover_wavelengths_from_hdf5(str(t.tmp), "g1", 10) is None
    assert "broken.h5" in t.caplog.text


@row
def clean_model_dir_raises_listing_unconverted_spectra(t):
    """It used to return a summary as if the grid were clean, leaving two files in their original units."""
    for i in range(3):
        _spectrum(t.tmp / f"good_{i}.txt", teff=4000.0 + 1000 * i)
    (t.tmp / "empty.txt").write_text("# teff = 5000\n")
    _spectrum(t.tmp / "index.txt", units="", wl=np.arange(30.0))
    with pytest.raises(RuntimeError) as err:
        clean_model_dir(str(t.tmp), sample_fraction=1.0)
    message = str(err.value)
    assert "2 of 5 spectra" in message
    assert "empty data (1): empty.txt" in message
    assert "index grid without HDF5 recovery (1): index.txt" in message
    assert "lookup_table.csv was rebuilt" in message
    assert all(_is_standardized(t.tmp / f"good_{i}.txt") for i in range(3))
    assert not _is_standardized(t.tmp / "empty.txt")
    assert (t.tmp / "lookup_table.csv").exists()


@row
def clean_model_dir_records_unreadable_file_and_finishes_the_rest(t):
    _spectrum(t.tmp / "a.txt")
    (t.tmp / "b.txt").mkdir()  # matches *.txt but cannot be read as a file
    _spectrum(t.tmp / "c.txt")
    t.mp.setattr("sed_tools.grids.units.random.sample", lambda files, n: [str(t.tmp / "a.txt")])
    with pytest.raises(RuntimeError, match=r"IsADirectoryError.*\(1\): b\.txt"):
        clean_model_dir(str(t.tmp), rebuild_lookup=False, sample_fraction=0.4)
    assert _is_standardized(t.tmp / "a.txt") and _is_standardized(t.tmp / "c.txt")


@row
def clean_model_dir_unreadable_sampled_file_stops_before_any_rewrite(t):
    """It used to be counted as an invalid sample without saying why."""
    _spectrum(t.tmp / "a.txt")
    (t.tmp / "b.txt").mkdir()
    with pytest.raises(IsADirectoryError, match=r"b\.txt"):
        clean_model_dir(str(t.tmp), sample_fraction=1.0)
    assert not _is_standardized(t.tmp / "a.txt")


@row
def clean_model_dir_stops_at_once_when_a_rewrite_fails(t):
    """A failed rewrite stops the batch (it used to record 'error' and carry on), and leaves the file unchanged."""
    for name in ("a.txt", "b.txt"):
        _spectrum(t.tmp / name)
    calls = []

    def disk_full(filepath, *args):
        calls.append(filepath)
        raise OSError(28, "No space left on device")

    t.mp.setattr("sed_tools.grids.clean.write_standardized_spectrum", disk_full)
    with pytest.raises(RuntimeError, match=r"a\.txt.*The file is unchanged"):
        clean_model_dir(str(t.tmp), sample_fraction=1.0)
    assert len(calls) == 1


@row
def clean_model_dir_warns_when_units_come_from_each_file(t):
    """The sample held only standardized files, so the grid-wide units were unknown and the rest were converted
    one by one, silently."""
    for i in range(9):
        _spectrum(t.tmp / f"raw_{i}.txt")
    done = t.tmp / "done.txt"
    done.write_text("# units_standardized = True\n4000 1.0\n5000 2.0\n")
    t.mp.setattr("sed_tools.grids.units.random.sample", lambda files, n: [str(done)])
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.grids.clean"):
        summary = clean_model_dir(str(t.tmp), sample_fraction=0.1)
    assert len(summary["converted"]) == 9
    assert "9 spectra were converted using units detected file by file" in t.caplog.text


@row
def clean_model_dir_counts_renormed_spectra_as_converted_and_warns(t):
    wl, grid = np.geomspace(100.0, 1e6, 2000), t.tmp / "grid"
    for name, teff, scale in (("a.txt", 5000.0, 100.0), ("b.txt", 6000.0, 120.0), ("c.txt", 7000.0, 1.0)):
        _write(grid / name, _header(teff), wl, scale * planck_flam(wl, teff))
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.grids.clean"):
        summary = clean_model_dir(str(grid), try_h5_recovery=False, sample_fraction=1.0)
    assert summary["converted"] == ["a.txt", "b.txt", "c.txt"]
    assert summary["error"] == []
    assert sorted(summary["renormed"]) == ["a.txt", "b.txt"]
    assert summary["renormed"]["a.txt"] == pytest.approx(100.0)
    assert summary["renormed"]["b.txt"] == pytest.approx(120.0)
    warning = [r.getMessage() for r in t.caplog.records if "bolometric renormalisation" in r.getMessage()]
    assert len(warning) == 1 and "2 spectra integrated to 100-120 times" in warning[0]
    for name, teff in (("a.txt", 5000.0), ("b.txt", 6000.0), ("c.txt", 7000.0)):
        wl_after, flux_after = read_text_spectrum(str(grid / name))
        assert np.trapezoid(flux_after, wl_after) / (SIGMA * teff ** 4) == pytest.approx(1.0, rel=0.01)


@row
def coverage_gives_the_normality_test_or_says_why_it_was_skipped(t):
    """It said "scipy required" whenever the test was skipped, with scipy installed."""
    from sed_tools.grids.coverage import coverage_summary, print_coverage

    def grid(directory, teffs):
        directory.mkdir()
        rows = "".join(f"s{i}.txt,{teff},4.0,0.0\n" for i, teff in enumerate(teffs))
        (directory / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\n" + rows)
        return directory

    print_coverage(coverage_summary(grid(t.tmp / "few", [5000, 5500, 6000])))
    out = t.capsys.readouterr().out
    assert "A-D     : skipped (needs at least 8 distinct values; this axis has 3)" in out
    assert "scipy" not in out
    # 14 Teffs, one spectrum without a log g: the values are elided, the missing one counted, the test made
    fourteen = grid(t.tmp / "fourteen", range(4000, 10500, 500))
    with (fourteen / "lookup_table.csv").open("a") as table:
        table.write("s13.txt,11000,,0.0\n")
    summary = coverage_summary(fourteen)
    print_coverage(summary)
    out = t.capsys.readouterr().out
    assert "values  : 14 unique  [4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, ..., 10000, 11000]" in out
    assert "missing : 1 spectra have no value on this axis" in out
    ad = summary["axes"]["teff"]["anderson_darling"]
    assert f"A-D     : {ad['statistic']:.3f} (5% crit {ad['crit_5pct']:.3f}; consistent with normal)" in out
    # SciPy 1.19 and later give only a p-value
    t.mp.setattr("scipy.stats.anderson", lambda x, dist: SimpleNamespace(statistic=0.25, pvalue=0.5))
    print_coverage(coverage_summary(fourteen))
    assert "A-D     : 0.250 (p=0.500; consistent with normal)" in t.capsys.readouterr().out
    t.mp.setitem(sys.modules, "scipy.stats", None)  # as if scipy were not installed
    print_coverage(coverage_summary(grid(t.tmp / "many", range(4000, 9000, 500))))
    out = t.capsys.readouterr().out
    assert "    A-D     : skipped (needs scipy, which is not installed)" in out
    (t.tmp / "empty").mkdir()
    with pytest.raises(RuntimeError, match="No lookup_table.csv and no .txt spectra"):
        coverage_summary(t.tmp / "empty")


@row
def coverage_lists_the_bands_the_grid_without_a_cube_does_not_sample(t):
    from sed_tools.grids.coverage import grid_coverage

    grid, filters = t.tmp / "Grid", t.tmp / "filters"
    _write(grid / "s.txt", _header(5000), np.arange(3000.0, 7000.0, 10.0), np.ones(400))
    (grid / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\ns.txt,5000,4.0,0.0\n")
    (filters / "Generic" / "Johnson").mkdir(parents=True)
    (filters / "Generic" / "Johnson" / "V.dat").write_text(CURVE)
    (filters / "Far").mkdir()
    (filters / "Far" / "K.dat").write_text("Wavelength,Transmission\n20000,0\n22000,1\n24000,0\n")
    t.mp.setattr("sed_tools.cube.model.FILTER_DIR_DEFAULT", filters)
    t.mp.setattr(config, "get_ui_setting", lambda name: "never")
    summary = grid_coverage(grid, plot=False, filters=["Generic/Johnson", str(filters / "Far" / "K.dat")])
    assert list(summary["bands_not_sampled"]) == ["Far/K"]
    assert list(summary["bands_not_sampled"]["Far/K"]) == ["(cube to be built)"]
    assert "Band sampling of Grid: 2 bands, 1 not sampled" in t.capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match=r"No filter curves found for 'Generic/None'"):
        grid_coverage(grid, plot=False, filters=["Generic/None"])


WL12, WL11, WL9 = np.linspace(3000.0, 9000.0, 12), np.linspace(3000.0, 9000.0, 11), np.geomspace(1000.0, 50000.0, 9)


def _ranges(f):
    rg = f.create_group("g1").create_group("range")
    rg.create_group("range2").create_dataset("wave", data=np.linspace(6000.0, 9000.0, 5))
    rg.create_group("range1").create_dataset("wave", data=np.linspace(3000.0, 5000.0, 5))
    f.create_group("g2").create_group("range").create_dataset("x", data=np.linspace(2000.0, 4000.0, 8))


def _among_others(f):
    """The grid comes last in name order, after a string and a compound dataset of the same length."""
    f.create_dataset("labels", data=np.array([b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h", b"i"]))
    f.create_dataset("records", data=np.zeros(9, dtype=[("a", "f8"), ("b", "i4")]))
    f.create_group("wave").create_group("er").create_dataset("grid", data=WL9)


def _short_segments(f):
    rg = f.create_group("g1").create_group("range")
    rg.create_group("range1").create_dataset("wave", data=np.linspace(3000.0, 5000.0, 5))
    rg.create_dataset("range2", data=np.linspace(6000.0, 9000.0, 5))  # a dataset, not a segment group


def _top_level(f):
    f.create_dataset("wavelength", data=WL11)


#: name: (what msg.h5 holds, the spectrum's group, its number of points, the wavelengths recovered, or None)
RECOVERED = {
    "recovery_reads_direct_dataset_in_spec_group": (
        lambda f: f.create_group("g1").create_dataset("lambda", data=WL12), "g1", 10, WL12[:10]),
    "recovery_reads_range_x_then_segments_segments": (
        _ranges, "g1", 10, np.concatenate([np.linspace(3000.0, 5000.0, 5), np.linspace(6000.0, 9000.0, 5)])),
    "recovery_reads_range_x_then_segments_x": (_ranges, "g2", 8, np.linspace(2000.0, 4000.0, 8)),
    "recovery_top_level_array_within_two_points": (_top_level, "absent", 10, WL11[:10]),
    "recovery_top_level_array_not_within_two_points": (_top_level, "absent", 20, None),
    "recovery_global_search_skips_non_numeric_datasets": (_among_others, "absent", 9, WL9),
    "recovery_rejects_index_like_arrays": (
        lambda f: f.create_group("g1").create_dataset("lambda", data=np.arange(10.0)), "g1", 10, None),
    "recovery_rejects_index_like_arrays_from_1": (
        lambda f: f.create_group("g1").create_dataset("lambda", data=np.arange(1.0, 11.0)), "g1", 10, None),
    "recovery_rejects_a_single_point": (
        lambda f: f.create_group("g1").create_dataset("lambda", data=WL12), "g1", 1, None),
    "recovery_rejects_a_dataset_shorter_than_the_spectrum": (
        lambda f: f.create_group("g1").create_dataset("lambda", data=WL9), "g1", 10, None),
    "recovery_rejects_segments_shorter_than_the_spectrum": (_short_segments, "g1", 10, None),
}


def _recovered(t, build, group, n, expected):
    from sed_tools.grids.clean import recover_wavelengths_from_hdf5

    with h5py.File(t.tmp / "msg.h5", "w") as f:
        build(f)
    found = recover_wavelengths_from_hdf5(str(t.tmp), group, n)
    if expected is None:
        assert found is None
    else:
        np.testing.assert_array_equal(found, expected)


#: name: (Teff, wavelengths, flux in units of pi*B_lambda): spectra the bolometric check converts unscaled.
LEFT_ALONE = {
    # Granada HRES PHOENIX giants at 3000-3500 K, log g <= 1, have 2-10 % of a blackbody's flux over 3000-7000 A
    # (TiO), which holds 8-15 % of sigma*T^4. They are correct; the first version, which also rescaled ratios below
    # 0.1, multiplied them by 10-50.
    "cool_giant_far_below_the_blackbody_in_the_optical_is_left_alone": (
        3000.0, np.linspace(3000.0, 7000.0, 4000), 0.03),
    # B_lambda instead of pi*B_lambda (MESA's bbody grid) is 1/pi of sigma*T^4: not rescaled.
    "pi_less_blackbody_is_within_the_thresholds": (5000.0, np.geomspace(100.0, 1e6, 3000), 1 / np.pi),
    # 3000-9000 A holds about 3 % of a 50000 K blackbody's flux: too little to judge the scale.
    "spectrum_covering_little_of_sigma_t4_is_not_checked": (50000.0, np.linspace(3000.0, 9000.0, 500), 100.0),
}


def _left_alone(t, teff, wl, scale):
    path = t.tmp / "spectrum.txt"
    _write(path, _header(teff), wl, scale * planck_flam(wl, teff))
    _, written_before = read_text_spectrum(str(path))
    status, _ = clean_spectrum_file(str(path))
    assert status == "converted"
    np.testing.assert_array_equal(read_text_spectrum(str(path))[1], written_before)


# ----------------------------------------------------------------------
# Flux units: cleaning stops on an undeclared unit and the CLI asks for one; declared units are converted
# ----------------------------------------------------------------------


def _grid(directory, flux_unit=None, n=200, flux=planck_flam, row="{:.6f} {:.8e}\n"):
    """Spectra at 3500, 3600 and 3700 K in Å, declaring ``flux_unit`` or no flux unit; by default pi*B_lambda, a
    surface flux in erg/s/cm2/A."""
    wl = np.linspace(3000.0, 10000.0, n)
    units = "# wavelength_unit = Angstrom\n" + (f"# flux_unit = {flux_unit}\n" if flux_unit else "")
    for i, teff in enumerate((3500.0, 3600.0, 3700.0)):
        _write(directory / f"s{i}.txt", _header(teff, 4.5, units), wl, flux(wl, teff), row)
    return directory


#: 1e5 erg/s/cm2/A at 5000 A, falling as wavelength^-2, written with fewer digits
FLAM_GRID = {"n": 50, "flux": lambda wl, teff: 1e5 * (wl / 5000.0) ** -2, "row": "{:.4f} {:.6e}\n"}


def _terminal(t, *replies):
    """input() on a terminal, answering ``replies``; the prompts it was given."""
    t.mp.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    return answer(t.mp, *replies)


@row
def clean_model_dir_stops_on_an_undeclared_flux_unit_naming_the_best_guess(t):
    from sed_tools.grids.units import undeclared_flux_unit

    grid = _grid(t.tmp / "Cool")
    before = _contents(grid)
    # The range detector reads these surface fluxes (1e5-1e6) as Jansky at Earth.
    assert undeclared_flux_unit(sorted(str(p) for p in grid.glob("*.txt")), 1.0) == "fnu_jy"
    with pytest.raises(ValueError) as err:
        clean_model_dir(str(grid), sample_fraction=1.0)
    message = str(err.value)
    assert f"Flux unit of {grid} is not declared; best guess: 'fnu_jy', Jansky" in message
    assert ("Pass the unit as flux_unit='fnu_jy', or as another of flam, fnu, fnu_jy, flam_si, flam_si_nm, "
            "flam_si_um, normalized") in message
    assert _contents(grid) == before
    assert not (grid / "lookup_table.csv").exists()


@row
def clean_model_dir_uses_the_flux_unit_it_is_given(t):
    """Declutter B1: an upper-case .TXT spectrum is cleaned too (until 0.6.0 it was left uncleaned, yet bundled and
    put in the cube)."""
    grid = _grid(t.tmp / "Cool")
    (grid / "s2.txt").rename(grid / "s2.TXT")
    (grid / "s0.txt").write_text("# vturb = 2.0\n" + (grid / "s0.txt").read_text())
    summary = clean_model_dir(str(grid), sample_fraction=1.0, flux_unit="flam")
    assert len(summary["converted"]) == 3 and summary["catalog_units"]["flux"] == "flam"
    wl, flux = read_text_spectrum(str(grid / "s0.txt"))
    np.testing.assert_allclose(flux, planck_flam(wl, 3500.0), rtol=1e-8)
    assert "# original_flux_unit = flam" in (grid / "s0.txt").read_text()
    assert "# vturb = 2.0\n" in (grid / "s0.txt").read_text()  # a header line the cleaner does not know is kept


@row
def declared_flux_units_clean_as_before(t):
    """A declared unit is used without asking, and gives the same files as passing it."""
    declared, passed = _grid(t.tmp / "Declared", "erg/cm2/s/A"), _grid(t.tmp / "Passed")
    clean_model_dir(str(declared), sample_fraction=1.0)
    clean_model_dir(str(passed), sample_fraction=1.0, flux_unit="flam")
    assert _contents(declared) == _contents(passed)


@row
def a_spectrum_cleaned_on_its_own_stops_on_an_undeclared_flux_unit(t):
    grid = _grid(t.tmp / "Cool")
    before = (grid / "s0.txt").read_bytes()
    with pytest.raises(ValueError, match=r"Flux unit of this spectrum is not declared; best guess: 'fnu_jy'"):
        clean_spectrum_file(str(grid / "s0.txt"))
    assert (grid / "s0.txt").read_bytes() == before


@row
def files_cleaned_one_by_one_without_a_declared_flux_unit_are_listed(t):
    """The sample held only a standardized file, so there are no grid-wide units."""
    grid = _grid(t.tmp / "Cool")
    done = grid / "done.txt"
    done.write_text("# units_standardized = True\n4000 1.0\n5000 2.0\n")
    t.mp.setattr("sed_tools.grids.units.random.sample", lambda files, n: [str(done)])
    with pytest.raises(RuntimeError, match=r"(?s)3 of 4 spectra .*Flux unit of this spectrum is not declared; "
                                           r"best guess: 'fnu_jy'.* \(3\): s0.txt, s1.txt, s2.txt"):
        clean_model_dir(str(grid), sample_fraction=0.1)
    summary = clean_model_dir(str(grid), sample_fraction=0.1, flux_unit="flam")
    assert summary["converted"] == ["s0.txt", "s1.txt", "s2.txt"]


@row
def cli_stops_on_no(t):
    grid = _grid(t.tmp / "Cool")
    prompts = _terminal(t, "n")
    with pytest.raises(ValueError, match="No flux unit chosen for Cool; it was not cleaned"):
        cli.grids._ask_flux_unit("Cool", sorted(str(p) for p in grid.glob("*.txt")))
    assert len(prompts) == 1


@row
def cli_does_not_ask_when_the_flux_unit_is_declared(t):
    grid = _grid(t.tmp / "Declared", "erg/cm2/s/A")
    prompts = _terminal(t)
    assert cli.grids._ask_flux_unit("Declared", sorted(str(p) for p in grid.glob("*.txt"))) is None
    assert prompts == []


@row
def cli_rebuild_asks_once_per_grid_and_cleans_with_the_answer(t):
    for name in ("A", "B"):
        _grid(t.tmp / name)
    prompts = _terminal(t, "flam", "fnu_jy")
    cli.run_rebuild_flow(base_dir=str(t.tmp), models=["A", "B"], rebuild_h5=False, rebuild_flux_cube=False)
    assert [p.split(" (")[0] for p in prompts] == ["Flux unit of A", "Flux unit of B"]
    wl, flux = read_text_spectrum(str(t.tmp / "A" / "s0.txt"))
    np.testing.assert_allclose(flux, planck_flam(wl, 3500.0), rtol=1e-8)
    assert "# original_flux_unit = fnu" in (t.tmp / "B" / "s0.txt").read_text()


@row
def cli_without_a_terminal_stops_naming_the_grid_and_the_guess(t):
    grid = _grid(t.tmp / "Cool")
    before = _contents(grid)
    t.mp.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    t.mp.setattr("builtins.input", lambda prompt="": pytest.fail("asked without a terminal"))
    with pytest.raises(RuntimeError) as err:
        cli.run_rebuild_flow(base_dir=str(t.tmp), models=["Cool"], rebuild_h5=False, rebuild_flux_cube=False)
    message = str(err.value)
    assert f"Cool: ValueError: Flux unit of {grid} is not declared; best guess: 'fnu_jy'" in message
    assert "--flux-unit fnu_jy" in message and "sed-tools asks for it when run in a terminal" in message
    assert _contents(grid) == before


@row
def import_grid_stops_before_moving_files_and_imports_with_a_unit(t):
    src = _grid(t.tmp / "mine")
    with pytest.raises(ValueError, match=rf"Flux unit of {src} is not declared; best guess: 'fnu_jy'"):
        SED.import_grid(src, model_root=t.tmp / "models", move=True, build_cube=False, build_h5=False)
    assert len(list(src.glob("*.txt"))) == 3
    assert not (t.tmp / "models" / "mine").exists()
    SED.import_grid(src, model_root=t.tmp / "models", move=True, build_cube=False, build_h5=False, flux_unit="flam")
    assert "# original_flux_unit = flam" in (t.tmp / "models" / "mine" / "s0.txt").read_text()


#: name: how an unknown flux unit is given. The unit is looked up before any file is copied, moved or rewritten,
#: so the spectra are left as they were and no grid folder is made (cleaning one by one used to convert the
#: spectra that declare their unit before stopping on one that does not).
UNKNOWN_UNIT = {
    "an_unknown_flux_unit_leaves_the_source_folder_untouched_on_import": "copy",
    "an_unknown_flux_unit_leaves_the_source_folder_untouched_on_import_with_move": "move",
    "an_unknown_flux_unit_leaves_every_spectrum_untouched_when_cleaning_one_by_one": "clean",
    "an_unknown_flux_unit_downloads_nothing_on_fetch": "fetch",
}


def _unknown_unit(t, how):
    src = _grid(t.tmp / "mine")
    (src / "declared.txt").write_text("# teff = 3500\n# flux_unit = erg/s/cm2/A\n3000 1.0\n4000 2.0\n")
    done = src / "done.txt"
    done.write_text("# units_standardized = True\n4000 1.0\n5000 2.0\n")
    t.mp.setattr("sed_tools.grids.units.random.sample", lambda files, n: [str(done)])  # no grid-wide units
    before = _contents(src)
    t.mp.setattr("sed_tools.api.SED._download_from_first_source", lambda *a: pytest.fail("downloaded"))
    with pytest.raises(KeyError, match="Jy"):
        if how == "fetch":
            SED.fetch("Grid", source="svo", model_root=t.tmp / "models", flux_unit="Jy")
        elif how == "clean":
            clean_model_dir(str(src), flux_unit="Jy")
        else:
            SED.import_grid(src, model_root=t.tmp / "models", move=how == "move", build_cube=False, build_h5=False,
                            flux_unit="Jy")
    assert _contents(src) == before
    assert not (t.tmp / "models").exists()


@row
def cli_import_asks_for_the_flux_unit_and_passes_it_on(t):
    src = _grid(t.tmp / "mine")
    calls = []
    dest = t.tmp / "models" / "mine"
    t.mp.setattr("sed_tools.api.SED.import_grid", lambda *a, **k: calls.append(k) or {"dest": str(dest)})
    _terminal(t, str(src), "", "n", "flam")
    dest.mkdir(parents=True)
    (dest / "lookup_table.csv").write_text("file_name,teff\n")
    cli.run_import_flow(base_dir=str(t.tmp / "models"))
    assert calls[0]["flux_unit"] == "flam"
    assert f"stellar_atm = '{dest}'" in t.capsys.readouterr().out  # as every other build prints it


@row
def cli_import_command_asks_on_a_terminal_and_not_for_a_dry_run(t):
    src = _grid(t.tmp / "mine")
    calls = []
    dest = t.tmp / "models" / "mine"
    dest.mkdir(parents=True)
    (dest / "lookup_table.csv").write_text("file_name,teff\n")
    t.mp.setattr("sed_tools.api.SED.import_grid", lambda *a, **k: calls.append(k) or {"dest": str(dest)})
    prompts = _terminal(t, "fnu")
    for extra in ([], ["--dry-run"]):
        t.mp.setattr(sys, "argv", ["sed-tools", "import", "--path", str(src), "--base", str(t.tmp / "models")] + extra)
        assert cli.main() == 0
        assert (f"stellar_atm = '{dest}'" in t.capsys.readouterr().out) == (not extra)  # the MESA path, once built
    assert [c["flux_unit"] for c in calls] == ["fnu", None]
    assert len(prompts) == 1


@row
def fetch_stops_on_an_undeclared_flux_unit_and_cleans_with_the_one_given(t):
    def download(cls, catalog, sources, failures, model_dir, workers, cuts):
        _grid(model_dir)
        return "svo"

    t.mp.setattr(SED, "_download_from_first_source", classmethod(download))
    with pytest.raises(ValueError, match=r"Flux unit of .*Cool is not declared; best guess: 'fnu_jy'"):
        SED.fetch("Cool", source="svo", model_root=t.tmp)
    sed = SED.fetch("Cool", source="svo", model_root=t.tmp, flux_unit="flam")
    assert len(sed.grid) == 3
    assert "# original_flux_unit = flam" in (t.tmp / "Cool" / "s0.txt").read_text()


@row
def units_are_guessed_from_the_range_of_the_values(t):
    from sed_tools.grids.units import detect_flux_unit_from_range, detect_wavelength_unit_from_range

    wavelengths = {  # the values: (unit, factor, confidence)
        (5000.0,): ("angstrom", 1.0, "low"),  # one point says nothing
        (0.0, 1.0, 2.0, 3.0): ("index", 0.0, "index_grid"),
        (300.0, 400.0): ("nm", 10.0, "medium"),
        (0.3, 2.5): ("um", 1e4, "medium"),
        (0.001, 0.04): ("cm", 1e8, "low"),
        (1e-7, 1e-6): ("m", 1e10, "low"),
        (1e-12, 1e-11): ("angstrom", 1.0, "low"),  # below every unit's range
    }
    for values, unit in wavelengths.items():
        assert detect_wavelength_unit_from_range(np.array(values)) == unit, values
    fluxes = {  # a typical flux: (flux type, factor, confidence)
        0.0: ("flam", 1.0, "low"),  # no positive flux
        1e-25: ("fnu", 1.0, "low"),
        1.0: ("normalized", 1.0, "low"),
        10 ** -17.5: ("flam", 1e3, "low"),
        1e8: ("normalized", 1.0, "low"),
        1e-17: ("flam", 1.0, "low"),  # on the edge of F_lambda's range
    }
    for value, unit in fluxes.items():
        assert detect_flux_unit_from_range(np.full(3, value)) == unit, value


@row
def grid_units_say_how_far_the_sampled_spectra_agree(t):
    from sed_tools.grids.units import detect_catalog_units

    def grid(name, units):
        for i, (wavelength_unit, flux_unit) in enumerate(units):
            _write(t.tmp / name / f"s{i}.txt", f"# wavelength_unit = {wavelength_unit}\n# flux_unit = {flux_unit}\n",
                   (400.0, 500.0, 600.0), (1.0, 2.0, 3.0))
        return sorted(str(p) for p in (t.tmp / name).glob("*.txt"))

    assert detect_catalog_units([]) == (None, {"error": "no files"})
    flam, fnu = "erg/s/cm2/A", "erg/s/cm2/Hz"
    units, stats = detect_catalog_units(grid("mostly", [("Angstrom", flam)] * 3 + [("nm", flam)]), 1.0)
    assert (units.wavelength_unit, units.flux_type, units.confidence) == ("angstrom", "flam", "catalog_medium")
    assert stats["wavelength_agreement"] == "75%"
    # No spectrum has both the most common wavelength unit (Angstrom, first of a tie) and flux type (flam): the
    # flux factor is the first Angstrom spectrum's
    units, stats = detect_catalog_units(grid("mixed", [("Angstrom", fnu), ("Angstrom", "normalized"), ("nm", flam),
                                                       ("nm", flam), ("um", flam)]), 1.0)
    assert (units.wavelength_unit, units.flux_type, units.flux_factor, units.confidence) == (
        "angstrom", "flam", 1.0, "catalog_low")


@row
def si_flux_units_have_the_factor_of_their_wavelength_unit(t):
    assert FLUX_UNITS["flam_si"][:2] == ("flam", 1e3)
    assert FLUX_UNITS["flam_si_nm"][:2] == ("flam", 1e2)
    assert FLUX_UNITS["flam_si_um"][:2] == ("flam", 0.1)


LAMBDA_A = np.linspace(3000.0, 10000.0, 41)
FLAM = 1e-5 * (LAMBDA_A / 5000.0) ** -2          # the known F_λ, erg/s/cm²/Å
FNU = FLAM * LAMBDA_A ** 2 / 2.99792458e18       # F_ν = F_λ λ²/c, erg/s/cm²/Hz, with c in Å/s
EXACT = "{:.17g} {:.17g}\n"
#: The known spectrum in each declared unit, by hand. 1 erg/s = 1e-7 W and 1 cm² = 1e-4 m², so
#: 1 erg/s/cm² = 1e-3 W/m²; an nm holds 10 Å and a µm 1e4 Å.
DECLARED = {"erg/s/cm2/A": FLAM, "erg/s/cm2/Hz": FNU, "Jy": FNU / 1e-23, "W/m2/A": FLAM * 1e-3,
            "W/m2/nm": FLAM * 1e-3 * 10.0, "W/m2/um": FLAM * 1e-3 * 1e4}


def _declared(path, flux_header, flux):
    _write(path, f"# wavelength_unit = Angstrom\n# flux_unit = {flux_header}\n", LAMBDA_A, flux, EXACT)


@row
def a_grid_declaring_w_m2_nm_in_some_spectra_names_that_unit_as_its_guess(t):
    """The guess is the unit of the first sampled spectrum; W/m²/nm used to be read with the W/m²/Å factor, which a
    FLUX_UNITS name had to match."""
    grid = t.tmp / "Mixed"
    _declared(grid / "a.txt", "W/m2/nm", DECLARED["W/m2/nm"])
    _write(grid / "b.txt", "", LAMBDA_A, FLAM, EXACT)
    with pytest.raises(ValueError, match=r"is not declared; best guess: 'flam_si_nm', W/m²/nm"):
        clean_model_dir(str(grid), sample_fraction=1.0)


@row
def a_declared_flux_unit_is_used_and_a_different_flux_unit_is_reported_unused(t):
    declared = _grid(t.tmp / "Declared", "erg/cm2/s/A", **FLAM_GRID)
    reference = _grid(t.tmp / "Reference", "erg/cm2/s/A", **FLAM_GRID)
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.grids.units"):
        summary = clean_model_dir(str(declared), sample_fraction=1.0, flux_unit="fnu")
    clean_model_dir(str(reference), sample_fraction=1.0)
    assert summary["catalog_units"]["flux"] == "flam"  # the header's unit, not fnu
    assert [p.read_text() for p in sorted(declared.glob("s*.txt"))] == \
        [p.read_text() for p in sorted(reference.glob("s*.txt"))]
    assert "the spectra declare their flux unit (flam); flux_unit='fnu' is not used" in t.caplog.text


@row
def flux_unit_is_used_for_spectra_that_declare_none(t):
    grid = _grid(t.tmp / "Undeclared", **FLAM_GRID)
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.grids.units"):
        summary = clean_model_dir(str(grid), sample_fraction=1.0, flux_unit="flam")
    assert summary["catalog_units"]["flux"] == "flam" and "is not used" not in t.caplog.text


#: name: (the answers typed, the unit taken). The guess is offered, not taken: the unit must be typed.
TYPED = {
    "cli_offers_the_best_guess_and_takes_a_typed_unit": (["flam"], "flam"),
}


def _typed(t, answers, unit):
    grid = _grid(t.tmp / "Cool")
    prompts = _terminal(t, *answers)
    assert cli.grids._ask_flux_unit("Cool", sorted(str(p) for p in grid.glob("*.txt"))) == unit
    assert prompts == ["Flux unit of Cool (type its name, e.g. fnu_jy), or n to leave it uncleaned: "] * len(answers)
    out = t.capsys.readouterr().out
    assert "flam        erg/s/cm²/Å (F_λ)" in out and "fnu_jy      Jansky (F_ν)" in out
    assert ("Best guess: fnu_jy, Jansky (F_ν); guessed from the size of the flux values, "
            "assuming flux received at Earth.") in " ".join(out.split())


#: name: (a declared flux unit, the known spectrum in it), for each unit in DECLARED
CONVERTED = {"a_declared_flux_unit_is_converted_to_erg_s_cm2_a_from_" + re.sub(r"\W", "_", unit).lower(): (unit, flux)
             for unit, flux in DECLARED.items()}


def _converted(t, flux_header, flux):
    grid = t.tmp / "Grid"
    _declared(grid / "s0.txt", flux_header, flux)
    summary = clean_model_dir(str(grid), sample_fraction=1.0, rebuild_lookup=False)
    assert summary["converted"] == ["s0.txt"]
    data = np.loadtxt(grid / "s0.txt", comments="#")
    np.testing.assert_allclose(data[:, 0], LAMBDA_A, rtol=0, atol=5e-7)
    np.testing.assert_allclose(data[:, 1], FLAM, rtol=1e-8, atol=0)  # written with 9 significant digits


#: name: (a header line, the flux unit read from it)
HEADERS = {
    # W/m² per nm and per µm used to be read as W/m²/Å
    "flux_unit_header_patterns_w_m2_nm": ("# flux_unit = W/m2/nm", "flam_si_nm"),
    "flux_unit_header_patterns_w_m2_um": ("# flux_unit = W/m^2/um", "flam_si_um"),
    "flux_unit_header_patterns_w_m2_a_in_brackets": ("# flux [W/m2/A]", "flam_si"),
    "flux_unit_header_patterns_w_m2_a_in_a_column_name": ("# columns = wavelength_A flux_W/m2/A", "flam_si"),
    # a unit next to a quantity name is read as the unit, not the name
    "flux_unit_header_patterns_f_nu_jy": ("# F_nu (Jy)", "fnu_jy"),
    "flux_unit_header_patterns_flambda_w_m2_um": ("# Flambda (W/m2/um)", "flam_si_um"),
    "flux_unit_header_patterns_f_lambda_w_m2_nm": ("# F_lambda (W/m^2/nm)", "flam_si_nm"),
    # words that used to be read as a unit
    "flux_unit_header_patterns_bt_settl_is_no_unit": ("# BT-Settl low mass models", None),
    "flux_unit_header_patterns_a_new_man_is_no_unit": ("# a new man", None),
    "flux_unit_header_patterns_per_arcsec2_is_no_unit": ("# flux erg/s/cm2/arcsec2", None),
    # spellings read as before
    "flux_unit_header_patterns_upper_case_erg_cm2_s_a": ("# column 2: FLUX (ERG/CM2/S/A)", "flam"),
    "flux_unit_header_patterns_erg_cm2_s_a": ("# flux_unit = erg/cm2/s/A", "flam"),
    "flux_unit_header_patterns_angstrom": ("# flux (erg/s/cm^2/Angstrom)", "flam"),
    "flux_unit_header_patterns_angstroms": ("# flux (erg/s/cm2/Angstroms)", "flam"),
    "flux_unit_header_patterns_aa": ("# erg/s/cm2/AA", "flam"),
    "flux_unit_header_patterns_ang": ("# erg/s/cm2/Ang", "flam"),
    "flux_unit_header_patterns_a_ring": ("# erg/s/cm2/Å", "flam"),
    "flux_unit_header_patterns_fnu_hz": ("# fnu (erg/s/cm2/Hz)", "fnu"),
}


def _header_unit(t, header, expected):
    assert detect_units_from_header(header + "\n")["flux_unit"] == expected


# ----------------------------------------------------------------------
# Downloads: MAST written in full, NJM outages, products out of date by content, selections
# ----------------------------------------------------------------------


class Routes(Session):
    """A session answering GET and HEAD from ``files`` {url: reply}: text or bytes, an HTTP status, a Resp, an
    exception to raise, or a callable of the request's keywords giving one of these. Any other URL is a 404;
    ``calls`` lists (method, url)."""

    def get(self, url, params=None, **kwargs):
        return self._answer("GET", url, params, kwargs)

    def head(self, url, params=None, **kwargs):
        return self._answer("HEAD", url, params, kwargs)

    def mount(self, prefix, adapter):
        pass

    def _answer(self, method, url, params, kwargs):
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        self.calls.append((method, url))
        reply = self.files.get(url, 404)
        reply = reply(**kwargs) if callable(reply) else reply
        if isinstance(reply, BaseException):
            raise reply
        return reply if isinstance(reply, Resp) else Resp(b"", reply) if isinstance(reply, int) else Resp(reply)


# Real BOSZ 2024 names, from MAST's bulk download script .../download_scripts/
# hlsp_bosz_bosz2024_sim_rorig_m-1.00_v1_bulkdl.sh, retrieved 2026-10-27: both signs of logg, [M/H], [alpha/M] and
# [C/M], and a five-digit Teff.
RORIG_NAMES = ["bosz2024_ap_t10500_g+2.0_m-1.00_a+0.25_c+0.50_v4_rorig_noresam.txt.gz",
               "bosz2024_ms_t2800_g-0.5_m-1.00_a-0.25_c-0.75_v2_rorig_noresam.txt.gz",
               "bosz2024_ms_t4250_g+3.0_m-1.00_a+0.50_c+0.25_v0_rorig_noresam.txt.gz"]
N_ROWS = 600
WAVE = 1000.0 + 10.0 * np.arange(N_ROWS)  # rises 50 % over the first 51 rows


def _flux_of(k):
    """A spectrum that is not monotonic, so it is never taken for wavelengths."""
    return (k + 1) * 1e-3 * (1.0 + 0.5 * np.sin(np.arange(N_ROWS)))


@row
def mast_rorig_writes_every_row_of_every_spectrum_including_the_first(t):
    urls = [f"{mast.DATA_BASE}/rorig/m-1.00/{name}" for name in RORIG_NAMES]
    script = "".join(f'curl -f --create-dirs --output "rorig/m-1.00/{n}" "{u}"\n' for n, u in zip(RORIG_NAMES, urls))
    routes = {f"{mast.SCRIPTS_DIR}/hlsp_bosz_bosz2024_sim_rorig_m-1.00_v1_bulkdl.sh": "#!/bin/sh\n" + script}
    for k, url in enumerate(urls):
        rows = "".join(f"{w:.3f} {f:.6e} {2 * f:.6e}\n" for w, f in zip(WAVE, _flux_of(k)))
        routes[url] = gzip.compress(rows.encode())
    grabber = mast.MASTSpectraGrabber(base_dir=str(t.tmp), max_workers=1, session=Routes(routes))
    n = grabber.download_model_spectra("BOSZ-2024-rorig", {"reskey": "rorig", "metals": ["m-1.00"]})
    assert n == len(RORIG_NAMES)
    for k, name in enumerate(RORIG_NAMES):  # the first URL in sorted order is RORIG_NAMES[0]
        data = np.loadtxt(t.tmp / "BOSZ-2024-rorig" / mast._sanitize_filename(name.replace(".txt.gz", ".txt")),
                          comments="#")
        assert data.shape == (N_ROWS, 3), name
        np.testing.assert_allclose(data[:, 0], WAVE)
        np.testing.assert_allclose(data[:, 1], _flux_of(k), rtol=1e-6)
        np.testing.assert_allclose(data[:, 2], 2 * _flux_of(k), rtol=1e-6)


INDEX = f"{NJM}/index.json"


def _outage(t, answers):
    """An NJMSpectraGrabber whose session answers each (method, url, verify) from ``answers``; and its calls."""
    calls = []

    def reply(method, url, verify=True, timeout=None, **kwargs):
        calls.append((method, url, verify, timeout))
        result = answers.get((method, url, verify), SimpleNamespace(status_code=404))
        if isinstance(result, Exception):
            raise result
        return result

    session = SimpleNamespace(headers={}, verify=True, get=lambda url, **kw: reply("GET", url, **kw),
                              head=lambda url, **kw: reply("HEAD", url, **kw))
    t.mp.setattr(requests, "Session", lambda: session)
    return njm.NJMSpectraGrabber(base_dir=str(t.tmp)), calls


@row
def a_silent_spectra_outage_costs_one_connect_timeout(t):
    grabber, calls = _outage(t, {("GET", INDEX, True): requests.ConnectTimeout("t")})
    assert not grabber.is_available()
    assert calls == [("GET", INDEX, True, (filters_njm.NJM_CONNECT_TIMEOUT, 10))]


@row
def a_spectra_mirror_that_answers_with_an_error_status_is_confirmed_by_head(t):
    grabber, calls = _outage(t, {("GET", INDEX, True): SimpleNamespace(status_code=503),
                                 ("HEAD", NJM, True): SimpleNamespace(status_code=403)})
    assert grabber.is_available() and [c[:2] for c in calls] == [("GET", INDEX), ("HEAD", NJM)]


@row
def plain_http_only_when_tls_fails_both_ways(t):
    plain = f"{NJM.replace('https', 'http')}/index.json"
    grabber, calls = _outage(t, {("GET", INDEX, True): requests.exceptions.SSLError("bad certificate"),
                                 ("GET", INDEX, False): requests.exceptions.SSLError("handshake failed"),
                                 ("GET", plain, False): SimpleNamespace(status_code=200,
                                                                        json=lambda: {"models": ["Grid"]})})
    assert grabber.is_available() and grabber.base_url.startswith("http://")
    assert [c[1:3] for c in calls] == [(INDEX, True), (INDEX, False), (plain, False)]


@row
def a_spectra_mirror_whose_head_does_not_answer_below_500_is_unreachable(t):
    for head in (requests.ConnectionError("refused"), SimpleNamespace(status_code=502)):
        _reset_sources()  # each grabber as in a new process: NJM is not paused by the one before
        grabber, calls = _outage(t, {("GET", INDEX, True): SimpleNamespace(status_code=503), ("HEAD", NJM, True): head})
        assert not grabber.is_available() and [c[:2] for c in calls] == [("GET", INDEX), ("HEAD", NJM)]


@row
def plain_http_that_fails_too_leaves_the_mirror_unreachable(t):
    plain = f"{NJM.replace('https', 'http')}/index.json"
    for reply in (requests.ConnectionError("refused"), SimpleNamespace(status_code=404)):
        _reset_sources()
        grabber, calls = _outage(t, {("GET", INDEX, True): requests.exceptions.SSLError("bad certificate"),
                                     ("GET", INDEX, False): requests.exceptions.SSLError("handshake failed"),
                                     ("GET", plain, False): reply})
        assert not grabber.is_available() and grabber.base_url == NJM
        assert [c[1:3] for c in calls] == [(INDEX, True), (INDEX, False), (plain, False)]


def digests(directory):
    """Each file's bytes and modification time."""
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(Path(directory).iterdir()) if p.is_file()}


def teff_axis(folder):
    return list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"])


@row
def a_whole_grid_copied_without_its_times_is_cut_as_it_stands(t):
    server = njm_grids(t)
    SED.fetch(GRID, source="njm", model_root=t.tmp / "a")
    folder = copy_without_times(t.tmp / "a" / GRID, t.tmp / "b" / GRID)
    assert _flux_cube_shortfall(folder) is None and _bundle_shortfall(folder / f"{GRID}.h5", folder) is None
    before = digests(folder)
    server.calls.clear()
    t.capsys.readouterr()
    sed = SED.fetch(GRID, source="njm", model_root=t.tmp / "b", teff_min=5900, teff_max=6100)
    assert digests(folder) == before  # no lookup table, bundle or cube rewritten
    assert server.spectra_requested() == [] and len(sed.grid) == 4
    assert sed(6000.0, 4.0, 0.0).flux.any()  # SED.fetch's return used to have no model to call
    out = t.capsys.readouterr().out
    assert "[fetch] 4 already present, 0 downloaded" in out
    assert "nothing was downloaded or rewritten" in " ".join(out.split())


@row
def grids_no_cube_on_a_grid_copied_without_its_times_exits_0(t):
    njm_grids(t)
    SED.fetch(GRID, source="njm", model_root=t.tmp / "a")
    folder = copy_without_times(t.tmp / "a" / GRID, t.tmp / "b" / GRID)
    cube = (folder / "flux_cube.bin").read_bytes()
    # The re-check's command, which 0.4.1 runs with exit 0 on such a copy
    assert run_cli(t.mp, "grids", "--source", "njm", "--grids", GRID, "--no-cube", "--base", str(t.tmp / "b")) == 0
    assert (folder / "flux_cube.bin").read_bytes() == cube
    assert "HDF5 bundle   : Kurucz2003all.h5 (exists)" in t.capsys.readouterr().out


@row
def grids_keeps_a_bundle_that_holds_every_spectrum(t):
    # sed-tools grids (without the --no-h5 it no longer has, Q10) rewrote the bundle on every run, with everything
    # present (notebooks review, 24 Nov 2026).
    njm_grids(t)
    bundle = t.tmp / GRID / f"{GRID}.h5"
    assert run_cli(t.mp, "grids", "--source", "njm", "--grids", GRID, "--base", str(t.tmp)) == 0
    before = (bundle.read_bytes(), bundle.stat().st_mtime_ns)
    t.capsys.readouterr()
    assert run_cli(t.mp, "grids", "--source", "njm", "--grids", GRID, "--base", str(t.tmp)) == 0
    assert (bundle.read_bytes(), bundle.stat().st_mtime_ns) == before
    assert f"HDF5 bundle   : {GRID}.h5 (exists)" in t.capsys.readouterr().out


@row
def a_selection_then_the_rest_is_rebuilt_whatever_the_file_times(t):
    njm_grids(t)
    SED.fetch(GRID, source="njm", model_root=t.tmp, teff_min=5900, teff_max=6100)
    folder = t.tmp / GRID
    future = time.time() + 3600  # the cube now looks newer than anything downloaded later
    for name in ("flux_cube.bin", f"{GRID}.h5"):
        os.utime(folder / name, (future, future))
    t.capsys.readouterr()
    sed = SED.fetch(GRID, source="njm", model_root=t.tmp)
    assert len(sed.grid) == 12 and teff_axis(folder) == [5750.0, 6000.0, 6250.0]
    assert spectra_not_in_cube(folder / "flux_cube.bin", folder / "lookup_table.csv")[1] == []
    out = t.capsys.readouterr().out
    assert "[fetch] 4 already present, 8 downloaded" in out
    assert "[fetch] flux_cube.bin does not include 8 of the 12 spectra beside it:" in out


@row
def a_whole_grid_and_a_selection_alone_are_complete(t):
    njm_grids(t)
    SED.fetch(GRID, source="njm", model_root=t.tmp / "whole")
    SED.fetch(GRID, source="njm", model_root=t.tmp / "cut", teff_min=5900, teff_max=6100)
    for folder in (t.tmp / "whole" / GRID, t.tmp / "cut" / GRID):
        assert _flux_cube_shortfall(folder) is None
        assert _bundle_shortfall(folder / f"{GRID}.h5", folder) is None


@row
def a_bundle_that_lacks_spectra_says_how_many(t):
    njm_grids(t)
    SED.fetch(GRID, source="njm", model_root=t.tmp, teff_min=5900, teff_max=6100)
    folder = t.tmp / GRID
    shutil.copyfile(KURUCZ / "Kurucz2003all_fid16515.txt", folder / "Kurucz2003all_fid16515.txt")
    assert _bundle_shortfall(folder / f"{GRID}.h5", folder) == "holds 4 of the 5 spectra beside it"


@row
def an_unreadable_cube_is_out_of_date(t):
    njm_grids(t)
    SED.fetch(GRID, source="njm", model_root=t.tmp)
    folder = t.tmp / GRID
    cube = folder / "flux_cube.bin"
    cube.write_bytes(cube.read_bytes()[:1000])  # truncated
    assert _flux_cube_shortfall(folder).startswith("cannot be compared with the spectra beside it")


@row
def import_helpers_take_a_single_spectrum_and_report_folders_without_spectra(t):
    spectrum = t.tmp / "one.txt"
    spectrum.write_text("# teff = 5000\n1000 1\n2000 1\n")
    assert pipeline._import_sources(spectrum) == ([spectrum], t.tmp)
    (t.tmp / "notes.dat").write_text("1000 1\n")
    (t.tmp / "empty").mkdir()
    for src in (t.tmp / "notes.dat", t.tmp / "empty"):
        with pytest.raises(RuntimeError, match=f"No .txt spectra found in {src}"):
            pipeline._import_sources(src)
    bare = [t.tmp / "bare" / f"s{i:02d}.txt" for i in range(12)]
    for path in bare:
        _write(path, "", (1000, 2000), (1, 1), "{} {}\n")
    assert pipeline._report_import_headers(bare, t.tmp / "bare") == [p.name for p in bare]
    out = t.capsys.readouterr().out
    assert "[import] missing one or more parameters: 12" in out and "s09.txt" in out and "s10.txt" not in out
    assert "... and 2 more" in out
    pipeline.build_h5_bundle_from_txt(str(t.tmp / "empty"), str(t.tmp / "empty" / "empty.h5"))
    assert f"No .txt spectra found in {t.tmp / 'empty'}; skipping" in t.capsys.readouterr().out
    assert not (t.tmp / "empty" / "empty.h5").exists()
    assert pipeline.local_grids(t.tmp / "absent") == [] and pipeline.rebuildable_grids(t.tmp / "absent") == []


@row
def the_rebuild_hint_names_base_only_when_the_grid_is_elsewhere(t):
    t.mp.setattr("sed_tools.grids.pipeline.STELLAR_DIR_DEFAULT", t.tmp / "default")
    assert pipeline._rebuild_command(GRID, str(t.tmp / "default" / GRID)) == f"sed-tools rebuild --grids {GRID}"
    elsewhere = t.tmp / "my grids"
    assert (pipeline._rebuild_command(GRID, str(elsewhere / GRID))
            == f"sed-tools rebuild --grids {GRID} --base '{elsewhere}'")


@row
def bounds_no_spectrum_meets_leave_no_folder(t):
    njm_grids(t)
    sed = SED.fetch(GRID, source="njm", model_root=t.tmp, teff_min=9000)
    assert len(sed.grid) == 0 and not (t.tmp / GRID).exists()
    (t.tmp / GRID).mkdir()  # a folder the call did not make stays, empty or not
    SED.fetch(GRID, source="njm", model_root=t.tmp, teff_min=9000)
    assert (t.tmp / GRID).is_dir()


#: Teff 5900-6100 selects the four spectra at 6000 K.
AT_6000 = ["Kurucz2003all_fid16526.txt", "Kurucz2003all_fid16527.txt",
           "Kurucz2003all_fid18906.txt", "Kurucz2003all_fid18907.txt"]


@row
def sed_local_then_uses_the_selection(t):
    njm_grids(t)
    fetched = SED.fetch(GRID, source="njm", model_root=t.tmp, teff_min=5900, teff_max=6100).grid
    local = SED.local(GRID, model_root=t.tmp)
    assert [s.filename for s in local.grid] == AT_6000
    assert [(s.teff, s.logg, s.metallicity) for s in local.grid] == [(s.teff, s.logg, s.metallicity) for s in fetched]
    assert local.parameter_ranges()["teff"] == (6000.0, 6000.0)
    assert np.all(np.isfinite(local(6000, 4.25, -0.25).flux))


@row
def svo_cannot_select_so_the_whole_grid_is_downloaded_then_cut(t):
    class SVO:
        def __init__(self, base_dir=None, max_workers=None):
            self.base_dir = Path(base_dir)

        def get_model_metadata(self, name):
            return [{"fid": n} for n in (1, 2, 3)]

        def download_model_spectra(self, name, spectra_info, **ranges):
            if any(r is not None for r in ranges.values()):
                raise ValueError("SVO downloads cannot be cut")  # as the real grabber
            for n, teff in zip((1, 2, 3), (5000, 6000, 7000)):
                _write(self.base_dir / name / f"{name}_fid{n}.txt",
                       _header(teff, units="# units_standardized = True\n"), (1000, 2000), ("1.0", "2.0"), "{} {}\n")
            return 3

    t.mp.setattr("sed_tools.grids.svo.SVOSpectraGrabber", SVO)
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.api"):
        sed = SED.fetch("Grid", source="svo", model_root=t.tmp, teff_min=5500, teff_max=6500)
    assert [r.getMessage() for r in t.caplog.records] == [
        "[fetch] svo cannot select spectra by Teff, logg or metallicity, so the whole of Grid is downloaded, then cut"]
    assert [s.teff for s in sed.grid] == [6000.0]
    assert len(list((t.tmp / "Grid").glob("*.txt"))) == 3 and not (t.tmp / "Grid" / "flux_cube.bin").exists()


@row
def grids_after_a_selection_rebuilds_or_says_why_it_did_not(t):
    njm_grids(t)
    base, folder = str(t.tmp), t.tmp / GRID
    assert run_cli(t.mp, "grids", "--grids", GRID, "--source", "njm", "--base", base, "--teff", "5900", "6100") == 0
    assert list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"]) == [6000.0]
    # The rest, without a cube build: the selection's cube is not taken for the whole grid's.
    assert run_cli(t.mp, "grids", "--grids", GRID, "--source", "njm", "--base", base, "--no-cube") == 1
    err = " ".join(t.capsys.readouterr().err.split())
    assert "flux_cube.bin does not include 8 of the 12 spectra beside it" in err
    assert f"Rebuild it with: sed-tools rebuild --grids {GRID} --base {base}" in err
    with h5py.File(folder / f"{GRID}.h5", "r") as bundle:  # rebuilt: it lacked the rest
        assert len(bundle["spectra"]) == 12
    assert run_cli(t.mp, "grids", "--grids", GRID, "--source", "njm", "--base", base) == 0
    assert list(_read_flux_cube_header(folder / "flux_cube.bin")["teff"]) == [5750.0, 6000.0, 6250.0]


#: Local grids: (logg, [M/H]) ranges where a bound of 0 keeps some grids and drops others.
LOCAL_GRIDS = {"Negative": ([-1.0, -0.5], [-2.0, -1.0]), "Positive": ([0.5, 1.0], [0.25, 0.5]),
               "Straddle": ([-0.5, 0.5], [-0.5, 0.5])}
#: name: (SED.query's bounds, the local grids it finds)
ZERO_BOUNDS = {
    "query_applies_a_bound_of_zero_teff_min": ({"teff_min": 0}, {"Negative", "Positive", "Straddle"}),
    "query_applies_a_bound_of_zero_teff_max": ({"teff_max": 0}, set()),
    "query_applies_a_bound_of_zero_logg_min": ({"logg_min": 0.0}, {"Positive", "Straddle"}),
    "query_applies_a_bound_of_zero_logg_max": ({"logg_max": 0.0}, {"Negative", "Straddle"}),
    "query_applies_a_bound_of_zero_metallicity_min": ({"metallicity_min": 0.0}, {"Positive", "Straddle"}),
    "query_applies_a_bound_of_zero_metallicity_max": ({"metallicity_max": 0.0}, {"Negative", "Straddle"}),
    "query_applies_a_bound_of_zero_metallicity_max_int": ({"metallicity_max": 0}, {"Negative", "Straddle"}),
    "query_applies_a_bound_of_zero_logg_min_and_metallicity_max": ({"logg_min": 0, "metallicity_max": 0.0},
                                                                   {"Straddle"}),
}


def _query_bound(t, bound, expected):
    for name, (logg, meta) in LOCAL_GRIDS.items():
        write_flux_cube(t.tmp / name / "flux_cube.bin", [4000.0, 6000.0], logg, meta, [4000.0, 5000.0],
                        np.ones((2, len(logg), len(meta), 2)))
    t.mp.setattr(SED, "_model_root", t.tmp)
    found = SED.query(source="local", include_remote=False, **bound)
    assert {info.name for info in found} == expected


# ----------------------------------------------------------------------
# Source failures: NJM, SVO, MSG and MAST stop, warn or raise, never vanish
# ----------------------------------------------------------------------
GRID_URL = f"{NJM}/stellar_models/Grid/"
ONLINE = {INDEX: json.dumps({"models": ["Grid", ".hidden"]})}
LOOKUP = "#file_name,teff,logg,meta\na.txt,5000,4.0,0.0\nb.txt,6000,4.5,0.0\n"
SPECTRUM = "# teff = 5000\n" + "".join(f"{wl} 1.0\n" for wl in range(1000, 1400))
SVO_INDEX = "http://svo2.cab.inta-csic.es/theory/newov2/index.php"
SSAP = "http://svo2.cab.inta-csic.es/theory/newov2/ssap.php"
SSAP_QUERY = f"{SSAP}?model=Grid&REQUEST=queryData&FORMAT=votable"
SVO_MODELS = f"{SVO_INDEX}?models=Grid"
SVO_SPECTRUM = "# teff = 5000\n# logg = 4.5\n" + "".join(f"{wl} 1.0e-10\n" for wl in range(1000, 1100))
FID_CACHE = Path(".cache") / "svo_fids" / "Grid.json"
MSG_INDEX = "https://madstar.richtownsend.dev/static.php?ref=msg-grids"  # the wisc.edu page redirects here (Nov 2026)
MSG_FILE = "https://madstar.richtownsend.dev/resource/download/msg/grids/Test/sg-Test.h5"
GRABBERS = {"njm": lambda root: njm.NJMSpectraGrabber(base_dir=root),
            "svo": lambda root: svo.SVOSpectraGrabber(base_dir=root),
            "msg": lambda root: msg.MSGSpectraGrabber(base_dir=root, max_workers=1)}


def _grabber(t, source, routes=None):
    """``source``'s spectra grabber (njm, svo or msg) in the row's folder, its sessions answering ``routes``; and
    the session."""
    session = Routes(routes or {})
    t.mp.setattr(requests, "Session", lambda: session)
    return GRABBERS[source](str(t.tmp)), session


def warnings_from(caplog, module):
    return [r.getMessage() for r in caplog.records
            if r.name == f"sed_tools.{module}" and r.levelno >= logging.WARNING]


@row
def lookup_table_leaves_out_parameters_no_spectrum_states(t):
    grid = t.tmp / "bbody"
    for teff in (3000, 4000):
        _write(grid / f"bb_{teff}.txt", f"# teff = {teff}\n", (1000, 2000), ("1.0", "2.0"), "{} {}\n")
    regenerate_lookup_table(str(grid))
    assert (grid / "lookup_table.csv").read_text().splitlines() == [
        "#file_name, teff", "bb_3000.txt,3000.0", "bb_4000.txt,4000.0"]
    assert "no logg column" in t.capsys.readouterr().out


@row
def lookup_table_raises_when_only_some_spectra_state_a_parameter(t):
    grid = t.tmp / "mixed"
    for name, header in (("a.txt", "# teff = 3000\n# logg = 4.0\n"), ("b.txt", "# teff = 4000\n"),
                         ("c.txt", "# teff = 5000\n# logg = n/a\n")):
        _write(grid / name, header, (1000, 2000), ("1.0", "2.0"), "{} {}\n")
    with pytest.raises(ValueError, match="2 of 3 spectra .* do not state logg in their headers: b.txt, c.txt"):
        regenerate_lookup_table(str(grid))
    assert not (grid / "lookup_table.csv").exists()
    (t.tmp / "empty").mkdir()
    with pytest.raises(RuntimeError, match="No spectra files found for model 'empty'"):
        regenerate_lookup_table(str(t.tmp / "empty"))


@row
def njm_unreachable_warns_with_every_attempt_and_named_grid_requests_raise(t):
    down = requests.ConnectionError("connection refused")
    with t.caplog.at_level(logging.WARNING):
        grabber, session = _grabber(t, "njm", {INDEX: down, NJM: down, INDEX.replace("https", "http"): down})
    assert not grabber.is_available()
    [message] = warnings_from(t.caplog, "grids.njm")
    # No answer means the mirror is down: one attempt, not three (a silent outage cost 26 s).
    assert "unreachable" in message and message.count("connection refused") == 1 and len(session.calls) == 1
    assert grabber.discover_models() == []
    with pytest.raises(RuntimeError, match="unreachable"):
        grabber.get_model_metadata("Grid")
    with pytest.raises(RuntimeError, match="unreachable"):
        grabber.download_model_spectra("Grid", {"txt_files": ["a.txt"]})


@row
def njm_dropping_certificate_checks_is_warned(t):
    def index(verify=True, **kwargs):
        return requests.exceptions.SSLError("certificate expired") if verify else json.dumps({"models": ["Grid"]})

    with t.caplog.at_level(logging.WARNING):
        grabber, session = _grabber(t, "njm", {INDEX: index})
    assert grabber.is_available() and session.verify is False
    assert any("certificate expired" in m for m in warnings_from(t.caplog, "grids.njm"))


@row
def njm_discovery_asks_for_the_index_again_when_its_first_answer_was_not_json(t):
    replies = iter(["not json", json.dumps({"models": ["Grid", ".hidden"]})])
    session = Routes({INDEX: lambda **kw: next(replies)})
    t.mp.setattr(requests, "Session", lambda: session)
    t.mp.setattr("sed_tools.cube.model.STELLAR_DIR_DEFAULT", t.tmp / "default")
    grabber = njm.NJMSpectraGrabber()  # in the default folder
    assert grabber.base_dir == str(t.tmp / "default") and (t.tmp / "default").is_dir()
    assert grabber.discover_models() == ["Grid"]
    assert session.calls == [("GET", INDEX), ("GET", INDEX)]


@row
def njm_grid_info_gives_the_ranges_of_the_values_the_lookup_table_knows(t):
    # c.txt's Teff is not a number and its row has no [M/H]
    grabber, _ = _grabber(t, "njm", {**ONLINE, GRID_URL + "lookup_table.csv": LOOKUP + "c.txt,n/a,5.0\n"})
    assert grabber.get_grid_info("Grid") == {
        "n_spectra": 3, "teff_min": 5000.0, "teff_max": 6000.0, "teff_unique": 2,
        "logg_min": 4.0, "logg_max": 5.0, "logg_unique": 3, "meta_min": 0.0, "meta_max": 0.0, "meta_unique": 1}


@row
def njm_metadata_falls_back_to_lookup_table(t):
    grabber, _ = _grabber(t, "njm", {**ONLINE, GRID_URL: 403, GRID_URL + "lookup_table.csv": LOOKUP})
    meta = grabber.get_model_metadata("Grid")
    assert meta["txt_files"] == ["a.txt", "b.txt"]
    assert meta["csv_files"] == ["lookup_table.csv"] and meta["bin_files"] == []


@row
def njm_parameter_cut_without_lookup_raises_instead_of_downloading_everything(t):
    grabber, session = _grabber(t, "njm", ONLINE)
    metadata = {"model_url": GRID_URL, "txt_files": ["a.txt", "b.txt"]}
    with pytest.raises(RuntimeError, match="lookup_table.csv"):
        grabber.download_model_spectra("Grid", metadata, teff_range=(4000, 5500))
    assert not any(url.endswith(".txt") for _, url in session.calls)


class Dropped(Resp):
    """A download whose connection drops after its first 2000 bytes."""

    def iter_content(self, chunk_size=1):
        yield self.content[:2000]
        raise requests.exceptions.ChunkedEncodingError("connection dropped")


@row
def njm_failed_downloads_raise_after_the_rest(t):
    grabber, _ = _grabber(t, "njm", {**ONLINE, GRID_URL + "a.txt": SPECTRUM, GRID_URL + "b.txt": Dropped(SPECTRUM),
                                     GRID_URL + "d.txt": "", GRID_URL + "lookup_table.csv": LOOKUP})
    metadata = {"model_url": GRID_URL, "txt_files": ["a.txt", "b.txt", "c.txt", "d.txt"],
                "csv_files": ["lookup_table.csv"], "bin_files": ["flux_cube.bin"]}
    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("Grid", metadata)
    message = str(err.value)
    assert "4 file(s)" in message and "b.txt: connection dropped" in message and "c.txt: 404" in message
    assert f"d.txt: {GRID_URL}d.txt returned an empty file" in message and "flux_cube.bin: 404" in message
    model_dir = t.tmp / "Grid"
    assert (model_dir / "a.txt").exists() and (model_dir / "lookup_table.csv").exists()
    assert not (model_dir / "b.txt").exists()  # partial file removed
    assert not (model_dir / "d.txt").exists()  # empty file removed


@row
def njm_split_grid_raises_naming_the_variant_file_that_failed(t):
    index = "variant_name,alpha,n_spectra,path\nalpha_0.0,0.0,1,alpha_0.0\nalpha_0.4,0.4,1,alpha_0.4\n"
    routes = {**ONLINE, GRID_URL + "a.txt": SPECTRUM, GRID_URL + "b.txt": SPECTRUM, GRID_URL + "variants_index.csv": index}
    for name, alpha in (("a.txt", "0.0"), ("b.txt", "0.4")):
        routes[f"{GRID_URL}alpha_{alpha}/lookup_table.csv"] = f"#file_name,teff,alpha\n{name},5000,{alpha}\n"
    routes[GRID_URL + "alpha_0.0/flux_cube.bin"] = b"\0" * 2000  # alpha_0.4's cube is missing
    grabber, _ = _grabber(t, "njm", routes)
    metadata = {"model_url": GRID_URL, "txt_files": ["a.txt", "b.txt"], "csv_files": ["variants_index.csv"],
                "bin_files": ["flux_cube.bin"]}
    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("Grid", metadata)
    assert "1 file(s)" in str(err.value) and "alpha_0.4/flux_cube.bin: 404" in str(err.value)
    variant = t.tmp / "Grid" / "alpha_0.0"
    assert (variant / "flux_cube.bin").is_file() and os.readlink(variant / "a.txt") == os.path.join("..", "a.txt")
    assert not (t.tmp / "Grid" / "flux_cube.bin").exists()  # a split grid's top cube is built, not downloaded


@row
def njm_wavelength_cut_failures_are_raised(t):
    grabber, _ = _grabber(t, "njm", {**ONLINE, GRID_URL + "a.txt": SPECTRUM, GRID_URL + "b.txt": SPECTRUM})
    real_trim = njm._trim_spectrum_file

    def trim(path, wl_range):
        if path.endswith("b.txt"):
            raise PermissionError("read-only")
        return real_trim(path, wl_range)

    t.mp.setattr(njm, "_trim_spectrum_file", trim)
    metadata = {"model_url": GRID_URL, "txt_files": ["a.txt", "b.txt", "c.txt"]}  # c.txt fails to download
    with pytest.raises(RuntimeError, match="b.txt: wavelength cut failed: read-only") as err:
        grabber.download_model_spectra("Grid", metadata, wl_range=(1100, 1200))
    assert "2 file(s)" in str(err.value) and "c.txt: 404" in str(err.value)
    wl, _ = read_text_spectrum(t.tmp / "Grid" / "a.txt")
    assert wl.min() == 1100 and wl.max() == 1200


@row
def njm_cuts_keep_spectra_the_lookup_table_does_not_list_and_rows_they_cannot_read(t):
    a_txt = SPECTRUM + "abc 1.0\n"
    grabber, _ = _grabber(t, "njm", {**ONLINE, GRID_URL + "lookup_table.csv": LOOKUP, GRID_URL + "a.txt": a_txt,
                                     GRID_URL + "z.txt": SPECTRUM})
    metadata = {"model_url": GRID_URL, "txt_files": ["a.txt", "b.txt", "z.txt"]}
    assert grabber.download_model_spectra("Grid", metadata, teff_range=(4000, 5500), wl_range=(0, 1e5)) == 2
    assert (t.tmp / "Grid" / "a.txt").read_text() == a_txt  # within the range, and its last row kept as it was
    assert sorted(p.name for p in (t.tmp / "Grid").iterdir()) == ["a.txt", "z.txt"]  # b.txt (6000 K) cut
    out = t.capsys.readouterr().out
    assert "Axis cut: 2 kept, 1 removed (1 not in lookup — kept)" in out
    assert "All spectra already within requested range" in out
    assert grabber.download_model_spectra("Grid", {"model_url": GRID_URL}) == 0
    assert "[njm] No files to download for Grid" in t.capsys.readouterr().out


def svo_spectrum_url(fid):
    return f"{SSAP}?model=Grid&fid={fid}&format=ascii"


@row
def svo_metadata_warns_once_before_probing_and_raises_when_a_probe_fails(t):
    grabber, _ = _grabber(t, "svo", {SSAP_QUERY: 503, SVO_MODELS: requests.ConnectionError("index down"),
                                     svo_spectrum_url(7): requests.ConnectionError("probe refused")})
    with t.caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="FID 7 of Grid: probe refused"):
        grabber.get_model_metadata("Grid")
    messages = warnings_from(t.caplog, "grids.svo")
    [message] = messages  # one warning for the fallback, naming both reasons
    assert "HTTP 503" in message and "index down" in message and "probing" in message
    assert not (t.tmp / FID_CACHE).exists()


@row
def svo_discovery_reads_grid_names_from_links_then_from_the_page_text(t):
    grabber, session = _grabber(t, "svo", {SVO_INDEX: '<a href="index.php?models=bt-settl&fmt=html">BT-Settl</a>'
                                                      '<a href="index.php?models=">none</a><a href="about.html">x</a>'})
    assert grabber.discover_models() == ["bt-settl"]
    session.files[SVO_INDEX] = "<script>var page = 'index.php?models=Kurucz2003';</script>"
    assert grabber.discover_models() == ["Kurucz2003"]


@row
def svo_probe_finds_the_spectra_neither_list_gives(t):
    spectrum = Resp()
    spectrum.headers["content-length"] = "2048"  # HEAD of a spectrum: its size, no body
    grabber, session = _grabber(t, "svo", {SSAP_QUERY: requests.ConnectionError("SSAP down"), SVO_MODELS: 404,
                                           svo_spectrum_url(5): spectrum})
    with t.caplog.at_level(logging.WARNING):
        assert grabber.get_model_metadata("Grid") == [{"fid": 5}]
    [message] = warnings_from(t.caplog, "grids.svo")
    assert "SSAP down" in message and "HTTP 404" in message and "probing FIDs directly" in message
    heads = {url for method, url in session.calls if method == "HEAD"}
    assert {svo_spectrum_url(fid) for fid in (1, 4, 6, 10000)} <= heads  # the probes, and around the hit
    assert json.loads((t.tmp / FID_CACHE).read_text()) == [5]


@row
def svo_probe_closes_the_get_it_streams(t):
    class Streamed(Resp):
        closed = False

        def __exit__(self, *exc):
            self.closed = True
            return False

    spectrum, error_page = Streamed(SVO_SPECTRUM), Streamed(b"", 404)
    grabber, session = _grabber(t, "svo")
    session.get = lambda url, params=None, **kw: {1: spectrum, 2: error_page}[params["fid"]]
    session.head = lambda url, params=None, **kw: Resp()  # HEAD with no Content-Length
    assert grabber._head_exists("Grid", 1) is True and spectrum.closed
    assert grabber._head_exists("Grid", 2) is False and error_page.closed


@row
def svo_metadata_raises_when_no_method_finds_spectra(t):
    grabber, _ = _grabber(t, "svo", {SSAP_QUERY: "<VOTABLE/>"})
    with pytest.raises(RuntimeError, match="no spectra for Grid: SSAP query: the response lists none"):
        grabber.get_model_metadata("Grid")


@row
def svo_truncated_ssap_response_raises_and_is_not_cached(t):
    grabber, _ = _grabber(t, "svo", {SSAP_QUERY: "".join(f"<TD>ssap.php?fid={i}</TD>" for i in range(1, 200))})
    grabber.ssap_max_bytes = 1000
    with pytest.raises(RuntimeError, match="cap"):
        grabber.get_model_metadata("Grid")
    assert not (t.tmp / FID_CACHE).exists()


@row
def svo_metadata_is_cached_and_a_corrupt_cache_is_warned_about(t):
    grabber, session = _grabber(t, "svo", {SSAP_QUERY: "<TD>ssap.php?fid=2</TD><TD>ssap.php?fid=1</TD>"})
    cache = t.tmp / FID_CACHE
    assert grabber.get_model_metadata("Grid") == [{"fid": 1}, {"fid": 2}]
    assert json.loads(cache.read_text()) == [1, 2]
    session.calls.clear()
    assert grabber.get_model_metadata("Grid") == [{"fid": 1}, {"fid": 2}]
    assert session.calls == []
    cache.write_text("[1, 2")
    with t.caplog.at_level(logging.WARNING):
        assert grabber.get_model_metadata("Grid") == [{"fid": 1}, {"fid": 2}]
    assert any("unreadable FID cache" in m for m in warnings_from(t.caplog, "grids.svo"))


@row
def svo_download_refuses_ranges_it_cannot_apply(t):
    grabber, session = _grabber(t, "svo")
    with pytest.raises(ValueError, match="cannot be cut"):
        grabber.download_model_spectra("Grid", [{"fid": 1}], teff_range=(4000, 5000))
    assert session.calls == []


@row
def svo_failed_downloads_raise_and_leave_no_lookup_table(t):
    grabber, _ = _grabber(t, "svo", {svo_spectrum_url(1): SVO_SPECTRUM,
                                     svo_spectrum_url(2): "<html>Error</html>" + " " * 600,
                                     svo_spectrum_url(3): requests.ConnectionError("reset by peer"),
                                     svo_spectrum_url(4): "1000 1.0\n"})
    model_dir = t.tmp / "Grid"
    model_dir.mkdir()
    (model_dir / "Grid_fid3.txt").write_text("# teff = 5000\n1000 1.0\n")  # part of an earlier download
    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("Grid", [{"fid": 1}, {"fid": 2}, {"fid": 3}, {"fid": 4}])
    message = str(err.value)
    assert "3 of 4 spectra" in message
    assert "FID 2: content is not spectral data" in message and "FID 3: reset by peer" in message
    assert "FID 4: file too small" in message
    assert (model_dir / "Grid_fid1.txt").exists()
    assert not (model_dir / "Grid_fid3.txt").exists()  # the earlier part is not taken for the spectrum
    assert not (model_dir / "lookup_table.csv").exists()


@row
def svo_complete_download_writes_lookup_table(t):
    grabber, session = _grabber(t, "svo", {svo_spectrum_url(1): SVO_SPECTRUM, svo_spectrum_url(2): SVO_SPECTRUM})
    assert grabber.download_model_spectra("Grid", []) == 0
    model_dir = t.tmp / "Grid"
    (model_dir / "Grid_fid3.txt").write_text(SVO_SPECTRUM)  # downloaded earlier: kept
    (model_dir / "Grid_fid2.txt").write_text("<html>Error</html>" + " " * 2000)  # an error page kept earlier
    assert grabber.download_model_spectra("Grid", [{"fid": 1}, {"fid": 2}, {"fid": 3}]) == 3
    assert sorted(url for _, url in session.calls) == [svo_spectrum_url(1), svo_spectrum_url(2)]
    assert (model_dir / "Grid_fid2.txt").read_text() == SVO_SPECTRUM
    lookup = (model_dir / "lookup_table.csv").read_text().splitlines()
    assert lookup[0] == "#file_name,logg,metallicity,teff" and len(lookup) == 4


@row
def svo_and_msg_lookup_rows_are_in_file_name_order(t):
    # The rows came in the order the threaded downloads finished, so the same download wrote different files.
    rows = [{"file_name": f"Grid_fid{fid}.txt", "teff": fid} for fid in (2, 10, 1)]
    for source in ("svo", "msg"):
        grabber, _ = _grabber(t, source)
        grabber._write_lookup(str(t.tmp), rows)
        names = [line.split(",")[0] for line in (t.tmp / "lookup_table.csv").read_text().splitlines()[1:]]
        assert names == ["Grid_fid1.txt", "Grid_fid10.txt", "Grid_fid2.txt"]


# A 2 x 3 x 2 x 2 grid laid out as MSG writes it (src/vgrid/vgrid_io_sm.fypp), C/O first as in sg-SPHINX. Its
# spectra specints[1..3] sit at the 1-based, Fortran-order (first axis fastest) linear vertices 2, 9 and 23: the last
# is below the grid's 24 vertices, as C3K's and Coelho14's last vertices are below theirs.
MSG_AXES = (("C/O", [0.5, 0.9]), ("Teff", [3000.0, 3500.0, 4000.0]), ("[Fe/H]", [-1.0, 0.0]), ("log(g)", [4.0, 5.0]))
MSG_VERTICES = [2, 9, 23]
#: Each spectrum's parameters, worked by hand: Teff, log g and [Fe/H] as teff, logg and meta, C/O under its label
MSG_PARAMETERS = [{"teff": 3000.0, "logg": 4.0, "meta": -1.0, "C/O": 0.9},
                  {"teff": 3500.0, "logg": 4.0, "meta": 0.0, "C/O": 0.5},
                  {"teff": 4000.0, "logg": 5.0, "meta": 0.0, "C/O": 0.5}]


def _msg_vgrid(f, revision=3):
    """MSG_AXES and MSG_VERTICES as a vgrid group of *revision*: 1 names the axes 'axes(r)' and 2 'axes[r]', both
    with a 'mask' over the vertices; 3 names them 'axes[r]' and lists the vertices in 'v_lin_seq'. Revision 1 labels
    the metallicity 'Z/Zo', as the revision-1 grids (OSTAR2002, BSTAR2006) do."""
    vgrid = f.create_group("vgrid")
    vgrid.attrs.update({"TYPE": np.bytes_("vgrid_t"), "REVISION": revision, "rank": len(MSG_AXES)})
    for r, (label, x) in enumerate(MSG_AXES, 1):
        axis = vgrid.create_group(f"axes({r})" if revision == 1 else f"axes[{r}]")
        axis.attrs["label"] = np.bytes_("Z/Zo" if revision == 1 and label == "[Fe/H]" else label)
        axis["x"] = x
    if revision == 3:
        vgrid["v_lin_seq"] = MSG_VERTICES
    else:
        vgrid["mask"] = np.isin(np.arange(1, 25), MSG_VERTICES).astype(np.int32)


def _msg_range(group, kind, **values):
    """A wavelength range group of MSG's *kind* (lin_range_t, ...): attributes, or for tab_range_t the dataset x;
    comp_range_t takes sub-ranges=[(kind, values), ...]."""
    group.attrs.update({"TYPE": np.bytes_(kind), "REVISION": 1})
    if kind == "tab_range_t":
        group["x"] = values["x"]
    elif kind == "comp_range_t":
        group.attrs["n_ranges"] = len(values["sub_ranges"])
        for j, (sub_kind, sub_values) in enumerate(values["sub_ranges"], 1):
            _msg_range(group.create_group(f"ranges[{j}]"), sub_kind, **sub_values)
    else:
        group.attrs.update(values)


def _msg_spectrum(f, number, c, law="CONST", kind="lin_range_t", range_values=None):
    """specsource/specints[number] as MSG writes a limb_specint_t: coefficients c (one row per bin), the wavelength
    range (by default 50 bins from 3000 A in steps of 120 A) and the limb-darkening law."""
    spectrum = f.create_group(f"specsource/specints[{number}]")
    spectrum.attrs["TYPE"] = np.bytes_("limb_specint_t")
    spectrum["c"] = np.asarray(c, dtype=float).reshape(len(c), -1)
    _msg_range(spectrum.create_group("range"), kind, **(range_values or {"x_0": 3000.0, "dx": 120.0, "n": 51}))
    spectrum.create_group("limb").attrs["law"] = np.bytes_(law)


def _msg_grid(t):
    """A grabber whose sg-Test.h5 holds three spectra: one good, one without its coefficients 'c', one with a 'c' of
    one row fewer than its bins."""
    grabber, _ = _grabber(t, "msg")
    grabber.model_urls["sg-Test"] = MSG_FILE
    (t.tmp / "sg-Test").mkdir()
    with h5py.File(t.tmp / "sg-Test" / "sg-Test.h5", "w") as f:
        _msg_vgrid(f)
        _msg_spectrum(f, 1, np.ones(50))
        _msg_spectrum(f, 2, np.ones(50))
        del f["specsource/specints[2]/c"]
        _msg_spectrum(f, 3, np.ones(49))
    return grabber, t.tmp / "sg-Test"


@row
def msg_metadata_raises_for_unknown_grid_and_failed_download_then_downloads_the_grid(t):
    grabber, session = _grabber(t, "msg", {MSG_INDEX: f'<a href="{MSG_FILE}">sg-Test</a>',
                                           MSG_FILE: requests.ConnectionError("dropped")})
    with pytest.raises(RuntimeError, match="unknown to this grabber"):
        grabber.get_model_metadata("sg-Test")
    assert grabber.discover_models() == ["sg-Test"]
    with pytest.raises(RuntimeError, match="could not download .*sg-Test.h5: dropped"):
        grabber.get_model_metadata("sg-Test")
    served = t.tmp / "served.h5"
    with h5py.File(served, "w") as f:
        _msg_vgrid(f)
        for n in (1, 2, 3):
            _msg_spectrum(f, n, np.ones(50))
    session.files[MSG_FILE] = served.read_bytes()
    assert [s["params"] for s in grabber.get_model_metadata("sg-Test")] == MSG_PARAMETERS
    assert (t.tmp / "sg-Test" / "sg-Test.h5").read_bytes() == served.read_bytes()


@row
def msg_discovery_takes_the_newest_version_of_a_grid(t):
    # MSG's index links a grid's old versions beside its current one, and discovery kept the last link, an old
    # version (CAP18 v1, Goettingen v2; review 24 Nov 2026). Nill, 25 Nov: "pick the newest".
    base = MSG_FILE.rsplit("/", 2)[0]
    page = "".join(f'<tr><td><a href="{base}/{current}/sg-{name}.h5">sg-{name}.h5</a></td><td>v{n}</td><td>'
                   + "".join(f'<a href="{base}/{name}/v{k}/sg-{name}.h5">v{k}</a>' for k in old) + "</td></tr>"
                   for name, current, n, old in (("A", "A", 2, [1]), ("B", "B/v3", 3, [2, 1]), ("C", "C", 1, [])))
    grabber, _ = _grabber(t, "msg", {MSG_INDEX: f"<table>{page}</table>"})
    assert grabber.discover_models() == ["sg-A", "sg-B", "sg-C"]
    assert grabber.model_urls == {"sg-A": f"{base}/A/sg-A.h5", "sg-B": f"{base}/B/v3/sg-B.h5",
                                  "sg-C": f"{base}/C/sg-C.h5"}


@row
def msg_metadata_raises_for_unreadable_or_empty_file(t):
    grabber, _ = _grabber(t, "msg")
    grabber.model_urls["sg-Test"] = MSG_FILE
    h5_path = t.tmp / "sg-Test" / "sg-Test.h5"
    h5_path.parent.mkdir()
    h5_path.write_bytes(b"\x89HDF partial download")
    with pytest.raises(RuntimeError, match="could not read .*partial"):
        grabber.get_model_metadata("sg-Test")
    with h5py.File(h5_path, "w") as f:
        f.create_group("other")
    with pytest.raises(RuntimeError, match="no 'specsource' group"):
        grabber.get_model_metadata("sg-Test")
    with h5py.File(h5_path, "w") as f:
        f.create_group("specsource")
    with pytest.raises(RuntimeError, match=r"holds no spectra \('specints\[...\]' groups\)"):
        grabber.get_model_metadata("sg-Test")


@row
def msg_failed_extractions_raise_after_the_rest(t):
    grabber, model_dir = _msg_grid(t)
    spectra = grabber.get_model_metadata("sg-Test")
    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("sg-Test", spectra)
    message = str(err.value)
    assert "2 of 3 spectra" in message
    assert "sg-Test_fid2.txt: specsource/specints[2]: " in message and "'c' doesn't exist" in message
    assert "sg-Test_fid3.txt: specsource/specints[3]: c has shape (49, 1), not (50, 1)" in message
    wl, _ = read_text_spectrum(model_dir / "sg-Test_fid1.txt")
    assert wl.size == 50
    assert not (model_dir / "lookup_table.csv").exists()


@row
def msg_cut_exclusions_are_not_failures(t):
    grabber, model_dir = _msg_grid(t)
    spectra = grabber.get_model_metadata("sg-Test")[:1]
    assert grabber.download_model_spectra("sg-Test", []) == 0
    assert grabber.download_model_spectra("sg-Test", spectra, teff_range=(3200, 4000)) == 0
    assert grabber.download_model_spectra("sg-Test", spectra, wl_range=(1.0, 2.0)) == 0
    assert not list(model_dir.glob("*.txt"))
    # A metallicity cut keeps a spectrum whose metallicity is not known
    unknown = [{**spectra[0], "params": {k: v for k, v in spectra[0]["params"].items() if k != "meta"}}]
    assert grabber.download_model_spectra("sg-Test", unknown, meta_range=(5.0, 6.0), wl_range=(3000, 4000)) == 1
    wl, _ = read_text_spectrum(model_dir / "sg-Test_fid1.txt")
    np.testing.assert_allclose(wl, np.arange(3060.0, 4000.0, 120.0))
    assert grabber.download_model_spectra("sg-Test", spectra, teff_range=(2500, 3200)) == 1
    assert (model_dir / "lookup_table.csv").exists()


@row
def an_msg_spectrum_whose_writing_stopped_is_extracted_again(t):
    # Spectra were written in place, and any file over 1 KB counted as extracted: one whose writing stopped part way
    # (a full disk, a killed run) was reused on the next run, its flux zero in the cube beyond where it stopped.
    import builtins

    grabber, model_dir = _msg_grid(t)
    spectrum = grabber.get_model_metadata("sg-Test")[0]
    extract = lambda: grabber._extract_to_text("sg-Test", str(model_dir / "sg-Test.h5"), str(model_dir), spectrum,
                                               dict.fromkeys(("teff", "logg", "meta", "wl")))

    class FullDisk:
        """A file that takes 1500 bytes, then fails as a full disk does."""
        def __init__(self, fh):
            self.fh, self.written = fh, 0

        def write(self, text):
            self.written += len(text)
            if self.written > 1500:
                raise OSError(28, "No space left on device")
            return self.fh.write(text)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.fh.close()

    t.mp.setattr(msg, "open", lambda *a, **k: FullDisk(builtins.open(*a, **k)), raising=False)
    with pytest.raises(OSError, match="No space left"):
        extract()
    t.mp.setattr(msg, "open", builtins.open, raising=False)
    status, name, _ = extract()
    assert status == "ok" and read_text_spectrum(model_dir / name)[0].size == 50
    assert sorted(p.name for p in model_dir.iterdir()) == ["sg-Test.h5", name]


def _msg_vgrid_read(t, revision, departure, error):
    grabber, _ = _grabber(t, "msg")
    grabber.model_urls["sg-Test"] = MSG_FILE
    (t.tmp / "sg-Test").mkdir()
    with h5py.File(t.tmp / "sg-Test" / "sg-Test.h5", "w") as f:
        _msg_vgrid(f, revision)
        for n in (1, 2, 3):
            _msg_spectrum(f, n, np.ones(50))
        if departure:
            departure(f)
    if error:
        with pytest.raises(RuntimeError, match=error):
            grabber.get_model_metadata("sg-Test")
        return
    spectra = grabber.get_model_metadata("sg-Test")
    assert [s["params"] for s in spectra] == MSG_PARAMETERS
    for _ in range(2):  # the second run finds the spectra on disk: their lookup rows keep their parameters
        assert grabber.download_model_spectra("sg-Test", spectra) == 3
        assert (t.tmp / "sg-Test" / "lookup_table.csv").read_text().splitlines() == ["file_name,teff,logg,meta,C/O"] + [
            f"sg-Test_fid{n}.txt," + ",".join(map(str, p.values())) for n, p in enumerate(MSG_PARAMETERS, 1)]
    assert "# meta = -1.0\n# C/O = 0.9\n" in (t.tmp / "sg-Test" / "sg-Test_fid1.txt").read_text()


def _msg_axes_from_0(f):
    for r in range(1, 5):
        f.move(f"vgrid/axes[{r}]", f"vgrid/axes[{r - 1}]")


def _msg_replace(path, data):
    def departure(f):
        del f[path]
        f[path] = data
    return departure


#: name: (MSG vgrid revision, how the file departs from MSG's layout or None, the error that stops its download or
#: None). Until 13 Nov 2026 the reader looked for axes from axes[0], so it read none from any MSG file, and every MSG
#: spectrum had NaN Teff, logg and [M/H]; it also took v_lin_seq as 0-based unless its maximum was the grid size, and
#: tried C order before Fortran order. Grids whose spectra lack parameters now fail before anything is extracted.
MSG_VGRIDS = {
    "msg_revision_3_reads_axes_from_1_and_v_lin_seq_in_fortran_order": (3, None, None),
    "msg_revision_2_reads_axes_from_1_and_the_mask": (2, None, None),
    "msg_revision_1_reads_parenthesised_axes_and_the_mask": (1, None, None),
    "msg_axes_numbered_from_0_fail": (3, _msg_axes_from_0, r"no vgrid group laid out as MSG writes it \(.*axes\[4\]"),
    "msg_v_lin_seq_from_0_fails": (3, _msg_replace("vgrid/v_lin_seq", [0, 8, 22]), "index -1 is out of bounds"),
    "msg_unknown_vgrid_revision_fails": (3, lambda f: f["vgrid"].attrs.__setitem__("REVISION", 4), "revision 4"),
    "msg_axis_without_a_label_fails": (3, lambda f: f["vgrid/axes[3]"].attrs.__delitem__("label"), "axis 3 has no label"),
    "msg_grid_without_a_teff_axis_fails": (
        3, lambda f: f["vgrid/axes[2]"].attrs.__setitem__("label", "T"),
        r"none of the axes of .*\('C/O', 'T', '\[Fe/H\]', 'log\(g\)'\) reads as teff"),
    "msg_spectra_unmatched_by_the_vgrid_fail": (
        3, lambda f: f.__delitem__("specsource/specints[3]"), "places 3 spectra but it holds 2"),
    "msg_spectra_without_a_finite_value_fail": (
        3, _msg_replace("vgrid/axes[4]/x", [4.0, np.nan]), "1 of 3 spectra have no finite logg"),
}


def _msg_spectrum_written(t, c, law, wavelength_range, expected):
    grabber, _ = _grabber(t, "msg")
    grabber.model_urls["sg-Test"] = MSG_FILE
    (t.tmp / "sg-Test").mkdir()
    kind, range_values = wavelength_range
    with h5py.File(t.tmp / "sg-Test" / "sg-Test.h5", "w") as f:
        _msg_vgrid(f)
        for n in (1, 2, 3):
            _msg_spectrum(f, n, c, law, kind, range_values)
    spectra = grabber.get_model_metadata("sg-Test")[:1]
    if isinstance(expected, str):
        with pytest.raises(RuntimeError, match=re.escape(f"sg-Test_fid1.txt: specsource/specints[1]: {expected}")):
            grabber.download_model_spectra("sg-Test", spectra)
        assert not (t.tmp / "sg-Test" / "sg-Test_fid1.txt").exists()
        return
    assert grabber.download_model_spectra("sg-Test", spectra) == 1
    written = t.tmp / "sg-Test" / "sg-Test_fid1.txt"
    wavelength, flux = read_text_spectrum(written)
    np.testing.assert_allclose(wavelength, expected[0])
    np.testing.assert_allclose(flux, expected[1], rtol=1e-8)
    noted = [line for line in written.read_text().splitlines() if line.startswith("# negative_flux_fraction")]
    assert noted == list(expected[2:])  # only a spectrum with negative fluxes says how much they are


CLARET_ROW = [1.0, 0.1, 0.2, 0.3, 0.4]
LIN = ("lin_range_t", {"x_0": 3000.0, "dx": 1000.0, "n": 4})
#: name: (c, limb-darkening law, (range type, its values), the bin-centre wavelengths and fluxes written and any
#: negative_flux_fraction header line, or the reason the spectrum fails). MSG's range points are bin edges and c holds one row per bin
#: (limb_specint_construct_sm.fypp); the flux is 2 pi times c dotted with the law's flux basis (eval_flux_basis,
#: limb_basis_sm.fypp). Until 13 Nov 2026 no lin_range_t, log_range_t or comp_range_t was read, nor a tab_range_t
#: from 0 A, so 24 of the 25 grids extracted nothing, and SPHINX wrote the intensity c as flux: pi times too low.
MSG_SPECTRA = {
    "msg_lin_range_bins_are_written_at_their_centres": (
        [[2.0]] * 3, "CONST", LIN, ([3500.0, 4500.0, 5500.0], [2 * np.pi] * 3)),
    "msg_log_range_is_read": (
        [[1.0], [3.0]], "CONST", ("log_range_t", {"logx_0": np.log(1000.0), "dlogx": np.log(10.0), "n": 3}),
        ([5500.0, 55000.0], [np.pi, 3 * np.pi])),
    "msg_tab_range_is_read_from_0_angstrom_with_zero_flux_kept": (
        [[0.0], [1.0]], "CONST", ("tab_range_t", {"x": [0.0, 2.0, 6.0]}), ([1.0, 4.0], [0.0, np.pi])),
    "msg_comp_range_joins_its_sub_ranges": (
        [[1.0]] * 3, "CONST", ("comp_range_t", {"sub_ranges": [("lin_range_t", {"x_0": 1000.0, "dx": 1000.0, "n": 2}),
                                                               ("tab_range_t", {"x": [3000.0, 5000.0]})]}),
        ([1500.0, 2500.0, 4000.0], [np.pi] * 3)),
    "msg_claret_flux_uses_msgs_flux_basis": (
        [CLARET_ROW] * 3, "CLARET", LIN,
        ([3500.0, 4500.0, 5500.0], [2 * np.pi * (1 / 2 - 0.1 / 10 - 0.2 / 6 - 3 * 0.3 / 14 - 0.4 / 4)] * 3)),
    "msg_c_without_a_row_per_bin_fails": ([[1.0]] * 2, "CONST", LIN, "c has shape (2, 1), not (3, 1)"),
    "msg_unknown_range_type_fails": ([[1.0]] * 3, "CONST", ("spline_range_t", {}), "wavelength range type 'spline_range_t'"),
    "msg_unknown_limb_law_fails": ([[1.0]] * 3, "POWER", LIN, "limb-darkening law 'POWER'"),
    "msg_non_finite_flux_fails": ([[1.0], [np.nan], [1.0]], "CONST", LIN, "1 of 3 fluxes are not finite"),
    # Negative fluxes are kept as MSG gives them while their integral is below 0.1 % of the absolute flux's
    "msg_negative_flux_below_a_tenth_of_a_percent_is_kept_and_noted": (
        [[1.0], [-1e-4], [1.0]], "CONST", LIN,
        ([3500.0, 4500.0, 5500.0], [np.pi, -1e-4 * np.pi, np.pi], "# negative_flux_fraction = 9.999e-05")),
    "msg_negative_flux_of_a_tenth_of_a_percent_or_more_fails": (
        [[1.0], [-2e-3], [1.0]], "CONST", LIN, "its negative fluxes are 0.002 of its absolute flux (limit 0.001)"),
    "msg_spectrum_without_a_positive_flux_fails": ([[0.0]] * 3, "CONST", LIN, "no flux is positive"),
    "msg_non_positive_wavelength_fails": (
        [[1.0]] * 3, "CONST", ("lin_range_t", {"x_0": -2000.0, "dx": 1000.0, "n": 4}),
        "2 of 3 wavelengths are not finite and positive"),
}


BOSZ_SCRIPT = f"{mast.SCRIPTS_DIR}/hlsp_bosz_bosz2024_sim_r1000_m%2B0.00_v1_bulkdl.sh"
BOSZ_WAVE = f"{mast.DATA_BASE}/wavelength_grids/bosz2024_wave_r1000.txt"
WAVE_12 = "\n".join(str(3000 + 100 * i) for i in range(12))
GZ = gzip.compress(b"1 2\n")


def bosz_url(teff):
    return f"{mast.DATA_BASE}/m+0.00/r1000/bosz2024_mp_t{teff:05d}_g+4.5_m+0.00_a+0.00_c+0.00_v2_r1000_resam.txt.gz"


def bosz_resam(n):
    """n rows of (flux, continuum); the flux is not monotonic, so it is not read as wavelengths."""
    return gzip.compress("".join(f"{1.0 + (i % 3)} 2.0\n" for i in range(n)).encode())


def _bosz(t, spectra, **kw):
    """A MAST grabber in the row's folder whose r1000 [M/H] = +0.00 script lists ``spectra`` {url: reply}."""
    routes = {BOSZ_SCRIPT: "\n".join(f"curl -O {u}" for u in spectra), BOSZ_WAVE: WAVE_12, **spectra}
    return mast.MASTSpectraGrabber(base_dir=str(t.tmp), session=Routes(routes), **kw)


@row
def mast_download_text_distinguishes_missing_from_failed(t):
    session = Routes({"https://x/missing": 404, "https://x/broken": 503})
    assert mast._download_text("https://x/missing", session=session) == ""
    with pytest.raises(requests.HTTPError):
        mast._download_text("https://x/broken", session=session)


@row
def mast_gz_download_raises_for_corrupt_gzip_instead_of_decoding_it(t):
    session = Routes({"https://x/plain": b"1 2\n", "https://x/good": GZ, "https://x/corrupt": GZ[:-6] + b"\x00" * 6,
                      "https://x/gone": 404})
    assert mast._download_text_gz("https://x/plain", session=session) == "1 2\n"
    assert mast._download_text_gz("https://x/good", session=session) == "1 2\n"
    with pytest.raises(ValueError, match="corrupt gzip"):
        mast._download_text_gz("https://x/corrupt", session=session)
    with pytest.raises(requests.HTTPError):
        mast._download_text_gz("https://x/gone", session=session)


@row
def mast_missing_wavelength_grid_raises_listing_each_location(t):
    with pytest.raises(RuntimeError, match="No BOSZ wavelength grid found for r1000") as err:
        mast._try_wave_urls("r1000", Routes({BOSZ_WAVE: "1\n2\n"}))
    assert "not an increasing wavelength grid (2 values)" in str(err.value)
    assert str(err.value).count("not found") == 4


@row
def mast_unreadable_or_empty_download_scripts_raise(t):
    session = Routes({BOSZ_SCRIPT: requests.ConnectionError("timed out")})
    with pytest.raises(RuntimeError, match="metallicities would be missing:\n  .*timed out"):
        mast._gather_urls_from_scripts("r1000", ["m+0.00"], session)
    with pytest.raises(RuntimeError, match="not found"):
        mast._gather_urls_from_scripts("r1000", ["m+0.00"], Routes({}))
    with pytest.raises(RuntimeError, match="The BOSZ download scripts for r1000 list no spectra"):
        mast._gather_urls_from_scripts("r1000", ["m+0.00"], Routes({BOSZ_SCRIPT: "#!/bin/sh\n"}))


@row
def mast_asks_for_metallicities_only_at_a_terminal_and_without_meta_range(t):
    # It asked in get_model_metadata, before the cuts: --meta did not stop it, and without a terminal the
    # download failed on EOF, leaving an empty grid folder.
    grabber = _bosz(t, {bosz_url(5750): bosz_resam(12)})
    t.mp.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    answer(t.mp)  # a question fails the row
    info = grabber.get_model_metadata("BOSZ-2024-r1000")
    assert grabber.download_model_spectra("BOSZ-2024-r1000", info, meta_range=(0.0, 0.0)) == 1
    answer(t.mp, "-1.0, +0.25,")
    assert mast._ask_metals(info["metals"]) == ["m-1.00", "m+0.25"]
    t.mp.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    answer(t.mp)
    assert mast._ask_metals(info["metals"]) == info["metals"] == mast.METALS


@row
def mast_failed_spectra_raise_after_the_rest(t):
    grabber = _bosz(t, {bosz_url(5750): bosz_resam(12), bosz_url(6000): 404, bosz_url(6250): bosz_resam(11)},
                    max_workers=1)
    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("BOSZ-2024-r1000", {"reskey": "r1000", "metals": ["m+0.00"]})
    message = str(err.value)
    assert "2 of 3 BOSZ spectra" in message
    assert "t06000" in message and "404" in message and "length mismatch" in message
    model_dir = t.tmp / "BOSZ-2024-r1000"
    assert [p.name for p in model_dir.glob("*.txt")] == [
        "bosz2024_mp_t05750_g_4.5_m_0.00_a_0.00_c_0.00_v2_r1000_resam.txt"]  # '+' sanitised to '_'
    assert not (model_dir / "lookup_table.csv").exists()


@row
def mast_cuts_that_leave_nothing_are_not_failures(t):
    grabber = _bosz(t, {bosz_url(5750): bosz_resam(12)})
    info = {"reskey": "r1000", "metals": ["m+0.00"]}
    assert grabber.download_model_spectra("BOSZ-2024-r1000", info, wl_range=(9000, 9500)) == 0
    assert grabber.download_model_spectra("BOSZ-2024-r1000", info, teff_range=(7000, 8000), meta_range=(0, 0)) == 0
    # A metallicity tag that is not BOSZ's ('solar') matches no range
    assert grabber.download_model_spectra("BOSZ-2024-r1000", {"reskey": "r1000", "metals": ["m+0.00", "solar"]},
                                          meta_range=(5.0, 6.0)) == 0
    out = t.capsys.readouterr().out
    assert ("[MAST] No BOSZ spectra URLs found for r1000 with teff_range=(7000, 8000), logg_range=None, "
            "meta_range=(0, 0)") in out
    assert "[MAST] No BOSZ metallicities match meta_range=(5.0, 6.0)" in out
    assert not list((t.tmp / "BOSZ-2024-r1000").glob("*.txt"))
    with pytest.raises(ValueError, match="Unrecognized BOSZ model name: BOSZ-2023-r1000"):
        grabber.get_model_metadata("BOSZ-2023-r1000")


@row
def mast_reports_each_spectrum_it_cannot_read_and_why(t):
    odd = f"{mast.DATA_BASE}/m+0.00/r1000/bosz2024_xx_t05750_r1000_resam.txt.gz"
    unreadable = "# no data\n\nflux continuum\n5\n"  # the first file: its first column is no wavelength grid
    grabber = _bosz(t, {bosz_url(5750): unreadable, bosz_url(6000): "", bosz_url(6250): bosz_resam(12),
                        odd: bosz_resam(12)}, max_workers=1)
    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("BOSZ-2024-r1000", {"reskey": "r1000", "metals": ["m+0.00"]})
    message = str(err.value)
    assert "3 of 4 BOSZ spectra" in message
    assert f"{bosz_url(5750)}: no spectrum could be read from the file" in message
    assert f"{bosz_url(6000)}: empty file" in message
    assert f"{odd}: file name does not follow the BOSZ 2024 pattern" in message
    [written] = (t.tmp / "BOSZ-2024-r1000").glob("*.txt")
    assert "t06250" in written.name and read_text_spectrum(written)[0].size == 12


@row
def svo_cache_that_cannot_be_read_or_written_is_warned_about(t):
    grabber, _ = _grabber(t, "svo", {SSAP_QUERY: "<TD>ssap.php?fid=3</TD>"})
    (t.tmp / FID_CACHE).mkdir()  # a directory where the cache file goes
    with t.caplog.at_level(logging.WARNING):
        assert grabber.get_model_metadata("Grid") == [{"fid": 3}]
    messages = warnings_from(t.caplog, "grids.svo")
    assert any("unreadable FID cache" in m for m in messages)
    assert any("could not cache the FID list" in m for m in messages)


@row
def mast_filter_cuts_by_teff_alone_and_keeps_names_it_cannot_parse(t):
    odd = "https://x/m+0.00/r1000/bosz2024_xx_t05750_r1000_resam.txt.gz"
    assert mast._filter_urls_by_params([odd], teff_range=(5000, 6000)) == [odd]  # the download reports it
    assert mast._filter_urls_by_params([bosz_url(5750), bosz_url(6250)], teff_range=(5000, 6000)) == [bosz_url(5750)]


@row
def mast_failed_first_spectrum_does_not_stop_the_rest(t):
    grabber = _bosz(t, {bosz_url(5750): requests.ConnectionError("first one dropped"), bosz_url(6000): bosz_resam(12)},
                    max_workers=1)
    with pytest.raises(RuntimeError, match="1 of 2 BOSZ spectra.*\n.*t05750.*first one dropped"):
        grabber.download_model_spectra("BOSZ-2024-r1000", {"reskey": "r1000", "metals": ["m+0.00"]})
    assert len(list((t.tmp / "BOSZ-2024-r1000").glob("*t06000*.txt"))) == 1


@row
def mast_download_accepts_generic_axis_kwargs(t):
    url = ("https://archive.stsci.edu/hlsps/bosz/bosz2024/m+0.00/r1000/"
           "bosz2024_mp_t05750_g+4.5_m+0.00_a+0.00_c+0.00_v2_r1000_resam.txt.gz")

    def fake_download_text_gz(download_url, session=None):
        assert download_url == url
        return "1000 1.0\n2000 1.0\n3000 1.0\n4000 1.0\n"

    t.mp.setattr("sed_tools.grids.mast._gather_urls_from_scripts", lambda reskey, metals, session: [url])
    t.mp.setattr("sed_tools.grids.mast._try_wave_urls", lambda reskey, session: (None, None))
    t.mp.setattr("sed_tools.grids.mast._download_text_gz", fake_download_text_gz)
    n = mast.MASTSpectraGrabber(base_dir=str(t.tmp)).download_model_spectra(
        "BOSZ-2024-r1000", {"reskey": "r1000", "metals": ["m+0.00"]},
        teff_range=(5700, 5800), logg_range=(4.0, 5.0), meta_range=(-0.1, 0.1), wl_range=(1500, 3500))
    assert n == 1
    model_dir = t.tmp / "BOSZ-2024-r1000"
    spectra = sorted(model_dir.glob("*.txt"))
    assert len(spectra) == 1
    body = [line for line in spectra[0].read_text().splitlines() if not line.startswith("#")]
    assert body[0].startswith("2000.000000 ")
    assert body[-1].startswith("3000.000000 ")
    assert (model_dir / "lookup_table.csv").exists()


@row
def try_wave_urls_uses_official_wavelength_grids_directory(t):
    expected = f"{mast.DATA_BASE}/wavelength_grids/bosz2024_wave_r1000.txt"
    calls = []

    def fake_download_text(url, session=None, timeout=180):
        calls.append(url)
        if url == expected:
            return "\n".join(str(x) for x in range(500, 512))
        return ""

    t.mp.setattr(mast, "_download_text", fake_download_text)
    wave, url = mast._try_wave_urls("r1000", requests.Session())
    assert url == expected
    assert wave.size == 12
    assert calls[0] == expected


#: name: (source, its routes, a call, what it returns, words of a warning, whether that is the source's only warning)
WARNED = {
    "njm_discovery_failure_warns_instead_of_vanishing": (  # one warning for both failures
        "njm", {INDEX: "not json", f"{NJM}/stellar_models": 403}, lambda g: g.discover_models(), [],
        ("grid list", "403", "not offered"), True),
    "njm_discovery_from_the_directory_listing_is_warned_about_once": (
        "njm", {INDEX: "not json", f"{NJM}/stellar_models": '<a href="?C=N;O=D">Name</a>'
                '<a href="/sed_tools/">Parent Directory</a><a href="Grid/">Grid/</a>'},
        lambda g: g.discover_models(), ["Grid"], ("using the directory listing instead",), True),
    "njm_grid_info_warns_and_returns_none_without_lookup": (
        "njm", ONLINE, lambda g: g.get_grid_info("Grid"), None, ("404",), False),
    # One plain line (the raw urllib3 text was several lines' worth for a proxy or timeout).
    "svo_discovery_failure_warns_instead_of_vanishing": (
        "svo", {SVO_INDEX: requests.ConnectionError("no route")}, lambda g: g.discover_models(), [],
        ("could not be reached", "not offered"), False),
    "svo_metadata_from_the_index_page_is_warned_about_once": (
        "svo", {SSAP_QUERY: 503, SVO_MODELS: '<a href="ssap.php?fid=2">b</a><a href="ssap.php?fid=1">a</a>'},
        lambda g: g.get_model_metadata("Grid"), [{"fid": 1}, {"fid": 2}], ("HTTP 503", "using the model index page"),
        True),
    "msg_discovery_failure_warns_instead_of_vanishing": (
        "msg", {MSG_INDEX: 500}, lambda g: g.discover_models(), [], ("500", "not offered"), False),
    "msg_discovery_failure_without_a_status_or_network_error_gives_its_reason": (
        "msg", {MSG_INDEX: requests.exceptions.InvalidURL("bad index address")}, lambda g: g.discover_models(), [],
        ("bad index address", "not offered"), True),
    "msg_index_without_grid_links_warns": (
        "msg", {MSG_INDEX: '<a href="about.html">About</a>'}, lambda g: g.discover_models(), [],
        ("no grids linked", "not offered"), True),
    "svo_discovery_timeout_says_how_long_it_waited": (
        "svo", {SVO_INDEX: requests.ReadTimeout("read timed out")}, lambda g: g.discover_models(), [],
        ("did not answer within 5+10 s", "not offered"), True),
    "svo_index_without_grid_links_warns": (
        "svo", {SVO_INDEX: "<html>No grids today</html>"}, lambda g: g.discover_models(), [],
        ("no spectral grids found", "not offered"), True),
    "njm_discovery_of_an_index_listing_no_models_uses_the_directory_listing": (
        "njm", {INDEX: json.dumps({"models": []}), f"{NJM}/stellar_models": '<a href="Grid/">Grid/</a>'},
        lambda g: g.discover_models(), ["Grid"], ("it lists no models", "using the directory listing instead"), True),
    "njm_grid_info_warns_about_an_empty_lookup_table": (
        "njm", {**ONLINE, GRID_URL + "lookup_table.csv": ""}, lambda g: g.get_grid_info("Grid"), None,
        ("lookup_table.csv is empty",), True),
    "njm_grid_info_warns_about_a_lookup_table_without_a_file_name_column": (
        "njm", {**ONLINE, GRID_URL + "lookup_table.csv": "teff,logg\n5000,4.0\n"}, lambda g: g.get_grid_info("Grid"),
        None, ("has no recognizable filename column",), True),
    "njm_grid_info_warns_about_a_lookup_table_listing_no_spectra": (
        "njm", {**ONLINE, GRID_URL + "lookup_table.csv": "#file_name,teff\n,5000\n"},
        lambda g: g.get_grid_info("Grid"), None, ("lists no spectra",), True),
}


def _warns(t, source, routes, call, result, words, alone):
    grabber, _ = _grabber(t, source, routes)
    with t.caplog.at_level(logging.WARNING):
        assert call(grabber) == result
    messages = warnings_from(t.caplog, f"grids.{source}")
    if alone:
        [message] = messages
        assert all(word in message for word in words)
    else:
        assert any(all(word in m for word in words) for m in messages)


# ----------------------------------------------------------------------
# Spectrum checks and readers
# ----------------------------------------------------------------------
#: name: (wavelengths, fluxes, why validate_converted_spectrum refuses them)
REFUSED = {
    "validate_rejects_empty_arrays": ([], [], "empty"),
    "validate_rejects_non_finite_wavelength": ([1.0, np.nan], [1.0, 1.0], "wavelength"),
    "validate_rejects_non_positive_wavelength": ([0.0, 1.0], [1.0, 1.0], "non-positive"),
    # 3 of 4 flux values non-finite (75%) exceeds the 50% threshold.
    "validate_rejects_mostly_non_finite_flux": ([1.0, 2.0, 3.0, 4.0], [1.0, np.nan, np.nan, np.nan], "non-finite flux"),
}


def _refused(t, wavelength, flux, reason):
    ok, msg = validate_converted_spectrum(np.array(wavelength), np.array(flux))
    assert ok is False
    assert reason in msg


@row
def single_valid_point_and_overflowing_flux_are_skipped_invalid(t):
    path = t.tmp / "one_point.txt"
    path.write_text("# teff = 5000\n4000 1.0\n")
    status, _ = clean_spectrum_file(str(path))
    assert status == "skipped_invalid"
    # A grid whose sample has nothing to read gives no grid-wide units, and says so
    with pytest.raises(RuntimeError, match=r"1 of 1 spectra"):
        clean_model_dir(str(t.tmp), sample_fraction=1.0)
    assert "Could not determine spectral grid units. Status: no_valid_detections" in t.capsys.readouterr().out
    # 1e306 W/m2/A is beyond a float in erg/s/cm2/A
    huge = t.tmp / "huge" / "s.txt"
    _write(huge, "# teff = 5000\n# flux_unit = W/m2/A\n", (4000.0, 5000.0, 6000.0), (1e306,) * 3)
    before = huge.read_bytes()
    with np.errstate(over="ignore"):
        assert clean_spectrum_file(str(huge)) == (
            "skipped_invalid", "validation failed: too many non-finite flux values: 3/3")
    assert huge.read_bytes() == before


@row
def an_msg_index_grid_is_cleaned_with_the_wavelengths_of_its_hdf5(t):
    path = t.tmp / "s.txt"
    _write(path, "# source = MSG HDF5\n# spec_group = g1\n" + _header(5000), np.arange(1.0, 11.0), np.ones(10))
    with h5py.File(t.tmp / "msg.h5", "w") as f:
        f.create_group("g1").create_dataset("lambda", data=WL12)
    assert clean_spectrum_file(str(path))[0] == "recovered"
    np.testing.assert_allclose(read_text_spectrum(str(path))[0], WL12[:10])
    assert clean_spectrum_file(str(path), try_h5_recovery=False)[0] == "skipped_already"


@row
def rebuild_lookup_table_empty_directory_returns_empty_string(t):
    assert rebuild_lookup_table(str(t.tmp)) == ""


@row
def bolometric_renorm_rescales_a_spectrum_100x_too_large(t):
    """clean_spectrum_file renormalises a spectrum whose integrated flux is more than 3x the blackbody's over the
    same wavelengths (the changelog's "NextGen T >= 5000 K flux 100x too large"). Until it was fixed this never
    happened: the block used the undefined name SIGMA_SB and a DEBUG-only handler hid the NameError."""
    path, wl = t.tmp / "hot_star.txt", np.linspace(3000.0, 9000.0, 500)
    flux_100x_too_large = planck_flam(wl, 6000.0) * 100.0  # mirrors the documented NextGen bug
    _write(path, _header(6000, units="# flux_unit = erg/s/cm2/A\n"), wl, flux_100x_too_large)
    status, detail = clean_spectrum_file(str(path))
    wl_after, flux_after = read_text_spectrum(str(path))
    ratio_after = float(np.trapezoid(flux_after, wl_after)) / float(
        np.trapezoid(planck_flam(wl_after, 6000.0), wl_after))
    assert status == "converted_renormed"
    assert "flux divided by 100" in detail
    assert ratio_after == pytest.approx(1.0, rel=1e-6)


@row
def text_spectrum_reader_keeps_the_finite_pairs(t):
    path = t.tmp / "sample.txt"
    path.write_text("# header\n1, 2\ninvalid\nwavelength flux\n3 4 extra\ninf 5\n")
    wavelength, flux = read_text_spectrum(path)
    np.testing.assert_array_equal(wavelength, [1.0, 3.0])
    np.testing.assert_array_equal(flux, [2.0, 4.0])


def test_grids(rows):
    rows(SCENARIOS)
    rows(RECOVERED, _recovered)
    rows(LEFT_ALONE, _left_alone)
    rows(TYPED, _typed)
    rows(CONVERTED, _converted)
    rows(HEADERS, _header_unit)
    rows(ZERO_BOUNDS, _query_bound)
    rows(WARNED, _warns)
    rows(MSG_VGRIDS, _msg_vgrid_read)
    rows(MSG_SPECTRA, _msg_spectrum_written)
    rows(REFUSED, _refused)
    rows(UNKNOWN_UNIT, _unknown_unit)
