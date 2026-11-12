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
from sed_tools import cli, njm_filter_grabber
from sed_tools import mast_spectra_grabber as mast, msg_spectra_grabber as msg, njm_spectra_grabber as njm
from sed_tools import svo_spectra_grabber as svo
from sed_tools._constants import SIGMA, planck_flam
from sed_tools.api import SED, _bundle_shortfall, _flux_cube_shortfall
from sed_tools.models import _read_flux_cube_header
from sed_tools.precompute_flux_cube import spectra_not_in_cube
from sed_tools.spectra_cleaner import (FLUX_UNITS, clean_model_dir, clean_spectrum_file, detect_units_from_header,
                                       rebuild_lookup_table, validate_converted_spectrum)
from sed_tools.spectrum_io import read_text_spectrum
from sed_tools.svo_regen_spectra_lookup import regenerate_lookup_table

from support import (GRID, KURUCZ, NJM, Resp, Session, answer, copy_without_times, njm_grids, run_cli, scenarios,
                     write_flux_cube)

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
    from sed_tools.grid_coverage import grid_coverage

    (t.tmp / "cool.txt").write_text("# teff = 4000\n# logg = 3.0\n1000 1.0\n2000 2.0\n")
    (t.tmp / "lookup_table.csv").write_text(
        "teff,logg,metallicity,filename\n4000,3.0,0.0,cool.txt\n8000,5.0,0.0,gone.txt\n")
    t.mp.setattr(config, "get_ui_setting", lambda name: "always" if name == "terminal_plots" else "never")
    grid_coverage(t.tmp, plot=False)
    output = t.capsys.readouterr().out
    assert "Representative SEDs" in output
    assert "spectrum file not found" in output
    assert "gone.txt" in output


@row
def recovery_warns_about_unreadable_h5(t):
    """An unreadable .h5 used to be skipped with a DEBUG message only."""
    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    (t.tmp / "broken.h5").write_bytes(b"not an hdf5 file")
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
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
    t.mp.setattr("sed_tools.spectra_cleaner.random.sample", lambda files, n: [str(t.tmp / "a.txt")])
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
    """A failed rewrite can leave a header-only file that later runs skip, so the batch stops (it used to record
    'error' and carry on)."""
    for name in ("a.txt", "b.txt"):
        _spectrum(t.tmp / name)
    calls = []

    def disk_full(filepath, *args):
        calls.append(filepath)
        raise OSError(28, "No space left on device")

    t.mp.setattr("sed_tools.spectra_cleaner.write_standardized_spectrum", disk_full)
    with pytest.raises(RuntimeError, match=r"a\.txt.*only its new header"):
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
    t.mp.setattr("sed_tools.spectra_cleaner.random.sample", lambda files, n: [str(done)])
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = clean_model_dir(str(t.tmp), sample_fraction=0.1)
    assert len(summary["converted"]) == 9
    assert "9 spectra were converted using units detected file by file" in t.caplog.text


@row
def clean_model_dir_counts_renormed_spectra_as_converted_and_warns(t):
    wl, grid = np.geomspace(100.0, 1e6, 2000), t.tmp / "grid"
    for name, teff, scale in (("a.txt", 5000.0, 100.0), ("b.txt", 6000.0, 120.0), ("c.txt", 7000.0, 1.0)):
        _write(grid / name, _header(teff), wl, scale * planck_flam(wl, teff))
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
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
def coverage_says_why_the_normality_test_was_skipped(t):
    """It said "scipy required" whenever the test was skipped, with scipy installed."""
    from sed_tools.grid_coverage import coverage_summary, print_coverage

    def grid(directory, teffs):
        directory.mkdir()
        rows = "".join(f"s{i}.txt,{teff},4.0,0.0\n" for i, teff in enumerate(teffs))
        (directory / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\n" + rows)
        return directory

    print_coverage(coverage_summary(grid(t.tmp / "few", [5000, 5500, 6000])))
    out = t.capsys.readouterr().out
    assert "A-D     : skipped (needs at least 8 distinct values; this axis has 3)" in out
    assert "scipy" not in out
    t.mp.setitem(sys.modules, "scipy.stats", None)  # as if scipy were not installed
    print_coverage(coverage_summary(grid(t.tmp / "many", range(4000, 9000, 500))))
    out = t.capsys.readouterr().out
    assert "    A-D     : skipped (needs scipy, which is not installed)" in out


WL12, WL11, WL9 = np.linspace(3000.0, 9000.0, 12), np.linspace(3000.0, 9000.0, 11), np.geomspace(1000.0, 50000.0, 9)


def _ranges(f):
    rg = f.create_group("g1").create_group("range")
    rg.create_group("range2").create_dataset("wave", data=np.linspace(6000.0, 9000.0, 5))
    rg.create_group("range1").create_dataset("wave", data=np.linspace(3000.0, 5000.0, 5))
    f.create_group("g2").create_group("range").create_dataset("x", data=np.linspace(2000.0, 4000.0, 8))


def _among_others(f):
    f.create_dataset("labels", data=np.array([b"a", b"b"]))
    f.create_dataset("records", data=np.zeros(9, dtype=[("a", "f8"), ("b", "i4")]))
    f.create_group("deep").create_group("er").create_dataset("grid", data=WL9)


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
}


def _recovered(t, build, group, n, expected):
    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

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
    from sed_tools.spectra_cleaner import undeclared_flux_unit

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
    summary = clean_model_dir(str(grid), sample_fraction=1.0, flux_unit="flam")
    assert len(summary["converted"]) == 3 and summary["catalog_units"]["flux"] == "flam"
    wl, flux = read_text_spectrum(str(grid / "s0.txt"))
    np.testing.assert_allclose(flux, planck_flam(wl, 3500.0), rtol=1e-8)
    assert "# original_flux_unit = flam" in (grid / "s0.txt").read_text()


@row
def declared_flux_units_clean_as_before(t):
    """A declared unit is used without asking, and gives the same files as passing it."""
    declared, passed = _grid(t.tmp / "Declared", "erg/cm2/s/A"), _grid(t.tmp / "Passed")
    clean_model_dir(str(declared), sample_fraction=1.0)
    clean_model_dir(str(passed), sample_fraction=1.0, flux_unit="flam")
    assert _contents(declared) == _contents(passed)


@row
def an_unknown_flux_unit_name_is_refused_before_anything_changes(t):
    grid = _grid(t.tmp / "Cool")
    before = _contents(grid)
    with pytest.raises(ValueError, match=r"flux_unit='Jy' is not one of flam, fnu, fnu_jy, flam_si, flam_si_nm, "
                                         r"flam_si_um, normalized"):
        clean_model_dir(str(grid), flux_unit="Jy")
    assert _contents(grid) == before


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
    t.mp.setattr("sed_tools.spectra_cleaner.random.sample", lambda files, n: [str(done)])
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
        cli._ask_flux_unit("Cool", sorted(str(p) for p in grid.glob("*.txt")))
    assert len(prompts) == 1


@row
def cli_does_not_ask_when_the_flux_unit_is_declared(t):
    grid = _grid(t.tmp / "Declared", "erg/cm2/s/A")
    prompts = _terminal(t)
    assert cli._ask_flux_unit("Declared", sorted(str(p) for p in grid.glob("*.txt"))) is None
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


@row
def cli_import_asks_for_the_flux_unit_and_passes_it_on(t):
    src = _grid(t.tmp / "mine")
    calls = []
    t.mp.setattr("sed_tools.api.SED.import_grid", lambda *a, **k: calls.append(k))
    _terminal(t, str(src), "", "n", "flam")
    cli.run_import_flow(base_dir=str(t.tmp / "models"))
    assert calls[0]["flux_unit"] == "flam"


@row
def cli_import_command_asks_on_a_terminal_and_not_for_a_dry_run(t):
    src = _grid(t.tmp / "mine")
    calls = []
    t.mp.setattr("sed_tools.api.SED.import_grid", lambda *a, **k: calls.append(k))
    prompts = _terminal(t, "fnu")
    for extra in ([], ["--dry-run"]):
        t.mp.setattr(sys, "argv", ["sed-tools", "import", "--path", str(src), "--base", str(t.tmp / "models")] + extra)
        assert cli.main() == 0
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
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = clean_model_dir(str(declared), sample_fraction=1.0, flux_unit="fnu")
    clean_model_dir(str(reference), sample_fraction=1.0)
    assert summary["catalog_units"]["flux"] == "flam"  # the header's unit, not fnu
    assert [p.read_text() for p in sorted(declared.glob("s*.txt"))] == \
        [p.read_text() for p in sorted(reference.glob("s*.txt"))]
    assert "the spectra declare their flux unit (flam); flux_unit='fnu' is not used" in t.caplog.text


@row
def flux_unit_is_used_for_spectra_that_declare_none(t):
    grid = _grid(t.tmp / "Undeclared", **FLAM_GRID)
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = clean_model_dir(str(grid), sample_fraction=1.0, flux_unit="flam")
    assert summary["catalog_units"]["flux"] == "flam" and "is not used" not in t.caplog.text


#: name: (the answers typed, the unit taken). The guess is offered, not taken: the unit must be typed.
TYPED = {
    "cli_offers_the_best_guess_and_takes_a_typed_unit": (["flam"], "flam"),
}


def _typed(t, answers, unit):
    grid = _grid(t.tmp / "Cool")
    prompts = _terminal(t, *answers)
    assert cli._ask_flux_unit("Cool", sorted(str(p) for p in grid.glob("*.txt"))) == unit
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
    assert calls == [("GET", INDEX, True, (njm_filter_grabber.NJM_CONNECT_TIMEOUT, 10))]


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
    assert run_cli(t.mp, "grids", "--source", "njm", "--grids", GRID, "--no-cube", "--no-h5",
                   "--base", str(t.tmp / "b")) == 0
    assert "HDF5 bundle   : Kurucz2003all.h5 (exists)" in t.capsys.readouterr().out


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
def the_rebuild_hint_names_base_only_when_the_grid_is_elsewhere(t):
    t.mp.setattr("sed_tools.grids.pipeline.STELLAR_DIR_DEFAULT", t.tmp / "default")
    assert cli._rebuild_command(GRID, str(t.tmp / "default" / GRID)) == f"sed-tools rebuild --grids {GRID}"
    elsewhere = t.tmp / "my grids"
    assert (cli._rebuild_command(GRID, str(elsewhere / GRID))
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

    t.mp.setattr("sed_tools.svo_spectra_grabber.SVOSpectraGrabber", SVO)
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
    assert run_cli(t.mp, "grids", "--grids", GRID, "--source", "njm", "--base", base, "--no-h5", "--no-cube") == 1
    err = " ".join(t.capsys.readouterr().err.split())
    assert "flux_cube.bin does not include 8 of the 12 spectra beside it" in err
    assert f"Rebuild it with: sed-tools rebuild --grids {GRID} --base {base}" in err
    with h5py.File(folder / f"{GRID}.h5", "r") as bundle:  # rebuilt although --no-h5: it lacked the rest
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
MSG_INDEX = "http://user.astro.wisc.edu/~townsend/static.php?ref=msg-grids"
MSG_FILE = "http://user.astro.wisc.edu/msg/grids/sg-Test.h5"
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


@row
def njm_unreachable_warns_with_every_attempt_and_named_grid_requests_raise(t):
    down = requests.ConnectionError("connection refused")
    with t.caplog.at_level(logging.WARNING):
        grabber, session = _grabber(t, "njm", {INDEX: down, NJM: down, INDEX.replace("https", "http"): down})
    assert not grabber.is_available()
    [message] = warnings_from(t.caplog, "njm_spectra_grabber")
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
    assert any("certificate expired" in m for m in warnings_from(t.caplog, "njm_spectra_grabber"))


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
                                     GRID_URL + "lookup_table.csv": LOOKUP})
    metadata = {"model_url": GRID_URL, "txt_files": ["a.txt", "b.txt", "c.txt"], "csv_files": ["lookup_table.csv"]}
    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("Grid", metadata)
    message = str(err.value)
    assert "2 file(s)" in message and "b.txt: connection dropped" in message and "c.txt: 404" in message
    model_dir = t.tmp / "Grid"
    assert (model_dir / "a.txt").exists() and (model_dir / "lookup_table.csv").exists()
    assert not (model_dir / "b.txt").exists()  # partial file removed


@row
def njm_wavelength_cut_failures_are_raised(t):
    grabber, _ = _grabber(t, "njm", {**ONLINE, GRID_URL + "a.txt": SPECTRUM, GRID_URL + "b.txt": SPECTRUM})
    real_trim = njm._trim_spectrum_file

    def trim(path, wl_range):
        if path.endswith("b.txt"):
            raise PermissionError("read-only")
        return real_trim(path, wl_range)

    t.mp.setattr(njm, "_trim_spectrum_file", trim)
    metadata = {"model_url": GRID_URL, "txt_files": ["a.txt", "b.txt"]}
    with pytest.raises(RuntimeError, match="b.txt: wavelength cut failed: read-only"):
        grabber.download_model_spectra("Grid", metadata, wl_range=(1100, 1200))
    wl, _ = read_text_spectrum(t.tmp / "Grid" / "a.txt")
    assert wl.min() == 1100 and wl.max() == 1200


def svo_spectrum_url(fid):
    return f"{SSAP}?model=Grid&fid={fid}&format=ascii"


@row
def svo_metadata_warns_once_before_probing_and_raises_when_a_probe_fails(t):
    grabber, _ = _grabber(t, "svo", {SSAP_QUERY: 503, SVO_MODELS: requests.ConnectionError("index down"),
                                     svo_spectrum_url(7): requests.ConnectionError("probe refused")})
    with t.caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="FID 7 of Grid: probe refused"):
        grabber.get_model_metadata("Grid")
    messages = warnings_from(t.caplog, "svo_spectra_grabber")
    [message] = messages  # one warning for the fallback, naming both reasons
    assert "HTTP 503" in message and "index down" in message and "probing" in message
    assert not (t.tmp / FID_CACHE).exists()


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
    assert any("unreadable FID cache" in m for m in warnings_from(t.caplog, "svo_spectra_grabber"))


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
                                     svo_spectrum_url(3): requests.ConnectionError("reset by peer")})
    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("Grid", [{"fid": 1}, {"fid": 2}, {"fid": 3}])
    message = str(err.value)
    assert "2 of 3 spectra" in message
    assert "FID 2: content is not spectral data" in message and "FID 3: reset by peer" in message
    model_dir = t.tmp / "Grid"
    assert (model_dir / "Grid_fid1.txt").exists()
    assert not (model_dir / "lookup_table.csv").exists()


@row
def svo_complete_download_writes_lookup_table(t):
    grabber, _ = _grabber(t, "svo", {svo_spectrum_url(1): SVO_SPECTRUM, svo_spectrum_url(2): SVO_SPECTRUM})
    assert grabber.download_model_spectra("Grid", [{"fid": 1}, {"fid": 2}]) == 2
    lookup = (t.tmp / "Grid" / "lookup_table.csv").read_text().splitlines()
    assert lookup[0] == "#file_name,logg,metallicity,teff" and len(lookup) == 3


def _msg_grid(t):
    """A grabber whose sg-Test.h5 holds three spectra: one good, one with no flux dataset, one with an undecodable
    wavelength segment."""
    grabber, _ = _grabber(t, "msg")
    grabber.model_urls["sg-Test"] = MSG_FILE
    wave = np.linspace(3000.0, 9000.0, 50)
    (t.tmp / "sg-Test").mkdir()
    with h5py.File(t.tmp / "sg-Test" / "sg-Test.h5", "w") as f:
        for i, values in enumerate(([5000.0, 6000.0], [-1.0, 0.0], [4.0, 4.5])):
            f[f"vgrid/axes[{i}]/x"] = values
        f["vgrid/v_lin_seq"] = [1, 2, 3]
        good = f.create_group("specsource/specints[1]")
        good["flux"] = np.ones(50)
        good["range/x"] = wave
        no_flux = f.create_group("specsource/specints[2]")
        no_flux["a"] = np.ones(50)
        no_flux["b"] = np.ones(50)
        gapped = f.create_group("specsource/specints[3]")
        gapped["flux"] = np.ones(60)
        gapped["range/ranges[1]/x"] = wave[:30]
        gapped.create_group("range/ranges[2]").attrs["label"] = "no numbers here"
    return grabber, t.tmp / "sg-Test"


@row
def msg_metadata_raises_for_unknown_grid_and_failed_download(t):
    grabber, _ = _grabber(t, "msg", {MSG_INDEX: f'<a href="{MSG_FILE}">sg-Test</a>',
                                     MSG_FILE: requests.ConnectionError("dropped")})
    with pytest.raises(RuntimeError, match="unknown to this grabber"):
        grabber.get_model_metadata("sg-Test")
    assert grabber.discover_models() == ["sg-Test"]
    with pytest.raises(RuntimeError, match="could not download .*sg-Test.h5: dropped"):
        grabber.get_model_metadata("sg-Test")


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


@row
def msg_failed_extractions_raise_after_the_rest(t):
    grabber, model_dir = _msg_grid(t)
    spectra = grabber.get_model_metadata("sg-Test")
    with pytest.raises(RuntimeError) as err:
        grabber.download_model_spectra("sg-Test", spectra)
    message = str(err.value)
    assert "2 of 3 spectra" in message
    assert "sg-Test_fid2.txt: No suitable flux dataset" in message
    assert "sg-Test_fid3.txt: Unable to recover wavelength grid" in message  # was written truncated to 30 points
    wl, _ = read_text_spectrum(model_dir / "sg-Test_fid1.txt")
    assert wl.size == 50
    assert not (model_dir / "lookup_table.csv").exists()


@row
def msg_cut_exclusions_are_not_failures(t):
    grabber, model_dir = _msg_grid(t)
    spectra = grabber.get_model_metadata("sg-Test")[:1]
    assert grabber.download_model_spectra("sg-Test", spectra, teff_range=(5500, 7000)) == 0
    assert grabber.download_model_spectra("sg-Test", spectra, teff_range=(4500, 5500)) == 1
    assert (model_dir / "lookup_table.csv").exists()


@row
def msg_axis_role_beyond_the_grid_raises(t):
    grabber, _ = _grabber(t, "msg")
    grabber.model_axes["sg-C3K-test"] = {
        "axes": [np.array([5000.0, 6000.0]), np.array([-1.0, 0.0]), np.array([4.0, 4.5])],
        "shape": (2, 2, 2), "vlin": np.array([0]), "roles": {"teff": 0, "meta": 1, "logg": 3}}
    with pytest.raises(RuntimeError, match="logg is mapped to axis 3, but the grid has 3 axes"):
        grabber._params_from_index("sg-C3K-test", 0)


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
def mast_unreadable_download_script_raises(t):
    session = Routes({BOSZ_SCRIPT: requests.ConnectionError("timed out")})
    with pytest.raises(RuntimeError, match="metallicities would be missing:\n  .*timed out"):
        mast._gather_urls_from_scripts("r1000", ["m+0.00"], session)
    with pytest.raises(RuntimeError, match="not found"):
        mast._gather_urls_from_scripts("r1000", ["m+0.00"], Routes({}))


@row
def mast_metallicity_prompt_rejects_values_it_would_have_dropped(t):
    grabber = mast.MASTSpectraGrabber(base_dir=str(t.tmp), session=Routes({}))
    for typed, error in (("-1.0,abc", "'abc' is not a number"), ("-3.0", "no \\[M/H\\] = -3.0")):
        t.mp.setattr("builtins.input", lambda prompt, typed=typed: typed)
        with pytest.raises(ValueError, match=error):
            grabber.get_model_metadata("BOSZ-2024-r1000")
    t.mp.setattr("builtins.input", lambda prompt: "-1.0, +0.25,")
    assert grabber.get_model_metadata("BOSZ-2024-r1000")["metals"] == ["m-1.00", "m+0.25"]


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
def mast_wavelength_cut_that_leaves_nothing_is_not_a_failure(t):
    grabber = _bosz(t, {bosz_url(5750): bosz_resam(12)})
    info = {"reskey": "r1000", "metals": ["m+0.00"]}
    assert grabber.download_model_spectra("BOSZ-2024-r1000", info, wl_range=(9000, 9500)) == 0


@row
def svo_cache_that_cannot_be_read_or_written_is_warned_about(t):
    grabber, _ = _grabber(t, "svo", {SSAP_QUERY: "<TD>ssap.php?fid=3</TD>"})
    (t.tmp / FID_CACHE).mkdir()  # a directory where the cache file goes
    with t.caplog.at_level(logging.WARNING):
        assert grabber.get_model_metadata("Grid") == [{"fid": 3}]
    messages = warnings_from(t.caplog, "svo_spectra_grabber")
    assert any("unreadable FID cache" in m for m in messages)
    assert any("could not cache the FID list" in m for m in messages)


@row
def mast_filter_keeps_names_it_cannot_parse_so_the_download_reports_them(t):
    odd = "https://x/m+0.00/r1000/bosz2024_xx_t05750_r1000_resam.txt.gz"
    assert mast._filter_urls_by_params([odd], teff_range=(5000, 6000)) == [odd]


@row
def mast_failed_first_spectrum_does_not_stop_the_rest(t):
    grabber = _bosz(t, {bosz_url(5750): requests.ConnectionError("first one dropped"), bosz_url(6000): bosz_resam(12)},
                    max_workers=1)
    with pytest.raises(RuntimeError, match="1 of 2 BOSZ spectra.*\n.*t05750.*first one dropped"):
        grabber.download_model_spectra("BOSZ-2024-r1000", {"reskey": "r1000", "metals": ["m+0.00"]})
    assert len(list((t.tmp / "BOSZ-2024-r1000").glob("*t06000*.txt"))) == 1


@row
def msg_discover_and_run_finishes_other_grids_then_raises(t):
    index = f'<a href="{MSG_FILE}">a</a><a href="http://user.astro.wisc.edu/msg/grids/sg-Other.h5">b</a>'
    grabber, _ = _grabber(t, "msg", {MSG_INDEX: index})
    done = []

    def metadata(name):
        if name == "sg-Test":
            raise RuntimeError("download failed")
        return [{"fid": 1}]

    t.mp.setattr(grabber, "get_model_metadata", metadata)
    t.mp.setattr(grabber, "download_model_spectra", lambda name, meta: done.append(name) or 1)
    with pytest.raises(RuntimeError, match="1 of 2 grid\\(s\\) failed:\n  sg-Test: download failed"):
        grabber.discover_and_run()
    assert done == ["sg-Other"]


@row
def mast_download_accepts_generic_axis_kwargs(t):
    url = ("https://archive.stsci.edu/hlsps/bosz/bosz2024/m+0.00/r1000/"
           "bosz2024_mp_t05750_g+4.5_m+0.00_a+0.00_c+0.00_v2_r1000_resam.txt.gz")

    def fake_download_text_gz(download_url, session=None):
        assert download_url == url
        return "1000 1.0\n2000 1.0\n3000 1.0\n4000 1.0\n"

    t.mp.setattr("sed_tools.mast_spectra_grabber._gather_urls_from_scripts", lambda reskey, metals, session: [url])
    t.mp.setattr("sed_tools.mast_spectra_grabber._try_wave_urls", lambda reskey, session: (None, None))
    t.mp.setattr("sed_tools.mast_spectra_grabber._download_text_gz", fake_download_text_gz)
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
}


def _warns(t, source, routes, call, result, words, alone):
    grabber, _ = _grabber(t, source, routes)
    with t.caplog.at_level(logging.WARNING):
        assert call(grabber) == result
    messages = warnings_from(t.caplog, f"{source}_spectra_grabber")
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
def single_valid_point_is_skipped_invalid(t):
    path = t.tmp / "one_point.txt"
    path.write_text("# teff = 5000\n4000 1.0\n")
    status, _ = clean_spectrum_file(str(path))
    assert status == "skipped_invalid"


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
def text_spectrum_compatibility_aliases(t):
    from sed_tools.cli import load_txt_spectrum
    from sed_tools.precompute_flux_cube import load_sed
    from sed_tools.spectra_cleaner import read_spectrum_data

    path = t.tmp / "sample.txt"
    path.write_text("# header\n1, 2\ninvalid\n3 4 extra\ninf 5\n")
    expected = read_text_spectrum(path)
    np.testing.assert_array_equal(expected[0], [1.0, 3.0])
    np.testing.assert_array_equal(expected[1], [2.0, 4.0])
    for loader in (load_txt_spectrum, load_sed, read_spectrum_data):
        actual = loader(str(path))
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


def test_grids(rows):
    rows(SCENARIOS)
    rows(RECOVERED, _recovered)
    rows(LEFT_ALONE, _left_alone)
    rows(TYPED, _typed)
    rows(CONVERTED, _converted)
    rows(HEADERS, _header_unit)
    rows(ZERO_BOUNDS, _query_bound)
    rows(WARNED, _warns)
    rows(REFUSED, _refused)
