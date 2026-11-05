"""Cleaning spectral grids: clean_model_dir's failures, the bolometric renormalisation,
wavelengths recovered from HDF5 files, coverage, and messages that name the CLI option or the model root.
"""

import shutil

import numpy as np
import pytest

import sed_tools.config as config
from sed_tools._constants import SIGMA, planck_flam

from helpers_cube import KURUCZ


# ---------------------------------------------------------------------
# grid_coverage
# ---------------------------------------------------------------------
def test_terminal_coverage_reports_missing_representative_spectra(tmp_path, monkeypatch, capsys):
    # A representative spectrum missing from disk used to be dropped silently.
    from sed_tools.grid_coverage import grid_coverage

    (tmp_path / "cool.txt").write_text("# teff = 4000\n# logg = 3.0\n1000 1.0\n2000 2.0\n")
    (tmp_path / "lookup_table.csv").write_text(
        "teff,logg,metallicity,filename\n4000,3.0,0.0,cool.txt\n8000,5.0,0.0,gone.txt\n"
    )
    monkeypatch.setattr(config, "get_ui_setting",
                        lambda name: "always" if name == "terminal_plots" else "never")
    grid_coverage(tmp_path, plot=False)
    output = capsys.readouterr().out
    assert "Representative SEDs" in output
    assert "spectrum file not found" in output
    assert "gone.txt" in output


# ---------------------------------------------------------------------
# spectra_cleaner: HDF5 wavelength recovery (characterisation)
# ---------------------------------------------------------------------
def _h5(path, build):
    import h5py

    with h5py.File(path, "w") as f:
        build(f)
    return path


def test_recovery_reads_direct_dataset_in_spec_group(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    wl = np.linspace(3000.0, 9000.0, 12)
    _h5(tmp_path / "msg.h5", lambda f: f.create_group("g1").create_dataset("lambda", data=wl))
    np.testing.assert_array_equal(recover_wavelengths_from_hdf5(str(tmp_path), "g1", 10), wl[:10])


def test_recovery_reads_range_x_then_segments(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    def build(f):
        rg = f.create_group("g1").create_group("range")
        rg.create_group("range2").create_dataset("wave", data=np.linspace(6000.0, 9000.0, 5))
        rg.create_group("range1").create_dataset("wave", data=np.linspace(3000.0, 5000.0, 5))
        g2 = f.create_group("g2").create_group("range")
        g2.create_dataset("x", data=np.linspace(2000.0, 4000.0, 8))

    _h5(tmp_path / "msg.h5", build)
    seg = recover_wavelengths_from_hdf5(str(tmp_path), "g1", 10)
    np.testing.assert_array_equal(seg, np.concatenate([np.linspace(3000.0, 5000.0, 5),
                                                       np.linspace(6000.0, 9000.0, 5)]))
    np.testing.assert_array_equal(recover_wavelengths_from_hdf5(str(tmp_path), "g2", 8),
                                  np.linspace(2000.0, 4000.0, 8))


def test_recovery_top_level_array_within_two_points(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    wl = np.linspace(3000.0, 9000.0, 11)
    _h5(tmp_path / "msg.h5", lambda f: f.create_dataset("wavelength", data=wl))
    np.testing.assert_array_equal(recover_wavelengths_from_hdf5(str(tmp_path), "absent", 10), wl[:10])
    assert recover_wavelengths_from_hdf5(str(tmp_path), "absent", 20) is None


def test_recovery_global_search_skips_non_numeric_datasets(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    wl = np.geomspace(1000.0, 50000.0, 9)

    def build(f):
        f.create_dataset("labels", data=np.array([b"a", b"b"]))
        f.create_dataset("records", data=np.zeros(9, dtype=[("a", "f8"), ("b", "i4")]))
        f.create_group("deep").create_group("er").create_dataset("grid", data=wl)

    _h5(tmp_path / "msg.h5", build)
    np.testing.assert_array_equal(recover_wavelengths_from_hdf5(str(tmp_path), "absent", 9), wl)


def test_recovery_rejects_index_like_arrays(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    _h5(tmp_path / "msg.h5", lambda f: f.create_group("g1").create_dataset("lambda", data=np.arange(10.0)))
    assert recover_wavelengths_from_hdf5(str(tmp_path), "g1", 10) is None


def test_recovery_warns_about_unreadable_h5(tmp_path, caplog):
    # An unreadable .h5 used to be skipped with a DEBUG message only.
    import logging

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    (tmp_path / "broken.h5").write_bytes(b"not an hdf5 file")
    with caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        assert recover_wavelengths_from_hdf5(str(tmp_path), "g1", 10) is None
    assert "broken.h5" in caplog.text


def test_recovery_still_uses_a_good_h5_beside_a_broken_one(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    (tmp_path / "broken.h5").write_bytes(b"not an hdf5 file")
    wl = np.linspace(3000.0, 9000.0, 10)
    _h5(tmp_path / "good.h5", lambda f: f.create_group("g1").create_dataset("lambda", data=wl))
    np.testing.assert_array_equal(recover_wavelengths_from_hdf5(str(tmp_path), "g1", 10), wl)


# ---------------------------------------------------------------------
# spectra_cleaner: clean_model_dir must not report success over
# spectra it left in their original units
# ---------------------------------------------------------------------
def _spectrum(path, teff=5000.0, unit_line="# wavelength_unit = nm\n# flux_unit = erg/s/cm2/A\n", wl=None, flux=None):
    import numpy as np

    wl = np.linspace(400.0, 700.0, 30) if wl is None else wl
    flux = np.full(len(wl), 1e-14) if flux is None else flux
    lines = [f"# source = testgrid\n# teff = {teff}\n# logg = 4.0\n# metallicity = 0.0\n", unit_line]
    lines += [f"{w:.6f} {f:.8e}\n" for w, f in zip(wl, flux)]
    path.write_text("".join(lines))


def _is_standardized(path):
    return "units_standardized = True" in path.read_text()


def test_clean_model_dir_raises_listing_unconverted_spectra(tmp_path):
    # Used to return a summary as if the grid were clean, leaving these two
    # files in their original units beside the converted ones.
    import numpy as np

    from sed_tools.spectra_cleaner import clean_model_dir

    for i in range(3):
        _spectrum(tmp_path / f"good_{i}.txt", teff=4000.0 + 1000 * i)
    (tmp_path / "empty.txt").write_text("# teff = 5000\n")
    _spectrum(tmp_path / "index.txt", unit_line="", wl=np.arange(30.0))

    with pytest.raises(RuntimeError) as err:
        clean_model_dir(str(tmp_path), sample_fraction=1.0)

    message = str(err.value)
    assert "2 of 5 spectra" in message
    assert "empty data (1): empty.txt" in message
    assert "index grid without HDF5 recovery (1): index.txt" in message
    assert "lookup_table.csv was rebuilt" in message
    assert all(_is_standardized(tmp_path / f"good_{i}.txt") for i in range(3))
    assert not _is_standardized(tmp_path / "empty.txt")
    assert (tmp_path / "lookup_table.csv").exists()


def test_clean_model_dir_records_unreadable_file_and_finishes_the_rest(tmp_path, monkeypatch):
    import sed_tools.spectra_cleaner as sc

    _spectrum(tmp_path / "a.txt")
    (tmp_path / "b.txt").mkdir()  # matches *.txt but cannot be read as a file
    _spectrum(tmp_path / "c.txt")
    monkeypatch.setattr(sc.random, "sample", lambda files, n: [str(tmp_path / "a.txt")])

    with pytest.raises(RuntimeError, match=r"IsADirectoryError.*\(1\): b\.txt"):
        sc.clean_model_dir(str(tmp_path), rebuild_lookup=False, sample_fraction=0.4)
    assert _is_standardized(tmp_path / "a.txt") and _is_standardized(tmp_path / "c.txt")


def test_clean_model_dir_unreadable_sampled_file_stops_before_any_rewrite(tmp_path):
    # Used to be counted as an invalid sample without saying why.
    from sed_tools.spectra_cleaner import clean_model_dir

    _spectrum(tmp_path / "a.txt")
    (tmp_path / "b.txt").mkdir()

    with pytest.raises(IsADirectoryError, match=r"b\.txt"):
        clean_model_dir(str(tmp_path), sample_fraction=1.0)
    assert not _is_standardized(tmp_path / "a.txt")


def test_clean_model_dir_stops_at_once_when_a_rewrite_fails(tmp_path, monkeypatch):
    # A failed rewrite can leave a header-only file that later runs skip, so
    # the batch must not carry on (it used to record 'error' and continue).
    import sed_tools.spectra_cleaner as sc

    for name in ("a.txt", "b.txt"):
        _spectrum(tmp_path / name)
    calls = []

    def disk_full(filepath, *args):
        calls.append(filepath)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sc, "write_standardized_spectrum", disk_full)
    with pytest.raises(RuntimeError, match=r"a\.txt.*only its new header"):
        sc.clean_model_dir(str(tmp_path), sample_fraction=1.0)
    assert len(calls) == 1


def test_write_standardized_spectrum_length_mismatch_leaves_file_intact(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import SpectrumMeta, UnitInfo, write_standardized_spectrum

    path = tmp_path / "s.txt"
    _spectrum(path)
    before = path.read_text()
    units = UnitInfo("angstrom", 1.0, "flam", 1.0, "high", "header")
    with pytest.raises(ValueError):
        write_standardized_spectrum(str(path), np.arange(1.0, 5.0), np.ones(3), SpectrumMeta(), units)
    assert path.read_text() == before


def test_clean_model_dir_warns_when_units_come_from_each_file(tmp_path, monkeypatch, caplog):
    # The sample happened to hold only standardized files, so the grid-wide
    # units were unknown and the rest were converted one by one, silently.
    import logging

    import sed_tools.spectra_cleaner as sc

    for i in range(9):
        _spectrum(tmp_path / f"raw_{i}.txt")
    done = tmp_path / "done.txt"
    done.write_text("# units_standardized = True\n4000 1.0\n5000 2.0\n")
    monkeypatch.setattr(sc.random, "sample", lambda files, n: [str(done)])

    with caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = sc.clean_model_dir(str(tmp_path), sample_fraction=0.1)
    assert len(summary["converted"]) == 9
    assert "9 spectra were converted using units detected file by file" in caplog.text


# ---------------------------------------------------------------------
# spectra_cleaner: bolometric renormalisation
# ---------------------------------------------------------------------
def _write_spectrum(path, teff, wl, flux):
    with path.open("w") as fh:
        fh.write(f"# teff = {teff}\n# logg = 4.0\n# metallicity = 0.0\n"
                 "# wavelength_unit = Angstrom\n# flux_unit = erg/cm2/s/A\n")
        for w, f in zip(wl, flux):
            fh.write(f"{w:.6f} {f:.8e}\n")


def _read(path):
    from sed_tools.spectrum_io import read_text_spectrum

    return read_text_spectrum(str(path))


def test_spectrum_100x_too_bright_is_renormalised_to_sigma_t4(tmp_path):
    # SVO's NextGen T >= 5000 K entries are about 100 times too bright.
    from sed_tools.spectra_cleaner import clean_spectrum_file

    wl = np.geomspace(100.0, 1e6, 3000)
    path = tmp_path / "nextgen_like.txt"
    _write_spectrum(path, 6000.0, wl, 100.0 * planck_flam(wl, 6000.0))

    status, detail = clean_spectrum_file(str(path))

    assert status == "converted_renormed"
    assert "flux divided by 100 " in detail
    wl_after, flux_after = _read(path)
    assert np.trapezoid(flux_after, wl_after) / (SIGMA * 6000.0 ** 4) == pytest.approx(1.0, rel=0.01)


def test_cool_giant_far_below_the_blackbody_in_the_optical_is_left_alone(tmp_path):
    # Granada HRES PHOENIX giants at 3000-3500 K, log g <= 1, have 2-10 % of a blackbody's
    # flux over 3000-7000 A (TiO), which holds 8-15 % of sigma*T^4. They are correct; the
    # first version, which also rescaled ratios below 0.1, multiplied them by 10-50.
    from sed_tools.spectra_cleaner import clean_spectrum_file

    wl = np.linspace(3000.0, 7000.0, 4000)
    path = tmp_path / "m_giant.txt"
    _write_spectrum(path, 3000.0, wl, 0.03 * planck_flam(wl, 3000.0))
    _, written_before = _read(path)

    status, _ = clean_spectrum_file(str(path))

    assert status == "converted"
    np.testing.assert_array_equal(_read(path)[1], written_before)


@pytest.mark.parametrize("teff", [20000.0, 30000.0, 37000.0])
def test_correct_hot_blackbody_on_a_wide_grid_is_left_alone(tmp_path, teff):
    # The first version integrated the blackbody on a 2000-point linear grid. Over
    # 10 A-1e7 A that misses a hot star's peak, so these correctly scaled spectra
    # came out 4-11 times too bright and would have been divided by that.
    from sed_tools.spectra_cleaner import clean_spectrum_file

    wl = np.geomspace(10.0, 1e7, 5000)
    flux = planck_flam(wl, teff)
    path = tmp_path / "hot.txt"
    _write_spectrum(path, teff, wl, flux)
    _, written_before = _read(path)

    status, _ = clean_spectrum_file(str(path))

    assert status == "converted"
    np.testing.assert_array_equal(_read(path)[1], written_before)


def test_pi_less_blackbody_is_within_the_thresholds(tmp_path):
    # B_lambda instead of pi*B_lambda (MESA's bbody grid) is 1/pi of sigma*T^4: not rescaled.
    from sed_tools.spectra_cleaner import clean_spectrum_file

    wl = np.geomspace(100.0, 1e6, 3000)
    path = tmp_path / "bbody.txt"
    _write_spectrum(path, 5000.0, wl, planck_flam(wl, 5000.0) / np.pi)
    assert clean_spectrum_file(str(path))[0] == "converted"


def test_spectrum_covering_little_of_sigma_t4_is_not_checked(tmp_path):
    # 3000-9000 A holds about 3 % of a 50000 K blackbody's flux: too little to judge the scale.
    from sed_tools.spectra_cleaner import clean_spectrum_file

    wl = np.linspace(3000.0, 9000.0, 500)
    path = tmp_path / "optical_only.txt"
    _write_spectrum(path, 50000.0, wl, 100.0 * planck_flam(wl, 50000.0))
    assert clean_spectrum_file(str(path))[0] == "converted"


def test_clean_model_dir_counts_renormed_spectra_as_converted_and_warns(tmp_path, caplog):
    import logging

    from sed_tools.spectra_cleaner import clean_model_dir

    wl = np.geomspace(100.0, 1e6, 2000)
    grid = tmp_path / "grid"
    grid.mkdir()
    _write_spectrum(grid / "a.txt", 5000.0, wl, 100.0 * planck_flam(wl, 5000.0))
    _write_spectrum(grid / "b.txt", 6000.0, wl, 120.0 * planck_flam(wl, 6000.0))
    _write_spectrum(grid / "c.txt", 7000.0, wl, planck_flam(wl, 7000.0))

    with caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = clean_model_dir(str(grid), try_h5_recovery=False, sample_fraction=1.0)

    assert summary["converted"] == ["a.txt", "b.txt", "c.txt"]
    assert summary["error"] == []
    assert sorted(summary["renormed"]) == ["a.txt", "b.txt"]
    assert summary["renormed"]["a.txt"] == pytest.approx(100.0)
    assert summary["renormed"]["b.txt"] == pytest.approx(120.0)
    warning = [r.getMessage() for r in caplog.records if "bolometric renormalisation" in r.getMessage()]
    assert len(warning) == 1 and "2 spectra integrated to 100-120 times" in warning[0]
    for name, teff in (("a.txt", 5000.0), ("b.txt", 6000.0), ("c.txt", 7000.0)):
        wl_after, flux_after = _read(grid / name)
        assert np.trapezoid(flux_after, wl_after) / (SIGMA * teff ** 4) == pytest.approx(1.0, rel=0.01)


def test_unstandardized_kurucz_slice_is_converted_without_renormalisation(tmp_path, caplog):
    # The golden clean_model_dir case only skips these files (they are marked standardized),
    # so it does not show the check leaves correctly scaled Kurucz spectra alone. This does.
    import logging

    from sed_tools.spectra_cleaner import clean_model_dir

    grid = shutil.copytree(KURUCZ, tmp_path / "Kurucz2003all")
    before = {}
    for path in sorted(grid.glob("*.txt")):
        path.write_text("".join(line for line in path.read_text().splitlines(keepends=True)
                                if not line.startswith("# units_standardized")))
        before[path.name] = _read(path)

    with caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = clean_model_dir(str(grid), try_h5_recovery=False, sample_fraction=1.0)

    assert len(summary["converted"]) == 12
    assert "renormed" not in summary
    assert not any("renormalisation" in r.getMessage() for r in caplog.records)
    for name, (wl, flux) in before.items():
        wl_after, flux_after = _read(grid / name)
        np.testing.assert_array_equal(wl_after, wl)
        np.testing.assert_array_equal(flux_after, flux)


# ---------------------------------------------------------------------
# Undeclared flux unit, coverage's A-D note, import's "Load with" hint
# ---------------------------------------------------------------------
def _undeclared_flux_grid(directory):
    """Three spectra of a surface flux in erg/s/cm2/A whose headers do not declare the flux unit."""
    from sed_tools._constants import planck_flam

    directory.mkdir(parents=True)
    wl = np.linspace(3000.0, 10000.0, 200)
    for i, teff in enumerate((3500.0, 3600.0, 3700.0)):
        lines = [f"# teff = {teff}\n", "# logg = 4.5\n", "# metallicity = 0.0\n", "# wavelength_unit = Angstrom\n"]
        lines += [f"{w:.6f} {f:.8e}\n" for w, f in zip(wl, planck_flam(wl, teff))]
        (directory / f"s{i}.txt").write_text("".join(lines))
    return directory


def test_undeclared_flux_unit_error_names_the_cli_option(tmp_path):
    from sed_tools.spectra_cleaner import check_flux_unit

    grid = _undeclared_flux_grid(tmp_path / "Cool")
    with pytest.raises(ValueError) as excinfo:
        check_flux_unit(str(grid), sorted(str(p) for p in grid.glob("*.txt")), sample_fraction=1.0)
    message = str(excinfo.value)
    assert "Pass the unit as flux_unit='fnu_jy'" in message
    assert "on the command line, pass --flux-unit fnu_jy to sed-tools grids, rebuild or import" in message


def _coverage_grid(directory, teffs):
    directory.mkdir()
    rows = "".join(f"s{i}.txt,{t},4.0,0.0\n" for i, t in enumerate(teffs))
    (directory / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\n" + rows)
    return directory


def test_coverage_says_why_the_normality_test_was_skipped(tmp_path, monkeypatch, capsys):
    # It said "scipy required" whenever the test was skipped, with scipy installed.
    import sys

    from sed_tools.grid_coverage import coverage_summary, print_coverage

    print_coverage(coverage_summary(_coverage_grid(tmp_path / "few", [5000, 5500, 6000])))
    out = capsys.readouterr().out
    assert "A-D     : skipped (needs at least 8 distinct values; this axis has 3)" in out
    assert "scipy" not in out

    monkeypatch.setitem(sys.modules, "scipy.stats", None)  # as if scipy were not installed
    print_coverage(coverage_summary(_coverage_grid(tmp_path / "many", range(4000, 9000, 500))))
    out = capsys.readouterr().out
    assert "    A-D     : skipped (needs scipy, which is not installed)" in out


def test_import_hint_names_the_model_root_it_imported_into(tmp_path, monkeypatch, capsys):
    from sed_tools.api import SED

    models = tmp_path / "models"
    SED.import_grid(_undeclared_flux_grid(tmp_path / "mine"), model_root=models, build_cube=False,
                    build_h5=False, flux_unit="flam")
    assert f"Load with: SED.local('mine', model_root={str(models)!r})" in capsys.readouterr().out

    monkeypatch.setattr(SED, "_model_root", models)
    SED.import_grid(_undeclared_flux_grid(tmp_path / "default"), build_cube=False, build_h5=False,
                    flux_unit="flam")
    assert "Load with: SED.local('default')\n" in capsys.readouterr().out
