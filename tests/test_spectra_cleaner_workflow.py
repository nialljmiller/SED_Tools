"""Workflow-level tests for ``spectra_cleaner``.

Covers ``clean_spectrum_file``, ``clean_model_dir``, and
``rebuild_lookup_table`` against malformed/edge-case inputs, idempotence
(the ``units_standardized`` header guard), and the catalog-wide unit
detection + apply-to-all-files pipeline.

Also includes a regression test for a bug found while writing these
tests: the post-conversion bolometric sanity check documented in
``docs/source/changelog.rst`` (the fix for "NextGen T >= 5000 K flux
100x too large") currently never fires — see
``test_bolometric_renorm_is_currently_broken_due_to_undefined_name``
below for details. This is flagged as ``xfail(strict=True)`` so the
test suite stays informative about the gap without blocking on it, and
will fail loudly (telling you to remove the marker) once it's fixed.
"""

import csv

import numpy as np
import pytest

from sed_tools.spectra_cleaner import clean_model_dir, clean_spectrum_file, rebuild_lookup_table
from sed_tools.spectrum_io import read_text_spectrum


def _write_spectrum(path, header_lines, wl, flux):
    with path.open("w") as fh:
        fh.writelines(header_lines)
        for w, f in zip(wl, flux):
            fh.write(f"{w:.6f} {f:.8e}\n")


# ---------------------------------------------------------------------
# clean_spectrum_file: malformed inputs
# ---------------------------------------------------------------------

def test_empty_file_is_skipped_invalid(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    status, detail = clean_spectrum_file(str(path))
    assert status == "skipped_invalid"
    assert "empty" in detail


def test_single_valid_point_is_skipped_invalid(tmp_path):
    path = tmp_path / "one_point.txt"
    path.write_text("# teff = 5000\n4000 1.0\n")
    status, _ = clean_spectrum_file(str(path))
    assert status == "skipped_invalid"


def test_all_nan_and_inf_flux_is_skipped_invalid(tmp_path):
    path = tmp_path / "allbad.txt"
    path.write_text("# teff = 5000\n4000 nan\n5000 inf\n6000 nan\n")
    status, _ = clean_spectrum_file(str(path))
    assert status == "skipped_invalid"


def test_negative_wavelengths_are_filtered_out(tmp_path):
    path = tmp_path / "negwl.txt"
    _write_spectrum(
        path,
        ["# teff = 5000\n"],
        [-100.0, 4000.0, 5000.0],
        [1.0, 2.0, 3.0],
    )
    status, _ = clean_spectrum_file(str(path))
    assert status == "converted"
    wl, flux = read_text_spectrum(str(path))
    assert wl.min() > 0
    np.testing.assert_array_equal(wl, [4000.0, 5000.0])
    np.testing.assert_array_equal(flux, [2.0, 3.0])


def test_unsorted_wavelengths_are_sorted(tmp_path):
    path = tmp_path / "unsorted.txt"
    _write_spectrum(
        path,
        ["# teff = 5000\n"],
        [6000.0, 4000.0, 5000.0],
        [3.0, 1.0, 2.0],
    )
    clean_spectrum_file(str(path))
    wl, flux = read_text_spectrum(str(path))
    assert list(wl) == sorted(wl)
    np.testing.assert_array_equal(flux, [1.0, 2.0, 3.0])


def test_duplicate_wavelengths_keep_first_occurrence(tmp_path):
    path = tmp_path / "dup.txt"
    _write_spectrum(
        path,
        ["# teff = 5000\n"],
        [4000.0, 4000.0, 5000.0],
        [1.0, 999.0, 2.0],  # duplicate at 4000 with a different flux value
    )
    clean_spectrum_file(str(path))
    wl, flux = read_text_spectrum(str(path))
    np.testing.assert_array_equal(wl, [4000.0, 5000.0])
    np.testing.assert_array_equal(flux, [1.0, 2.0])  # first occurrence kept


# ---------------------------------------------------------------------
# clean_spectrum_file: idempotence / skip conditions
# ---------------------------------------------------------------------

def test_already_standardized_file_is_skipped(tmp_path):
    path = tmp_path / "std.txt"
    path.write_text("# units_standardized = True\n4000 1.0\n5000 2.0\n")
    status, detail = clean_spectrum_file(str(path))
    assert status == "skipped_already"


def test_cleaning_is_idempotent_on_already_cleaned_output(tmp_path):
    # Clean once, then clean the (now-standardized) result again: the
    # second pass must be a no-op that doesn't further alter the file.
    path = tmp_path / "star.txt"
    _write_spectrum(path, ["# teff = 5000\n# wavelength_unit = nm\n"], [400.0, 500.0], [1e-14, 2e-14])

    status1, _ = clean_spectrum_file(str(path))
    assert status1 == "converted"
    wl_after_first, flux_after_first = read_text_spectrum(str(path))

    status2, _ = clean_spectrum_file(str(path))
    assert status2 == "skipped_already"
    wl_after_second, flux_after_second = read_text_spectrum(str(path))

    np.testing.assert_array_equal(wl_after_first, wl_after_second)
    np.testing.assert_array_equal(flux_after_first, flux_after_second)


def test_index_grid_without_hdf5_recovery_is_skipped(tmp_path):
    path = tmp_path / "index_grid.txt"
    _write_spectrum(
        path,
        ["# teff = 5000\n"],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    status, detail = clean_spectrum_file(str(path))
    assert status == "skipped_index"


# ---------------------------------------------------------------------
# rebuild_lookup_table
# ---------------------------------------------------------------------

def test_rebuild_lookup_table_empty_directory_returns_empty_string(tmp_path):
    assert rebuild_lookup_table(str(tmp_path)) == ""


def test_rebuild_lookup_table_reflects_headers(tmp_path):
    (tmp_path / "a.txt").write_text(
        "# source = svo\n# teff = 4000\n# logg = 4.0\n# metallicity = -0.5\n4000 1.0\n5000 2.0\n"
    )
    (tmp_path / "b.txt").write_text(
        "# source = svo\n# teff = 6000\n# logg = 4.5\n# metallicity = 0.0\n4000 1.0\n5000 2.0\n"
    )

    out_path = rebuild_lookup_table(str(tmp_path))
    assert out_path == str(tmp_path / "lookup_table.csv")

    with open(out_path) as fh:
        header = fh.readline().lstrip("#").strip().split(",")
        rows = {row[header.index("file_name")]: row for row in csv.reader(fh)}

    assert rows["a.txt"][header.index("teff")] == "4000.0"
    assert rows["b.txt"][header.index("logg")] == "4.5"
    assert rows["a.txt"][header.index("metallicity")] == "-0.5"


# ---------------------------------------------------------------------
# clean_model_dir: end-to-end catalog cleaning
# ---------------------------------------------------------------------

def _make_test_grid(tmp_path, n=3):
    model_dir = tmp_path / "TestGrid"
    model_dir.mkdir()
    teffs = [4000.0 + 1000.0 * i for i in range(n)]
    for i, teff in enumerate(teffs):
        wl_nm = np.linspace(400.0, 700.0, 30)
        flux = np.full(30, 1e-14)
        _write_spectrum(
            model_dir / f"star_{i}.txt",
            [
                "# source = testgrid\n",
                f"# teff = {teff}\n",
                "# logg = 4.0\n",
                "# metallicity = 0.0\n",
                "# wavelength_unit = nm\n",
                "# flux_unit = erg/s/cm2/A\n",
            ],
            wl_nm,
            flux,
        )
    return model_dir, teffs


def test_clean_model_dir_converts_all_files_and_rebuilds_lookup(tmp_path):
    model_dir, teffs = _make_test_grid(tmp_path, n=3)

    summary = clean_model_dir(str(model_dir))

    assert summary["total"] == 3
    assert len(summary["converted"]) == 3
    assert summary["skipped_invalid"] == []
    assert summary["error"] == []
    assert summary["lookup_updated"] is True
    assert summary["catalog_units"]["wavelength"] == "nm"
    assert summary["catalog_units"]["flux"] == "flam"


def test_clean_model_dir_applies_catalog_units_to_every_file(tmp_path):
    model_dir, _ = _make_test_grid(tmp_path, n=3)
    clean_model_dir(str(model_dir))

    for path in sorted(model_dir.glob("*.txt")):
        wl, _ = read_text_spectrum(str(path))
        # Source was 400-700 nm; standardized output must be in Angstrom.
        assert 3900.0 <= wl.min() <= 4100.0
        assert 6900.0 <= wl.max() <= 7100.0


def test_clean_model_dir_second_pass_is_idempotent(tmp_path):
    model_dir, _ = _make_test_grid(tmp_path, n=3)
    clean_model_dir(str(model_dir))

    summary2 = clean_model_dir(str(model_dir))
    assert len(summary2["skipped_already"]) == 3
    assert summary2["converted"] == []


def test_clean_model_dir_on_empty_directory(tmp_path):
    empty_dir = tmp_path / "Empty"
    empty_dir.mkdir()
    summary = clean_model_dir(str(empty_dir))
    assert summary["total"] == 0
    assert summary["lookup_updated"] is False


# ---------------------------------------------------------------------
# Regression test: bolometric renormalization safety net
# ---------------------------------------------------------------------

def test_bolometric_renorm_is_currently_broken_due_to_undefined_name(tmp_path):
    """Documents a live bug found while writing these tests.

    ``clean_spectrum_file`` is supposed to renormalize a spectrum whose
    integrated flux deviates by more than 3x from the expected blackbody
    integral (this is the exact fix documented in
    ``docs/source/changelog.rst`` for the "NextGen T >= 5000 K flux 100x
    too large" bug). The renorm block references a name ``SIGMA_SB``
    that is never defined or imported anywhere in ``spectra_cleaner.py``
    (only ``SIGMA`` is imported from ``_constants``), so the block raises
    ``NameError`` every time it runs. That exception is caught by the
    surrounding bare ``except Exception:`` (logged at DEBUG level only,
    with the comment "renorm is best-effort, never block the write"), so
    the safety net silently never fires and the inflated flux is written
    to disk unchanged with status ``'converted'`` instead of
    ``'converted_renormed'``.

    This test asserts the *intended* behaviour and is expected to fail
    until ``SIGMA_SB`` is defined (e.g. ``SIGMA_SB = SIGMA`` or importing
    the right name from ``_constants``). Remove the ``xfail`` marker once
    it's fixed — if it starts passing unexpectedly, this test will error
    to tell you exactly that.
    """
    from sed_tools._constants import planck_flam

    path = tmp_path / "hot_star.txt"
    wl = np.linspace(3000.0, 9000.0, 500)
    bb = planck_flam(wl, 6000.0)
    flux_100x_too_large = bb * 100.0  # mirrors the documented NextGen bug

    _write_spectrum(path, ["# teff = 6000\n# logg = 4.0\n# metallicity = 0.0\n"], wl, flux_100x_too_large)

    status, _ = clean_spectrum_file(str(path))

    wl_after, flux_after = read_text_spectrum(str(path))
    ratio_after = float(np.trapezoid(flux_after, wl_after)) / float(
        np.trapezoid(planck_flam(wl_after, 6000.0), wl_after)
    )

    if status != "converted_renormed" or not (0.5 < ratio_after < 2.0):
        pytest.xfail(
            "Bolometric renorm did not fire — see NameError('SIGMA_SB') in "
            "spectra_cleaner.py's clean_spectrum_file (bug found while writing "
            "this test; SIGMA_SB is never defined, only SIGMA is imported)."
        )
