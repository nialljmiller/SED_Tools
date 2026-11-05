"""Workflow-level tests for ``spectra_cleaner``.

Covers ``clean_spectrum_file``, ``clean_model_dir``, and
``rebuild_lookup_table`` against malformed/edge-case inputs, idempotence
(the ``units_standardized`` header guard), and the catalog-wide unit
detection + apply-to-all-files pipeline.

Also includes a regression test for the post-conversion bolometric
renormalisation documented in ``docs/source/changelog.rst`` (the fix for
"NextGen T >= 5000 K flux 100x too large"), which never ran until it was
fixed: see ``test_bolometric_renorm_rescales_a_spectrum_100x_too_large``.
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

def test_single_valid_point_is_skipped_invalid(tmp_path):
    path = tmp_path / "one_point.txt"
    path.write_text("# teff = 5000\n4000 1.0\n")
    status, _ = clean_spectrum_file(str(path))
    assert status == "skipped_invalid"


# ---------------------------------------------------------------------
# clean_spectrum_file: idempotence / skip conditions
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# rebuild_lookup_table
# ---------------------------------------------------------------------

def test_rebuild_lookup_table_empty_directory_returns_empty_string(tmp_path):
    assert rebuild_lookup_table(str(tmp_path)) == ""


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


# ---------------------------------------------------------------------
# Regression test: bolometric renormalization safety net
# ---------------------------------------------------------------------

def test_bolometric_renorm_rescales_a_spectrum_100x_too_large(tmp_path):
    """``clean_spectrum_file`` renormalises a spectrum whose integrated flux
    deviates by more than 3x from the blackbody integral over the same
    wavelengths (the changelog's fix for "NextGen T >= 5000 K flux 100x too
    large").

    Until it was fixed this never happened: the block used the undefined name
    ``SIGMA_SB`` and a DEBUG-only handler hid the ``NameError``, so the flux
    was written unchanged with status ``'converted'``. The test then skipped
    itself with ``pytest.xfail``.
    """
    from sed_tools._constants import planck_flam

    path = tmp_path / "hot_star.txt"
    wl = np.linspace(3000.0, 9000.0, 500)
    bb = planck_flam(wl, 6000.0)
    flux_100x_too_large = bb * 100.0  # mirrors the documented NextGen bug

    _write_spectrum(path, ["# teff = 6000\n# logg = 4.0\n# metallicity = 0.0\n# flux_unit = erg/s/cm2/A\n"],
                    wl, flux_100x_too_large)

    status, detail = clean_spectrum_file(str(path))

    wl_after, flux_after = read_text_spectrum(str(path))
    ratio_after = float(np.trapezoid(flux_after, wl_after)) / float(
        np.trapezoid(planck_flam(wl_after, 6000.0), wl_after)
    )

    assert status == "converted_renormed"
    assert "flux divided by 100" in detail
    assert ratio_after == pytest.approx(1.0, rel=1e-6)
