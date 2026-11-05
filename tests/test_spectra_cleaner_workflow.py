"""``spectra_cleaner`` workflows: a one-point spectrum is skipped as invalid, ``rebuild_lookup_table``
on an empty directory, and the bolometric renormalisation of a spectrum 100x too bright (the
changelog's fix for "NextGen T >= 5000 K flux 100x too large").
"""


import numpy as np
import pytest

from sed_tools.spectra_cleaner import clean_spectrum_file, rebuild_lookup_table
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
# rebuild_lookup_table
# ---------------------------------------------------------------------

def test_rebuild_lookup_table_empty_directory_returns_empty_string(tmp_path):
    assert rebuild_lookup_table(str(tmp_path)) == ""


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
