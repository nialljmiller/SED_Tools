"""Combining grids: a missing spectrum, a row without parameters or a failed copy stops the
combine; the normalisation warning, the common wavelength grid and the plots.
"""

import pytest


# ---------------------------------------------------------------------
# combine_stellar_atm
# ---------------------------------------------------------------------
GOLDEN_MODELS = __import__("pathlib").Path(__file__).parent / "golden" / "inputs" / "stellar_models"


def _copy_grids(tmp_path):
    import shutil

    models = tmp_path / "models"
    for name in ("Kurucz2003all", "Kurucz2003all__alpha_04"):
        shutil.copytree(GOLDEN_MODELS / name, models / name)
    return models


def _append_lookup_row(grid_dir, row):
    with open(grid_dir / "lookup_table.csv", "a") as fh:
        fh.write(row + "\n")


@pytest.mark.parametrize("grid", ["Kurucz2003all", "Kurucz2003all__alpha_04"])
def test_combine_stops_on_a_missing_spectrum(tmp_path, grid):
    # Used to give the spectrum a normalisation factor of 1 (alpha_04) or
    # leave its node empty for nearest-neighbour filling (Kurucz), silently.
    import numpy as np

    from sed_tools.api import SED

    models = _copy_grids(tmp_path)
    _append_lookup_row(models / grid, "missing.txt,0,999.9,2,1.25,4.0,0.0,999.9,999.9,999.9,9000.0,999.9,2,999.9")
    np.random.seed(0)
    with pytest.raises(ValueError, match=r"No such file or directory: '.*missing\.txt'"):
        SED.combine(["Kurucz2003all", "Kurucz2003all__alpha_04"], output="combined",
                    model_root=models, visualization=False)
    assert not (models / "combined").exists()


def test_combine_stops_on_a_row_without_parameters(tmp_path):
    # A NaN Teff used to land on grid node [0, ...] and overwrite it.
    import numpy as np

    from sed_tools.api import SED

    models = _copy_grids(tmp_path)
    grid = models / "Kurucz2003all"
    _append_lookup_row(grid, "Kurucz2003all_fid16515.txt,0,999.9,2,1.25,4.0,-0.5,999.9,999.9,999.9,,999.9,2,999.9")
    np.random.seed(0)
    with pytest.raises(ValueError, match=r"non-finite parameters \(teff=nan"):
        SED.combine(["Kurucz2003all", "Kurucz2003all__alpha_04"], output="combined",
                    model_root=models, visualization=False)


def test_combine_raises_after_writing_when_a_copy_fails(tmp_path, monkeypatch):
    # Copy failures used to be printed and the run reported success.
    import shutil

    import numpy as np

    import sed_tools.combine_stellar_atm as csa
    from sed_tools.api import SED

    models = _copy_grids(tmp_path)
    real_copy = shutil.copy2

    def flaky_copy(src, dst):
        if src.endswith("Kurucz2003all_fid16516.txt"):
            raise PermissionError(13, "Permission denied", src)
        return real_copy(src, dst)

    monkeypatch.setattr(csa.shutil, "copy2", flaky_copy)
    np.random.seed(0)
    with pytest.raises(OSError, match=r"(?s)Could not copy 1 spectra.*fid16516"):
        SED.combine(["Kurucz2003all", "Kurucz2003all__alpha_04"], output="combined",
                    model_root=models, visualization=False)
    assert (models / "combined" / "lookup_table.csv").exists()


def test_normalisation_warns_when_spectra_keep_native_scale(tmp_path, caplog):
    import logging

    import numpy as np

    from sed_tools.combine_stellar_atm import compute_normalization_factors, load_model_data

    models = _copy_grids(tmp_path)
    ref = load_model_data(str(models / "Kurucz2003all"))
    target = load_model_data(str(models / "Kurucz2003all__alpha_04"))
    coarse = np.linspace(3000.0, 10000.0, 50)  # fewer than 100 points in any overlap
    with caplog.at_level(logging.WARNING, logger="sed_tools.combine_stellar_atm"):
        factors = compute_normalization_factors(target, ref, coarse)
    np.testing.assert_array_equal(factors, np.ones(len(target["files"])))
    assert f"{len(target['files'])} spectra of Kurucz2003all__alpha_04 keep their native flux scale" in caplog.text


def test_common_wavelength_grid_needs_a_usable_spectrum(tmp_path):
    # Used to fail with OverflowError (int of -inf).
    import numpy as np

    from sed_tools.combine_stellar_atm import create_common_wavelength_grid

    (tmp_path / "far_ir.txt").write_text("".join(f"{w} 1.0\n" for w in np.linspace(3e4, 9e4, 50)))
    data = {"files": np.array(["far_ir.txt"]), "model_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="between 3000 and 25000"):
        create_common_wavelength_grid([data])


