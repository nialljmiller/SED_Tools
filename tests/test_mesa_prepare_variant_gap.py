"""Regression test for an integration gap found while writing the
collision/extra-axis test suite (PR3).

``mesa_prepare.py`` used to read per-variant flux cubes only from a
``fluxcube_library/`` directory, falling back to a master
``model_dir/flux_cube.bin``. ``precompute_flux_cube``'s default ``split``
strategy writes neither: it writes per-variant subgrids as sibling model
directories (``{Model}_{axis}_{value}/flux_cube.bin``) plus a
``variants_index.csv``, and no master cube. So ``list_variants()`` on a
grid that ``sed-tools rebuild`` had just split raised ``FileNotFoundError``
telling the user to run rebuild.

Since 0.5.0 ``list_variants()`` reads ``variants_index.csv`` (this test was
a strict ``xfail`` until then).
"""

import numpy as np

from sed_tools.mesa_prepare import list_variants
from sed_tools.precompute_flux_cube import precompute_flux_cube
from sed_tools.spectra_cleaner import clean_model_dir


def _make_collision_grid(model_dir):
    model_dir.mkdir()
    wl_nm = np.linspace(400.0, 700.0, 8)
    for fname, teff, alpha, flux_jy in [
        ("a1.txt", 5000.0, 0.0, 100.0),
        ("a2.txt", 5000.0, 0.4, 300.0),
        ("b1.txt", 6000.0, 0.0, 400.0),
        ("b2.txt", 6000.0, 0.4, 600.0),
    ]:
        lines = [
            "# source = synthetic\n",
            f"# teff = {teff}\n",
            "# logg = 4.0\n",
            "# metallicity = 0.0\n",
            f"# alpha = {alpha}\n",
            "# wavelength_unit = nm\n",
            "# flux in Jansky\n",
        ]
        with (model_dir / fname).open("w") as fh:
            fh.writelines(lines)
            for w in wl_nm:
                fh.write(f"{w:.6f} {flux_jy:.8e}\n")
    clean_model_dir(str(model_dir))


def test_mesa_prepare_can_list_variants_after_default_split_build(tmp_path):
    model_dir = tmp_path / "CollisionGrid"
    _make_collision_grid(model_dir)

    precompute_flux_cube(str(model_dir), str(model_dir / "flux_cube.bin"))  # default strategy = split

    variants = list_variants(model_dir)
    assert {v.label for v in variants} == {"alpha_0.0", "alpha_0.4"}
