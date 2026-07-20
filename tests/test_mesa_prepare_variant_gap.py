"""Regression test for an integration gap found while writing the
collision/extra-axis test suite (PR3).

``mesa_prepare.py`` (variant selection / export for MESA) is documented
and written to read per-variant flux cubes from a ``fluxcube_library/``
directory (``model_dir/fluxcube_library/flux_cube__{label}.bin`` plus
``fluxcube_library/index.csv``), falling back to a master
``model_dir/flux_cube.bin`` if that directory doesn't exist.

But ``precompute_flux_cube.py``'s default and physically-safe collision
strategy — ``split`` — never produces either of those. It writes
per-variant subgrids as *sibling model directories*
(``{Model}_{axis}_{value}/flux_cube.bin``) plus a ``variants_index.csv``,
and explicitly does *not* write a master ``flux_cube.bin`` at the parent
level ("The parent directory is not MESA-ready"). The function that
would write ``fluxcube_library/index.csv``
(``precompute_flux_cube._write_library_index``) exists but is never
called from anywhere in the codebase.

Net effect: running ``precompute_flux_cube`` with the default strategy
on a grid with extra axes, then calling
``mesa_prepare.list_variants()`` on that same parent directory —
exactly the workflow ``mesa_prepare``'s own docstring and
``run_interactive()`` describe — raises ``FileNotFoundError`` telling
the user to "Run 'sed-tools rebuild' first," even though they just did.

This is marked ``xfail`` rather than asserting the failure outright, so
it will loudly tell you (via an "unexpectedly passed" report) once the
two are reconciled — whether that means ``precompute_flux_cube`` starts
writing ``fluxcube_library/``, or ``mesa_prepare`` is updated to read
the ``{Model}_{axis}_{value}/`` + ``variants_index.csv`` layout that
``split`` actually produces.
"""

import numpy as np
import pytest

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


@pytest.mark.xfail(
    reason=(
        "precompute_flux_cube's default 'split' strategy produces "
        "{Model}_{axis}_{value}/ subgrids + variants_index.csv, not the "
        "fluxcube_library/ layout mesa_prepare.list_variants() reads. "
        "See module docstring."
    ),
    strict=True,
)
def test_mesa_prepare_can_list_variants_after_default_split_build(tmp_path):
    model_dir = tmp_path / "CollisionGrid"
    _make_collision_grid(model_dir)

    precompute_flux_cube(str(model_dir), str(model_dir / "flux_cube.bin"))  # default strategy = split

    variants = list_variants(model_dir)  # currently raises FileNotFoundError
    assert {v.label for v in variants} == {"alpha_0.0", "alpha_0.4"}
