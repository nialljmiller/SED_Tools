"""Integration tests for the full pipeline:

    raw (non-standard-unit) .txt spectra
    -> spectra_cleaner.clean_model_dir (unit standardization + lookup rebuild)
    -> precompute_flux_cube.precompute_flux_cube (flux_cube.bin construction)
    -> models.SEDModel (public API load + exact-node evaluation)

This is the "principal integration test" called for in the testing
review: expected flux values are computed independently in each test
(from the raw Jansky/nm inputs, via the textbook F_nu -> F_lambda
Jacobian) rather than by re-using any of the pipeline's own conversion
code, so a bug anywhere in the chain — wrong catalog units applied,
wrong node written to the cube, wavelength grid corruption on disk I/O,
axis-order mixups — will show up as a mismatch here even if each stage
passes its own unit tests in isolation.
"""

import itertools

import numpy as np
import pytest

from sed_tools.models import SEDModel
from sed_tools.precompute_flux_cube import precompute_flux_cube
from sed_tools.spectra_cleaner import clean_model_dir

# Speed of light in Angstrom/s, hardcoded independently (see
# test_spectra_cleaner_units.py for the same convention).
C_ANGSTROM_PER_S = 2.99792458e18


def _write_raw_node(path, teff, logg, meta, wl_nm, flux_jy):
    """Write one *uncleaned* spectrum: nm wavelengths, Jansky flux."""
    lines = [
        "# source = synthetic\n",
        f"# teff = {teff}\n",
        f"# logg = {logg}\n",
        f"# metallicity = {meta}\n",
        "# wavelength_unit = nm\n",
        "# flux in Jansky\n",
    ]
    with path.open("w") as fh:
        fh.writelines(lines)
        for w, f in zip(wl_nm, np.broadcast_to(flux_jy, wl_nm.shape)):
            fh.write(f"{w:.6f} {f:.8e}\n")


def test_clean_to_cube_round_trip_exact_nodes(tmp_path):
    model_dir = tmp_path / "RawGrid"
    model_dir.mkdir()

    teffs = [4000.0, 6000.0]
    loggs = [4.0, 5.0]
    metas = [-0.5, 0.0]
    wl_nm = np.linspace(400.0, 700.0, 10)
    nodes = list(itertools.product(teffs, loggs, metas))
    # Distinct flux level per node so a mislabeled/scrambled node is
    # caught, not just a wrong-but-plausible-looking value.
    flux_jy_for_node = {node: 50.0 + 5.0 * i for i, node in enumerate(nodes)}

    for i, (T, L, M) in enumerate(nodes):
        _write_raw_node(model_dir / f"node_{i}.txt", T, L, M, wl_nm, flux_jy_for_node[(T, L, M)])

    summary = clean_model_dir(str(model_dir))
    assert len(summary["converted"]) == 8
    assert summary["error"] == []

    cube_path = model_dir / "flux_cube.bin"
    precompute_flux_cube(str(model_dir), str(cube_path))

    model = SEDModel(name="RawGrid", flux_cube_path=cube_path)
    np.testing.assert_array_equal(model.teff_grid, teffs)
    np.testing.assert_array_equal(model.logg_grid, loggs)
    np.testing.assert_array_equal(model.meta_grid, metas)

    wl_ang_expected = wl_nm * 10.0
    for T, L, M in nodes:
        result = model(teff=T, logg=L, metallicity=M)
        expected_flam = (flux_jy_for_node[(T, L, M)] * 1e-23) * C_ANGSTROM_PER_S / (wl_ang_expected ** 2)

        # atol/rtol account for the %.6f / %.8e text round-trip through
        # the standardized .txt file on disk, not pipeline imprecision.
        np.testing.assert_allclose(result.wavelength, wl_ang_expected, atol=1e-4)
        np.testing.assert_allclose(result.flux, expected_flam, rtol=1e-6)


def test_clean_to_cube_round_trip_lookup_table_matches_grid(tmp_path):
    model_dir = tmp_path / "RawGrid"
    model_dir.mkdir()
    wl_nm = np.linspace(400.0, 700.0, 5)
    _write_raw_node(model_dir / "a.txt", 4000.0, 4.0, 0.0, wl_nm, 10.0)
    _write_raw_node(model_dir / "b.txt", 5000.0, 4.5, -0.3, wl_nm, 20.0)

    summary = clean_model_dir(str(model_dir))
    lookup_text = (model_dir / "lookup_table.csv").read_text()

    assert lookup_text == open(summary["lookup_path"]).read()
    assert "4000.0" in lookup_text
    assert "5000.0" in lookup_text
    assert "True" in lookup_text  # units_standardized column


def test_clean_to_cube_rebuild_is_reproducible(tmp_path):
    # Building the same cleaned directory twice must produce a
    # byte-identical flux_cube.bin.
    model_dir = tmp_path / "RawGrid"
    model_dir.mkdir()
    wl_nm = np.linspace(400.0, 700.0, 5)
    for i, teff in enumerate([4000.0, 5000.0]):
        _write_raw_node(model_dir / f"n{i}.txt", teff, 4.0, 0.0, wl_nm, 10.0 + i)

    clean_model_dir(str(model_dir))

    cube_a = model_dir / "cube_a.bin"
    cube_b = model_dir / "cube_b.bin"
    precompute_flux_cube(str(model_dir), str(cube_a))
    precompute_flux_cube(str(model_dir), str(cube_b))

    assert cube_a.read_bytes() == cube_b.read_bytes()


def test_partial_wavelength_coverage_zero_fills_without_cross_contamination(tmp_path):
    # One node has full coverage, the other a narrower native range.
    # The narrow node's cube slice must be exactly zero outside its own
    # coverage, and the full-coverage node's values at those same
    # wavelengths must be unaffected (no cross-node bleed).
    model_dir = tmp_path / "PartialGrid"
    model_dir.mkdir()
    wl_full = np.linspace(400.0, 700.0, 16)
    wl_narrow = np.linspace(450.0, 650.0, 16)

    _write_raw_node(model_dir / "nodeA.txt", 4000.0, 4.0, 0.0, wl_full, 100.0)
    _write_raw_node(model_dir / "nodeB.txt", 6000.0, 4.0, 0.0, wl_narrow, 200.0)

    clean_model_dir(str(model_dir))
    cube_path = model_dir / "flux_cube.bin"
    precompute_flux_cube(str(model_dir), str(cube_path))

    model = SEDModel(name="PartialGrid", flux_cube_path=cube_path)
    fc = model._load_cube()
    wl = model.wavelengths

    outside_narrow = (wl < 4500.0) | (wl > 6500.0)
    assert outside_narrow.any()

    idx_a = list(model.teff_grid).index(4000.0)
    idx_b = list(model.teff_grid).index(6000.0)
    flux_a = fc.flux[idx_a, 0, 0, :]
    flux_b = fc.flux[idx_b, 0, 0, :]

    assert np.all(flux_a[outside_narrow] > 0)  # full-coverage node unaffected
    np.testing.assert_array_equal(flux_b[outside_narrow], 0.0)  # narrow node zero-filled
    assert np.all(flux_b[~outside_narrow] > 0)  # narrow node has real data inside its range
