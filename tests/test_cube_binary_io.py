"""Binary flux-cube (``flux_cube.bin``) read-path correctness tests.

Covers the parts of ``flux_cube_tool.FluxCube.from_file`` and
``models._read_flux_cube_header`` that the existing suite doesn't touch:
lossless round-tripping, the on-disk axis layout, truncated/corrupt
files, and the trailing-byte warning path.
"""

import struct

import numpy as np
import pytest

from sed_tools._flux import FluxCube
from sed_tools.models import SEDModel, _read_flux_cube_header


# ---------------------------------------------------------------------
# Lossless round trip
# ---------------------------------------------------------------------

def test_round_trip_grids_and_flux_are_lossless(make_affine_cube):
    cube = make_affine_cube()
    fc = FluxCube.from_file(str(cube.path))

    np.testing.assert_array_equal(fc.teff_grid, cube.teff)
    np.testing.assert_array_equal(fc.logg_grid, cube.logg)
    np.testing.assert_array_equal(fc.meta_grid, cube.meta)
    np.testing.assert_array_equal(fc.wavelengths, cube.wavelengths)
    assert fc.flux.shape == (cube.teff.size, cube.logg.size, cube.meta.size, cube.wavelengths.size)


def test_from_file_preserves_axis_order_at_specific_nodes(make_affine_cube):
    # Directly index fc.flux[t, l, m, w] (bypassing interpolate_spectrum)
    # for several asymmetric node combinations, to pin down the on-disk
    # axis convention independently of the Hermite interpolation path.
    cube = make_affine_cube()
    fc = FluxCube.from_file(str(cube.path))

    for ti, T in enumerate(cube.teff):
        for li, L in enumerate(cube.logg):
            for mi, M in enumerate(cube.meta):
                expected = cube.expected(T, L, M)
                np.testing.assert_allclose(fc.flux[ti, li, mi, :], expected, rtol=1e-10)


# ---------------------------------------------------------------------
# Truncated / corrupt files
# ---------------------------------------------------------------------

def test_truncated_header_raises_low_level(tmp_path):
    bad = tmp_path / "trunc_header.bin"
    bad.write_bytes(b"\x00" * 10)  # header is 16 bytes (4 int32s)
    with pytest.raises(ValueError, match="truncated"):
        FluxCube.from_file(str(bad))


def test_truncated_header_raises_via_read_flux_cube_header(tmp_path):
    bad = tmp_path / "trunc_header.bin"
    bad.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="truncated"):
        _read_flux_cube_header(bad)


def test_truncated_header_raises_via_sedmodel_construction(tmp_path):
    bad = tmp_path / "trunc_header.bin"
    bad.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="truncated"):
        SEDModel(name="bad", flux_cube_path=bad)


def test_truncated_body_raises_with_expected_and_found_counts(tmp_path):
    bad = tmp_path / "trunc_body.bin"
    teff = np.array([4000.0, 6000.0])
    logg = np.array([3.0])
    meta = np.array([0.0])
    wl = np.array([4000.0, 5000.0])
    with bad.open("wb") as fh:
        fh.write(struct.pack("4i", teff.size, logg.size, meta.size, wl.size))
        for grid in (teff, logg, meta, wl):
            grid.tofile(fh)
        # Body should hold 2*1*1*2 = 4 float64 values; write only 2.
        np.array([1.0, 2.0], dtype=np.float64).tofile(fh)

    with pytest.raises(ValueError) as excinfo:
        FluxCube.from_file(str(bad))
    message = str(excinfo.value)
    assert "4" in message  # expected count
    assert "2" in message  # actual count found


def test_truncated_body_only_surfaces_when_flux_is_loaded(tmp_path):
    # SEDModel's constructor only reads the header + grids, so a cube
    # with a truncated flux body still constructs successfully; the
    # error should only surface once the flux data is actually needed.
    bad = tmp_path / "trunc_body.bin"
    teff = np.array([4000.0, 6000.0])
    logg = np.array([3.0])
    meta = np.array([0.0])
    wl = np.array([4000.0, 5000.0])
    with bad.open("wb") as fh:
        fh.write(struct.pack("4i", teff.size, logg.size, meta.size, wl.size))
        for grid in (teff, logg, meta, wl):
            grid.tofile(fh)
        np.array([1.0, 2.0], dtype=np.float64).tofile(fh)

    model = SEDModel(name="bad", flux_cube_path=bad)  # should not raise
    with pytest.raises(ValueError):
        model(teff=5000.0, logg=3.0, metallicity=0.0)


def test_trailing_bytes_emit_warning_but_do_not_raise(tmp_path, make_affine_cube, capsys):
    cube = make_affine_cube()
    # Append one stray trailing byte after an otherwise-complete file.
    with cube.path.open("ab") as fh:
        fh.write(b"\x00")

    fc = FluxCube.from_file(str(cube.path))  # should not raise
    np.testing.assert_allclose(
        fc.flux[0, 0, 0, :], cube.expected(cube.teff[0], cube.logg[0], cube.meta[0]), rtol=1e-10
    )
    captured = capsys.readouterr()
    assert "trailing data" in captured.out.lower()


def test_empty_file_raises(tmp_path):
    bad = tmp_path / "empty.bin"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="truncated"):
        FluxCube.from_file(str(bad))
