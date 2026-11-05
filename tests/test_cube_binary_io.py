"""Reading ``flux_cube.bin``: a truncated header or body raises, naming the expected and found
counts; trailing bytes are warned about.
"""

import struct

import numpy as np
import pytest

from sed_tools._flux import FluxCube


# ---------------------------------------------------------------------
# Truncated / corrupt files
# ---------------------------------------------------------------------

def test_truncated_header_raises_low_level(tmp_path):
    bad = tmp_path / "trunc_header.bin"
    bad.write_bytes(b"\x00" * 10)  # header is 16 bytes (4 int32s)
    with pytest.raises(ValueError, match="truncated"):
        FluxCube.from_file(str(bad))


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
