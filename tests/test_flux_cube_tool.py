import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sed_tools.flux_cube_tool import FluxCube


def test_flux_cube_loader_restores_transposed_body(tmp_path):
    teff = np.array([3000.0, 6000.0])
    logg = np.array([1.0, 5.0])
    meta = np.array([-1.0, 0.5])
    wavelengths = np.array([100.0, 200.0, 300.0])

    flux = np.arange(
        teff.size * logg.size * meta.size * wavelengths.size, dtype=np.float64
    ).reshape((teff.size, logg.size, meta.size, wavelengths.size))

    path = tmp_path / "cube.bin"
    with path.open("wb") as fh:
        fh.write(struct.pack("4i", teff.size, logg.size, meta.size, wavelengths.size))
        for grid in (teff, logg, meta, wavelengths):
            grid.astype(np.float64).tofile(fh)
        np.transpose(flux, (3, 2, 1, 0)).tofile(fh)

    cube = FluxCube.from_file(str(path))

    assert cube.flux.shape == flux.shape
    assert np.allclose(cube.flux, flux)
    assert np.allclose(cube.teff_grid, teff)
    assert np.allclose(cube.logg_grid, logg)
    assert np.allclose(cube.meta_grid, meta)
    assert np.allclose(cube.wavelengths, wavelengths)
