import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sed_tools import SED


def _write_flux_cube(path: Path, teff, logg, meta, wavelengths) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("4i", len(teff), len(logg), len(meta), len(wavelengths)))
        for grid in (teff, logg, meta, wavelengths):
            np.asarray(grid, dtype=np.float64).tofile(fh)
        # The interpolated flux values are not needed for metadata inspection


def test_find_atmospheres_prefers_full_coverage(tmp_path):
    root = tmp_path / "stellar_models"
    model_a = root / "model_a" / "flux_cube.bin"
    model_b = root / "model_b" / "flux_cube.bin"

    _write_flux_cube(
        model_a,
        teff=[4000.0, 5000.0, 6000.0],
        logg=[1.0, 2.0, 3.0],
        meta=[-1.0, 0.0, 1.0],
        wavelengths=[4000.0, 5000.0],
    )
    _write_flux_cube(
        model_b,
        teff=[4500.0, 5200.0],
        logg=[1.5, 2.5],
        meta=[-0.5, 0.2],
        wavelengths=[4000.0, 5000.0],
    )

    sed = SED(model_root=root)
    matches = sed.find_atmospheres(
        teff_range=(4600.0, 5500.0),
        logg_range=(1.8, 2.4),
        metallicity_range=(-0.3, 0.1),
    )

    assert [m.name for m in matches] == ["model_a"]
    assert matches[0].covers_range is True

    partial = sed.find_atmospheres(
        teff_range=(4600.0, 5500.0),
        logg_range=(1.8, 2.4),
        metallicity_range=(-0.3, 0.1),
        allow_partial=True,
    )

    assert {m.name for m in partial} == {"model_a", "model_b"}
    assert any(not m.covers_range for m in partial)
