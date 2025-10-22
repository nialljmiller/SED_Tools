import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import SED_tools
from sed_tools._flux import AB_ZERO_FLUX


def _write_flux_cube(
    path: Path,
    teff: np.ndarray,
    logg: np.ndarray,
    meta: np.ndarray,
    wavelengths: np.ndarray,
    flux: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("4i", teff.size, logg.size, meta.size, wavelengths.size))
        for grid in (teff, logg, meta, wavelengths):
            np.asarray(grid, dtype=np.float64).tofile(fh)
        flux.astype(np.float64).transpose(3, 2, 1, 0).ravel().tofile(fh)


def _write_filter(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# wavelength_unit = angstrom\n4000 1.0\n5000 1.0\n")


def test_pipeline_like_usage(tmp_path: Path) -> None:
    model_root = tmp_path / "stellar_models"
    filter_root = tmp_path / "filters"

    teff = np.array([1000.0, 2000.0])
    logg = np.array([1.0, 2.0])
    meta = np.array([-1.0, 0.0])
    wavelengths = np.array([4000.0, 5000.0])
    flux_value = 2.0
    flux = np.full((teff.size, logg.size, meta.size, wavelengths.size), flux_value)

    flux_cube = model_root / "demo" / "flux_cube.bin"
    _write_flux_cube(flux_cube, teff, logg, meta, wavelengths, flux)

    filter_file = filter_root / "GAIA" / "GAIA.dat"
    _write_filter(filter_file)

    atmospheres = SED_tools.find_atm(
        teff_range=(1000.0, 2000.0),
        logg_range=(1.0, 2.0),
        Z_range=(-1.0, 0.0),
        model_root=model_root,
    )

    assert len(atmospheres) == 1
    assert bool(atmospheres[0].contains_point)

    sed_model = SED_tools.SED(
        model_root=model_root,
        filter_root=filter_root,
        atm=atmospheres[0],
        interpolation="hermite",
        fill_gaps=True,
    )

    evaluated = sed_model(teff=1500.0, logg=1.5, z=-0.5)

    assert np.allclose(evaluated.wavelength, wavelengths)
    assert np.allclose(evaluated.flux, flux_value)

    gaia_photometry = evaluated.photometry("Gaia")

    assert set(gaia_photometry) == {"GAIA"}
    gaia = gaia_photometry["GAIA"]
    assert gaia.system == "AB"
    assert math.isfinite(gaia.magnitude)
    expected_mag = -2.5 * math.log10(gaia.flux_density / AB_ZERO_FLUX)
    assert math.isclose(gaia.magnitude, expected_mag)
