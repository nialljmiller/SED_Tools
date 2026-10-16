"""Shared pytest fixtures for the SED_Tools test suite.

Centralizes the flux-cube binary writer and the analytic-field cube
factory used by the interpolation/cube-correctness tests, so this logic
lives in one place instead of being copy-pasted (and drifting) across
individual test files.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest


def write_flux_cube(
    path: Path,
    teff: Sequence[float],
    logg: Sequence[float],
    meta: Sequence[float],
    wavelengths: Sequence[float],
    flux: np.ndarray,
) -> None:
    """Write a ``flux_cube.bin`` in the format read by
    ``sed_tools.flux_cube_tool.FluxCube.from_file`` and
    ``sed_tools.models._read_flux_cube_header``.

    ``flux`` must be supplied with shape ``(nt, nl, nm, nw)`` — the
    in-memory convention used throughout ``sed_tools``. On disk the
    array is stored axis-reversed as ``(nw, nm, nl, nt)``, which is why
    the ``.transpose(3, 2, 1, 0)`` below matters: getting this backwards
    is exactly the historical "Flux cube axis mismatch" bug recorded in
    the changelog, so this helper is the single place that encodes the
    correct on-disk layout for tests.
    """
    teff = np.asarray(teff, dtype=np.float64)
    logg = np.asarray(logg, dtype=np.float64)
    meta = np.asarray(meta, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    expected_shape = (teff.size, logg.size, meta.size, wavelengths.size)
    if flux.shape != expected_shape:
        raise ValueError(f"flux shape {flux.shape} != expected {expected_shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("4i", teff.size, logg.size, meta.size, wavelengths.size))
        for grid in (teff, logg, meta, wavelengths):
            grid.tofile(fh)
        flux.transpose(3, 2, 1, 0).ravel().tofile(fh)


@pytest.fixture
def cube_writer() -> Callable[..., None]:
    """Expose :func:`write_flux_cube` to tests as a fixture."""
    return write_flux_cube


@dataclass
class AffineCube:
    """An analytic flux cube: F(T, g, Z, wl) = a + b*T + c*g + d*Z + e*T*g.

    This is reproduced *exactly* by the cubic-Hermite interpolation in
    ``FluxCube.interpolate_spectrum`` — each sequential 1-D interpolation
    step is linear-exact for affine/bilinear inputs — so any mismatch
    against :meth:`expected` indicates a real interpolation or
    axis-ordering bug, not curvature/approximation error.
    """

    path: Path
    teff: np.ndarray
    logg: np.ndarray
    meta: np.ndarray
    wavelengths: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    e: np.ndarray

    def expected(self, teff: float, logg: float, meta: float) -> np.ndarray:
        return self.a + self.b * teff + self.c * logg + self.d * meta + self.e * teff * logg


@pytest.fixture
def make_affine_cube(tmp_path, cube_writer) -> Callable[..., AffineCube]:
    """Factory fixture building an analytic-field flux cube on disk.

    Coefficients are distinct per wavelength *and* per axis, and the
    default grids are non-uniformly spaced, so a swapped or scrambled
    axis produces a detectably wrong answer rather than accidentally
    matching by symmetry.
    """

    def _make(
        teff=(4000.0, 4500.0, 6000.0, 6200.0),
        logg=(1.0, 2.5, 5.0),
        meta=(-2.0, -0.3, 0.0),
        wavelengths=(4000.0, 4500.0, 6000.0),
        name="affine_demo",
        include_cross_term=True,
    ) -> AffineCube:
        teff = np.asarray(teff, dtype=np.float64)
        logg = np.asarray(logg, dtype=np.float64)
        meta = np.asarray(meta, dtype=np.float64)
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        nw = wavelengths.size

        rng = np.arange(nw, dtype=np.float64)
        a = 1.0 + rng
        b = 0.0010 + 0.0003 * rng  # teff coefficient
        c = 0.30 - 0.05 * rng      # logg coefficient
        d = 0.90 + 0.20 * rng      # metallicity coefficient
        e = (1e-6 * (1.0 + rng)) if include_cross_term else np.zeros(nw)

        flux = np.zeros((teff.size, logg.size, meta.size, nw))
        for ti, T in enumerate(teff):
            for li, L in enumerate(logg):
                for mi, M in enumerate(meta):
                    flux[ti, li, mi, :] = a + b * T + c * L + d * M + e * T * L

        path = tmp_path / name / "flux_cube.bin"
        write_flux_cube(path, teff, logg, meta, wavelengths, flux)
        return AffineCube(path, teff, logg, meta, wavelengths, a, b, c, d, e)

    return _make
