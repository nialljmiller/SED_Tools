"""FluxCube: a flux_cube.bin (the format is in cube/fileio.py) in memory, with cubic Hermite interpolation in Teff,
log g and [M/H]. SEDModel (cube/model.py) and the flux cube tool (cube/tool.py, which re-exports it) use it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .fileio import read_axes


@dataclass
class FluxCube:
    """In-memory representation of a pre-computed flux cube."""

    teff_grid: np.ndarray
    logg_grid: np.ndarray
    meta_grid: np.ndarray
    wavelengths: np.ndarray
    flux: np.ndarray  # shape (nt, nl, nm, nw)

    @classmethod
    def from_file(cls, path: str) -> "FluxCube":
        """Load a :class:`FluxCube` from a binary file."""
        with open(path, "rb") as fh:
            header = fh.read(16)
            if len(header) != 16:
                raise ValueError("Flux cube header truncated")
            nt, nl, nm, nw = struct.unpack("4i", header)
            teff, logg, meta, wavelengths = read_axes(fh, nt, nl, nm, nw)
            flux_expected = nt * nl * nm * nw
            flux_flat = np.fromfile(fh, dtype=np.float64, count=flux_expected)
            leftover = fh.read(1) if flux_flat.size == flux_expected else b""

        if flux_flat.size != flux_expected:
            raise ValueError(f"Flux cube body truncated: expected {flux_expected} values, found {flux_flat.size}")

        if leftover:
            print("Warning: trailing data detected in flux cube; ignoring extra bytes.")

        # The body is stored in Fortran order (Teff varies fastest) for MESA,
        # so read it as (nw, nm, nl, nt) and transpose back.
        flux = flux_flat.reshape((nw, nm, nl, nt)).transpose(3, 2, 1, 0)
        return cls(teff, logg, meta, wavelengths, flux)

    def _hermite_interp_axis(self, values: np.ndarray, grid: np.ndarray, x: float, axis: int) -> np.ndarray:
        """Cubic Hermite interpolation along one axis."""
        if values.shape[axis] < 2 or len(grid) < 2:
            # Degenerate axis: return the lone slice without interpolation.
            return np.take(values, 0, axis=axis)

        if not (grid[0] <= x <= grid[-1]):
            raise ValueError(f"Requested value {x} is outside interpolation range [{grid[0]}, {grid[-1]}]")

        idx = int(np.searchsorted(grid, x) - 1)
        idx = max(0, min(idx, len(grid) - 2))

        x0 = grid[idx]
        h = grid[idx + 1] - x0
        t = (x - x0) / h if h != 0 else 0.0

        f0 = np.take(values, idx, axis=axis)
        f1 = np.take(values, idx + 1, axis=axis)

        m0 = self._axis_derivative(values, grid, idx, axis)
        m1 = self._axis_derivative(values, grid, idx + 1, axis)

        h00 = (2 * t**3) - (3 * t**2) + 1
        h10 = (t**3) - (2 * t**2) + t
        h01 = (-2 * t**3) + (3 * t**2)
        h11 = (t**3) - (t**2)

        return h00 * f0 + h10 * h * m0 + h01 * f1 + h11 * h * m1

    def _axis_derivative(self, values: np.ndarray, grid: np.ndarray, idx: int, axis: int) -> np.ndarray:
        n = len(grid)
        if idx <= 0:
            lo, hi = 0, 1
        elif idx >= n - 1:
            lo, hi = n - 2, n - 1
        else:
            lo, hi = idx - 1, idx + 1
        return (np.take(values, hi, axis=axis) - np.take(values, lo, axis=axis)) / (grid[hi] - grid[lo])

    def interpolate_spectrum(self, teff: float, logg: float, meta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(wavelength, flux)`` at the requested parameter point."""
        tmp = self._hermite_interp_axis(self.flux, self.teff_grid, teff, axis=0)
        tmp = self._hermite_interp_axis(tmp, self.logg_grid, logg, axis=0)
        flux = self._hermite_interp_axis(tmp, self.meta_grid, meta, axis=0)
        return self.wavelengths.copy(), flux
