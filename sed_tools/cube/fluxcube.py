"""FluxCube: a 3-axis flux_cube.bin (the format is in cube/fileio.py) in memory, with cubic Hermite interpolation in
Teff, log g and [M/H]. SEDModel (cube/model.py) and the flux cube tool (cube/tool.py, which re-exports it) use it.
AllAxesCube: a split grid's all-axes flux_cube_nd.bin, memory-mapped, with the same cubic Hermite on every numeric axis.
"""

from __future__ import annotations

import struct  # noqa: F401  (kept: older code star-imports this module)
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .fileio import CubeHeader, read_axes, read_dims


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
            nt, nl, nm, nw = read_dims(fh, path)
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


def _hermite_stencil(grid: np.ndarray, x: float) -> Tuple[slice, np.ndarray]:
    """FluxCube's cubic Hermite along one axis at ``x``, as weights on at most four neighbouring nodes:
    (the nodes, as a slice of ``grid``; their weights). The slopes are FluxCube's finite differences, so this is
    FluxCube's interpolation; at a node the weights are exactly 1 there and 0 elsewhere. A length-1 axis gives its
    node. ValueError outside the grid."""
    n = len(grid)
    if n < 2:
        return slice(0, 1), np.ones(1)
    if not grid[0] <= x <= grid[-1]:
        raise ValueError(f"Requested value {x} is outside interpolation range [{grid[0]}, {grid[-1]}]")
    idx = max(0, min(int(np.searchsorted(grid, x)) - 1, n - 2))
    h = grid[idx + 1] - grid[idx]
    t = (x - grid[idx]) / h if h != 0 else 0.0
    weights = {idx: 2 * t**3 - 3 * t**2 + 1, idx + 1: -2 * t**3 + 3 * t**2}
    for node, factor in ((idx, (t**3 - 2 * t**2 + t) * h), (idx + 1, (t**3 - t**2) * h)):
        lo, hi = (0, 1) if node <= 0 else (n - 2, n - 1) if node >= n - 1 else (node - 1, node + 1)
        slope = factor / (grid[hi] - grid[lo])
        weights[hi] = weights.get(hi, 0.0) + slope
        weights[lo] = weights.get(lo, 0.0) - slope
    nodes = sorted(node for node, weight in weights.items() if weight != 0)
    return slice(nodes[0], nodes[-1] + 1), np.array([weights.get(k, 0.0) for k in range(nodes[0], nodes[-1] + 1)])


@dataclass
class AllAxesCube:
    """A split grid's all-axes flux cube (cube/fileio.py), its flux memory-mapped and never read whole."""

    header: CubeHeader
    flux: np.ndarray  # as stored: (n_P, ..., n_4, nw, n_meta, n_logg, n_teff)

    @classmethod
    def from_header(cls, path, header: CubeHeader) -> "AllAxesCube":
        return cls(header, np.memmap(path, dtype="<f8", mode="r", offset=header.offset, shape=header.shape))

    def interpolate_spectrum(self, values: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        """``(wavelength, flux)`` at ``values``, one per parameter axis: a number on a numeric axis (cubic Hermite,
        as FluxCube), the index of a label on a text axis (exact). Nodes no spectrum fills are blended in as zeros,
        as in FluxCube. Read a part of wavelengths at a time, so memory stays near 256 MiB."""
        stencils = [(slice(int(v), int(v) + 1), np.ones(1)) if k in self.header.labels
                    else _hermite_stencil(axis, float(v))
                    for k, (axis, v) in enumerate(zip(self.header.axes, values))]
        corners = int(np.prod([len(weights) for _, weights in stencils]))
        nw = len(self.header.wavelengths)
        step = max(1, (256 * 2**20) // (8 * corners))
        p = len(stencils)
        # stored order (P..4, w, meta, logg, teff) to (teff, logg, meta, 4..P, w), so axis k is contracted k-th
        order = [p, p - 1, p - 2, *range(p - 4, -1, -1), p - 3]
        flux = np.empty(nw)
        for w0 in range(0, nw, step):
            where = (*(s for s, _ in stencils[:2:-1]), slice(w0, w0 + step), *(s for s, _ in stencils[2::-1]))
            part = np.asarray(self.flux[where], dtype=np.float64).transpose(order)
            for _, weights in stencils:
                part = np.tensordot(weights, part, axes=(0, 0))
            flux[w0:w0 + step] = part
        return self.header.wavelengths.copy(), flux
