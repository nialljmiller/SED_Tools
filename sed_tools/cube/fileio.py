"""The flux cube in its two formats: flux_cube.bin (3-axis) and flux_cube_nd.bin (all-axes).

3-axis, flux_cube.bin (every grid's cube, and each variant's of a split grid; MESA reads it): four int32 axis lengths (Teff, log g, [M/H],
wavelength), the four float64 axes, then the float64 flux in (wavelength, [M/H], log g, Teff) order with Teff varying
fastest, as MESA reads it.

All-axes, flux_cube_nd.bin (the top folder of a split grid, one cube for every variant; MESA does not read it, and a
split grid's top folder holds no flux_cube.bin), little-endian: int32 -1 (where a 3-axis cube
holds its Teff count, at least 1), the magic ``FLUXCUBE``, int32 version 1, int32 P (parameter axes, wavelength not
counted), int32 n_1..n_P and n_wavelength, int32 kind_1..kind_P (0 numeric, 1 text), the P names (32 bytes each,
space-padded UTF-8), the labels of each text axis in axis order (32 bytes each), the float64 axes one after another
(a text axis stores 0..n-1), the float64 wavelengths, then the flux: one block per combination of the extra axes
(4..P, axis 4 varying fastest), each laid out exactly as a 3-axis cube's flux on axes 1-3 (Teff, log g, [M/H]).
A node no spectrum fills is zero.

Each reader checks a short or corrupt file itself.
"""

import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

ALL_AXES_NAME = "flux_cube_nd.bin"
ALL_AXES_MARKER = -1
MAGIC = b"FLUXCUBE"
VERSION = 1
NAME_BYTES = 32
NUMERIC, TEXT = 0, 1


def flux_offset(nt, nl, nm, nw):
    """Byte offset where the flux starts."""
    return 16 + 8 * (nt + nl + nm + nw)


def read_axes(fh, nt, nl, nm, nw):
    """The Teff, log g, [M/H] and wavelength axes, read from ``fh`` just after the 16-byte header."""
    return [np.fromfile(fh, dtype=np.float64, count=n) for n in (nt, nl, nm, nw)]


def is_all_axes(path) -> bool:
    """Whether the file at ``path`` starts as an all-axes cube does (its first int32 is -1)."""
    with open(path, "rb") as fh:
        return fh.read(4) == struct.pack("<i", ALL_AXES_MARKER)


def read_dims(fh, path):
    """The four axis lengths of a 3-axis cube, read from the start of ``fh``; ValueError for a short header or an
    all-axes cube, which a 3-axis reader must not read."""
    header = fh.read(16)
    if len(header) != 16:
        raise ValueError(f"Flux cube header truncated in {path}")
    dims = struct.unpack("4i", header)
    if dims[0] == ALL_AXES_MARKER:
        raise ValueError(f"{path} is the all-axes flux cube of a split grid, not a 3-axis (Teff, log g, [M/H]) "
                         "cube; use one of the grid's variant cubes, or open it with sed_tools.SED")
    return dims


def load_axes(path):
    """Teff, log g, [M/H] and wavelengths of the flux cube at ``path`` (either format), read without its flux."""
    header = read_header(path)
    return [*header.axes[:3], header.wavelengths]


def load(path):
    """Teff, log g, [M/H], wavelengths and the flux as stored, shape (nw, nm, nl, nt), of a 3-axis cube."""
    with open(path, "rb") as fh:
        nt, nl, nm, nw = read_dims(fh, path)
        teff, logg, meta, wavelengths = read_axes(fh, nt, nl, nm, nw)
        flux = np.fromfile(fh, dtype=np.float64, count=nt * nl * nm * nw).reshape((nw, nm, nl, nt))
    return teff, logg, meta, wavelengths, flux


def write_header(fh, teff, logg, meta, wavelengths):
    """The axis lengths and the four axes, as float64."""
    fh.write(struct.pack("4i", len(teff), len(logg), len(meta), len(wavelengths)))
    for axis in (teff, logg, meta, wavelengths):
        np.asarray(axis).astype(np.float64).tofile(fh)


@dataclass
class CubeHeader:
    """A flux cube's axes: ``names`` and ``axes`` of its P parameter axes (teff, logg, meta, then a split grid's
    extra axes), ``labels`` {axis number: its text values} for a text axis, the wavelengths, and where the flux
    starts. ``shape`` is the flux as stored, in C order: (n_P, ..., n_4, n_wavelength, n_meta, n_logg, n_teff)."""
    names: List[str]
    axes: List[np.ndarray]
    wavelengths: np.ndarray
    offset: int
    labels: Dict[int, List[str]] = field(default_factory=dict)
    all_axes: bool = False

    @property
    def shape(self):
        n = [len(axis) for axis in self.axes]
        return (*n[:2:-1], len(self.wavelengths), n[2], n[1], n[0])


def _text(raw: bytes) -> str:
    return raw.decode("utf-8").rstrip(" ")


def read_header(path) -> CubeHeader:
    """The header of the flux cube at ``path``, in either format; ValueError for a truncated or malformed file, or
    one whose size is not what its header gives."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if fh.read(4) != struct.pack("<i", ALL_AXES_MARKER):
            fh.seek(0)
            nt, nl, nm, nw = read_dims(fh, path)
            if min(nt, nl, nm, nw) < 1:
                raise ValueError(f"{path}: the header gives dimensions {nt}, {nl}, {nm}, {nw}")
            teff, logg, meta, wavelengths = read_axes(fh, nt, nl, nm, nw)
            if len(wavelengths) != nw:
                raise ValueError(f"Flux cube axes truncated in {path}")
            return CubeHeader(["teff", "logg", "meta"], [teff, logg, meta], wavelengths, flux_offset(nt, nl, nm, nw))
        fixed = fh.read(16)
        if len(fixed) != 16 or fixed[:8] != MAGIC:
            raise ValueError(f"{path} starts as an all-axes flux cube but has no {MAGIC.decode()} header")
        version, p = struct.unpack("<2i", fixed[8:])
        if version != VERSION or p < 3:
            raise ValueError(f"{path}: all-axes flux cube version {version} with {p} axes is not one this reads")
        counts = np.fromfile(fh, dtype="<i4", count=2 * p + 1)
        if len(counts) != 2 * p + 1 or counts[:p + 1].min() < 1:
            raise ValueError(f"{path}: the all-axes header is truncated or gives an empty axis")
        n, nw, kinds = [int(c) for c in counts[:p]], int(counts[p]), [int(k) for k in counts[p + 1:]]
        names = [_text(fh.read(NAME_BYTES)) for _ in range(p)]
        labels = {k: [_text(fh.read(NAME_BYTES)) for _ in range(n[k])] for k in range(p) if kinds[k] == TEXT}
        axes = [np.fromfile(fh, dtype="<f8", count=count) for count in n]
        wavelengths = np.fromfile(fh, dtype="<f8", count=nw)
        offset = fh.tell()
    header = CubeHeader(names, axes, wavelengths, offset, labels, all_axes=True)
    expected = offset + 8 * int(np.prod(header.shape, dtype=np.int64))
    if len(wavelengths) != nw or size != expected:
        raise ValueError(f"{path}: {size} bytes, but its all-axes header gives {expected}")
    return header


def write_all_axes_header(fh, names, axes, labels, wavelengths) -> None:
    """The header of an all-axes cube: ``names`` and ``axes`` of its parameter axes, ``labels`` {axis number: text
    values} for each text axis (whose axis is then 0..n-1), and the wavelengths. ValueError for a name or label
    longer than NAME_BYTES."""
    p = len(names)

    def fixed(text):
        raw = str(text).encode("utf-8")
        if len(raw) > NAME_BYTES:
            raise ValueError(f"'{text}' is longer than the {NAME_BYTES} bytes an all-axes flux cube stores")
        return raw.ljust(NAME_BYTES, b" ")

    fh.write(struct.pack("<i", ALL_AXES_MARKER) + MAGIC + struct.pack("<2i", VERSION, p))
    np.array([*map(len, axes), len(wavelengths)] + [TEXT if k in labels else NUMERIC for k in range(p)],
             dtype="<i4").tofile(fh)
    fh.write(b"".join(fixed(name) for name in names))
    fh.write(b"".join(fixed(label) for k in sorted(labels) for label in labels[k]))
    for axis in (*axes, wavelengths):
        np.asarray(axis, dtype="<f8").tofile(fh)
