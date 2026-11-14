"""A grid's text spectra and its HDF5 bundle: list_text_spectra, read_text_spectrum, build_h5_bundle and
bundled_spectra, with where the bundle goes (_bundle_path); and write_atomically, the one way a downloaded or
rewritten spectrum is written."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Union

import h5py
import numpy as np

from .header_parser import parse_header

PathLike = Union[str, Path]


def write_atomically(path: PathLike, write: Callable[[str], None]) -> None:
    """Write ``path`` whole or not at all: ``write(part)`` writes the new content to ``part`` (``path`` + '.part'),
    which then replaces ``path``. If ``write`` raises, or the process stops first, ``path`` is as it was and the part
    is gone (removed, or left under its own name), so a file at ``path`` is always complete."""
    part = f"{path}.part"
    try:
        write(part)
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)


def list_text_spectra(model_dir: PathLike) -> list[Path]:
    """Return the top-level ``.txt`` spectra in stable name order."""
    root = Path(model_dir)
    return sorted((path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".txt"), key=lambda p: p.name)


def read_text_spectrum(path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Read the first two finite numeric columns of a text spectrum.

    Rows whose first two fields are not numbers are skipped. Raises OSError
    if the file cannot be read.
    """
    wavelength: list[float] = []
    flux: list[float] = []
    with Path(path).expanduser().open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith(("#", ";", "!")):
                continue
            fields = text.replace(",", " ").split()
            if len(fields) < 2:
                continue
            try:
                x, y = float(fields[0]), float(fields[1])
            except ValueError:
                continue
            if np.isfinite(x) and np.isfinite(y):
                wavelength.append(x)
                flux.append(y)
    return np.asarray(wavelength, dtype=float), np.asarray(flux, dtype=float)


def bundled_spectra(bundle_path: PathLike) -> list[str]:
    """The file names of the spectra an HDF5 bundle from :func:`build_h5_bundle` holds.

    Raises ValueError when the file has no ``spectra`` group, and OSError when it cannot be read.
    """
    with h5py.File(bundle_path, "r") as h5:
        if not isinstance(h5.get("spectra"), h5py.Group):
            raise ValueError(f"{bundle_path} has no 'spectra' group")
        return sorted(h5["spectra"].keys())


def _bundle_path(model_dir: PathLike, name: str) -> Path:
    """Where grid ``name``'s HDF5 bundle goes: ``{name}.h5``, unless a file there is not a bundle (MSG's downloaded
    grid has that name), which is never written over; then ``{name}_bundle.h5``."""
    path = Path(model_dir) / f"{name}.h5"
    if path.exists():
        try:
            bundled_spectra(path)
        except (OSError, ValueError):
            return path.with_name(f"{name}_bundle.h5")
    return path


def build_h5_bundle(model_dir: PathLike, output_path: PathLike) -> bool:
    """Bundle text spectra and canonical header metadata; return whether written."""
    spectra = list_text_spectra(model_dir)
    if not spectra:
        return False
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as h5:
        group = h5.create_group("spectra")
        for path in spectra:
            wavelength, flux = read_text_spectrum(path)
            item = group.create_group(path.name)
            item.create_dataset("lambda", data=wavelength, dtype="f8")
            item.create_dataset("flux", data=flux, dtype="f8")
            for key, value in parse_header(str(path)).items():
                if isinstance(value, float) and np.isnan(value):
                    continue
                if key == "metallicity":
                    item.attrs["feh"] = value
                if key in {"teff", "logg"}:
                    item.attrs[key] = value
                item.attrs[f"raw:{key}"] = value
    return True
