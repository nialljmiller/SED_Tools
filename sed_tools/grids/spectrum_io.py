"""Canonical text-spectrum I/O and simple HDF5 bundle support."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np

from .header_parser import parse_header

PathLike = Union[str, Path]


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
