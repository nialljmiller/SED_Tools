"""Shared filesystem and spectrum file helpers for SED Tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from .header_parser import parse_header

PathLike = Union[str, os.PathLike[str]]


def ensure_dir(path: PathLike) -> None:
    """Create *path* and any missing parents if it is not empty."""
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def list_txt_spectra(model_dir: PathLike) -> list[str]:
    """Return sorted ``.txt`` spectrum file names in *model_dir*."""
    return sorted(
        entry.name
        for entry in Path(model_dir).iterdir()
        if entry.is_file() and entry.suffix.lower() == ".txt"
    )


def load_txt_spectrum(txt_path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Load wavelength and flux columns from a two-column text spectrum."""
    wavelength: list[float] = []
    flux: list[float] = []
    with Path(txt_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            wavelength.append(float(parts[0]))
            flux.append(float(parts[1]))
    return np.asarray(wavelength, dtype=float), np.asarray(flux, dtype=float)


def build_h5_bundle_from_txt(model_dir: PathLike, out_h5: PathLike) -> None:
    """Bundle all text spectra in *model_dir* into one HDF5 file."""
    model_path = Path(model_dir)
    out_path = Path(out_h5)
    txt_files = list_txt_spectra(model_path)
    if not txt_files:
        print(f"[H5 bundle] No .txt spectra found in {model_path}; skipping.")
        return

    ensure_dir(out_path.parent)
    with h5py.File(out_path, "w") as h5:
        spectra_group = h5.create_group("spectra")
        for filename in txt_files:
            path = model_path / filename
            wavelength, flux = load_txt_spectrum(path)
            if wavelength.size == 0 or flux.size == 0:
                print(f"[H5 bundle] Empty or invalid spectrum: {filename}")
                continue

            group = spectra_group.create_group(filename)
            group.create_dataset("lambda", data=wavelength, dtype="f8")
            group.create_dataset("flux", data=flux, dtype="f8")

            metadata = parse_header(path)
            teff = metadata.get("teff", np.nan)
            logg = metadata.get("logg", np.nan)
            metallicity = metadata.get("metallicity", np.nan)
            if not np.isnan(teff):
                group.attrs["teff"] = teff
            if not np.isnan(logg):
                group.attrs["logg"] = logg
            if not np.isnan(metallicity):
                group.attrs["feh"] = metallicity
            for key, value in metadata.items():
                group.attrs[f"raw:{key}"] = value

    print(f"[H5 bundle] Wrote {out_path}")
