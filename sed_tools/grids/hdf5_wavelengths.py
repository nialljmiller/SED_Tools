"""Real wavelengths for index-grid spectra, recovered from the grid's HDF5 files (MSG grids)."""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger("sed_tools.spectra_cleaner")


WAVE_KEYS = ('lambda', 'wavelength', 'wave', 'wl', 'wavelength_A', 'x')

def _is_physical_wavelength_array(wl: np.ndarray) -> bool:
    """Return True if wl looks like a physical wavelength grid in Angstroms."""
    if wl.size < 2:
        return False
    if wl[0] < 1.0 or wl[-1] > 1e9:
        return False
    # Reject pure integer index sequences
    if wl[0] < 2 and np.allclose(wl, np.arange(wl.size, dtype=float), atol=0.5):
        return False
    if wl[0] < 2 and np.allclose(wl, np.arange(1, wl.size + 1, dtype=float), atol=0.5):
        return False
    if np.allclose(np.diff(wl), 1.0, atol=0.01):
        return False
    return True


def recover_wavelengths_from_hdf5(
    model_dir: str,
    spec_group: str,
    expected_len: int
) -> Optional[np.ndarray]:
    """
    Attempt to recover wavelength array from HDF5 file for index grids.

    Searches every *.h5 in model_dir, in priority order:
      1. Direct wavelength dataset within spec_group
      2. range/x structure within spec_group
      3. Top-level wavelength arrays in the HDF5 root
      4. Global search for any physically plausible array of the right length

    An unreadable .h5 is reported with a warning and the next one is tried.
    Returns None if no physical wavelength grid is found.
    """
    for h5_path in glob.glob(os.path.join(model_dir, "*.h5")):
        try:
            with h5py.File(h5_path, 'r') as f:
                wl = _wavelengths_in_h5(f, spec_group, expected_len)
        except OSError as exc:
            logger.warning("Skipping unreadable HDF5 file %s while recovering wavelengths "
                           "for %s: %s", h5_path, spec_group, exc)
            continue
        if wl is not None:
            return wl
    return None


def _wavelengths_in_h5(f, spec_group: str, expected_len: int) -> Optional[np.ndarray]:
    """Apply the four recovery searches to one open HDF5 file."""
    if spec_group in f:
        wl = _wavelengths_in_group(f[spec_group], expected_len)
        if wl is not None:
            return wl
    wl = _top_level_wavelengths(f, expected_len)
    if wl is not None:
        return wl
    return _search_all_datasets(f, expected_len)


def _increasing_physical_prefix(wl: np.ndarray, expected_len: int) -> Optional[np.ndarray]:
    """wl[:expected_len] if wl is long enough and that prefix is a rising physical grid."""
    if (wl.size >= expected_len and
            np.all(np.diff(wl[:expected_len]) > 0) and
            _is_physical_wavelength_array(wl[:expected_len])):
        return wl[:expected_len]
    return None


def _wavelengths_in_group(group, expected_len: int) -> Optional[np.ndarray]:
    """Searches 1 and 2: a wavelength dataset in the spectrum's group, then its range/ subgroup."""
    for key in WAVE_KEYS:
        if key in group and isinstance(group[key], h5py.Dataset):
            wl = _increasing_physical_prefix(np.array(group[key]).astype(float).ravel(), expected_len)
            if wl is not None:
                return wl
    if 'range' in group and isinstance(group['range'], h5py.Group):
        return _wavelengths_in_range_group(group['range'], expected_len)
    return None


def _segment_number(name: str) -> int:
    match = re.search(r'\d+', name)
    return int(match.group()) if match else 0


def _wavelengths_in_range_group(rg, expected_len: int) -> Optional[np.ndarray]:
    """range/x, else the range segments' wavelength datasets concatenated in numeric order."""
    if 'x' in rg and isinstance(rg['x'], h5py.Dataset):
        wl = _increasing_physical_prefix(np.array(rg['x']).astype(float).ravel(), expected_len)
        if wl is not None:
            return wl

    segs = []
    seg_names = sorted([k for k in rg.keys() if 'range' in k.lower()], key=_segment_number)
    for sn in seg_names:
        if not isinstance(rg[sn], h5py.Group):
            continue
        sg = rg[sn]
        for key in WAVE_KEYS:
            if key in sg and isinstance(sg[key], h5py.Dataset):
                segs.append(np.array(sg[key]).astype(float).ravel())
                break
    if segs:
        wl = np.concatenate(segs)
        if wl.size >= expected_len and _is_physical_wavelength_array(wl[:expected_len]):
            return wl[:expected_len]
    return None


def _top_level_wavelengths(f, expected_len: int) -> Optional[np.ndarray]:
    """Search 3: a root-level wavelength dataset within two points of the expected length."""
    for key in WAVE_KEYS:
        if key in f and isinstance(f[key], h5py.Dataset):
            wl = np.array(f[key]).astype(float).ravel()
            if (wl.size >= 2 and np.all(np.diff(wl) > 0) and
                    _is_physical_wavelength_array(wl) and
                    abs(wl.size - expected_len) <= 2):
                return wl[:expected_len] if wl.size > expected_len else wl
    return None


def _iter_datasets(group):
    for item in group.values():
        if isinstance(item, h5py.Dataset):
            yield item
        elif isinstance(item, h5py.Group):
            yield from _iter_datasets(item)


def _search_all_datasets(f, expected_len: int) -> Optional[np.ndarray]:
    """Search 4: any rising physical numeric dataset of exactly the expected length."""
    for dataset in _iter_datasets(f):
        try:
            arr = np.array(dataset[()]).astype(float).ravel()
        except (TypeError, ValueError):
            continue  # string and compound datasets cannot be wavelength grids
        if (arr.size == expected_len and
                np.all(np.diff(arr) > 0) and
                _is_physical_wavelength_array(arr)):
            return arr
    return None


def is_index_grid(wl: np.ndarray) -> bool:
    """Detect if wavelength array is actually an index grid (0, 1, 2, ...)."""
    if wl.size < 4:
        return False
    
    # Must start near 0 or 1
    if wl[0] > 2:
        return False
    
    # Check if it matches 0..N-1 or 1..N
    if np.allclose(wl, np.arange(wl.size), atol=0.01):
        return True
    if np.allclose(wl, np.arange(1, wl.size + 1), atol=0.01):
        return True
    
    # Check uniform step of ~1
    steps = np.diff(wl)
    if np.allclose(steps, 1.0, atol=0.01):
        return True
    
    return False
