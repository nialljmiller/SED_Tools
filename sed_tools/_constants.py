"""
sed_tools/_constants.py

Physical constants and the canonical Planck blackbody function, plus the
available-RAM measurement that sizes memory use when flux cubes are written.

All modules that need these quantities should import from here.
Adding support for a new grid never requires touching this file;
all physics is kept in one place.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

# Physical constants (CGS)
H    = 6.62607015e-27   # erg s       — Planck constant
C    = 2.99792458e10    # cm/s        — speed of light
K    = 1.380649e-16     # erg/K       — Boltzmann constant
SIGMA = 5.670374419e-5  # erg/s/cm²/K⁴ — Stefan-Boltzmann constant


def planck_flam(wl_ang: np.ndarray, teff: float) -> np.ndarray:
    """π·B_λ(T) in erg/cm²/s/Å, safe against overflow at short wavelengths.

    Parameters
    ----------
    wl_ang : array
        Wavelength in Angstroms (must be positive).
    teff : float
        Effective temperature in Kelvin.

    Returns
    -------
    array
        Hemisphere-integrated Planck flux in erg/cm²/s/Å.
    """
    wl_cm = np.asarray(wl_ang, dtype=np.float64) * 1e-8
    exp = np.minimum((H * C) / (wl_cm * K * teff), 709.0)
    return np.pi * (2.0 * H * C**2 / wl_cm**5) / (np.exp(exp) - 1.0) / 1e8


# ---------------------------------------------------------------------------
# Available RAM
# ---------------------------------------------------------------------------

_MEMINFO = "/proc/meminfo"
# When only the physical RAM can be read (macOS without psutil), half of it is
# taken as available, leaving the rest to the system and other programs.
# Callers keep their own margin on top: precompute_flux_cube plans for 75% of
# the figure and the densifier for half, so at most 3/8 of physical RAM.
_SHARE_OF_PHYSICAL_RAM = 0.5
_ASSUMED_RAM_BYTES = 2 * (1024 ** 3)


def available_memory_bytes() -> int:
    """Available RAM in bytes. It only sizes memory use; no result depends on it.

    Measured by psutil if it is installed, else by MemAvailable in /proc/meminfo
    (Linux), else as half the physical RAM given by os.sysconf (macOS). If none
    of these works, a warning is logged and 2 GiB is assumed.
    """
    if psutil is not None:
        return psutil.virtual_memory().available
    reasons = ["psutil is not installed"]
    try:
        available = _meminfo_available_bytes()
    except OSError as exc:
        reasons.append(f"{_MEMINFO} could not be read ({exc})")
    else:
        if available is not None:
            return available
        reasons.append(f"{_MEMINFO} has no MemAvailable line")
    try:
        return int(_physical_memory_bytes() * _SHARE_OF_PHYSICAL_RAM)
    except (AttributeError, ValueError, OSError) as exc:
        # os.sysconf does not exist on Windows; an unknown name raises ValueError.
        reasons.append(f"os.sysconf could not give the physical RAM ({exc})")
    logger.warning(
        "Could not measure available RAM: %s. Assuming %d GiB; this affects "
        "memory use only, not the flux cube.", "; ".join(reasons), _ASSUMED_RAM_BYTES // 1024**3,
    )
    return _ASSUMED_RAM_BYTES


def _meminfo_available_bytes() -> Optional[int]:
    """MemAvailable from /proc/meminfo in bytes, or None if the file has no such line."""
    with open(_MEMINFO) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def _physical_memory_bytes() -> int:
    """Physical RAM in bytes from os.sysconf."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    if page_size <= 0 or pages <= 0:
        raise ValueError(f"SC_PAGE_SIZE is {page_size} and SC_PHYS_PAGES is {pages}")
    return page_size * pages
