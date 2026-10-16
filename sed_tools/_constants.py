"""
sed_tools/_constants.py

Physical constants and the canonical Planck blackbody function.

All modules that need these quantities should import from here.
Adding support for a new grid never requires touching this file;
all physics is kept in one place.
"""

from __future__ import annotations

import numpy as np

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
