"""
sed_tools/_resample.py

Single authoritative function for resampling a spectrum onto a target
wavelength grid.  Used by both precompute_flux_cube.py and
combine_stellar_atm.py so the behaviour is identical for every stellar
atmosphere model (Kurucz, BT-Settl, NextGen, SPHINX, tmap, bbody, …).

Rule:
  - Inside the spectrum's actual wavelength coverage  → log-linear interp
  - Outside coverage                                  → blackbody scaled to
                                                        match the bolometric
                                                        flux of the real data

The blackbody extrapolation is model-agnostic: at the boundaries a stellar
atmosphere must approach a Planck function, so this is physically motivated
regardless of which grid you are processing.
"""

from __future__ import annotations

import numpy as np

# Physical constants (CGS)
_H    = 6.62607015e-27    # erg s
_C    = 2.99792458e18     # Å/s
_K    = 1.380649e-16      # erg/K
_SIGM = 5.670374419e-5    # erg s-1 cm-2 K-4

# Minimum sensible flux to avoid log(0)
_FLUX_FLOOR = 1e-300


def _planck_flam(wl_ang: np.ndarray, teff: float) -> np.ndarray:
    """
    Planck function in erg/cm²/s/Å (hemisphere-integrated, i.e. π × B_λ).
    Safe against overflow at short wavelengths.
    """
    l   = wl_ang * 1e-8                         # Å → cm
    exp = np.minimum(_H * _C / (l * _K * teff), 700.0)
    return np.pi * (2 * _H * _C**2 / l**5) / (np.exp(exp) - 1.0) * 1e-8


def resample_to_grid(
    wl_src:  np.ndarray,
    fl_src:  np.ndarray,
    teff:    float,
    wl_tgt:  np.ndarray,
) -> np.ndarray:
    """
    Resample a stellar spectrum onto *wl_tgt*.

    Parameters
    ----------
    wl_src : array
        Source wavelengths in Angstroms, must be strictly increasing.
    fl_src : array
        Source flux in erg/cm²/s/Å, same length as wl_src.
    teff : float
        Effective temperature of the star in Kelvin.  Used only to set
        the *shape* of the blackbody extrapolation; the amplitude is
        determined from the spectrum's own bolometric flux.
    wl_tgt : array
        Target wavelength grid in Angstroms.

    Returns
    -------
    fl_tgt : array
        Flux on *wl_tgt* in erg/cm²/s/Å.  Always >= 0.
    """
    wl_src = np.asarray(wl_src, dtype=np.float64)
    fl_src = np.asarray(fl_src, dtype=np.float64)
    wl_tgt = np.asarray(wl_tgt, dtype=np.float64)

    wl_min = float(wl_src[0])
    wl_max = float(wl_src[-1])

    # ------------------------------------------------------------------
    # Interpolate within coverage in log space.
    # Outside coverage: zero.  We do not invent flux the model does not
    # provide.  The bolometric integral is dominated by the covered range
    # and injecting a blackbody outside it inflates the integral.
    # ------------------------------------------------------------------
    fl_safe = np.maximum(fl_src, _FLUX_FLOOR)
    log_fl  = np.log10(fl_safe)
    in_cov  = (wl_tgt >= wl_min) & (wl_tgt <= wl_max)

    fl_tgt = np.zeros(len(wl_tgt), dtype=np.float64)

    if in_cov.any():
        fl_tgt[in_cov] = 10.0 ** np.interp(
            wl_tgt[in_cov], wl_src, log_fl
        )

    return fl_tgt