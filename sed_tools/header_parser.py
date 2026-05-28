"""
sed_tools/header_parser.py

Single source of truth for parsing spectrum file headers.

All modules that need to read parameter metadata from .txt spectrum files
(svo_spectra_grabber, svo_spectra_filter, svo_regen_spectra_lookup,
spectra_cleaner, njm_spectra_grabber, precompute_flux_cube) should import
parse_header() from here rather than implementing their own logic.

To add support for a new key alias, edit the ALIASES table below — that is
the only place a developer needs to touch.
"""

from __future__ import annotations

import re
from typing import Dict, Optional
import math

# =============================================================================
# ALIAS TABLE
# All keys are lowercased before lookup.
# Each entry maps one or more raw header key variants to a canonical name.
# To support a new grid's naming convention, add its key(s) here.
# =============================================================================

ALIASES: Dict[str, str] = {
    # ---- effective temperature ----
    "teff":                  "teff",
    "t_eff":                 "teff",
    "t eff":                 "teff",
    "t-eff":                 "teff",
    "effective_temperature": "teff",
    "temperature":           "teff",
    "temp":                  "teff",

    # ---- surface gravity ----
    "logg":                  "logg",
    "log_g":                 "logg",
    "log(g)":                "logg",
    "log g":                 "logg",
    "surface_gravity":       "logg",
    "gravity":               "logg",

    # ---- metallicity ----
    "metallicity":           "metallicity",
    "meta":                  "metallicity",
    "feh":                   "metallicity",
    "[fe/h]":                "metallicity",
    "[m/h]":                 "metallicity",
    "m/h":                   "metallicity",
    "m_h":                   "metallicity",
    "mh":                    "metallicity",
    "z":                     "metallicity",   # TLUSTY OSTAR/BSTAR convention
    "zh":                    "metallicity",
    "zmet":                  "metallicity",
    "z/z0":                  "metallicity",
    "z/zsun":                "metallicity",

    # ---- bookkeeping (passed through as-is under canonical name) ----
    "source":                "source",
    "spec_group":            "spec_group",
    "units_standardized":    "units_standardized",
    "wavelength_unit":       "wavelength_unit",
    "flux_unit":             "flux_unit",
    "original_wavelength_unit": "original_wavelength_unit",
    "original_flux_unit":    "original_flux_unit",
    "conversion_confidence": "conversion_confidence",
}

# Numeric regex — matches int or float including scientific notation
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Canonical numeric parameters — values stored as float, nan if unparseable
_NUMERIC_PARAMS = {"teff", "logg", "metallicity"}


def parse_header(filepath: str) -> Dict[str, object]:
    """
    Parse the comment header of a spectrum .txt file.
    Returns a flat dict with canonical keys. Numeric parameters (teff, logg,
    metallicity) are stored as float (nan if the value couldn't be parsed).
    All other recognised keys are stored as strings. Unrecognised keys are
    stored under their original lowercased name in the returned dict so no
    information is lost.
    Parameters
    ----------
    filepath : str
        Path to the spectrum file.
    Returns
    -------
    dict
        Keys include at minimum 'teff', 'logg', 'metallicity' (all float).
        Additional keys depend on what the file header contains.
    """
    result: Dict[str, object] = {
        "teff":        float("nan"),
        "logg":        float("nan"),
        "metallicity": float("nan"),
    }
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped.startswith("#"):
                    break
                if "=" not in stripped:
                    continue
                raw_key, _, raw_val = stripped.lstrip("#").partition("=")
                key_lower = raw_key.strip().lower()
                # Strip parenthetical unit annotations: "15000 K (...)" → "15000"
                val_clean = raw_val.split("(")[0].strip()
                canonical = ALIASES.get(key_lower, key_lower)

                if canonical in _NUMERIC_PARAMS:
                    m = _NUM_RE.search(val_clean)
                    parsed = float(m.group(0)) if m else float("nan")
                    # never let nan overwrite an already-finite value
                    if not (math.isnan(parsed) and not math.isnan(result.get(canonical, float("nan")))):
                        result[canonical] = parsed
                else:
                    result[canonical] = val_clean
    except Exception:
        pass
    return result
