"""Spectrum file headers: parse_header reads the '# key = value' lines of a .txt spectrum.

ALIASES maps the key names grids use to the canonical teff, logg and metallicity.
"""

from __future__ import annotations

import math
import re
from typing import Dict

# Header keys (lowercased) and the canonical name each is read as. To read a new
# grid's naming convention, add its keys here.
ALIASES: Dict[str, str] = {
    **dict.fromkeys(["teff", "t_eff", "t eff", "t-eff", "effective_temperature", "temperature", "temp"], "teff"),
    **dict.fromkeys(["logg", "log_g", "log(g)", "log g", "surface_gravity", "gravity"], "logg"),
    **dict.fromkeys(["metallicity", "meta", "feh", "[fe/h]", "[m/h]", "m/h", "m_h", "mh",
                     "z",  # TLUSTY OSTAR/BSTAR convention
                     "zh", "zmet", "z/z0", "z/zsun"], "metallicity"),
    # bookkeeping keys, kept under their own name
    **{key: key for key in ["source", "spec_group", "units_standardized", "wavelength_unit", "flux_unit",
                            "original_wavelength_unit", "original_flux_unit", "conversion_confidence"]},
}

# An int or float, including scientific notation
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Canonical numeric parameters — values stored as float, nan if unparseable
_NUMERIC_PARAMS = ("teff", "logg", "metallicity")


def parse_header(filepath: str) -> Dict[str, object]:
    """Parse the comment header of a spectrum .txt file.

    Returns a flat dict with canonical keys. Numeric parameters (teff, logg,
    metallicity) are stored as float (nan if the value couldn't be parsed).
    All other recognised keys are stored as strings. Unrecognised keys are
    stored under their original lowercased name, so no information is lost.

    Parameters
    ----------
    filepath : str
        Path to the spectrum file.

    Returns
    -------
    dict
        Keys include at minimum 'teff', 'logg', 'metallicity' (all float).
        Additional keys depend on what the file header contains.

    Raises
    ------
    OSError
        If the file cannot be opened or read.
    """
    result: Dict[str, object] = {param: float("nan") for param in _NUMERIC_PARAMS}
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
                if not math.isnan(parsed) or math.isnan(result[canonical]):
                    result[canonical] = parsed
            else:
                result[canonical] = val_clean
    return result
