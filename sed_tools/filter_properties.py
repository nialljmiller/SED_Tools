"""Characteristic properties of filter curves, as the SVO Filter Profile Service lists them.

Wavelengths are in Angstrom.  ``T`` is the tabulated transmission and ``W`` the
synthetic-photometry weight used everywhere in SED_Tools (:func:`filter_weight`:
``T`` for energy-counting curves, ``T·λ`` for photon-counting ones; Bessell &
Murphy 2012, PASP 124, 140).  Integrals use the trapezoidal rule on the curve's
own samples, and a reference spectrum is interpolated onto them.

=================  ==============================================================
WavelengthMean     ∫ λ T dλ / ∫ T dλ
WavelengthPivot    sqrt(∫ W dλ / ∫ W λ⁻² dλ): converts the band-averaged f_λ and
                   f_ν exactly; for a photon-counting curve sqrt(∫ λT dλ / ∫ T/λ dλ)
WavelengthPeak     λ of the maximum of T
WavelengthMin/Max  first/last tabulated λ where T is at least 1 % of its maximum
WavelengthCen      midpoint of the outermost half-maximum points
FWHM               distance between the outermost half-maximum points
WidthEff           ∫ T dλ / max(T)
WavelengthEff      ∫ λ T F dλ / ∫ T F dλ, F the Vega spectrum
WavelengthPhot     ∫ λ² T F dλ / ∫ λ T F dλ
ZeroPoint*_flam    band-averaged f_λ of the reference source, erg s⁻¹ cm⁻² Å⁻¹
ZeroPoint*_Jy      band-averaged f_ν of the reference source, Jy
ABminusVega        m_AB − m_Vega for any source: −2.5 log₁₀(ZeroPointVega_Jy / 3631 Jy)
STminusVega        m_ST − m_Vega: −2.5 log₁₀(ZeroPointVega_flam / 3.631e-9)
=================  ==============================================================

The references are Vega (a supplied spectrum), AB (3631 Jy) and ST
(3.631e-9 erg s⁻¹ cm⁻² Å⁻¹).  As in SED_Tools photometry, the Vega spectrum is
taken as zero outside its range.  Vega quantities are ``None`` when no spectrum is
given or it does not cover WavelengthMin to WavelengthMax.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .filter_metadata import normalize_detector_type, read_filter_header
from .flux_cube_tool import (
    SPEED_OF_LIGHT_ANG_PER_S,
    filter_weight,
    load_filter_curve,
    load_spectrum,
)

AB_JY = 3631.0
ST_FLAM = 3.631e-9  # erg/s/cm²/Å
JY = 1e-23  # erg/s/cm²/Hz

UNITS = {
    "WavelengthMean": "Angstrom", "WavelengthPivot": "Angstrom", "WavelengthPeak": "Angstrom",
    "WavelengthMin": "Angstrom", "WavelengthMax": "Angstrom", "WavelengthCen": "Angstrom",
    "FWHM": "Angstrom", "WidthEff": "Angstrom", "WavelengthEff": "Angstrom",
    "WavelengthPhot": "Angstrom",
    "ZeroPointVega_flam": "erg/s/cm2/Angstrom", "ZeroPointVega_Jy": "Jy",
    "ZeroPointAB_flam": "erg/s/cm2/Angstrom", "ZeroPointAB_Jy": "Jy",
    "ZeroPointST_flam": "erg/s/cm2/Angstrom", "ZeroPointST_Jy": "Jy",
    "ABminusVega": "mag", "STminusVega": "mag",
}

PathLike = Union[str, Path]
Spectrum = Tuple[np.ndarray, np.ndarray]


def _half_maximum_points(wavelength: np.ndarray, transmission: np.ndarray) -> Tuple[float, float]:
    """Outermost wavelengths where T crosses half its maximum, linearly interpolated."""
    half = transmission.max() / 2.0
    above = np.flatnonzero(transmission >= half)
    i, j = above[0], above[-1]

    def crossing(a: int, b: int) -> float:
        t0, t1 = transmission[a], transmission[b]
        if t1 == t0:
            return float(wavelength[b])
        return float(wavelength[a] + (half - t0) * (wavelength[b] - wavelength[a]) / (t1 - t0))

    blue = crossing(i - 1, i) if i > 0 else float(wavelength[0])
    red = crossing(j, j + 1) if j < transmission.size - 1 else float(wavelength[-1])
    return blue, red


def filter_properties(
    wavelength,
    transmission,
    detector_type: str = "energy",
    vega: Optional[Spectrum] = None,
) -> Dict[str, Optional[float]]:
    """Return the properties of one curve (Angstrom, transmission) as a dict.

    ``vega`` is ``(wavelength in Å, f_λ)``; without it the Vega quantities are ``None``.
    """
    lam = np.asarray(wavelength, dtype=float)
    t = np.asarray(transmission, dtype=float)
    order = np.argsort(lam)
    lam, t = lam[order], t[order]
    if lam.size < 2 or not np.all(np.isfinite(lam)) or not np.all(np.isfinite(t)) or np.any(lam <= 0):
        raise ValueError("a filter curve needs at least two finite, positive wavelengths")
    if t.max() <= 0:
        raise ValueError("the filter curve has no positive transmission")
    if normalize_detector_type(detector_type) is None:
        raise ValueError(f"Unknown detector type: {detector_type!r}")

    w = filter_weight(lam, t, detector_type)
    area_t, area_w = np.trapezoid(t, lam), np.trapezoid(w, lam)
    pivot = math.sqrt(area_w / np.trapezoid(w / lam**2, lam))
    significant = np.flatnonzero(t >= 0.01 * t.max())
    blue, red = _half_maximum_points(lam, t)
    nu_factor = pivot**2 / SPEED_OF_LIGHT_ANG_PER_S  # band-averaged f_ν = f_λ · λp² / c

    props: Dict[str, Optional[float]] = {
        "WavelengthMean": float(np.trapezoid(lam * t, lam) / area_t),
        "WavelengthPivot": pivot,
        "WavelengthPeak": float(lam[np.argmax(t)]),
        "WavelengthMin": float(lam[significant[0]]),
        "WavelengthMax": float(lam[significant[-1]]),
        "WavelengthCen": (blue + red) / 2.0,
        "FWHM": red - blue,
        "WidthEff": float(area_t / t.max()),
        "WavelengthEff": None,
        "WavelengthPhot": None,
        "ZeroPointVega_flam": None,
        "ZeroPointVega_Jy": None,
        "ZeroPointAB_flam": AB_JY * JY * SPEED_OF_LIGHT_ANG_PER_S / pivot**2,
        "ZeroPointAB_Jy": AB_JY,
        "ZeroPointST_flam": ST_FLAM,
        "ZeroPointST_Jy": ST_FLAM * nu_factor / JY,
        "ABminusVega": None,
        "STminusVega": None,
    }
    if vega is not None:
        vl, vf = (np.asarray(a, dtype=float) for a in vega)
        # Like SED_Tools photometry, the spectrum is zero outside its range; it must
        # still cover every wavelength where T is at least 1 % of its maximum.
        if vl[0] <= props["WavelengthMin"] and vl[-1] >= props["WavelengthMax"]:
            f = np.interp(lam, vl, vf, left=0.0, right=0.0)
            flam = float(np.trapezoid(f * w, lam) / area_w)
            props.update({
                "WavelengthEff": float(np.trapezoid(lam * t * f, lam) / np.trapezoid(t * f, lam)),
                "WavelengthPhot": float(np.trapezoid(lam**2 * t * f, lam) / np.trapezoid(lam * t * f, lam)),
                "ZeroPointVega_flam": flam,
                "ZeroPointVega_Jy": flam * nu_factor / JY,
            })
            if flam > 0:
                props["ABminusVega"] = -2.5 * math.log10(flam * nu_factor / JY / AB_JY)
                props["STminusVega"] = -2.5 * math.log10(flam / ST_FLAM)
    return props


def load_vega(path: PathLike, wavelength_unit: str = "angstrom") -> Spectrum:
    """Read a Vega spectrum (wavelength, f_λ in erg/s/cm²/Å), e.g. MESA's vega_flam.csv (Å)."""
    spectrum = load_spectrum(str(path), wavelength_unit=wavelength_unit)
    return spectrum.wavelength, spectrum.flux


def curve_properties(path: PathLike, vega: Optional[Spectrum] = None) -> Dict[str, Optional[float]]:
    """Properties of a filter file, using its WavelengthUnit and DetectorType metadata."""
    curve = load_filter_curve(str(path))
    return filter_properties(curve.wavelength, curve.transmission, curve.detector_type, vega)


def instrument_properties(instrument_dir: PathLike, vega: Optional[Spectrum] = None) -> Dict[str, dict]:
    """Properties of every published ``<band>.dat`` curve of an instrument, keyed by band.

    Each entry also records the ``DetectorType`` the values were computed with.
    """
    out: Dict[str, dict] = {}
    for path in sorted(Path(instrument_dir).glob("*.dat")):
        curve = load_filter_curve(str(path))
        props: Dict[str, object] = dict(filter_properties(curve.wavelength, curve.transmission,
                                                          curve.detector_type, vega))
        props["DetectorType"] = curve.detector_type
        props["DetectorTypeSource"] = read_filter_header(path).get("DetectorTypeSource", "assumed")
        out[path.stem] = props
    return out
