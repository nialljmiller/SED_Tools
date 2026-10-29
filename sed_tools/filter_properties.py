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
Fsun               ∫ W S dλ / ∫ W dλ, S the Sun's spectrum at 1 AU: the Sun's
                   band-averaged f_λ, erg s⁻¹ cm⁻² Å⁻¹ (SVO's ``Fsun``)
AfAv               A(λ)/A(V) at λ = WavelengthPivot, from the Fitzpatrick (1999)
                   extinction law with R_V = 3.1 (SVO's ``Af/AV``, which uses a
                   different law)
=================  ==============================================================

The references are Vega (a supplied spectrum), AB (3631 Jy) and ST
(3.631e-9 erg s⁻¹ cm⁻² Å⁻¹).  As in SED_Tools photometry, a supplied spectrum
(Vega or the Sun) is taken as zero outside its range.  Vega quantities are
``None`` when no Vega spectrum is given or it does not cover WavelengthMin to
WavelengthMax; ``Fsun`` follows the same rule with the solar spectrum.  The
NJM mirror uses CALSPEC's ``sun_reference_stis_002`` below 9600 Å and
``sun_mod_001`` above (Bohlin, Gordon & Tremblay 2014, PASP 126, 711), the
composite SVO describes; switching at 9600 Å reproduces SVO's values.

``AfAv`` is evaluated with the ``F99`` class of the optional package
``dust_extinction`` (Gordon 2024, JOSS 9(100), 7023, doi:10.21105/joss.07023;
install with ``pip install "sed-tools[extinction]"``); the law is Fitzpatrick
1999, PASP 111, 63 (1999PASP..111...63F, doi:10.1086/316293).  A(V) is the
law's own normalisation, the extinction averaged over the V band, so
A(λ)/A(V) = 1 at 5414 Å and 0.978 at 5500 Å (dust_extinction's "Normalization
Details" page).  ``AfAv`` is ``None`` where the pivot lies outside the range
the implementation defines for the law (``F99.x_range``: 0.3–10 µm⁻¹, i.e.
1000 Å to 3.33 µm), which is never extrapolated, and ``None``, with a warning,
when ``dust_extinction`` is not installed.  dust_extinction 1.6 changed the
law's optical–infrared spline (by up to 1 % near 3.3 µm), so 1.6 or later is
required.
"""

from __future__ import annotations

import math
import warnings
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .filter_metadata import read_filter_header
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
    "Fsun": "erg/s/cm2/Angstrom", "AfAv": "dimensionless",
}

EXTINCTION_RV = 3.1
EXTINCTION_LAW = {
    "quantity": "AfAv = A(lambda)/A(V) at the pivot wavelength",
    "law": "Fitzpatrick (1999) Milky Way R(V)-dependent extinction curve",
    "R_V": EXTINCTION_RV,
    "implementation": "dust_extinction.parameter_averages.F99 (dust_extinction >= 1.6)",
    "normalisation": ("A(V) is the law's own normalisation, the extinction averaged over the V band: "
                      "A(lambda)/A(V) = 1 at 5414 Angstrom and 0.978 at 5500 Angstrom "
                      "(https://dust-extinction.readthedocs.io/en/latest/dust_extinction/normalization_details.html)"),
    "out_of_range": "null where the pivot lies outside the implementation's F99.x_range; never extrapolated",
    "references": [
        "Fitzpatrick 1999, PASP 111, 63 (1999PASP..111...63F, doi:10.1086/316293): the law",
        "Gordon 2024, JOSS 9(100), 7023 (doi:10.21105/joss.07023): dust_extinction",
    ],
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


_f99_model = None  # the F99 model once built; False when dust_extinction cannot be imported


def _extinction_model():
    """dust_extinction's F99 model at R_V = 3.1, or None (with a warning) when it is not installed."""
    global _f99_model
    if _f99_model is None:
        try:
            from dust_extinction.parameter_averages import F99
        except ImportError:
            warnings.warn(
                "AfAv needs the optional package dust_extinction (pip install \"sed-tools[extinction]\"); "
                "AfAv is None without it",
                stacklevel=3,
            )
            _f99_model = False
        else:
            _f99_model = F99(Rv=EXTINCTION_RV)
    return _f99_model or None


def extinction_ratio(wavelength: float) -> Optional[float]:
    """A(λ)/A(V) of Fitzpatrick (1999) with R_V = 3.1 at ``wavelength`` (Å), from dust_extinction.

    ``None`` outside the law's valid range (``F99.x_range``, in µm⁻¹) or without dust_extinction.
    """
    model = _extinction_model()
    if model is None:
        return None
    lo, hi = model.x_range
    if not lo <= 1e4 / wavelength <= hi:
        return None
    import astropy.units as u

    return float(np.squeeze(model(wavelength * u.AA)))


def extinction_law() -> dict:
    """:data:`EXTINCTION_LAW` with the installed dust_extinction version and the law's range in Å.

    ``dust_extinction_version`` is ``None`` when the package is not installed (``AfAv`` is then ``None``).
    """
    law = dict(EXTINCTION_LAW)
    try:
        law["dust_extinction_version"] = metadata.version("dust_extinction")
    except metadata.PackageNotFoundError:
        law["dust_extinction_version"] = None
    model = _extinction_model()
    if model is not None:
        lo, hi = model.x_range
        law["valid_range_per_micron"] = [float(lo), float(hi)]
        law["valid_range_angstrom"] = [1e4 / hi, 1e4 / lo]
    return law


def _on_curve(spectrum: Optional[Spectrum], lam: np.ndarray, lo: float, hi: float) -> Optional[np.ndarray]:
    """A reference spectrum interpolated onto the curve's samples, or None if it misses lo..hi.

    Like SED_Tools photometry, the spectrum is zero outside its range; it must still
    cover every wavelength where T is at least 1 % of its maximum (lo..hi).
    """
    if spectrum is None:
        return None
    sl, sf = (np.asarray(a, dtype=float) for a in spectrum)
    if sl[0] > lo or sl[-1] < hi:
        return None
    return np.interp(lam, sl, sf, left=0.0, right=0.0)


def filter_properties(
    wavelength,
    transmission,
    detector_type: str = "energy",
    vega: Optional[Spectrum] = None,
    sun: Optional[Spectrum] = None,
) -> Dict[str, Optional[float]]:
    """Return the properties of one curve (Angstrom, transmission) as a dict.

    ``vega`` and ``sun`` are ``(wavelength in Å, f_λ in erg/s/cm²/Å)``, the Sun's at 1 AU;
    without them the Vega quantities and ``Fsun`` are ``None``.
    """
    lam = np.asarray(wavelength, dtype=float)
    t = np.asarray(transmission, dtype=float)
    order = np.argsort(lam)
    lam, t = lam[order], t[order]
    if lam.size < 2 or not np.all(np.isfinite(lam)) or not np.all(np.isfinite(t)) or np.any(lam <= 0):
        raise ValueError("a filter curve needs at least two finite, positive wavelengths")
    if t.max() <= 0:
        raise ValueError("the filter curve has no positive transmission")

    w = filter_weight(lam, t, detector_type)  # ValueError for an unknown detector type
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
        "Fsun": None,
        "AfAv": extinction_ratio(pivot),
    }
    lo, hi = props["WavelengthMin"], props["WavelengthMax"]
    f = _on_curve(vega, lam, lo, hi)
    if f is not None:
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
    solar = _on_curve(sun, lam, lo, hi)
    if solar is not None:
        props["Fsun"] = float(np.trapezoid(solar * w, lam) / area_w)
    return props


def load_vega(path: PathLike, wavelength_unit: str = "angstrom") -> Spectrum:
    """Read a Vega spectrum (wavelength, f_λ in erg/s/cm²/Å), e.g. MESA's vega_flam.csv (Å)."""
    spectrum = load_spectrum(str(path), wavelength_unit=wavelength_unit)
    return spectrum.wavelength, spectrum.flux


def load_sun(path: PathLike, wavelength_unit: str = "angstrom") -> Spectrum:
    """Read a solar spectrum at 1 AU (wavelength, f_λ in erg/s/cm²/Å), e.g. NJM's sun/sun_flam.csv (Å)."""
    return load_vega(path, wavelength_unit)


def curve_properties(path: PathLike, vega: Optional[Spectrum] = None,
                     sun: Optional[Spectrum] = None) -> Dict[str, Optional[float]]:
    """Properties of a filter file, using its WavelengthUnit and DetectorType metadata."""
    curve = load_filter_curve(str(path))
    return filter_properties(curve.wavelength, curve.transmission, curve.detector_type, vega, sun)


def instrument_properties(instrument_dir: PathLike, vega: Optional[Spectrum] = None,
                          sun: Optional[Spectrum] = None) -> Dict[str, dict]:
    """Properties of every published ``<band>.dat`` curve of an instrument, keyed by band.

    Each entry also records the ``DetectorType`` the values were computed with.
    """
    out: Dict[str, dict] = {}
    for path in sorted(Path(instrument_dir).glob("*.dat")):
        curve = load_filter_curve(str(path))
        props: Dict[str, object] = dict(filter_properties(curve.wavelength, curve.transmission,
                                                          curve.detector_type, vega, sun))
        props["DetectorType"] = curve.detector_type
        props["DetectorTypeSource"] = read_filter_header(path).get("DetectorTypeSource", "assumed")
        out[path.stem] = props
    return out
