"""Photometry on spectra and filter curves: units, curve and spectrum loaders, band averages, magnitudes
and bolometric flux. The flux-cube tool, the SED API, the filter modules and the builders use these.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .filters.curve_metadata import ENERGY, PHOTON, normalize_detector_type, read_filter_header

# These were flux_cube_tool's; their warnings keep its logger name.
logger = logging.getLogger("sed_tools.flux_cube_tool")


FILTER_EXTENSIONS = {".dat", ".txt", ".csv"}

PC_TO_M = 3.085677581e16
RSUN_TO_M = 6.957e8
SPEED_OF_LIGHT_ANG_PER_S = 2.99792458e18  # speed of light in Angstrom/s
AB_ZERO_FLUX = 3.631e-20  # 3631 Jy expressed in erg s^-1 cm^-2 Hz^-1
UNIT_TO_ANGSTROM: Dict[str, float] = {name: factor for factor, names in (
    (1.0, ("angstrom", "angstroms", "\u00e5", "ang", "a", "aa")),
    (10.0, ("nanometer", "nanometre", "nm")),
    (1.0e4, ("micrometer", "micrometre", "micron", "microns", "\u00b5m", "\u03bcm", "um")),
    (1.0e7, ("millimeter", "millimetre", "mm")),
    (1.0e10, ("meter", "metre", "m")),
) for name in names}
VEGA_ZP_KEYS = [
    "zp_vega", "vega_zp", "vega_flux", "zero_point_flux_vega", "zero_point_flux", "zeropointflux",
    "zeropoint_flux", "zeropointflux_ergcm2sa", "zeropointflux_ergcm2sang", "zeropointfluxergcm2sa",
]


@dataclass
class FilterCurve:
    name: str
    wavelength: np.ndarray
    transmission: np.ndarray
    metadata: Dict[str, object]
    wavelength_unit: str = "angstrom"
    detector_type: str = ENERGY

    @property
    def weight(self) -> np.ndarray:
        """Integration weight: ``T`` (energy counting) or ``T*lambda`` (photon counting)."""
        return filter_weight(self.wavelength, self.transmission, self.detector_type)


@dataclass
class Spectrum:
    wavelength: np.ndarray
    flux: np.ndarray
    metadata: Dict[str, object]
    wavelength_unit: str = "angstrom"


def normalize_metadata_key(key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower())
    return normalized.strip("_")


def parse_metadata_value(value: str) -> object:
    value = value.strip()
    if not value:
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def infer_unit_from_values(values: np.ndarray) -> str:
    """Guess the wavelength unit of a filter curve or spectrum from its values.

    Angstrom is assumed unless that reading is physically implausible: a curve
    or spectrum whose reddest point lies below 1000 A would sit entirely in the
    extreme UV, so such values are read as nm, then um, then m.

    The guess is inherently ambiguous where units overlap (e.g. a K-band curve
    tabulated in nm, 1950-2400, reads as a UV curve in A), so files should
    declare ``WavelengthUnit`` (on a filter curve's column row, see
    :mod:`sed_tools.filter_metadata`) and callers may pass an explicit unit.
    """
    if values.size == 0:
        return "angstrom"
    max_value = float(np.max(values))
    if max_value >= 1000:
        return "angstrom"
    if max_value >= 100:
        return "nanometer"
    if max_value >= 0.01:
        return "micrometer"
    return "meter"


def standardize_wavelength_array(
    values: Sequence[float], unit_hint: str | None = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("No wavelength data found in file")
    order = np.argsort(arr)
    arr = arr[order]
    unit_key = (unit_hint or "").strip().lower() or infer_unit_from_values(arr)
    if unit_key not in UNIT_TO_ANGSTROM:
        logger.warning("Unrecognised wavelength unit %r; inferring from values instead", unit_hint)
        unit_key = infer_unit_from_values(arr)
    return arr * UNIT_TO_ANGSTROM[unit_key], order, unit_key or "angstrom"


def load_two_column_file(path: str) -> Tuple[List[float], List[float], Dict[str, object]]:
    """The first two columns of a text curve, and its ``# key: value`` metadata.

    Non-numeric lines before the first data row (column names, titles) are
    skipped; one after the data has started raises ValueError.
    """
    x_values: List[float] = []
    y_values: List[float] = []
    metadata: Dict[str, object] = {}

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                meta = line[1:].strip()
                for sep in (":", "="):
                    if sep in meta:
                        key, value = meta.split(sep, 1)
                        metadata[normalize_metadata_key(key)] = parse_metadata_value(value)
                        break
                else:
                    metadata.setdefault("comments", []).append(meta)
                continue

            parts = line.replace(",", " ").split()
            try:
                x = float(parts[0])
                y = float(parts[1])
            except (IndexError, ValueError):
                if x_values:
                    raise ValueError(f"{path}, line {line_no}: expected two numbers, found {line!r}") from None
                continue
            x_values.append(x)
            y_values.append(y)

    return x_values, y_values, metadata


def get_metadata_float(metadata: Dict[str, object], keys: Sequence[str]) -> float | None:
    """The value of the first of *keys* present in *metadata*, or None if none is.

    Raises ValueError if that value is not a number.
    """
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata {key} = {value!r} is not a number") from exc
    return None

def integrate_flux(wavelength: np.ndarray, flux: np.ndarray) -> float:
    """Compute bolometric flux via trapezoidal integration."""

    return float(np.trapezoid(flux, wavelength))


def bolometric_magnitude(
    wavelength: np.ndarray,
    flux: np.ndarray,
    reference_flux: float = 1.0,
    reference_magnitude: float = 0.0,
) -> Tuple[float, float]:
    """Return bolometric flux and magnitude.

    A zero or negative bolometric flux gives a magnitude of ``inf`` or NaN, and a
    zero or negative reference flux ``-inf`` or NaN, as computed; no error.
    """

    fbol = integrate_flux(wavelength, flux)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = float(reference_magnitude - 2.5 * np.log10(np.divide(fbol, reference_flux)))
    return fbol, mag


def _standardize_file_wavelengths(
    path: str,
    raw_wavelengths: Sequence[float],
    metadata: Dict[str, object],
    wavelength_unit: str | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a file's wavelength column to Angstrom, recording how the unit was found."""
    hint = wavelength_unit or metadata.get("wavelength_unit") or metadata.get("wavelengthunit")
    hint = str(hint) if hint else None
    wavelength, order, unit_key = standardize_wavelength_array(raw_wavelengths, hint)
    if wavelength_unit:
        metadata["wavelength_unit_source"] = "argument"
    elif hint:
        metadata["wavelength_unit_source"] = "header"
    else:
        metadata["wavelength_unit_source"] = "inferred"
        logger.warning(
            "%s declares no wavelength unit; assuming %s from its values. "
            "Declare WavelengthUnit (filter curves: on the column row, via "
            "sed_tools.filter_metadata.write_filter_header) or pass wavelength_unit= to be explicit.", path, unit_key)
    metadata["wavelength_unit_inferred"] = unit_key or "angstrom"
    metadata["wavelength_unit_resolved"] = "angstrom"
    return wavelength, order


def load_filter_curve(
    path: str, name: str | None = None, wavelength_unit: str | None = None
) -> FilterCurve:
    """Load a filter throughput curve along with its metadata.

    ``wavelength_unit`` overrides any ``WavelengthUnit`` header; without either
    the unit is inferred from the values (see :func:`infer_unit_from_values`).
    """

    raw_wavelengths, raw_transmission, metadata = load_two_column_file(path)
    # SED_Tools metadata sits on the column row, which the sample parser skips.
    metadata.update({normalize_metadata_key(key): parse_metadata_value(value)
                     for key, value in read_filter_header(path).items()})
    wavelength, order = _standardize_file_wavelengths(path, raw_wavelengths, metadata, wavelength_unit)
    transmission = np.asarray(raw_transmission, dtype=float)[order]
    detector_type = normalize_detector_type(metadata.get("detectortype"))
    if detector_type is None:
        # Untagged file: keep the historical energy-counting behaviour.
        detector_type = ENERGY
        metadata.setdefault("detectortypesource", "assumed")
    return FilterCurve(name or Path(path).stem, wavelength, transmission, metadata,
                       wavelength_unit="angstrom", detector_type=detector_type)


def filter_weight(
    wavelength: np.ndarray, transmission: np.ndarray, detector_type: str = ENERGY
) -> np.ndarray:
    """Return the synthetic-photometry weight for a transmission curve.

    Photon-counting curves are weighted by ``T*lambda`` so the band average is
    over photons rather than energy (Bessell & Murphy 2012, PASP 124, 140,
    eq. A9).  Energy-counting curves use ``T`` directly.
    """
    resolved = normalize_detector_type(detector_type)
    if resolved is None:
        raise ValueError(f"Unknown detector type: {detector_type!r}")
    transmission = np.asarray(transmission, dtype=float)
    if resolved == PHOTON:
        return transmission * np.asarray(wavelength, dtype=float)
    return transmission


def band_average_flux_lambda_from_arrays(
    wavelength: np.ndarray,
    flux: np.ndarray,
    transmission: np.ndarray,
    detector_type: str = ENERGY,
) -> float:
    weight = filter_weight(wavelength, transmission, detector_type)
    numerator = np.trapezoid(flux * weight, wavelength)
    denom = np.trapezoid(weight, wavelength)
    if denom == 0:
        raise ValueError("Filter transmission integrates to zero")
    return float(numerator / denom)


def _warn_outside_sed(sed_wavelength: np.ndarray, curves: Iterable[FilterCurve]) -> None:
    """One warning naming the filters with transmission outside the SED, where band averages count zero flux."""
    lo, hi = np.min(sed_wavelength), np.max(sed_wavelength)
    outside = []
    for curve in curves:
        total = np.trapezoid(curve.transmission, curve.wavelength)
        if not total > 0:
            continue  # the band average itself raises for a curve that integrates to zero
        beyond = (curve.wavelength < lo) | (curve.wavelength > hi)
        fraction = np.trapezoid(np.where(beyond, curve.transmission, 0.0), curve.wavelength) / total
        if fraction > 0:
            outside.append(f"{curve.name} ({100 * fraction:.3g}%)")
    if outside:
        logger.warning(
            "Filter transmission outside the SED's wavelength range (%.6g-%.6g Å), where the band "
            "average counts the flux as zero, so these bands are biased low: %s.",
            lo, hi, ", ".join(outside),
        )


def band_average_flux_lambda(
    sed_wavelength: np.ndarray, sed_flux: np.ndarray, curve: FilterCurve
) -> float:
    interp_flux = np.interp(curve.wavelength, sed_wavelength, sed_flux, left=0.0, right=0.0)
    return band_average_flux_lambda_from_arrays(curve.wavelength, interp_flux, curve.transmission, curve.detector_type)


def band_average_flux_nu(
    sed_wavelength: np.ndarray, sed_flux: np.ndarray, curve: FilterCurve
) -> float:
    interp_flux = np.interp(curve.wavelength, sed_wavelength, sed_flux, left=0.0, right=0.0)
    weight = curve.weight
    numerator = np.trapezoid(interp_flux * weight, curve.wavelength)
    denom = np.trapezoid(weight * SPEED_OF_LIGHT_ANG_PER_S / (curve.wavelength**2), curve.wavelength)
    if denom == 0:
        raise ValueError("Filter transmission integrates to zero")
    return float(numerator / denom)

def load_spectrum(path: str, wavelength_unit: str | None = None) -> Spectrum:
    wavelengths, fluxes, metadata = load_two_column_file(path)
    wavelength, order = _standardize_file_wavelengths(path, wavelengths, metadata, wavelength_unit)
    flux = np.asarray(fluxes, dtype=float)[order]
    return Spectrum(wavelength, flux, metadata, wavelength_unit="angstrom")

def _band_magnitude(wl, flux, curve, photometric_system, vega_spectrum):
    """Return (flux label, band-averaged flux, magnitude) for one filter.

    In both systems the magnitude of a zero band flux is ``inf`` and of a
    negative one NaN; a zero or negative Vega zero point gives ``-inf`` or NaN.
    *vega_spectrum* is called for the Vega reference only when the filter has
    no Vega zero point in its metadata.
    """
    if photometric_system == "AB":
        flux_density = band_average_flux_nu(wl, flux, curve)
        # A zero or negative band flux gives inf or NaN, as computed; no error.
        with np.errstate(divide="ignore", invalid="ignore"):
            return "f_nu", flux_density, float(-2.5 * np.log10(flux_density / AB_ZERO_FLUX))

    flux_density = band_average_flux_lambda(wl, flux, curve)
    vega_zero = get_metadata_float(curve.metadata, VEGA_ZP_KEYS)
    if vega_zero is None:
        vega_curve = vega_spectrum()
        vega_zero = band_average_flux_lambda(vega_curve.wavelength, vega_curve.flux, curve)
    # A zero or negative band flux or zero point gives inf, -inf or NaN, as computed; no error.
    with np.errstate(divide="ignore", invalid="ignore"):
        return "f_lambda", flux_density, float(-2.5 * np.log10(np.divide(flux_density, vega_zero)))
