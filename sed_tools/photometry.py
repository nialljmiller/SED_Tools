"""Photometry on spectra and filter curves: units, curve and spectrum loaders, band averages, magnitudes
and bolometric flux. The flux-cube tool, the SED API, the filter modules and the builders use these.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .filters.curve_metadata import ENERGY, PHOTON, normalize_detector_type, read_filter_header

logger = logging.getLogger(__name__)


FILTER_EXTENSIONS = {".dat", ".txt", ".csv"}

PC_TO_M = 3.085677581e16
RSUN_TO_M = 6.957e8
SPEED_OF_LIGHT_ANG_PER_S = 2.99792458e18  # speed of light in Angstrom/s
AB_ZERO_FLUX = 3.630e-20  # 3631 Jy expressed in erg s^-1 cm^-2 Hz^-1
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
    path: str | None = None  # the file it was read from


@dataclass
class Spectrum:
    wavelength: np.ndarray
    flux: np.ndarray
    metadata: Dict[str, object]
    wavelength_unit: str = "angstrom"


def normalize_metadata_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


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
    :mod:`sed_tools.filters.curve_metadata`) and callers may pass an explicit unit.
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
        raise ValueError(f"Unknown wavelength unit {unit_hint!r}: declare WavelengthUnit as angstrom, nm, um, mm or m")
    return arr * UNIT_TO_ANGSTROM[unit_key], order, unit_key


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
            "sed_tools.filters.curve_metadata.write_filter_header) or pass wavelength_unit= to be explicit.", path, unit_key)
    metadata["wavelength_unit_inferred"] = unit_key
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
                       wavelength_unit="angstrom", detector_type=detector_type, path=str(path))


#: :func:`band_grid` splits every step longer than lambda / BAND_RESOLVING_POWER, so that :func:`band_integral` is
#: exact to about 1e-9 for the negative powers of lambda too (the pivot's lambda^-2), not only for polynomials.
BAND_RESOLVING_POWER = 100.0


def band_grid(
    wavelength: Sequence[float], transmission: Sequence[float], points: Sequence[float] = ()
) -> Tuple[np.ndarray, np.ndarray]:
    """The wavelengths every band integral is taken on, and the curve's transmission there: (wavelength, transmission).

    They are the curve's samples, the ``points`` (a spectrum's wavelengths) within the curve's range, and evenly spaced
    points that split each step longer than lambda / :data:`BAND_RESOLVING_POWER`.  The transmission is linear between
    the curve's samples, on every step however long.  The curve's wavelengths increase; one may repeat (a vertical
    edge), and keeps its order.
    """
    lam = np.asarray(wavelength, dtype=float)
    t = np.asarray(transmission, dtype=float)
    points = np.asarray(points, dtype=float)
    added = points[(points > lam[0]) & (points < lam[-1])]
    added = added[~np.isin(added, lam)]
    if added.size:
        order = np.argsort(np.concatenate([lam, added]), kind="stable")
        t = np.concatenate([t, np.interp(added, lam, t)])[order]
        lam = np.concatenate([lam, added])[order]
    step = np.diff(lam)
    n = np.maximum(np.ceil(step * BAND_RESOLVING_POWER / lam[:-1]).astype(int) - 1, 0)  # points added in each step
    if not n.any():
        return lam, t
    i = np.repeat(np.arange(step.size), n)
    frac = (np.arange(i.size) - np.repeat(np.cumsum(n) - n, n) + 1) / (n[i] + 1)
    return (np.insert(lam, i + 1, lam[i] + frac * step[i]),
            np.insert(t, i + 1, t[i] + frac * (t[i + 1] - t[i])))


def _step_integrals(wavelength, transmission, flux=None, power: int = 0) -> np.ndarray:
    """∫ T F λ^power dλ over each step of a band grid, T and F linear across the step: Simpson's rule with T and F
    at the step's middle taken from that line, which is exact for power 0 to 2 (a cubic at most)."""
    lam = np.asarray(wavelength, dtype=float)
    product = np.asarray(transmission, dtype=float)
    middle = (product[1:] + product[:-1]) / 2
    if flux is not None:
        flux = np.asarray(flux, dtype=float)
        product, middle = product * flux, middle * (flux[1:] + flux[:-1]) / 2
    mid = (lam[1:] + lam[:-1]) / 2
    ends = product * lam**power
    return np.diff(lam) / 6 * (ends[:-1] + 4 * middle * mid**power + ends[1:])


def band_integral(wavelength, transmission, flux=None, power: int = 0) -> float:
    """∫ T F λ^power dλ on a :func:`band_grid` (``flux`` F omitted: 1), the one rule for every band quantity: band
    averages, zero points, pivot and other wavelengths, and shares of the band.

    T and F are taken as linear between the grid's wavelengths, on which the integral of their product is exact
    (see :func:`_step_integrals`). T is linear between the curve's samples, so that is exact for T; F is log-linear
    between a spectrum's positive samples (see :func:`on_band_grid`), and the grid's steps of at most
    lambda / :data:`BAND_RESOLVING_POWER` follow it closely.
    """
    return float(_step_integrals(wavelength, transmission, flux, power).sum())


def _weight_power(detector_type: str) -> int:
    """The power of lambda in a band's weight T λ^power: 0 (energy counting) or 1 (photon counting, which averages
    over photons rather than energy; Bessell & Murphy 2012, PASP 124, 140, eq. A9)."""
    resolved = normalize_detector_type(detector_type)
    if resolved is None:
        raise ValueError(f"Unknown detector type: {detector_type!r}")
    return 1 if resolved == PHOTON else 0


def band_average_flux_lambda_from_arrays(
    wavelength: np.ndarray,
    flux: np.ndarray,
    transmission: np.ndarray,
    detector_type: str = ENERGY,
) -> float:
    """∫ F W dλ / ∫ W dλ by :func:`band_integral` on the given wavelengths (a :func:`band_grid`), W the weight."""
    k = _weight_power(detector_type)
    denom = band_integral(wavelength, transmission, None, k)
    if denom == 0:
        raise ValueError("Filter transmission integrates to zero")
    return band_integral(wavelength, transmission, flux, k) / denom


def pivot_wavelength(wavelength: Sequence[float], transmission: Sequence[float], detector_type: str = ENERGY) -> float:
    """sqrt(∫ W dλ / ∫ W λ⁻² dλ) by :func:`band_integral`, W the weight: the band-averaged f_ν is f_λ λp² / c."""
    k = _weight_power(detector_type)
    grid, t = band_grid(wavelength, transmission)
    denom = band_integral(grid, t, None, k - 2)
    if not denom > 0:
        raise ValueError("Filter transmission integrates to zero")
    return math.sqrt(band_integral(grid, t, None, k) / denom)


def _band_steps(wavelength, transmission, detector_type):
    """Each step of the band grid: its middle wavelength and its part of ∫ W dλ (negative T counts as 0)."""
    grid, t = band_grid(wavelength, np.clip(np.asarray(transmission, dtype=float), 0.0, None))
    return (grid[1:] + grid[:-1]) / 2, _step_integrals(grid, t, None, _weight_power(detector_type))


def band_share(wavelength: Sequence[float], transmission: Sequence[float], where, detector_type: str = ENERGY) -> float:
    """The share of a band's weight ∫ W dλ (by :func:`band_integral`) in the steps of the band grid whose middle
    wavelength ``where`` accepts (negative transmission counts as zero); NaN for a band whose weight integrates to
    zero."""
    middle, part = _band_steps(wavelength, transmission, detector_type)
    total = part.sum()
    if not total > 0:
        return math.nan
    return float(part[where(middle)].sum() / total)


# ---- How a grid's wavelengths sample a band ----

#: A stretch between two consecutive grid wavelengths is coarse where lambda/delta-lambda is below this: across it,
#: linear and power-law interpolation of a Rayleigh-Jeans continuum (F_lambda ~ lambda^-4) differ by up to 1%, so
#: the grid no longer fixes the flux there.
COARSE_RESOLVING_POWER = 16.0
#: The largest share of a band's weight that may lie beyond a grid or in its coarse stretches: 1% of the band average.
UNSAMPLED_WEIGHT_LIMIT = 0.01


@dataclass(frozen=True)
class BandSampling:
    """How a grid's wavelengths sample one band: see :func:`band_sampling`."""

    inside: int                   # grid wavelengths where the band transmits
    outside: float                # share of the band's weight beyond the grid's first and last wavelength
    coarse: float                 # share of its weight in coarse stretches between grid wavelengths
    stretch: Tuple[float, float]  # the coarse stretch (Å) under the band's heaviest coarse step; (nan, nan) if none

    @property
    def sampled(self) -> bool:
        """Two or more grid wavelengths where the band transmits, and at most UNSAMPLED_WEIGHT_LIMIT of its weight
        beyond the grid or in coarse stretches.

        A band average taken on the grid's own wavelengths (MESA's: the curve interpolated onto the SED) is empty
        with no grid wavelength in the band and is one flux sample with one; elsewhere it is an interpolation across
        the grid's gap, whatever the method.
        """
        return self.inside >= 2 and self.outside + self.coarse <= UNSAMPLED_WEIGHT_LIMIT

    def describe(self) -> str:
        parts = [f"{self.inside} grid wavelength{'' if self.inside == 1 else 's'} where it transmits"]
        if self.outside > 0:
            parts.append(f"{100 * self.outside:.3g}% of its weight beyond the grid")
        if self.coarse > 0:
            parts.append(f"{100 * self.coarse:.3g}% where lambda/delta-lambda < {COARSE_RESOLVING_POWER:g}, "
                         f"as between {self.stretch[0]:.6g} and {self.stretch[1]:.6g} Å")
        return ", ".join(parts)


def band_sampling(grid_wavelength: Sequence[float], curve: FilterCurve) -> BandSampling:
    """How the increasing wavelengths ``grid_wavelength`` (Å; a flux cube's axis or a spectrum's) sample ``curve``.

    The shares are of the band's weight (``T``, or ``T·λ`` for photon counting) on :func:`band_grid`.
    """
    grid = np.asarray(grid_wavelength, dtype=float)
    lam = curve.wavelength
    near = grid[np.searchsorted(grid, lam[0]):np.searchsorted(grid, lam[-1], side="right")]  # 0 elsewhere
    inside = int(np.count_nonzero(np.interp(near, lam, np.clip(curve.transmission, 0.0, None)) > 0))

    def beyond(x):
        return (x < grid[0]) | (x > grid[-1])

    def stretch_of(x):  # each wavelength's stretch grid[k-1]..grid[k]
        k = np.clip(np.searchsorted(grid, x, side="right"), 1, grid.size - 1)
        return grid[k - 1], grid[k]

    def coarse(x):
        lo, hi = stretch_of(x)
        return ~beyond(x) & ((hi - lo) * 2 * COARSE_RESOLVING_POWER > hi + lo)

    outside = band_share(lam, curve.transmission, beyond, curve.detector_type)
    if math.isnan(outside):
        return BandSampling(inside, 0.0, 0.0, (math.nan, math.nan))  # the band average itself raises
    if grid.size < 2:
        return BandSampling(inside, outside, 0.0, (math.nan, math.nan))
    share = band_share(lam, curve.transmission, coarse, curve.detector_type)
    stretch = (math.nan, math.nan)
    if share > 0:
        middle, part = _band_steps(lam, curve.transmission, curve.detector_type)
        j = int(np.argmax(np.where(coarse(middle), part, -1.0)))
        lo, hi = stretch_of(middle[j:j + 1])
        stretch = (float(lo[0]), float(hi[0]))
    return BandSampling(inside, outside, share, stretch)


def _warn_outside_sed(sed_wavelength: np.ndarray, curves: Iterable[FilterCurve]) -> None:
    """One warning naming the filters with weight outside the SED, where band averages count zero flux."""
    lo, hi = np.min(sed_wavelength), np.max(sed_wavelength)
    outside = []
    for curve in curves:
        fraction = band_share(curve.wavelength, curve.transmission, lambda x: (x < lo) | (x > hi), curve.detector_type)
        if fraction > 0:  # NaN (weight integrating to zero): the band average itself raises
            outside.append(f"{curve.name} ({100 * fraction:.3g}%)")
    if outside:
        logger.warning(
            "Filter transmission outside the SED's wavelength range (%.6g-%.6g Å), where the band "
            "average counts the flux as zero, so these bands are biased low: %s.",
            lo, hi, ", ".join(outside),
        )


#: Stands in for log10 of a zero or negative sample, which the log rule of :func:`resample_to_grid` never uses.
_FLUX_FLOOR = 1e-300


def resample_to_grid(wl_src: Sequence[float], fl_src: Sequence[float], wl_tgt: Sequence[float]) -> np.ndarray:
    """A spectrum's flux at the wavelengths ``wl_tgt`` (Å): the one rule for a spectrum between its samples, used by
    every flux cube builder and every band average (:func:`on_band_grid`).

    Between two samples that are both positive, log flux is linear in wavelength, which is closer than linear flux
    on coarsely sampled continua; between two where either is zero or negative, the flux is linear (Nill, 25 Nov
    2026, Q12). A sample keeps its own value: a positive one as ``10**log10`` of it, as cubes have always stored
    it, and a zero or negative one as it is. Outside the spectrum's range the flux is zero: no flux is invented
    beyond a spectrum's coverage. ``wl_src`` increases.
    """
    wl = np.asarray(wl_src, dtype=np.float64)
    fl = np.asarray(fl_src, dtype=np.float64)
    at = np.asarray(wl_tgt, dtype=np.float64)
    i = np.clip(np.searchsorted(wl, at, side="right") - 1, 0, max(wl.size - 2, 0))  # the step each point is in
    j = np.minimum(i + 1, wl.size - 1)
    log_rule = ((fl[i] > 0) & (fl[j] > 0)) | ((at == wl[i]) & (fl[i] > 0)) | ((at == wl[j]) & (fl[j] > 0))
    flux = np.where(log_rule, 10.0 ** np.interp(at, wl, np.log10(np.maximum(fl, _FLUX_FLOOR))), np.interp(at, wl, fl))
    flux[(at < wl[0]) | (at > wl[-1])] = 0.0
    return flux


def on_band_grid(
    sed_wavelength: Sequence[float], sed_flux: Sequence[float], wavelength: Sequence[float],
    transmission: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A spectrum and a filter curve on :func:`band_grid` with the spectrum's wavelengths: (wavelength, transmission,
    flux).

    The spectrum is put onto the grid by :func:`resample_to_grid` (log flux linear between positive samples,
    flux linear next to a zero or negative one), the curve linearly between its own samples, and the spectrum is
    zero beyond its range: an end of the spectrum inside the band is a vertical edge (its wavelength twice, the
    second with flux 0), so no flux is ramped down beyond it.  A coarsely tabulated curve (Johnson U: 13 samples)
    then sees every spectral feature in its band, not the spectrum at its 13 wavelengths.
    """
    sed_wavelength = np.asarray(sed_wavelength, dtype=float)
    grid, t = band_grid(wavelength, transmission, sed_wavelength)
    flux = resample_to_grid(sed_wavelength, sed_flux, grid)
    blue = np.searchsorted(grid, sed_wavelength[0], side="left")  # its ends inside the curve are on the grid
    red = np.searchsorted(grid, sed_wavelength[-1], side="right")
    if 0 < blue < grid.size:
        grid, t, flux = np.insert(grid, blue, grid[blue]), np.insert(t, blue, t[blue]), np.insert(flux, blue, 0.0)
        red += 1
    if 0 < red < grid.size:
        grid, t, flux = np.insert(grid, red, grid[red - 1]), np.insert(t, red, t[red - 1]), np.insert(flux, red, 0.0)
    return grid, t, flux


def band_average_flux_lambda(
    sed_wavelength: np.ndarray, sed_flux: np.ndarray, curve: FilterCurve
) -> float:
    """The band-averaged f_λ of a spectrum: ∫ F W dλ / ∫ W dλ on :func:`on_band_grid`, W the curve's weight."""
    wavelength, transmission, flux = on_band_grid(sed_wavelength, sed_flux, curve.wavelength, curve.transmission)
    return band_average_flux_lambda_from_arrays(wavelength, flux, transmission, curve.detector_type)


def band_average_flux_nu(
    sed_wavelength: np.ndarray, sed_flux: np.ndarray, curve: FilterCurve
) -> float:
    """The band-averaged f_ν of a spectrum: ∫ F W dλ / ∫ W c/λ² dλ, that is the band-averaged f_λ times λp² / c with
    λp the :func:`pivot_wavelength`."""
    pivot = pivot_wavelength(curve.wavelength, curve.transmission, curve.detector_type)
    return band_average_flux_lambda(sed_wavelength, sed_flux, curve) * pivot**2 / SPEED_OF_LIGHT_ANG_PER_S

def load_spectrum(path: str, wavelength_unit: str | None = None) -> Spectrum:
    wavelengths, fluxes, metadata = load_two_column_file(path)
    wavelength, order = _standardize_file_wavelengths(path, wavelengths, metadata, wavelength_unit)
    return Spectrum(wavelength, np.asarray(fluxes, dtype=float)[order], metadata, wavelength_unit="angstrom")


def save_spectrum(path, wavelength, flux, parameters: Dict[str, float], extra: Dict[str, object]) -> None:
    """Write a spectrum in SED_Tools' text format, as its grids store theirs: a ``# key = value`` line for each
    parameter (one that is NaN, unknown, is left out), the units and each ``extra`` entry, then a line per sample:
    wavelength (Å) and flux (erg/s/cm²/Å)."""
    units = {"wavelength_unit": "angstrom", "flux_unit": "erg/cm2/s/A", "units_standardized": True}
    header = {**{k: v for k, v in parameters.items() if not np.isnan(v)}, **units,
              **{k: v for k, v in extra.items() if k not in parameters and k not in units}}
    with open(Path(path), "w") as f:
        f.writelines(f"# {key} = {value}\n" for key, value in header.items())
        f.writelines(f"{w:.6f} {fl:.8e}\n" for w, fl in zip(wavelength, flux))

def vega_zero_point(curve: FilterCurve, vega: Spectrum | None = None, find_vega=None) -> float:
    """The zero point of ``curve``'s Vega magnitudes, Vega's band-averaged f_lambda: from ``vega``, a Vega spectrum,
    when one is given; otherwise the zero point the curve states, in its header (:data:`VEGA_ZP_KEYS`) or as
    ``ZeroPointVega_flam`` in the ``properties.json`` beside its file (NJM's, saved with the curves); otherwise from
    the spectrum ``find_vega()`` returns. ValueError when there is none."""
    if vega is None:
        try:
            stated = get_metadata_float(curve.metadata, VEGA_ZP_KEYS)
        except ValueError as exc:
            raise ValueError(f"Filter '{curve.name}': {exc}") from exc
        if stated is None and curve.path is not None:
            stated = _published_vega_zero_point(Path(curve.path))
        if stated is not None:
            return stated
        if find_vega is None:
            raise ValueError(f"Filter '{curve.name}' states no Vega zero point, in its header or a properties.json "
                             "beside it. Supply a Vega spectrum via the 'vega_spectrum' argument.")
        vega = find_vega()
    return band_average_flux_lambda(vega.wavelength, vega.flux, curve)


def _published_vega_zero_point(path: Path) -> float | None:
    """``ZeroPointVega_flam`` of the band named as the file ``path`` in the ``properties.json`` beside it, if any."""
    properties = path.with_name("properties.json")
    if not properties.is_file():
        return None
    value = (json.loads(properties.read_text(encoding="utf-8"))["bands"].get(path.stem) or {}).get("ZeroPointVega_flam")
    return None if value is None else float(value)


def _band_magnitude(wl, flux, curve, photometric_system, find_vega, given_vega: bool = False):
    """Return (flux label, band-averaged flux, magnitude) for one filter.

    In both systems the magnitude of a zero band flux is ``inf`` and of a
    negative one NaN; a zero or negative Vega zero point gives ``-inf`` or NaN.
    The Vega zero point is :func:`vega_zero_point`'s: *find_vega* is called for
    a Vega spectrum only when the curve states no zero point, or first when
    ``given_vega`` (it returns a spectrum the user supplied, which wins).
    """
    if photometric_system == "AB":
        flux_density = band_average_flux_nu(wl, flux, curve)
        # A zero or negative band flux gives inf or NaN, as computed; no error.
        with np.errstate(divide="ignore", invalid="ignore"):
            return "f_nu", flux_density, float(-2.5 * np.log10(flux_density / AB_ZERO_FLUX))

    flux_density = band_average_flux_lambda(wl, flux, curve)
    vega_zero = vega_zero_point(curve, find_vega() if given_vega else None, find_vega)
    # A zero or negative band flux or zero point gives inf, -inf or NaN, as computed; no error.
    with np.errstate(divide="ignore", invalid="ignore"):
        return "f_lambda", flux_density, float(-2.5 * np.log10(np.divide(flux_density, vega_zero)))
