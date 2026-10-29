"""The local spectral grids' flux cubes as SEDs: SED (sed_tools.models.SED) finds the installed grids that cover a
point or range, SEDModel interpolates one grid's flux_cube.bin, and EvaluatedSED gives photometry. sed_tools.SED
(api/) is the public class built on them; "model" here means an atmosphere grid, not an ML model.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..config import get_data_dir
from ..filters.compose import is_auxiliary_path
from ..photometry import (AB_ZERO_FLUX, FILTER_EXTENSIONS, VEGA_ZP_KEYS, FilterCurve, Spectrum, _warn_outside_sed,
                          band_average_flux_lambda, band_average_flux_lambda_from_arrays, band_average_flux_nu,
                          get_metadata_float, load_filter_curve, load_spectrum)
from .fileio import read_axes
from .tool import FluxCube

Number = Union[int, float, np.floating]
FilterSpec = Union[str, os.PathLike[str], FilterCurve, Tuple[str, Union[str, os.PathLike[str]]]]

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR_DEFAULT     = get_data_dir()
STELLAR_DIR_DEFAULT  = Path(os.environ.get("SED_STELLAR_DIR", DATA_DIR_DEFAULT / "stellar_models")).expanduser()
FILTER_DIR_DEFAULT   = Path(os.environ.get("SED_FILTER_DIR",  DATA_DIR_DEFAULT / "filters")).expanduser()

@dataclass(order=True)
class ModelMatch:
    """Represents how well a stored flux cube covers a requested parameter."""

    distance: float
    name: str
    flux_cube: Path
    teff_range: Tuple[float, float]
    logg_range: Tuple[float, float]
    metallicity_range: Tuple[float, float]
    contains_point: bool
    covers_range: bool = field(default=False, compare=False)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "flux_cube": str(self.flux_cube),
            "teff_range": self.teff_range,
            "logg_range": self.logg_range,
            "metallicity_range": self.metallicity_range,
            "distance": self.distance,
            "contains_point": self.contains_point,
            "covers_range": self.covers_range,
        }


@dataclass
class PhotometryResult:
    """Synthetic photometry for a single filter band."""

    magnitude: float
    flux_density: float
    system: str
    filter_name: str


@dataclass
class EvaluatedSED:
    """Interpolated spectrum at a specific point in parameter space."""

    wavelength: np.ndarray
    flux: np.ndarray
    metadata: Dict[str, float]
    _model: "SEDModel"

    @property
    def teff(self) -> float:
        return self.metadata["teff"]

    @property
    def logg(self) -> float:
        return self.metadata["logg"]

    @property
    def metallicity(self) -> float:
        return self.metadata["metallicity"]

    @property
    def wl(self) -> np.ndarray:
        """Alias for wavelength."""
        return self.wavelength

    @property
    def fl(self) -> np.ndarray:
        """Alias for flux."""
        return self.flux

    def photometry(
        self,
        *filters: FilterSpec,
        system: str = "AB",
        vega_spectrum: Union[None, Spectrum, str, os.PathLike[str]] = None,
    ) -> Dict[str, PhotometryResult]:
        """Compute synthetic photometry for the supplied filters.

        Parameters
        ----------
        filters:
            Filter specifications. Each entry can be a name (matched against the
            filter repository), a path to a transmission file, or a pre-loaded
            :class:`FilterCurve`. ``(name, path)`` tuples are also accepted.
        system:
            Photometric system to use. "AB" (default) computes magnitudes using
            the AB zero-point. "Vega" relies on either the filter metadata or on
            a provided Vega reference spectrum. In both, a band flux of zero
            gives an infinite magnitude and a negative one NaN. A Vega zero
            point of zero gives ``-inf``, and a negative one NaN.
        vega_spectrum:
            Optional spectrum to use as Vega reference when ``system="Vega"``
            and the filter file does not encode a zero point. Can be provided as
            a :class:`Spectrum` instance or a filesystem path.
        """

        if not filters:
            raise ValueError("At least one filter must be supplied")

        resolved = self._model._resolve_filters(filters)
        results: Dict[str, PhotometryResult] = {}

        system = system.upper()
        if system not in {"AB", "VEGA"}:
            raise ValueError("Photometric system must be either 'AB' or 'Vega'")

        vega_curve: Optional[Spectrum] = vega_spectrum
        if vega_spectrum is not None and not isinstance(vega_spectrum, Spectrum):
            vega_curve = load_spectrum(os.fspath(vega_spectrum))

        _warn_outside_sed(self.wavelength, resolved.values())
        for name, curve in resolved.items():
            # A zero or negative band flux or Vega zero point gives inf, -inf or NaN, as computed; no error.
            if system == "AB":
                flux_density = band_average_flux_nu(self.wavelength, self.flux, curve)
                with np.errstate(divide="ignore", invalid="ignore"):
                    magnitude = float(-2.5 * np.log10(flux_density / AB_ZERO_FLUX))
            else:
                flux_density = band_average_flux_lambda(self.wavelength, self.flux, curve)
                vega_zero = _vega_zero_point(curve, vega_curve)
                with np.errstate(divide="ignore", invalid="ignore"):
                    magnitude = float(-2.5 * np.log10(np.divide(flux_density, vega_zero)))
            results[name] = PhotometryResult(
                magnitude=magnitude,
                flux_density=float(flux_density),
                system="AB" if system == "AB" else "Vega",
                filter_name=name,
            )

        return results


def _vega_zero_point(curve: FilterCurve, vega_curve: Optional[Spectrum]) -> float:
    try:
        metadata_value = get_metadata_float(curve.metadata, VEGA_ZP_KEYS)
    except ValueError as exc:
        raise ValueError(f"Filter '{curve.name}': {exc}") from exc
    if metadata_value is not None:
        return float(metadata_value)
    if vega_curve is None:
        raise ValueError(
            "Filter does not provide a Vega zero point. Supply a Vega spectrum via "
            "the 'vega_spectrum' argument."
        )
    return band_average_flux_lambda(vega_curve.wavelength, vega_curve.flux, curve)


class SEDModel:
    """Thin wrapper around a flux cube providing interpolation utilities.

    With ``fill_gaps=True`` (the default) a requested Teff, logg or [M/H]
    outside the grid is clamped to the nearest grid edge; with ``False`` it
    raises ValueError.
    """

    def __init__(
        self,
        name: str,
        flux_cube_path: Union[str, os.PathLike[str]],
        filters_dir: Optional[Union[str, os.PathLike[str]]] = None,
        *,
        interpolation: str = "hermite",
        fill_gaps: bool = True,
    ) -> None:
        self.name = name
        self.flux_cube_path = Path(flux_cube_path)
        self.filters_dir = Path(filters_dir) if filters_dir is not None else None
        self.interpolation = interpolation.lower()
        if self.interpolation != "hermite":
            raise ValueError("Only Hermite interpolation is supported at present")
        self.fill_gaps = fill_gaps

        metadata = _read_flux_cube_header(self.flux_cube_path)
        self.teff_grid = metadata["teff"]
        self.logg_grid = metadata["logg"]
        self.meta_grid = metadata["meta"]
        self.wavelengths = metadata["wavelengths"]

        self._cube: Optional[FluxCube] = None
        self._filter_cache: Dict[str, FilterCurve] = {}
        self._filter_index: Optional[Dict[str, List[Path]]] = None

    # --- Flux cube loading & interpolation ---

    def _load_cube(self) -> FluxCube:
        if self._cube is None:
            self._cube = FluxCube.from_file(str(self.flux_cube_path))
        return self._cube

    def __call__(self, teff: Number, logg: Number, metallicity: Number) -> EvaluatedSED:
        cube = self._load_cube()
        teff_val = self._prepare_value(float(teff), self.teff_grid, "Teff")
        logg_val = self._prepare_value(float(logg), self.logg_grid, "logg")
        meta_val = self._prepare_value(float(metallicity), self.meta_grid, "[M/H]")
        wavelength, flux = cube.interpolate_spectrum(teff_val, logg_val, meta_val)
        metadata = {"teff": teff_val, "logg": logg_val, "metallicity": meta_val}
        return EvaluatedSED(wavelength=wavelength, flux=flux, metadata=metadata, _model=self)

    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            "teff": (float(self.teff_grid[0]), float(self.teff_grid[-1])),
            "logg": (float(self.logg_grid[0]), float(self.logg_grid[-1])),
            "metallicity": (float(self.meta_grid[0]), float(self.meta_grid[-1])),
        }

    # --- Filter handling ---

    def _resolve_filters(self, filters: Sequence[FilterSpec]) -> Dict[str, FilterCurve]:
        entries = _normalize_filter_sequence(filters)
        resolved: Dict[str, FilterCurve] = {}

        for item in entries:
            if isinstance(item, FilterCurve):
                resolved[item.name] = item
            elif isinstance(item, tuple):
                custom_name, source = item
                curve = self._load_filter(source, custom_name=custom_name)
                resolved[curve.name] = curve
            elif isinstance(item, (str, os.PathLike)):
                resolved.update(self._load_filter_group(item))
            else:
                raise TypeError(f"Unsupported filter specification: {item!r}")
        return resolved

    def _load_filter_group(self, spec: Union[str, os.PathLike[str]]) -> Dict[str, FilterCurve]:
        curves: Dict[str, FilterCurve] = {}
        for path in self._locate_filter_paths(spec):
            key = (os.fspath(path), "")
            if key not in self._filter_cache:
                self._filter_cache[key] = load_filter_curve(str(path))
            curve = self._filter_cache[key]
            curves[curve.name] = curve
        return curves

    def _load_filter(
        self,
        spec: Union[str, os.PathLike[str]],
        *,
        custom_name: Optional[str] = None,
    ) -> FilterCurve:
        key = (os.fspath(spec), custom_name or "")
        if key not in self._filter_cache:
            self._filter_cache[key] = load_filter_curve(str(self._locate_filter_path(spec)), name=custom_name)
        return self._filter_cache[key]

    def _locate_filter_paths(self, spec: Union[str, os.PathLike[str]]) -> List[Path]:
        path = Path(spec)
        if path.is_file():
            return [path]

        if path.is_dir():
            files = _filter_files_in_dir(path)
            if files:
                return files

        if self.filters_dir is None:
            raise FileNotFoundError(
                f"Filter '{spec}' could not be resolved because no filters directory is configured."
            )

        filters_dir = Path(self.filters_dir)

        candidate = filters_dir / path
        if candidate.is_file():
            return [candidate]
        if candidate.is_dir():
            files = _filter_files_in_dir(candidate)
            if files:
                return files

        # Common SED_Tools/MESA layout:
        #   filters/<Facility>/<Instrument>/*.dat
        # This lets photometry("GAIA") resolve filters/GAIA/GAIA/*.dat,
        # photometry("2MASS") resolve filters/2MASS/2MASS/*.dat, etc.
        if not path.is_absolute() and len(path.parts) == 1:
            lookup_dir = str(path).lower()
            dir_matches: List[Path] = []
            for candidate_dir in filters_dir.rglob("*"):
                if not candidate_dir.is_dir() or is_auxiliary_path(candidate_dir, filters_dir):
                    continue
                if candidate_dir.name.lower() != lookup_dir:
                    continue
                files = _filter_files_in_dir(candidate_dir)
                if files:
                    dir_matches.extend(files)

            if dir_matches:
                return sorted({p.resolve(): p for p in dir_matches}.values())

        if self._filter_index is None:
            self._filter_index = _build_filter_index(filters_dir)

        lookup = str(spec).lower()
        if lookup in self._filter_index and len(self._filter_index[lookup]) == 1:
            return [self._filter_index[lookup][0]]

        matches = [p for key, paths in self._filter_index.items() if lookup in key for p in paths]
        if not matches:
            raise FileNotFoundError(f"No filter file matching '{spec}' was found under {filters_dir}")
        if len(matches) > 1:
            options = ", ".join(str(p) for p in matches)
            raise FileExistsError(f"Filter specification '{spec}' is ambiguous; matches: {options}")
        return [matches[0]]

    def _locate_filter_path(self, spec: Union[str, os.PathLike[str]]) -> Path:
        paths = self._locate_filter_paths(spec)
        if len(paths) != 1:
            options = ", ".join(str(p) for p in paths)
            raise FileExistsError(
                f"Filter specification '{spec}' resolves to multiple filters; "
                f"use photometry(...) without a custom name or pass one file. Matches: {options}"
            )
        return paths[0]

    # --- Helpers ---

    def _prepare_value(self, value: float, grid: np.ndarray, label: str) -> float:
        lower = float(grid[0])
        upper = float(grid[-1])
        if lower <= value <= upper:
            return value
        if not self.fill_gaps:
            raise ValueError(f"Requested {label}={value} is outside the grid range [{lower}, {upper}].")
        return float(min(max(value, lower), upper))


class SED:
    """The installed spectral grids under model_root: find the ones covering a point or range, and evaluate one.

    sed_tools.SED (api/) wraps this class; it is importable as sed_tools.models.SED.
    """

    def __init__(
        self,
        *,
        model_root: Union[str, os.PathLike[str], None] = None,
        filter_root: Union[str, os.PathLike[str], None] = None,
        atm: Optional[Union[str, os.PathLike[str], ModelMatch, "SEDModel"]] = None,
        interpolation: str = "hermite",
        fill_gaps: bool = True,
    ) -> None:
        self.model_root = Path(model_root) if model_root else STELLAR_DIR_DEFAULT
        self.filter_root = Path(filter_root) if filter_root else FILTER_DIR_DEFAULT
        self._metadata_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._active_model: Optional[SEDModel] = None

        if atm is not None:
            self.model(atm, interpolation=interpolation, fill_gaps=fill_gaps, activate=True)

    # --- Model discovery ---

    def available_grids(self) -> List[str]:
        """List locally installed spectral grids with a flux cube."""
        if not self.model_root.exists():
            return []
        entries = sorted(self.model_root.iterdir())
        return [entry.name for entry in entries if entry.is_dir() and (entry / "flux_cube.bin").is_file()]

    available_models = available_grids

    def find_model(
        self,
        teff: Number,
        logg: Number,
        metallicity: Number,
        *,
        limit: Optional[int] = None,
        allow_partial: bool = False,
    ) -> List[ModelMatch]:
        candidates: List[ModelMatch] = []
        for name in self.available_models():
            cube_path = self.model_root / name / "flux_cube.bin"
            metadata = self._get_metadata(name, cube_path)
            grids = [metadata[key] for key in ("teff", "logg", "meta")]
            bounds = [(float(grid[0]), float(grid[-1])) for grid in grids]
            point = (float(teff), float(logg), float(metallicity))
            contains = _contains_point(*point, metadata)
            if not contains and not allow_partial:
                continue
            candidates.append(ModelMatch(
                distance=_parameter_distance(*zip(point, grids)),
                name=name,
                flux_cube=cube_path,
                teff_range=bounds[0],
                logg_range=bounds[1],
                metallicity_range=bounds[2],
                contains_point=contains,
            ))
        return sorted(candidates)[:limit]

    def find_grids(
        self,
        *,
        teff_range: Optional[Sequence[Number]] = None,
        logg_range: Optional[Sequence[Number]] = None,
        metallicity_range: Optional[Sequence[Number]] = None,
        limit: Optional[int] = None,
        allow_partial: bool = False,
    ) -> List[ModelMatch]:
        """Discover local spectral grids compatible with the provided parameter ranges.

        Parameters
        ----------
        teff_range, logg_range, metallicity_range:
            Inclusive parameter ranges expressed as ``(min, max)`` pairs. Any
            parameter left as ``None`` will not be used for filtering.
        limit:
            Optional maximum number of matches to return after sorting by the
            proximity of the requested range to the spectral grid.
        allow_partial:
            When ``False`` (default) only spectral grids that fully cover the supplied
            ranges are returned. Set to ``True`` to also include partial overlaps
            which can be useful when exploring sparse grids.
        """

        if teff_range is None and logg_range is None and metallicity_range is None:
            raise ValueError("Provide at least one parameter range to search for")

        requested = [_normalize_range(teff_range, "teff"), _normalize_range(logg_range, "logg"),
                     _normalize_range(metallicity_range, "metallicity")]

        candidates: List[ModelMatch] = []
        for name in self.available_models():
            cube_path = self.model_root / name / "flux_cube.bin"
            metadata = self._get_metadata(name, cube_path)
            grids = [metadata[key] for key in ("teff", "logg", "meta")]
            bounds = [(float(grid[0]), float(grid[-1])) for grid in grids]
            coverage = [_range_coverage(wanted, bound) for wanted, bound in zip(requested, bounds)]
            if not all(overlaps for overlaps, _ in coverage):
                continue
            covers_range = all(covers for _, covers in coverage)
            if not covers_range and not allow_partial:
                continue
            centers = [_range_center(wanted, grid) for wanted, grid in zip(requested, grids)]
            candidates.append(ModelMatch(
                distance=_parameter_distance(*zip(centers, grids)),
                name=name,
                flux_cube=cube_path,
                teff_range=bounds[0],
                logg_range=bounds[1],
                metallicity_range=bounds[2],
                contains_point=_contains_point(*centers, metadata),
                covers_range=covers_range,
            ))
        return sorted(candidates)[:limit]

    find_atmospheres = find_grids

    # --- Model loading ---

    def model(
        self,
        model: Union[str, os.PathLike[str], ModelMatch, SEDModel],
        *,
        interpolation: str = "hermite",
        fill_gaps: bool = True,
        activate: bool = True,
    ) -> SEDModel:
        if isinstance(model, SEDModel):
            result = model
        else:
            flux_path, name = self._resolve_model_inputs(model)
            result = SEDModel(
                name=name,
                flux_cube_path=flux_path,
                filters_dir=self.filter_root if self.filter_root.exists() else None,
                interpolation=interpolation,
                fill_gaps=fill_gaps,
            )

        if activate:
            self._active_model = result
        return result

    def __call__(
        self,
        teff: Number,
        logg: Number,
        metallicity: Optional[Number] = None,
        **aliases: Number,
    ) -> EvaluatedSED:
        if self._active_model is None:
            raise RuntimeError(
                "No atmosphere selected. Call 'model(...)' or pass 'atm=' when constructing SED."
            )
        if metallicity is None:
            for key in ("z", "Z"):
                if key in aliases:
                    metallicity = aliases.pop(key)
                    break
        if aliases:
            unexpected = ", ".join(sorted(aliases.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if metallicity is None:
            raise TypeError("Metallicity value missing; supply 'metallicity' or 'z'.")
        return self._active_model(teff, logg, metallicity)

    # --- Helpers ---

    def _resolve_model_inputs(
        self, model: Union[str, os.PathLike[str], ModelMatch]
    ) -> Tuple[Path, str]:
        if isinstance(model, ModelMatch):
            return Path(model.flux_cube), model.name

        flux_path = self._resolve_model_path(model)
        name = flux_path.parent.name if flux_path.parent != flux_path else flux_path.stem
        return flux_path, name

    def _resolve_model_path(self, model: Union[str, os.PathLike[str]]) -> Path:
        path = Path(model)
        if path.is_file() and path.suffix.lower() == ".bin":
            return path
        for directory in (path, self.model_root / path):
            if directory.is_dir() and (directory / "flux_cube.bin").is_file():
                return directory / "flux_cube.bin"

        candidate = self.model_root / f"{path}.bin"
        if candidate.is_file():
            return candidate

        raise FileNotFoundError(
            f"Could not resolve model '{model}'. Provide either a model directory or a flux_cube.bin file."
        )

    def _get_metadata(self, name: str, cube_path: Path) -> Dict[str, np.ndarray]:
        cache_key = str(cube_path)
        if cache_key not in self._metadata_cache:
            self._metadata_cache[cache_key] = _read_flux_cube_header(cube_path)
        return self._metadata_cache[cache_key]



def _read_flux_cube_header(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as fh:
        header = fh.read(16)
        if len(header) != 16:
            raise ValueError(f"Flux cube header truncated in {path}")
        teff, logg, meta, wavelengths = read_axes(fh, *struct.unpack("4i", header))
    return {"teff": teff, "logg": logg, "meta": meta, "wavelengths": wavelengths}


def _contains_point(teff: float, logg: float, meta: float, metadata: Mapping[str, np.ndarray]) -> bool:
    teff_grid = metadata["teff"]
    logg_grid = metadata["logg"]
    meta_grid = metadata["meta"]
    return (
        teff_grid[0] <= teff <= teff_grid[-1]
        and logg_grid[0] <= logg <= logg_grid[-1]
        and meta_grid[0] <= meta <= meta_grid[-1]
    )


def _parameter_distance(*entries: Tuple[float, np.ndarray]) -> float:
    total = 0.0
    for value, grid in entries:
        if grid.size == 0:
            continue
        nearest = _nearest_on_grid(value, grid)
        span = float(max(grid) - min(grid)) or 1.0
        total += ((value - nearest) / span) ** 2
    return math.sqrt(total)


def _nearest_on_grid(value: float, grid: np.ndarray) -> float:
    idx = int(np.searchsorted(grid, value))
    if idx <= 0:
        return float(grid[0])
    if idx >= len(grid):
        return float(grid[-1])
    left = float(grid[idx - 1])
    right = float(grid[idx])
    return left if abs(value - left) <= abs(value - right) else right


def _filter_files_in_dir(directory: Union[str, os.PathLike[str]]) -> List[Path]:
    directory = Path(directory)

    direct = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in FILTER_EXTENSIONS
    )
    if direct:
        return direct

    child_dirs = sorted(p for p in directory.iterdir() if p.is_dir())
    if len(child_dirs) == 1:
        return _filter_files_in_dir(child_dirs[0])

    return []


def _build_filter_index(root: Union[str, os.PathLike[str]]) -> Dict[str, List[Path]]:
    root = Path(root)
    index: Dict[str, List[Path]] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file() and not is_auxiliary_path(path, root) and path.suffix.lower() in FILTER_EXTENSIONS:
            index.setdefault(path.stem.lower(), []).append(path)
    return index


def _normalize_filter_sequence(filters: Sequence[FilterSpec]) -> List[FilterSpec]:
    """The filter specs given, with a lone list, set or tuple of specs unpacked; a lone ``(name, path)`` tuple
    stays one spec when its second item is a path object or names an existing file (``("V", "B")`` is two)."""
    if len(filters) == 1 and isinstance(filters[0], (list, tuple, set)):
        only = filters[0]
        is_named_path = (isinstance(only, tuple) and len(only) == 2 and isinstance(only[0], str)
                         and (isinstance(only[1], os.PathLike)
                              or (isinstance(only[1], str) and os.path.isfile(only[1]))))
        if not is_named_path:
            return list(only)  # type: ignore[return-value]
    return list(filters)


def _normalize_range(value: Optional[Sequence[Number]], label: str) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{label} range must contain exactly two entries")
    start, end = float(value[0]), float(value[1])
    if start > end:
        start, end = end, start
    return start, end


def _range_coverage(
    requested: Optional[Tuple[float, float]],
    available: Tuple[float, float],
) -> Tuple[bool, bool]:
    if requested is None:
        return True, True
    req_min, req_max = requested
    avail_min, avail_max = available
    overlaps = req_max >= avail_min and req_min <= avail_max
    covers = avail_min <= req_min and avail_max >= req_max
    return overlaps, covers


def _range_center(requested: Optional[Tuple[float, float]], grid: np.ndarray) -> float:
    if requested is not None:
        return float((requested[0] + requested[1]) / 2.0)
    if grid.size == 0:
        return 0.0
    return float(grid[len(grid) // 2])


__all__ = [
    "SED",
    "SEDModel",
    "EvaluatedSED",
    "PhotometryResult",
    "ModelMatch",
    "STELLAR_DIR_DEFAULT",
    "FILTER_DIR_DEFAULT",
]
