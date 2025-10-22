from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ._flux import (
    AB_ZERO_FLUX,
    FILTER_EXTENSIONS,
    FilterCurve,
    FluxCube,
    Spectrum,
    VEGA_ZP_KEYS,
    band_average_flux_lambda,
    band_average_flux_lambda_from_arrays,
    band_average_flux_nu,
    load_filter_curve,
    load_spectrum,
)

Number = Union[int, float, np.floating]
FilterSpec = Union[str, os.PathLike[str], FilterCurve, Tuple[str, Union[str, os.PathLike[str]]]]

PACKAGE_ROOT = Path(__file__).resolve().parent
STELLAR_DIR_DEFAULT = Path(
    os.environ.get("SED_STELLAR_DIR", PACKAGE_ROOT / "data" / "stellar_models")
).expanduser()
FILTER_DIR_DEFAULT = Path(
    os.environ.get("SED_FILTER_DIR", PACKAGE_ROOT / "data" / "filters")
).expanduser()


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
            a provided Vega reference spectrum.
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

        vega_curve: Optional[Spectrum]
        if isinstance(vega_spectrum, Spectrum):
            vega_curve = vega_spectrum
        elif vega_spectrum is None:
            vega_curve = None
        else:
            vega_curve = load_spectrum(os.fspath(vega_spectrum))

        for name, curve in resolved.items():
            if system == "AB":
                flux_density = band_average_flux_nu(
                    self.wavelength, self.flux, curve
                )
                if flux_density <= 0:
                    magnitude = math.inf
                else:
                    magnitude = -2.5 * math.log10(flux_density / AB_ZERO_FLUX)
                results[name] = PhotometryResult(
                    magnitude=magnitude,
                    flux_density=float(flux_density),
                    system="AB",
                    filter_name=name,
                )
                continue

            # Vega system
            flux_density = band_average_flux_lambda(
                self.wavelength, self.flux, curve
            )
            vega_zero = _vega_zero_point(curve, vega_curve)
            if vega_zero <= 0:
                raise ValueError(
                    f"Vega zero point for filter '{name}' is non-positive; "
                    "cannot compute magnitude."
                )
            magnitude = -2.5 * math.log10(flux_density / vega_zero)
            results[name] = PhotometryResult(
                magnitude=magnitude,
                flux_density=float(flux_density),
                system="Vega",
                filter_name=name,
            )

        return results


def _vega_zero_point(curve: FilterCurve, vega_curve: Optional[Spectrum]) -> float:
    metadata_value = _extract_metadata_float(curve.metadata, VEGA_ZP_KEYS)
    if metadata_value is not None:
        return float(metadata_value)
    if vega_curve is None:
        raise ValueError(
            "Filter does not provide a Vega zero point. Supply a Vega spectrum via "
            "the 'vega_spectrum' argument."
        )
    interp_flux = np.interp(
        curve.wavelength,
        vega_curve.wavelength,
        vega_curve.flux,
        left=0.0,
        right=0.0,
    )
    return band_average_flux_lambda_from_arrays(
        curve.wavelength, interp_flux, curve.transmission
    )


def _extract_metadata_float(metadata: Mapping[str, object], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


class SEDModel:
    """Thin wrapper around a flux cube providing interpolation utilities."""

    def __init__(
        self,
        name: str,
        flux_cube_path: Union[str, os.PathLike[str]],
        filters_dir: Optional[Path] = None,
        *,
        interpolation: str = "hermite",
        fill_gaps: bool = True,
    ) -> None:
        self.name = name
        self.flux_cube_path = Path(flux_cube_path)
        self.filters_dir = filters_dir
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

    # ------------------------------------------------------------------
    # Flux cube loading & interpolation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Filter handling
    # ------------------------------------------------------------------

    def _resolve_filters(self, filters: Sequence[FilterSpec]) -> Dict[str, FilterCurve]:
        entries = _normalize_filter_sequence(filters)
        resolved: Dict[str, FilterCurve] = {}

        for item in entries:
            if isinstance(item, FilterCurve):
                resolved[item.name] = item
                continue

            if isinstance(item, tuple):
                custom_name, source = item
                curve = self._load_filter(source, custom_name=custom_name)
                resolved[curve.name] = curve
                continue

            if isinstance(item, (str, os.PathLike)):
                curve = self._load_filter(item)
                resolved[curve.name] = curve
                continue

            raise TypeError(f"Unsupported filter specification: {item!r}")

        return resolved

    def _load_filter(
        self,
        spec: Union[str, os.PathLike[str]],
        *,
        custom_name: Optional[str] = None,
    ) -> FilterCurve:
        key = (os.fspath(spec), custom_name or "")
        if key in self._filter_cache:
            return self._filter_cache[key]

        path = self._locate_filter_path(spec)
        curve = load_filter_curve(str(path), name=custom_name)
        self._filter_cache[key] = curve
        return curve

    def _locate_filter_path(self, spec: Union[str, os.PathLike[str]]) -> Path:
        path = Path(spec)
        if path.is_file():
            return path

        if self.filters_dir is None:
            raise FileNotFoundError(
                f"Filter '{spec}' could not be resolved because no filters directory is configured."
            )

        if self._filter_index is None:
            self._filter_index = _build_filter_index(self.filters_dir)

        lookup = str(spec).lower()
        if lookup in self._filter_index and len(self._filter_index[lookup]) == 1:
            return self._filter_index[lookup][0]

        matches = [p for key, paths in self._filter_index.items() if lookup in key for p in paths]
        if not matches:
            raise FileNotFoundError(f"No filter file matching '{spec}' was found under {self.filters_dir}")
        if len(matches) > 1:
            options = ", ".join(str(p) for p in matches)
            raise FileExistsError(
                f"Filter specification '{spec}' is ambiguous; matches: {options}"
            )
        return matches[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_value(self, value: float, grid: np.ndarray, label: str) -> float:
        lower = float(grid[0])
        upper = float(grid[-1])
        if lower <= value <= upper:
            return value
        if not self.fill_gaps:
            raise ValueError(
                f"Requested {label}={value} is outside the grid range [{lower}, {upper}]."
            )
        return float(min(max(value, lower), upper))


class SED:
    """Facade for working with the local SED model repository."""

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
            self.model(
                atm,
                interpolation=interpolation,
                fill_gaps=fill_gaps,
                activate=True,
            )

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def available_models(self) -> List[str]:
        if not self.model_root.exists():
            return []
        models: List[str] = []
        for entry in sorted(self.model_root.iterdir()):
            if not entry.is_dir():
                continue
            cube = entry / "flux_cube.bin"
            if cube.is_file():
                models.append(entry.name)
        return models

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
            teff_range = (float(metadata["teff"][0]), float(metadata["teff"][-1]))
            logg_range = (float(metadata["logg"][0]), float(metadata["logg"][-1]))
            meta_range = (float(metadata["meta"][0]), float(metadata["meta"][-1]))

            contains = _contains_point(
                float(teff), float(logg), float(metallicity), metadata
            )
            if not contains and not allow_partial:
                continue

            distance = _parameter_distance(
                (float(teff), metadata["teff"]),
                (float(logg), metadata["logg"]),
                (float(metallicity), metadata["meta"]),
            )
            match = ModelMatch(
                distance=distance,
                name=name,
                flux_cube=cube_path,
                teff_range=teff_range,
                logg_range=logg_range,
                metallicity_range=meta_range,
                contains_point=contains,
            )
            candidates.append(match)

        candidates.sort()
        if limit is not None:
            return candidates[:limit]
        return candidates

    def find_atmospheres(
        self,
        *,
        teff_range: Optional[Sequence[Number]] = None,
        logg_range: Optional[Sequence[Number]] = None,
        metallicity_range: Optional[Sequence[Number]] = None,
        limit: Optional[int] = None,
        allow_partial: bool = False,
    ) -> List[ModelMatch]:
        """Discover flux cubes compatible with the provided parameter ranges.

        Parameters
        ----------
        teff_range, logg_range, metallicity_range:
            Inclusive parameter ranges expressed as ``(min, max)`` pairs. Any
            parameter left as ``None`` will not be used for filtering.
        limit:
            Optional maximum number of matches to return after sorting by the
            proximity of the requested range to the model grid.
        allow_partial:
            When ``False`` (default) only models that fully cover the supplied
            ranges are returned. Set to ``True`` to also include partial overlaps
            which can be useful when exploring sparse grids.
        """

        if teff_range is None and logg_range is None and metallicity_range is None:
            raise ValueError("Provide at least one parameter range to search for")

        teff_requested = _normalize_range(teff_range, "teff")
        logg_requested = _normalize_range(logg_range, "logg")
        meta_requested = _normalize_range(metallicity_range, "metallicity")

        candidates: List[ModelMatch] = []
        for name in self.available_models():
            cube_path = self.model_root / name / "flux_cube.bin"
            metadata = self._get_metadata(name, cube_path)
            teff_grid = metadata["teff"]
            logg_grid = metadata["logg"]
            meta_grid = metadata["meta"]

            teff_bounds = (float(teff_grid[0]), float(teff_grid[-1]))
            logg_bounds = (float(logg_grid[0]), float(logg_grid[-1]))
            meta_bounds = (float(meta_grid[0]), float(meta_grid[-1]))

            overlap_teff, cover_teff = _range_coverage(teff_requested, teff_bounds)
            overlap_logg, cover_logg = _range_coverage(logg_requested, logg_bounds)
            overlap_meta, cover_meta = _range_coverage(meta_requested, meta_bounds)

            if not (overlap_teff and overlap_logg and overlap_meta):
                continue

            covers_range = cover_teff and cover_logg and cover_meta
            if not covers_range and not allow_partial:
                continue

            teff_value = _range_center(teff_requested, teff_grid)
            logg_value = _range_center(logg_requested, logg_grid)
            meta_value = _range_center(meta_requested, meta_grid)

            distance = _parameter_distance(
                (teff_value, teff_grid),
                (logg_value, logg_grid),
                (meta_value, meta_grid),
            )

            match = ModelMatch(
                distance=distance,
                name=name,
                flux_cube=cube_path,
                teff_range=teff_bounds,
                logg_range=logg_bounds,
                metallicity_range=meta_bounds,
                contains_point=_contains_point(teff_value, logg_value, meta_value, metadata),
                covers_range=covers_range,
            )
            candidates.append(match)

        candidates.sort()
        if limit is not None:
            return candidates[:limit]
        return candidates

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

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
            filters_dir = self.filter_root if self.filter_root.exists() else None
            result = SEDModel(
                name=name,
                flux_cube_path=flux_path,
                filters_dir=filters_dir,
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_model_inputs(
        self, model: Union[str, os.PathLike[str], ModelMatch]
    ) -> Tuple[Path, str]:
        if isinstance(model, ModelMatch):
            path = Path(model.flux_cube)
            name = model.name
            return path, name

        flux_path = self._resolve_model_path(model)
        name = flux_path.parent.name if flux_path.parent != flux_path else flux_path.stem
        return flux_path, name

    def _resolve_model_path(self, model: Union[str, os.PathLike[str]]) -> Path:
        path = Path(model)
        if path.is_file() and path.suffix.lower() == ".bin":
            return path
        if path.is_dir():
            cube = path / "flux_cube.bin"
            if cube.is_file():
                return cube

        candidate = self.model_root / path
        if candidate.is_dir():
            cube = candidate / "flux_cube.bin"
            if cube.is_file():
                return cube

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


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------


def _read_flux_cube_header(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as fh:
        header = fh.read(16)
        if len(header) != 16:
            raise ValueError(f"Flux cube header truncated in {path}")
        nt, nl, nm, nw = struct.unpack("4i", header)
        teff = np.fromfile(fh, dtype=np.float64, count=nt)
        logg = np.fromfile(fh, dtype=np.float64, count=nl)
        meta = np.fromfile(fh, dtype=np.float64, count=nm)
        wavelengths = np.fromfile(fh, dtype=np.float64, count=nw)
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


def _build_filter_index(root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in FILTER_EXTENSIONS:
            continue
        key = path.stem.lower()
        index.setdefault(key, []).append(path)
    return index


def _normalize_filter_sequence(filters: Sequence[FilterSpec]) -> List[FilterSpec]:
    if len(filters) == 1 and isinstance(filters[0], (list, tuple, set)):
        inner = filters[0]
        if isinstance(inner, (list, tuple, set)):
            return list(inner)  # type: ignore[return-value]
    return list(filters)


def _normalize_range(
    value: Optional[Sequence[Number]], label: str
) -> Optional[Tuple[float, float]]:
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


def _range_center(
    requested: Optional[Tuple[float, float]], grid: np.ndarray
) -> float:
    if requested is not None:
        return float((requested[0] + requested[1]) / 2.0)
    if grid.size == 0:
        return 0.0
    midpoint = grid[len(grid) // 2]
    return float(midpoint)


__all__ = [
    "SED",
    "SEDModel",
    "EvaluatedSED",
    "PhotometryResult",
    "ModelMatch",
    "STELLAR_DIR_DEFAULT",
    "FILTER_DIR_DEFAULT",
]
