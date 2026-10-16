"""Separable filter components: bandpass, optics, detector QE and atmosphere.

A published passband (e.g. from SVO) is usually the product of several
physical throughputs, and which ones varies by source: Gaia DR3 is
"Filter + CCD + Instrument", SDSS is "Filter + Instrument + Atmosphere",
Johnson is "Filter" only.  A product cannot be divided back into its factors,
so separate components exist only where the original team published them.

This module gives those components a home next to the published curves and
composes them on request.  Layout (all additive; the published ``<band>.dat``
files and the ``<Instrument>`` index that MESA reads are untouched)::

    filters/<Facility>/<Instrument>/
        <Instrument>, <band>.dat        published curves (as before)
        manifest.json                   what is available, with provenance
        components/<role>/<file>.dat    per-band or shared component curves
        products/<recipe>/              composed curves; a valid MESA instrument
            <recipe>, <band>.dat

    atmospheres/<site>/
        manifest.json                   airmass/PWV grid (ESO SkyCalc)
        <file>.dat

Component roles
---------------
``filter``      the bandpass filter transmission
``optics``      telescope + instrument optical throughput (mirrors, lenses, ...)
``qe``          detector quantum efficiency
``atmosphere``  telluric transmission (site, airmass, PWV dependent)
``instrument``  an unspecified mix of optics and detector, as some sources
                publish it; never composed, only used to describe published
                curves
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .filter_metadata import (ENERGY, PHOTON, normalize_detector_type,
                              read_filter_header, write_filter_file)

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

FILTER, OPTICS, QE, ATMOSPHERE, INSTRUMENT = "filter", "optics", "qe", "atmosphere", "instrument"
#: Pseudo-role selecting the published root-level curve as the base of a product.
PUBLISHED = "published"
#: Roles in canonical order; recipes are always written in this order.
ROLES = (FILTER, OPTICS, QE, INSTRUMENT, ATMOSPHERE)
#: Roles that can be selected when composing a product.
COMPOSABLE_ROLES = (FILTER, OPTICS, QE, ATMOSPHERE)

#: Sub-directories of an instrument that hold components and composed
#: products rather than published curves.
COMPONENTS_DIR, PRODUCTS_DIR = "components", "products"
AUXILIARY_DIRS = (COMPONENTS_DIR, PRODUCTS_DIR)


def is_auxiliary_path(path: PathLike, root: Optional[PathLike] = None, dirs: Sequence[str] = AUXILIARY_DIRS) -> bool:
    """True if ``path`` lies inside a ``components/`` or ``products/`` directory.

    Scanners that look for *published* curves by name skip these so that a
    composed ``products/filter/G.dat`` never shadows ``G.dat``.
    """
    path = Path(path)
    if root is not None:
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
    return any(part in dirs for part in path.parts)


MANIFEST_NAME = "manifest.json"
#: Suffixes SVO uses for the filter-only sibling of a total passband.
FILTER_SIBLING_SUFFIXES = ("_filter", "_fil", "prime_filter")
MANIFEST_SCHEMA = 1
SHARED = "*"  # component key meaning "applies to every band"

_SVO_TOKENS = {
    "filter": FILTER,
    "filters": FILTER,
    "theoretically defined": FILTER,  # idealised bandpasses (SVO)
    "ccd": QE,
    "detector": QE,
    "qe": QE,
    "quantum efficiency": QE,
    "instrument": INSTRUMENT,
    "optics": OPTICS,
    "other optical components": OPTICS,
    "telescope": OPTICS,
    "mirror": OPTICS,
    "mirrors": OPTICS,
    "atmosphere": ATMOSPHERE,
}


class ComponentError(ValueError):
    """A requested component combination cannot be built."""


class ComponentCoverageError(ComponentError):
    """A component does not cover the wavelength range of the band."""


# --------------------------------------------------------------------------
# Component vocabulary
# --------------------------------------------------------------------------
def parse_components(text: Union[str, Iterable[str], None]) -> Tuple[str, ...]:
    """Parse ``"filter+qe"``, ``["qe", "filter"]`` or SVO's ``"Filter + CCD"``.

    Returns the roles in canonical order.  Unknown tokens raise.
    """
    if text is None:
        return ()
    tokens = re.split(r"\s*\+\s*|\s*,\s*", text.strip()) if isinstance(text, str) else list(text)
    roles = set()
    published = False
    for token in tokens:
        key = token.strip().lower()
        if not key:
            continue
        if key == PUBLISHED:
            published = True
            continue
        if key not in _SVO_TOKENS:
            raise ComponentError(f"Unknown filter component: {token!r}")
        roles.add(_SVO_TOKENS[key])
    ordered = tuple(r for r in ROLES if r in roles)
    return ((PUBLISHED,) + ordered) if published else ordered


def recipe_name(roles: Iterable[str]) -> str:
    """Canonical ``+``-joined name for a set of roles, e.g. ``filter+optics+qe``."""
    return "+".join(parse_components(list(roles)))


# --------------------------------------------------------------------------
# Atmosphere selection
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AtmosphereSpec:
    """Observing conditions for the telluric component.

    ``site`` is always the instrument's own site (``manifest.default_site``);
    it may be omitted, and is only accepted if it matches.  Atmospheres are
    never borrowed from another site.
    """

    site: Optional[str]
    airmass: float
    pwv: Optional[float] = None  # precipitable water vapour, mm

    @classmethod
    def coerce(cls, value: Union["AtmosphereSpec", Mapping[str, object], None]) -> Optional["AtmosphereSpec"]:
        if value is None or isinstance(value, AtmosphereSpec):
            return value
        if "airmass" not in value:
            raise ComponentError("atmosphere needs at least 'airmass' (and 'pwv' for SkyCalc sites)")
        return cls(
            site=None if value.get("site") is None else str(value["site"]),
            airmass=float(value["airmass"]),  # type: ignore[arg-type]
            pwv=None if value.get("pwv") is None else float(value["pwv"]),  # type: ignore[arg-type]
        )

    @property
    def label(self) -> str:
        pwv = "" if self.pwv is None else f"_pwv{self.pwv:g}"
        return f"{self.site}_am{self.airmass:g}{pwv}"


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------
@dataclass
class InstrumentManifest:
    """What an instrument directory provides.

    ``components`` maps role -> {band or ``"*"``: path relative to the
    instrument directory}.  ``published`` describes the root-level curves.
    """

    facility: str
    instrument: str
    bands: List[str]
    published: Dict[str, str] = field(default_factory=dict)
    published_bands: Dict[str, str] = field(default_factory=dict)  # band -> recipe of its published curve
    components: Dict[str, Dict[str, str]] = field(default_factory=dict)
    detector_type: str = ENERGY
    ground_based: Optional[bool] = None
    default_site: Optional[str] = None
    sources: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    #: How non-filter components are treated outside their tabulated range:
    #: "error" (refuse), or the source team's own convention: "zero" or
    #: "edge" (hold the edge value).
    outside_range: str = "error"
    schema: int = MANIFEST_SCHEMA

    # ---- persistence -------------------------------------------------
    @classmethod
    def load(cls, instrument_dir: PathLike) -> "InstrumentManifest":
        """Read ``manifest.json``, or describe a legacy directory from its headers."""
        instrument_dir = Path(instrument_dir)
        path = instrument_dir / MANIFEST_NAME
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            known = {f for f in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in data.items() if k in known})
        return cls.from_published(instrument_dir)

    @classmethod
    def from_published(cls, instrument_dir: PathLike) -> "InstrumentManifest":
        """Describe a directory of published curves from their headers.

        Bands whose published curve is the bare filter are registered as
        their own ``filter`` component (no copy is made).
        """
        instrument_dir = Path(instrument_dir)
        bands = sorted(p.stem for p in instrument_dir.glob("*.dat"))
        published_bands: Dict[str, str] = {}
        sources = set()
        detector_types = set()
        filter_components: Dict[str, str] = {}
        for band in bands:
            header = read_filter_header(instrument_dir / f"{band}.dat")
            try:
                roles = parse_components(header.get("Components"))
            except ComponentError:
                roles = ()
            if roles:
                published_bands[band] = recipe_name(roles)
                if roles == (FILTER,):
                    filter_components[band] = f"{band}.dat"
            if header.get("ComponentsSource"):
                sources.add(header["ComponentsSource"])
            dtype = normalize_detector_type(header.get("DetectorType"))
            if dtype:
                detector_types.add(dtype)
        # SVO publishes many filter-only curves as sibling bands of the total
        # (HSC.g_filter, ZTF.g_fil, SDSS.gprime_filter); they are that band's
        # own filter component.
        for band in bands:
            if band in filter_components:
                continue
            for suffix in FILTER_SIBLING_SUFFIXES:
                sibling = f"{band}{suffix}"
                if published_bands.get(sibling) == FILTER:
                    filter_components[band] = f"{sibling}.dat"
                    break
        recipes = set(published_bands.values())
        if len(recipes) == 1 and len(published_bands) == len(bands):
            summary = recipes.pop()
        else:
            summary = "mixed" if published_bands else ""
        published = {"components": summary} if summary else {}
        if len(sources) == 1:
            published["source"] = sources.pop()
        from .facility_sites import is_ground_based, own_site

        return cls(
            facility=instrument_dir.parent.name,
            instrument=instrument_dir.name,
            bands=bands,
            ground_based=is_ground_based(instrument_dir.parent.name),
            default_site=own_site(instrument_dir.parent.name),
            published=published,
            published_bands=published_bands,
            components={FILTER: filter_components} if filter_components else {},
            detector_type=detector_types.pop() if len(detector_types) == 1 else ENERGY,
        )

    def save(self, instrument_dir: PathLike) -> Path:
        path = Path(instrument_dir) / MANIFEST_NAME
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    # ---- queries -----------------------------------------------------
    def component_file(self, role: str, band: str) -> Optional[str]:
        """File for a component key (a role or composite recipe) and band."""
        entries = self.components.get(role, {})
        rel = entries.get(band) or entries.get(SHARED)
        if rel is not None:
            _require_inside(rel, f"{self.facility}/{self.instrument}")
        return rel

    def pieces(self, band: str) -> List[Tuple[Tuple[str, ...], str]]:
        """Component files usable for ``band`` as (roles they contain, relative path).

        Keys may be single roles (``qe``) or composites (``filter+optics+qe``)
        for sources that only publish multiplied curves.
        """
        out = []
        for key in self.components:
            rel = self.component_file(key, band)
            if rel:
                out.append((parse_components(key), rel))
        return out

    def cover(self, band: str, roles: Iterable[str]) -> Optional[List[Tuple[Tuple[str, ...], str]]]:
        """Pieces that together contain exactly ``roles`` (no overlap), or ``None``.

        Larger (composite) pieces are preferred: they are what the source
        team actually measured.
        """
        target = frozenset(roles)
        pieces = sorted(self.pieces(band), key=lambda p: -len(p[0]))

        def search(remaining: frozenset, start: int) -> Optional[List[Tuple[Tuple[str, ...], str]]]:
            if not remaining:
                return []
            for i in range(start, len(pieces)):
                contains = frozenset(pieces[i][0])
                if contains and contains <= remaining:
                    rest = search(remaining - contains, i + 1)
                    if rest is not None:
                        return [pieces[i]] + rest
            return None

        found = search(target, 0)
        if found:
            # The piece containing the filter defines the band; put it first.
            found.sort(key=lambda p: FILTER not in p[0])
        return found

    def primary_bands(self) -> List[str]:
        """Bands that are not merely another band's filter-only sibling."""
        bands = set(self.bands)
        siblings = {
            b for b in self.bands
            if self.published_bands.get(b) == FILTER and any(
                b.endswith(sfx) and b[: -len(sfx)] in bands for sfx in FILTER_SIBLING_SUFFIXES
            )
        }
        return [b for b in self.bands if b not in siblings]

    def buildable(self) -> List[str]:
        """Recipes (without atmosphere) that can be built for every primary band."""
        bands = self.primary_bands()
        out = []
        for extra in ((), (OPTICS,), (QE,), (OPTICS, QE)):
            roles = (FILTER,) + extra
            if bands and all(self.cover(b, roles) for b in bands):
                out.append(recipe_name(roles))
        return out

    def available_roles(self) -> Tuple[str, ...]:
        """Single roles available on their own for every primary band."""
        bands = self.primary_bands()
        return tuple(r for r in (FILTER, OPTICS, QE) if bands and all(self.cover(b, (r,)) for b in bands))

    def bands_with(self, role: str) -> List[str]:
        """Primary bands that have a ``role`` component on its own."""
        return [b for b in self.primary_bands() if self.cover(b, (role,))]

    def describe(self) -> Dict[str, object]:
        return {
            "facility": self.facility,
            "instrument": self.instrument,
            "bands": list(self.bands),
            "published": self.published.get("components") or "unknown",
            "published_source": self.published.get("source", ""),
            "published_bands": dict(self.published_bands) if self.published.get("components") == "mixed" else {},
            "components": list(self.available_roles()),
            "buildable": self.buildable(),
            "partial_components": {
                r: self.bands_with(r) for r in (FILTER, OPTICS, QE)
                if r not in self.available_roles() and self.bands_with(r)
            },
            "atmosphere": bool(self.ground_based and self.default_site),
            "atmosphere_site": self.default_site,
            "platform": _platform_of(self.facility),
            "default_site": self.default_site,
            "detector_type": self.detector_type,
            "component_status": _component_status(self.facility, self.instrument),
        }


@dataclass
class AtmosphereGrid:
    site: str
    entries: List[Dict[str, object]]  # {"airmass": float, "pwv": float|None, "file": str}
    source: str = ""
    altitude_m: Optional[float] = None
    notes: str = ""
    schema: int = MANIFEST_SCHEMA

    @classmethod
    def load(cls, site_dir: PathLike) -> "AtmosphereGrid":
        data = json.loads((Path(site_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, site_dir: PathLike) -> Path:
        path = Path(site_dir) / MANIFEST_NAME
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def select(self, spec: AtmosphereSpec) -> str:
        """Return the file for an exact grid point, or explain what exists."""
        for entry in self.entries:
            if not math_isclose(float(entry["airmass"]), spec.airmass):
                continue
            pwv = entry.get("pwv")
            if spec.pwv is None or (pwv is not None and math_isclose(float(pwv), spec.pwv)):
                return str(entry["file"])
        airmasses = sorted({float(e["airmass"]) for e in self.entries})
        pwvs = sorted({float(e["pwv"]) for e in self.entries if e.get("pwv") is not None})
        raise ComponentError(
            f"No {self.site} atmosphere at airmass={spec.airmass:g}"
            + (f", pwv={spec.pwv:g} mm" if spec.pwv is not None else "")
            + f". Grid: airmass {airmasses}" + (f", pwv {pwvs} mm" if pwvs else "")
        )


def _component_status(facility: str, instrument: str) -> Optional[str]:
    from .component_sources.catalog import builder_for, status_for

    status = status_for(facility, instrument)
    if status:
        return status[1]
    ref = builder_for(facility, instrument)
    if ref and ref.heavy:
        return f"Components available via `sed-tools filters-build-components {ref.name}` (large download)."
    return None


def _require_inside(rel: str, owner: str) -> None:
    """Components must live inside their own instrument directory.

    Each instrument is composed only from its own published factors; a
    manifest can never point at another instrument's files.
    """
    parts = Path(rel).parts
    if not parts or Path(rel).is_absolute() or ".." in parts:
        raise ComponentError(f"{owner}: component path {rel!r} is outside the instrument directory")


def _platform_of(facility: str) -> Optional[str]:
    from .facility_sites import platform

    return platform(facility)


def math_isclose(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------
@dataclass
class Curve:
    wavelength: np.ndarray  # Angstrom, ascending
    transmission: np.ndarray
    header: Dict[str, str]
    path: Optional[Path] = None


def read_curve(path: PathLike) -> Curve:
    """Load a component/filter file in Angstrom (honouring ``WavelengthUnit``)."""
    from .flux_cube_tool import load_filter_curve

    curve = load_filter_curve(str(path))
    return Curve(curve.wavelength, curve.transmission, read_filter_header(path), Path(path))


def _support(curve: Curve, rel_threshold: float = 1e-3) -> Tuple[float, float]:
    """Wavelength range where ``curve`` is non-negligible."""
    peak = float(np.max(curve.transmission)) if curve.transmission.size else 0.0
    if peak <= 0:
        raise ComponentError(f"{curve.path}: transmission is zero everywhere")
    above = np.nonzero(curve.transmission > rel_threshold * peak)[0]
    return float(curve.wavelength[above[0]]), float(curve.wavelength[above[-1]])


def multiply_curves(
    curves: Sequence[Curve],
    *,
    allow_partial: bool = False,
    outside: str = "error",
) -> Tuple[np.ndarray, np.ndarray]:
    """Multiply component curves on a merged grid spanning the first curve.

    The first curve (normally the filter) defines the band.  Every other
    curve must cover the band's non-negligible range; outside its tabulated
    range a component is otherwise unknown, so this raises
    :class:`ComponentCoverageError`, unless the source team's convention says
    otherwise: ``outside="zero"`` (or ``allow_partial``) treats it as zero,
    ``outside="edge"`` holds its edge value.
    """
    if allow_partial and outside == "error":
        outside = "zero"
    if outside not in ("error", "zero", "edge"):
        raise ComponentError(f"Unknown outside-range policy {outside!r}")
    if not curves:
        raise ComponentError("Nothing to compose")
    base = curves[0]
    lo, hi = float(base.wavelength[0]), float(base.wavelength[-1])
    band_lo, band_hi = _support(base)

    for other in curves[1:]:
        o_lo, o_hi = float(other.wavelength[0]), float(other.wavelength[-1])
        if (o_lo > band_lo or o_hi < band_hi) and outside == "error":
            raise ComponentCoverageError(
                f"{other.path.name if other.path else 'component'} covers {o_lo:.0f}-{o_hi:.0f} A "
                f"but the band spans {band_lo:.0f}-{band_hi:.0f} A"
            )

    grid = np.unique(np.concatenate(
        [base.wavelength] + [c.wavelength[(c.wavelength > lo) & (c.wavelength < hi)] for c in curves[1:]]
    ))
    product = np.interp(grid, base.wavelength, base.transmission)
    for other in curves[1:]:
        if outside == "edge":
            factor = np.interp(grid, other.wavelength, other.transmission)
        else:
            factor = np.interp(grid, other.wavelength, other.transmission, left=0.0, right=0.0)
        product = product * factor
    return grid, product


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------
def atmosphere_file(atmosphere_root: PathLike, spec: AtmosphereSpec) -> Path:
    site_dir = Path(atmosphere_root) / spec.site
    if not (site_dir / MANIFEST_NAME).exists():
        sites = sorted(p.name for p in Path(atmosphere_root).glob("*") if (p / MANIFEST_NAME).exists())
        raise ComponentError(f"No atmosphere grid for site '{spec.site}'. Available: {sites or 'none'}")
    return site_dir / AtmosphereGrid.load(site_dir).select(spec)


def resolve_atmosphere(manifest: InstrumentManifest, spec: Optional[AtmosphereSpec]) -> AtmosphereSpec:
    """Pin ``spec`` to the instrument's own site, refusing anything else.

    Products are the instrument's own throughput with or without each
    perturber, so the atmosphere is the one at the instrument's site; only
    the observing conditions (airmass, PWV) are chosen by the user.
    """
    owner = f"{manifest.facility}/{manifest.instrument}"
    if spec is None:
        raise ComponentError("Including the atmosphere needs atmosphere={'airmass': ..., 'pwv': ...}")
    if manifest.ground_based is False:
        from .facility_sites import platform

        raise ComponentError(
            f"{manifest.facility} is {platform(manifest.facility) or 'not ground'}-based; "
            "a telluric component does not apply"
        )
    site = manifest.default_site
    if not site:
        raise ComponentError(f"{owner} has no atmosphere site of its own, so no telluric component is offered")
    if spec.site and spec.site != site:
        raise ComponentError(
            f"{owner} observes from '{site}'; atmospheres from other sites ('{spec.site}') are not mixed in"
        )
    return AtmosphereSpec(site, spec.airmass, spec.pwv)


def product_detector_type(roles: Iterable[str], manifest: InstrumentManifest) -> str:
    """QE converts photons to counts, so anything with QE is photon-counting.

    Without QE the product describes the same detector as the published
    curve, so its detector type is inherited.
    """
    return PHOTON if QE in set(roles) else (normalize_detector_type(manifest.detector_type) or ENERGY)


def _band_detector_type(instrument_dir: Path, band: str, roles: Sequence[str], manifest: InstrumentManifest) -> str:
    if PUBLISHED in roles and QE not in roles:
        # The product is the published curve times extra factors: same detector.
        published = normalize_detector_type(read_filter_header(instrument_dir / f"{band}.dat").get("DetectorType"))
        if published:
            return published
    return product_detector_type(roles, manifest)


def compose_band(
    instrument_dir: PathLike,
    band: str,
    components: Union[str, Iterable[str]],
    *,
    atmosphere: Union[AtmosphereSpec, Mapping[str, object], None] = None,
    atmosphere_root: Optional[PathLike] = None,
    allow_partial: bool = False,
    manifest: Optional[InstrumentManifest] = None,
) -> Curve:
    """Compose one band from its components (see module docstring)."""
    instrument_dir = Path(instrument_dir)
    manifest = manifest or InstrumentManifest.load(instrument_dir)
    roles = set(parse_components(components))
    spec = AtmosphereSpec.coerce(atmosphere)
    if spec is not None:
        roles.add(ATMOSPHERE)

    if INSTRUMENT in roles:
        raise ComponentError("'instrument' describes published curves; request 'optics' and/or 'qe'")
    if PUBLISHED in roles and FILTER in roles:
        raise ComponentError("'published' already contains the filter; combine it with optics/qe/atmosphere only")
    if FILTER not in roles and PUBLISHED not in roles:
        raise ComponentError("A composed band needs the 'filter' component (or the 'published' curve)")
    if band not in manifest.bands:
        raise ComponentError(f"{manifest.facility}/{manifest.instrument} has no band '{band}'")
    if ATMOSPHERE in roles and spec is None:
        raise ComponentError("Including the atmosphere needs atmosphere={'airmass': ..., 'pwv': ...}")
    if ATMOSPHERE in roles:
        spec = resolve_atmosphere(manifest, spec)

    ordered = [r for r in (PUBLISHED,) + COMPOSABLE_ROLES if r in roles]
    instrument_roles = [r for r in ordered if r in (FILTER, OPTICS, QE)]
    covered: List[str] = []
    steps: List[Tuple[str, Path]] = []
    if PUBLISHED in roles:
        if not manifest.published_bands.get(band):
            raise ComponentError(
                f"The published {band} curve does not say what it contains; tag it first "
                "(sed-tools filters-detector) so components are not double counted"
            )
        steps.append((PUBLISHED, instrument_dir / f"{band}.dat"))
    if instrument_roles:
        pieces = manifest.cover(band, instrument_roles)
        if pieces is None:
            can = ", ".join(manifest.buildable()) or "none"
            raise ComponentError(
                f"{manifest.facility}/{manifest.instrument} cannot provide '{recipe_name(instrument_roles)}' "
                f"for band '{band}' (buildable: {can}; published curve: "
                f"{manifest.published_bands.get(band) or manifest.published.get('components') or 'unknown'})"
            )
        steps.extend(("+".join(contains), instrument_dir / rel) for contains, rel in pieces)
    if ATMOSPHERE in roles:
        if atmosphere_root is None:
            raise ComponentError("atmosphere_root is required to include the atmosphere")
        steps.append((ATMOSPHERE, atmosphere_file(atmosphere_root, spec)))  # type: ignore[arg-type]

    curves: List[Curve] = []
    sources: List[str] = []
    for key, path in steps:
        curve = read_curve(path)
        # A file may itself already include other factors; never count one twice.
        if key == PUBLISHED:
            contains = parse_components(manifest.published_bands[band])
        else:
            contains = parse_components(curve.header.get("Components")) or parse_components(key)
        overlap = set(contains) & set(covered)
        if overlap:
            raise ComponentError(f"{path.name} already includes {sorted(overlap)}; refusing to double count")
        covered.extend(contains)
        curves.append(curve)
        sources.append(curve.header.get("FilterID") or (
            str(path.relative_to(path.parents[2])) if len(path.parents) > 2 else path.name))

    wavelength, transmission = multiply_curves(
        curves, allow_partial=allow_partial, outside=manifest.outside_range
    )
    recipe = recipe_name(ordered)
    header = {
        "Components": recipe,
        "ComponentsSource": "composed",
        "Recipe": recipe + (f" @ {spec.label}" if spec else ""),
        "Sources": "; ".join(sources),
    }
    if spec is not None:
        header["Atmosphere"] = f"site={spec.site} airmass={spec.airmass:g}" + (
            f" pwv={spec.pwv:g}" if spec.pwv is not None else "")
    return Curve(wavelength, transmission, header)


def product_name(components: Iterable[str], atmosphere: Optional[AtmosphereSpec] = None) -> str:
    roles = [r for r in parse_components(list(components)) if r != ATMOSPHERE]
    name = recipe_name(roles)
    return f"{name}+atm_{atmosphere.label}" if atmosphere else name


def build_product(
    instrument_dir: PathLike,
    components: Union[str, Iterable[str]],
    *,
    atmosphere: Union[AtmosphereSpec, Mapping[str, object], None] = None,
    atmosphere_root: Optional[PathLike] = None,
    bands: Optional[Sequence[str]] = None,
    allow_partial: bool = False,
    overwrite: bool = True,
) -> Path:
    """Compose every (or the given) band into ``products/<recipe>/``.

    The output directory is a self-contained MESA instrument: ``.dat`` files
    plus an index file named after the directory.
    """
    from .combine_filters import write_filter_index

    instrument_dir = Path(instrument_dir)
    manifest = InstrumentManifest.load(instrument_dir)
    spec = AtmosphereSpec.coerce(atmosphere)
    if spec is not None:
        spec = resolve_atmosphere(manifest, spec)
    roles = parse_components(components)
    name = product_name(roles, spec)
    out_dir = instrument_dir / "products" / name
    if out_dir.exists() and not overwrite:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for band in bands or manifest.primary_bands():
        detector_type = _band_detector_type(instrument_dir, band, roles, manifest)
        curve = compose_band(
            instrument_dir, band, roles,
            atmosphere=spec, atmosphere_root=atmosphere_root,
            allow_partial=allow_partial, manifest=manifest,
        )
        write_filter_file(
            out_dir / f"{band}.dat",
            curve.wavelength,
            curve.transmission,
            detector_type=detector_type,
            source="composed",
            wavelength_unit="Angstrom",
            extra=curve.header,
        )
    write_filter_index(out_dir, name)
    return out_dir


def build_all_products(
    instrument_dir: PathLike,
    *,
    conditions: Sequence[Mapping[str, object]] = (),
    atmosphere_root: Optional[PathLike] = None,
) -> List[Path]:
    """Build every with/without variant the instrument's own components allow.

    That is each buildable recipe (``filter``, ``filter+optics``, ...,
    ``filter+optics+qe``) and, for each observing condition in
    ``conditions`` (``{"airmass": 1.2, "pwv": 2.5}``), the same recipes plus
    the instrument's own atmosphere, and the published curve plus that
    atmosphere when the published curve lacks one.
    """
    instrument_dir = Path(instrument_dir)
    manifest = InstrumentManifest.load(instrument_dir)
    recipes = [r for r in manifest.buildable() if not (r == FILTER and manifest.published.get("components") == FILTER)]
    built: List[Path] = [build_product(instrument_dir, r) for r in recipes]
    published_has_atm = all(
        ATMOSPHERE in parse_components(manifest.published_bands.get(b, "")) for b in manifest.bands
    ) if manifest.published_bands else True
    published_recipes = set(manifest.published_bands.values())
    # The published curve + atmosphere is only a distinct variant when the
    # published curve is not itself one of the buildable recipes.
    add_published = (
        not published_has_atm
        and len(manifest.published_bands) == len(manifest.bands)
        and not (len(published_recipes) == 1 and published_recipes <= set(manifest.buildable()))
    )
    for cond in conditions:
        variants = list(manifest.buildable())
        if add_published:
            variants.append(PUBLISHED)
        for recipe in variants:
            built.append(build_product(instrument_dir, recipe, atmosphere=cond, atmosphere_root=atmosphere_root))
    return built


def add_component(
    instrument_dir: PathLike,
    role: str,
    wavelength_angstrom: Sequence[float],
    transmission: Sequence[float],
    *,
    band: str = SHARED,
    name: Optional[str] = None,
    source: str = "",
    manifest: Optional[InstrumentManifest] = None,
) -> str:
    """Write a component curve and register it in the manifest (not saved).

    ``role`` is ``filter``, ``optics`` or ``qe``, or a composite such as
    ``filter+optics+qe`` when the source only publishes the product.
    """
    roles = parse_components(role)
    if not roles or not set(roles) <= {FILTER, OPTICS, QE}:
        raise ComponentError(f"Instrument components are filter/optics/qe (or composites), not {role!r}")
    role = recipe_name(roles)
    peak = float(np.nanmax(np.asarray(transmission, dtype=float)))
    if not np.isfinite(peak) or peak > 1.5:
        raise ComponentError(f"{role} component peaks at {peak:g}; expected a fraction (is it in percent?)")
    instrument_dir = Path(instrument_dir)
    stem = name or (band if band != SHARED else role)
    rel = f"components/{role}/{stem}.dat"
    path = instrument_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    write_filter_file(
        path, wavelength_angstrom, transmission,
        detector_type=PHOTON if QE in roles else ENERGY,
        source="component",
        wavelength_unit="Angstrom",
        extra={"Role": role, "Components": role, "Sources": source or None},
    )
    if manifest is not None:
        manifest.components.setdefault(role, {})[band] = rel
        if source:
            manifest.sources.setdefault(role, source)
    return rel
