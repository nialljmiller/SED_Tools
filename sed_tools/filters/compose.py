"""Composing filter curves: an instrument's own components multiplied into products.

A published passband is a product of throughputs (Gaia DR3: filter + CCD + instrument) that cannot be divided
back, so components exist only where the team published them; the folder model is :mod:`.manifest`, re-exported here.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .curve_metadata import (ENERGY, PHOTON, normalize_detector_type,
                              read_filter_header, write_filter_file)
from .manifest import (  # noqa: F401 - every manifest name stays importable from here (filter_components)
    ATMOSPHERE, ATMOSPHERE_KINDS, AUXILIARY_DIRS, COMPONENTS_DIR, COMPOSABLE_ROLES, DERIVED, FILTER,
    FILTER_SIBLING_SUFFIXES, INSTRUMENT, MANIFEST_NAME, MANIFEST_SCHEMA, OPTICS, PRODUCTS_DIR, PUBLISHED, QE,
    ROLES, SHARED, TEAM, _RECIPES, _SVO_TOKENS, _VARIANT_NAME, AtmosphereGrid, AtmosphereSpec, ComponentError,
    InstrumentManifest, PathLike, _require_inside, archived_bands, index_listing, index_names, is_auxiliary_path,
    math_isclose, parse_components, recipe_name, write_filter_index,
)
from .sites import instrument_sites, is_ground_based, own_site, platform  # noqa: F401

logger = logging.getLogger("sed_tools.filter_components")


class ComponentCoverageError(ComponentError):
    """A component does not cover the wavelength range of the band."""


# ---- Curves
@dataclass
class Curve:
    wavelength: np.ndarray  # Angstrom, ascending
    transmission: np.ndarray
    header: Dict[str, str]
    path: Optional[Path] = None


def read_curve(path: PathLike) -> Curve:
    """Load a component/filter file in Angstrom (honouring ``WavelengthUnit``)."""
    from ..photometry import load_filter_curve

    if not Path(path).is_file():
        raise ComponentError(f"Missing throughput file: {path}; fetch this instrument's components again")
    curve = load_filter_curve(str(path))
    if (curve.wavelength.size < 2 or not np.all(np.isfinite(curve.wavelength))
            or np.any(curve.wavelength <= 0) or np.any(np.diff(curve.wavelength) <= 0)
            or not np.all(np.isfinite(curve.transmission)) or np.any(curve.transmission < 0)):
        raise ComponentError(f"{path}: throughput needs increasing positive wavelengths and finite nonnegative values")
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
    policies: Optional[Sequence[Optional[str]]] = None,
    tolerances: Optional[Sequence[Optional[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multiply component curves on a merged grid spanning the first curve.

    The first curve (normally the filter) defines the band.  Every other
    curve must cover the band's non-negligible range; outside its tabulated
    range a component is otherwise unknown, so this raises
    :class:`ComponentCoverageError`, unless the source team's convention says
    otherwise: ``outside="zero"`` (or ``allow_partial``) treats it as zero,
    ``outside="edge"`` holds its edge value.  ``policies`` may override
    ``outside`` per curve (``None`` entries keep it), e.g. for an atmosphere
    whose own manifest sets ``outside_range``.  ``tolerances`` (per curve)
    relax an ``"error"`` rule: the curve may miss at most that fraction of
    the throughput accumulated so far, and holds its edge values there.
    """
    if allow_partial and outside == "error":
        outside = "zero"
    if not curves:
        raise ComponentError("Nothing to compose")
    rules = [outside if p is None else p for p in (policies or [None] * len(curves))]
    if len(rules) != len(curves):
        raise ComponentError("one outside-range policy per curve")
    if allow_partial:
        rules = ["zero" if r == "error" else r for r in rules]
    for rule in [outside, *rules]:
        if rule not in ("error", "zero", "edge"):
            raise ComponentError(f"Unknown outside-range policy {rule!r}")
    tols = list(tolerances or [None] * len(curves))
    if len(tols) != len(curves):
        raise ComponentError("one coverage tolerance per curve")
    base = curves[0]
    lo, hi = float(base.wavelength[0]), float(base.wavelength[-1])
    band_lo, band_hi = _support(base)

    for other, rule, tol in zip(curves[1:], rules[1:], tols[1:]):
        o_lo, o_hi = float(other.wavelength[0]), float(other.wavelength[-1])
        if (o_lo > band_lo or o_hi < band_hi) and rule == "error" and tol is None:
            raise ComponentCoverageError(
                f"{other.path.name if other.path else 'component'} covers {o_lo:.0f}-{o_hi:.0f} A "
                f"but the band spans {band_lo:.0f}-{band_hi:.0f} A"
            )

    grid = np.unique(np.concatenate(
        [base.wavelength] + [c.wavelength[(c.wavelength > lo) & (c.wavelength < hi)] for c in curves[1:]]
    ))
    product = np.interp(grid, base.wavelength, base.transmission)
    for other, rule, tol in zip(curves[1:], rules[1:], tols[1:]):
        o_lo, o_hi = float(other.wavelength[0]), float(other.wavelength[-1])
        if rule == "error" and tol is not None:
            total = float(np.trapezoid(product, grid))
            beyond = float(np.trapezoid(np.where((grid < o_lo) | (grid > o_hi), product, 0.0), grid))
            if total <= 0 or beyond > tol * total:
                raise ComponentCoverageError(
                    f"{other.path.name if other.path else 'component'} covers {o_lo:.0f}-{o_hi:.0f} A but "
                    f"{beyond / total if total > 0 else 1:.2%} of the band lies beyond it (tolerance {tol:g})"
                )
            rule = "edge"
        if rule == "edge":
            factor = np.interp(grid, other.wavelength, other.transmission)
        else:
            factor = np.interp(grid, other.wavelength, other.transmission, left=0.0, right=0.0)
        product = product * factor
    return grid, product


# ---- Composition
def atmosphere_grid(atmosphere_root: PathLike, site: str) -> Optional[AtmosphereGrid]:
    """The cached manifest of ``site``, or None."""
    site_dir = Path(atmosphere_root) / site
    if not (site_dir / MANIFEST_NAME).exists():
        return None
    grid = AtmosphereGrid.load(site_dir)
    if grid.site != site:
        raise ComponentError(f"Atmosphere manifest site {grid.site!r} does not match {site!r}")
    return grid


def find_atmosphere(atmosphere_root: PathLike, spec: AtmosphereSpec) -> Optional[Path]:
    """Find a cached grid point without treating a cache miss as an error."""
    site_dir = Path(atmosphere_root) / spec.site
    grid = atmosphere_grid(atmosphere_root, spec.site)
    if grid is None:
        return None
    filename = grid.find(spec)
    if filename is None:
        return None
    path = site_dir / filename
    if not path.resolve().is_relative_to(site_dir.resolve()):
        raise ComponentError("Atmosphere file resolves outside its site directory")
    return path if path.is_file() else None


def atmosphere_file(atmosphere_root: PathLike, spec: AtmosphereSpec) -> Path:
    """The cached file for ``spec``; ComponentError saying what the site does have otherwise."""
    path = find_atmosphere(atmosphere_root, spec)
    if path is not None:
        return path
    grid = atmosphere_grid(atmosphere_root, spec.site)
    if grid is None:
        raise ComponentError(f"No atmosphere of site {spec.site} is cached; fetch or generate the site first")
    if grid.find(spec) is None:
        raise ComponentError(f"Site {spec.site} has no atmosphere for {_conditions(spec)}; "
                             f"it has {_available_conditions(grid)}")
    raise ComponentError(f"The {spec.site} atmosphere for {_conditions(spec)} is listed but its file "
                         "is not cached; fetch it first")


def _conditions(spec: AtmosphereSpec) -> str:
    """``airmass 1.2``, ``airmass 1.2 and PWV 2.5 mm`` or ``unstated airmass``, for messages."""
    if spec.airmass is None:
        return "unstated airmass"
    return f"airmass {spec.airmass:g}" + (f" and PWV {spec.pwv:g} mm" if spec.pwv is not None else "")


def _available_conditions(grid: AtmosphereGrid) -> str:
    """The airmasses and PWVs a site's manifest lists, and whether it has an unstated-airmass atmosphere."""
    parts = []
    airmasses = sorted({float(e["airmass"]) for e in grid.entries})
    if airmasses:
        listed = "airmass " + ", ".join(f"{a:g}" for a in airmasses)
        pwvs = sorted({float(e["pwv"]) for e in grid.entries if e.get("pwv") is not None})
        if pwvs:
            listed += " with PWV " + ", ".join(f"{p:g}" for p in pwvs) + " mm"
        parts.append(listed)
    if grid.unstated:
        parts.append("unstated airmass (the atmosphere as published)")
    return "; or ".join(parts) or "no atmospheres"


_NEEDS_CONDITIONS = ("Including the atmosphere needs the observing conditions: an airmass, "
                     "and a PWV where the site models it")


def resolve_atmosphere(manifest: InstrumentManifest, spec: Optional[AtmosphereSpec]) -> AtmosphereSpec:
    """Pin ``spec`` to one of the instrument's own sites, refusing anything else.

    Products are the instrument's own throughput with or without each
    perturber, so the atmosphere is one at the instrument's location: its
    ``default_site`` unless another of its own sites (a team's standard
    atmosphere, say) is named.  Only the observing conditions (airmass, PWV)
    are otherwise chosen by the user.
    """
    owner = f"{manifest.facility}/{manifest.instrument}"
    if spec is None:
        raise ComponentError(_NEEDS_CONDITIONS)
    if manifest.ground_based is False:
        raise ComponentError(
            f"{manifest.facility} is {platform(manifest.facility) or 'not ground'}-based; "
            "a telluric component does not apply"
        )
    sites = manifest.sites()
    if not sites:
        raise ComponentError(f"{owner} has no atmosphere site of its own, so no telluric component is offered")
    if spec.site and spec.site not in sites:
        raise ComponentError(
            f"{owner} observes from {', '.join(sites)}; atmospheres from other sites ({spec.site}) are not mixed in"
        )
    site = spec.site or manifest.default_site or (sites[0] if len(sites) == 1 else None)
    if site is None:
        raise ComponentError(f"{owner} has several atmosphere sites; choose a site: {', '.join(sites)}")
    return AtmosphereSpec(site, spec.airmass, spec.pwv)


def product_detector_type(roles: Iterable[str], manifest: InstrumentManifest) -> str:
    """QE converts photons to counts, so anything with QE is photon-counting.

    Without QE the product describes the same detector as the published
    curve, so its detector type is inherited.
    """
    return PHOTON if QE in set(roles) else (normalize_detector_type(manifest.detector_type) or ENERGY)


def _band_detector_type(instrument_dir: Path, band: str, roles: Sequence[str], manifest: InstrumentManifest) -> str:
    if manifest.detector_type_sources.get(band) == "user":
        # A convention stated for the instrument (e.g. by its team, as for
        # Argus) holds for every product of the band, QE or not.
        return manifest.detector_types[band]
    if PUBLISHED in roles and QE not in roles:
        # The product is the published curve times extra factors: same detector.
        published = normalize_detector_type(read_filter_header(instrument_dir / f"{band}.dat").get("DetectorType"))
        if published:
            return published
    if QE not in roles:
        return manifest.detector_types.get(band) or product_detector_type(roles, manifest)
    return product_detector_type(roles, manifest)


def _check_composition(roles: set, band: str, manifest: InstrumentManifest,
                       spec: Optional[AtmosphereSpec]) -> None:
    """Refuse requests that do not describe one band's own throughput."""
    if INSTRUMENT in roles:
        raise ComponentError("'instrument' describes published curves; request 'optics' and/or 'qe'")
    if PUBLISHED in roles and FILTER in roles:
        raise ComponentError("'published' already contains the filter; combine it with optics/qe/atmosphere only")
    if FILTER not in roles and PUBLISHED not in roles:
        raise ComponentError("A composed band needs the 'filter' component (or the 'published' curve)")
    if band not in manifest.bands:
        raise ComponentError(f"{manifest.facility}/{manifest.instrument} has no band '{band}'")
    if ATMOSPHERE in roles and spec is None:
        raise ComponentError(_NEEDS_CONDITIONS)


def _composition_steps(instrument_dir: Path, band: str, roles: set, manifest: InstrumentManifest,
                       spec: Optional[AtmosphereSpec], atmosphere_root: Optional[PathLike]) -> List[Tuple[str, Path]]:
    """The files to multiply for ``band`` as (component key, path), the one holding the filter first."""
    instrument_roles = [r for r in (FILTER, OPTICS, QE) if r in roles]
    steps: List[Tuple[str, Path]] = []
    if PUBLISHED in roles:
        if not manifest.published_bands.get(band) and roles != {PUBLISHED}:
            raise ComponentError(
                f"The published {band} curve does not say what it contains; tag it first "
                "(sed-tools filters-detector) so components are not double counted"
            )
        steps.append((PUBLISHED, instrument_dir / f"{band}.dat"))
        if (INSTRUMENT in parse_components(manifest.published_bands.get(band))
                and roles & {OPTICS, QE}):
            raise ComponentError(f"{band}: published 'instrument' contribution is unresolved; adding optics/QE risks double counting")
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
    return steps


def compose_band(
    instrument_dir: PathLike,
    band: str,
    components: Union[str, Iterable[str]],
    *,
    atmosphere: Union[AtmosphereSpec, Mapping[str, object], None] = None,
    atmosphere_root: Optional[PathLike] = None,
    allow_partial: bool = False,
    manifest: Optional[InstrumentManifest] = None,
    variants: Optional[Mapping[str, str]] = None,
) -> Curve:
    """Compose one band from its components (see module docstring).

    ``variants`` selects alternative curves for component keys, e.g.
    ``{"qe": "single_layer"}`` (see :meth:`InstrumentManifest.with_variants`).
    """
    instrument_dir = Path(instrument_dir)
    manifest = (manifest or InstrumentManifest.load(instrument_dir)).with_variants(variants)
    roles = set(parse_components(components))
    spec = AtmosphereSpec.coerce(atmosphere)
    if spec is not None:
        roles.add(ATMOSPHERE)
    _check_composition(roles, band, manifest, spec)
    if ATMOSPHERE in roles:
        spec = resolve_atmosphere(manifest, spec)
    ordered = [r for r in (PUBLISHED,) + COMPOSABLE_ROLES if r in roles]
    steps = _composition_steps(instrument_dir, band, roles, manifest, spec, atmosphere_root)

    covered: List[str] = []
    curves: List[Curve] = []
    sources: List[str] = []
    policies: List[Optional[str]] = []
    tolerances: List[Optional[float]] = []
    for key, path in steps:
        policies.append(None)
        tolerances.append(None)
        if key == ATMOSPHERE:
            grid = atmosphere_grid(atmosphere_root, spec.site)  # type: ignore[arg-type,union-attr]
            if grid is not None:
                policies[-1], tolerances[-1] = grid.outside_range, grid.coverage_tolerance
        elif not path.resolve().is_relative_to(instrument_dir.resolve()):
            raise ComponentError(f"{path}: component resolves outside its instrument directory")
        curve = read_curve(path)
        # A file may itself already include other factors; never count one twice.
        if key == PUBLISHED:
            contains = parse_components(manifest.published_bands.get(band))
        else:
            contains = parse_components(curve.header.get("Components")) or parse_components(key)
            if set(contains) != set(parse_components(key)):
                raise ComponentError(f"{path.name}: component contents {contains} disagree with its manifest role {key!r}")
        overlap = set(contains) & set(covered)
        if overlap:
            raise ComponentError(f"{path.name} already includes {sorted(overlap)}; refusing to double count")
        covered.extend(contains)
        curves.append(curve)
        sources.append(curve.header.get("FilterID") or (
            str(path.relative_to(path.parents[2])) if len(path.parents) > 2 else path.name))

    try:
        wavelength, transmission = multiply_curves(
            curves, allow_partial=allow_partial, outside=manifest.outside_range, policies=policies,
            tolerances=tolerances,
        )
    except ComponentCoverageError as exc:
        raise ComponentCoverageError(f"{manifest.facility}/{manifest.instrument} band {band}: {exc}") from exc
    recipe = recipe_name(ordered)
    header = {
        "Components": recipe_name(covered),
        "ComponentsSource": "composed",
        "Recipe": recipe + "".join(f" [{k}={v}]" for k, v in sorted((variants or {}).items()))
                  + (f" @ {spec.label}" if spec else ""),
        "Sources": "; ".join(sources),
    }
    if spec is not None:
        header["Atmosphere"] = spec.describe()
    return Curve(wavelength, transmission, header)


def product_name(components: Iterable[str], atmosphere: Optional[AtmosphereSpec] = None,
                 variants: Optional[Mapping[str, str]] = None) -> str:
    """Directory name of a product; selected variants are part of it.

    ``filter+qe`` with ``{"qe": "single_layer"}`` is ``filter+qe+qe_single_layer``,
    so products of different variants never replace each other.
    """
    roles = [r for r in parse_components(list(components)) if r != ATMOSPHERE]
    name = recipe_name(roles)
    for key, value in sorted((variants or {}).items()):
        if not _VARIANT_NAME.fullmatch(str(value)):
            raise ComponentError(f"Invalid {key} variant name: {value!r}")
        name += f"+{recipe_name(parse_components(key))}_{value}"
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
    detector_type: Optional[str] = None,
    variants: Optional[Mapping[str, str]] = None,
) -> Path:
    """Compose every (or the given) band into ``products/<recipe>/``.

    The output directory is a self-contained MESA instrument: ``.dat`` files
    plus an index file named after the directory.  ``variants`` selects
    alternative component curves (:meth:`InstrumentManifest.with_variants`);
    the choice is part of the product name.
    """
    instrument_dir = Path(instrument_dir)
    manifest = InstrumentManifest.load(instrument_dir)
    spec = AtmosphereSpec.coerce(atmosphere)
    if spec is not None:
        spec = resolve_atmosphere(manifest, spec)
    roles = parse_components(components)
    override = normalize_detector_type(detector_type)
    if detector_type is not None and override is None:
        raise ComponentError(f"Unknown detector type: {detector_type!r}; choose energy or photon")
    name = product_name(roles, spec, variants) + (f"+detector_{override}" if override else "")
    out_dir = instrument_dir / "products" / name
    if out_dir.exists() and not overwrite:
        return out_dir
    manifest.with_variants(variants)  # refuse unknown variants before writing anything
    selected = list(manifest.primary_bands() if bands is None else bands)
    if not selected:
        raise ComponentError("No bands selected for composition")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # Finish every band before replacing an existing product. A failed band
    # must not leave a partial product or overwrite a previously valid one.
    with tempfile.TemporaryDirectory(prefix=".compose-", dir=out_dir.parent) as tmp:
        staging = Path(tmp) / "ready"
        staging.mkdir()
        for band in selected:
            curve = compose_band(
                instrument_dir, band, roles,
                atmosphere=spec, atmosphere_root=atmosphere_root,
                allow_partial=allow_partial, manifest=manifest, variants=variants,
            )
            original = _band_detector_type(instrument_dir, band, roles, manifest)
            if override:
                curve.header["DetectorTypeOriginal"] = original
            write_filter_file(
                staging / f"{band}.dat",
                curve.wavelength,
                curve.transmission,
                detector_type=override or original,
                source="user" if override else "composed",
                wavelength_unit="Angstrom",
                extra=curve.header,
            )
        write_filter_index(staging, name)
        previous = Path(tmp) / "previous"
        if out_dir.exists():
            out_dir.rename(previous)
        try:
            staging.rename(out_dir)
        except OSError:
            if previous.exists():
                previous.rename(out_dir)
            raise
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

    A recipe with optics or QE that only some bands have is not built; a
    warning names it and the :func:`build_product` call that composes it.
    """
    instrument_dir = Path(instrument_dir)
    manifest = InstrumentManifest.load(instrument_dir)
    _warn_partial_recipes(instrument_dir, manifest)
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


def _warn_partial_recipes(instrument_dir: Path, manifest: InstrumentManifest) -> None:
    """Warn about recipes with optics or QE that only some primary bands have (SLOAN/SDSS's primed filters).

    build_all_products builds a recipe only for every primary band, so without this such a product silently is not made.
    """
    bands = manifest.primary_bands()
    recipes_by_bands: Dict[Tuple[str, ...], List[str]] = {}
    for roles in _RECIPES[1:]:
        covered = tuple(b for b in bands if manifest.cover(b, roles))
        if covered and len(covered) < len(bands):
            recipes_by_bands.setdefault(covered, []).append(recipe_name(roles))
    for covered, recipes in recipes_by_bands.items():
        missing = [b for b in bands if b not in covered]
        logger.warning(
            "%s/%s: %s not built, because only %d of %d bands have the components (none for %s). "
            "Build for those bands with build_product(%r, %r, bands=%r)",
            manifest.facility, manifest.instrument, ", ".join(recipes), len(covered), len(bands),
            ", ".join(missing), str(instrument_dir), recipes[-1], list(covered),
        )


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
    variant: Optional[str] = None,
    variant_info: Optional[Mapping[str, object]] = None,
) -> str:
    """Write a component curve and register it in the manifest (not saved).

    ``role`` is ``filter``, ``optics`` or ``qe``, or a composite such as
    ``filter+optics+qe`` when the source only publishes the product.

    With ``variant`` the curve is registered as that named alternative of
    ``role`` (``manifest.variants``) instead of as the default component;
    ``variant_info`` adds its description, epoch, etc.  A band-specific
    curve's source is recorded under ``"<role>:<band>"`` so each band
    cites its own file.
    """
    if variant is not None and not _VARIANT_NAME.fullmatch(variant):
        raise ComponentError(f"Invalid variant name: {variant!r}")
    roles = parse_components(role)
    if not roles or not set(roles) <= {FILTER, OPTICS, QE}:
        hint = ("; an atmosphere is a site of its own: register it with "
                "component_sources._common.SiteAtmosphere (see write_site)" if ATMOSPHERE in roles else "")
        raise ComponentError(f"Instrument components are filter/optics/qe (or composites), not {role!r}{hint}")
    role = recipe_name(roles)
    peak = float(np.nanmax(np.asarray(transmission, dtype=float)))
    if not np.isfinite(peak) or peak > 1.5:
        raise ComponentError(f"{role} component peaks at {peak:g}; expected a fraction (is it in percent?)")
    instrument_dir = Path(instrument_dir)
    # A peak between 1 and 1.5 is kept as published, not clipped: clipping would change the
    # shape.  Builders record why in the manifest notes (e.g. DECam's quotients).
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
    if manifest is not None and variant is not None:
        entry = manifest.variants.setdefault(role, {}).setdefault(variant, {"files": {}})
        entry.setdefault("files", {})[band] = rel
        if source:
            entry.setdefault("sources", {})[band] = source
        entry.update(dict(variant_info or {}))
    elif manifest is not None:
        manifest.components.setdefault(role, {})[band] = rel
        if source:
            if band == SHARED:
                manifest.sources[role] = source
            else:
                manifest.sources[f"{role}:{band}"] = source
    return rel
