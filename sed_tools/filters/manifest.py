"""One instrument folder as it is on disk: its manifest, the component vocabulary and the MESA index file.

Nothing here downloads or multiplies curves.  :mod:`.compose` multiplies components into products
(and re-exports every name here); :mod:`.library` decides where a set's curves come from.
"""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .curve_metadata import ENERGY, normalize_detector_type, read_filter_header
from .sites import instrument_sites, is_ground_based, own_site, platform

logger = logging.getLogger("sed_tools.filter_components")

PathLike = Union[str, os.PathLike]

# Component roles: "filter" is the bandpass filter's transmission, "optics" the telescope and instrument
# optics (mirrors, lenses, ...), "qe" the detector's quantum efficiency, "atmosphere" the telluric
# transmission (site, airmass and PWV dependent), and "instrument" an unspecified mix of optics and
# detector, as some sources publish it: it only describes published curves and is never composed.
FILTER, OPTICS, QE, ATMOSPHERE, INSTRUMENT = "filter", "optics", "qe", "atmosphere", "instrument"
#: Pseudo-role selecting the published root-level curve as the base of a product.
PUBLISHED = "published"
#: Roles in canonical order; recipes are always written in this order.
ROLES = (FILTER, OPTICS, QE, INSTRUMENT, ATMOSPHERE)
#: Roles that can be selected when composing a product.
COMPOSABLE_ROLES = (FILTER, OPTICS, QE, ATMOSPHERE)
# The instrument's own recipes, without atmosphere, in the order they are offered.
_RECIPES = ((FILTER,), (FILTER, OPTICS), (FILTER, QE), (FILTER, OPTICS, QE))

#: Sub-directories of an instrument that hold components and composed
#: products rather than published curves.
COMPONENTS_DIR, PRODUCTS_DIR = "components", "products"
AUXILIARY_DIRS = (COMPONENTS_DIR, PRODUCTS_DIR)


def is_auxiliary_path(path: PathLike, root: Optional[PathLike] = None, dirs: Sequence[str] = AUXILIARY_DIRS) -> bool:
    """True if ``path`` lies inside a ``components/`` or ``products/`` directory.

    Scanners of *published* curves skip these, so a composed ``products/filter/G.dat`` never shadows ``G.dat``.
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
#: SVO's SDSS ``uprime_filter`` ... ``zprime_filter`` are not siblings of
#: ``u`` ... ``z``: they are the USNO 40-in filters of the primed system,
#: kept in air, whereas the 2.5 m camera's filters sit in vacuum, which
#: moves their red edges about 2.5 per cent blueward (SDSS DR7 "Photometric
#: flux calibration", classic.sdss.org/dr7/algorithms/fluxcal.html).
FILTER_SIBLING_SUFFIXES = ("_filter", "_fil")
#: Variant names: letters, digits, ``_`` and ``-`` (they appear in product names).
_VARIANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
MANIFEST_SCHEMA = 1
SHARED = "*"  # component key meaning "applies to every band"

#: Atmosphere site kinds: published by an instrument team, or computed by
#: SED_Tools from a model (e.g. ESO SkyCalc).  Atmosphere ``.dat`` headers
#: carry the same word as ``ComponentsSource`` (``original`` for team data
#: stored as published, ``derived`` for anything SED_Tools computed).
TEAM, DERIVED = "team", "derived"
ATMOSPHERE_KINDS = (TEAM, DERIVED)

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


# ---- Component vocabulary
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


# ---- Atmosphere selection
@dataclass(frozen=True)
class AtmosphereSpec:
    """Observing conditions for the telluric component.

    ``site`` is one of the instrument's own sites (``manifest.default_site``
    when omitted); any other site is refused.  Atmospheres are never
    borrowed from another location.

    ``airmass=None`` selects a site's atmosphere *as published* when the
    source does not state its airmass (e.g. the 2MASS PLEXUS models); it
    must be passed explicitly, ``{"site": ..., "airmass": None}``.
    """

    site: Optional[str]
    airmass: Optional[float]
    pwv: Optional[float] = None  # precipitable water vapour, mm

    def __post_init__(self):
        if self.airmass is None:
            if self.pwv is not None:
                raise ComponentError("an atmosphere of unstated airmass has no pwv to choose")
        elif not np.isfinite(self.airmass) or self.airmass < 1:
            raise ComponentError("airmass must be finite and at least 1")
        if self.pwv is not None and (not np.isfinite(self.pwv) or self.pwv < 0):
            raise ComponentError("pwv must be finite and nonnegative")
        if self.site is not None:
            _require_inside(self.site, "atmosphere site")
            if Path(self.site).name != self.site:
                raise ComponentError("atmosphere site must be a single directory name")

    @classmethod
    def coerce(cls, value: Union["AtmosphereSpec", Mapping[str, object], None]) -> Optional["AtmosphereSpec"]:
        if value is None or isinstance(value, AtmosphereSpec):
            return value
        if "airmass" not in value:
            raise ComponentError("atmosphere needs at least 'airmass' (and 'pwv' for SkyCalc sites)")
        return cls(
            site=None if value.get("site") is None else str(value["site"]),
            airmass=None if value["airmass"] is None else float(value["airmass"]),  # type: ignore[arg-type]
            pwv=None if value.get("pwv") is None else float(value["pwv"]),  # type: ignore[arg-type]
        )

    @property
    def label(self) -> str:
        pwv = "" if self.pwv is None else f"_pwv{self.pwv:g}"
        airmass = "unstated" if self.airmass is None else f"{self.airmass:g}"
        return f"{self.site}_am{airmass}{pwv}"

    def describe(self) -> str:
        """``site=... airmass=... [pwv=...]`` for curve headers."""
        airmass = "unstated" if self.airmass is None else f"{self.airmass:g}"
        return f"site={self.site} airmass={airmass}" + (f" pwv={self.pwv:g}" if self.pwv is not None else "")


# ---- Manifests
def index_names(text: str) -> List[str]:
    """The names a MESA index file lists: its lines stripped, without blank and ``#`` lines."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def index_listing(instrument_dir: PathLike) -> Optional[List[str]]:
    """The curve files the instrument's MESA index file lists, or None without one (or an empty one)."""
    index = Path(instrument_dir) / Path(instrument_dir).name
    if not index.is_file():
        return None
    return index_names(index.read_text(encoding="utf-8")) or None


def archived_bands(instrument_dir: PathLike) -> Dict[str, Dict[str, object]]:
    """The ``archived_bands`` of the directory's manifest (a list of names is read as names), else {}."""
    path = Path(instrument_dir) / MANIFEST_NAME
    if not path.is_file():
        return {}
    archived = json.loads(path.read_text(encoding="utf-8")).get("archived_bands") or {}
    return {band: {} for band in archived} if isinstance(archived, list) else dict(archived)


def write_filter_index(filter_dir: Path, instrument: str) -> Path:
    """Write the MESA index file for all ``*.dat`` files in ``filter_dir``.

    Bands its ``manifest.json`` lists under ``archived_bands`` (superseded
    copies kept beside the set) are left out: MESA must not load them.
    """
    archived = archived_bands(filter_dir)
    dat_files = sorted(child.name for child in filter_dir.glob("*.dat")
                       if child.is_file() and child.stem not in archived)
    index_path = filter_dir / instrument
    index_path.write_text("\n".join(dat_files) + "\n", encoding="utf-8")
    return index_path


@dataclass
class InstrumentManifest:
    """What an instrument directory provides.

    ``components`` maps role -> {band or ``"*"``: path relative to the
    instrument directory}.  ``published`` describes the root-level curves.
    The layout (``manifest.json``, ``components/`` and ``products/`` are
    additions; the published curves and the index MESA reads are untouched)::

        filters/<Facility>/<Instrument>/
            <Instrument>, <band>.dat        published curves
            manifest.json                   what is available, with provenance
            components/<role>/<file>.dat    per-band or shared component curves
            products/<recipe>/              composed curves; a valid MESA instrument
                <recipe>, <band>.dat

        atmospheres/<site>/
            manifest.json                   airmass/PWV grid with its provenance
            <file>.dat
    """

    facility: str
    instrument: str
    bands: List[str]
    published: Dict[str, str] = field(default_factory=dict)
    published_bands: Dict[str, str] = field(default_factory=dict)  # band -> recipe of its published curve
    components: Dict[str, Dict[str, str]] = field(default_factory=dict)
    detector_type: str = ENERGY
    detector_types: Dict[str, str] = field(default_factory=dict)
    detector_type_sources: Dict[str, str] = field(default_factory=dict)
    ground_based: Optional[bool] = None
    default_site: Optional[str] = None
    #: Further atmosphere sites at the instrument's own location, e.g. a
    #: team's standard atmosphere next to a derived SkyCalc grid.  Saved
    #: only when set, so existing manifests are written unchanged.
    atmosphere_sites: List[str] = field(default_factory=list)
    sources: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    #: How non-filter components are treated outside their tabulated range:
    #: "error" (refuse), or the source team's own convention: "zero" or
    #: "edge" (hold the edge value).
    outside_range: str = "error"
    schema: int = MANIFEST_SCHEMA
    #: Alternative curves for a component key (a role or composite recipe)
    #: that are used only when selected, e.g. QE per detector coating or
    #: epoch: key -> variant name -> {"files": {band or "*": path},
    #: "description": ..., "source": ..., optional "epoch", "derived",
    #: "default"}.  The curve in ``components`` stays the default; a
    #: variant never replaces it unless asked for.  Saved only when set,
    #: so existing manifests are written unchanged.
    variants: Dict[str, Dict[str, Dict[str, object]]] = field(default_factory=dict)
    #: Superseded copies kept beside the set but not part of it (e.g. SVO's
    #: ``H_svo`` beside ESA's ``H``): band -> {"detector_type",
    #: "detector_type_source", "published"}.  Never in ``bands`` or the MESA
    #: index file.  Saved only when set, so existing manifests are written unchanged.
    archived_bands: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def __post_init__(self):
        for band in self.bands:
            if Path(band).name != band or band in (".", ".."):
                raise ComponentError(f"Invalid band name in manifest: {band!r}")
        for entries in self.components.values():
            for rel in entries.values():
                _require_inside(rel, f"{self.facility}/{self.instrument}")
        for key, named in self.variants.items():
            for name, entry in named.items():
                if not _VARIANT_NAME.fullmatch(name):
                    raise ComponentError(f"Invalid {key} variant name: {name!r}")
                for rel in dict(entry.get("files") or {}).values():
                    _require_inside(str(rel), f"{self.facility}/{self.instrument} {key} variant {name}")

    # ---- persistence -------------------------------------------------
    @classmethod
    def load(cls, instrument_dir: PathLike) -> "InstrumentManifest":
        """Read ``manifest.json``, or describe a legacy directory from its headers."""
        instrument_dir = Path(instrument_dir)
        path = instrument_dir / MANIFEST_NAME
        if path.exists():
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return cls.from_published(instrument_dir)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "InstrumentManifest":
        """Read supported manifest fields, allowing metadata from newer versions."""
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})

    @classmethod
    def from_published(cls, instrument_dir: PathLike) -> "InstrumentManifest":
        """Describe a directory of published curves from their headers.

        Bands whose published curve is the bare filter are registered as
        their own ``filter`` component (no copy is made).  When the directory
        has a MESA index file, only the curves it lists are bands; bands an
        existing manifest lists under ``archived_bands`` never are, and that
        record is kept.
        """
        instrument_dir = Path(instrument_dir)
        archived = archived_bands(instrument_dir)
        listed = index_listing(instrument_dir)
        curves = [p for p in sorted(instrument_dir.glob("*.dat"))
                  if p.stem not in archived and (listed is None or p.name in listed)]
        manifest = cls.from_headers(instrument_dir.parent.name, instrument_dir.name,
                                    {p.stem: read_filter_header(p) for p in curves})
        manifest.archived_bands = archived
        return manifest

    @classmethod
    def from_headers(cls, facility: str, instrument: str,
                     headers: Mapping[str, Mapping[str, str]]) -> "InstrumentManifest":
        """Describe local headers or SVO metadata without downloading files."""
        bands = sorted(headers)
        published_bands: Dict[str, str] = {}
        sources = set()
        detector_types = set()
        band_detectors = {}
        band_detector_sources = {}
        filter_components: Dict[str, str] = {}
        for band in bands:
            header = headers[band]
            try:
                roles = parse_components(header.get("Components"))
            except ComponentError as exc:
                logger.warning("%s/%s band %s: %s in its Components header; its published contents "
                               "are treated as unknown", facility, instrument, band, exc)
                roles = ()
            if roles:
                published_bands[band] = recipe_name(roles)
                if roles == (FILTER,):
                    filter_components[band] = f"{band}.dat"
            if header.get("ComponentsSource"):
                sources.add(header["ComponentsSource"])
            dtype = normalize_detector_type(header.get("DetectorType"))
            band_detector_sources[band] = header.get("DetectorTypeSource", "unknown") if dtype else "assumed"
            if dtype:
                detector_types.add(dtype)
                band_detectors[band] = dtype
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
        return cls(
            facility=facility,
            instrument=instrument,
            bands=bands,
            ground_based=is_ground_based(facility),
            default_site=own_site(facility),
            published=published,
            published_bands=published_bands,
            components={FILTER: filter_components} if filter_components else {},
            detector_type=detector_types.pop() if len(detector_types) == 1 else ENERGY,
            detector_types=band_detectors,
            detector_type_sources=band_detector_sources,
        )

    def save(self, instrument_dir: PathLike) -> Path:
        path = Path(instrument_dir) / MANIFEST_NAME
        data = asdict(self)
        for key in ("variants", "atmosphere_sites", "archived_bands"):  # saved only when set
            if not data[key]:
                del data[key]
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    # ---- variants ----------------------------------------------------
    def variant_files(self) -> List[str]:
        """Every file referenced by a variant."""
        return sorted({str(rel) for named in self.variants.values() for entry in named.values()
                       for rel in dict(entry.get("files") or {}).values()})

    def with_variants(self, selection: Optional[Mapping[str, str]]) -> "InstrumentManifest":
        """A copy whose components use the selected variants.

        ``selection`` maps a component key (``"qe"``, ``"filter+optics+qe"``)
        to a variant name.  The variant's files replace that key's default
        entries; bands the variant does not cover keep no entry for the key,
        so composing them fails rather than silently using the default.
        """
        if not selection:
            return self
        view = deepcopy(self)
        for key, name in selection.items():
            canonical = recipe_name(parse_components(key))
            named = self.variants.get(canonical, {})
            if name not in named:
                have = ", ".join(sorted(named)) or "none"
                raise ComponentError(f"{self.facility}/{self.instrument} has no {canonical} variant "
                                     f"{name!r} (available: {have})")
            view.components[canonical] = {str(b): str(rel) for b, rel in dict(named[name]["files"]).items()}
        return view

    # ---- queries -----------------------------------------------------
    def sites(self) -> List[str]:
        """Every atmosphere site of the instrument's own location, default first.

        That is ``default_site``, then ``atmosphere_sites``, then the team
        sites registered for this instrument in :mod:`facility_sites`.
        """
        names = [self.default_site, *self.atmosphere_sites,
                 *instrument_sites(self.facility, self.instrument)]
        return list(dict.fromkeys(name for name in names if name))

    def component_file(self, role: str, band: str) -> Optional[str]:
        """File for a component key (a role or composite recipe) and band."""
        entries = self.components.get(role, {})
        return entries.get(band) or entries.get(SHARED)

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
        return [recipe_name(roles) for roles in _RECIPES if bands and all(self.cover(b, roles) for b in bands)]

    def available_roles(self) -> Tuple[str, ...]:
        """Single roles available on their own for every primary band."""
        bands = self.primary_bands()
        return tuple(r for r in (FILTER, OPTICS, QE) if bands and all(self.cover(b, (r,)) for b in bands))

    def bands_with(self, role: str) -> List[str]:
        """Primary bands that have a ``role`` component on its own."""
        return [b for b in self.primary_bands() if self.cover(b, (role,))]

    def local(self, instrument_dir: PathLike) -> "InstrumentManifest":
        """An availability view containing only files actually present locally."""
        view = deepcopy(self)
        root = Path(instrument_dir).resolve()
        view.bands = [b for b in self.bands if (root / f"{b}.dat").is_file()]
        view.components = {role: _present(root, entries, "component") for role, entries in view.components.items()}
        for named in view.variants.values():
            for entry in named.values():
                entry["files"] = _present(root, dict(entry.get("files") or {}), "variant")
        return view

    def missing_files(self, instrument_dir: PathLike) -> List[str]:
        root = Path(instrument_dir)
        paths = {f"{b}.dat" for b in self.bands}
        for entries in self.components.values():
            paths.update(entries.values())
        paths.update(self.variant_files())
        return sorted(rel for rel in paths if not (root / rel).is_file())

    def describe(self, instrument_dir: Optional[PathLike] = None) -> Dict[str, object]:
        if instrument_dir is not None:
            info = self.local(instrument_dir).describe()
            info["missing_files"] = self.missing_files(instrument_dir)
            # The set's bands, as FilterLibrary.published_complete counts them: never an archived copy.
            info["expected_bands"] = [band for band in self.bands if band not in self.archived_bands]
            return info
        from ..component_sources.catalog import component_availability

        recipes = {}
        primary_bands = self.primary_bands()
        available_roles = self.available_roles()
        role_bands = {role: self.bands_with(role) for role in (FILTER, OPTICS, QE)}
        source_availability = component_availability(self.facility, self.instrument)
        for roles in _RECIPES:
            bands = [b for b in primary_bands if self.cover(b, roles)]
            if bands:
                recipes[recipe_name(roles)] = bands
        dtypes = set(self.detector_types.get(b, self.detector_type) for b in self.bands)
        detector_type = None
        if dtypes:
            detector_type = dtypes.pop() if len(dtypes) == 1 else "mixed"
        return {
            "facility": self.facility,
            "instrument": self.instrument,
            "bands": list(self.bands),
            "published": self.published.get("components") or "unknown",
            "published_source": self.published.get("source", ""),
            "published_bands": dict(self.published_bands),
            "components": list(available_roles),
            "buildable": self.buildable(),
            "recipes_by_band": recipes,
            "partial_components": {
                role: bands for role, bands in role_bands.items()
                if role not in available_roles and bands
            },
            "atmosphere": bool(self.ground_based and self.sites()),
            "atmosphere_site": self.default_site,
            "atmosphere_sites": self.sites() if self.ground_based else [],
            "platform": platform(self.facility),
            "default_site": self.default_site,
            "detector_type": detector_type,
            "detector_types": dict(self.detector_types),
            "detector_type_sources": dict(self.detector_type_sources),
            "component_status": source_availability["note"] or None,
            "component_availability": source_availability,
            "missing_files": [],
            "variants": {
                key: {name: {k: v for k, v in entry.items() if k != "files"}
                      | {"bands": sorted(dict(entry.get("files") or {}))}
                      for name, entry in named.items()}
                for key, named in self.variants.items()
            },
        }


@dataclass
class AtmosphereGrid:
    """An atmosphere site: its grid of observing conditions and provenance.

    ``entries`` rows always carry a numeric ``airmass`` (released SED_Tools
    reads every served row that way); they may add their own ``source`` and
    ``derivation``.  Atmospheres whose source does not state the airmass are
    listed under ``unstated`` instead and selected with ``airmass=None``.

    The provenance fields (``kind`` onwards) are optional and only written
    when set, so manifests saved by earlier builders keep their bytes.
    """

    site: str
    entries: List[Dict[str, object]]  # {"airmass": float, "pwv": float|None, "file": str}
    source: str = ""
    altitude_m: Optional[float] = None
    notes: str = ""
    schema: int = MANIFEST_SCHEMA
    #: "team" (published by an instrument team) or "derived" (computed by SED_Tools)
    kind: Optional[str] = None
    model: Optional[str] = None            # e.g. "MODTRAN", "ATRAN", "ESO SkyCalc ..."
    model_version: Optional[str] = None    # as the source states it
    citation: Optional[str] = None         # papers / DOIs / bibcodes
    parameters: Optional[Dict[str, object]] = None  # fixed model inputs, derivation choices
    location: Optional[str] = None         # the observatory the atmosphere describes
    terms: Optional[str] = None            # usage/credit terms, verbatim where published
    #: How composition treats wavelengths beyond this atmosphere's table:
    #: "error", "zero" or "edge"; unset = the instrument's own policy.
    outside_range: Optional[str] = None
    #: With outside_range "error": the largest fraction of a band's throughput
    #: (the product of its other factors) allowed beyond the table, where the
    #: edge values are then held; e.g. a filter's faint leak wing.
    coverage_tolerance: Optional[float] = None
    #: [{"file": str, "source": str, ...}] atmospheres of unstated airmass
    unstated: Optional[List[Dict[str, object]]] = None

    _OPTIONAL = ("kind", "model", "model_version", "citation", "parameters", "location", "terms",
                 "outside_range", "coverage_tolerance", "unstated")

    def __post_init__(self):
        if self.kind is not None and self.kind not in ATMOSPHERE_KINDS:
            raise ComponentError(f"{self.site}: atmosphere kind must be one of {ATMOSPHERE_KINDS}")
        if self.outside_range not in (None, "error", "zero", "edge"):
            raise ComponentError(f"{self.site}: unknown outside-range policy {self.outside_range!r}")
        for entry in self.entries:
            if entry.get("airmass") is None:
                raise ComponentError(f"{self.site}: grid entries need a numeric airmass; "
                                     "list atmospheres of unstated airmass under 'unstated'")

    @classmethod
    def load(cls, site_dir: PathLike) -> "AtmosphereGrid":
        data = json.loads((Path(site_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        for key in self._OPTIONAL:
            if data.get(key) in (None, "", [], {}):
                data.pop(key, None)
        return data

    def save(self, site_dir: PathLike) -> Path:
        path = Path(site_dir) / MANIFEST_NAME
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def files(self) -> List[str]:
        """Every data file the site lists."""
        return [str(e["file"]) for e in self.entries] + [str(e["file"]) for e in (self.unstated or [])]

    def find(self, spec: AtmosphereSpec) -> Optional[str]:
        """Look up exact observing conditions; return None when absent."""
        if spec.airmass is None:
            unstated = list(self.unstated or [])
            if len(unstated) > 1:
                raise ComponentError(f"{self.site}: {len(unstated)} atmospheres of unstated airmass; "
                                     "the site manifest must name one")
            if not unstated:
                return None
            rel = str(unstated[0]["file"])
            _require_inside(rel, f"atmosphere {self.site}")
            return rel
        candidates = []
        for entry in self.entries:
            if not math_isclose(float(entry["airmass"]), spec.airmass):
                continue
            pwv = entry.get("pwv")
            if spec.pwv is None or (pwv is not None and math_isclose(float(pwv), spec.pwv)):
                candidates.append(entry)
        if len(candidates) > 1 and spec.pwv is None:
            pwvs = sorted({float(e["pwv"]) for e in candidates if e.get("pwv") is not None})
            raise ComponentError(f"{self.site}: choose pwv explicitly; available at this airmass: {pwvs}")
        if candidates:
            rel = str(candidates[0]["file"])
            _require_inside(rel, f"atmosphere {self.site}")
            return rel
        return None

    def select(self, spec: AtmosphereSpec) -> str:
        """Require an exact grid point, or explain which conditions exist."""
        filename = self.find(spec)
        if filename is not None:
            return filename
        airmasses = sorted({float(e["airmass"]) for e in self.entries})
        pwvs = sorted({float(e["pwv"]) for e in self.entries if e.get("pwv") is not None})
        asked = "unstated airmass" if spec.airmass is None else f"airmass={spec.airmass:g}"
        raise ComponentError(
            f"No {self.site} atmosphere at {asked}"
            + (f", pwv={spec.pwv:g} mm" if spec.pwv is not None else "")
            + f". Grid: airmass {airmasses}" + (f", pwv {pwvs} mm" if pwvs else "")
            + (" (and one of unstated airmass: airmass=None)" if self.unstated else "")
        )


def _present(root: Path, files: Mapping[str, str], kind: str) -> Dict[str, str]:
    """``files`` (band -> path) less those not installed; ComponentError for a path outside ``root``."""
    present = {}
    for band, rel in files.items():
        path = (root / str(rel)).resolve()
        if not path.is_relative_to(root):
            raise ComponentError(f"{rel}: {kind} resolves outside its instrument directory")
        if path.is_file():
            present[band] = rel
    return present


def _require_inside(rel: str, owner: str) -> None:
    """Refuse a component path outside its own instrument directory: an instrument is composed only from its own."""
    parts = Path(rel).parts
    if not parts or Path(rel).is_absolute() or ".." in parts:
        raise ComponentError(f"{owner}: component path {rel!r} is outside the instrument directory")


def math_isclose(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))
