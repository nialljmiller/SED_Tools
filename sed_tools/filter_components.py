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
        manifest.json                   airmass/PWV grid with its provenance
        <file>.dat

An atmosphere *site* is either a standard atmosphere published by an
instrument team (``kind="team"``: DES, Rubin, HSC, Pan-STARRS, ...) or a
grid SED_Tools computes from a model (``kind="derived"``: ESO SkyCalc).
An instrument may use only atmospheres of its own location: its
``default_site``, its ``atmosphere_sites``, and the team sites registered
for it in :mod:`sed_tools.facility_sites`.

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
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .filter_metadata import (ENERGY, PHOTON, normalize_detector_type,
                              read_filter_header, write_filter_file)
from .facility_sites import instrument_sites, is_ground_based, own_site, platform

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


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------
def index_listing(instrument_dir: PathLike) -> Optional[List[str]]:
    """The curve files the instrument's MESA index file lists, or None without one (or an empty one)."""
    index = Path(instrument_dir) / Path(instrument_dir).name
    if not index.is_file():
        return None
    listed = [line.strip() for line in index.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    return listed or None


def archived_bands(instrument_dir: PathLike) -> Dict[str, Dict[str, object]]:
    """The ``archived_bands`` of the directory's manifest (a list of names is read as names), else {}."""
    path = Path(instrument_dir) / MANIFEST_NAME
    if not path.is_file():
        return {}
    archived = json.loads(path.read_text(encoding="utf-8")).get("archived_bands") or {}
    return {band: {} for band in archived} if isinstance(archived, list) else dict(archived)


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
        if not data["variants"]:
            del data["variants"]
        if not data["atmosphere_sites"]:
            del data["atmosphere_sites"]
        if not data["archived_bands"]:
            del data["archived_bands"]
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
        out: List[str] = []
        for name in names:
            if name and name not in out:
                out.append(name)
        return out

    def component_file(self, role: str, band: str) -> Optional[str]:
        """File for a component key (a role or composite recipe) and band."""
        entries = self.components.get(role, {})
        rel = entries.get(band) or entries.get(SHARED)
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

    def local(self, instrument_dir: PathLike) -> "InstrumentManifest":
        """An availability view containing only files actually present locally."""
        view = deepcopy(self)
        root = Path(instrument_dir).resolve()
        view.bands = [b for b in self.bands if (root / f"{b}.dat").is_file()]
        for role, entries in view.components.items():
            for band, rel in list(entries.items()):
                path = (root / rel).resolve()
                if not path.is_relative_to(root):
                    raise ComponentError(f"{rel}: component resolves outside its instrument directory")
                if not path.is_file():
                    del entries[band]
        for named in view.variants.values():
            for entry in named.values():
                files = dict(entry.get("files") or {})
                for band, rel in list(files.items()):
                    path = (root / str(rel)).resolve()
                    if not path.is_relative_to(root):
                        raise ComponentError(f"{rel}: variant resolves outside its instrument directory")
                    if not path.is_file():
                        del files[band]
                entry["files"] = files
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
        from .component_sources.catalog import component_availability

        recipes = {}
        primary_bands = self.primary_bands()
        available_roles = self.available_roles()
        role_bands = {role: self.bands_with(role) for role in (FILTER, OPTICS, QE)}
        source_availability = component_availability(self.facility, self.instrument)
        for extra in ((), (OPTICS,), (QE,), (OPTICS, QE)):
            roles = (FILTER,) + extra
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
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

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


def _require_inside(rel: str, owner: str) -> None:
    """Components must live inside their own instrument directory.

    Each instrument is composed only from its own published factors; a
    manifest can never point at another instrument's files.
    """
    parts = Path(rel).parts
    if not parts or Path(rel).is_absolute() or ".." in parts:
        raise ComponentError(f"{owner}: component path {rel!r} is outside the instrument directory")


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


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------
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
    from .combine_filters import write_filter_index

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
    """Warn about recipes with optics or QE that only some primary bands have.

    :func:`build_all_products` builds a recipe only for every primary band,
    so without this such a product silently does not appear.  SLOAN/SDSS is
    one: its camera response exists for u ... z but not for the primed
    filters, which are bands of their own.  A bare ``filter`` recipe is not
    reported; for most instruments it would only copy filter-only curves.
    """
    bands = manifest.primary_bands()
    recipes_by_bands: Dict[Tuple[str, ...], List[str]] = {}
    for extra in ((OPTICS,), (QE,), (OPTICS, QE)):
        roles = (FILTER,) + extra
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
    stem = name or (band if band != SHARED else role)
    if peak > 1.0:
        # Kept as published, not clipped: clipping would change the shape.
        # Builders record why in the manifest notes (e.g. DECam's quotients).
        logger.warning("%s: %s component %s peaks at %.6g > 1 (written unclipped)",
                       instrument_dir, role, stem, peak)
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
