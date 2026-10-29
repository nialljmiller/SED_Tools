"""Shared machinery for instruments rebuilt from their teams' original files.

These builders reproduce the curated published curves (and any separately
published components) that the njm mirror serves, directly from the original
sources, so that :meth:`FilterLibrary.published` can rebuild them when the
mirror is unavailable.  Each source module (``surveys``, ``space``,
``observatories``, ``standards``) maps an instrument to a *recipe*: a
function that downloads and parses the team's files through a
:class:`Fetcher` and returns an :class:`Instrument`.  Nothing is written
until every file of the instrument has been fetched and checked.

Cleaning follows the format contract of the curated library and is the same
for every recipe (:func:`clean_curve`): undefined (and sentinel) samples are
dropped, wavelengths sorted, repeated wavelengths averaged, negative
measurement noise set to 0 and, where the recipe asks, the peak scaled to 1.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..filters.compose import InstrumentManifest, add_component, parse_components, recipe_name
from ..filters.curve_metadata import DETECTOR_TYPES, write_filter_file

logger = logging.getLogger(__name__)

#: Header value for curves taken from the instrument team.
ORIGINAL = "original"
C_ANGSTROM_PER_S = 2.99792458e18
_UNIT_SCALE = {"angstrom": 1.0, "a": 1.0, "aa": 1.0, "nm": 10.0, "micron": 1e4, "um": 1e4, "mm": 1e7}
USER_AGENT = "SED_Tools-original-sources"


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------
#: Seconds an original source's server has to accept a connection, as for SVO: a host
#: that is down fails fast, while a large file still has minutes to arrive.
CONNECT_TIMEOUT = 10.0


class Fetcher:
    """Download each URL once per build through a ``requests``-like session."""

    def __init__(self, session=None, timeout: Union[float, Tuple[float, float]] = (CONNECT_TIMEOUT, 300)):
        if session is None:
            import requests

            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
        self.session = session
        self.timeout = timeout
        self._cache: Dict[str, bytes] = {}

    def get(self, url: str, *, allow_html: bool = False) -> bytes:
        if url not in self._cache:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.content
            head = data[:512].lstrip().lower()
            if not allow_html and (head.startswith(b"<!doctype html") or head.startswith(b"<html")):
                raise ValueError(f"{url} returned HTML, not data")
            self._cache[url] = data
        return self._cache[url]

    def text(self, url: str, encoding: str = "utf-8", errors: str = "strict", **kw) -> str:
        return self.get(url, **kw).decode(encoding, errors)

    def stream(self, url: str) -> io.BytesIO:
        return io.BytesIO(self.get(url))


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------
def to_angstrom(x: Sequence[float], unit: str) -> np.ndarray:
    """Convert wavelengths, frequencies (GHz/Hz) or wavenumbers (1/cm) to Angstrom."""
    x = np.asarray(x, dtype=float)
    u = unit.strip().lower().replace("µ", "u")
    if u in _UNIT_SCALE:
        return x * _UNIT_SCALE[u]
    if u in ("ghz", "hz"):
        hz = x * (1e9 if u == "ghz" else 1.0)
        with np.errstate(divide="ignore"):
            return C_ANGSTROM_PER_S / hz
    if u in ("1/cm", "cm-1", "cm^-1"):
        with np.errstate(divide="ignore"):
            return 1e8 / x
    raise ValueError(f"Unknown wavelength unit {unit!r}")


def clean_curve(wl, tr, *, sentinel: Optional[float] = None, normalize: bool = False
                ) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return a curve that meets the library's format contract, and a note of the changes.

    Drops non-finite (and ``sentinel``) samples, sorts by wavelength,
    averages repeated wavelengths, sets negative transmissions (measurement
    noise) to 0 and optionally scales the peak to 1 (for effective areas or
    relative responses; synthetic photometry is unaffected by a constant).
    """
    wl = np.asarray(wl, dtype=float)
    tr = np.asarray(tr, dtype=float)
    notes = []
    ok = np.isfinite(wl) & np.isfinite(tr) & (wl > 0)
    if sentinel is not None:
        ok &= ~np.isclose(tr, sentinel)
    dropped = int((~ok).sum())
    wl, tr = wl[ok], tr[ok]
    order = np.argsort(wl, kind="stable")
    wl, tr = wl[order], tr[order]
    uniq, inv, counts = np.unique(wl, return_inverse=True, return_counts=True)
    merged = len(wl) - len(uniq)
    if merged:
        tr = np.bincount(inv, weights=tr) / counts
        wl = uniq
    neg = tr < 0
    if neg.any():
        notes.append(f"{int(neg.sum())} negative samples (min {float(tr.min()):.3g}) set to 0")
        tr = np.where(neg, 0.0, tr)
    if dropped:
        notes.append(f"{dropped} undefined samples dropped")
    if merged:
        notes.append(f"{merged} duplicate wavelengths averaged")
    if normalize:
        peak = float(tr.max())
        if peak > 0:
            if peak != 1.0:
                notes.append(f"scaled by {1.0 / peak:.6g} (peak normalised)")
            tr = tr / peak
    if len(wl) < 2 or not (tr > 0).any():
        raise ValueError("curve has fewer than 2 samples or no positive transmission")
    return wl, tr, "; ".join(notes)


def safe_band(name: str) -> str:
    """A band name usable as ``<band>.dat``: exactly one dot in the filename."""
    import re

    band = re.sub(r"[^A-Za-z0-9_+-]", "_", name.replace(".", "p")).strip("_")
    if not band:
        raise ValueError(f"empty band name from {name!r}")
    return band


@dataclass
class Band:
    name: str
    wavelength: np.ndarray          # Angstrom, cleaned
    transmission: np.ndarray
    detector_type: str              # photon | energy
    components: str                 # e.g. "filter+optics+qe+atmosphere"
    sources: str                    # URL(s) of the numeric curve, ' | ' separated
    detector_source: str = ORIGINAL
    components_source: str = ORIGINAL
    clean_note: str = ""            # what clean_curve changed (for notes and logs)
    atmosphere: str = ""            # conditions of an included atmosphere, e.g. "airmass=1.2"

    def __post_init__(self):
        if self.detector_type not in DETECTOR_TYPES:
            raise ValueError(f"{self.name}: detector_type must be photon or energy")
        self.components = recipe_name(parse_components(self.components))


@dataclass
class Component:
    role: str                       # filter | optics | qe | filter+optics ...
    wavelength: np.ndarray          # Angstrom
    transmission: np.ndarray        # fraction
    band: str = "*"                 # band name or "*" (shared)
    source: str = ""
    name: Optional[str] = None      # file stem; lets several bands share one file


def make_band(name, wl, tr, *, unit, detector_type, components, sources,
              sentinel=None, normalize=False) -> Band:
    """Convert units, clean, and wrap a parsed curve as a :class:`Band`."""
    w, t, note = clean_curve(to_angstrom(wl, unit), tr, sentinel=sentinel, normalize=normalize)
    return Band(safe_band(name), w, t, detector_type, components, sources, clean_note=note)


@dataclass
class Instrument:
    facility: str
    instrument: str
    bands: Sequence[Band]
    notes: str
    ground_based: bool
    components: Sequence[Component] = ()
    outside_range: str = "zero"
    #: Atmospheres the team published (``_common.SiteAtmosphere``), written as
    #: named sites; ``default_site`` (one of them, or a SkyCalc site) is the
    #: one composition uses when none is named.
    sites: Sequence["SiteAtmosphere"] = ()
    default_site: Optional[str] = None


Recipe = Callable[[Fetcher], Instrument]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def write_instrument(filter_root: Union[str, Path], spec: Instrument, *,
                     atmosphere_root: Union[str, Path, None] = None, session=None) -> Path:
    """Write the published curves, index, components and manifest of ``spec``.

    Existing published ``.dat`` files and MESA index files are kept as they
    are; only missing curves are written.  The team's atmospheres
    (``spec.sites``) are written under ``atmosphere_root`` (default
    ``<filter_root>/../atmospheres``) before anything else, so a failed
    download leaves the instrument untouched.
    """
    if spec.sites:
        from ._common import write_site

        if session is None:
            import requests

            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
        root = Path(atmosphere_root) if atmosphere_root else Path(filter_root).parent / "atmospheres"
        for site in spec.sites:
            write_site(root, site, session)
    names = [b.name for b in spec.bands]
    if len({n.lower() for n in names}) != len(names):
        raise ValueError(f"{spec.facility}/{spec.instrument}: duplicate band names {names}")
    inst_dir = Path(filter_root) / spec.facility / spec.instrument
    inst_dir.mkdir(parents=True, exist_ok=True)
    for b in spec.bands:
        path = inst_dir / f"{b.name}.dat"
        if path.exists():
            logger.info("%s exists; keeping it", path)
            continue
        write_filter_file(
            path, b.wavelength, b.transmission,
            detector_type=b.detector_type, source=b.detector_source,
            filter_id=f"{spec.facility}/{spec.instrument}.{b.name}",
            wavelength_unit="Angstrom",
            extra={"Components": b.components, "ComponentsSource": b.components_source,
                   "Atmosphere": b.atmosphere or None, "Sources": b.sources})
    index = inst_dir / spec.instrument
    if not index.exists():
        index.write_text("".join(f"{n}.dat\n" for n in sorted(names)), encoding="utf-8")
    manifest = InstrumentManifest.from_published(inst_dir)
    manifest.facility, manifest.instrument = spec.facility, spec.instrument
    manifest.ground_based = spec.ground_based
    for c in spec.components:
        add_component(inst_dir, c.role, c.wavelength, c.transmission, band=c.band,
                      name=c.name, source=c.source, manifest=manifest)
    manifest.notes = spec.notes
    manifest.outside_range = spec.outside_range
    if spec.sites or spec.default_site:
        from ._common import register_sites

        register_sites(manifest, spec.default_site, spec.sites)
    manifest.detector_type_sources = {b.name: b.detector_source for b in spec.bands}
    for b in spec.bands:
        manifest.sources.setdefault("published", b.sources.split(" | ")[0])
    manifest.save(inst_dir)
    return inst_dir


def make_builder(recipes: Mapping[str, Tuple[str, Recipe]]):
    """``build(filter_root, *, instrument, session=None)`` for one source module."""

    def build(filter_root, *, instrument, session=None, facility=None, atmosphere_root=None,
              **_ignored) -> Path:
        try:
            fac, recipe = recipes[instrument]
        except KeyError:
            raise ValueError(f"No original-source recipe for {instrument!r}; "
                             f"available: {sorted(recipes)}") from None
        if facility is not None and facility != fac:
            raise ValueError(f"{instrument} belongs to {fac}, not {facility}")
        fetcher = Fetcher(session)
        spec = recipe(fetcher)
        return write_instrument(filter_root, spec, atmosphere_root=atmosphere_root, session=fetcher.session)

    return build


# --------------------------------------------------------------------------
# Small parsing helpers shared by several recipes
# --------------------------------------------------------------------------
def safe_member(name: str, *, flat: bool = False) -> None:
    """Refuse absolute or parent-relative archive members (and nested ones if ``flat``)."""
    from pathlib import PurePosixPath

    pp = PurePosixPath(name)
    if pp.is_absolute() or ".." in pp.parts or "\\" in name or (flat and "/" in name):
        raise ValueError(f"refusing archive member {name!r}")
