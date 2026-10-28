"""Shared machinery for builders whose teams publish plain tabulated factors.

Most sources publish per-band filter curves plus shared optics/QE curves
(and sometimes the standard atmosphere they used).  A builder describes
these as a :class:`SimpleSource` and :func:`build_simple` downloads them,
writes the components, registers the team's atmosphere as the instrument's
own site, and saves the manifest.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..filter_components import (ATMOSPHERE, FILTER, OPTICS, QE, SHARED, AtmosphereGrid,
                                 InstrumentManifest, add_component)
from ..filter_metadata import (ENERGY, PHOTON, SOURCE_SVO, SVOMetadataClient, read_filter_header,
                               svo_components_header, svo_fps_url, write_filter_file)
from ._original import CONNECT_TIMEOUT

Curve = Tuple[np.ndarray, np.ndarray]  # (Angstrom, fraction)


@dataclass
class Table:
    """A remote two-column table and how to read it."""

    url: str
    wl_scale: float = 1.0      # multiply wavelengths by this to get Angstrom
    tr_scale: float = 1.0      # multiply values by this to get a fraction
    cols: Tuple[int, int] = (0, 1)
    delimiter: Optional[str] = None
    skip_header: int = 0
    note: str = ""

    def read(self, session) -> Curve:
        resp = session.get(self.url, timeout=(CONNECT_TIMEOUT, 300))
        resp.raise_for_status()
        return self.parse(resp.content)

    def parse(self, raw: bytes) -> Curve:
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        text = raw.decode("utf-8", "replace")
        rows = []
        for line in text.splitlines()[self.skip_header:]:
            line = line.strip()
            if not line or line[0] in "#!;|" or line[0].isalpha():
                continue
            parts = line.split(self.delimiter) if self.delimiter else line.replace(",", " ").split()
            try:
                pair = (float(parts[self.cols[0]]), float(parts[self.cols[1]]))
                if np.isfinite(pair).all():
                    rows.append(pair)
            except (ValueError, IndexError):
                continue
        if not rows:
            raise RuntimeError(f"No numeric rows in {self.url}")
        data = np.array(rows)
        wl = data[:, 0] * self.wl_scale
        tr = data[:, 1] * self.tr_scale
        # Repeated wavelengths (e.g. overlapping scan segments) are averaged,
        # so the result does not depend on the sort algorithm.
        wl, inverse, counts = np.unique(wl, return_inverse=True, return_counts=True)
        tr = np.bincount(inverse, weights=tr) / counts
        # Published measurements can dip slightly below zero in the noise.
        # A physical transmission cannot be negative, and the component
        # loader requires nonnegative values.
        return wl, np.maximum(tr, 0.0)


def product(curves: Sequence[Curve]) -> Curve:
    """Product on the merged grid over the common range."""
    if len(curves) == 1:
        return curves[0]
    lo = max(c[0][0] for c in curves)
    hi = min(c[0][-1] for c in curves)
    grid = np.unique(np.concatenate([c[0][(c[0] >= lo) & (c[0] <= hi)] for c in curves]))
    out = np.ones_like(grid)
    for wl, tr in curves:
        out *= np.interp(grid, wl, tr)
    return grid, out


@dataclass
class DerivedCurve:
    """A curve SED_Tools computes from a team's files (a ratio, product, power).

    ``derivation`` says exactly how; ``url`` lists the source file(s).  Used
    as a :class:`SiteAtmosphere` entry where a :class:`Table` would hold data
    stored as published.
    """

    url: str
    make: Callable[..., Curve]
    derivation: str
    note: str = ""

    def read(self, session) -> Curve:
        wl, tr = self.make(session)
        return np.asarray(wl, dtype=float), np.asarray(tr, dtype=float)


@dataclass
class SiteAtmosphere:
    """An atmosphere supplied by an instrument team, registered as a named site.

    ``entries`` are ``(airmass, pwv, source)``; ``source`` is a :class:`Table`
    (data stored as published, units converted) or a :class:`DerivedCurve`.
    ``airmass=None`` records an atmosphere whose source does not state the
    airmass; it is listed under the site's ``unstated`` atmospheres.

    The provenance fields (``kind`` onwards) are written to the site
    manifest only when set; ``kind`` is normally ``"team"``.
    """

    site: str
    altitude_m: Optional[float]
    entries: Sequence[Tuple[Optional[float], Optional[float], Table]]  # (airmass, pwv, table)
    source: str
    notes: str = ""
    kind: Optional[str] = None
    model: Optional[str] = None
    model_version: Optional[str] = None
    citation: Optional[str] = None
    location: Optional[str] = None
    terms: Optional[str] = None
    parameters: Optional[Dict[str, object]] = None
    outside_range: Optional[str] = None  # unset: the instrument's own policy
    coverage_tolerance: Optional[float] = None  # see AtmosphereGrid.coverage_tolerance


@dataclass
class SimpleSource:
    facility: str
    instrument: str
    filters: Mapping[str, Table]                  # published band name -> filter-only curve
    optics: Sequence[Table] = ()                  # multiplied into one shared optics curve
    qe: Optional[Table] = None
    svo_ids: Mapping[str, str] = field(default_factory=dict)  # band -> SVO filterID (to add published curves)
    detector_type: str = PHOTON
    ground_based: bool = True
    site: Optional[SiteAtmosphere] = None
    default_site: Optional[str] = None            # e.g. a SkyCalc site for ESO facilities
    composites: Mapping[str, Tuple[str, Callable[..., Curve]]] = field(default_factory=dict)
    outside_range: str = "error"
    references: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""
    published_from_filters: bool = False  # original filter scans can stand alone without SVO
    #: band -> file stem of its filter component when it is not the band name
    #: (e.g. a newer filter generation must not reuse an older curve's file)
    filter_names: Mapping[str, str] = field(default_factory=dict)
    #: Further team atmospheres of the same location, offered next to the
    #: default site (``site`` or ``default_site``) as ``atmosphere_sites``.
    sites: Sequence[SiteAtmosphere] = ()


def ensure_published(inst_dir: Path, svo_ids: Mapping[str, str], client: SVOMetadataClient) -> None:
    """Write any missing published curves from SVO (the MESA-facing default).

    If SVO has no record for some of ``svo_ids``, the others are written and
    a RuntimeError lists the missing ones.  The MESA index is written only
    when nothing is missing, so a partial set never looks complete.
    """
    from ..combine_filters import write_filter_index

    inst_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for band, fid in svo_ids.items():
        path = inst_dir / f"{band}.dat"
        if path.exists():
            continue
        record = client.get_filter(fid)
        if record is None:
            missing.append(fid)
            continue
        write_filter_file(
            path, record.wavelength, record.transmission,
            detector_type=record.detector_type or PHOTON, source=SOURCE_SVO, filter_id=fid,
            wavelength_unit=record.wavelength_unit,
            extra={**svo_components_header(record.components), "Sources": f"{svo_fps_url()}?ID={fid}"},
        )
    if missing:
        raise RuntimeError(
            f"{inst_dir}: SVO has no {', '.join(missing)}, so the published curves are incomplete "
            "(the others were written). Check these IDs at SVO, or install the curves from the njm mirror.")
    if not (inst_dir / inst_dir.name).exists() and list(inst_dir.glob("*.dat")):
        write_filter_index(inst_dir, inst_dir.name)


#: File name of a site's atmosphere of unstated airmass.
UNSTATED_FILE = "am_unstated.dat"


def write_site(atmosphere_root: Path, site: SiteAtmosphere, session) -> Path:
    """Write a team atmosphere site: its curves and its grid manifest.

    Every curve is checked (increasing wavelengths, finite values in 0-1)
    before anything is written, so a bad download leaves no partial site.
    """
    atmosphere_root = Path(atmosphere_root)
    site_dir = atmosphere_root / site.site
    curves = []
    for airmass, pwv, table in site.entries:
        if airmass is None and pwv is not None:
            raise ValueError(f"{site.site}: an atmosphere of unstated airmass cannot state a pwv")
        wl, tr = table.read(session)
        wl, tr = np.asarray(wl, dtype=float), np.asarray(tr, dtype=float)
        if (len(wl) < 2 or not np.all(np.isfinite(wl)) or not np.all(np.isfinite(tr))
                or np.any(np.diff(wl) <= 0) or np.any(tr < 0) or np.any(tr > 1.0 + 1e-6)):
            raise ValueError(f"{site.site}: {table.url} is not a transmission (0-1) on increasing wavelengths")
        curves.append((airmass, pwv, table, wl, tr))
    site_dir.mkdir(parents=True, exist_ok=True)
    entries, unstated = [], []
    for airmass, pwv, table, wl, tr in curves:
        if airmass is None:
            name, condition = UNSTATED_FILE, f"site={site.site} airmass=unstated"
        else:
            name = f"am{airmass:.2f}" + (f"_pwv{pwv:g}" if pwv is not None else "") + ".dat"
            condition = f"site={site.site} airmass={airmass:g}" + (f" pwv={pwv:g}" if pwv is not None else "")
        derivation = getattr(table, "derivation", "")
        extra = {"Role": ATMOSPHERE, "Components": ATMOSPHERE, "Atmosphere": condition, "Sources": table.url}
        if site.kind is not None:
            extra["ComponentsSource"] = "derived" if derivation else "original"
        write_filter_file(
            site_dir / name, wl, tr, detector_type=ENERGY, source="component", wavelength_unit="Angstrom",
            extra=extra,
        )
        row: Dict[str, object] = {"airmass": airmass, "pwv": pwv, "file": name}
        if site.kind is not None:
            row["source"] = table.url
            if derivation:
                row["derivation"] = derivation
            elif getattr(table, "note", ""):
                row["note"] = table.note
        if airmass is None:
            del row["airmass"], row["pwv"]
            unstated.append(row)
        else:
            entries.append(row)
    AtmosphereGrid(site=site.site, entries=entries, altitude_m=site.altitude_m,
                   source=site.source, notes=site.notes, kind=site.kind, model=site.model,
                   model_version=site.model_version, citation=site.citation, location=site.location,
                   terms=site.terms, parameters=site.parameters, outside_range=site.outside_range,
                   coverage_tolerance=site.coverage_tolerance, unstated=unstated or None).save(site_dir)
    return site_dir


def register_sites(manifest: InstrumentManifest, default: Optional[str], sites: Sequence[SiteAtmosphere]) -> None:
    """Record an instrument's own atmosphere sites in its manifest."""
    if default:
        manifest.default_site = default
    extra = [s.site for s in sites if s.site != manifest.default_site]
    manifest.atmosphere_sites = list(dict.fromkeys([*manifest.atmosphere_sites, *extra]))


def _publish_original_filters(spec: SimpleSource, inst_dir: Path, session) -> Dict[str, Curve]:
    """Write the original filter scans as the published curves, and the MESA index.

    Existing published curves are kept.  Every band is fetched and checked
    before anything is written: a failed upstream request must not leave a
    seemingly complete partial instrument.
    """
    from ..combine_filters import write_filter_index

    curves = {}
    for band, table in spec.filters.items():
        wl, tr = table.read(session)
        if (len(wl) < 2 or not np.all(np.isfinite(wl)) or
                not np.all(np.isfinite(tr)) or np.any(np.diff(wl) <= 0) or
                np.any(tr > 1.001)):
            raise ValueError(f"Invalid filter transmission from {table.url}")
        curves[band] = (wl, tr)
    inst_dir.mkdir(parents=True, exist_ok=True)
    for band, curve in curves.items():
        path = inst_dir / f"{band}.dat"
        if not path.exists():
            write_filter_file(
                path, *curve, detector_type=spec.detector_type, source="assumed",
                wavelength_unit="Angstrom",
                extra={"Components": FILTER, "ComponentsSource": "original",
                       "Sources": spec.filters[band].url},
            )
    write_filter_index(inst_dir, spec.instrument)
    return curves


def _add_components(spec: SimpleSource, inst_dir: Path, manifest: InstrumentManifest,
                    original_filters: Mapping[str, Curve], session) -> None:
    """Write the filter, optics, qe and composite components of ``spec``."""
    for band, table in spec.filters.items():
        if (spec.published_from_filters and
                read_filter_header(inst_dir / f"{band}.dat").get("ComponentsSource") == "original"):
            continue  # the published filter scan is already this component
        wl, tr = original_filters.get(band) or table.read(session)
        add_component(inst_dir, FILTER, wl, tr, band=band, name=spec.filter_names.get(band, band),
                      source=table.url + (f" ({table.note})" if table.note else ""), manifest=manifest)
        # Each band cites its own filter file ("filter:<band>"); drop a
        # role-wide entry left by an older build, which named one band's file.
        manifest.sources.pop(FILTER, None)
    if spec.optics:
        wl, tr = product([t.read(session) for t in spec.optics])
        add_component(inst_dir, OPTICS, wl, tr, band=SHARED, name="optics",
                      source="product of " + "; ".join(t.url for t in spec.optics), manifest=manifest)
    if spec.qe is not None:
        wl, tr = spec.qe.read(session)
        add_component(inst_dir, QE, wl, tr, band=SHARED, name="qe",
                      source=spec.qe.url + (f" ({spec.qe.note})" if spec.qe.note else ""), manifest=manifest)
    for band, (recipe, make) in spec.composites.items():
        wl, tr = make(session)
        add_component(inst_dir, recipe, wl, tr, band=band, name=band,
                      source=f"{spec.references.get(recipe, '')}", manifest=manifest)


def _write_sites(spec: SimpleSource, manifest: InstrumentManifest, atmosphere_root: Path, session) -> None:
    """Write the team atmospheres of ``spec`` and record its sites in ``manifest``."""
    if spec.site is not None:
        write_site(atmosphere_root, spec.site, session)
        manifest.default_site = spec.site.site
    elif spec.default_site:
        manifest.default_site = spec.default_site
    for extra in spec.sites:
        write_site(atmosphere_root, extra, session)
    if spec.sites:
        register_sites(manifest, None, spec.sites)


def build_simple(
    spec: SimpleSource,
    filter_root: Union[str, Path],
    *,
    atmosphere_root: Union[str, Path, None] = None,
    session=None,
    svo_client: Optional[SVOMetadataClient] = None,
) -> Path:
    """Write the published curves (where missing), components, atmospheres and manifest of ``spec``."""
    if session is None:
        import requests

        session = requests.Session()
    filter_root = Path(filter_root)
    inst_dir = filter_root / spec.facility / spec.instrument
    if spec.published_from_filters:
        original_filters = _publish_original_filters(spec, inst_dir, session)
    else:
        ensure_published(inst_dir, spec.svo_ids, svo_client or SVOMetadataClient())
        original_filters = {}

    # Every band given a component must have a published curve to belong to.
    published = {p.stem for p in inst_dir.glob("*.dat")}
    unpublished = sorted((set(spec.filters) | set(spec.composites)) - published)
    if unpublished:
        raise FileNotFoundError(
            f"{inst_dir} has no published curve for {', '.join(unpublished)}; add them to the "
            f"{spec.facility}/{spec.instrument} builder's svo_ids, or install them from the njm mirror first")

    manifest = (InstrumentManifest.from_published(inst_dir) if spec.published_from_filters
                else InstrumentManifest.load(inst_dir))
    # Keep what the published curves already provide (e.g. SVO filter-only siblings).
    manifest.components = InstrumentManifest.from_published(inst_dir).components
    _add_components(spec, inst_dir, manifest, original_filters, session)

    manifest.facility, manifest.instrument = spec.facility, spec.instrument
    manifest.detector_type = spec.detector_type
    manifest.ground_based = spec.ground_based
    manifest.outside_range = spec.outside_range
    manifest.sources.update(spec.references)
    manifest.notes = spec.notes
    root = Path(atmosphere_root) if atmosphere_root else filter_root.parent / "atmospheres"
    _write_sites(spec, manifest, root, session)
    manifest.save(inst_dir)
    return inst_dir


def register_variant(manifest: InstrumentManifest, key: str, name: str,
                     files: Mapping[str, str], **info: object) -> None:
    """Record curves already written as the named alternative ``name`` of ``key``.

    ``files`` maps band (or ``"*"``) to a path inside the instrument;
    ``info`` adds ``description``, ``source``, ``epoch``, ``derived``,
    ``default`` and the like (see :attr:`InstrumentManifest.variants`).
    """
    entry = {"files": dict(files)}
    entry.update(info)
    manifest.variants.setdefault(key, {})[name] = entry
