"""Shared machinery for builders whose teams publish plain tabulated factors.

Most sources publish per-band filter curves plus shared optics/QE curves
(and sometimes the standard atmosphere they used).  A builder describes
these as a :class:`SimpleSource` and :func:`build_simple` downloads them,
writes the components, registers the team's atmosphere as the instrument's
own site, and saves the manifest.
"""

from __future__ import annotations

import gzip
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..filter_components import (ATMOSPHERE, FILTER, OPTICS, QE, SHARED, AtmosphereGrid,
                                 InstrumentManifest, add_component)
from ..filter_metadata import (ENERGY, PHOTON, SOURCE_SVO, SVOMetadataClient,
                               read_filter_header, svo_components_header, write_filter_file)

logger = logging.getLogger(__name__)

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
        resp = session.get(self.url, timeout=300)
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
        order = np.argsort(wl)
        wl, idx = np.unique(wl[order], return_index=True)
        # Published measurements can dip slightly below zero in the noise.
        # A physical transmission cannot be negative, and the component
        # loader requires nonnegative values.
        return wl, np.maximum(tr[order][idx], 0.0)


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
class SiteAtmosphere:
    site: str
    altitude_m: float
    entries: Sequence[Tuple[float, Optional[float], Table]]  # (airmass, pwv, table)
    source: str
    notes: str = ""


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


def ensure_published(inst_dir: Path, svo_ids: Mapping[str, str], client: SVOMetadataClient) -> None:
    """Write any missing published curves from SVO (the MESA-facing default)."""
    from ..combine_filters import write_filter_index

    inst_dir.mkdir(parents=True, exist_ok=True)
    for band, fid in svo_ids.items():
        path = inst_dir / f"{band}.dat"
        if path.exists():
            continue
        record = client.get_filter(fid)
        if record is None:
            logger.warning("SVO has no %s", fid)
            continue
        write_filter_file(
            path, record.wavelength, record.transmission,
            detector_type=record.detector_type or PHOTON, source=SOURCE_SVO, filter_id=fid,
            wavelength_unit=record.wavelength_unit, extra=svo_components_header(record.components),
        )
    if not (inst_dir / inst_dir.name).exists() and list(inst_dir.glob("*.dat")):
        write_filter_index(inst_dir, inst_dir.name)


def write_site(atmosphere_root: Path, site: SiteAtmosphere, session) -> None:
    site_dir = atmosphere_root / site.site
    site_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for airmass, pwv, table in site.entries:
        wl, tr = table.read(session)
        name = f"am{airmass:.2f}" + (f"_pwv{pwv:g}" if pwv is not None else "") + ".dat"
        write_filter_file(
            site_dir / name, wl, tr, detector_type=ENERGY, source="component", wavelength_unit="Angstrom",
            extra={"Role": ATMOSPHERE, "Components": ATMOSPHERE,
                   "Atmosphere": f"site={site.site} airmass={airmass:g}" + (f" pwv={pwv:g}" if pwv is not None else ""),
                   "Sources": table.url},
        )
        entries.append({"airmass": airmass, "pwv": pwv, "file": name})
    AtmosphereGrid(site=site.site, entries=entries, altitude_m=site.altitude_m,
                   source=site.source, notes=site.notes).save(site_dir)


def build_simple(
    spec: SimpleSource,
    filter_root: Union[str, Path],
    *,
    atmosphere_root: Union[str, Path, None] = None,
    session=None,
    svo_client: Optional[SVOMetadataClient] = None,
) -> Path:
    if session is None:
        import requests

        session = requests.Session()
    filter_root = Path(filter_root)
    inst_dir = filter_root / spec.facility / spec.instrument
    if not spec.published_from_filters:
        ensure_published(inst_dir, spec.svo_ids, svo_client or SVOMetadataClient())

    # Fetch every original band before changing the local library. A failed
    # upstream request must not leave a seemingly complete partial instrument.
    original_filters = {}
    if spec.published_from_filters:
        for band, table in spec.filters.items():
            wl, tr = table.read(session)
            if (len(wl) < 2 or not np.all(np.isfinite(wl)) or
                    not np.all(np.isfinite(tr)) or np.any(np.diff(wl) <= 0) or
                    np.any(tr > 1.001)):
                raise ValueError(f"Invalid filter transmission from {table.url}")
            original_filters[band] = (wl, tr)
        inst_dir.mkdir(parents=True, exist_ok=True)
        for band, curve in original_filters.items():
            path = inst_dir / f"{band}.dat"
            if not path.exists():
                write_filter_file(
                    path, *curve, detector_type=spec.detector_type, source="assumed",
                    wavelength_unit="Angstrom",
                    extra={"Components": FILTER, "ComponentsSource": "original",
                           "Sources": spec.filters[band].url},
                )
        from ..combine_filters import write_filter_index

        write_filter_index(inst_dir, spec.instrument)

    manifest = (InstrumentManifest.from_published(inst_dir) if spec.published_from_filters
                else InstrumentManifest.load(inst_dir))
    # Keep what the published curves already provide (e.g. SVO filter-only siblings).
    manifest.components = InstrumentManifest.from_published(inst_dir).components
    published = {p.stem for p in inst_dir.glob("*.dat")}
    for band, table in spec.filters.items():
        if band not in published:
            logger.info("%s/%s: no published %s curve; skipping its filter", spec.facility, spec.instrument, band)
            continue
        if (spec.published_from_filters and
                read_filter_header(inst_dir / f"{band}.dat").get("ComponentsSource") == "original"):
            continue  # the published filter scan is already this component
        wl, tr = original_filters.get(band) or table.read(session)
        add_component(inst_dir, FILTER, wl, tr, band=band, name=band,
                      source=table.url + (f" ({table.note})" if table.note else ""), manifest=manifest)
    if spec.optics:
        wl, tr = product([t.read(session) for t in spec.optics])
        add_component(inst_dir, OPTICS, wl, tr, band=SHARED, name="optics",
                      source="product of " + "; ".join(t.url for t in spec.optics), manifest=manifest)
    if spec.qe is not None:
        wl, tr = spec.qe.read(session)
        add_component(inst_dir, QE, wl, tr, band=SHARED, name="qe",
                      source=spec.qe.url + (f" ({spec.qe.note})" if spec.qe.note else ""), manifest=manifest)
    for band, (recipe, make) in spec.composites.items():
        if band in published:
            wl, tr = make(session)
            add_component(inst_dir, recipe, wl, tr, band=band, name=band,
                          source=f"{spec.references.get(recipe, '')}", manifest=manifest)

    manifest.facility, manifest.instrument = spec.facility, spec.instrument
    manifest.detector_type = spec.detector_type
    manifest.ground_based = spec.ground_based
    manifest.outside_range = spec.outside_range
    manifest.sources.update(spec.references)
    manifest.notes = spec.notes
    if spec.site is not None:
        root = Path(atmosphere_root) if atmosphere_root else filter_root.parent / "atmospheres"
        write_site(root, spec.site, session)
        manifest.default_site = spec.site.site
    elif spec.default_site:
        manifest.default_site = spec.default_site
    manifest.save(inst_dir)
    return inst_dir
