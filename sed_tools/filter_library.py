"""Shared filter acquisition and composition workflow for the API and CLI.

The library owns source selection: installed files, NJM, then original
sources. Manifests describe capabilities; filter_components performs the
numerical work. Source catalogues remain data-only descriptions.
"""

from __future__ import annotations

import logging
import re
from functools import cached_property
from pathlib import Path

from requests import RequestException

from .component_sources import get_builder
from .component_sources.catalog import BUILDERS_FOR, builder_for, component_availability
from .filter_components import (
    FILTER, OPTICS, QE, PUBLISHED, AtmosphereGrid, AtmosphereSpec, ComponentError,
    InstrumentManifest, atmosphere_file, build_product, find_atmosphere, parse_components,
    resolve_atmosphere,
)
from .filter_metadata import migrate_filter_tree, svo_components_header
from .njm_filter_grabber import NJMFilterGrabber
from .svo_filter_grabber import SVOFilterBrowser, _clean_path

logger = logging.getLogger(__name__)


class FilterLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self.atmosphere_root = self.root.parent / "atmospheres"

    @cached_property
    def mirror(self):
        return NJMFilterGrabber(base_dir=str(self.root))

    @cached_property
    def svo(self):
        return SVOFilterBrowser(base_dir=str(self.root))

    def directory(self, instrument):
        path = Path(instrument)
        return path if path.is_absolute() or path.is_dir() else self.root / path

    def svo_filters(self, facility, instrument):
        """Resolve user-facing names to SVO keys before requesting curves."""
        fac = self.svo_facility(facility)
        if fac is None:
            raise ComponentError(f"SVO facility not found: {facility}")
        instruments = self.svo.list_instruments(fac.key)
        inst = next((i for i in instruments if instrument.casefold() in
                     (i.key.casefold(), i.label.casefold())), None)
        if inst is None:
            raise ComponentError(f"SVO instrument not found: {facility}/{instrument}")
        return self.svo.list_filters(fac.key, inst.key)

    def svo_facility(self, name):
        identity = self._catalogue_key(name)
        return next((fac for fac in self.svo.list_facilities() if identity in
                     (self._catalogue_key(fac.key), self._catalogue_key(fac.label))), None)

    @staticmethod
    def published_complete(directory):
        """Check every declared band; undeclared partial downloads are unverified."""
        directory = Path(directory)
        if not list(directory.glob("*.dat")):
            return False
        index = directory / directory.name
        if not (directory / "manifest.json").is_file():
            return False
        manifest = InstrumentManifest.load(directory)
        expected = {f"{b}.dat" for b in manifest.bands}
        if index.is_file():
            expected.update(line.strip() for line in index.read_text().splitlines()
                            if line.strip() and not line.lstrip().startswith("#"))
        return bool(expected) and all(Path(name).name == name and (directory / name).is_file()
                                      for name in expected)

    def published(self, facility, instrument):
        directory = self.root / facility / instrument
        if self.published_complete(directory):
            # Sets downloaded by SED_Tools 0.3.0 carry metadata on leading '#' lines.
            migrate_filter_tree(directory)
            return directory

        if self.mirror.is_available():
            self.mirror.download_filters(facility, instrument)
            if self.published_complete(directory):
                return directory

        ref = builder_for(facility, instrument)
        if ref is not None and ref.publishes:
            try:
                get_builder(ref.name)(self.root, **dict(ref.kwargs))
                if self.published_complete(directory):
                    return directory
            except (RequestException, OSError, RuntimeError, ValueError) as exc:
                logger.warning("Original source unavailable for %s/%s: %s", facility, instrument, exc)

        rows = self.svo_filters(facility, instrument)
        if not rows:
            raise ComponentError(f"SVO returned no filters for {facility}/{instrument}")
        self.svo.download_filters(rows)
        directories = {self.root / _clean_path(row.facility) / _clean_path(row.instrument) for row in rows}
        for candidate in [directory, *sorted(directories)]:
            if self.published_complete(candidate):
                return candidate
        raise ComponentError(f"Incomplete passband download for {facility}/{instrument}; retry to recover missing files")

    def available(self, facility, instrument, *, remote=True):
        directory = self.root / facility / instrument
        if (directory / "manifest.json").is_file() or list(directory.glob("*.dat")):
            info = InstrumentManifest.load(directory).describe(directory)
            return dict(info, availability_source="local", errors=[])
        errors = []
        if remote:
            data = self.mirror.fetch_manifest(facility, instrument)
            if data:
                manifest = InstrumentManifest.from_dict(data)
                return dict(manifest.describe(), availability_source="njm", errors=[])
        ref = builder_for(facility, instrument)
        if ref is not None and ref.name == "tabular":
            from .component_sources.tabular import SOURCES

            spec = SOURCES[instrument]
            headers = {band: {"Components": FILTER, "ComponentsSource": "original",
                              "DetectorType": spec.detector_type,
                              "DetectorTypeSource": "assumed"} for band in spec.filters}
            manifest = InstrumentManifest.from_headers(facility, instrument, headers)
            return dict(manifest.describe(), availability_source="original", errors=errors)
        if remote:
            try:
                rows = self.svo_filters(facility, instrument)
                headers = {}
                for row in rows:
                    record = self.svo.metadata_client.get_filter(row.filter_id)
                    if record is None:
                        errors.append(f"SVO: no metadata for {row.filter_id}")
                        headers[row.band] = {}
                    else:
                        headers[row.band] = dict(svo_components_header(record.components),
                                                 DetectorType=record.detector_type or "",
                                                 DetectorTypeSource="svo" if record.detector_type else "assumed")
                if headers:
                    manifest = InstrumentManifest.from_headers(facility, instrument, headers)
                    return dict(manifest.describe(), availability_source="svo", errors=errors)
            except (ComponentError, RequestException, RuntimeError) as exc:
                errors.append(f"SVO: {exc}")
        manifest = InstrumentManifest.from_headers(facility, instrument, {})
        return dict(manifest.describe(), availability_source="unknown", errors=errors)

    @staticmethod
    def components_ready(directory, components=None, bands=None):
        declared = InstrumentManifest.load(directory)
        manifest = declared.local(directory)
        selected = declared.primary_bands() if bands is None else list(bands)
        if not selected or any(b not in manifest.bands for b in selected):
            return False
        if components is None:
            return bool(manifest.buildable())
        requested = parse_components(components)
        if PUBLISHED in requested and len(requested) > 1 and any(not manifest.published_bands.get(b) for b in selected):
            return False
        roles = [r for r in requested if r in (FILTER, OPTICS, QE)]
        return not roles or all(manifest.cover(b, roles) for b in selected)

    def prepare_components(self, directory, *, components=None, bands=None, allow_heavy=False):
        directory = Path(directory)
        if self.components_ready(directory, components, bands):
            return True
        facility, instrument = directory.parent.name, directory.name
        if self.mirror.is_available():
            self.mirror.download_components(facility, instrument)
            if self.components_ready(directory, components, bands):
                return True
        ref = builder_for(facility, instrument)
        if ref is None or not ref.components:
            return False
        if ref.heavy and not allow_heavy:
            logger.warning("%s/%s requires a large source download; use allow_heavy=True or --allow-heavy",
                           facility, instrument)
            return False
        get_builder(ref.name)(self.root, **dict(ref.kwargs))
        return self.components_ready(directory, components, bands)

    def prepare_atmosphere(self, manifest, spec, *, allow_heavy=False):
        spec = resolve_atmosphere(manifest, spec)
        cached = find_atmosphere(self.atmosphere_root, spec)
        if cached is not None:
            return cached
        if self.mirror.download_atmosphere(str(self.atmosphere_root), spec.site):
            grid = AtmosphereGrid.load(self.atmosphere_root / spec.site)
            filename = grid.find(spec)
            if filename is not None:
                self.mirror.download_atmosphere(str(self.atmosphere_root), spec.site, filename)
                cached = find_atmosphere(self.atmosphere_root, spec)
                if cached is not None:
                    return cached
        # Only registered site models may be recovered. Never substitute a
        # different observatory or invent weather conditions for a standard.
        from .component_sources.skycalc import SITES
        from .component_sources.team_atmospheres import SITES as TEAM_SITES

        if spec.site in SITES:
            if spec.pwv is None or spec.airmass is None:
                raise ComponentError(f"{spec.site}: specify airmass and pwv to generate a SkyCalc atmosphere")
            get_builder("skycalc")(self.root, atmosphere_root=self.atmosphere_root,
                                   sites=[spec.site], airmasses=[spec.airmass], pwvs=[spec.pwv])
        elif spec.site in TEAM_SITES:
            # A team's published atmosphere: rebuild the whole site from its source.
            get_builder("team_atmospheres")(self.root, atmosphere_root=self.atmosphere_root,
                                            sites=[spec.site], overwrite=True)
        else:
            ref = builder_for(manifest.facility, manifest.instrument)
            if ref is not None and (allow_heavy or not ref.heavy):
                get_builder(ref.name)(self.root, **dict(ref.kwargs))
        return atmosphere_file(self.atmosphere_root, spec)

    def atmospheres(self, *, remote=True):
        sites = {p.parent.name for p in self.atmosphere_root.glob("*/manifest.json")}
        if remote:
            for site in self.mirror.list_atmosphere_sites():
                if site not in sites and self.mirror.download_atmosphere(str(self.atmosphere_root), site):
                    sites.add(site)
        result = {}
        for site in sorted(sites):
            grid = AtmosphereGrid.load(self.atmosphere_root / site)
            result[site] = {
                "altitude_m": grid.altitude_m,
                "airmass": sorted({float(e["airmass"]) for e in grid.entries}),
                "pwv": sorted({float(e["pwv"]) for e in grid.entries if e.get("pwv") is not None}),
                "unstated_airmass": bool(grid.unstated),
                "source": grid.source,
                "kind": grid.kind,
                "model": grid.model,
                "model_version": grid.model_version,
                "citation": grid.citation,
            }
        return result

    def compose(self, instrument, components, *, atmosphere=None, bands=None,
                detector_type=None, allow_heavy=False, allow_partial=False, variants=None):
        directory = self.directory(instrument)
        if not list(directory.glob("*.dat")):
            directory = self.published(directory.parent.name, directory.name)
        roles = parse_components(components)
        self.prepare_components(directory, components=roles, bands=bands, allow_heavy=allow_heavy)
        spec = AtmosphereSpec.coerce(atmosphere)
        if spec is not None:
            manifest = InstrumentManifest.load(directory)
            self.prepare_atmosphere(manifest, spec, allow_heavy=allow_heavy)
        return build_product(directory, roles, atmosphere=spec, atmosphere_root=self.atmosphere_root,
                             bands=bands, detector_type=detector_type, allow_partial=allow_partial,
                             variants=variants)

    @staticmethod
    def _catalogue_key(name):
        return re.sub(r"[^a-z0-9]", "", name.casefold())

    @classmethod
    def _catalogue_add(cls, entries, key, label, source):
        identity = cls._catalogue_key(key)
        entry = entries.setdefault(identity, {"key": key, "label": label, "sources": []})
        if source == "svo":
            entry.update(key=key, label=label)
        if source not in entry["sources"]:
            entry["sources"].append(source)

    def facilities(self, *, remote=True):
        """Discover the union of installed, registered, SVO and NJM facilities."""
        entries, errors = {}, []
        for directory in sorted(self.root.glob("*")):
            if directory.is_dir() and any(directory.glob("*/*.dat")):
                self._catalogue_add(entries, directory.name, directory.name, "local")
        for facility, _ in BUILDERS_FOR:
            self._catalogue_add(entries, facility, facility, "original")
        if remote:
            try:
                facilities = self.svo.list_facilities()
                if not facilities:
                    raise ComponentError("SVO returned no facilities")
                for facility in facilities:
                    self._catalogue_add(entries, facility.key, facility.label, "svo")
            except (ComponentError, RequestException, RuntimeError) as exc:
                errors.append(f"SVO: {exc}")
            if self.mirror.is_available():
                for facility in self.mirror.discover_facilities():
                    self._catalogue_add(entries, facility, facility, "njm")
            else:
                errors.append("NJM unavailable")
        return {"entries": sorted(entries.values(), key=lambda e: e["key"].casefold()), "errors": errors}

    def instruments(self, facility, *, remote=True):
        """Discover every instrument under one facility without downloading curves."""
        entries, errors = {}, []
        same_facility = lambda name: self._catalogue_key(name) == self._catalogue_key(facility)
        for directory in sorted(self.root.glob("*/*")):
            if same_facility(directory.parent.name) and directory.is_dir() and list(directory.glob("*.dat")):
                self._catalogue_add(entries, directory.name, directory.name, "local")
        for fac, instrument in BUILDERS_FOR:
            if same_facility(fac):
                self._catalogue_add(entries, instrument, instrument, "original")
        if remote:
            try:
                fac = self.svo_facility(facility)
                if fac is not None:
                    instruments = self.svo.list_instruments(fac.key)
                    if not instruments:
                        raise ComponentError("No instruments returned for a listed SVO facility")
                    for instrument in instruments:
                        self._catalogue_add(entries, instrument.key, instrument.label, "svo")
            except (ComponentError, RequestException, RuntimeError) as exc:
                errors.append(f"SVO {facility}: {exc}")
            if self.mirror.is_available():
                fac = next((name for name in self.mirror.discover_facilities() if same_facility(name)), None)
                if fac:
                    for instrument in self.mirror.discover_instruments(fac):
                        self._catalogue_add(entries, instrument, instrument, "njm")
        return {"entries": sorted(entries.values(), key=lambda e: e["key"].casefold()), "errors": errors}

    def inventory(self, *, remote=True, facility=None):
        """List every discovered instrument; retain errors and unreviewed sources."""
        catalogue = self.facilities(remote=remote)
        rows, errors = [], list(catalogue["errors"])
        for fac in catalogue["entries"]:
            if facility and self._catalogue_key(facility) not in (
                    self._catalogue_key(fac["key"]), self._catalogue_key(fac["label"])):
                continue
            source_catalogue = any(s in fac["sources"] for s in ("svo", "njm"))
            result = self.instruments(fac["key"], remote=remote and source_catalogue)
            errors.extend(result["errors"])
            for inst in result["entries"]:
                rows.append({"facility": fac["key"], "instrument": inst["key"],
                             "sources": inst["sources"],
                             "component_availability": component_availability(fac["key"], inst["key"])})
        return {"instruments": rows, "errors": errors}
