"""Shared filter acquisition and composition workflow for the API and CLI.

The library owns source selection: installed files, NJM, then original
sources. Manifests describe capabilities; filter_components performs the
numerical work. Source catalogues remain data-only descriptions.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
import time
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Dict, List

from requests import RequestException

from .component_sources import get_builder
from .component_sources.catalog import (
    BUILDERS_FOR, builder_for, component_availability, match_name, name_key, registered_name,
)
from .filter_components import (
    FILTER, OPTICS, QE, PUBLISHED, AtmosphereGrid, AtmosphereSpec, ComponentError,
    InstrumentManifest, atmosphere_file, build_product, find_atmosphere, index_listing, parse_components,
    recipe_name, resolve_atmosphere,
)
from .filter_catalogue import cache_root
from .filter_metadata import SVOUnreachableError, migrate_filter_tree, svo_components_header
from .njm_filter_grabber import NJMFilterGrabber, _status, curve_file
from .svo_filter_grabber import FilterRow, SVOFilterBrowser, _clean_filename, _clean_path, svo_unavailable

logger = logging.getLogger(__name__)

#: Whose spelling of a name the picker shows and downloads with, when several sources list it.
FETCH_ORDER = ("njm", "original", "local", "svo")
#: Seconds between checks of an installed atmosphere site's manifest against NJM's.
ATMOSPHERE_CHECK_AGE = 24 * 3600
#: Why NJM gave nothing while it is down (the NJM client announces each outage once per process).
NJM_UNAVAILABLE = "the mirror is unavailable"
_NJM_HAS_NO_COMPONENTS = "the mirror does not have them"
_COMPONENTS_ONLY = "its builder makes only the separate components, not the published curves"


@dataclass
class _SVOPlan:
    """How SVO fills a product of NJM's (see :meth:`FilterLibrary._svo_plan`)."""

    facility: str  # NJM's names
    instrument: str
    bands: List[str]  # every band NJM serves
    rows: Dict[str, FilterRow]  # NJM band -> the SVO filter installed as it
    differs: List[str]  # bands taken from SVO although NJM's current curve differs
    svo_instruments: List[str]  # the SVO instruments NJM names as the same product


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
        """SVO's filters of an instrument, as :class:`FilterRow` s.

        For a product of NJM's that SVO holds under other names (the bundled
        catalogue's ``svo_bands``: NJM's ``GAIA/GAIA`` is SVO's ``GAIA/GAIA3``),
        the rows of the SVO instruments that hold its bands, renamed to NJM's
        facility, instrument and band names (see :meth:`_svo_plan`).
        Otherwise SVO's instrument of that name, matched ignoring case and
        punctuation.
        """
        plan = self._svo_plan(facility, instrument)
        if plan is not None:
            return list(plan.rows.values())
        return self._svo_own_filters(facility, instrument)

    def _svo_own_filters(self, facility, instrument):
        """SVO's instrument of this name (user-facing names resolved to SVO keys)."""
        fac = self.svo_facility(facility)
        if fac is None:
            raise ComponentError(f"SVO facility not found: {facility}")
        instruments = self.svo.list_instruments(fac.key)
        key = match_name(instrument, [i.key for i in instruments])
        if key is None:
            label = match_name(instrument, [i.label for i in instruments])
            key = next((i.key for i in instruments if i.label == label), None)
        if key is None:
            raise ComponentError(f"SVO instrument not found: {facility}/{instrument}")
        return self.svo.list_filters(fac.key, key)

    def svo_facility(self, name):
        identity = name_key(name)
        return next((fac for fac in self.svo.list_facilities() if identity in
                     (name_key(fac.key), name_key(fac.label))), None)

    @staticmethod
    def published_complete(directory):
        """Whether an installed instrument holds every curve it declares.

        Every curve its MESA index file lists must be present, and every band
        its manifest declares except those under ``archived_bands`` (superseded
        copies NJM keeps beside the set but never serves as part of it).  The
        manifest counts as well because a download that failed part-way leaves
        an index of only the curves it saved beside a manifest of the bands it
        expected.
        """
        directory = Path(directory)
        if not list(directory.glob("*.dat")) or not (directory / "manifest.json").is_file():
            return False
        expected = {f"{band}.dat" for band in _declared_bands(directory)}
        expected.update(index_listing(directory) or [])
        return bool(expected) and all(Path(name).name == name and (directory / name).is_file()
                                      for name in expected)

    def installed_directory(self, facility, instrument):
        """The installed folder of an instrument, whatever its spelling, or None when there is none.

        Folders holding curves are matched ignoring case and punctuation, so
        that NJM's ``Gemini/GMOS_N`` or ``Planck/HFI`` finds a ``Gemini/GMOS-N``
        or ``PLANCK/HFI`` installed earlier and no second copy is made beside
        it.  A folder named exactly as another product is never taken for this
        one (NJM serves both ``Cameras/Sony_NEX5N`` and ``Cameras/Sony_NEX_5N``).
        When several folders match, the one spelled exactly as asked is used;
        otherwise ComponentError lists them.
        """
        key = (name_key(facility), name_key(instrument))
        others = self._other_products(facility, instrument)
        found = [d for d in sorted(self.root.glob("*/*"))
                 if (name_key(d.parent.name), name_key(d.name)) == key and d.name not in others
                 and d.is_dir() and any(d.glob("*.dat"))]
        if len(found) <= 1:
            return found[0] if found else None
        exact = [d for d in found if (d.parent.name, d.name) == (facility, instrument)]
        if len(exact) == 1:
            return exact[0]
        raise ComponentError(f"{facility}/{instrument} is installed more than once, under different spellings: "
                             f"{', '.join(str(d) for d in found)}. Keep one of them (move the others out of "
                             f"{self.root}) and run again.")

    def _other_products(self, facility, instrument):
        """Other instruments of the facility (NJM's or a builder's) whose names differ from this one's only
        in case or punctuation; empty unless ``instrument`` is itself one of those names."""
        fac = match_name(facility, self._bundled)
        known = set(self._bundled[fac]["instruments"]) if fac else set()
        known.update(inst for fac_name, inst in BUILDERS_FOR if name_key(fac_name) == name_key(facility))
        if instrument not in known:
            return set()
        return {name for name in known if name != instrument and name_key(name) == name_key(instrument)}

    def published(self, facility, instrument):
        """The directory of the instrument's complete published curves, downloading them if needed.

        Sources, in order: an installed complete copy, NJM, the instrument's
        original-source builder, SVO.  NJM and the builder are asked under
        their own spelling of the name, matched ignoring case and punctuation
        (``CTIO/S-PLUS`` is NJM's ``CTIO/S_PLUS``).  An instrument installed
        under another spelling is completed in that folder
        (:meth:`installed_directory`).  A product of NJM's that SVO holds under
        other names is taken from those SVO instruments, under NJM's names
        (:meth:`_svo_plan`).  Each source that cannot give a complete set is
        reported with a warning naming the next one tried; when none can,
        ComponentError carries every source's reason.  When SVO holds only
        some of the bands of NJM's product, those are installed and returned,
        with one warning naming the bands missing; the manifest still declares
        every band, so the set is not taken as complete and the next fetch
        asks NJM again (``Filters.available`` gives ``expected_bands`` beside
        the installed ``bands``).  Once NJM has put part of its set in place,
        SVO is not tried: its curves would replace or mix with NJM's.
        """
        installed = self.installed_directory(facility, instrument)
        if installed is not None and self.published_complete(installed):
            # Sets downloaded by SED_Tools 0.3.0 carry metadata on leading '#' lines.
            migrate_filter_tree(installed)
            return installed

        njm_directory, reason = self._published_from_njm(facility, instrument, installed)
        if reason is None:
            return njm_directory
        reasons = [f"NJM: {reason}"]
        njm_filled = njm_directory is not None
        builder, built, why_not = self._publishing_builder(facility, instrument, installed)
        if why_not:
            reasons.append(f"original source: {why_not}")
        beside = why_not if why_not != _COMPONENTS_ONLY else None  # a builder skipped to avoid a second copy
        if builder is not None:
            next_source = "its original-source builder"
        else:
            next_source = None if njm_filled else self._svo_next()
            if next_source and beside:
                next_source += f" (the original source is not used: {beside})"
        # An NJM outage is announced once per process by the NJM client, not again for each instrument.
        if next_source and (beside or not (njm_directory is None and _njm_outage(reason))):
            logger.warning("NJM cannot provide %s/%s (%s); trying %s", facility, instrument, reason, next_source)
        if builder is not None:
            reason = self._published_from_builder(builder, built)
            if reason is None:
                return built
            reasons.append(f"original source: {reason}")
            next_source = None if njm_filled else self._svo_next()
            if next_source:
                logger.warning("The original source cannot provide %s/%s (%s); trying %s",
                               facility, instrument, reason, next_source)
        if njm_filled:
            raise ComponentError(f"NJM provided only part of {facility}/{instrument}; run again to fetch the rest. "
                                 f"SVO is not used, so that its curves never replace or mix with NJM's in "
                                 f"{njm_directory}: " + "; ".join(reasons))
        try:
            directory, shortfall = self._published_from_svo(facility, instrument, installed)
        except (ComponentError, RequestException, RuntimeError) as exc:
            reasons.append(f"SVO: {exc}")
            raise ComponentError(f"No source can provide {facility}/{instrument}: " + "; ".join(reasons)) from exc
        if shortfall is not None:
            text = (f"{facility}/{instrument} is incomplete: {shortfall}. Fetch again when NJM is reachable to "
                    f"complete it ({'; '.join(reasons)}).")
            # One warning, wrapped to 80 columns, with the folder on a line of its own
            logger.warning("%s\nInstalled in: %s",
                           textwrap.fill(text, width=80, break_long_words=False, break_on_hyphens=False),
                           directory)
        return directory

    def _publishing_builder(self, facility, instrument, installed):
        """``(builder, the directory it writes, None)``, or ``(None, None, why it is not used or None)``.

        A builder that makes only the separate components is not a source of
        the published curves.  Nor is one that would write a second copy beside
        an instrument installed under another spelling.
        """
        names = registered_name(facility, instrument)
        if names is None:
            return None, None, None
        builder, built = BUILDERS_FOR[names], self.root / names[0] / names[1]
        if not builder.publishes:
            return None, None, _COMPONENTS_ONLY
        if installed is not None and not _same_directory(installed, built):
            return None, None, f"its builder writes {names[0]}/{names[1]}, which would be a second copy beside {installed}"
        return builder, built, None

    @staticmethod
    def _svo_next():
        """"SVO" as the next source to try, or None when SVO did not answer lately (its error ends the run)."""
        return None if svo_unavailable() else "SVO"

    def _published_from_njm(self, facility, instrument, installed=None):
        """Download the set from NJM, into ``installed`` when the instrument is installed under another spelling.

        Returns ``(directory, None)`` once every curve NJM's index lists is
        installed and matches NJM's copy, else ``(directory holding curves NJM
        wrote or confirmed now, or None, why the set is incomplete)``.
        """
        if not self.mirror.is_available():
            return None, NJM_UNAVAILABLE
        spellings = [(facility, instrument)]
        njm_names = self._njm_spelling(facility, instrument)
        if njm_names is not None and njm_names != (facility, instrument):
            spellings.insert(0, njm_names)
        for fac, inst in spellings:
            directory = installed if installed is not None else self.root / fac / inst
            before = _curve_state(directory)
            partial_manifest = _partial_fill_manifest(directory)
            try:
                result = (self.mirror._download_set(fac, inst) if installed is None
                          else self.mirror._download_set(fac, inst, output_dir=installed))
            except (RequestException, OSError, RuntimeError, ValueError) as exc:
                return (directory if _curve_state(directory) != before else None), str(exc)
            if result.listed:
                # NJM writes its own manifest only when it serves one
                manifest_kept = partial_manifest is not None and _manifest_text(directory) == partial_manifest
                reason = self._njm_shortfall(fac, inst, directory, result)
                if reason is None and manifest_kept:
                    _describe_completed_set(directory)
                filled = reason is None or result.downloaded + result.verified > 0  # NJM's curves are there now
                return (directory if filled else None), reason
        return None, "it does not serve this instrument"

    def _njm_shortfall(self, facility, instrument, directory, result):
        """Why an NJM download left the set incomplete, or None when it is complete."""
        missing = [name for name in result.listed if not (directory / curve_file(name)).is_file()]
        if result.curve_failures:
            return (f"{len(result.curve_failures)} of {len(result.listed)} curves could not be downloaded or "
                    "checked: " + "; ".join(result.curve_failures))
        if missing:
            return "the mirror has no complete set"
        if result.other_failures:
            # Every curve arrived; only an extra (components, say) failed, and is fetched again when needed.
            logger.warning("NJM provided every curve of %s/%s, but: %s", facility, instrument,
                           "; ".join(result.other_failures))
        _drop_unlisted_bands(directory, {curve_file(name) for name in result.listed})
        return None

    def _published_from_builder(self, ref, directory):
        """Build the set from its original source; ``None`` once it is complete, else why it is not."""
        try:
            get_builder(ref.name)(self.root, **dict(ref.kwargs))
        except (RequestException, OSError, RuntimeError, ValueError) as exc:
            return str(exc)
        return None if self.published_complete(directory) else "the builder left an incomplete set"

    def _published_from_svo(self, facility, instrument, installed=None):
        """Download the set from SVO: ``(directory, None)`` when complete, ``(directory, what is missing)``
        when SVO holds only some of the bands of one of NJM's products; raises when SVO cannot provide it,
        or a curve it has could not be saved."""
        plan = self._svo_plan(facility, instrument)
        if plan is not None:
            return self._njm_product_from_svo(plan, installed)
        rows = self._svo_own_filters(facility, instrument)
        if not rows:
            raise ComponentError(f"SVO returned no filters for {facility}/{instrument}")
        directories = {self.root / _clean_path(row.facility) / _clean_path(row.instrument) for row in rows}
        elsewhere = installed is not None and any(not _same_directory(installed, d) for d in directories)
        failure = None
        try:
            if elsewhere:  # into the folder installed under another spelling, not beside it
                self.svo.download_into(installed, {_clean_filename(row.band): row for row in rows})
            else:
                self.svo.download_filters(rows)
        except SVOUnreachableError:
            raise
        except RuntimeError as exc:  # some curves could not be saved; their bands stay expected
            failure = exc
        if failure is None:
            for candidate in [installed or self.root / facility / instrument, *sorted(directories)]:
                if self.published_complete(candidate):
                    return candidate, None
        raise ComponentError(f"Incomplete passband download for {facility}/{instrument}; retry to recover "
                             "missing files" + (f" ({failure})" if failure else "")) from failure

    def _svo_plan(self, facility, instrument):
        """How SVO can fill a product of NJM's that it holds under other names, or None.

        Uses the bundled catalogue: the SVO instruments NJM names as the same
        product (``svo_instruments``) are listed, and each filter that holds one
        of NJM's bands (``svo_bands``) is taken under that band's name (SVO's
        ``GAIA/GAIA2.Gbp`` is NJM's ``GAIA/DR2/BP``).  A filter of theirs whose
        name is one of NJM's bands that no filter holds is taken as SVO
        publishes it, since NJM's current curve differs (``differs``).  A
        claimed SVO instrument SVO does not list is skipped, so its bands are
        missing; ComponentError when SVO lists none of them.
        """
        names = self._njm_spelling(facility, instrument)
        meta = self._bundled_record(facility, instrument)
        if meta is None or not meta["svo_bands"]:
            return None
        rows, others, unlisted = {}, [], []
        for claim in meta["svo_instruments"]:
            try:
                listed = self._svo_own_filters(*claim.split("/", 1))
            except ComponentError as exc:
                unlisted.append(f"{claim}: {exc}")
                continue
            for row in listed:
                band = meta["svo_bands"].get(row.filter_id)
                if band is None:
                    others.append(row)
                else:
                    rows.setdefault(band, replace(row, facility=names[0], instrument=names[1], band=band))
        by_name = {band.casefold(): band for band in meta["bands"]}
        differs = []
        for row in others:
            band = by_name.get(row.band.casefold())
            if band is not None and band not in rows:
                rows[band] = replace(row, facility=names[0], instrument=names[1], band=band)
                differs.append(band)
        if not rows:
            raise ComponentError(f"SVO lists none of the filters of {names[0]}/{names[1]}'s SVO equivalents "
                                 f"({', '.join(meta['svo_instruments'])})" + (f": {'; '.join(unlisted)}" if unlisted
                                                                              else ""))
        return _SVOPlan(names[0], names[1], list(meta["bands"]), rows, differs, list(meta["svo_instruments"]))

    def _njm_product_from_svo(self, plan, installed):
        """Install a product of NJM's from SVO under NJM's names.

        Returns ``(directory, None)`` when every band is installed, else
        ``(directory, what is missing)``: the bands SVO has no curve of.  The
        bands taken from SVO although NJM's current curve differs are named in
        that text, or in a warning of their own when nothing is missing.
        ComponentError, once the others are saved, when a curve SVO has could
        not be saved.
        """
        directory = installed if installed is not None else self.root / plan.facility / plan.instrument
        try:
            self.svo.download_into(directory, plan.rows, expected=plan.bands)
        except SVOUnreachableError:
            raise
        except RuntimeError as exc:  # some curves could not be saved; every band stays expected
            have = [band for band in plan.bands if (directory / f"{band}.dat").is_file()]
            raise ComponentError(f"{plan.facility}/{plan.instrument} is only partly installed, in {directory}: "
                                 f"{len(have)} of NJM's {len(plan.bands)} bands came from SVO's "
                                 f"{', '.join(plan.svo_instruments)} ({', '.join(have) or 'none'}); {exc}") from exc
        missing = [band for band in plan.bands if not (directory / f"{band}.dat").is_file()]
        differs = (f"SVO's {', '.join(plan.differs)} of {plan.facility}/{plan.instrument} differ from NJM's current "
                   "curves; they are installed as SVO publishes them, and replaced by NJM's (with a warning) when "
                   "it is fetched from NJM") if plan.differs else ""
        if not missing:
            if differs:
                logger.warning("%s", differs)
            return directory, None
        return directory, (f"{len(plan.bands) - len(missing)} of NJM's {len(plan.bands)} bands were installed from "
                           f"SVO's {', '.join(plan.svo_instruments)}; missing {', '.join(missing)}, which SVO "
                           "does not have" + (f". {differs}" if differs else ""))

    def available(self, facility, instrument, *, remote=True):
        """What can be requested for an instrument, from the first source that can say.

        Besides the description it has ``availability_source`` (``local``,
        ``njm``, ``original``, ``svo`` or ``unknown``), ``errors`` (each
        failure as ``"NJM: ..."`` or ``"SVO: ..."``), ``sources_failed``
        (``{"njm": [...], "svo": [...]}``, the same failures by source) and
        ``fallback_from`` (the sources whose failure only made a later one
        answer: they leave nothing out).  An instrument installed under another
        spelling is described from that folder (:meth:`installed_directory`).
        When no source can say, the bands are those of NJM's catalogue bundled
        with SED_Tools, if it lists the instrument (with a note).
        """
        directory = self.installed_directory(facility, instrument) or self.root / facility / instrument
        if (directory / "manifest.json").is_file() or list(directory.glob("*.dat")):
            return _availability(InstrumentManifest.load(directory).describe(directory), "local", {})
        failed = {}
        if remote:
            try:
                info = self._njm_availability(facility, instrument)
            except (RequestException, ValueError) as exc:
                failed.setdefault("njm", []).append(str(exc))
            else:
                if info is not None:
                    return info
        ref = builder_for(facility, instrument)
        if ref is not None and ref.name == "tabular":
            from .component_sources.tabular import SOURCES

            spec = SOURCES[dict(ref.kwargs)["instrument"]]
            headers = {band: {"Components": FILTER, "ComponentsSource": "original",
                              "DetectorType": spec.detector_type,
                              "DetectorTypeSource": "assumed"} for band in spec.filters}
            manifest = InstrumentManifest.from_headers(facility, instrument, headers)
            return _availability(manifest.describe(), "original", failed)
        builder, _, _ = self._publishing_builder(facility, instrument, None)
        if builder is not None and (self._bundled_record(facility, instrument) or {}).get("bands"):
            # published() would build it before asking SVO: its bands as NJM lists them
            return self._bundled_availability(facility, instrument, failed, source="original")
        if remote:
            try:
                info = self._svo_availability(facility, instrument, failed)
            except (ComponentError, RequestException, RuntimeError) as exc:
                failed.setdefault("svo", []).append(str(exc))
            else:
                if info is not None:
                    return info
        return self._bundled_availability(facility, instrument, failed)

    def _svo_availability(self, facility, instrument, failed):
        """What SVO has of an instrument, described from its metadata, or None if it lists no filters.

        A product of NJM's is described under NJM's band names, with a note
        naming NJM's bands SVO has no copy of (see :meth:`_svo_plan`).
        """
        headers = {}
        for row in self.svo_filters(facility, instrument):
            record = self.svo.metadata_client.get_filter(row.filter_id)
            if record is None:
                failed.setdefault("svo", []).append(f"no metadata for {row.filter_id}")
                headers[row.band] = {}
            else:
                headers[row.band] = dict(svo_components_header(record.components),
                                         DetectorType=record.detector_type or "",
                                         DetectorTypeSource="svo" if record.detector_type else "assumed")
        if not headers:
            return None
        info = _availability(InstrumentManifest.from_headers(facility, instrument, headers).describe(), "svo", failed)
        meta = self._bundled_record(facility, instrument)
        absent = [band for band in meta["bands"] if band not in headers] if meta and meta["svo_bands"] else []
        if absent:
            info["notes"] = [f"SVO has no curve that is NJM's {', '.join(absent)}"]
        return info

    def _bundled_availability(self, facility, instrument, failed, source="unknown"):
        """The bands NJM served when this SED_Tools was released, else none: no source could say
        (``source="unknown"``), or the original-source builder makes them when fetched (``"original"``)."""
        meta = self._bundled_record(facility, instrument)
        bands = meta["bands"] if meta else []
        manifest = InstrumentManifest.from_headers(facility, instrument, {band: {} for band in bands})
        info = _availability(manifest.describe(), source, failed)
        # No curve has been seen, so the detector convention is unknown; the empty headers would read as
        # energy, which the curves NJM, a builder or SVO then write need not be (GAIA/DR2's are photon).
        info["detector_type"] = None
        info["detector_type_sources"] = {band: "unknown" for band in bands}
        notes = ["its original-source builder makes the curves when it is fetched"] if source == "original" else []
        if bands:
            notes.append("bands as NJM served them when this SED_Tools was released; what each curve contains "
                         "is known once " + ("they are built" if source == "original" else
                                             "NJM or another source can be asked"))
        if notes:
            info["notes"] = notes
        return info

    def _njm_availability(self, facility, instrument):
        """What NJM serves for an instrument (under its name as given, else NJM's spelling), or None.

        An instrument NJM serves without a component manifest (GAIA/GAIA, say)
        is described from the curve headers its ``info.json`` records, with a
        note in ``notes`` rather than an error.
        """
        spellings = [(facility, instrument)]
        njm_names = self._njm_spelling(facility, instrument)
        if njm_names is not None and njm_names != (facility, instrument):
            spellings.append(njm_names)
        for fac, inst in spellings:
            data = self.mirror.fetch_manifest(fac, inst)
            if data:
                return _availability(InstrumentManifest.from_dict(data).describe(), "njm", {})
            listed = self.mirror.discover_filters(fac, inst)
            if listed:
                return self._njm_without_manifest(fac, inst, listed)
        return None

    def _njm_without_manifest(self, facility, instrument, listed):
        """Describe an instrument NJM serves without ``manifest.json``, from its ``info.json``."""
        bands = [curve_file(name)[:-len(".dat")] for name in listed]
        headers = {band: {} for band in bands}
        failed = {}
        try:
            info = json.loads(self.mirror._catalogue_file(facility, instrument, "info.json"))
        except (RequestException, ValueError) as exc:
            if _status(exc) != 404:
                failed["njm"] = [str(exc)]
        else:
            for band in bands:
                curve = (info["bands"].get(band) or {}).get("curve")
                if isinstance(curve, dict):
                    headers[band] = {key: value for key, value in curve.items() if isinstance(value, str)}
        manifest = InstrumentManifest.from_headers(facility, instrument, headers)
        return dict(_availability(manifest.describe(), "njm", failed),
                    notes=["no component manifest: NJM serves only the published curves, described "
                           "here from their headers"])

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
        njm_facility, njm_instrument = self._njm_spelling(facility, instrument) or (facility, instrument)
        reason = NJM_UNAVAILABLE
        if self.mirror.is_available():
            try:
                if (njm_facility, njm_instrument) == (facility, instrument):
                    self.mirror.download_components(facility, instrument)
                else:  # NJM's spelling, into the folder installed under another one
                    self.mirror._save_components(njm_facility, njm_instrument, directory)
            except (RequestException, RuntimeError, ValueError) as exc:
                reason = str(exc)
            else:
                if self.components_ready(directory, components, bands):
                    return True
                reason = _NJM_HAS_NO_COMPONENTS
        announce = not _njm_outage(reason)  # an outage is announced once per process by the NJM client
        ref = builder_for(facility, instrument)
        if ref is None or not ref.components:
            if announce and reason != _NJM_HAS_NO_COMPONENTS:  # for 'none', build_product says what is missing
                logger.warning("NJM cannot provide the components of %s/%s (%s), and no original source is "
                               "registered", facility, instrument, reason)
            return False
        names = registered_name(facility, instrument)
        if not _same_directory(directory, self.root / names[0] / names[1]):
            logger.warning("NJM cannot provide the components of %s/%s (%s), and its original-source builder "
                           "writes only %s/%s, which would be a second copy beside %s", facility, instrument,
                           reason, names[0], names[1], directory)
            return False
        if ref.heavy and not allow_heavy:
            logger.warning("%s/%s requires a large source download; use allow_heavy=True or --allow-heavy",
                           facility, instrument)
            return False
        if announce:
            logger.warning("NJM cannot provide the components of %s/%s (%s); trying their original source",
                           facility, instrument, reason)
        get_builder(ref.name)(self.root, **dict(ref.kwargs))
        return self.components_ready(directory, components, bands)

    def prepare_atmosphere(self, manifest, spec, *, allow_heavy=False):
        spec = resolve_atmosphere(manifest, spec)
        cached = find_atmosphere(self.atmosphere_root, spec)
        if cached is not None:
            return cached
        try:
            cached = self._atmosphere_from_njm(spec)
        except RequestException as exc:
            if not _njm_outage(str(exc)):
                logger.warning("NJM cannot provide the %s atmosphere (%s); trying its source", spec.label, exc)
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

    def _atmosphere_from_njm(self, spec):
        """The requested grid point downloaded from NJM, or None when NJM does not have it."""
        if not self.mirror.download_atmosphere(str(self.atmosphere_root), spec.site):
            return None
        filename = AtmosphereGrid.load(self.atmosphere_root / spec.site).find(spec)
        if filename is None:
            return None
        self.mirror.download_atmosphere(str(self.atmosphere_root), spec.site, filename)
        return find_atmosphere(self.atmosphere_root, spec)

    def atmospheres(self, *, remote=True):
        """The installed atmosphere sites and, with ``remote``, NJM's, by site.

        With ``remote`` an installed site's manifest is checked against NJM's
        once a day and updated (its grid files are kept); an unreachable NJM
        leaves the installed sites, with a warning.
        """
        sites = {p.parent.name for p in self.atmosphere_root.glob("*/manifest.json")}
        if remote:
            self._atmospheres_from_njm(sites)
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

    def _atmospheres_from_njm(self, sites):
        """Add NJM's sites to ``sites``, downloading their manifests; check installed ones once a day."""
        checked = cache_root(self.root) / "atmospheres_checked.json"
        try:
            when = json.loads(checked.read_text(encoding="utf-8")) if checked.is_file() else {}
        except ValueError:
            when = {}  # rewritten below
        try:
            for site in self.mirror.list_atmosphere_sites():
                if site in sites and time.time() - float(when.get(site, 0)) < ATMOSPHERE_CHECK_AGE:
                    continue
                if self.mirror.download_atmosphere(str(self.atmosphere_root), site):
                    sites.add(site)
                    when[site] = time.time()
        except (RequestException, ValueError) as exc:
            if not _njm_outage(str(exc)):
                logger.warning("NJM's atmospheres could not be listed or checked (%s); the installed ones are used",
                               exc)
        checked.parent.mkdir(parents=True, exist_ok=True)
        checked.write_text(json.dumps(when, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    def compose(self, instrument, components, *, atmosphere=None, bands=None,
                detector_type=None, allow_heavy=False, allow_partial=False, variants=None):
        directory = self.directory(instrument)
        if not list(directory.glob("*.dat")):
            directory = self.published(directory.parent.name, directory.name)
        roles = parse_components(components)
        # Before any source is asked for components a band could never use.
        manifest = InstrumentManifest.load(directory)
        for band in bands or ():
            if band not in manifest.bands:
                raise ComponentError(f"{manifest.facility}/{manifest.instrument} has no band '{band}'")
        _refuse_uncovered_bands(manifest, roles, bands)
        self.prepare_components(directory, components=roles, bands=bands, allow_heavy=allow_heavy)
        spec = AtmosphereSpec.coerce(atmosphere)
        if spec is not None:
            manifest = InstrumentManifest.load(directory)
            self.prepare_atmosphere(manifest, spec, allow_heavy=allow_heavy)
        return build_product(directory, roles, atmosphere=spec, atmosphere_root=self.atmosphere_root,
                             bands=bands, detector_type=detector_type, allow_partial=allow_partial,
                             variants=variants)

    # ------------------------------------------------------------------
    # Discovery: what the facility and instrument pickers offer
    # ------------------------------------------------------------------
    @cached_property
    def _bundled(self):
        """NJM's names, descriptions and SVO equivalents as shipped (see :func:`load_instrument_catalogue`)."""
        return load_instrument_catalogue()

    def _njm_spelling(self, facility, instrument):
        """NJM's spelling of an instrument from the bundled catalogue (``CTIO/S-PLUS`` -> ``CTIO/S_PLUS``), or None."""
        fac = match_name(facility, self._bundled)
        inst = match_name(instrument, self._bundled[fac]["instruments"]) if fac else None
        return (fac, inst) if inst else None

    def _bundled_record(self, facility, instrument):
        """The bundled catalogue's record of an instrument (names matched as :meth:`_njm_spelling` does), or None."""
        names = self._njm_spelling(facility, instrument)
        return self._bundled[names[0]]["instruments"][names[1]] if names else None

    def _njm_instruments(self, facility, live):
        """NJM's instruments of a facility -> ``{"description", "svo_instruments"}``.

        The bundled catalogue, with each field a reachable NJM's ``index.json``
        gives (``live``: name -> record) added or overriding it.
        """
        fac = match_name(facility, self._bundled)
        known = {name: dict(meta) for name, meta in (self._bundled[fac]["instruments"] if fac else {}).items()}
        for name, record in live.items():
            fresh = _instrument_metadata(record)
            known[name] = dict(known.get(name, fresh), **{k: v for k, v in fresh.items() if k in record})
        return known

    def facilities(self, *, remote=True):
        """Discover the union of installed, registered, SVO and NJM facilities.

        Returns ``{"entries": [...], "errors": [...]}``; :func:`_picker_entries`
        describes an entry.  NJM is not needed: without it the other sources
        are listed, named and described as when it is reachable.
        """
        listings, errors = [], []
        listings.append(("local", [(d.name, "") for d in sorted(self.root.glob("*"))
                                   if d.is_dir() and any(d.glob("*/*.dat"))]))
        listings.append(("original", [(facility, "") for facility, _ in _publishing_builders()]))
        if remote:
            try:
                facilities = self.svo.list_facilities()
            except (ComponentError, RequestException, RuntimeError) as exc:
                errors.append(f"SVO: {exc}")
            else:
                if facilities:
                    listings.append(("svo", [(f.key, f.description) for f in facilities]))
                else:
                    errors.append("SVO: SVO returned no facilities")
            if self.mirror.is_available():
                try:
                    listings.append(("njm", [(name, "") for name in self.mirror.discover_facilities()]))
                except RequestException as exc:
                    errors.append(f"NJM: {exc}")
            else:
                errors.append("NJM unavailable")
        return {"entries": _picker_entries(_merge(listings), self._bundled), "errors": errors}

    def instruments(self, facility, *, remote=True):
        """Discover every instrument under one facility without downloading curves.

        Returns ``{"entries": [...], "errors": [...]}`` as :meth:`facilities`
        does.  An SVO instrument that NJM lists as the same product as one of
        its own (``svo_instruments``) is listed once, under NJM's name: while
        NJM is reachable NJM's owner-sourced copy is offered instead, and while
        it is not, SVO's copy is offered under NJM's name (see
        :func:`_adopt_claimed_svo`).  Builders that make only the separate
        components are not listed: they cannot provide the curves.
        """
        same_facility = lambda name: name_key(name) == name_key(facility)
        listings, errors, live = [], [], {}
        listings.append(("local", [(d.name, "") for d in sorted(self.root.glob("*/*"))
                                   if same_facility(d.parent.name) and d.is_dir() and list(d.glob("*.dat"))]))
        listings.append(("original", [(inst, "") for fac, inst in _publishing_builders() if same_facility(fac)]))
        if remote:
            try:
                fac = self.svo_facility(facility)
                instruments = self.svo.list_instruments(fac.key) if fac is not None else None
            except SVOUnreachableError as exc:
                errors.append(f"SVO: {exc}")  # as facilities() words it: one outage, reported once
            except (ComponentError, RequestException, RuntimeError) as exc:
                errors.append(f"SVO {facility}: {exc}")
            else:
                if instruments:
                    listings.append(("svo", [(i.key, i.description) for i in instruments]))
                elif fac is not None:
                    errors.append(f"SVO {facility}: No instruments returned for a listed SVO facility")
            if self.mirror.is_available():
                try:
                    fac = next((name for name in self.mirror.discover_facilities() if same_facility(name)), None)
                    live = self.mirror._instrument_records(fac) if fac else {}
                except RequestException as exc:
                    errors.append(f"NJM {facility}: {exc}")
                listings.append(("njm", [(name, "") for name in live]))
            else:
                errors.append("NJM unavailable")
        known = self._njm_instruments(facility, live)
        claims = {name_key(claim.partition("/")[2]): njm for njm, meta in known.items()
                  for claim in meta["svo_instruments"] if name_key(claim.partition("/")[0]) == name_key(facility)}
        entries = _merge(listings)
        _drop_svo_duplicates(entries, claims)
        _adopt_claimed_svo(entries, claims, known)
        return {"entries": _picker_entries(entries, known, claims), "errors": errors}

    def inventory(self, *, remote=True, facility=None):
        """List every discovered instrument; retain errors and unreviewed sources."""
        catalogue = self.facilities(remote=remote)
        rows, errors = [], list(catalogue["errors"])
        for fac in catalogue["entries"]:
            if facility and name_key(facility) not in (name_key(fac["key"]), name_key(fac["label"])):
                continue
            source_catalogue = any(s in fac["sources"] for s in ("svo", "njm"))
            result = self.instruments(fac["key"], remote=remote and source_catalogue)
            errors.extend(error for error in result["errors"] if error not in errors)
            for inst in result["entries"]:
                rows.append({"facility": fac["key"], "instrument": inst["key"],
                             "sources": inst["sources"],
                             "component_availability": component_availability(fac["key"], inst["key"])})
        return {"instruments": rows, "errors": errors}


#: How available() names each source in its ``errors`` strings.
_SOURCE_LABELS = {"njm": "NJM", "svo": "SVO"}


def _njm_outage(reason):
    """Whether NJM's failure is its outage, which the NJM client announces once per process."""
    return reason == NJM_UNAVAILABLE or NJMFilterGrabber._paused()


def _publishing_builders():
    """The ``(facility, instrument)`` of each builder that can make the published curves."""
    return [names for names, ref in BUILDERS_FOR.items() if ref.publishes]


def _availability(description, source, failed):
    """``description`` with ``availability_source``, ``errors``, ``sources_failed`` and ``fallback_from``.

    ``failed`` maps a source to its failure messages, in the order they happened.
    """
    errors = [f"{_SOURCE_LABELS[name]}: {message}" for name, messages in failed.items() for message in messages]
    fallback = [name for name in failed if name != source] if source != "unknown" else []
    return dict(description, availability_source=source, errors=errors,
                sources_failed={name: list(messages) for name, messages in failed.items()},
                fallback_from=fallback)


def _refuse_uncovered_bands(manifest, roles, bands):
    """ComponentError, before any download, for bands the manifest gives none of the requested components.

    Only when the manifest declares those components for some of its bands:
    NJM and the original-source builders, which wrote it, have them for no
    others (Euclid/NISP's grisms Blue and Red have no factors). A manifest
    that declares none, as for curves from SVO, is left to prepare_components,
    since a builder may add them.
    """
    needed = [role for role in roles if role in (FILTER, OPTICS, QE)]
    if not needed:
        return
    covered = [band for band in manifest.primary_bands() if manifest.cover(band, needed)]
    selected = manifest.primary_bands() if bands is None else list(bands)
    missing = [band for band in selected if not manifest.cover(band, needed)]
    if covered and missing:
        raise ComponentError(
            f"{manifest.facility}/{manifest.instrument} has no {recipe_name(needed)} components for "
            f"{', '.join(missing)}; they exist for {', '.join(covered)}. Compose only those bands "
            "(bands=[...], or --bands on the command line).")


# ----------------------------------------------------------------------
# Installed sets
# ----------------------------------------------------------------------
def _declared_bands(directory):
    """The bands an installed manifest declares part of the set: ``bands`` minus ``archived_bands``."""
    manifest = InstrumentManifest.load(directory)
    return [band for band in manifest.bands if band not in manifest.archived_bands]


def _manifest_text(directory):
    """The installed ``manifest.json`` as text, or None when there is none."""
    path = Path(directory) / "manifest.json"
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _partial_fill_manifest(directory):
    """The manifest (as text) of a set installed in part, which declares bands whose curves are not
    installed (SVO's bands of NJM's set, with NJM down); None for any other folder."""
    text = _manifest_text(directory)
    if text is None or all((Path(directory) / f"{band}.dat").is_file() for band in _declared_bands(directory)):
        return None
    return text


def _describe_completed_set(directory):
    """Write the manifest of a set NJM has just completed as a fresh NJM download writes it.

    The manifest of the partial fill described only the curves it installed,
    so the others' published contents and detector types were missing.  Now
    every curve NJM's index file lists is described from its header.
    """
    manifest = InstrumentManifest.from_published(directory)
    manifest.bands = sorted(Path(name).stem for name in index_listing(directory) or [] if name.endswith(".dat"))
    manifest.save(directory)


def _drop_unlisted_bands(directory, listed):
    """Remove from an NJM set's manifest the bands NJM's index does not list and that are not installed.

    Such bands are never downloaded: an archive copy NJM's manifest did not
    move to ``archived_bands``, or one left by an earlier release's SVO
    fallback.  Kept, they would make the set look incomplete for ever.
    """
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    archived = data.get("archived_bands") or {}
    stale = [band for band in data.get("bands", []) if band not in archived and f"{band}.dat" not in listed
             and not (Path(directory) / f"{band}.dat").is_file()]
    if not stale:
        return
    data["bands"] = [band for band in data["bands"] if band not in stale]
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.warning("%s declared bands that NJM's index does not list and that are not installed (%s); "
                   "they are no longer expected", path, ", ".join(stale))


def _curve_state(directory):
    """Each curve file's modification time, to tell whether a download wrote any."""
    return {path.name: path.stat().st_mtime_ns for path in Path(directory).glob("*.dat")}


def _same_directory(a, b):
    """Whether two paths are one directory (they are, on a case-insensitive disk, when only case differs)."""
    a, b = Path(a), Path(b)
    return a == b or (a.exists() and b.exists() and os.path.samefile(a, b))


# ----------------------------------------------------------------------
# The bundled instrument catalogue
# ----------------------------------------------------------------------
#: NJM's facilities and instruments, their descriptions and SVO equivalents, as
#: of the release; regenerate with scripts/make_instrument_catalogue.py.
INSTRUMENT_CATALOGUE = Path(__file__).parent / "instrument_catalogue.json"


def load_instrument_catalogue(path=INSTRUMENT_CATALOGUE):
    """The bundled catalogue: ``{facility: {"description", "instruments": {name: {...}}}}``.

    Keyed by NJM's spelling; each instrument has ``description`` and
    ``svo_instruments`` (SVO ``Facility/Instrument`` names of the same
    product), empty where NJM gives none.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return instrument_catalogue(data["facilities"])


def instrument_catalogue(facilities):
    """Normalise ``filters.facilities_detailed`` of NJM's ``index.json`` (or a saved catalogue).

    Only ``description``, ``svo_instruments``, and the bundled catalogue's
    ``bands`` (NJM's served bands) and ``svo_bands`` (SVO filter ID -> one of
    those bands) are kept; a field in another form than the contract's is left
    empty rather than guessed at.
    """
    if not isinstance(facilities, dict):
        raise ValueError("expected a mapping of facilities")
    catalogue = {}
    for facility, record in sorted(facilities.items()):
        record = record if isinstance(record, dict) else {}
        instruments = record.get("instruments") if isinstance(record.get("instruments"), dict) else {}
        catalogue[facility] = {
            "description": _plain_text(record.get("description")),
            "instruments": {name: _instrument_metadata(meta) for name, meta in sorted(instruments.items())},
        }
    return catalogue


def _instrument_metadata(record):
    record = record if isinstance(record, dict) else {}
    claims = record.get("svo_instruments")
    claims = [c for c in claims if isinstance(c, str) and "/" in c] if isinstance(claims, list) else []
    bands = record.get("bands")
    bands = [b for b in bands if _band_name(b)] if isinstance(bands, list) else []
    return {"description": _plain_text(record.get("description")), "svo_instruments": claims,
            "bands": bands, "svo_bands": _svo_bands(record.get("svo_bands"), claims, bands)}


def _svo_bands(stored, claims, bands):
    """``{SVO filter ID: NJM band}`` from the stored ``{SVO instrument: ["G", "Gbp=BP", ...]}``.

    Only SVO instruments in ``claims`` and NJM bands in ``bands`` count.
    """
    held = {}
    for claim, entries in (stored.items() if isinstance(stored, dict) else ()):
        if claim not in claims or not isinstance(entries, list):
            continue
        for entry in entries:
            svo_band, _, band = entry.partition("=") if isinstance(entry, str) else ("", "", "")
            band = band or svo_band
            if svo_band and band in bands:
                held[f"{claim}.{svo_band}"] = band
    return held


def _band_name(value):
    """Whether ``value`` can name a curve file in an instrument directory."""
    return isinstance(value, str) and value not in ("", ".", "..") and Path(value).name == value


def _plain_text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


# ----------------------------------------------------------------------
# Picker entries
# ----------------------------------------------------------------------
def _merge(listings):
    """Group the names each source lists (``[(source, [(name, description), ...]), ...]``) into entries.

    A name joins an entry holding the same name from another source, else one
    whose name matches it under :func:`name_key` and has no name from this
    source yet.  One source's own names never join each other: NJM serves
    both ``Cameras/Sony_NEX5N`` and ``Cameras/Sony_NEX_5N``.
    """
    entries = []
    for source, listing in listings:
        named = {}
        for name, description in listing:
            named.setdefault(name, description)
        unmatched = []
        for name, description in named.items():
            entry = next((e for e in entries if source not in e["names"] and name in e["names"].values()), None)
            if entry is None:
                unmatched.append((name, description))
            else:
                _join(entry, source, name, description)
        for name, description in unmatched:
            entry = next((e for e in entries if source not in e["names"] and name_key(name) in e["keys"]), None)
            if entry is None:
                entry = {"names": {}, "descriptions": {}, "keys": set()}
                entries.append(entry)
            _join(entry, source, name, description)
    return entries


def _join(entry, source, name, description):
    entry["names"][source] = name
    entry["keys"].add(name_key(name))
    if description:
        entry["descriptions"][source] = description


def _drop_svo_duplicates(entries, claims):
    """Remove SVO's copy of a product NJM lists as its own, keeping SVO's description for NJM's entry.

    ``claims`` maps an SVO instrument (:func:`name_key`) to the NJM instrument
    that is the same product.  Only instruments NJM lists now count: while NJM
    is unreachable, SVO's copies are what can be downloaded.
    """
    listed = {e["names"]["njm"]: e for e in entries if "njm" in e["names"]}
    for entry in [e for e in entries if "svo" in e["names"]]:
        claimer = listed.get(claims.get(name_key(entry["names"]["svo"])))
        if claimer is None:
            continue
        del entry["names"]["svo"]
        text = entry["descriptions"].pop("svo", "")
        if text:
            claimer["descriptions"].setdefault("svo", text)
    entries[:] = [e for e in entries if e["names"]]


def _adopt_claimed_svo(entries, claims, njm_known):
    """Name SVO's copy of a product of NJM's after NJM, when NJM does not list it now (NJM is down, say).

    ``claims`` maps an SVO instrument (:func:`name_key`) to the NJM instrument
    that is the same product.  SVO's name joins the entry that already stands
    for that product (an original-source builder's or an installed copy's,
    under NJM's spelling), else its own entry takes NJM's name
    (``njm_name``).  Of two SVO instruments one NJM product claims (Euclid's
    NISP and NISP0) one entry remains, holding the first.  So the list names
    each product once, as when NJM is reachable.
    """
    listed = {e["names"]["njm"] for e in entries if "njm" in e["names"]}
    for entry in [e for e in entries if "svo" in e["names"]]:
        owner = claims.get(name_key(entry["names"]["svo"]))
        if owner is None or owner in listed:
            continue  # not NJM's product, or NJM lists it now (_drop_svo_duplicates kept NJM's)
        target = next((e for e in entries if e is not entry and _stands_for(e, owner, njm_known)), None)
        if target is None:
            if set(entry["names"]) == {"svo"} or _stands_for(entry, owner, njm_known):
                entry["njm_name"] = owner
                continue
            target = {"names": {}, "descriptions": {}, "keys": set(), "njm_name": owner}
            entries.append(target)
        name = entry["names"].pop("svo")
        text = entry["descriptions"].pop("svo", "")
        if "svo" not in target["names"]:
            _join(target, "svo", name, text)
    entries[:] = [e for e in entries if e["names"]]


def _stands_for(entry, owner, njm_known):
    """Whether an entry is NJM's product ``owner``: named so already, or by a source other than SVO."""
    return entry.get("njm_name") == owner or any(
        match_name(name, njm_known) == owner for source, name in entry["names"].items() if source != "svo")


def _picker_entries(entries, njm_known, claims=None):
    """The entries as the pickers show them, sorted by label.

    Each has ``label`` (the name shown), ``fetch_key`` (the spelling to
    download with: NJM's, else the original builder's, else the installed
    directory's, else SVO's), ``key`` (the same), ``sources`` (every source
    listing it now: ``local``, ``original``, ``svo``, ``njm``),
    ``description`` (NJM's, else SVO's own text, else ``""``; SVO's can run to
    several sentences) and ``names`` (each source's own spelling, NJM's from
    the bundled catalogue when NJM does not list it now).  ``njm_known`` (NJM's names -> their catalogue record)
    names and describes entries NJM does not list now, so names do not change
    with NJM's availability; an entry only SVO lists takes them only if
    ``claims`` (SVO name key -> NJM name) says it is NJM's product.
    """
    rows = []
    for entry in entries:
        names = dict(entry["names"])
        if "njm" not in names:
            names["njm"] = entry.get("njm_name") or _catalogue_name(names, njm_known, claims)
        names = {source: name for source, name in names.items() if name}
        fetch_key = next(names[source] for source in FETCH_ORDER if source in names)
        description = (njm_known.get(names.get("njm")) or {}).get("description") or entry["descriptions"].get("svo", "")
        rows.append({"key": fetch_key, "label": fetch_key, "sources": list(entry["names"]),
                     "description": description, "fetch_key": fetch_key, "names": names})
    return sorted(rows, key=lambda row: (row["label"].casefold(), row["label"]))


def _catalogue_name(names, njm_known, claims):
    """NJM's name for an entry it does not list now, from the bundled catalogue, or None."""
    for name in names.values():
        found = match_name(name, njm_known)
        if found is None:
            continue
        if claims is not None and set(names) == {"svo"} and claims.get(name_key(names["svo"])) != found:
            return None  # the same name, but NJM does not say it is the same product
        return found
    return None
