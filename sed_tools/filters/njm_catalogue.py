"""NJM's descriptive files about the curves it serves (index.json, properties.json, info.json): search and citations.

Cached under ``<filter root>/../cache/njm/``, revalidated by ETag when ``index.json`` (rechecked daily) changes,
and used as cached without NJM.  No curve is downloaded here: :mod:`.library` chooses sources, :mod:`.njm` fetches.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import pandas as pd
from requests import RequestException

from ..component_sources.catalog import name_key
from ..njm_mirror import _body, _write_atomic, cache_root
from .curve_metadata import normalize_detector_type, read_filter_header
from .manifest import index_names

logger = logging.getLogger("sed_tools.filter_catalogue")

INDEX_MAX_AGE = 24 * 3600  # seconds between checks of index.json
MAX_CONCURRENT = 6  # NJM blocks bursts from one address
PROPERTIES, INFO = "properties", "info"
VEGA_URL = "vega/vega_flam.csv"
KEY_COLUMNS = ["facility", "instrument", "band"]
# Leading property columns of a search result; the rest follow in the served order.
LEADING_COLUMNS = ["WavelengthPivot", "WavelengthMin", "WavelengthMax", "FWHM", "WidthEff",
                   "DetectorType", "DetectorTypeSource"]

Range = Union[Tuple[Optional[float], Optional[float]], Sequence[Optional[float]]]


class CatalogueError(RuntimeError):
    """The catalogue cannot answer: unknown instrument, file not served, mirror offline."""


def parse_target(target: str) -> Tuple[str, str, Optional[str]]:
    """``"Facility/Instrument"`` or ``"Facility/Instrument/Band"`` -> (facility, instrument, band)."""
    parts = [p for p in str(target).strip().strip("/").split("/") if p]
    if len(parts) in (2, 3):
        return parts[0], parts[1], parts[2] if len(parts) == 3 else None
    raise ValueError(f"expected Facility/Instrument or Facility/Instrument/Band, got {target!r}")


class FilterCatalogue:
    """Cached, read-only access to NJM's ``index.json``, ``properties.json`` and ``info.json``.

    ``index.json`` lists every facility, instrument and band.  ``properties.json``
    holds the values computed from each curve by :mod:`sed_tools.filter_properties`
    and :mod:`sed_tools.filter_colors` (pivot and other wavelengths, widths, zero
    points, AB−Vega, DisplayColor, ...).  ``info.json`` holds descriptions, the
    references to cite (with BibTeX where a service supplied it), the providers'
    requested acknowledgements, the licence and what SVO recorded about each band.
    Per-instrument files are fetched only for the instruments a search still
    needs, at most six at a time (NJM blocks bursts).

    ``mirror`` is an :class:`~sed_tools.njm_filter_grabber.NJMFilterGrabber`, or a
    callable returning one, so that nothing touches the network until it must.
    ``offline=True`` reads installed and cached copies only.
    """

    def __init__(self, filter_root: Union[str, Path], *, mirror: Any = None, offline: bool = False,
                 max_workers: int = MAX_CONCURRENT, index_max_age: float = INDEX_MAX_AGE):
        self.filter_root = Path(filter_root)
        self.cache_root = cache_root(self.filter_root)
        self.offline = offline
        self.max_workers = max(1, min(int(max_workers), MAX_CONCURRENT))
        self.index_max_age = index_max_age
        self._mirror = mirror
        self._mirror_lock = threading.Lock()
        self._index: Optional[dict] = None
        self._index_loaded = 0.0
        self._origins: Dict[Any, dict] = {}  # see origin()

    def origin(self, facility: Optional[str] = None, instrument: Optional[str] = None,
               kind: str = INFO) -> Optional[dict]:
        """Where the last read of an instrument's ``kind`` file (without an instrument: of index.json) came from.

        ``{"source": "installed" | "cache" | "njm", "checked": when NJM last
        confirmed that copy (time.time(); None for an installed copy),
        "njm_unreachable": True when the cached copy was used because NJM
        could not be asked}``, or None if it has not been read.
        """
        return self._origins.get("index.json" if facility is None else (facility, instrument, kind))

    def _came_from(self, key, source: str, checked: Optional[float], njm_unreachable: bool = False) -> None:
        self._origins[key] = {"source": source, "checked": checked, "njm_unreachable": njm_unreachable}

    # ---- HTTP with an on-disk cache
    def _online_mirror(self, *, have_copy: bool = False):
        """The mirror client, or ``None`` to use the cached copy instead.

        An unavailable mirror is still asked for files not yet cached: the check also fails during NJM's brief
        burst blocks, which the request retries wait out.
        """
        if self.offline:
            return None
        with self._mirror_lock:  # fetch threads must not each build a client
            if self._mirror is None:
                from .njm import NJMFilterGrabber

                self._mirror = NJMFilterGrabber(base_dir=str(self.filter_root))
            elif callable(self._mirror) and not hasattr(self._mirror, "is_available"):
                self._mirror = self._mirror()
        return None if have_copy and not self._mirror.is_available() else self._mirror

    def _cache_path(self, rel: str) -> Path:
        path = self.cache_root / rel
        if ".." in Path(rel).parts or not path.resolve().is_relative_to(self.cache_root.resolve()):
            raise CatalogueError(f"catalogue path outside the cache: {rel}")
        return path

    def _not_cached(self, rel: str) -> str:
        why = "working offline" if self.offline else "NJM is unreachable"
        return f"{rel} is not cached and {why}"

    @staticmethod
    def _read_meta(path: Path) -> dict:
        try:
            return json.loads(path.with_name(path.name + ".http.json").read_text())
        except (OSError, ValueError):
            return {}

    def _fetch(self, rel: str, *, generation: Optional[str] = None, max_age: Optional[float] = None,
               refresh: bool = False, validate: Optional[Callable[[str], None]] = None) -> Optional[Path]:
        """The cached copy of ``<mirror>/<rel>``, fetched or revalidated if needed; None if NJM and the cache lack it.

        A copy is fresh while the index ``generation`` it was checked under is unchanged, or for ``max_age`` seconds.
        """
        path = self._cache_path(rel)
        meta = self._read_meta(path)
        if path.is_file() and not refresh and (
                (generation is not None and meta.get("generation") == generation)
                or (max_age is not None and time.time() - float(meta.get("checked", 0)) < max_age)):
            self._came_from(rel, "cache", meta.get("checked"))
            return path
        mirror = self._online_mirror(have_copy=path.is_file() and not refresh)
        if mirror is None:
            # Offline, the cache is what was asked for; only a refresh it cannot honour is reported.
            if path.is_file() and (refresh or not self.offline):
                logger.warning("Cannot ask NJM (%s); using the cached %s, which may be out of date",
                               "offline" if self.offline else "unreachable", rel)
            if path.is_file():
                self._came_from(rel, "cache", meta.get("checked"), njm_unreachable=not self.offline)
            return path if path.is_file() else None
        conditional = (("If-None-Match", "etag"), ("If-Modified-Since", "last_modified"))
        headers = {header: meta[key] for header, key in conditional if path.is_file() and meta.get(key)}
        url = f"{mirror.base_url}/{quote(rel, safe='/')}"
        try:
            response = mirror._get(url, timeout=60, headers=headers or None)
        except RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                raise CatalogueError(f"NJM does not serve {rel}") from exc
            if path.is_file():
                logger.warning("Cannot ask NJM (%s); using the cached %s, which may be out of date", exc, rel)
                self._came_from(rel, "cache", meta.get("checked"), njm_unreachable=True)
                return path
            if status is None:  # no answer: the mirror client has said why, once
                raise CatalogueError(self._not_cached(rel)) from exc
            raise CatalogueError(f"Cannot fetch {rel} from NJM: {exc}") from exc
        self._store(path, rel, url, response, meta, generation, validate)
        self._came_from(rel, "njm", time.time())
        return path

    def _fetched(self, rel: str, **kwargs) -> Path:
        """:meth:`_fetch`, with CatalogueError when neither NJM nor the cache has the file."""
        path = self._fetch(rel, **kwargs)
        if path is None:
            raise CatalogueError(self._not_cached(rel))
        return path

    @staticmethod
    def _store(path: Path, rel: str, url: str, response, meta: dict, generation: Optional[str],
               validate: Optional[Callable[[str], None]]) -> None:
        """Save a fetched file (unless NJM answered 304 Not Modified) and record when it was checked."""
        response_headers = getattr(response, "headers", None) or {}
        if getattr(response, "status_code", 200) != 304:
            content = _body(response)
            if validate is not None:
                try:
                    validate(content.decode("utf-8"))
                except ValueError as exc:
                    raise CatalogueError(f"NJM served an unreadable {rel}: {exc}") from exc
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(path, content)
            meta = {}
        meta.update(url=url, checked=time.time(), generation=generation,
                    etag=response_headers.get("ETag") or meta.get("etag"),
                    last_modified=(response_headers.get("Last-Modified") or meta.get("last_modified")
                                   or formatdate(path.stat().st_mtime, usegmt=True)))
        path.with_name(path.name + ".http.json").write_text(json.dumps(meta, indent=1) + "\n")

    @staticmethod
    def _json_object(text: str, require_bands: bool = True) -> dict:
        data = json.loads(text)
        if not isinstance(data, dict) or (require_bands and not isinstance(data.get("bands"), dict)):
            raise ValueError("not an NJM catalogue file")
        return data

    # ---- index.json
    def index(self, *, refresh: bool = False) -> dict:
        """NJM's ``index.json`` (rechecked after ``index_max_age`` seconds)."""
        if self._index is not None and not refresh and time.time() - self._index_loaded < self.index_max_age:
            return self._index

        def check(text):
            data = json.loads(text)
            if not isinstance(data, dict) or not isinstance(data.get("filters"), dict):
                raise ValueError("no 'filters' section")

        path = self._fetched("index.json", max_age=self.index_max_age, refresh=refresh, validate=check)
        self._index = json.loads(path.read_text(encoding="utf-8"))
        self._index_loaded = time.time()
        return self._index

    def instruments(self, *, refresh: bool = False) -> List[dict]:
        """One record per served instrument: names, bands, detector types and its catalogue files."""
        facilities = self.index(refresh=refresh)["filters"].get("facilities_detailed", {})
        rows = []
        for fac_name, facility in sorted(facilities.items(), key=lambda kv: kv[0].casefold()):
            for inst_name, inst in sorted(facility.get("instruments", {}).items(), key=lambda kv: kv[0].casefold()):
                rows.append({
                    "facility": fac_name, "instrument": inst_name,
                    "bands": list(inst.get("filters", [])),
                    "detector_types": dict(inst.get("detector_types", {})),
                    PROPERTIES: inst.get(PROPERTIES), INFO: inst.get(INFO),
                })
        return rows

    def resolve(self, facility: str, instrument: str) -> dict:
        """The served instrument matching these names (exactly, then ignoring case and separators)."""
        rows = self.instruments()
        for match in (lambda a, b: a == b, lambda a, b: a.casefold() == b.casefold(),
                      lambda a, b: name_key(a) == name_key(b)):
            found = [r for r in rows if match(r["facility"], facility) and match(r["instrument"], instrument)]
            if len(found) == 1:
                return found[0]
        raise CatalogueError(f"NJM does not serve {facility}/{instrument}")

    # ---- Per-instrument files
    def _installed(self, facility: str, instrument: str, kind: str) -> Optional[Path]:
        """The copy kept beside the installed curves (:meth:`.FilterLibrary.installed_directory`), if any."""
        directory = self._library.installed_directory(facility, instrument)
        path = directory / f"{kind}.json" if directory is not None else None
        return path if path is not None and path.is_file() else None

    def instrument_file(self, facility: str, instrument: str, kind: str, *,
                        prefer_installed: bool = True, refresh: bool = False) -> dict:
        """``properties.json`` (``kind="properties"``) or ``info.json`` (``"info"``) for an instrument.

        The copy saved beside installed curves describes exactly those curves and is
        used first, unless ``refresh`` or ``prefer_installed=False``; otherwise NJM's
        current file, cached.  Bands NJM keeps beside the set but does not serve as
        part of it (superseded copies such as ``BP_svo``) are moved from ``bands``
        to ``archived_bands`` (see :meth:`_split_archived`).  The records of
        curves installed under other band names are listed under the installed
        names (see :meth:`_named_as_installed`).
        """
        data = self._instrument_file(facility, instrument, kind, prefer_installed=prefer_installed, refresh=refresh)
        data = self._named_as_installed(facility, instrument, kind, data, refresh=refresh)
        return self._split_archived(facility, instrument, data)

    def _named_as_installed(self, facility: str, instrument: str, kind: str, data: dict, *,
                            refresh: bool) -> dict:
        """``data`` with the records of curves installed under other band names listed under those names.

        0.4.1 installed PLANCK/HFI from SVO as ``100GHz`` …, which NJM keeps as ``100_svo`` …; curves are
        matched by the FilterID of their header and of NJM's ``info.json``, only when no band name is installed.
        """
        bands = data.get("bands") or {}
        directory = self._library.installed_directory(facility, instrument)
        listed = directory / directory.name if directory is not None else None
        if not bands or listed is None or not listed.is_file():
            return data
        installed = _index_bands(listed.read_text(encoding="utf-8"))
        if any(band in installed for band in bands):
            return data
        by_id = {}
        for band in sorted(installed):
            curve = directory / f"{band}.dat"
            filter_id = read_filter_header(curve).get("FilterID") if curve.is_file() else None
            if filter_id:
                by_id[filter_id] = band
        if not by_id:
            return data
        try:
            info = data if kind == INFO else self._instrument_file(facility, instrument, INFO,
                                                                   prefer_installed=True, refresh=refresh)
        except (CatalogueError, OSError, ValueError):
            return data  # without info.json the curves cannot be matched; properties.json is still as served
        renamed = {}
        for band, record in (info.get("bands") or {}).items():
            filter_id = ((record or {}).get("curve") or {}).get("FilterID")
            if band in bands and filter_id in by_id:
                renamed[band] = by_id[filter_id]
        if not renamed:
            return data
        return dict(data, bands={renamed.get(band, band): record for band, record in bands.items()})

    def _instrument_file(self, facility: str, instrument: str, kind: str, *,
                         prefer_installed: bool, refresh: bool) -> dict:
        """The file as installed, cached or served (see :meth:`instrument_file`)."""
        if kind not in (PROPERTIES, INFO):
            raise ValueError(f"kind must be {PROPERTIES!r} or {INFO!r}")
        installed = self._installed(facility, instrument, kind) if prefer_installed and not refresh else None
        if installed is not None:
            try:
                data = self._json_object(installed.read_text(encoding="utf-8"))
                self._came_from((facility, instrument, kind), "installed", None)
                return data
            except (OSError, ValueError) as exc:
                # NJM's current file may describe other curves than the installed ones.
                raise CatalogueError(f"{installed} is unreadable ({exc}); repair or remove it, or pass "
                                     "refresh=True to read NJM's current file instead") from exc
        index = self.index()
        entry = self.resolve(facility, instrument)
        name = entry[kind]
        if not name:
            raise CatalogueError(f"NJM has no {kind} file for {entry['facility']}/{entry['instrument']}")
        rel = f"filters/{entry['facility']}/{entry['instrument']}/{name}"
        path = self._fetched(rel, generation=index.get("generated"), refresh=refresh, validate=self._json_object)
        data = self._json_object(path.read_text(encoding="utf-8"))
        origin = dict(self._origins[rel])
        if origin["source"] == "cache" and (self._origins.get("index.json") or {}).get("njm_unreachable"):
            origin["njm_unreachable"] = True  # current only as of an index.json NJM could not confirm
        self._origins[(facility, instrument, kind)] = origin
        return data

    def _split_archived(self, facility: str, instrument: str, data: dict) -> dict:
        """``data`` with the bands that are not part of the served set under ``archived_bands``.

        The set is the first of the installed index file, NJM's and the bundled catalogue's that has any of its bands.
        """
        bands = data.get("bands") or {}
        served = next((names for names in self._served_band_sets(facility, instrument)
                       if any(band in names for band in bands)), None)
        if served is None:
            return data
        archived = {band: record for band, record in bands.items() if band not in served}
        if not archived:
            return data
        return dict(data, bands={band: record for band, record in bands.items() if band in served},
                    archived_bands=dict(data.get("archived_bands") or {}, **archived))

    def _served_band_sets(self, facility: str, instrument: str) -> Iterator[set]:
        """The bands of the instrument's set from each place that lists them, in :meth:`_split_archived`'s order."""
        installed = self._library.installed_directory(facility, instrument)
        if installed is not None:
            listed = installed / installed.name
            if listed.is_file():
                yield _index_bands(listed.read_text(encoding="utf-8"))
        try:
            entry = self.resolve(facility, instrument)
            rel = f"filters/{entry['facility']}/{entry['instrument']}/{entry['instrument']}"
            path = self._fetch(rel, generation=self.index().get("generated"))
        except CatalogueError:
            path = None
        if path is not None:
            yield _index_bands(path.read_text(encoding="utf-8"))
        bundled = self._library._bundled_record(facility, instrument)
        if bundled is not None:
            yield set(bundled["bands"])

    @cached_property
    def _library(self):
        from .library import FilterLibrary

        return FilterLibrary(self.filter_root)

    def properties(self, facility: str, instrument: str, **kwargs) -> dict:
        """The instrument's ``properties.json``: ``{"bands": {band: {...}}, "definitions": {...}}``."""
        return self.instrument_file(facility, instrument, PROPERTIES, **kwargs)

    def info(self, facility: str, instrument: str, **kwargs) -> dict:
        """The instrument's ``info.json`` (schema in NJM's ``filter_info/README.md``)."""
        return self.instrument_file(facility, instrument, INFO, **kwargs)

    def many(self, entries: Sequence[dict], kind: str, *, refresh: bool = False
             ) -> Tuple[Dict[Tuple[str, str], dict], List[str]]:
        """Fetch one catalogue file for several served instruments, at most six at a time."""
        from tqdm import tqdm

        generation = self.index().get("generated")
        todo = [e for e in entries if e.get(kind)]
        errors = [f"{e['facility']}/{e['instrument']}: NJM has no {kind} file" for e in entries if not e.get(kind)]

        def one(entry):
            rel = f"filters/{entry['facility']}/{entry['instrument']}/{entry[kind]}"
            path = self._fetched(rel, generation=generation, refresh=refresh, validate=self._json_object)
            return self._json_object(path.read_text(encoding="utf-8"))

        results: Dict[Tuple[str, str], dict] = {}
        bar = tqdm(total=len(todo), desc=f"[njm] {kind}", unit="instrument",
                   disable=len(todo) < 10 or not sys.stderr.isatty())
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(one, e): e for e in todo}
            for future, entry in futures.items():
                try:
                    results[(entry["facility"], entry["instrument"])] = future.result()
                except (CatalogueError, OSError, ValueError) as exc:
                    errors.append(f"{entry['facility']}/{entry['instrument']}: {exc}")
                bar.update()
        bar.close()
        return results, errors

    def vega(self, *, refresh: bool = False) -> Path:
        """Path to the cached Vega spectrum NJM's properties use (``vega/vega_flam.csv``, Å and f_λ)."""
        definitions = self.index()["filters"].get("properties_definitions") or {}
        rel = (definitions.get("vega_spectrum") or {}).get("url") or VEGA_URL
        return self._fetched(rel, max_age=30 * 24 * 3600, refresh=refresh)

    # ---- Search
    def _named_bands(self, text: Optional[str], needles: Mapping[str, Optional[str]],
                     wanted_detector: Optional[str], refresh: bool) -> Tuple[List[dict], Dict[Tuple[str, str], dict]]:
        """Bands whose names match, from ``index.json``: their rows and the instruments they belong to."""
        rows, entries = [], {}
        for entry in self.instruments(refresh=refresh):
            if wanted_detector and entry["detector_types"] and not entry["detector_types"].get(wanted_detector):
                continue
            for name in entry["bands"]:
                values = {"facility": entry["facility"], "instrument": entry["instrument"], "band": name}
                if any(n and n.casefold() not in values[k].casefold() for k, n in needles.items()):
                    continue
                if text and text.casefold() not in f"{entry['facility']}/{entry['instrument']}/{name}".casefold():
                    continue
                rows.append(values)
                entries[(entry["facility"], entry["instrument"])] = entry
        return rows, entries

    def search(self, text: Optional[str] = None, *, facility: Optional[str] = None,
               instrument: Optional[str] = None, band: Optional[str] = None,
               wavelength: Any = None, pivot: Optional[Range] = None, fwhm: Optional[Range] = None,
               detector_type: Optional[str] = None, where: Optional[Mapping[str, Any]] = None,
               properties: bool = True, refresh: bool = False) -> pd.DataFrame:
        """Find served bands; see :meth:`sed_tools.api.Filters.search`."""
        wanted_detector = None
        if detector_type is not None:
            wanted_detector = normalize_detector_type(detector_type)
            if wanted_detector is None:
                raise ValueError(f"Unknown detector type: {detector_type!r}")
        criteria = _criteria(wavelength, pivot, fwhm, where, wanted_detector)
        if criteria and not properties:
            raise ValueError("property criteria need properties=True")
        needles = {"facility": facility, "instrument": instrument, "band": band}
        rows, entries = self._named_bands(text, needles, wanted_detector, refresh)

        errors: List[str] = []
        gaps: List[str] = []  # permanent: NJM lists the instrument but serves no properties.json for it
        if properties and rows:
            served = [e for e in entries.values() if e.get(PROPERTIES)]
            gaps = [f"{e['facility']}/{e['instrument']}: NJM has no {PROPERTIES} file"
                    for e in entries.values() if not e.get(PROPERTIES)]
            fetched, errors = self.many(served, PROPERTIES, refresh=refresh)
            for row in rows:
                served = fetched.get((row["facility"], row["instrument"]), {}).get("bands", {})
                row.update(served.get(row["band"], {}))
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=KEY_COLUMNS)
        mask = pd.Series(True, index=frame.index)
        for column, criterion in criteria.items():
            mask &= _apply(frame, column, criterion)
        frame = frame[mask]
        leading = [c for c in LEADING_COLUMNS if c in frame.columns]
        rest = [c for c in frame.columns if c not in KEY_COLUMNS and c not in leading]
        frame = frame[KEY_COLUMNS + leading + rest]
        order = [frame[c].str.casefold() for c in ("facility", "instrument")]
        sort = pd.DataFrame({"f": order[0], "i": order[1],
                             "p": frame["WavelengthPivot"] if "WavelengthPivot" in frame else 0,
                             "b": frame["band"].str.casefold()}, index=frame.index)
        frame = frame.loc[sort.sort_values(["f", "i", "p", "b"]).index].reset_index(drop=True)
        frame.attrs["errors"] = errors
        frame.attrs["gaps"] = gaps
        frame.attrs["units"] = (self.index()["filters"].get("properties_definitions") or {}).get("units", {})
        return frame


def _index_bands(text: str) -> set:
    """The bands a MESA index file lists (its ``.dat`` names without the suffix)."""
    return {name[:-len(".dat")] for name in index_names(text) if name.endswith(".dat")}


# ---- Search criteria
def _criteria(wavelength, pivot, fwhm, where, detector_type: Optional[str]) -> Dict[str, tuple]:
    """The property criteria of a search, by column."""
    criteria = {}
    if wavelength is not None:
        criteria["wavelength"] = _wavelength_criterion(wavelength)
    if pivot is not None:
        criteria["WavelengthPivot"] = _range(pivot, angstrom=True)
    if fwhm is not None:
        criteria["FWHM"] = _range(fwhm, angstrom=True)
    for column, value in (where or {}).items():
        criteria[column] = _range(value) if _is_range(value) else ("equals", value)
    if detector_type:
        criteria["DetectorType"] = ("equals", detector_type)
    return criteria


def _to_angstrom(value) -> float:
    """A wavelength in Å; astropy quantities in any spectral unit are converted."""
    if hasattr(value, "unit"):
        from astropy import units as u

        return float(value.to_value(u.AA, equivalencies=u.spectral()))
    return float(value)


def _is_range(value) -> bool:
    return isinstance(value, (tuple, list)) and len(value) == 2 or (
        hasattr(value, "unit") and getattr(value, "shape", ()) == (2,))


def _range(value, *, angstrom: bool = False):
    if not _is_range(value):
        raise ValueError(f"expected a (low, high) range, got {value!r}")
    if hasattr(value, "unit"):
        value = list(value)
    convert = _to_angstrom if angstrom else float
    lo, hi = (None if v is None else convert(v) for v in value)
    return ("range", lo, hi)


def _wavelength_criterion(value):
    if _is_range(value):
        return ("overlaps",) + _range(value, angstrom=True)[1:]
    return ("covers", _to_angstrom(value))


def _apply(frame: pd.DataFrame, column: str, criterion) -> pd.Series:
    """Rows meeting one criterion; missing values (e.g. Vega quantities NJM left null) never match."""
    def numeric(name):
        if name not in frame:
            return pd.Series(float("nan"), index=frame.index)
        return pd.to_numeric(frame[name], errors="coerce")

    kind = criterion[0]
    if kind == "covers":
        return (numeric("WavelengthMin") <= criterion[1]) & (numeric("WavelengthMax") >= criterion[1])
    if kind in ("overlaps", "range"):
        lo, hi = criterion[1:]
        # overlaps: the curve reaches past lo and starts before hi; range: the value lies between them
        if kind == "overlaps":
            reaches, starts = numeric("WavelengthMax"), numeric("WavelengthMin")
        else:
            reaches = starts = numeric(column)
        mask = reaches.notna() & starts.notna()
        if lo is not None:
            mask &= reaches >= lo
        if hi is not None:
            mask &= starts <= hi
        return mask
    wanted = criterion[1]
    if column not in frame:
        return pd.Series(False, index=frame.index)
    if isinstance(wanted, str):
        return frame[column].map(lambda v: isinstance(v, str) and v.casefold() == wanted.casefold())
    return frame[column].map(lambda v: v == wanted)


# ---- Citations
def _band_references(info: dict, bands: Optional[Iterable[str]]) -> List[str]:
    """References for the chosen bands: the instrument-wide ones plus each band's own."""
    everything = list(info.get("references", []))
    if bands is None:
        return everything
    band_info = {**(info.get("archived_bands") or {}), **info.get("bands", {})}
    per_band = {ref for b in band_info.values() for ref in b.get("references", [])}
    chosen = [ref for band in bands for ref in band_info.get(band, {}).get("references", [])]
    keep = [ref for ref in everything if ref not in per_band or ref in chosen]
    return keep + [ref for ref in dict.fromkeys(chosen) if ref not in keep]


def collect_citations(infos: Sequence[Tuple[str, dict, Optional[Sequence[str]]]]) -> dict:
    """Merge what to cite for several instruments.

    ``infos`` holds ``(label, info_json, bands or None)``.  References are merged
    across instruments by key, following ``same_as`` links to the canonical
    (bibcode) form; acknowledgements are merged by their text.
    """
    bibliography: Dict[str, dict] = {}
    for _, info, _ in infos:
        for key, entry in (info.get("bibliography") or {}).items():
            bibliography.setdefault(key, entry)

    def canonical(key):
        seen = set()
        while bibliography.get(key, {}).get("same_as") and key not in seen:
            seen.add(key)
            key = bibliography[key]["same_as"]
        return key

    references: Dict[str, dict] = {}
    acknowledgements: Dict[str, dict] = {}
    instruments = []
    for label, info, bands in infos:
        instruments.append({"instrument": label, "bands": list(bands) if bands is not None else None,
                            "description": info.get("description"), "licence": info.get("licence"),
                            "links": info.get("links", [])})
        for ref in _band_references(info, bands):
            key = canonical(ref)
            record = references.setdefault(key, {"key": key, "aliases": [], "used_by": [],
                                                 **(bibliography.get(key) or bibliography.get(ref) or {})})
            if ref != key and ref not in record["aliases"]:
                record["aliases"].append(ref)
            if label not in record["used_by"]:
                record["used_by"].append(label)
        for ack in info.get("acknowledgements", []):
            text = " ".join(str(ack.get("text", "")).split())
            if not text:
                continue
            record = acknowledgements.setdefault(text, {**ack, "used_by": []})
            if label not in record["used_by"]:
                record["used_by"].append(label)
    return {"instruments": instruments, "references": list(references.values()),
            "acknowledgements": list(acknowledgements.values()), "errors": []}


def _reference_line(ref: dict) -> str:
    label = ref.get("short") or ref["key"]
    if label.startswith("url:"):  # a reference known only by its address
        label = label[len("url:"):]
    parts = [label]
    venue = " ".join(str(ref[k]) for k in ("journal", "volume") if ref.get(k))
    if venue:
        parts.append(venue + (f", {ref['page']}" if ref.get("page") else ""))
    line = ", ".join(parts)
    if ref.get("title"):
        line += f'. "{" ".join(str(ref["title"]).split())}"'
    link = (f"doi:{ref['doi']}" if ref.get("doi") else ref.get("url") or ref.get("publisher_url"))
    return line + (f". {link}" if link and link != label else "")


def format_bibtex(citations: dict) -> str:
    """BibTeX for the merged references.

    Entries NJM has no BibTeX for are listed as ``%`` comments (BibTeX ignores
    them) rather than made up.
    """
    entries, missing = [], []
    for ref in citations["references"]:
        if ref.get("bibtex"):
            entries.append(ref["bibtex"].strip())
        else:
            missing.append(f"% No BibTeX served for {ref['key']}: {_reference_line(ref)}")
    blocks = (["\n".join(missing)] if missing else []) + entries
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_citations(citations: dict) -> str:
    """Plain text: acknowledgements, licences, references and their BibTeX."""
    out: List[str] = []
    labels = [i["instrument"] + (f" ({', '.join(i['bands'])})" if i["bands"] else "")
              for i in citations["instruments"]]
    out.append("Citations for " + ", ".join(labels) if labels else "No instruments")
    if citations["acknowledgements"]:
        out += ["", "Acknowledgements (requested by the providers; quote verbatim)"]
        for ack in citations["acknowledgements"]:
            head = f"- {ack.get('for') or 'Provider'}"
            if ack.get("applies_to"):
                head += f" [{ack['applies_to']}]"
            out.append(head + f"  <- {', '.join(ack['used_by'])}")
            out.append(f"  {' '.join(str(ack['text']).split())}")
            if ack.get("url"):
                out.append(f"  {ack['url']}")
    licences = [(i["instrument"], i["licence"]) for i in citations["instruments"] if i.get("licence")]
    if licences:
        out += ["", "Licences"]
        for label, lic in licences:
            line = f"- {label}: {lic.get('redistribution', 'unknown')}"
            if lic.get("terms"):
                line += f". {lic['terms']}"
            if lic.get("url"):
                line += f" {lic['url']}"
            out.append(line)
    if citations["references"]:
        out += ["", "References"]
        for n, ref in enumerate(citations["references"], 1):
            out.append(f"[{n}] {_reference_line(ref)}  <- {', '.join(ref['used_by'])}")
        out += ["", "BibTeX", format_bibtex(citations).rstrip()]
    for error in citations.get("errors", []):
        out.append(f"[filters] {error}")
    return "\n".join(out) + "\n"
