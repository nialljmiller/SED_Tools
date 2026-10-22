"""Detector-type metadata for filter transmission curves.

A transmission curve is defined either for an *energy-counting* detector
(e.g. classical photomultiplier-era curves such as Bessell's Johnson-Cousins)
or for a *photon-counting* detector (CCDs; e.g. Gaia DR3).  Synthetic
photometry must weight the integrand by ``T(λ)`` in the first case and by
``T(λ)·λ`` in the second (Bessell & Murphy 2012, PASP 124, 140, eq. A9).

Every filter ``.dat`` file managed by SED_Tools carries this information in a
small comment header placed before the CSV body::

    # FilterID: GAIA/GAIA3.G
    # DetectorType: photon
    # DetectorTypeSource: svo
    # WavelengthUnit: Angstrom
    Wavelength,Transmission
    ...

``DetectorType`` is ``energy`` or ``photon``.  ``DetectorTypeSource`` records
how it was established:

``svo``
    Read from the SVO Filter Profile Service record for ``FilterID``.
``svo-curve-match``
    The file had no ``FilterID``; its curve was matched point-by-point against
    the SVO candidates with the same facility and band.
``assumed``
    No authoritative source could be found; ``energy`` is written so that
    every file is explicit, but the value is unverified.

``WavelengthUnit`` is copied from SVO whenever the curve is identified, so
the loader never has to guess the unit from the values.
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

ENERGY = "energy"
PHOTON = "photon"
DETECTOR_TYPES = (ENERGY, PHOTON)

SOURCE_SVO = "svo"
SOURCE_SVO_MATCH = "svo-curve-match"
SOURCE_ASSUMED = "assumed"

HEADER_KEYS = (
    "FilterID",
    "DetectorType",
    "DetectorTypeSource",
    "WavelengthUnit",
    # Component bookkeeping (see sed_tools.filter_components)
    "Role",
    "Components",
    "ComponentsSource",
    "Recipe",
    "Atmosphere",
    "Sources",
    "DetectorTypeOriginal",
)
DETECTOR_KEYS = HEADER_KEYS[:4]

SVO_FPS_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php"

PathLike = Union[str, os.PathLike]


def normalize_detector_type(value: object) -> Optional[str]:
    """Map SVO codes (0/1) and textual labels onto ``"energy"``/``"photon"``."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) == 0:
            return ENERGY
        if float(value) == 1:
            return PHOTON
        return None
    text = str(value).strip().lower()
    if text in {"0", "0.0", "energy", "energy counter", "energy-counting", "energy_counting"}:
        return ENERGY
    if text in {"1", "1.0", "photon", "photon counter", "photon-counting", "photon_counting"}:
        return PHOTON
    return None


# --------------------------------------------------------------------------
# Header I/O
# --------------------------------------------------------------------------
_HEADER_RE = re.compile(r"^#\s*([A-Za-z]+)\s*[:=]\s*(.*?)\s*$")


def read_filter_header(path: PathLike) -> Dict[str, str]:
    """Return the SED_Tools header fields (``HEADER_KEYS``) present in a filter file."""
    fields: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break
            match = _HEADER_RE.match(stripped)
            if match and match.group(1) in HEADER_KEYS:
                fields[match.group(1)] = match.group(2)
    return fields


def write_filter_header(path: PathLike, fields: Dict[str, Optional[str]]) -> None:
    """Rewrite ``path`` with ``fields`` as its leading header, keeping everything else."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    body = []
    for line in lines:
        match = _HEADER_RE.match(line.strip())
        if match and match.group(1) in HEADER_KEYS:
            continue
        body.append(line)

    header = [f"# {key}: {fields[key]}\n" for key in HEADER_KEYS if fields.get(key)]
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.writelines(header + body)
    os.replace(tmp_path, path)


def write_filter_file(
    path: PathLike,
    wavelength: Sequence[float],
    transmission: Sequence[float],
    *,
    detector_type: str,
    source: str,
    filter_id: Optional[str] = None,
    wavelength_unit: Optional[str] = None,
    extra: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    """Write a filter curve in the SED_Tools ``.dat`` format, header included.

    ``extra`` holds further header fields (``Role``, ``Components``, ...);
    keys must be listed in ``HEADER_KEYS``.
    """
    fields = {
        "FilterID": filter_id,
        "DetectorType": detector_type,
        "DetectorTypeSource": source,
        "WavelengthUnit": wavelength_unit,
    }
    for key, value in (extra or {}).items():
        if key not in HEADER_KEYS:
            raise ValueError(f"Unknown filter header key: {key}")
        fields[key] = value
    with open(path, "w", encoding="utf-8") as fh:
        for key in HEADER_KEYS:
            if fields.get(key):
                fh.write(f"# {key}: {fields[key]}\n")
        fh.write("Wavelength,Transmission\n")
        for wl, tr in zip(wavelength, transmission):
            fh.write(f"{float(wl)!r},{float(tr)!r}\n")


def _read_curve(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    wl: List[float] = []
    tr: List[float] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 2 or parts[0].startswith("#"):
                continue
            try:
                wl.append(float(parts[0]))
                tr.append(float(parts[1]))
            except ValueError:
                continue
    return np.asarray(wl), np.asarray(tr)


def _curves_match(a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray]) -> bool:
    (wa, ta), (wb, tb) = a, b
    if wa.size == 0 or wa.shape != wb.shape:
        return False
    peak = max(float(np.max(np.abs(ta))), float(np.max(np.abs(tb))), 1e-30)
    return bool(np.allclose(wa, wb, rtol=1e-6, atol=0.0) and np.allclose(ta, tb, rtol=1e-4, atol=1e-6 * peak))


def _band_key(value: str) -> str:
    """Normalise a band label the same way the SVO grabber names files."""
    cleaned = "".join(ch if ch.isalnum() or ch in (" ", ".", "_") else "_" for ch in value).strip()
    return cleaned.replace(" ", "_").lower()


# --------------------------------------------------------------------------
# SVO lookups
# --------------------------------------------------------------------------
@dataclass
class SVOFilterRecord:
    filter_id: str
    detector_type: Optional[str]
    wavelength: np.ndarray
    transmission: np.ndarray
    wavelength_unit: Optional[str] = None
    components: Optional[str] = None


class SVOMetadataClient:
    """Small cached client for the SVO FPS VOTable endpoint.

    ``astroquery.svo_fps`` drops the VOTable PARAMs (including
    ``DetectorType``) so the response is parsed here directly.
    """

    def __init__(self, session=None, timeout: float = 120.0):
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.timeout = timeout
        self._records: Dict[str, Optional[SVOFilterRecord]] = {}
        self._facility_ids: Dict[str, List[str]] = {}

    def _get_votable(self, params: Dict[str, str]):
        from astropy.io.votable import parse

        resp = self.session.get(SVO_FPS_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return parse(io.BytesIO(resp.content), verify="ignore")

    def get_filter(self, filter_id: str) -> Optional[SVOFilterRecord]:
        """Fetch one filter (curve + DetectorType). ``None`` if SVO has no such ID."""
        if filter_id in self._records:
            return self._records[filter_id]
        record: Optional[SVOFilterRecord] = None
        vot = self._get_votable({"ID": filter_id})
        tables = list(vot.iter_tables())
        if tables:
            table = tables[0]
            params = {p.name: p.value for p in table.params}
            data = table.to_table()
            record = SVOFilterRecord(
                filter_id=filter_id,
                detector_type=normalize_detector_type(_param_text(params.get("DetectorType"))),
                wavelength=np.asarray(data["Wavelength"], dtype=float),
                transmission=np.asarray(data["Transmission"], dtype=float),
                wavelength_unit=_param_text(params.get("WavelengthUnit")) or None,
                components=_param_text(params.get("components")) or None,
            )
        self._records[filter_id] = record
        return record

    def facility_filter_ids(self, facility: str) -> List[str]:
        """Return every SVO filterID registered under ``facility`` (empty if unknown)."""
        if facility in self._facility_ids:
            return self._facility_ids[facility]
        ids: List[str] = []
        try:
            vot = self._get_votable({"Facility": facility})
            tables = list(vot.iter_tables())
            if tables:
                data = tables[0].to_table()
                ids = [_param_text(v) for v in data["filterID"]]
        except Exception:
            logger.debug("SVO facility listing failed for %s", facility, exc_info=True)
        self._facility_ids[facility] = ids
        return ids


def svo_components_header(components: Optional[str]) -> Dict[str, str]:
    """``Components``/``ComponentsSource`` header fields from SVO's components string."""
    if not components:
        return {}
    from .filter_components import ComponentError, parse_components, recipe_name

    try:
        roles = parse_components(components)
    except ComponentError:
        logger.warning("Unrecognised SVO components string %r", components)
        return {}
    return {"Components": recipe_name(roles), "ComponentsSource": SOURCE_SVO} if roles else {}


def _param_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value) if value is not None else ""


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------
@dataclass
class DetectorTypeResult:
    path: Path
    detector_type: str
    source: str
    filter_id: Optional[str] = None
    changed: bool = False
    note: str = ""


def resolve_detector_type(
    path: PathLike,
    *,
    client: Optional[SVOMetadataClient] = None,
    facility: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str, str]:
    """Work out the detector type for one filter file.

    Returns ``(detector_type, filter_id, source, note)``; ``detector_type`` is
    ``None`` when nothing authoritative was found.
    """
    path = Path(path)
    header = read_filter_header(path)
    filter_id = header.get("FilterID") or None
    if client is None:
        client = SVOMetadataClient()

    if filter_id:
        try:
            record = client.get_filter(filter_id)
        except Exception as exc:
            return None, filter_id, SOURCE_ASSUMED, f"SVO lookup for {filter_id} failed: {exc}"
        if record is not None and record.detector_type:
            return record.detector_type, filter_id, SOURCE_SVO, ""
        return None, filter_id, SOURCE_ASSUMED, f"SVO has no DetectorType for {filter_id}"

    facility = facility or path.parent.parent.name
    candidates = _candidate_filter_ids(path, facility, client)
    if not candidates:
        return None, None, SOURCE_ASSUMED, f"no SVO filter with band '{path.stem}' under facility '{facility}'"

    local_curve = _read_curve(path)
    matches: List[SVOFilterRecord] = []
    for fid in candidates:
        try:
            record = client.get_filter(fid)
        except Exception:
            continue
        if record is not None and _curves_match(local_curve, (record.wavelength, record.transmission)):
            matches.append(record)

    types = {m.detector_type for m in matches if m.detector_type}
    if len(types) == 1:
        # Several SVO IDs can share an identical curve; any of them identifies it.
        return types.pop(), matches[0].filter_id, SOURCE_SVO_MATCH, ""
    if len(types) > 1:
        ids = ", ".join(m.filter_id for m in matches)
        return None, None, SOURCE_ASSUMED, f"curve matches SVO filters with conflicting detector types ({ids})"
    return None, None, SOURCE_ASSUMED, f"curve does not match any of {', '.join(candidates)}"


def _candidate_filter_ids(path: Path, facility: str, client: SVOMetadataClient) -> List[str]:
    """SVO IDs that could correspond to ``path``; each is later verified by curve match."""
    guesses: List[str] = []
    stem = path.stem
    # <Facility>/<Instrument>/<Band>.dat as written by the SVO grabber
    guesses.append(f"{facility}/{path.parent.name}.{stem}")
    # Files named after their SVO ID with '/' flattened, e.g. SLOAN_SDSS.g.dat
    if "." in stem:
        prefix, band = stem.rsplit(".", 1)
        for i, ch in enumerate(prefix):
            if ch == "_":
                guesses.append(f"{prefix[:i]}/{prefix[i + 1:]}.{band}")

    band = _band_key(stem)
    listed = [fid for fid in client.facility_filter_ids(facility) if _band_key(fid.rsplit(".", 1)[-1]) == band]

    candidates: List[str] = []
    for fid in guesses:
        try:
            if client.get_filter(fid) is not None and fid not in candidates:
                candidates.append(fid)
        except Exception:
            continue
    for fid in listed:
        if fid not in candidates:
            candidates.append(fid)
    return candidates


def tag_filter_file(
    path: PathLike,
    *,
    client: Optional[SVOMetadataClient] = None,
    facility: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> DetectorTypeResult:
    """Ensure ``path`` has a ``DetectorType`` header, resolving it if necessary.

    Files already tagged from an authoritative source are left alone unless
    ``force`` is set.  Files previously tagged ``assumed`` are always retried.
    """
    path = Path(path)
    client = client or SVOMetadataClient()
    header = read_filter_header(path)
    existing = normalize_detector_type(header.get("DetectorType"))
    existing_source = header.get("DetectorTypeSource", "")
    complete = not header.get("FilterID") or bool(header.get("WavelengthUnit") and header.get("Components"))
    if existing and existing_source != SOURCE_ASSUMED and complete and not force:
        return DetectorTypeResult(path, existing, existing_source or "header", header.get("FilterID"))

    detector_type, filter_id, source, note = resolve_detector_type(path, client=client, facility=facility)
    if detector_type is None:
        if existing:
            # An unavailable catalogue cannot replace an explicit source
            # assumption with the historical energy default.
            return DetectorTypeResult(path, existing, existing_source or "header", header.get("FilterID"), note=note)
        detector_type, source = ENERGY, SOURCE_ASSUMED
    elif source == SOURCE_SVO and existing_source == SOURCE_SVO_MATCH:
        # The FilterID itself came from a curve match; keep that provenance.
        source = SOURCE_SVO_MATCH

    wavelength_unit = header.get("WavelengthUnit")
    published_components = None
    if filter_id and source != SOURCE_ASSUMED:
        try:
            record = client.get_filter(filter_id)
        except Exception:
            record = None
        if record is not None and record.wavelength_unit:
            wavelength_unit = record.wavelength_unit
        if record is not None and not header.get("Components"):
            published_components = svo_components_header(record.components)
    new_fields: Dict[str, Optional[str]] = dict(header)
    new_fields.update({
        "FilterID": filter_id,
        "DetectorType": detector_type,
        "DetectorTypeSource": source,
        "WavelengthUnit": wavelength_unit,
    })
    if published_components:
        new_fields.update(published_components)
    changed = any(
        (header.get(k) or None) != (new_fields.get(k) or None)
        for k in DETECTOR_KEYS + ("Components", "ComponentsSource")
    )
    if changed and not dry_run:
        write_filter_header(path, new_fields)
    return DetectorTypeResult(path, detector_type, source, filter_id, changed=changed, note=note)


def iter_filter_files(root: PathLike) -> Iterable[Path]:
    """Yield every ``.dat`` filter file under ``root`` (or ``root`` itself)."""
    root = Path(root)
    if root.is_file():
        yield root
        return
    yield from sorted(p for p in root.rglob("*.dat") if p.is_file())


def tag_filter_tree(
    root: PathLike,
    *,
    client: Optional[SVOMetadataClient] = None,
    force: bool = False,
    dry_run: bool = False,
    progress: Optional[Callable[[DetectorTypeResult], None]] = None,
) -> List[DetectorTypeResult]:
    """Tag every filter file below ``root``; see :func:`tag_filter_file`."""
    root = Path(root)
    client = client or SVOMetadataClient()
    results = []
    for path in iter_filter_files(root):
        facility = None
        if root.is_dir():
            rel = path.relative_to(root).parts
            # Layout is <root>/<Facility>/<Instrument>/<Band>.dat
            facility = rel[0] if len(rel) >= 3 else None
        result = tag_filter_file(path, client=client, facility=facility, force=force, dry_run=dry_run)
        results.append(result)
        if progress is not None:
            progress(result)
    return results
