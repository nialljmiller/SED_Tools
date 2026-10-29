"""HST components from STScI's synphot throughput tables.

STScI describes every HST observing mode as a chain of component throughput
tables (``comp/<instrument>/*_syn.fits``) via the master graph table (TMG)
and component lookup table (TMC).  For each band, the chain for its obsmode
(e.g. ``acs,wfc1,f606w``) is walked and its components grouped:

* ``filter`` - the component named after the band (e.g. ``acs_f606w_wfc``)
* ``qe``     - the detector response (``*_ccd*``, ``*_qe``, ``*dqe*``,
  ``*_mama*``) together with STScI's empirical detector-side corrections
  (``*_cor``, ``*_tds*``), so filter x optics x qe equals synphot's total
* ``optics`` - everything else: OTA, pick-off and relay mirrors, windows

Graph and component tables are taken from the newest files in
https://ssb.stsci.edu/trds/mtab/ unless pinned.  Individual component files
can be pinned per instrument (:data:`PINNED`) to the release the published
curves were computed with.
"""

from __future__ import annotations

import datetime
import io
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..filter_components import FILTER, OPTICS, QE, InstrumentManifest, add_component
from ..filter_metadata import PHOTON, SVOMetadataClient
from ._original import CONNECT_TIMEOUT

logger = logging.getLogger(__name__)

TRDS = "https://ssb.stsci.edu/trds"
FACILITY = "HST"

#: SVO instrument key -> obsmode prefix (detector keywords)
MODES: Dict[str, str] = {
    "ACS_WFC": "acs,wfc1",
    "ACS_HRC": "acs,hrc",
    "ACS_SBC": "acs,sbc",
    "WFC3_UVIS1": "wfc3,uvis1",
    "WFC3_UVIS2": "wfc3,uvis2",
    "WFC3_IR": "wfc3,ir",
    "WFPC2-PC": "wfpc2,1",
    "WFPC2-WF": "wfpc2,3",
    "NICMOS1": "nicmos,1",
    "NICMOS2": "nicmos,2",
    "NICMOS3": "nicmos,3",
}

#: Component files pinned per instrument, as {component: file in the same TRDS
#: directory}.  STScI revises ACS/SBC's time-dependent sensitivity (TDS) every
#: year; each file's default column extrapolates to a future epoch, and the
#: factor is achromatic per filter.  The published SBC curves (SVO) equal
#: synphot with the TDS release of 2021-04-25 (``*_tds_def_004``, default
#: extrapolated to 2023-04-01) to 5e-6 of the peak in every filter, so the
#: components use that release and reproduce the published totals.  The
#: prisms' TDS is 1 in every release.
SBC_TDS_RELEASE = "004"
PINNED: Dict[str, Dict[str, str]] = {
    "ACS_SBC": {f"acs_{f}_tds_def": f"acs_{f}_tds_def_{SBC_TDS_RELEASE}_syn.fits"
                for f in ("f115lp", "f122m", "f125lp", "f140lp", "f150lp", "f165lp")},
}

_QE_PATTERN = re.compile(r"(ccd|_qe\b|_qe$|dqe|mama)")
_CORRECTION = re.compile(r"(_cor$|_tds)")


def classify(component: str, band: str) -> str:
    """Role of an HST component within a band's chain."""
    name = component.lower()
    if _CORRECTION.search(name):
        return QE  # empirical sensitivity corrections, even when named after the filter
    if re.search(rf"(^|_|\d){re.escape(band.lower())}(_|$)", name):
        return FILTER
    if _QE_PATTERN.search(name):
        return QE
    return OPTICS


class SynphotTables:
    """Graph traversal over the HST TMG/TMC tables."""

    def __init__(self, session=None, tmg: Optional[str] = None, tmc: Optional[str] = None):
        from astropy.io import fits

        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.tmg_name = tmg or self._latest("tmg")
        self.tmc_name = tmc or self._latest("tmc")
        g = fits.open(io.BytesIO(self._get(f"{TRDS}/mtab/{self.tmg_name}")))[1].data
        c = fits.open(io.BytesIO(self._get(f"{TRDS}/mtab/{self.tmc_name}")))[1].data
        self.innode = np.asarray(g["INNODE"])
        self.outnode = np.asarray(g["OUTNODE"])
        self.keyword = np.array([k.strip().lower() for k in g["KEYWORD"]])
        self.compname = np.array([k.strip().lower() for k in g["COMPNAME"]])
        self.files = {k.strip().lower(): f.strip() for k, f in zip(c["COMPNAME"], c["FILENAME"])}
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _get(self, url: str) -> bytes:
        resp = self.session.get(url, timeout=(CONNECT_TIMEOUT, 300))
        resp.raise_for_status()
        return resp.content

    def _latest(self, kind: str) -> str:
        html = self._get(f"{TRDS}/mtab/").decode("utf-8", "replace")
        rows = re.findall(rf'href="([^"]+_{kind}\.fits)"[^\n]*?(\d{{2}}-\w{{3}}-\d{{4}} \d{{2}}:\d{{2}})', html)
        if not rows:
            raise RuntimeError(f"No *_{kind}.fits found at {TRDS}/mtab/")
        return max(rows, key=lambda r: datetime.datetime.strptime(r[1], "%d-%b-%Y %H:%M"))[0]

    def chain(self, obsmode: str) -> List[str]:
        keys = {k.strip().lower() for k in obsmode.split(",")}
        node = int(self.innode.min())
        used, comps = set(), []
        while True:
            idx = np.nonzero(self.innode == node)[0]
            if idx.size == 0:
                break
            pick = next((i for i in idx if self.keyword[i] in keys), None)
            if pick is None:
                pick = next((i for i in idx if self.keyword[i] == "default"), None)
                if pick is None:
                    raise ValueError(f"{obsmode}: no branch at graph node {node}")
            else:
                used.add(self.keyword[pick])
            if self.compname[pick] != "clear":
                comps.append(str(self.compname[pick]))
            if self.outnode[pick] == node:
                break
            node = int(self.outnode[pick])
        unused = keys - used
        if unused:
            raise ValueError(f"{obsmode}: keywords not in graph: {sorted(unused)}")
        return comps

    def _ref(self, component: str, pins: Optional[Mapping[str, str]] = None) -> Tuple[str, str]:
        prefix, name = self.files[component].split("$", 1)
        return prefix, (pins or {}).get(component, name)

    def throughput(self, component: str, pins: Optional[Mapping[str, str]] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
        from astropy.io import fits

        prefix, name = self._ref(component, pins)
        key = f"{prefix}${name}"
        if key not in self._cache:
            subdir = re.sub(r"^cr|comp$", "", prefix)
            data = fits.open(io.BytesIO(self._get(f"{TRDS}/comp/{subdir}/{name}")))[1].data
            wl = np.asarray(data["WAVELENGTH"], dtype=float)
            # Some tables (e.g. the NICMOS filters) carry small negative
            # measurement noise; a transmission cannot be negative, and the
            # published totals have none.
            tr = np.maximum(np.asarray(data["THROUGHPUT"], dtype=float), 0.0)
            order = np.argsort(wl)
            self._cache[key] = (wl[order], tr[order])
        return self._cache[key]

    def source(self, component: str, pins: Optional[Mapping[str, str]] = None) -> str:
        name = self._ref(component, pins)[1]
        pinned = component in (pins or {})
        return f"STScI synphot {name} ({self.tmg_name}{'; pinned release' if pinned else ''})"


def _pin_note(tables: SynphotTables, pins: Mapping[str, str], used: Iterable[str]) -> str:
    """How the newest TRDS files differ from the pinned ones (both achromatic)."""
    parts = []
    for comp in sorted(set(used) & set(pins)):
        new = tables.throughput(comp)[1]
        old = tables.throughput(comp, pins)[1]
        new, old = new[new > 0], old[old > 0]
        newest = tables.files[comp].split("$", 1)[1]
        if new.size and old.size and np.ptp(new) == 0 and np.ptp(old) == 0:
            parts.append(f"{newest} = {new[0] / old[0]:.4f} x {pins[comp]}")
        else:
            parts.append(f"{newest} differs from {pins[comp]} (not achromatic)")
    return "; ".join(parts)


def _tail_beyond_published(path: Path, wl: np.ndarray, total: np.ndarray) -> float:
    """Largest total outside the published curve's wavelength range, relative to its peak."""
    try:
        pw, pt = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1), unpack=True)
    except ValueError as exc:
        raise ValueError(f"{path}: cannot read the published curve: {exc}") from exc
    out = (wl < pw.min()) | (wl > pw.max())
    return float(total[out].max() / pt.max()) if out.any() and pt.max() > 0 else 0.0


def _product(tables: SynphotTables, comps: Sequence[str],
             pins: Optional[Mapping[str, str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Product of components on their merged grid, over their common range."""
    curves = [tables.throughput(c, pins) for c in comps]
    lo = max(c[0][0] for c in curves)
    hi = min(c[0][-1] for c in curves)
    grid = np.unique(np.concatenate([c[0][(c[0] >= lo) & (c[0] <= hi)] for c in curves]))
    out = np.ones_like(grid)
    for wl, tr in curves:
        out *= np.interp(grid, wl, tr)
    return grid, out


def _split_chain(tables: SynphotTables, prefix: str, band: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """A band's synphot components in chain order, and the same grouped by role.

    Raises ValueError if the graph has no chain for the band, or the chain
    does not have exactly one filter and at least one qe component.
    """
    comps = tables.chain(f"{prefix},{band.lower()}")
    roles = {FILTER: [], OPTICS: [], QE: []}
    for comp in comps:
        roles[classify(comp, band)].append(comp)
    if len(roles[FILTER]) != 1 or not roles[QE]:
        raise ValueError(f"unclassifiable chain {comps}")
    return comps, roles


def _notes(tables: SynphotTables, prefix: str, skipped: Mapping[str, str], tails: Mapping[str, float],
           pins: Mapping[str, str], used: set) -> str:
    """Manifest notes: the obsmode, bands not decomposed, the unpublished tail and any pins."""
    notes = (
        f"Components from STScI synphot obsmode '{prefix},<band>'. QE includes STScI's empirical "
        "detector-side corrections (*_cor, *_tds*) so filter+optics+qe reproduces synphot's total."
        + (f" Not decomposed: {', '.join(sorted(skipped))}." if skipped else "")
    )
    widest = max(tails, key=tails.get) if tails else None
    if widest and tails[widest] > 0:
        notes += (
            " The components extend beyond the wavelength range of the published curves, which "
            f"omit that tail; there the composed product reaches up to {tails[widest]:.1e} of the "
            f"published peak ({widest}), which matters for sources much brighter there."
        )
    if pins and used & set(pins):
        notes += (
            " Pinned to the release the published curves were computed with: "
            + ", ".join(sorted(pins[c] for c in used & set(pins)))
            + ". The newest TRDS files differ by an achromatic factor per band ("
            + _pin_note(tables, pins, used) + "); for the current calibration use synphot itself."
        )
    return notes


def _build_instrument(inst: str, prefix: str, inst_dir: Path, bands: Sequence[str],
                      tables: SynphotTables) -> None:
    """Write the components of one instrument's published ``bands``, and its manifest."""
    manifest = InstrumentManifest.load(inst_dir)
    manifest.components = {}
    pins = PINNED.get(inst, {})
    skipped, used, tails = {}, set(), {}
    for band in bands:
        try:
            comps, roles = _split_chain(tables, prefix, band)
        except ValueError as exc:
            # Ramp, quad and grism bands have no single chain; the notes list them.
            skipped[band] = str(exc)
            continue
        used.update(comps)
        for role, comps_for_role in roles.items():
            if not comps_for_role:
                continue
            wl, tr = _product(tables, comps_for_role, pins)
            add_component(
                inst_dir, role, wl, tr, band=band, name=band,
                source="; ".join(tables.source(c, pins) for c in comps_for_role), manifest=manifest,
            )
        tails[band] = _tail_beyond_published(inst_dir / f"{band}.dat", *_product(tables, comps, pins))
    manifest.detector_type = PHOTON
    manifest.ground_based = False
    manifest.outside_range = "zero"  # synphot evaluates throughput tables as zero outside their range
    manifest.sources["synphot"] = f"{TRDS} ({tables.tmg_name}, {tables.tmc_name}); obsmode {prefix},<band>"
    manifest.notes = _notes(tables, prefix, skipped, tails, pins, used)
    manifest.save(inst_dir)
    if skipped:  # expected on every build of ACS_WFC, ACS_HRC, WFC3_UVIS1, WFPC2-PC; the notes list them
        logger.info("HST/%s: %d bands not decomposed into components: %s", inst, len(skipped), skipped)


def build(
    filter_root: Union[str, Path],
    *,
    instruments: Optional[Iterable[str]] = None,
    session=None,
    svo_client: Optional[SVOMetadataClient] = None,
    tables: Optional[SynphotTables] = None,
) -> List[Path]:
    """Populate components for the HST imagers in :data:`MODES`.

    Instruments without published curves are left out; once the others are
    built, a FileNotFoundError lists them.  ``svo_client`` is accepted for
    compatibility and is not used.
    """
    filter_root = Path(filter_root)
    tables = tables or SynphotTables(session=session)
    built, unpublished = [], []
    for inst in instruments or MODES:
        prefix = MODES[inst]
        inst_dir = filter_root / FACILITY / inst
        bands = sorted(p.stem for p in inst_dir.glob("*.dat"))
        if not bands:
            unpublished.append(inst)
            continue
        _build_instrument(inst, prefix, inst_dir, bands, tables)
        built.append(inst_dir)
    if unpublished:
        raise FileNotFoundError(
            f"{filter_root / FACILITY} has no published curves for {', '.join(unpublished)}; fetch them "
            f"first (njm mirror or SVO). Built: {', '.join(p.name for p in built) or 'none'}.")
    return built
