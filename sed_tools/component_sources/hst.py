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
https://ssb.stsci.edu/trds/mtab/ unless pinned.
"""

from __future__ import annotations

import datetime
import io
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..filter_components import FILTER, OPTICS, QE, InstrumentManifest, add_component
from ..filter_metadata import PHOTON, SOURCE_SVO, SVOMetadataClient, svo_components_header, write_filter_file

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
        resp = self.session.get(url, timeout=300)
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

    def throughput(self, component: str) -> Tuple[np.ndarray, np.ndarray]:
        from astropy.io import fits

        if component not in self._cache:
            ref = self.files[component]
            prefix, name = ref.split("$", 1)
            subdir = re.sub(r"^cr|comp$", "", prefix)
            data = fits.open(io.BytesIO(self._get(f"{TRDS}/comp/{subdir}/{name}")))[1].data
            wl = np.asarray(data["WAVELENGTH"], dtype=float)
            tr = np.asarray(data["THROUGHPUT"], dtype=float)
            order = np.argsort(wl)
            self._cache[component] = (wl[order], tr[order])
        return self._cache[component]

    def source(self, component: str) -> str:
        return f"STScI synphot {self.files[component].split('$', 1)[1]} ({self.tmg_name})"


def _product(tables: SynphotTables, comps: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Product of components on their merged grid, over their common range."""
    curves = [tables.throughput(c) for c in comps]
    lo = max(c[0][0] for c in curves)
    hi = min(c[0][-1] for c in curves)
    grid = np.unique(np.concatenate([c[0][(c[0] >= lo) & (c[0] <= hi)] for c in curves]))
    out = np.ones_like(grid)
    for wl, tr in curves:
        out *= np.interp(grid, wl, tr)
    return grid, out


def build(
    filter_root: Union[str, Path],
    *,
    instruments: Optional[Iterable[str]] = None,
    session=None,
    svo_client: Optional[SVOMetadataClient] = None,
    tables: Optional[SynphotTables] = None,
) -> List[Path]:
    """Populate components for the HST imagers in :data:`MODES`."""
    filter_root = Path(filter_root)
    tables = tables or SynphotTables(session=session)
    svo_client = svo_client or SVOMetadataClient()
    built = []
    for inst in instruments or MODES:
        prefix = MODES[inst]
        inst_dir = filter_root / FACILITY / inst
        inst_dir.mkdir(parents=True, exist_ok=True)
        bands = sorted(p.stem for p in inst_dir.glob("*.dat"))
        if not bands:
            logger.warning("HST/%s: no published curves present; fetch them first", inst)
            continue
        manifest = InstrumentManifest.load(inst_dir)
        manifest.components = {}
        skipped = {}
        for band in bands:
            try:
                comps = tables.chain(f"{prefix},{band.lower()}")
            except (ValueError, KeyError) as exc:
                skipped[band] = str(exc)
                continue
            roles = {FILTER: [], OPTICS: [], QE: []}
            for comp in comps:
                roles[classify(comp, band)].append(comp)
            if len(roles[FILTER]) != 1 or not roles[QE]:
                skipped[band] = f"unclassifiable chain {comps}"
                continue
            for role, comps_for_role in roles.items():
                if not comps_for_role:
                    continue
                wl, tr = _product(tables, comps_for_role)
                add_component(
                    inst_dir, role, wl, tr, band=band, name=band,
                    source="; ".join(tables.source(c) for c in comps_for_role), manifest=manifest,
                )
        manifest.detector_type = PHOTON
        manifest.ground_based = False
        manifest.outside_range = "zero"  # synphot evaluates throughput tables as zero outside their range
        manifest.sources["synphot"] = f"{TRDS} ({tables.tmg_name}, {tables.tmc_name}); obsmode {prefix},<band>"
        manifest.notes = (
            f"Components from STScI synphot obsmode '{prefix},<band>'. QE includes STScI's empirical "
            "detector-side corrections (*_cor, *_tds*) so filter+optics+qe reproduces synphot's total."
            + (f" Not decomposed: {', '.join(sorted(skipped))}." if skipped else "")
        )
        manifest.save(inst_dir)
        if skipped:
            logger.info("HST/%s skipped %d bands: %s", inst, len(skipped), skipped)
        built.append(inst_dir)
    return built
