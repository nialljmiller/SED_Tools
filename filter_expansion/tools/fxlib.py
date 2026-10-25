"""Filter-expansion builder: stage new instruments in the NJM mirror layout.

Every curve is written with sed_tools.filter_metadata.write_filter_file
(SED_Tools 0.4.1), so it follows the format contract in
agent-share/projects/sed-tools-work/filter-expansion/BRIEF.md:

    stage/filters/<Facility>/<Instrument>/
        <Instrument>                 index file, one <band>.dat per line
        <band>.dat                   published curve, metadata on the column row
        manifest.json                InstrumentManifest
        components/<role>/<x>.dat    separately published components
    stage/provenance/<Facility>__<Instrument>.json

Recipes (tools/recipes/*.py) download sources into downloads/<Facility>/,
parse them, and call build_instrument().  Nothing here touches the mirror.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sed_tools.filter_components import (InstrumentManifest, add_component,
                                         parse_components, recipe_name)
from sed_tools.filter_metadata import (DETECTOR_TYPES, read_filter_header,
                                       released_mesa_reads, write_filter_file)

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "downloads"
STAGE = ROOT / "stage"
C_ANGSTROM_PER_S = 2.99792458e18

# Header values for curves taken from the instrument team rather than SVO.
ORIGINAL = "original"
SVO = "svo"

_UNIT_SCALE = {"angstrom": 1.0, "a": 1.0, "aa": 1.0, "nm": 10.0, "micron": 1e4, "um": 1e4, "mm": 1e7}


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------
def fetch(url: str, facility: str, name: Optional[str] = None, *, timeout: float = 120) -> Path:
    """Download ``url`` once into downloads/<facility>/ and return the local path."""
    target_dir = DOWNLOADS / facility
    target_dir.mkdir(parents=True, exist_ok=True)
    name = name or re.sub(r"[^A-Za-z0-9._-]", "_", url.rstrip("/").rsplit("/", 1)[-1].split("?")[0])
    path = target_dir / name
    if not path.exists() or path.stat().st_size == 0:
        req = urllib.request.Request(url, headers={"User-Agent": "SED_Tools-filter-expansion"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        head = data[:512].lstrip().lower()
        if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
            raise ValueError(f"{url} returned HTML, not data")
        path.write_bytes(data)
    return path


def svo_curve(filter_id: str, facility: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """Fetch an SVO FPS VOTable; return wavelength (Å), transmission and its PARAMs."""
    from astropy.io.votable import parse
    url = f"https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={filter_id}"
    path = fetch(url, facility, filter_id.replace("/", "__") + ".xml", timeout=30)
    vot = parse(str(path))
    table = vot.get_first_table()
    params = {p.name: p.value for p in table.params}
    unit = str(params.get("WavelengthUnit", "Angstrom"))
    wl = to_angstrom(np.asarray(table.array["Wavelength"], float), unit)
    return wl, np.asarray(table.array["Transmission"], float), {k: str(v) for k, v in params.items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------
def to_angstrom(x: Sequence[float], unit: str) -> np.ndarray:
    """Convert wavelengths, frequencies (GHz/Hz) or wavenumbers (1/cm) to Å."""
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


@dataclass
class CleanReport:
    n_in: int
    n_out: int
    dropped_nonfinite: int = 0
    merged_duplicates: int = 0
    clipped_negative: int = 0
    min_negative: float = 0.0
    scaled_by: float = 1.0

    def note(self) -> str:
        parts = []
        if self.clipped_negative:
            parts.append(f"{self.clipped_negative} negative samples (min {self.min_negative:.3g}) set to 0")
        if self.dropped_nonfinite:
            parts.append(f"{self.dropped_nonfinite} undefined samples dropped")
        if self.merged_duplicates:
            parts.append(f"{self.merged_duplicates} duplicate wavelengths averaged")
        if self.scaled_by != 1.0:
            parts.append(f"scaled by {self.scaled_by:.6g} (peak normalised)")
        return "; ".join(parts)


def clean_curve(wl, tr, *, sentinel: Optional[float] = None, normalize: bool = False
                ) -> Tuple[np.ndarray, np.ndarray, CleanReport]:
    """Return a curve that meets the format contract, with a report of every change.

    Drops non-finite (and ``sentinel``) samples, sorts by wavelength, averages
    repeated wavelengths, sets negative transmissions (measurement noise) to 0
    and optionally scales the peak to 1 (for effective areas / relative
    responses; synthetic photometry is unaffected by a constant factor).
    """
    wl = np.asarray(wl, dtype=float)
    tr = np.asarray(tr, dtype=float)
    rep = CleanReport(n_in=len(wl), n_out=0)
    ok = np.isfinite(wl) & np.isfinite(tr) & (wl > 0)
    if sentinel is not None:
        ok &= ~np.isclose(tr, sentinel)
    rep.dropped_nonfinite = int((~ok).sum())
    wl, tr = wl[ok], tr[ok]
    order = np.argsort(wl, kind="stable")
    wl, tr = wl[order], tr[order]
    uniq, inv, counts = np.unique(wl, return_inverse=True, return_counts=True)
    if len(uniq) != len(wl):
        rep.merged_duplicates = int(len(wl) - len(uniq))
        tr = np.bincount(inv, weights=tr) / counts
        wl = uniq
    neg = tr < 0
    if neg.any():
        rep.clipped_negative = int(neg.sum())
        rep.min_negative = float(tr.min())
        tr = np.where(neg, 0.0, tr)
    if normalize:
        peak = float(tr.max())
        if peak > 0:
            rep.scaled_by = 1.0 / peak
            tr = tr / peak
    rep.n_out = len(wl)
    if rep.n_out < 2 or not (tr > 0).any():
        raise ValueError("curve has fewer than 2 samples or no positive transmission")
    return wl, tr, rep


def safe_band(name: str) -> str:
    """A band name usable as ``<band>.dat``: exactly one dot in the filename."""
    band = re.sub(r"[^A-Za-z0-9_+-]", "_", name.replace(".", "p")).strip("_")
    if not band:
        raise ValueError(f"empty band name from {name!r}")
    return band


def pivot_wavelength(wl, tr, detector_type: str) -> float:
    """Pivot wavelength (Å) of a photon- or energy-counting response."""
    wl = np.asarray(wl, float)
    tr = np.asarray(tr, float)
    w = tr * wl if detector_type == "photon" else tr
    return float(np.sqrt(np.trapezoid(w, wl) / np.trapezoid(w / wl**2, wl)))


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------
@dataclass
class Band:
    name: str
    wavelength: np.ndarray          # Å, cleaned
    transmission: np.ndarray
    detector_type: str              # photon | energy
    detector_source: str            # original | svo
    components: str                 # e.g. "filter+optics+qe+atmosphere"
    components_source: str          # original | svo
    sources: str                    # URL(s) of the numeric curve, ' | ' separated
    filter_id: Optional[str] = None  # "<Facility>/<Instrument>.<band>" by default
    svo_id: Optional[str] = None     # SVO counterpart for the pivot spot check
    clean_note: str = ""
    ref_pivot: Optional[Tuple[float, str]] = None  # (pivot Å, citation) from the source team

    def __post_init__(self):
        if self.detector_type not in DETECTOR_TYPES:
            raise ValueError(f"{self.name}: detector_type must be photon or energy")
        self.components = recipe_name(parse_components(self.components))


@dataclass
class Component:
    role: str                       # filter | optics | qe | filter+qe ...
    wavelength: np.ndarray          # Å
    transmission: np.ndarray        # fraction, peak <= 1.5
    band: str = "*"                 # band name or "*" (shared)
    source: str = ""


def build_instrument(facility: str, instrument: str, bands: Sequence[Band], *,
                     notes: str, components: Sequence[Component] = (),
                     outside_range: str = "zero", raw_files: Sequence[Path] = (),
                     extra_provenance: Optional[dict] = None, root: Path = STAGE,
                     ground_based: Optional[bool] = None) -> Path:
    """Write one instrument into ``root``/filters and record its provenance.

    ``ground_based`` is required when SED_Tools' facility_sites does not know the
    facility (e.g. OGLE, Magellan, VST, KMTNet in 0.4.1); it is stored in the manifest.
    """
    inst_dir = root / "filters" / facility / instrument
    if inst_dir.exists() and any(inst_dir.iterdir()):
        # Re-run: keep the earlier build (nothing is deleted), out of the deploy tree.
        aside = root / "_superseded" / f"{facility}__{instrument}__{time.strftime('%Y%m%dT%H%M%S')}"
        aside.parent.mkdir(parents=True, exist_ok=True)
        inst_dir.rename(aside)
    inst_dir.mkdir(parents=True, exist_ok=True)
    names = [b.name for b in bands]
    if len({n.lower() for n in names}) != len(names):
        raise ValueError(f"{facility}/{instrument}: duplicate band names {names}")
    for b in bands:
        if safe_band(b.name) != b.name:
            raise ValueError(f"band {b.name!r} is not filename-safe; use safe_band()")
        write_filter_file(
            inst_dir / f"{b.name}.dat", b.wavelength, b.transmission,
            detector_type=b.detector_type, source=b.detector_source,
            filter_id=b.filter_id or f"{facility}/{instrument}.{b.name}",
            wavelength_unit="Angstrom",
            extra={"Components": b.components, "ComponentsSource": b.components_source,
                   "Sources": b.sources})
    (inst_dir / instrument).write_text("".join(f"{n}.dat\n" for n in sorted(names)), encoding="utf-8")
    manifest = InstrumentManifest.from_published(inst_dir)
    manifest.facility, manifest.instrument = facility, instrument
    if ground_based is not None:
        manifest.ground_based = ground_based
    if manifest.ground_based is None:
        raise ValueError(f"{facility} is unknown to facility_sites; pass ground_based=True/False")
    for c in components:
        add_component(inst_dir, c.role, c.wavelength, c.transmission, band=c.band,
                      source=c.source, manifest=manifest)
    manifest.notes = notes
    manifest.outside_range = outside_range
    manifest.detector_type_sources = {b.name: b.detector_source for b in bands}
    for b in bands:
        manifest.sources.setdefault("published", b.sources.split(" | ")[0])
    manifest.save(inst_dir)
    prov = {
        "facility": facility, "instrument": instrument, "notes": notes,
        "bands": {b.name: {"sources": b.sources, "svo_id": b.svo_id, "detector_type": b.detector_type,
                           "detector_source": b.detector_source, "components": b.components,
                           "components_source": b.components_source, "cleaning": b.clean_note,
                           "ref_pivot": list(b.ref_pivot) if b.ref_pivot else None,
                           "n": int(len(b.wavelength)),
                           "pivot_A": round(pivot_wavelength(b.wavelength, b.transmission, b.detector_type), 2)}
                  for b in bands},
        "raw_files": {str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else str(p): sha256(p)
                      for p in raw_files},
        **(extra_provenance or {}),
    }
    prov_dir = root / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / f"{facility}__{instrument}.json").write_text(json.dumps(prov, indent=2) + "\n")
    bad = [p for p in inst_dir.rglob("*.dat") if not released_mesa_reads(p)]
    if bad:
        raise RuntimeError(f"released MESA would misread {bad}")
    return inst_dir


def make_band(name, wl, tr, *, unit, detector_type, detector_source, components,
              components_source, sources, svo_id=None, sentinel=None, normalize=False,
              filter_id=None, ref_pivot=None) -> Band:
    """Convert units, clean, and wrap a parsed curve as a Band."""
    wl_a = to_angstrom(wl, unit)
    w, t, rep = clean_curve(wl_a, tr, sentinel=sentinel, normalize=normalize)
    return Band(safe_band(name), w, t, detector_type, detector_source, components,
                components_source, sources, filter_id=filter_id, svo_id=svo_id, clean_note=rep.note(),
                ref_pivot=ref_pivot)
