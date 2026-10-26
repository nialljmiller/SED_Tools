"""Helpers for builders that add components to instruments already published.

CFHT MegaPrime/MegaCam, ERIS/NIX and 2MASS are served from SVO's (or the
mirror's) published curves; their teams also publish the separate factors.
These builders only add ``components/`` and ``manifest.json``; the published
``.dat`` files and the MESA index are never written or changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ..filter_components import InstrumentManifest, add_component


def published_bands(inst_dir: Path) -> set:
    """Band names of the published curves; refuse an empty instrument."""
    bands = {p.stem for p in inst_dir.glob("*.dat")}
    if not bands:
        raise FileNotFoundError(
            f"{inst_dir} has no published curves; fetch the instrument (njm mirror or SVO) first")
    return bands


def fresh_manifest(inst_dir: Path, facility: str, instrument: str) -> InstrumentManifest:
    """Describe the published curves from their headers, as a clean start.

    Components from an earlier build are dropped from the description (their
    files stay on disk); filter-only published curves are registered as
    their own filter, as :meth:`InstrumentManifest.from_published` does.
    """
    manifest = InstrumentManifest.from_published(inst_dir)
    manifest.facility, manifest.instrument = facility, instrument
    return manifest


EARLIER = " Earlier note: "


def with_earlier_note(inst_dir: Path, notes: str) -> str:
    """``notes`` followed by the note an earlier manifest carried (kept, not rewritten).

    Only a note this builder did not write is carried, once: a rebuild finds
    its own text (and any carried note) and keeps the carried part as is.
    """
    path = inst_dir / "manifest.json"
    if not path.exists():
        return notes
    earlier = InstrumentManifest.load(inst_dir).notes or ""
    if EARLIER in earlier:
        earlier = earlier.split(EARLIER, 1)[1]
    elif earlier[:60] == notes[:60]:
        earlier = ""
    return notes + (EARLIER + earlier if earlier.strip() else "")


def register(manifest: InstrumentManifest, inst_dir: Path, role: str, wl, tr, *,
             bands: Iterable[str], name: str, source: str) -> str:
    """Write one component file and register it for each of ``bands``."""
    rel = add_component(inst_dir, role, wl, tr, name=name, source=source)
    entries = manifest.components.setdefault(role, {})
    for band in bands:
        entries[band] = rel
    manifest.sources.setdefault(role, source)
    return rel


def write_unregistered(inst_dir: Path, role: str, wl, tr, *, name: str, source: str) -> str:
    """An alternative factor kept next to the registered one, not composed by default."""
    return add_component(inst_dir, role, wl, tr, name=name, source=source)


def common_product(curves):
    """Product of (Angstrom, fraction) curves over their common range.

    Each source table is zero outside its own range, so the product exists
    only where every factor is tabulated.  The grid is the union of the
    factors' samples there.
    """
    lo = max(float(c[0][0]) for c in curves)
    hi = min(float(c[0][-1]) for c in curves)
    if hi <= lo:
        raise ValueError("factors do not overlap")
    grid = np.unique(np.concatenate([c[0][(c[0] >= lo) & (c[0] <= hi)] for c in curves]))
    out = np.ones_like(grid)
    for wl, tr in curves:
        out = out * np.interp(grid, wl, tr)
    return grid, out
