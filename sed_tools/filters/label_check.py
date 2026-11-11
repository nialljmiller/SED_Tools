"""The rules a served filter tree's labels must keep; NJM checks them on every refresh, a MESA build against its last tree.

``python -m sed_tools.filters.label_check FILTERS_DIR [--against OLD_FILTERS_DIR]`` prints every
violation and exits 1, or prints a one-line summary and exits 0.  Each rule holds for every instrument:

1. ``DetectorType`` is ``energy`` or ``photon``, and ``DetectorTypeSource`` is a word SED_Tools ranks
   (:data:`DETECTOR_SOURCE_PRECEDENCE`) or sets itself (``user``, ``composed``, ``component``).  A label
   set downstream under another word, such as ``manual``, cannot be ranked against SVO's.
2. ``Components`` names known roles only.
3. A band's filter-only sibling (``<band>_filter`` or ``<band>_fil`` beside ``<band>``, SVO's naming) is
   labelled ``filter`` or not at all.
4. A curve equal to a one-role component its ``manifest.json`` registers is labelled with that role.
5. With ``against``, a curve whose ``DetectorType`` differs from the old tree's has a source that
   outranks the old one (:data:`DETECTOR_SOURCE_PRECEDENCE`), or ``user``: a rebuild cannot undo a
   correction with weaker evidence, nor follow SVO reclassifying a curve (``svo`` to ``svo``) unseen.
6. ``Sources`` is as SED_Tools writes it (:func:`normalize_sources`): distinct references separated by
   ``" | "``, none run together in one item, and no item crediting a reference another item already gives
   (as the Internet Archive ingest's credit sentence did).
7. Curves with the same samples carry the same ``DetectorType``, across the whole tree: the same curve
   served under two instruments cannot be read in two conventions.  Samples are the same when the
   wavelengths agree and the peak-normalised transmissions agree to :data:`SAME_SAMPLES` (float32
   precision, so a copy written at float32 is the same curve).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .compose import read_curve
from .curve_metadata import (DETECTOR_SOURCE_PRECEDENCE, SOURCE_ASSUMED, PathLike, normalize_detector_type,
                             normalize_sources, read_filter_header)
from .manifest import FILTER, MANIFEST_NAME, ComponentError, filter_sibling_base, parse_components

#: Every ``DetectorTypeSource`` SED_Tools writes.
DETECTOR_SOURCES = DETECTOR_SOURCE_PRECEDENCE + ("user", "composed", "component")
#: A curve is a component when it follows it to SAME_CURVE of the component's peak on its own wavelengths and
#: leaves out only wings below WING of that peak (copies are often trimmed there).
SAME_CURVE, WING = 1e-4, 1e-2
#: Rule 7's tolerance on wavelengths (relative) and on peak-normalised transmissions: float32 precision.
SAME_SAMPLES = 1e-6

Problem = Tuple[Path, str]


def problems(filters_dir: PathLike, against: Optional[PathLike] = None) -> Iterator[Problem]:
    """Every rule broken in ``filters_dir`` (``Facility/Instrument/*.dat``), as (curve or folder, message)."""
    labelled: Dict[int, List] = defaultdict(list)  # rule 7: (curve, path, DetectorType, source) by sample count
    for folder in sorted(p for p in Path(filters_dir).glob("*/*") if p.is_dir()):
        headers = {p.stem: read_filter_header(p) for p in sorted(folder.glob("*.dat"))}
        yield from _label_problems(folder, headers)
        yield from _sibling_problems(folder, headers)
        yield from _component_problems(folder, headers)
        if against is not None:
            yield from _regressions(folder, headers, Path(against) / folder.parent.name / folder.name)
        for band, header in headers.items():
            detector = normalize_detector_type(header.get("DetectorType"))
            try:
                curve = read_curve(folder / f"{band}.dat") if detector else None
            except ComponentError:
                continue  # an unreadable curve is no one's copy
            if curve is not None and np.max(curve.transmission, initial=0.0) > 0:
                labelled[len(curve.wavelength)].append((curve, folder / f"{band}.dat", detector,
                                                        header.get("DetectorTypeSource") or "no source"))
    yield from _clashes(Path(filters_dir), labelled)


def _roles(header) -> Tuple[str, ...]:
    try:
        return parse_components(header.get("Components"))
    except ComponentError:
        return ()


def _label_problems(folder: Path, headers) -> Iterator[Problem]:
    for band, header in headers.items():
        path = folder / f"{band}.dat"
        if header.get("DetectorType") and not normalize_detector_type(header["DetectorType"]):
            yield path, f"DetectorType {header['DetectorType']!r} is neither energy nor photon"
        source = header.get("DetectorTypeSource")
        if header.get("DetectorType") and source not in DETECTOR_SOURCES:
            yield path, f"DetectorTypeSource {source!r} is not one of {', '.join(DETECTOR_SOURCES)}"
        try:
            parse_components(header.get("Components"))
        except ComponentError as exc:
            yield path, f"Components {header['Components']!r}: {exc}"
        sources = header.get("Sources")
        if sources and normalize_sources(sources) != sources:
            yield path, f"Sources {sources!r} should read {normalize_sources(sources)!r}"


def _sibling_problems(folder: Path, headers) -> Iterator[Problem]:
    for sibling, header in headers.items():
        band = filter_sibling_base(sibling)
        if band not in headers:
            continue
        if _roles(header) not in ((), (FILTER,)):
            yield (folder / f"{sibling}.dat",
                   f"named as {band}'s filter-only curve but labelled Components={header['Components']}")


def _component_problems(folder: Path, headers) -> Iterator[Problem]:
    path = folder / MANIFEST_NAME
    if not path.is_file():
        return
    registered = json.loads(path.read_text(encoding="utf-8")).get("components") or {}
    components: Dict[Path, str] = {}
    for key, entries in registered.items():
        roles = parse_components(key)
        for rel in (entries or {}).values():
            file = folder / rel
            if len(roles) == 1 and file.parent != folder and file.is_file():
                components[file] = roles[0]
    if not components:
        return
    loaded = {file: read_curve(file) for file in components}
    for band, header in headers.items():
        try:
            curve = read_curve(folder / f"{band}.dat")
        except ComponentError as exc:
            yield folder / f"{band}.dat", str(exc)
            continue
        for file, role in components.items():
            if _roles(header) != (role,) and _same_curve(curve, loaded[file]):
                yield (folder / f"{band}.dat", f"is the registered {role} component {file.relative_to(folder)} "
                                               f"but labelled Components={header.get('Components') or '(none)'}")


def _same_curve(curve, component) -> bool:
    """Whether ``curve`` is ``component`` sampled on its own wavelengths: equal there, with only wings left out."""
    peak = float(np.max(component.transmission))
    if peak <= 0:
        return False
    inside = (component.wavelength >= curve.wavelength[0]) & (component.wavelength <= curve.wavelength[-1])
    if np.any(component.transmission[~inside] > WING * peak):
        return False
    on_curve = np.interp(curve.wavelength, component.wavelength, component.transmission, left=0.0, right=0.0)
    return bool(np.max(np.abs(curve.transmission - on_curve)) <= SAME_CURVE * peak)


def _clashes(filters_dir: Path, labelled: Dict[int, List]) -> Iterator[Problem]:
    """Rule 7: each group of curves with the same samples (:func:`_same_samples`) and more than one DetectorType."""
    for entries in labelled.values():
        groups: List[List] = []
        for entry in entries:
            group = next((g for g in groups if _same_samples(g[0][0], entry[0])), None)
            if group is None:
                groups.append([entry])
            else:
                group.append(entry)
        for group in groups:
            if len({detector for _, _, detector, _ in group}) > 1:
                yield group[0][1], "the same curve carries more than one DetectorType: " + ", ".join(
                    f"{path.relative_to(filters_dir)} ({detector}, {source})" for _, path, detector, source in group)


def _same_samples(a, b) -> bool:
    """Whether curves ``a`` and ``b`` hold the same samples after peak normalisation, to SAME_SAMPLES."""
    if not (np.allclose(a.wavelength[[0, -1]], b.wavelength[[0, -1]], rtol=SAME_SAMPLES, atol=0)
            and np.allclose(a.wavelength, b.wavelength, rtol=SAME_SAMPLES, atol=0)):
        return False
    return bool(np.allclose(a.transmission / np.max(a.transmission), b.transmission / np.max(b.transmission),
                            rtol=0, atol=SAME_SAMPLES))


def _regressions(folder: Path, headers, old_folder: Path) -> Iterator[Problem]:
    for band, header in headers.items():
        old_path = old_folder / f"{band}.dat"
        if not old_path.is_file():
            continue
        old = read_filter_header(old_path)
        before, now = normalize_detector_type(old.get("DetectorType")), normalize_detector_type(header.get("DetectorType"))
        if not before or before == now:
            continue
        source, old_source = header.get("DetectorTypeSource"), old.get("DetectorTypeSource")
        if now and (source == "user" or _outranks(source, old_source or SOURCE_ASSUMED)):
            continue
        yield (folder / f"{band}.dat", f"DetectorType {before} ({old_source or 'no source'}) became "
                                       f"{now or 'unlabelled'} ({source or 'no source'}), which does not outrank it")


def _outranks(source: Optional[str], old: str) -> bool:
    order = DETECTOR_SOURCE_PRECEDENCE
    return source in order and old in order and order.index(source) < order.index(old)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filters_dir", type=Path, help="Facility/Instrument/*.dat tree, e.g. data/filters")
    parser.add_argument("--against", type=Path, help="the tree it replaces, e.g. the last release's filters")
    args = parser.parse_args(argv)
    if not args.filters_dir.is_dir():
        parser.error(f"{args.filters_dir} is not a directory")
    found = list(problems(args.filters_dir, args.against))
    for path, message in found:
        print(f"ERROR {path}: {message}", file=sys.stderr)
    folders = sum(1 for p in args.filters_dir.glob("*/*") if p.is_dir())
    if found:
        print(f"label_check: {len(found)} problems in {folders} instrument folders", file=sys.stderr)
        return 1
    print(f"label_check: the labels of {folders} instrument folders keep every rule")
    return 0


if __name__ == "__main__":
    sys.exit(main())
