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
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from .compose import read_curve
from .curve_metadata import (DETECTOR_SOURCE_PRECEDENCE, SOURCE_ASSUMED, PathLike, normalize_detector_type,
                             read_filter_header)
from .manifest import FILTER, MANIFEST_NAME, ComponentError, filter_sibling_base, parse_components

#: Every ``DetectorTypeSource`` SED_Tools writes.
DETECTOR_SOURCES = DETECTOR_SOURCE_PRECEDENCE + ("user", "composed", "component")
#: A curve is a component when it follows it to SAME_CURVE of the component's peak on its own wavelengths and
#: leaves out only wings below WING of that peak (copies are often trimmed there).
SAME_CURVE, WING = 1e-4, 1e-2

Problem = Tuple[Path, str]


def problems(filters_dir: PathLike, against: Optional[PathLike] = None) -> Iterator[Problem]:
    """Every rule broken in ``filters_dir`` (``Facility/Instrument/*.dat``), as (curve or folder, message)."""
    for folder in sorted(p for p in Path(filters_dir).glob("*/*") if p.is_dir()):
        headers = {p.stem: read_filter_header(p) for p in sorted(folder.glob("*.dat"))}
        yield from _label_problems(folder, headers)
        yield from _sibling_problems(folder, headers)
        yield from _component_problems(folder, headers)
        if against is not None:
            yield from _regressions(folder, headers, Path(against) / folder.parent.name / folder.name)


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
