"""Two bands of one served set that hold the same curve, compared by value, not by bytes (Nill, 25 Nov 2026: the
byte-level dedupe of 29 Oct missed LCO/e2v i, which held the r curve at another decimal precision).

``problems`` is the class guard filter_info.check_manifests runs: two curves of one set whose samples agree at every
wavelength to 1e-6 of the curve's peak (the same number of samples, the same wavelengths to 1e-6 A) are one curve under
two names when their data rows differ as text (bytes equal is the 29 Oct dedupe's class). Archive copies (``archived_bands``) and curves that are components are not compared.
"""
import hashlib
from pathlib import Path

import numpy as np

TOLERANCE = 1e-6


def _samples(path):
    rows = []
    for line in Path(path).read_text().split("\n")[1:]:
        parts = line.split(",")
        if len(parts) >= 2 and not line.startswith("#"):
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return np.array(rows) if rows else np.zeros((0, 2))


def _rows(path):
    return [l for l in Path(path).read_text().split("\n")[1:] if l.strip()]


def problems(set_dir, manifest):
    set_dir = Path(set_dir)
    archived = set(manifest.get("archived_bands") or [])
    groups = {}
    for curve in sorted(set_dir.glob("*.dat")):
        if curve.stem in archived:
            continue
        d = _samples(curve)
        if d.shape[0] < 2 or not d[:, 1].max() > 0:
            continue
        key = (d.shape[0], round(float(d[0, 0]), 3), round(float(d[-1, 0]), 3))
        groups.setdefault(key, []).append((curve.stem, d))
    for members in groups.values():
        for i, (a, da) in enumerate(members):
            for b, db in members[i + 1:]:
                if (_rows(set_dir / f"{a}.dat") != _rows(set_dir / f"{b}.dat")  # bytes equal: the 29 Oct dedupe's class
                        and np.allclose(da[:, 0], db[:, 0], rtol=0, atol=1e-6)
                        and np.max(np.abs(da[:, 1] - db[:, 1])) <= TOLERANCE * max(da[:, 1].max(), db[:, 1].max())):
                    yield f"{set_dir}: {a}.dat and {b}.dat hold the same curve (to {TOLERANCE:g} of the peak)"
