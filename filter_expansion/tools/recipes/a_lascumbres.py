"""Las Cumbres Observatory Sinistro and Spectral imager filters: LCO filter_curves zip (codex.csv).

Members are two-column ASCII (wavelength nm, transmission fraction), laboratory filter scans
only (no QE, optics or atmosphere).  The SDSS-type and PSTR (zs, Y) scans contain small
negative measurement noise that clean_curve sets to 0.  Same filter files for both cameras.
"""
import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
from fxlib import DOWNLOADS, ORIGINAL, build_instrument, fetch, make_band

URL = "https://lco.global/documents/2/lco_filter_curves.zip"
PAGE = "https://lco.global/observatory/instruments/filters/"
LOCAL = Path("/home/nill/agent-share/projects/sed-tools-components/round2/claude/ground/evidence/"
             "lco_filter_curves.zip")
SHA = "4e9ecf215af1d1210c43342dd7604492d07cea588239b148836eff6f1bf727f0"
# Published band name (case-insensitively unique; LCO filter codes) -> zip member, codex.csv band.
FILES = {"U": ("bssl-ux-022.txt", "U"), "B": ("bssl-bx-004.txt", "B"), "V": ("bssl-vx-022.txt", "V"),
         "R": ("bssl-rx-007.txt", "R"), "I": ("bssl-ix-023.txt", "I"),
         "up": ("SDSS.up.txt", "u"), "gp": ("SDSS.gp.txt", "g"), "rp": ("SDSS.rp.txt", "r"),
         "ip": ("SDSS.ip.txt", "i"), "zs": ("PSTR-ZS-avg.txt", "zs"), "Y": ("PSTR-YX-avg.txt", "Y")}
SETS = {"Sinistro": ["U", "B", "V", "R", "I", "up", "gp", "rp", "ip", "zs", "Y"],
        "Spectral": ["B", "V", "R", "I", "up", "gp", "rp", "ip", "zs", "Y"]}


def _zip():
    target = DOWNLOADS / "LasCumbres" / "lco_filter_curves.zip"
    if not target.exists() and LOCAL.exists():  # reuse the verified round-2 copy
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LOCAL, target)
    path = fetch(URL, "LasCumbres")
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA:
        raise ValueError("lco_filter_curves.zip differs from the round-2 verified copy")
    return path


def _read(zf, member):
    info = zf.getinfo(member)
    if info.filename.startswith("/") or ".." in info.filename:
        raise ValueError(f"refusing zip member {member!r}")
    rows = []
    for line in zf.read(info).decode("latin-1").splitlines():
        parts = line.split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (IndexError, ValueError):
            continue  # header such as "\lambda [nm]"
    return np.array(rows)


def run():
    zpath = _zip()
    with zipfile.ZipFile(zpath) as zf:
        data = {band: _read(zf, member) for band, (member, _) in FILES.items()}
    mapping = "; ".join(f"{b}={m} (codex band {c})" for b, (m, c) in FILES.items())
    for instrument, names in SETS.items():
        bands = [make_band(n, data[n][:, 0], data[n][:, 1], unit="nm", detector_type="photon",
                           detector_source=ORIGINAL, components="filter", components_source=ORIGINAL,
                           sources=f"{URL}#{FILES[n][0]} | {PAGE}") for n in names]
        cleaned = "; ".join(f"{b.name}: {b.clean_note}" for b in bands if b.clean_note)
        note = (f"LCO {instrument} filters from LCO's lco_filter_curves.zip (wavelength nm -> Angstrom). "
                "Laboratory filter transmission scans only: no telescope, detector QE or "
                "atmosphere, so these are filter-only curves (DetectorType photon for the CCD "
                "imager; a total response needs the detector generation, which changed over time). "
                f"Band names follow LCO filter codes to avoid case collisions: {mapping}. "
                f"Cleaning (measurement noise below 0 set to 0): {cleaned or 'none'}. "
                "No published pivot wavelengths. The w filter on LCO's filter page is not in this "
                "zip and is not substituted."
                + (" Spectral is historical (replaced at FTN/FTS)." if instrument == "Spectral" else ""))
        build_instrument("LasCumbres", instrument, bands, notes=note, raw_files=[zpath],
                         ground_based=True)
