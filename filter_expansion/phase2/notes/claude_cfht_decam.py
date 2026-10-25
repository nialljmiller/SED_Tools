"""BRIEF-2 step 2: stage fixed CFHT and DECam instrument dirs (Claude worker).

Run: .venv/bin/python -E -P phase2/fix/scripts/claude_cfht_decam.py
Reads phase2/fix/current/<F>/<I>/ and downloads/{CFHT,DECam}/; writes
phase2/fix/stage/filters/<F>/<I>/, phase2/fix/changes/<F>__<I>.json and
phase2/fix/stage/provenance/<F>__<I>.json.  Refuses to overwrite a staged dir.
"""
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

TOP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TOP / "tools"))
import fxlib  # noqa: E402
from sed_tools.filter_components import InstrumentManifest  # noqa: E402
from sed_tools.filter_metadata import (read_filter_header, released_mesa_reads,  # noqa: E402
                                       write_filter_file, write_filter_header)

import os
FIX = TOP / "phase2" / "fix"
CUR = FIX / "current"
OUT = Path(os.environ.get("FIX_OUT", FIX))  # dry run: FIX_OUT=/tmp/somewhere
STAGE = OUT / "stage" / "filters"
EV = "source_evidence/filters/2026-10-25-fixes"
CFHT_CURVES = "https://www.cfht.hawaii.edu/Instruments/Filters/curves/"
CFHT_DOC = "https://www.cfht.hawaii.edu/Instruments/Filters/complete.html"
NOIR = "https://noirlab.edu/science/sites/default/files/media/archives/documents/"
DECAM_PAGE = "https://noirlab.edu/science/programs/ctio/filters/Dark-Energy-Camera"
DECAM_INST = "https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera"
BROAD_URL, MN_URL = NOIR + "scidoc0472.txt", NOIR + "scidoc3138.xls"
WIRCAM_MAP = {"H2": "cfh8304", "W": "cfh8105", "LowOH1": "cfh8104", "LowOH2": "cfh8102"}

flags = {}
with open(FIX / "audit_flags.csv") as fh:
    for row in csv.DictReader(fh):
        flags[row["path"]] = row


def read_cur(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return a[:, 0], a[:, 1]


def pivot(wl, tr, dt):
    o = np.argsort(wl, kind="stable")
    return fxlib.pivot_wavelength(wl[o], tr[o], dt)


def read_cfht(path):
    rows = []
    for ln in Path(path).read_text().splitlines():
        t = [x for x in re.split(r"[,\s]+", ln.strip()) if x]
        if len(t) >= 2:
            rows.append((float(t[0]), float(t[1])))
    a = np.array(rows)
    if not np.all(np.diff(a[:, 0]) < 0):
        raise ValueError(f"{path}: not a single strictly descending scan")
    wl = np.round(fxlib.to_angstrom(a[:, 0], "nm"), 6)
    tr = np.array([float(f"{x / 100:.10g}") for x in a[:, 1]])
    return wl, tr


class Inst:
    def __init__(self, fi):
        self.fi, self.changes, self.prov = fi, [], {"instrument": fi, "curves": {}, "raw_files": {}}
        self.src, self.dst = CUR / fi, STAGE / fi
        if self.dst.exists():
            raise SystemExit(f"{self.dst} already staged; not overwriting")
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.src, self.dst)

    def raw(self, path, url):
        self.prov["raw_files"][str(Path(path).relative_to(TOP))] = {"url": url, "sha256": fxlib.sha256(path)}

    def record(self, name, op, *, url=None, pivots=None):
        cur, new = self.src / name, self.dst / name
        e = {"path": f"data/filters/{self.fi}/{name}", "raw_copy": f"{EV}/{self.fi}/{name}",
             "operation": op, "raw_sha256": fxlib.sha256(cur), "curated_sha256": fxlib.sha256(new)}
        if url:
            e["source_url"] = url
        if pivots:
            e["pivot_current_A"], e["pivot_new_A"] = round(pivots[0], 3), round(pivots[1], 3)
            e["pivot_shift_pct"] = round(100 * (pivots[1] - pivots[0]) / pivots[0], 4)
        self.changes.append(e)
        self.prov["curves"][name] = {k: v for k, v in e.items() if k not in ("path", "raw_copy")}

    def finish(self, notes):
        man = InstrumentManifest.from_published(self.dst)
        man.notes = notes
        man.save(self.dst)
        self.changes.append({"path": f"data/filters/{self.fi}/manifest.json", "raw_copy": None,
                             "operation": "created manifest.json (none existed) with "
                                          "InstrumentManifest.from_published of the staged curves",
                             "raw_sha256": None, "curated_sha256": fxlib.sha256(self.dst / "manifest.json")})
        bad = [p for p in self.dst.glob("*.dat") if not released_mesa_reads(p)]
        if bad:
            raise RuntimeError(bad)
        key = self.fi.replace("/", "__")
        (OUT / "changes").mkdir(parents=True, exist_ok=True)
        (OUT / "changes" / f"{key}.json").write_text(json.dumps(self.changes, indent=1) + "\n")
        self.prov["notes"] = notes
        pdir = OUT / "stage" / "provenance"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / f"{key}.json").write_text(json.dumps(self.prov, indent=1) + "\n")
        print(self.fi, len(self.changes) - 1, "curves changed")


def write_curve(path, wl, tr, hdr):
    fields = dict(hdr)
    dt, src = fields.pop("DetectorType"), fields.pop("DetectorTypeSource")
    fid, unit = fields.pop("FilterID", None), fields.pop("WavelengthUnit", None)
    write_filter_file(path, wl, tr, detector_type=dt, source=src, filter_id=fid,
                      wavelength_unit=unit, extra=fields)


def clip_only(inst, name, extra_hdr=None, prefix=""):
    """Minimal curation: fxlib.clean_curve on the current samples."""
    wl0, tr0 = read_cur(inst.src / name)
    wl, tr, rep = fxlib.clean_curve(wl0, tr0)
    hdr = {**read_filter_header(inst.src / name), **(extra_hdr or {})}
    write_curve(inst.dst / name, wl, tr, hdr)
    dt = hdr["DetectorType"]
    op = prefix + "fxlib.clean_curve minimal curation: " + (rep.note() or "no sample change")
    inst.record(name, op, pivots=(pivot(wl0, tr0, dt), pivot(wl, tr, dt)))


def resource_cfht(inst, name, cfh):
    url = CFHT_CURVES + cfh + ".dat"
    raw = fxlib.fetch(url, "CFHT")
    inst.raw(raw, url)
    wl_r, tr_r = read_cfht(raw)
    wl, tr, rep = fxlib.clean_curve(wl_r, tr_r)
    hdr = read_filter_header(inst.src / name)
    hdr.update({"ComponentsSource": "original", "Sources": url})
    write_curve(inst.dst / name, wl, tr, hdr)
    wl0, tr0 = read_cur(inst.src / name)
    dt = hdr["DetectorType"]
    op = (f"re-sourced from {url} (CFHT filter {cfh[3:]}, listed in {CFHT_DOC}; raw sha256 {fxlib.sha256(raw)}): "
          f"{len(wl_r)} rows nm->Angstrom (x10), percent->fraction (/100), strictly descending scan reversed "
          f"to increasing; {rep.note() or 'no negatives'}. Replaces the SVO copy ({len(wl0)} rows, "
          f"wavelengths rounded to whole Angstrom). Header: ComponentsSource svo->original, Sources added; "
          f"FilterID/DetectorType/DetectorTypeSource=svo/Components kept")
    inst.record(name, op, url=url, pivots=(pivot(wl0, tr0, dt), pivot(wl, tr, dt)))


def main():
    # ---------------- CFHT ----------------
    i = Inst("CFHT/CFHT")
    for path in sorted(p for p in flags if p.startswith("CFHT/CFHT/")):
        name = path.rsplit("/", 1)[1]
        if name.startswith("cfh"):
            resource_cfht(i, name, name[:-4])
        else:  # I.dat, R.dat: no CFHT filter number, no original located
            clip_only(i, name)
    i.finish("BRIEF-2 step 2 fix: flagged cfhNNNN curves re-sourced from CFHT's own filter curve files; "
             "I and R (no CFHT filter number) clipped with fxlib.clean_curve.")

    i = Inst("CFHT/Megaprime")
    for path in sorted(p for p in flags if p.startswith("CFHT/Megaprime/")):
        clip_only(i, path.rsplit("/", 1)[1])
    i.finish("BRIEF-2 step 2 fix: one negative sample per flagged curve clipped to 0 with fxlib.clean_curve "
             "(SVO-built curves; totals cannot be rebuilt from CFHT filter files alone).")

    i = Inst("CFHT/Wircam")
    for path in sorted(p for p in flags if p.startswith("CFHT/Wircam/")):
        name = path.rsplit("/", 1)[1]
        resource_cfht(i, name, WIRCAM_MAP[name[:-4]])
    i.finish("BRIEF-2 step 2 fix: flagged curves re-sourced from CFHT's own WIRCam filter curve files.")

    # ---------------- DECam ----------------
    broad = fxlib.fetch(BROAD_URL, "DECam")
    mn = fxlib.fetch(MN_URL, "DECam")
    tab = np.loadtxt(broad, comments="#")
    cols = dict(zip(["u", "g", "r", "i", "z", "Y", "VR"], range(1, 8)))
    import xlrd
    book = xlrd.open_workbook(str(mn))

    def original(band):
        if band in cols:
            return tab[:, 0] * 10, tab[:, cols[band]] / 100, BROAD_URL, broad
        s = book.sheet_by_name(band)
        a = np.array([(s.cell_value(r, 0), s.cell_value(r, 1)) for r in range(1, s.nrows)], float)
        a = a[np.argsort(a[:, 0])]
        return a[:, 0] * 10, a[:, 1] / 100, MN_URL, mn

    for fi in ("DECam/Broad", "DECam/Medium", "DECam/Narrow", "DECam/all"):
        i = Inst(fi)
        for p in sorted(i.src.glob("*.dat")):
            name, band = p.name, p.stem
            ow, ot, url, raw = original(band)
            i.raw(raw, url)
            cw, ct = read_cur(p)
            if not (len(ow) == len(cw) and np.allclose(ow, cw) and np.allclose(ot, ct, rtol=1e-9, atol=0)):
                raise RuntimeError(f"{fi}/{name} does not match {url}")
            hdr = {"FilterID": f"{fi}.{band}", "DetectorType": "photon", "DetectorTypeSource": "original",
                   "WavelengthUnit": "Angstrom", "Components": "filter", "ComponentsSource": "original",
                   "Sources": url}
            tag = (f"verified sample-for-sample identical to the original {url} (raw sha256 {fxlib.sha256(raw)}; "
                   f"nm x10, percent /100). DetectorType=photon tagged from NOIRLab DECam documentation "
                   f"({DECAM_PAGE}; {DECAM_INST}: DECam is a CCD camera, photon-counting), DetectorTypeSource=original; "
                   f"header added (none before): FilterID={fi}.{band};WavelengthUnit=Angstrom;Components=filter;"
                   f"ComponentsSource=original;Sources={url}. ")
            if f"{fi}/{name}" in flags:
                clip_only(i, name, extra_hdr=hdr, prefix=tag)
            else:
                write_filter_header(i.dst / name, hdr)
                i.record(name, tag + "Samples unchanged byte for byte (write_filter_header).")
        i.finish("BRIEF-2 step 2 fix: DetectorType tagged (photon, CCD) from NOIRLab DECam documentation; "
                 "curves verified against NOIRLab scidoc0472 (broad) / scidoc3138 (medium, narrow); "
                 "negative noise samples clipped to 0 with fxlib.clean_curve.")


if __name__ == "__main__":
    main()
