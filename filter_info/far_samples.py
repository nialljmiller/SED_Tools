"""Curation of isolated far samples in two classes of NJM filter curves (Nill, 24 Nov 2026; evidence and the class
analysis in the SED_Tools report far-samples.md and hst-tails.md, copies under source_evidence/far-samples/).

An isolated sample is one whose gaps to both neighbours exceed lambda/10 and 20 times the curve's median step. A
curve with one is in scope when it belongs to one of two classes (class C also needs its isolated samples to carry
at least 1 % of its weight, trapezoid of lambda*T on its own samples, the threshold of the approved list; class A
has no threshold since Nill's Q1 of 24 Nov 2026):

A  ESO La Silla filter tables: the curve's Sources cite an ESO-MIDAS XFilter member (``eso*.mt``) or an
   ls.eso.org WFI or SUSI2 filter file. ESO's opticians scanned 300-1200 nm and left out readings below about
   1e-4, so a gap next to an isolated reading is zero, not a line ("Zero in those gaps"). Each such gap gets
   T = 0 samples half a median step inside both of its ends. A reading at 1100 or 1200 nm for which ESO's RLEAK
   keyword declares a leak there stays as it is (the gaps next to it are not zeroed).
C  HST FOC (f/48, f/96) and WFPC1 (PC, WF) curves that SVO computed with stsynphot and that hold STScI's
   30000 A placeholder: beyond the red end of the curve's shortest component table, stsynphot held every
   component at its last value, so the curve ends there ("Cut at measured end"); the end per curve is in
   source_evidence/far-samples/stsci/hst_component_ends.json (STScI TRDS tables of the curve's obsmode).

Every path whose data rows equal a curve of a class is that class too (exact copies). The header row is kept
(class C adds the component table that ends the curve to Sources); the change, its reason and its evidence go to
filter_expansion/provenance/<Facility>__<Instrument>.json, bands.<band>.cleaning, which info.json shows.

  python3 -m filter_info.far_samples DATA_DIR --dry-run         # list what would change
  python3 -m filter_info.far_samples DATA_DIR --apply BACKUP    # back up every changed file there, then write
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import urllib.parse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "source_evidence" / "far-samples"
PROVENANCE = ROOT / "filter_expansion" / "provenance"
SHARE_MIN = 0.01
ESO_MEMBER = re.compile(r"filters/(eso\d+\.mt)")
ESO_WEB = ("ls.eso.org/sci/facilities/lasilla/instruments/wfi/", "ls.eso.org/sci/facilities/lasilla/instruments/susi/")
HST_INSTRUMENTS = {"FOC_F48", "FOC_F96", "WFPC1-PC", "WFPC1-WF"}
TRDS = "https://ssb.stsci.edu/trds/comp/"
NOTE_A = ("T = 0 samples added inside the {n} gap(s) next to the isolated ESO reading(s) at {where} A: ESO's La Silla "
          "tables leave out readings below about 1e-4, so these gaps are zero, not a straight line (Nill, 24 Nov 2026; "
          "{src}). ESO's readings are kept.")
NOTE_A_LEAK = " The reading(s) at {kept} A stay with their gaps: ESO's RLEAK keyword declares a leak there ({rleak})."
NOTE_C = ("Cut at {end:g} A, where the shortest component table of this obsmode ends ({comp}): beyond it SVO's "
          "stsynphot curve held every component at its last value, with STScI's 30000 A placeholder "
          "(hst_ota_007_syn.fits: 'Extrapolated to 30000A for NICMOS'), so it was not measured (Nill, 24 Nov 2026). "
          "{n} samples beyond it removed from NJM's copy; no longer SVO's curve.")


def read_curve(path):
    text = path.read_text()
    head, _, body = text.partition("\n")
    lines = [l for l in body.split("\n") if l.strip()]
    data = np.array([[float(v) for v in l.split(",")[:2]] for l in lines])
    meta = dict(kv.split("=", 1) for kv in head.split(",", 2)[2].split(";") if "=" in kv) if head.count(",") >= 2 else {}
    return head, lines, data, {k: urllib.parse.unquote(v) for k, v in meta.items()}


def isolated(lam, t):
    """Indices of the isolated samples and the share of lambda*T in the steps next to them."""
    step = np.diff(lam)
    if lam.size < 4 or not np.all(step >= 0):
        return [], 0.0
    seg = 0.5 * (t[1:] * lam[1:] + t[:-1] * lam[:-1]) * step
    if seg.sum() <= 0:
        return [], 0.0
    big = (step > 0.1 * lam[:-1]) & (step > 20 * np.median(step))
    n = lam.size
    far = [i for i in range(n) if (i == 0 or big[i - 1]) and (i == n - 1 or big[i])
           and not (i == 0 and not big[0]) and not (i == n - 1 and not big[-1])]
    steps = {j for i in far for j in (i - 1, i) if 0 <= j < n - 1}
    return far, float(sum(seg[j] for j in steps) / seg.sum())


def rleak_table():
    """ESO-MIDAS XFilter member -> RLEAK (% at 1100 / 1200 nm, as written), read from ESO's archive."""
    from astropy.io import fits
    out = {}
    with tarfile.open(EVIDENCE / "eso" / "calib_src.tar.gz") as tar:
        for m in tar.getmembers():
            name = Path(m.name).name
            if re.fullmatch(r"eso\d+\.mt", name):
                header = fits.open(tar.extractfile(m))[1].header
                out[name] = str(header.get("RLEAK", "")).strip()
    return out


def declared_leak(rleak, wavelength):
    """The RLEAK value (%) declared at 1100 or 1200 nm for a reading at that wavelength, or None."""
    values = [v for v in re.split(r"\s*/\s*", rleak.strip().strip("'")) if v.strip()]
    for nm, k in ((11000.0, 0), (12000.0, 1)):
        if abs(wavelength - nm) <= 5 and len(values) > k:
            try:
                if float(values[k].lstrip("<")) > 0:
                    return values[k]
            except ValueError:
                return None
    return None


def plan(data_dir):
    filters = Path(data_dir) / "filters"
    ends = {(r["inst"], r["band"]): r for r in json.loads((EVIDENCE / "stsci" / "hst_component_ends.json").read_text())}
    leaks = rleak_table()
    curves, by_rows = {}, {}
    for path in sorted(filters.glob("*/*/*.dat")):
        try:
            head, lines, data, meta = read_curve(path)
        except (ValueError, IndexError):
            continue
        rel = str(path.relative_to(filters))
        curves[rel] = (path, head, lines, data, meta)
        by_rows.setdefault(hashlib.sha256("\n".join(lines).encode()).hexdigest(), []).append(rel)
    changes = {}
    for rel, (path, head, lines, data, meta) in curves.items():
        far, share = isolated(data[:, 0], data[:, 1])
        if not far:
            continue
        src = meta.get("Sources", "")
        fac, inst, band = rel[:-4].split("/")
        member = ESO_MEMBER.search(src)
        if member or any(w in src for w in ESO_WEB):
            changes[rel] = {"class": "A", "far": far, "share": share, "member": member and member.group(1)}
        elif share >= SHARE_MIN and fac == "HST" and inst in HST_INSTRUMENTS and "svo2.cab.inta-csic.es" in src and data[-1, 0] >= 30000:
            changes[rel] = {"class": "C", "far": far, "share": share, "end": ends[(inst, band)]["red"]}
    for rel, change in list(changes.items()):  # exact copies of a curve of a class
        digest = hashlib.sha256("\n".join(curves[rel][2]).encode()).hexdigest()
        for other in by_rows[digest]:
            changes.setdefault(other, {**change, "copy_of": rel})
    return curves, changes, leaks


def new_curve(curve, change, leaks):
    path, head, lines, data, meta = curve
    lam = data[:, 0]
    fac, inst, band = str(path)[:-4].split("/")[-3:]
    if change["class"] == "C":
        end, comp = change["end"]["end"], change["end"]["comp"]
        keep = [l for l, x in zip(lines, lam) if x <= end]
        if lam[len(keep) - 1] < end:
            keep.append(f"{float(end)!r},{float(np.interp(end, lam, data[:, 1]))!r}")
        note = NOTE_C.format(end=end, comp=comp.replace(TRDS, ""), n=len(lines) - len(keep) + (lam[len(keep) - 1] < end))
        return head, keep, note, {"sources": comp, "add_source": comp}
    h = float(np.median(np.diff(lam))) / 2
    rleak = leaks.get(change.get("member") or "", "")
    kept, zero_at = [], set()
    for i in change["far"]:
        if declared_leak(rleak, lam[i]):
            kept.append(i)
            continue
        if i > 0:
            zero_at |= {float(lam[i - 1] + h), float(lam[i] - h)}
        if i < lam.size - 1:
            zero_at |= {float(lam[i] + h), float(lam[i + 1] - h)}
    out = list(lines)
    for x in sorted(zero_at, reverse=True):
        k = int(np.searchsorted(lam, x))
        out.insert(k, f"{x!r},0.0")
    zeroed = [i for i in change["far"] if i not in kept]
    src = (f"ESO-MIDAS XFilter {change['member']}, in source_evidence/far-samples/eso/calib_src.tar.gz"
           if change.get("member") else "the La Silla filter file this curve copies")
    note = NOTE_A.format(n=2 * len(zeroed) - sum(i in (0, lam.size - 1) for i in zeroed),
                         where=", ".join(f"{lam[i]:g}" for i in zeroed), src=src) if zeroed else ""
    if kept:
        note += NOTE_A_LEAK.format(kept=", ".join(f"{lam[i]:g}" for i in kept), rleak=f"RLEAK '{rleak}'")
    return head, (out if zeroed else list(lines)), note.strip(), {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir", type=Path)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", type=Path, metavar="BACKUP_DIR")
    ap.add_argument("--log", type=Path)
    args = ap.parse_args(argv)
    curves, changes, leaks = plan(args.data_dir)
    log, provenance = {}, {}
    for rel in sorted(changes):
        change = changes[rel]
        head, rows, note, extra = new_curve(curves[rel], change, leaks)
        old_rows = curves[rel][2]
        changed = rows != old_rows or head != curves[rel][1]
        log[rel] = {"class": change["class"], "copy_of": change.get("copy_of"), "share": round(change["share"], 4),
                    "rows": [len(old_rows), len(rows)], "changed": changed, "note": note}
        if not changed:
            continue
        fac, inst, band = rel[:-4].split("/")
        provenance.setdefault(f"{fac}__{inst}", {})[band] = {"cleaning": note,
                                                             **({"sources": extra["sources"]} if extra else {})}
        if args.apply:
            path = curves[rel][0]
            target = args.apply / "data" / "filters" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            path.write_text(head + "\n" + "\n".join(rows) + "\n")
            if extra.get("add_source"):  # the header row through SED_Tools' writer: Sources gains the component table
                from sed_tools.filters.curve_metadata import read_filter_header, write_filter_header
                fields = read_filter_header(path)
                if extra["add_source"] not in fields.get("Sources", ""):
                    fields["Sources"] = " | ".join(x for x in (fields.get("Sources"), extra["add_source"]) if x)
                    write_filter_header(path, fields)
    for name, bands in sorted(provenance.items()):
        path = PROVENANCE / f"{name}.json"
        doc = json.loads(path.read_text()) if path.is_file() else {"facility": name.split("__")[0],
                                                                    "instrument": name.split("__")[1], "bands": {}}
        for band, fields in bands.items():
            doc.setdefault("bands", {}).setdefault(band, {}).update(fields)
        if args.apply:
            if path.is_file():
                target = args.apply / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    if args.log:
        args.log.write_text(json.dumps(log, indent=1) + "\n")
    n = sum(v["changed"] for v in log.values())
    for cls in ("A", "C"):
        rows = [v for v in log.values() if v["class"] == cls]
        print(f"class {cls}: {len(rows)} paths in scope ({sum(not v['copy_of'] for v in rows)} found by the rule, "
              f"{sum(bool(v['copy_of']) for v in rows)} exact copies), {sum(v['changed'] for v in rows)} changed")
    print(f"{'changed' if args.apply else 'would change'} {n} curves; provenance files: {len(provenance)}")


if __name__ == "__main__":
    sys.exit(main())
