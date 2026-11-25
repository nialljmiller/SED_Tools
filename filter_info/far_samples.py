"""Curation of isolated far samples in two classes of NJM filter curves (Nill, 24 Nov 2026; evidence and the class
analysis in the SED_Tools report far-samples.md and hst-tails.md, copies under source_evidence/far-samples/).

Two classes of curves are curated:

A  ESO La Silla filter tables: the curve's Sources cite an ESO-MIDAS XFilter member (``eso*.mt``) or an
   ls.eso.org WFI, SUSI2 or EMMI-archive filter file. ESO's opticians scanned 300-1200 nm and left out readings below about
   1e-4, so a gap next to an isolated reading is zero, not a line ("Zero in those gaps"). Each such gap is
   filled with T = 0 samples whose spacing starts from the neighbouring real step at each end and grows by at most
   a factor 2 per step, symmetric from both ends ("Space the zeros out", Nill, 25 Nov 2026), so no integration
   rule that works on the curve's own samples (MESA's Simpson zero points) meets a short step beside a long one. A reading at 1100 or 1200 nm for which ESO's RLEAK
   keyword declares a leak there stays as it is (the gaps next to it are not zeroed).
C  HST curves that SVO computed with stsynphot, which holds each component table at its last value beyond its end
   and at its first value below its start. Where STScI's tables confirm such a hold (the curve divided by the other
   components equals the held value to within 1e-4; source_evidence/far-samples/stsci/hst_component_ends.json,
   from hst-tails.md) and the curve still transmits beyond the measured range, it is cut at the end of its shortest
   component table and at the start of its latest-starting one ("Cut at measured end", 24 Nov 2026; both ends,
   Nill's Q2), with one interpolated sample at each cut.

An isolated sample (class A) is one whose gaps to both neighbours exceed lambda/10 and 20 times the curve's median
step; class A has no weight threshold (Nill's Q1 of 24 Nov 2026).

Every path whose data rows equal a curve of a class is that class too (exact copies). The header row is kept
(class C adds the component table that ends the curve to Sources); the change, its reason and its evidence go to
filter_expansion/provenance/<Facility>__<Instrument>.json, bands.<band>.cleaning, which info.json shows.

  python3 -m filter_info.far_samples DATA_DIR --dry-run         # list what would change
  ... --originals DIR [DIR ...]   take each class A curve as it was published from the first DIR holding a copy
                                  (e.g. the backups of an earlier run of this rule) and apply the rule to it
  python3 -m filter_info.far_samples DATA_DIR --apply BACKUP    # back up every changed file there, then write
"""
import argparse
import hashlib
import json
import math
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
ESO_MEMBER = re.compile(r"filters/(eso\d+\.mt)")
ESO_WEB = ("ls.eso.org/sci/facilities/lasilla/instruments/wfi/", "ls.eso.org/sci/facilities/lasilla/instruments/susi/",
           "ls.eso.org/lasilla/sciops/ntt/emmi/archive/filters/")
HELD_MAXDEV = 1e-4  # the hold is confirmed when curve / other components equals the held value to this
TRDS = "https://ssb.stsci.edu/trds/comp/"
NOTE_A = ("T = 0 samples fill the {n} gap(s) next to the isolated ESO reading(s) at {where} A, spaced from the "
          "neighbouring real step and growing by at most 2x per step: ESO's La Silla tables leave out readings below "
          "about 1e-4, so these gaps are zero, not a straight line (Nill, 24 Nov 2026; spacing, 25 Nov 2026; {src}). "
          "ESO's readings are kept.")
GROWTH = 2.0
NOTE_A_LEAK = " The reading(s) at {kept} A stay with their gaps: ESO's RLEAK keyword declares a leak there ({rleak})."
NOTE_C = ("Cut {where}: beyond the measured range of {tables}, SVO's stsynphot curve held the table at its value at that end "
          "(curve / other components = that value to {dev:.0e}), so it was not measured (Nill, 24 Nov 2026). {n} samples "
          "removed from NJM's copy; no longer SVO's curve.")


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


def plan(data_dir, originals=()):
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
        source = next((Path(o) / "data" / "filters" / rel for o in originals
                       if (Path(o) / "data" / "filters" / rel).is_file()), None)
        if source is not None:  # the class A rule is applied to the curve as it was published
            _, o_lines, o_data, o_meta = read_curve(source)
            if ESO_MEMBER.search(o_meta.get("Sources", "")) or any(w in o_meta.get("Sources", "") for w in ESO_WEB):
                curves[rel] = (path, head, o_lines, o_data, meta, lines)
                by_rows.setdefault(hashlib.sha256("\n".join(o_lines).encode()).hexdigest(), []).append(rel)
                continue
        curves[rel] = (path, head, lines, data, meta, lines)
        by_rows.setdefault(hashlib.sha256("\n".join(lines).encode()).hexdigest(), []).append(rel)
    changes = {}
    for rel, (path, head, lines, data, meta, _) in curves.items():
        far, share = isolated(data[:, 0], data[:, 1])
        if not far:
            continue
        if not (ESO_MEMBER.search(meta.get("Sources", "")) or any(w in meta.get("Sources", "") for w in ESO_WEB)):
            continue
        src = meta.get("Sources", "")
        fac, inst, band = rel[:-4].split("/")
        member = ESO_MEMBER.search(src)
        if member or any(w in src for w in ESO_WEB):
            changes[rel] = {"class": "A", "far": far, "share": share, "member": member and member.group(1)}
    for rel, (path, head, lines, data, meta, _) in curves.items():
        fac, inst, band = rel[:-4].split("/")
        record = ends.get((inst, band)) if fac == "HST" and "svo2.cab.inta-csic.es" in meta.get("Sources", "") else None
        cuts = {}
        for side, beyond in (("red", lambda x, e: x > e), ("blue", lambda x, e: x < e)):
            r = (record or {}).get(side) or {}
            if (r.get("ratio_vs_held_maxdev") is not None and r["ratio_vs_held_maxdev"] < HELD_MAXDEV
                    and np.any(beyond(data[:, 0], r["end"]) & (data[:, 1] > 0))):
                cuts[side] = r
        if cuts:
            changes[rel] = {"class": "C", "far": [], "share": 0.0, "cuts": cuts}
    for rel, change in list(changes.items()):  # exact copies of a curve of a class
        digest = hashlib.sha256("\n".join(curves[rel][2]).encode()).hexdigest()
        for other in by_rows[digest]:
            changes.setdefault(other, {**change, "copy_of": rel})
    return curves, changes, leaks


def gap_points(left, right, step_left, step_right, q=GROWTH):
    """Points strictly inside (left, right): step i is min(step_left q^i, step_right q^(n-1-i), C), with the fewest
    steps n that span the gap and the cap C that makes them span it exactly; so neighbouring steps differ by at most a
    factor q, and the first and last steps by at most q from the real steps beside the gap."""
    width = right - left

    def steps(n, cap):
        return [min(step_left * q ** i, step_right * q ** (n - 1 - i), cap) for i in range(n)]

    n = 1
    while sum(steps(n, math.inf)) < width:
        n += 1
    lo, hi = 0.0, max(steps(n, math.inf))
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(steps(n, mid)) < width else (lo, mid)
    out, x = [], left
    for st in steps(n, hi)[:-1]:
        x += st
        out.append(round(x, 6))
    return [x for x in out if left < x < right]


def new_curve(curve, change, leaks):
    path, head, lines, data, meta, _ = curve
    lam = data[:, 0]
    fac, inst, band = str(path)[:-4].split("/")[-3:]
    if change["class"] == "C":
        t, keep_lines, keep_lam = data[:, 1], list(lines), lam
        red, blue = change["cuts"].get("red"), change["cuts"].get("blue")
        where, tables, comps, devs = [], [], [], []
        if red:
            k = keep_lam <= red["end"]
            keep_lines = [l for l, ok in zip(keep_lines, k) if ok] + ([] if red["end"] in keep_lam else
                         [f"{float(red['end'])!r},{float(np.interp(red['end'], lam, t))!r}"])
            keep_lam = np.append(keep_lam[k], [] if red["end"] in keep_lam else [red["end"]])
            where.append(f"at {red['end']:g} A (red end)")
        if blue:
            k = keep_lam >= blue["end"]
            keep_lines = ([] if blue["end"] in keep_lam else
                          [f"{float(blue['end'])!r},{float(np.interp(blue['end'], lam, t))!r}"]) + \
                         [l for l, ok in zip(keep_lines, k) if ok]
            where.append(f"from {blue['end']:g} A (blue start)")
        for r in (red, blue):
            if r:
                comps.append(r["comp"]); tables.append(r["comp"].replace(TRDS, "")); devs.append(r["ratio_vs_held_maxdev"])
        removed = len(lines) - len([l for l in keep_lines if l in set(lines)])
        note = NOTE_C.format(where=" and ".join(where), tables=" and ".join(dict.fromkeys(tables)), dev=max(devs), n=removed)
        return head, keep_lines, note, {"sources": " | ".join(dict.fromkeys(comps)), "add_source": comps}
    step = np.diff(lam)
    median = float(np.median(step))
    big = (step > 0.1 * lam[:-1]) & (step > 20 * median)
    rleak = leaks.get(change.get("member") or "", "")
    kept, zero_at = [], set()
    for i in change["far"]:
        if declared_leak(rleak, lam[i]):
            kept.append(i)
            continue
        for a in (i - 1, i):  # the gaps on each side of the isolated reading
            if 0 <= a < lam.size - 1:
                left = float(step[a - 1]) if a > 0 and not big[a - 1] else median
                right = float(step[a + 1]) if a + 1 < step.size and not big[a + 1] else median
                zero_at |= set(gap_points(float(lam[a]), float(lam[a + 1]), left, right))
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
    ap.add_argument("--originals", type=Path, nargs="+", default=())
    args = ap.parse_args(argv)
    curves, changes, leaks = plan(args.data_dir, args.originals)
    log, provenance = {}, {}
    for rel in sorted(changes):
        change = changes[rel]
        head, rows, note, extra = new_curve(curves[rel], change, leaks)
        old_rows = curves[rel][5]  # the rows served now
        changed = rows != old_rows or head != curves[rel][1]
        log[rel] = {"class": change["class"], "copy_of": change.get("copy_of"), "share": round(change["share"], 4),
                    "rows": [len(old_rows), len(rows)], "changed": changed, "note": note}
        if not changed:
            continue
        fac, inst, band = rel[:-4].split("/")
        provenance.setdefault(f"{fac}__{inst}", {})[band] = {"cleaning": note, "replace": change["class"] == "A",
                                                             **({"sources": extra["sources"]} if extra else {})}
        if args.apply:
            path = curves[rel][0]
            target = args.apply / "data" / "filters" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            path.write_text(head + "\n" + "\n".join(rows) + "\n")
            if extra.get("add_source"):  # the header row through SED_Tools' writer: Sources gains the component tables
                from sed_tools.filters.curve_metadata import read_filter_header, write_filter_header
                fields = read_filter_header(path)
                new = [c for c in extra["add_source"] if c not in fields.get("Sources", "")]
                if new:
                    fields["Sources"] = " | ".join(x for x in (fields.get("Sources"), *new) if x)
                    write_filter_header(path, fields)
    for name, bands in sorted(provenance.items()):
        path = PROVENANCE / f"{name}.json"
        doc = json.loads(path.read_text()) if path.is_file() else {"facility": name.split("__")[0],
                                                                    "instrument": name.split("__")[1], "bands": {}}
        for band, fields in bands.items():
            entry = doc.setdefault("bands", {}).setdefault(band, {})
            replace = fields.pop("replace", False)  # class A's note describes the whole curve: it replaces its own
            for key, value in fields.items():  # an earlier curation's note and sources are kept, the new one added
                old = None if replace and key == "cleaning" else entry.get(key)
                entry[key] = value if not old or value in old else f"{old} | {value}" if key == "sources" else f"{old} {value}"
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
