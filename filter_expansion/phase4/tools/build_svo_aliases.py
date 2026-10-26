#!/usr/bin/env python3
"""Build the SVO -> NJM alias map (svo_aliases.json) and recount the SVO missing list.

An *alias* is an SVO FPS filter ID (``Facility/Instrument.Band``) whose band NJM
already serves under another path.  The generic rule is SVO ``Fac/Inst.Band`` ->
NJM ``Fac/Inst/Band``; this tool records only the IDs where that rule (or the
curve's own ``FilterID`` header) does not already give the answer, each with its
basis.  Evidence, in order of strength:

  info      info.json bands.<b>.svo.filterID (or bands.<b>.svo_filterID): the
            swarm's own SVO match for that NJM band (matched_by recorded).
  path      generic rule, band name passed through safe_band() (dots -> "p").
  curve     the IA-archived SVO VOTable curve equals an NJM curve in an approved
            instrument directory (peak-normalised overlap >= 0.995, pivot within
            0.2 %).
  name      approved instrument rename + the same band under a documented NJM
            naming scheme; checked against the archived curve when there is one
            (rejected if overlap < 0.90 or pivot off by > 2 %).
  curated   a named rule (DECISIONS-20261026 item 7, C-sources.md) applied to a
            band whose SVO curve was not archived, or whose curve is an older
            SVO version of the NJM band.

IDs that only relate to an NJM band (filter-only component of an NJM total
curve, twin instrument with an identical curve, different detector temperature
or realisation) are listed under "related" and are NOT counted as held or
served.

Inputs (all read-only): a TSV of NJM curve headers (path<TAB>first line), a JSON
of the info.json SVO matches, a local copy of NJM's data/filters/*.dat, and the
IA harvest manifest + raw VOTables.  See --help.  Re-run after new curves are
deployed; the map only grows when evidence does.
"""
import argparse
import collections
import datetime
import html
import json
import pathlib
import re
import sys

import numpy as np

OVERLAP_SAME, PIVOT_SAME = 0.995, 0.002
OVERLAP_NAME, PIVOT_NAME = 0.90, 0.02

# SVO instrument -> NJM instrument dirs, beyond those evidenced by FilterID headers
# and info.json matches.  Basis recorded in the output.
INSTRUMENT_RENAMES = {
    "WFIRST/WFI": (["Roman/WFI"], "DECISIONS-20261026 item 7: WFIRST = Roman/WFI (mission renamed 2020); "
                                  "SVO's WFIRST/WFI.* are the older SVO Roman curves"),
    "PLANCK/LFI": (["Planck/LFI"], "case only: NJM FilterIDs are Planck/LFI.<GHz>"),
    "PLANCK/HFI": (["Planck/HFI"], "case only: NJM FilterIDs are Planck/HFI.<GHz>"),
    "Paranal/SPHERE": (["SPHERE/IRDIS", "SPHERE/ZIMPOL"],
                       "NJM splits SPHERE into its IRDIS and ZIMPOL arms; SVO prefixes the band "
                       "with IRDIS_/ZIMPOL_; archived SVO curves are identical to NJM's"),
    "CAHA/ALHAMBRA": (["CAHA/LAICA", "CAHA/Omega2000"],
                      "ALHAMBRA optical bands are NJM CAHA/LAICA (FilterID headers); its J/H/Ks "
                      "curves are identical to NJM CAHA/Omega2000"),
    "NOT/NOT": (["NOT/ALFOSC"], "SVO NOT/NOT.<filter number> curves identical to NJM NOT/ALFOSC bands"),
    "LasCumbres/LasCumbres": (["LasCumbres/Spectral"],
                              "archived SVO curves identical to NJM LasCumbres/Spectral"),
    "Generic/Johnson_UBVRIJHKL": (["Generic/Johnson"],
                                  "archived SVO curves identical to NJM Generic/Johnson (both SVO-derived)"),
}
# Plausible renames that count only where the archived SVO curve confirms the band
# (name candidates need overlap >= 0.98; otherwise only an identical curve counts).
UNCERTAIN_RENAMES = {
    "COSMOSOMAS/COSMO11": (["Teide/COSMO11"], "same instrument name; COSMOSOMAS ran at Teide Observatory"),
    "COSMOSOMAS/COSMO15": (["Teide/COSMO15"], "same instrument name; COSMOSOMAS ran at Teide Observatory"),
    "MKO/NSFCam": (["IRTF/NSFCam"], "NSFCam is the IRTF camera on Maunakea (NJM also has its MKO-system bands)"),
    "NIRT/NSFCAM": (["IRTF/NSFCam"], "NSFCam is the IRTF camera"),
    "TYCHO/TYCHO": (["Hipparcos/Tycho"], "Tycho photometry from Hipparcos"),
    "Generic/Cousins": (["GCPD/Cousins"], "SVO also lists GCPD/Cousins"),
    "Generic/Stromgren": (["GCPD/Stromgren"], "SVO also lists GCPD/Stromgren"),
    "Geneva/Geneva": (["GCPD/Geneva"], "SVO also lists GCPD/Geneva"),
}
REQUIRE_CURVE = set(UNCERTAIN_RENAMES)
# Renames decided by Nill: same-name bands are aliases even when the curve version differs.
CURATED_INSTRUMENTS = {"WFIRST/WFI", "PLANCK/LFI", "PLANCK/HFI"}
# Evidence pairs that are NOT instrument renames (different spacecraft or twin instruments).
NOT_RENAMES = {
    ("Beijing1/SLIM6", "AlSat1/SLIM6"): "Beijing-1 and AlSat-1 are different DMC satellites with the same "
                                       "SLIM6 imager design; one NJM AlSat1 header carries a Beijing1 FilterID",
}
TWINS = {
    ("Gemini/Zorro", "Gemini/Alopeke"): "Zorro (Gemini South) and 'Alopeke (Gemini North) are twin "
                                       "imagers; SVO lists identical curves for both",
    ("JPSS/VIIRS", "JPSS/VIIRS-J1"): "SVO JPSS/VIIRS is Suomi-NPP VIIRS, a different unit from NOAA-20 (J1)",
    ("JPSS/VIIRS", "JPSS/VIIRS-J2"): "SVO JPSS/VIIRS is Suomi-NPP VIIRS, a different unit from NOAA-21 (J2)",
}


def band_rules(svo_inst, band):
    """Candidate NJM band names for an SVO band under a documented naming scheme: [(name, basis)]."""
    out = []
    if svo_inst == "WFIRST/WFI":
        m = re.fullmatch(r"[A-Z](\d{3})", band)
        if m:
            out.append(("F" + m.group(1), "WFIRST <letter><nnn> = Roman F<nnn> (same element, renamed)"))
    if svo_inst in ("PLANCK/LFI", "PLANCK/HFI"):
        m = re.fullmatch(r"0*(\d+)GHz", band)
        if m:
            out.append((m.group(1), "SVO <nnn>GHz = NJM <GHz> (Planck channel)"))
    if svo_inst == "Herschel/SPIRE":
        for s, n in (("PSW", "250"), ("PMW", "350"), ("PLW", "500")):
            if band == s:
                out.append((n, f"DECISIONS-20261026 item 7 and C-sources.md: SPIRE {s} = {n} um array"))
    if svo_inst in ("Spitzer/MIPS", "IRAS/IRAS"):
        m = re.fullmatch(r"(\d+)mu", band)
        if m:
            out.append((m.group(1), "SVO <n>mu = NJM <n> (info.json matches IRAS/IRAS.12mu to IRAS/12; "
                                    "DECISIONS item 7: MIPS 24mu = 24)"))
    if svo_inst == "Herschel/Pacs":
        for s, n in (("blue", "70"), ("green", "100"), ("red", "160")):
            if band == s:
                out.append((n, "info.json matches Herschel/Pacs.blue/green to PACS/70,100 (provenance); red = 160"))
    if svo_inst == "TYCHO/TYCHO" and band in ("B", "V"):
        out.append((band + "T", "Tycho B/V = BT/VT"))
    if svo_inst in ("GAIA/GAIA2", "GAIA/GAIA2r"):
        for s, n in (("Gbp", "BP"), ("Grp", "RP"), ("G", "G"), ("BP", "BP"), ("RP", "RP")):
            if band == s:
                out.append((n, "info.json matches GAIA/GAIA2(r).Gbp/G to GAIA/DR2(r)/BP,G; Grp = RP "
                               "(DECISIONS item 7: GAIA2r = DR2r)"))
    if svo_inst == "Paranal/NACO":
        m = re.fullmatch(r"(IB|NB)(\d)(\d\d)", band)
        if m:
            out.append((f"{m.group(1)}_{m.group(2)}p{m.group(3)}",
                        "NACO scheme IB<nnn>/NB<nnn> = NJM IB_<n>p<nn>/NB_<n>p<nn> (info.json matches "
                        "IB200..IB248, NB104..NB374 by band name and pivot)"))
        if band in ("H", "J", "Ks"):
            out.append((band + "_CONICA", "NACO scheme <X> = NJM <X>_CONICA (info.json matches H, J)"))
    if svo_inst == "Paranal/SPHERE":
        for pre in ("IRDIS_", "ZIMPOL_"):
            if band.startswith(pre):
                out.append((band[len(pre):], f"SVO SPHERE {pre}<band> = NJM SPHERE/{pre[:-1]}/<band>"))
        m = re.fullmatch(r"IRDIS_D_(ND-)?([A-Z])(\d\d)_(\d)", band)
        if m:
            nd = "ND_" if m.group(1) else ""
            out.append((f"D_{nd}{m.group(2)}{m.group(3)}_{m.group(2)}{m.group(4)}",
                        "SVO SPHERE IRDIS_D_[ND-]<X><ab>_<n> = NJM D_[ND_]<X><ab>_<X><n> (IRDIS dual-band)"))
    return out


def norm(name):
    """Loose band key: case-folded alphanumerics, digit-p-digit read as a decimal point."""
    name = re.sub(r"(?<=\d)p(?=\d)", "", name)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def safe_band(name):
    return re.sub(r"[^A-Za-z0-9_+-]", "_", name.replace(".", "p")).strip("_")


def split(svo_id):
    fac, _, rest = svo_id.partition("/")
    inst, _, band = rest.partition(".")
    return fac, inst, band


# ---- curves ----
TR_RE = re.compile(r"<TR>\s*<TD>([^<]*)</TD>\s*<TD>([^<]*)</TD>", re.S | re.I)
P_RE = re.compile(r"<PARAM\b([^>]*)/?>", re.S)
A_RE = re.compile(r'(\w+)="([^"]*)"')


def read_votable(path):
    t = pathlib.Path(path).read_bytes().decode("utf-8", "replace")
    params = {}
    for m in P_RE.finditer(t):
        a = dict(A_RE.findall(m.group(1)))
        if "name" in a and a["name"] not in params:
            params[a["name"]] = html.unescape(a.get("value", ""))
    rows = []
    for x, y in TR_RE.findall(t):
        try:
            rows.append((float(x), float(y)))
        except ValueError:
            pass
    return params, (np.array(rows) if rows else None)


def read_dat(path):
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            p = re.split(r"[\s,]+", line.strip())
            if len(p) >= 2:
                try:
                    rows.append((float(p[0]), float(p[1])))
                except ValueError:
                    pass
    return np.array(rows) if len(rows) > 1 else None


def prep(a):
    if a is None or len(a) < 2:
        return None
    a = a[np.argsort(a[:, 0])]
    w, t = a[:, 0], np.clip(a[:, 1], 0, None)
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(t)) or t.max() <= 0 or w.min() <= 0:
        return None
    t = t / t.max()
    num, den = np.trapezoid(t * w, w), np.trapezoid(t / w, w)
    return (w, t, float(np.sqrt(num / den))) if num > 0 and den > 0 else None


def compare(a, b):
    """(overlap, |pivot_b/pivot_a - 1|) of two prepared curves."""
    g = np.union1d(a[0], b[0])
    x = np.interp(g, a[0], a[1], left=0, right=0)
    y = np.interp(g, b[0], b[1], left=0, right=0)
    s = np.trapezoid(x + y, g)
    ov = float(1 - np.trapezoid(np.abs(x - y), g) / s) if s > 0 else 0.0
    return ov, abs(b[2] / a[2] - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--headers", required=True, help="TSV: Facility/Instrument/band.dat<TAB>first line")
    ap.add_argument("--info", required=True, help="JSON {Fac/Inst: {bands: {b: {svo, matched_by, curve, svo_filterID}}}}")
    ap.add_argument("--curves", required=True, help="local copy of NJM data/filters")
    ap.add_argument("--ia", required=True, help="IA harvest dir (manifest.jsonl, raw/)")
    ap.add_argument("--missing", required=True, help="svo_missing.txt")
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()
    out = pathlib.Path(args.out)

    paths, fid = {}, {}
    for line in open(args.headers):
        p, _, h = line.rstrip("\n").partition("\t")
        if p.count("/") != 2:
            continue
        paths[p[:-4]] = h
        m = re.search(r"FilterID=([^;,]+)", h)
        if m:
            fid.setdefault(m.group(1), p[:-4])
    info = json.load(open(args.info))
    info_match = {}
    for fi, v in info.items():
        for b, e in v["bands"].items():
            for key in ("svo", "svo_filterID"):
                if e.get(key) and f"{fi}/{b}" in paths:
                    info_match.setdefault(e[key], (f"{fi}/{b}", e.get("matched_by") or key, e.get("curve")))
    held = set(fid) | set(info_match)
    njm_svo = {}                      # NJM band -> SVO ID it already represents
    for k, p in fid.items():
        njm_svo.setdefault(p, k)
    for k, (p, _, _) in info_match.items():
        njm_svo.setdefault(p, k)
    by_dir = collections.defaultdict(list)
    for p in paths:
        by_dir[p.rsplit("/", 1)[0]].append(p)

    # approved SVO instrument -> NJM dirs
    inst_map = collections.defaultdict(dict)
    for src, label in ((fid, "FilterID headers"), ({k: v[0] for k, v in info_match.items()}, "info.json SVO matches")):
        for k, p in src.items():
            f, i, _ = split(k)
            d = p.rsplit("/", 1)[0]
            if f"{f}/{i}" != d and (f"{f}/{i}", d) not in NOT_RENAMES:
                inst_map[f"{f}/{i}"].setdefault(d, f"{label} place {f}/{i}.* bands in {d}")
    for k, (dirs, basis) in {**INSTRUMENT_RENAMES, **UNCERTAIN_RENAMES}.items():
        for d in dirs:
            inst_map[k].setdefault(d, basis)

    vot = {}
    for line in open(pathlib.Path(args.ia) / "manifest.jsonl"):
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("kind") == "fps_votable":
            vot.setdefault(r["id"], (r["file"], r.get("capture_timestamp"), r.get("sha256")))
    ccache = {}

    def ncurve(p):
        if p not in ccache:
            f = pathlib.Path(args.curves) / (p + ".dat")
            ccache[p] = prep(read_dat(f)) if f.is_file() else None
        return ccache[p]

    def scurve(svo_id):
        if svo_id not in vot:
            return None, {}
        try:
            params, arr = read_votable(pathlib.Path(args.ia) / vot[svo_id][0])
        except OSError:
            return None, {}
        return prep(arr), params

    missing = [l.strip() for l in open(args.missing) if l.strip() and not l.startswith("#")]
    aliases, related, now_held, rejected = {}, {}, {}, {}

    # info.json matches whose SVO ID differs from the NJM band's FilterID: needed by the service.
    for k, (p, by, curve) in sorted(info_match.items()):
        if k not in fid and fid.get(k) != p:
            aliases[k] = {"njm": p, "kind": "info", "basis": f"info.json bands.{p.rsplit('/', 1)[1]}.svo "
                          f"(matched by {by}; SVO curve {curve or 'n/a'})"}

    for m in missing:
        f, i, b = split(m)
        if m in held:
            now_held[m] = fid.get(m) or info_match[m][0]
            continue
        if not b:
            continue
        A, params = scurve(m)
        p = f"{f}/{i}/{safe_band(b)}"
        if p in paths and not njm_svo.get(p):
            rec = {"njm": p, "kind": "path", "basis": "generic rule SVO Fac/Inst.Band -> NJM Fac/Inst/Band"
                   + ("" if safe_band(b) == b else " (band through safe_band)")}
            B = ncurve(p)
            if A is not None and B is not None:
                ov, dp = compare(A, B)
                rec.update(overlap=round(ov, 4), pivot_diff_pct=round(100 * dp, 3))
            if rec.get("overlap") is None or (rec["overlap"] >= OVERLAP_NAME and rec["pivot_diff_pct"] <= 100 * PIVOT_NAME):
                aliases[m] = rec
                continue
            rejected[m] = {"njm": p, "reason": f"path match but curve differs (overlap {rec['overlap']}, "
                           f"pivot {rec['pivot_diff_pct']} %)"}
        comp = params.get("components", "")
        dirs = {f"{f}/{i}": "same instrument directory"} if f"{f}/{i}" in by_dir else {}
        dirs.update({d: why for d, why in inst_map.get(f"{f}/{i}", {}).items() if d in by_dir})
        # 1. named / scheme candidates
        cands = []
        for d, why in dirs.items():
            for name, rule in band_rules(f"{f}/{i}", b):
                if f"{d}/{name}" in paths:
                    cands.append((f"{d}/{name}", "curated", f"{rule}; instrument: {why}"))
            for q in by_dir[d]:
                if norm(q.rsplit("/", 1)[1]) == norm(b) and all(q != c[0] for c in cands):
                    cands.append((q, "name", f"band name equal up to case/punctuation; instrument: {why}"))
        best = None
        for q, kind, why in cands:
            B = ncurve(q)
            ov = dp = None
            if f"{f}/{i}" in REQUIRE_CURVE and (A is None or B is None):
                continue
            if A is not None and B is not None:
                ov, dp = compare(A, B)
                if f"{f}/{i}" in REQUIRE_CURVE and ov < 0.98:
                    continue
                if ov < OVERLAP_NAME or dp > PIVOT_NAME:
                    rejected[m] = {"njm": q, "reason": f"name match but curve differs (overlap {ov:.4f}, "
                                   f"pivot {100 * dp:.2f} %)", "components": comp}
                    continue
            other = njm_svo.get(q)
            if kind == "name" and f"{f}/{i}" in CURATED_INSTRUMENTS:
                kind = "curated"
            if "ND" in re.split(r"[_-]", b) and "ND" not in re.split(r"[_-]", q.rsplit("/", 1)[1]):
                continue   # neutral-density variant: same shape, different throughput
            # The NJM band already is another SVO filter: only an identical curve or a curated rule makes it an alias.
            if other and other != m and kind != "curated" and not (ov is not None and ov >= OVERLAP_SAME and dp <= PIVOT_SAME):
                rejected[m] = {"njm": q, "reason": f"NJM band already holds SVO {other}; no identical curve"}
                continue
            rec = {"njm": q, "kind": kind, "basis": why}
            if ov is not None:
                rec.update(overlap=round(ov, 4), pivot_diff_pct=round(100 * dp, 3))
            if other and other != m:
                rec["njm_filterID"] = other
            if best is None or (rec.get("overlap") or 0) > (best.get("overlap") or 0):
                best = rec
        if best:
            if best.get("overlap") is not None and best["overlap"] < OVERLAP_SAME:
                best["note"] = "same band, curve version differs"
            aliases[m] = best
            rejected.pop(m, None)
            continue
        # 2. curve identity within approved dirs
        if A is not None:
            hits = []
            for d, why in dirs.items():
                for q in by_dir[d]:
                    B = ncurve(q)
                    if B is None or abs(B[2] / A[2] - 1) > 0.01:
                        continue
                    if "ND" in re.split(r"[_-]", b) and "ND" not in re.split(r"[_-]", q.rsplit("/", 1)[1]):
                        continue
                    ov, dp = compare(A, B)
                    hits.append((ov, -dp, q, why))
            hits.sort(reverse=True)
            if hits and hits[0][0] >= OVERLAP_SAME and -hits[0][1] <= PIVOT_SAME:
                ov, ndp, q, why = hits[0]
                rec = {"njm": q, "kind": "curve", "overlap": round(ov, 4), "pivot_diff_pct": round(-100 * ndp, 3),
                       "basis": f"archived SVO curve (IA capture {vot[m][1]}) identical to NJM curve; instrument: {why}"}
                if njm_svo.get(q) and njm_svo[q] != m:
                    rec["njm_filterID"] = njm_svo[q]
                ties = [h[2] for h in hits[1:] if h[0] >= OVERLAP_SAME and -h[1] <= PIVOT_SAME]
                if ties:
                    rec["also_identical"] = ties
                aliases[m] = rec
                continue
            # related: filter-only components, twins
            for (si, nd), why in TWINS.items():
                if f"{f}/{i}" == si:
                    for q in by_dir.get(nd, []):
                        B = ncurve(q)
                        if B is not None and abs(B[2] / A[2] - 1) <= 0.01:
                            ov, dp = compare(A, B)
                            if ov >= 0.95 and (m not in related or ov > related[m]["overlap"]):
                                related[m] = {"njm": q, "relation": "twin or different unit, similar curve",
                                              "overlap": round(ov, 4), "basis": why}
            if m not in related and hits and hits[0][0] >= 0.90:
                ov, ndp, q, why = hits[0]
                rel = ("filter-only component of the NJM curve" if b.endswith(("_filter", "_fil"))
                       else "similar curve, not the same band")
                related[m] = {"njm": q, "relation": rel, "overlap": round(ov, 4),
                              "pivot_diff_pct": round(-100 * ndp, 3), "svo_components": comp}
        if m not in aliases and m not in related and b.endswith(("_filter", "_fil")):
            base = b.rsplit("_", 1)[0]
            for d in dirs:
                for q in by_dir[d]:
                    if norm(q.rsplit("/", 1)[1]) == norm(base):
                        related[m] = {"njm": q, "relation": "filter-only component of the NJM curve (by name)"}

    # An NJM curve identical to one SVO ID is not also a lower-overlap SVO variant of it.
    ident = collections.defaultdict(list)
    for k, a in aliases.items():
        if (a.get("overlap") or 0) >= OVERLAP_SAME:
            ident[a["njm"]].append(k)
    for k, a in list(aliases.items()):
        o = a.get("overlap")
        if o is not None and o < OVERLAP_SAME and ident.get(a["njm"]) and a["kind"] != "curated":
            related[k] = {"njm": a["njm"], "relation": f"NJM curve is identical to {', '.join(ident[a['njm']])} "
                          "instead", "overlap": o, "pivot_diff_pct": a.get("pivot_diff_pct")}
            del aliases[k]
    mset = set(missing)
    still = [m for m in missing if m not in now_held and m not in aliases]
    kinds = collections.Counter(a["kind"] for k, a in aliases.items() if k in mset and k not in now_held)
    doc = {
        "schema": 1,
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": "SVO Filter Profile Service IDs (Facility/Instrument.Band) that NJM serves under another "
                       "path. Generic rule: SVO Fac/Inst.Band -> NJM Fac/Inst/Band; IDs equal to an NJM FilterID "
                       "header need no alias. 'njm' is Facility/Instrument/band (file band.dat). 'related' IDs are "
                       "NOT the same band and are not served. Built by filter_expansion/phase4/tools/build_svo_aliases.py.",
        "generic_rule": "SVO Facility/Instrument.Band -> NJM Facility/Instrument/Band",
        "thresholds": {"identical_overlap": OVERLAP_SAME, "identical_pivot": PIVOT_SAME,
                       "name_check_overlap": OVERLAP_NAME, "name_check_pivot": PIVOT_NAME},
        "instrument_aliases": {k: [{"njm": d, "basis": why} for d, why in sorted(v.items())]
                               for k, v in sorted(inst_map.items())},
        "not_instrument_aliases": {f"{a} -> {b}": why for (a, b), why in {**NOT_RENAMES, **TWINS}.items()},
        "counts": {"missing_list": len(missing), "now_held_directly": len(now_held),
                   "aliases_from_missing_list": sum(kinds.values()), "aliases_by_kind": dict(kinds),
                   "aliases_total_including_info_matches": len(aliases),
                   "related_not_counted": len(related), "still_missing": len(still)},
        "aliases": dict(sorted(aliases.items())),
        "related": dict(sorted(related.items())),
        "rejected_candidates": dict(sorted(rejected.items())),
        "now_held_directly": dict(sorted(now_held.items())),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "svo_aliases.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    with open(out / "svo_missing_after_aliases.txt", "w") as fh:
        fh.write(f"# SVO IDs from svo_missing.txt ({len(missing)}) still not on NJM after the alias map "
                 f"({doc['generated']}): {len(still)} IDs.\n# Removed: {len(now_held)} now held directly "
                 f"(FilterID header or info.json match), {sum(kinds.values())} aliases (svo_aliases.json). "
                 f"{len(related)} 'related' IDs stay listed.\n")
        fh.writelines(m + "\n" for m in still)
    print(json.dumps(doc["counts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
