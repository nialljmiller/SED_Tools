"""Phase-4 metrics from NJM's info.json files. Run on Cirrus: python3 - < metrics.py [--list]"""
import json, glob, os, sys, collections
root = "/media/data3/MESA/SED_Tools/data/filters"
paper = lambda k: k.split(":", 1)[0] in ("bibcode", "doi", "arxiv")
I = B = desc = bcite = lic = 0; unk = []; assumed = collections.Counter(); dispute = 0; rows = []
for f in sorted(glob.glob(root + "/*/*/info.json")):
    d = json.load(open(f)); I += 1
    fi = os.path.relpath(os.path.dirname(f), root)
    bands = d.get("bands", {}); B += len(bands)
    desc += bool(d.get("description"))
    L = d.get("licence") or {}
    lic += bool(L.get("redistribution")) and L.get("redistribution") != "unknown"
    if not L.get("redistribution") or L.get("redistribution") == "unknown": unk.append(fi)
    nc = 0
    for b in bands.values():
        if any(paper(k) for k in b.get("references", [])): bcite += 1; nc += 1
        c = b.get("curve", {})
        if c.get("DetectorTypeSource") == "assumed": assumed[fi] += 1
        if "DetectorType_note" in json.dumps(b): dispute += 1
    rows.append((fi, len(bands), bool(d.get("description")), nc, L.get("redistribution", "")))
print(f"Instruments / bands on NJM | {I} / {B}")
print(f"Instruments with a description | {desc}")
print(f"Bands citing a paper | {bcite} ({100*bcite/B:.1f} %)")
print(f"Instruments with recorded redistribution terms | {lic}")
print(f"Bands with an assumed DetectorType | {sum(assumed.values())} ({', '.join(f'{k} {v}' for k,v in assumed.most_common())})")
print(f"Bands with a DetectorType disagreement note | {dispute}")
# SVO coverage (W1, 2026-10-26): SVO IDs NJM holds directly (curve FilterID header or info.json
# SVO match) or through filter_expansion/svo_aliases.json, against phase4/svo_missing.txt.
repo = os.path.dirname(os.path.dirname(root))
alias_path = os.path.join(repo, "filter_expansion/svo_aliases.json")
miss_path = os.path.join(repo, "filter_expansion/phase4/svo_missing.txt")
if os.path.isfile(alias_path) and os.path.isfile(miss_path):
    held = set()
    for f in glob.glob(root + "/*/*/*.dat"):
        with open(f, errors="replace") as h: line = h.readline()
        for item in line.strip().split(",", 2)[-1].split(";"):
            if item.startswith("FilterID="): held.add(item[9:])
    for f in glob.glob(root + "/*/*/info.json"):
        for b in json.load(open(f)).get("bands", {}).values():
            s = b.get("svo") if isinstance(b.get("svo"), dict) else {}
            for k in (s.get("filterID"), b.get("svo_filterID")):
                if k: held.add(k)
    al = json.load(open(alias_path))
    alias = {k for k, a in al["aliases"].items() if os.path.isfile(os.path.join(root, a["njm"] + ".dat"))}
    miss = [l.strip() for l in open(miss_path) if l.strip() and not l.startswith("#")]
    direct = sum(m in held for m in miss); via = sum(m not in held and m in alias for m in miss)
    rel = sum(m not in held and m not in alias and m in al.get("related", {}) for m in miss)
    print(f"SVO IDs on the missing list (svo_missing.txt) | {len(miss)}")
    print(f"... now held directly (FilterID or info.json match) | {direct}")
    print(f"... aliases of NJM bands (svo_aliases.json) | {via}")
    print(f"... still missing | {len(miss) - direct - via} (incl. {rel} related, not the same band)")
    print(f"SVO IDs NJM serves (direct or alias) | {len(held | alias)}")
if "--list" in sys.argv:
    for r in rows: print("\t".join(map(str, r)))
