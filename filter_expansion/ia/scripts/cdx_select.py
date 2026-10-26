"""Select curve-bearing captures (getdata.php ascii, fps.php?ID=) and pick the newest per (kind, ID)."""
import json, glob, re, collections, urllib.parse, pathlib
HERE = pathlib.Path(__file__).resolve().parent.parent
caps = {}
for fn in sorted(glob.glob(str(HERE / "cdx/*.json"))):
    for urlkey, ts, orig, mime, st, dig, ln in json.load(open(fn))[1:]:
        p = urllib.parse.urlsplit(orig)
        path = p.path.lower()
        q = {k.lower(): v for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True)}
        fid = q.get("id", "").strip()
        if not fid or "/" not in fid: continue
        if path.endswith("/getdata.php"):
            fmt = q.get("format", "").lower()
            kind = "getdata_" + (fmt or "none")
        elif path.endswith("/fps.php") and set(q) == {"id"}:
            kind = "fps_votable"
        else:
            continue
        caps[(urlkey, ts)] = dict(kind=kind, id=fid, ts=ts, original=orig, mime=mime, digest=dig, length=ln)
by = collections.defaultdict(list)
for c in caps.values(): by[(c["kind"], c["id"].lower())].append(c)
sel = []
for (kind, lid), cs in by.items():
    cs.sort(key=lambda c: c["ts"])
    newest = cs[-1]
    sel.append(dict(newest, n_captures=len(cs), all_ts=[c["ts"] for c in cs]))
print("captures", len(caps), collections.Counter(c["kind"] for c in caps.values()))
print("selected", len(sel), collections.Counter(s["kind"] for s in sel))
ids = {s["id"].lower() for s in sel}
print("unique ids", len(ids))
json.dump(sorted(sel, key=lambda s: (s["id"].lower(), s["kind"])), open(HERE / "cdx/selected.json", "w"), indent=0)
