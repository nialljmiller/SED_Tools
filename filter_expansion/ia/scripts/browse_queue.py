"""Build cdx/queue_browse.json: one entry per SVO filter ID with an archived browse page
(index.php?id=<ID>&mode=browse), carrying every capture newest-first so the downloader can
fall back past stubs. Order: IDs NJM serves (FilterID in njm_headers.tsv), then IDs seen under
theory/fps, svo/theory/fps or svo/theory/fps3, then IDs seen only under theory/fps3."""
import json, glob, re, collections, urllib.parse, pathlib
HERE = pathlib.Path(__file__).resolve().parent.parent
njm = set()
for line in open(HERE / "njm_headers.tsv"):
    m = re.search(r"FilterID=([^;\t\n]+)", line)
    if m: njm.add(m.group(1).strip().lower())
caps = {}
for fn in sorted(glob.glob(str(HERE / "cdx/*STAR.json"))):
    for urlkey, ts, orig, mime, st, dig, ln in json.load(open(fn))[1:]:
        p = urllib.parse.urlsplit(orig)
        q = {k.lower(): v for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True)}
        if not p.path.lower().endswith("/index.php") or q.get("mode", "").lower() != "browse": continue
        fid = q.get("id", "").strip()
        if "/" not in fid: continue
        tree = re.sub(r"/+", "/", p.path)[:-len("/index.php")]
        caps[(urlkey, ts)] = dict(ts=ts, original=orig, digest=dig, length=int(ln) if ln.isdigit() else -1,
                                  tree=tree, id=fid)
by = collections.defaultdict(list)
for c in caps.values(): by[c["id"]].append(c)
queue = []
for fid, cs in by.items():
    cs.sort(key=lambda c: c["ts"], reverse=True)
    trees = {c["tree"] for c in cs}
    tier = 0 if fid.lower() in njm else (1 if trees - {"/theory/fps3"} else 2)
    queue.append(dict(kind="fps_browse", id=fid, tier=tier, n_captures=len(cs),
                      captures=[{k: c[k] for k in ("ts", "original", "digest", "length")} for c in cs]))
queue.sort(key=lambda e: (e["tier"], e["id"].lower()))
json.dump(queue, open(HERE / "cdx/queue_browse.json", "w"), indent=0)
print("ids", len(queue), "by tier", dict(collections.Counter(e["tier"] for e in queue)),
      "njm ids", len(njm), "njm ids without browse capture", len(njm - {e["id"].lower() for e in queue}))
