"""Split the IA queues so the SVO IDs NJM serves come first (room request ddf8e20d).
NJM IDs: cdx/njm_svo_ids.txt (header FilterIDs plus provenance svo_id, collected on Cirrus).
Writes cdx/chunks/<NN>_<segment>.json; segments: njm_votable, njm_browse, rest_votable, rest_browse.
Entries already recorded ok/rejected in manifest.jsonl are left out."""
import json, pathlib, collections
HERE = pathlib.Path(__file__).resolve().parent.parent
njm = {l.strip().lower() for l in open(HERE / "cdx/njm_svo_ids.txt") if l.strip()}
done = set()
for l in open(HERE / "manifest.jsonl"):
    r = json.loads(l)
    if r["status"] in ("ok", "rejected"): done.add((r["kind"], r["id"].lower()))
vot = [c for c in json.load(open(HERE / "cdx/selected.json")) if c["kind"] == "fps_votable"]
bro = json.load(open(HERE / "cdx/queue_browse.json"))
seg = collections.OrderedDict((k, []) for k in ("njm_votable", "njm_browse", "rest_votable", "rest_browse"))
for c in vot:
    if ("fps_votable", c["id"].lower()) in done: continue
    seg["njm_votable" if c["id"].lower() in njm else "rest_votable"].append(c)
for c in bro:
    if ("fps_browse", c["id"].lower()) in done: continue
    seg["njm_browse" if c["id"].lower() in njm else "rest_browse"].append(c)
size = {"njm_votable": 500, "njm_browse": 500, "rest_votable": 1000, "rest_browse": 1000}
n = 0
for name, q in seg.items():
    q.sort(key=lambda c: (not c["id"].lower().startswith("generic/"), c["id"].lower()))
    for i in range(0, len(q), size[name]):
        n += 1
        json.dump(q[i:i + size[name]], open(HERE / f"cdx/chunks/{n:02d}_{name}.json", "w"), indent=0)
    print(name, len(q))
print("chunks", n)
