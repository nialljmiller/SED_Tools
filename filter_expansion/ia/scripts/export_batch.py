"""Collect manifest.jsonl records not yet published into batches/<NAME>/ (manifest.jsonl + raw/ hard links).
Already published: every batches/*.manifest.jsonl and batches/*/manifest.jsonl (by file + sha256).
Prints the number of records exported; exits 3 if there are none."""
import json, os, sys, pathlib, glob
HERE = pathlib.Path(__file__).resolve().parent.parent
name = sys.argv[1]
out = HERE / "batches" / name
pub = set()
for m in glob.glob(str(HERE / "batches/*.manifest.jsonl")) + glob.glob(str(HERE / "batches/*/manifest.jsonl")):
    for l in open(m):
        r = json.loads(l); pub.add((r.get("file"), r.get("sha256"), r.get("retrieved")))
new = []
for l in open(HERE / "manifest.jsonl"):
    r = json.loads(l)
    if r["status"] not in ("ok", "rejected") or (r.get("file"), r.get("sha256"), r.get("retrieved")) in pub: continue
    new.append((l, r))
if not new: print(0); sys.exit(3)
if out.exists(): sys.exit(f"{out} exists")
(out / "raw" / "_rejected").mkdir(parents=True)
with open(out / "manifest.jsonl", "w") as f:
    for l, r in new:
        if r.get("file"): os.link(HERE / r["file"], out / r["file"])
        f.write(l if l.endswith("\n") else l + "\n")
print(len(new))
