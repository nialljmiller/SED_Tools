import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/media/data3/MESA/SED_Tools")
from sed_tools.filter_metadata import iter_filter_files, read_filter_header
root = Path("/media/data3/MESA/SED_Tools/data/filters")
unt = Counter(); dotted = []; ex = {}
for p in iter_filter_files(root):
    rel = p.relative_to(root); inst = "/".join(rel.parts[:2])
    if "components" in rel.parts: continue
    if not read_filter_header(p).get("DetectorType"): unt[inst] += 1; ex.setdefault(inst, []).append(rel.name)
    if p.name.count(".") > 1: dotted.append(str(rel))
print("untagged", sum(unt.values()), {k: (v, ex[k][:3]) for k, v in sorted(unt.items())})
print("dotted", len(dotted), sorted({d.rsplit('/',1)[0] for d in dotted}))
