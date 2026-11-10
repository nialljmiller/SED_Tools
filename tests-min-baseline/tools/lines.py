import collections, json, sys
from pathlib import Path
import coverage
root, data, out = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
cov = coverage.CoverageData(data); cov.read()
per = collections.defaultdict(set)
allx = set()
for f in cov.measured_files():
    p = Path(f).resolve()
    try:
        rel = str(p.relative_to(root))
    except ValueError:
        rel = str(p)
    if not rel.startswith("sed_tools/"):
        print("skip", rel, file=sys.stderr); continue
    for ln, ctxs in cov.contexts_by_lineno(f).items():
        k = f"{rel}:{ln}"
        allx.add(k)
        for c in ctxs:
            per[c].add(k)
json.dump({"all": sorted(allx), "per": {c: sorted(v) for c, v in per.items()}}, open(out, "w"))
print(len(allx), "lines;", len(per), "contexts", file=sys.stderr)
