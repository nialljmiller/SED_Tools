"""Gate: staged instruments may not add fx_validate errors vs the current NJM snapshot.
gate.py F/I ...  (prints new errors; exit 1 if any)"""
import io, sys, contextlib, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import fx_validate
def errs(root, keys):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fx_validate.main([str(root), "--mirror-index", str(HERE / "empty_index.json"), "--only", *keys])
    out, cur = {}, None
    for line in buf.getvalue().splitlines():
        m = re.match(r"(ok  |FAIL) (\S+):", line)
        if m: cur = m.group(2); out.setdefault(cur, set()); continue
        if cur and line.startswith("     ") and not line.strip().startswith("warn:"):
            # normalise renamed band names so a rename isn't a "new" error
            out[cur].add(re.sub(r"(\d)\.(\d)", r"\1p\2", line.strip().replace(str(root) + "/filters/", "").replace(str(root.resolve()) + "/", "")))
    return out
keys = sys.argv[1:]
old, new = errs(HERE / "current_root", keys), errs(HERE / "stage", keys)
bad = 0
for k in keys:
    added = new.get(k, set()) - old.get(k, set())
    print(f"{k}: {len(old.get(k, ()))} errors before, {len(new.get(k, ()))} after, {len(added)} new")
    for a in sorted(added): print("   NEW", a)
    bad += bool(added)
sys.exit(1 if bad else 0)
