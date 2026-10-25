"""Record source URL for each file in downloads/ by running recipes with fetch/build stubbed."""
import importlib, json, re, sys, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/recipes")]
import fxlib
rec = {}
def fetch(url, facility, name=None, *, timeout=120):
    name = name or re.sub(r"[^A-Za-z0-9._-]", "_", url.rstrip("/").rsplit("/", 1)[-1].split("?")[0])
    p = fxlib.DOWNLOADS / facility / name
    rec.setdefault(str(p.relative_to(fxlib.DOWNLOADS)), url)
    if not p.exists():
        raise FileNotFoundError(p)
    return p
class Stop(Exception): pass
def build_instrument(*a, **k): return None
fxlib.fetch = fetch; fxlib.build_instrument = build_instrument
for r in sorted((ROOT / "tools/recipes").glob("*.py")):
    try:
        importlib.import_module(r.stem).run()
    except Exception as e:
        print(r.stem, "stopped:", type(e).__name__, str(e)[:120], file=sys.stderr)
json.dump(rec, open(ROOT / "phase2/urlmap.json", "w"), indent=1)
have = {str(p.relative_to(fxlib.DOWNLOADS)) for p in fxlib.DOWNLOADS.rglob("*") if p.is_file()}
print("files", len(have), "mapped", len(have & rec.keys()), "unmapped:", sorted(have - rec.keys()))
print("recorded-but-missing:", sorted(rec.keys() - have))
