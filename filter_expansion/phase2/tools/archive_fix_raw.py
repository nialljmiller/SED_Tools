import glob, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; DL = ROOT / "downloads"
DEST = "source_evidence/filters/2026-10-25-fixes/raw_downloads"
text = "\n".join(open(f).read() for f in glob.glob(str(ROOT / "phase2/fix/changes/*.json")) + glob.glob(str(ROOT / "phase2/fix/stage/provenance/*.json")) + glob.glob(str(ROOT / "phase2/fix/scripts/*.py")) + glob.glob(str(ROOT / "phase2/fix/notes/*")))
urls = set(re.findall(r"https?://[^\s\"'|,;)<>\\]+", text))
ref = (ROOT / "phase2/archive/manifest.json").stat().st_mtime
out = []
for p in sorted(x for x in DL.rglob("*") if x.is_file() and x.stat().st_mtime > ref):
    cand = sorted(u for u in urls if u.rstrip("/").rsplit("/", 1)[-1].split("?")[0].split("#")[0] in (p.name, p.name.split("_", 1)[-1]))
    e = {"path": f"{DEST}/{p.relative_to(DL)}", "source_url": cand[0] if len(cand) == 1 else (cand or None)}
    e["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest(); e["bytes"] = p.stat().st_size
    e["retrieved_utc"] = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.append(e)
json.dump(out, open(ROOT / "phase2/fix/raw_downloads_manifest.json", "w"), indent=2)
print(len(out), "files; unresolved:", [(e["path"].rsplit("/",2)[-2:], e["source_url"]) for e in out if not isinstance(e["source_url"], str)])
