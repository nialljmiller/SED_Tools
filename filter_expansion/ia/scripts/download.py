"""Throttled, resumable download of selected Wayback captures (raw id_ mode).
Appends one JSON line per attempt to manifest.jsonl; skips entries already recorded (ok or rejected)."""
import json, re, sys, time, hashlib, urllib.request, urllib.error, pathlib, datetime
HERE = pathlib.Path(__file__).resolve().parent.parent
RAW = HERE / "raw"; MAN = HERE / "manifest.jsonl"
queue = json.load(open(sys.argv[1]))
PACE = float(sys.argv[2]) if len(sys.argv) > 2 else 4.3
done = set()
if MAN.exists():
    for l in open(MAN):
        r = json.loads(l)
        if r["status"] in ("ok", "rejected"): done.add((r["kind"], r["capture_timestamp"], r["original_url"]))
UA = {"User-Agent": "SED_Tools-archive-recovery/1.0 (+https://nillmill.ddns.net/sed_tools)"}
def classify(b, kind):
    s = b.lstrip()[:600].lower()
    if not b.strip(): return "empty"
    if s.startswith(b"<!doctype html") or s.startswith(b"<html") or b"<html" in s[:200]: return "html"
    if kind == "fps_votable":
        if b"<votable" not in b[:3000].lower(): return "not_votable"
        if b"<td>" not in b.lower() and b"<stream" not in b.lower(): return "votable_no_data"
        return None
    rows = [l for l in b.decode("latin-1").splitlines() if re.match(r"^\s*[-+0-9.eE]+\s+[-+0-9.eE]+", l)]
    return None if len(rows) >= 2 else "ascii_lt2_rows"
last = 0.0
for i, c in enumerate(queue):
    key = (c["kind"], c["ts"], c["original"])
    if key in done: continue
    wb = f"https://web.archive.org/web/{c['ts']}id_/{c['original']}"
    fn = re.sub(r"[^A-Za-z0-9_+.-]", "_", c["id"]) + f"__{c['kind']}__{c['ts']}" + (".xml" if c["kind"] == "fps_votable" else ".txt")
    delay, rec = 330, None
    for attempt in range(7):
        wait = PACE - (time.time() - last)
        if wait > 0: time.sleep(wait)
        last = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(wb, headers=UA), timeout=90) as r:
                b = r.read(); code = r.status
            break
        except urllib.error.HTTPError as e:
            code = e.code
            if e.code in (404, 403, 410): b = None; break
            print("retry", e.code, wb, flush=True)
        except Exception as e:
            code = None; print("retry", repr(e)[:100], flush=True)
        time.sleep(delay); delay = min(delay * 2, 1800)
    else:
        b = None
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rec = dict(id=c["id"], kind=c["kind"], original_url=c["original"], wayback_url=wb,
               capture_timestamp=c["ts"], cdx_digest=c["digest"], retrieved=now, http_status=code)
    if b is None:
        rec["status"] = "rejected" if code in (404, 403, 410) else "error"; rec["reason"] = f"http {code}"
    else:
        why = classify(b, c["kind"])
        rec.update(size=len(b), sha256=hashlib.sha256(b).hexdigest())
        if why:
            rec["status"] = "rejected"; rec["reason"] = why
            (RAW / "_rejected").mkdir(exist_ok=True); (RAW / "_rejected" / fn).write_bytes(b); rec["file"] = "raw/_rejected/" + fn
        else:
            rec["status"] = "ok"; (RAW / fn).write_bytes(b); rec["file"] = "raw/" + fn
    with open(MAN, "a") as f: f.write(json.dumps(rec) + "\n")
    if i % 100 == 0: print(i, len(queue), now, flush=True)
print("finished", flush=True)
