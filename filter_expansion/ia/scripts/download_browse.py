"""Throttled, resumable download of SVO per-filter browse pages (kind fps_browse) from Wayback.
For each ID, tries captures newest-first and keeps the first that actually contains the requested ID
(skipping "Redirecting... to /" stubs, database-error pages and other HTML without it). Requests gzip
and stores the decoded page. Appends one JSON line per attempt to manifest.jsonl, same format as
download.py plus content_encoding/size_encoded; skips IDs already ok and captures already tried."""
import json, re, sys, time, gzip, html, hashlib, urllib.request, urllib.error, pathlib, datetime
HERE = pathlib.Path(__file__).resolve().parent.parent
RAW = HERE / "raw"; MAN = HERE / "manifest.jsonl"
queue = json.load(open(sys.argv[1]))
PACE = float(sys.argv[2]) if len(sys.argv) > 2 else 4.3
MAX_TRIES = 6        # captures fetched per ID before giving up
MIN_LEN = 1000       # CDX (compressed) length below this is a stub or error page; not fetched
ok_ids, tried = set(), set()
if MAN.exists():
    for l in open(MAN):
        r = json.loads(l)
        if r["kind"] != "fps_browse": continue
        if r["status"] == "ok": ok_ids.add(r["id"])
        if r["status"] in ("ok", "rejected"): tried.add((r["capture_timestamp"], r["original_url"]))
UA = {"User-Agent": "SED_Tools-archive-recovery/1.0 (+https://nillmill.ddns.net/sed_tools)",
      "Accept-Encoding": "gzip"}
def classify(b, fid):
    if not b.strip(): return "empty"
    t = b.decode("utf-8", "replace")
    if re.search(r"Redirecting\.*\s*to", t[:5000], re.I): return "redirect_stub"
    if "Could not connect to database" in t: return "db_error"
    if fid not in t and fid not in html.unescape(t): return "id_absent"
    return None
last = 0.0
for i, e in enumerate(queue):
    fid = e["id"]
    if fid in ok_ids: continue
    caps = [c for c in e["captures"] if c["length"] < 0 or c["length"] >= MIN_LEN]
    seen_digest = set(); n = 0
    for c in caps:
        if n >= MAX_TRIES: break
        if (c["ts"], c["original"]) in tried or c["digest"] in seen_digest: continue
        seen_digest.add(c["digest"]); n += 1
        wb = f"https://web.archive.org/web/{c['ts']}id_/{c['original']}"
        fn = re.sub(r"[^A-Za-z0-9_+.-]", "_", fid) + f"__fps_browse__{c['ts']}.html"
        delay, b, enc, code = 330, None, None, None
        for attempt in range(7):
            wait = PACE - (time.time() - last)
            if wait > 0: time.sleep(wait)
            last = time.time()
            try:
                with urllib.request.urlopen(urllib.request.Request(wb, headers=UA), timeout=90) as r:
                    b = r.read(); code = r.status; enc = r.headers.get("Content-Encoding")
                break
            except urllib.error.HTTPError as ex:
                code = ex.code
                if ex.code in (404, 403, 410): break
                print("retry", ex.code, wb, flush=True)
            except Exception as ex:
                code = None; print("retry", repr(ex)[:100], flush=True)
            time.sleep(delay); delay = min(delay * 2, 1800)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        rec = dict(id=fid, kind="fps_browse", original_url=c["original"], wayback_url=wb,
                   capture_timestamp=c["ts"], cdx_digest=c["digest"], retrieved=now, http_status=code)
        if b is None:
            rec["status"] = "rejected" if code in (404, 403, 410) else "error"; rec["reason"] = f"http {code}"
        else:
            size_enc = len(b)
            if enc == "gzip" or b[:2] == b"\x1f\x8b":
                try: b = gzip.decompress(b); enc = "gzip"
                except Exception: rec["reason"] = "gzip_decode_failed"
            why = rec.get("reason") or classify(b, fid)
            rec.update(size=len(b), sha256=hashlib.sha256(b).hexdigest(), content_encoding=enc, size_encoded=size_enc)
            if why:
                rec["status"] = "rejected"; rec["reason"] = why
                (RAW / "_rejected").mkdir(exist_ok=True); (RAW / "_rejected" / fn).write_bytes(b); rec["file"] = "raw/_rejected/" + fn
            else:
                rec["status"] = "ok"; (RAW / fn).write_bytes(b); rec["file"] = "raw/" + fn
        with open(MAN, "a") as f: f.write(json.dumps(rec) + "\n")
        if rec["status"] == "ok": ok_ids.add(fid); break
    if i % 100 == 0: print(i, len(queue), datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), flush=True)
print("finished", flush=True)
