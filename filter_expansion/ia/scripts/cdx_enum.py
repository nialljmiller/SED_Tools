"""Enumerate IA CDX captures of SVO FPS URLs. Saves raw CDX JSON per pattern in ../cdx/."""
import json, sys, time, urllib.request, urllib.parse, pathlib
OUT = pathlib.Path(__file__).resolve().parent.parent / "cdx"
HOSTS = ["svo2.cab.inta-csic.es", "svo.cab.inta-csic.es"]
PATHS = ["theory/fps/getdata.php*", "svo/theory/fps3/getdata.php*", "theory/fps3/*",
         "theory/fps/fps.php*", "svo/theory/fps/*", "theory/fps/*", "svo/theory/fps3/*", "theory/fps2/*"]
def get(url, tries=8):
    d = 5
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "SED_Tools-archive-recovery/1.0 (niall.j.miller research)"}), timeout=180) as r:
                return r.read()
        except Exception as e:
            print("  retry", k, e, flush=True); time.sleep(d); d = min(d * 2, 300)
    raise RuntimeError(url)
for h in HOSTS:
    for p in PATHS:
        u = f"{h}/{p}"
        fn = OUT / (u.replace("/", "_").replace("*", "STAR") + ".json")
        if fn.exists(): continue
        q = urllib.parse.urlencode({"url": u, "output": "json", "filter": "statuscode:200",
                                    "fl": "urlkey,timestamp,original,mimetype,statuscode,digest,length"})
        b = get("https://web.archive.org/cdx/search/cdx?" + q)
        rows = json.loads(b) if b.strip() else []
        fn.write_bytes(b if b.strip() else b"[]")
        print(u, max(len(rows) - 1, 0), flush=True); time.sleep(2)
