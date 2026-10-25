"""verify_https.py against phase2/fix/stage/filters: verify_fix.py Facility/Instrument ..."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import verify_https
verify_https.STAGE = HERE / "stage" / "filters"
# NJM runs mod_evasive (DOSSiteCount 500 per 5 s, 60 s block): stay well under it.
import time
_get = verify_https.get
def _throttled(url):
    time.sleep(0.03)
    return _get(url)
verify_https.get = _throttled
# NJM_TUNNEL=PORT: fetch through `ssh -L PORT:127.0.0.1:443 cirrus` (same Apache vhost and files),
# for when the home IP is being throttled on the public path. No cert check on the tunnel.
import os, ssl, urllib.request
if os.environ.get("NJM_TUNNEL"):
    verify_https.BASE = f"https://127.0.0.1:{os.environ['NJM_TUNNEL']}/sed_tools"
    ctx = ssl._create_unverified_context()
    def _tget(url):
        time.sleep(0.03)
        req = urllib.request.Request(url, headers={"User-Agent": "SED_Tools-filter-expansion"})
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            return r.read()
    verify_https.get = _tget
sys.exit(verify_https.main(sys.argv[1:]))
