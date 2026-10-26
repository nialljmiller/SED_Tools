"""Parse downloaded captures -> work/parsed.json (one record per SVO ID, no curves) + work/curves.npz.
Classify against NJM FilterIDs / directories."""
import json, re, html, collections, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from fxlib import clean_curve, safe_band, pivot_wavelength
P_RE = re.compile(r'<PARAM\b([^>]*)/?>', re.S)
A_RE = re.compile(r'(\w+)="([^"]*)"')
TR_RE = re.compile(r'<TR>\s*<TD>([^<]*)</TD>\s*<TD>([^<]*)</TD>', re.S | re.I)

def parse_votable(b):
    t = b.decode("utf-8", "replace")
    params = {}
    for m in P_RE.finditer(t):
        a = dict(A_RE.findall(m.group(1)))
        if "name" in a and a["name"] not in params:
            params[a["name"]] = html.unescape(a.get("value", ""))
    rows = [(float(x), float(y)) for x, y in TR_RE.findall(t)]
    f = re.search(r'<FIELD[^>]*name="Wavelength"[^>]*>', t)
    unit = (re.search(r'unit="([^"]*)"', f.group(0)).group(1) if f and 'unit="' in f.group(0) else params.get("WavelengthUnit", ""))
    return params, rows, unit

def parse_ascii(b):
    rows = []
    for l in b.decode("latin-1").splitlines():
        s = l.split()
        if len(s) >= 2:
            try: rows.append((float(s[0]), float(s[1])))
            except ValueError: pass
    return rows

recs = [json.loads(l) for l in open(HERE / "manifest.jsonl")]
ok = collections.defaultdict(dict)
for r in recs:
    if r["status"] == "ok": ok[r["id"].lower()][r["kind"]] = r
njm_id = {}; njm_dirs = collections.defaultdict(set)
for line in open(HERE / "njm_headers.tsv"):
    p, h = line.rstrip("\n").split("\t", 1)
    p = p[2:]; d = p.rsplit("/", 1)[0]
    njm_dirs[d.lower()].add(pathlib.Path(p).stem.lower())
    m = re.search(r"FilterID=([^;]+)", h)
    if m: njm_id[m.group(1).lower()] = p
out, curves = {}, {}
for lid, kinds in sorted(ok.items()):
    rec = {"id": None, "captures": {k: {x: v[x] for x in ("wayback_url", "capture_timestamp", "sha256", "size", "file", "original_url")} for k, v in kinds.items()}}
    params, rows, unit, used = {}, [], "Angstrom", None
    if "fps_votable" in kinds:
        params, rows, unit = parse_votable((HERE / kinds["fps_votable"]["file"]).read_bytes())
        used = "fps_votable"
        pid = params.get("filterID", "")
        if pid and pid.lower() != lid: rec["problem"] = f"votable filterID {pid} != requested"
    if len(rows) < 2 and "getdata_ascii" in kinds:
        rows = parse_ascii((HERE / kinds["getdata_ascii"]["file"]).read_bytes()); used = "getdata_ascii"; unit = "Angstrom"
    rec["id"] = params.get("filterID") or kinds[next(iter(kinds))]["id"]
    rec["used"] = used; rec["params"] = {k: v for k, v in params.items() if k in (
        "DetectorType", "components", "ProfileReference", "CalibrationReference", "Description", "Comments",
        "WavelengthPivot", "WavelengthUnit", "Facility", "Instrument", "Band", "PhotSystem")}
    rec["unit"] = unit
    if rec.get("problem"): pass
    elif unit.lower() not in ("angstrom", "a", ""): rec["problem"] = f"unit {unit}"
    else:
        try:
            wl, tr, rep = clean_curve(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
            rec["clean_note"] = rep.note(); rec["n"] = int(len(wl)); curves[lid] = np.vstack([wl, tr])
        except Exception as e:
            rec["problem"] = f"curve: {e}"
    rec["njm"] = njm_id.get(lid)
    out[lid] = rec
json.dump(out, open(HERE / "work/parsed.json", "w"), indent=1)
np.savez_compressed(HERE / "work/curves.npz", **{k.replace("/", "|"): v for k, v in curves.items()})
print(len(out), "ids;", collections.Counter(r.get("problem", "ok").split(":")[0] for r in out.values()))
print(collections.Counter(r["params"].get("DetectorType", "none") for r in out.values()))
print(collections.Counter(r["params"].get("components", "none") for r in out.values()).most_common(30))
