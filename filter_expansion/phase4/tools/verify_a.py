"""Model-free verifier for phase-4 block-A research files (BRIEF-A.md).

Usage: .venv/bin/python -I verify_a.py CHUNK_DIR [CHUNK_DIR ...]
For each <Facility>__<Instrument>.json in a chunk directory it checks, and repairs by removal only:
  - the instrument and band names exist on NJM (reports/njm_instruments.json);
  - every reference key resolves (NJM's resolver, phase3/research/lib/references_copy.py);
  - links load (HTTP 200 after redirects, not a login page);
  - acknowledgement text appears verbatim on its url; licence terms quote appears on its url;
  - each per-band field has an evidence quote found on its url.
Writes phase4/research/verified/<name>.json (evidence moved to verified/evidence/), and
CHUNK_DIR/VERIFY.tsv: name, verdict, problems. Fetched pages are cached in phase4/research/cache/.
"""
import hashlib, html, importlib.util, json, pathlib, re, subprocess, sys, time

P4 = pathlib.Path("/home/nill/SED_Workspace/filter-expansion/phase4")
LIB = pathlib.Path("/home/nill/SED_Workspace/filter-expansion/phase3/research/lib/references_copy.py")
CACHE = P4 / "research/cache"; OUT = P4 / "research/verified"
NJM = json.loads((P4 / "reports/njm_instruments.json").read_text())
CLASSES = {"public", "attribution", "permission", "restricted", "unknown"}
BAND_FIELDS = {"Description", "Comments", "PhotSystem", "Band", "MagSys", "ProfileReference",
               "CalibrationReference", "references"}
UA = "Mozilla/5.0 (X11; Linux x86_64) NJM-filter-metadata-check (niall.j.miller@gmail.com)"

spec = importlib.util.spec_from_file_location("refs", LIB)
refs = importlib.util.module_from_spec(spec); spec.loader.exec_module(refs)
LOOK = refs._Lookup()
_ref_ok = {}


def ref_ok(key):
    if key not in _ref_ok:
        e = LOOK.resolve(key)
        _ref_ok[key] = bool(e.get("resolved_by") is not None or e.get("title"))
    return _ref_ok[key]


def fetch(url):
    """(status, text) with whitespace-normalised lower-case text; cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    meta = CACHE / f"{h}.json"
    if meta.exists():
        m = json.loads(meta.read_text()); return m["status"], (CACHE / f"{h}.txt").read_text()
    raw = CACHE / f"{h}.raw"
    r = subprocess.run(["curl", "-sL", "--compressed", "--max-time", "40", "-A", UA, "-o", str(raw),
                        "-w", "%{http_code} %{url_effective} %{content_type}", url],
                       capture_output=True, text=True)
    parts = (r.stdout or "0 - -").split(" ", 2)
    status, final, ctype = int(parts[0] or 0), parts[1], (parts[2] if len(parts) > 2 else "")
    data = raw.read_bytes() if raw.exists() else b""
    if "pdf" in ctype.lower() or data[:5] == b"%PDF-":
        t = subprocess.run(["pdftotext", "-q", str(raw), "-"], capture_output=True).stdout.decode("utf-8", "replace")
    else:
        t = data.decode("utf-8", "replace")
        t = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", t)
        t = html.unescape(re.sub(r"(?s)<[^>]+>", " ", t))
    if re.search(r"(?i)(login|signin|sign-in|/sso/|auth\?)", final) and final != url:
        status = 401
    text = norm(t)
    if status == 200 and re.search(r"(document|page|file) not found|error 404|404 not found", text[:600]):
        status = 404  # soft 404
    (CACHE / f"{h}.txt").write_text(text)
    meta.write_text(json.dumps({"url": url, "status": status, "final": final, "ctype": ctype,
                                "fetched": time.strftime("%FT%TZ", time.gmtime())}))
    time.sleep(1)
    return status, text


def norm(s):
    s = s.replace("­", "").replace("-\n", "")
    s = s.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                   "–": "-", "—": "-", " ": " "}))
    return re.sub(r"\s+", " ", s).strip().lower()


def found(quote, url):
    if not quote or not url:
        return False
    status, text = fetch(url)
    return status == 200 and norm(quote) in text


def quotes(terms):
    return [q for q in re.findall(r'"([^"]{15,})"', norm(terms or "").replace("“", '"'))]


def verify(path):
    d = json.loads(path.read_text())
    name = path.stem
    fac_inst = name.replace("__", "/", 1)
    probs, fixes = [], []
    if fac_inst not in NJM:
        return name, "REJECT", ["not an NJM instrument"], None
    njm_bands = set(NJM[fac_inst]["bands"])
    evidence = d.pop("evidence", []) or []

    def keep_refs(keys, where):
        good = [k for k in keys if ref_ok(k)]
        for k in keys:
            if k not in good:
                fixes.append(f"dropped unresolved {k} ({where})")
        return good

    d["references"] = keep_refs(d.get("references", []), "references")
    links = []
    for l in d.get("links", []):
        st, _ = fetch(l.get("url", ""))
        (links.append(l) if st == 200 else fixes.append(f"dropped link {l.get('url')} (HTTP {st})"))
    d["links"] = links
    acks = []
    for a in d.get("acknowledgements", []):
        if not (a.get("for") and a.get("text") and a.get("url")):
            fixes.append(f"dropped incomplete acknowledgement {a.get('for')}"); continue
        if not found(a["text"], a["url"]):
            fixes.append(f"dropped acknowledgement {a['for']}: text not found verbatim on {a['url']}"); continue
        a["references"] = keep_refs(a.get("references", []), f"acknowledgement {a['for']}")
        acks.append(a)
    d["acknowledgements"] = acks
    lic = d.get("licence")
    if lic:
        cls = lic.get("redistribution")
        qs = quotes(lic.get("terms", ""))
        if cls not in CLASSES or not lic.get("url"):
            fixes.append(f"dropped licence (class {cls!r}, url {lic.get('url')!r})"); d.pop("licence")
        elif cls != "unknown" and not (qs and all(found(q, lic["url"]) for q in qs)):
            fixes.append(f"dropped licence {cls}: terms quote not found on {lic['url']}"); d.pop("licence")
    ev = {}
    for e in evidence:
        if found(e.get("quote"), e.get("url")):
            ev.setdefault(e.get("field"), e)
    bands = {}
    for b, fields in (d.get("bands") or {}).items():
        if b != "*" and b not in njm_bands:
            fixes.append(f"dropped band {b} (not on NJM)"); continue
        kept = {}
        for f, v in fields.items():
            if f not in BAND_FIELDS:
                fixes.append(f"dropped bands.{b}.{f} (not a curated field)"); continue
            if f"bands.{b}.{f}" not in ev:
                fixes.append(f"dropped bands.{b}.{f} (no verified evidence)"); continue
            if f in ("ProfileReference", "CalibrationReference"):
                v = [r for r in v if ref_ok(r["ref"] if isinstance(r, dict) else r)]
            elif f == "references":
                v = keep_refs(v, f"bands.{b}")
            if v:
                kept[f] = v
        if kept:
            bands[b] = kept
    d["bands"] = bands
    for k in [k for k, v in d.items() if v in ([], {}, "", None)]:
        d.pop(k)
    if not d.get("description"): probs.append("no description")
    if not d.get("facility_description"): probs.append("no facility_description")
    if not any(k.split(":")[0] in ("bibcode", "doi", "arxiv") for k in d.get("references", [])):
        probs.append("no paper reference")
    if "licence" not in d: probs.append("no licence")
    elif d["licence"]["redistribution"] in ("unknown", "permission", "restricted"):
        probs.append(f"licence {d['licence']['redistribution']}")
    if not d.get("acknowledgements"): probs.append("no acknowledgement")
    verdict = "PASS" if not probs else "PARTIAL"
    used = [e for f, e in ev.items() if not (f or "").startswith("bands.") or f.split(".")[1] in bands]
    return name, verdict, probs + fixes, (d, used)


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "evidence").mkdir(exist_ok=True)
    for cdir in map(pathlib.Path, sys.argv[1:]):
        rows = []
        for path in sorted(cdir.glob("*__*.json")):
            try:
                name, verdict, notes, res = verify(path)
            except Exception as exc:  # malformed file
                name, verdict, notes, res = path.stem, "REJECT", [f"error {exc!r}"], None
            if res:
                d, used = res
                (OUT / f"{name}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
                (OUT / "evidence" / f"{name}.json").write_text(json.dumps(used, indent=2, ensure_ascii=False) + "\n")
            rows.append(f"{name}\t{verdict}\t{'; '.join(notes)}")
            print(rows[-1], flush=True)
        (cdir / "VERIFY.tsv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
