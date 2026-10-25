import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; DL = ROOT / "downloads"
DEST = "source_evidence/filters/2026-10-25-expansion"
urls = json.load(open(ROOT / "phase2/urlmap.json"))
extra = {  # verified by fetching and comparing SHA-256 on 2026-10-25, or archive members
 "KPNO/Mosaic_k1020.txt": ("https://noirlab.edu/science/sites/default/files/media/archives/documents/k1020.txt", None),
 "NOT/ALFOSC_112.txt": ("https://www.not.iac.es/instruments/filters/curves-ascii/112.txt", None),
 "IRAS/tabC5-ipac.html": ("https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch2/tabC5.html", None),
 "IRAS/tabC5.html": ("https://irsa.ipac.caltech.edu/IRASdocs/exp.sup/ch2/tabC5.html",
     "earlier copy of the same page; differs from the 2026-10-25 fetch; the recipe uses tabC5-ipac.html"),
 "Keck/RG850.dat": ("http://www2.keck.hawaii.edu/inst/lris/data_files/RG850.dat", "held: long-pass, needs CCD QE red edge"),
 "Keck/Vbess3redonorm.txt": ("http://www2.keck.hawaii.edu/inst/deimos/Vbess3redonorm.txt", "fetched for review; not used"),
}
members = {"ATLAS/tonry_eprint_tex/": "ATLAS/tonry_1802.00879.eprint",
 "UKIRT/hewett_eprint_tex/": "UKIRT/hewett_astro-ph_0601592.eprint",
 "UKIRT/hewett_tables/": "UKIRT/hewett-ukidss-tables.tar",
 "GAIA/GaiaDR2_Passbands_ZeroPoints/": "GAIA/GaiaDR2_Passbands_ZeroPoints.zip",
 "GAIA/GaiaDR2_Revised_Passbands_ZeroPoints/": "GAIA/GaiaDR2_Revised_Passbands_ZeroPoints.zip"}
out = []
for p in sorted(x for x in DL.rglob("*") if x.is_file()):
    rel = str(p.relative_to(DL)); e = {"path": f"{DEST}/{rel}"}
    arch = next((a for m, a in members.items() if rel.startswith(m)), None)
    if rel in urls: e["source_url"] = urls[rel]
    elif rel in extra:
        e["source_url"], note = extra[rel]
        if note: e["note"] = note
    elif arch:
        e["source_url"] = urls[arch]; e["extracted_from"] = f"{DEST}/{arch}"
    else:
        sys.exit(f"no URL for {rel}")
    e["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    e["bytes"] = p.stat().st_size
    e["retrieved_utc"] = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.append(e)
(ROOT / "phase2/archive").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "phase2/archive/manifest.json", "w"), indent=2); open(ROOT / "phase2/archive/manifest.json", "a").write("\n")
print(len(out), "entries")
