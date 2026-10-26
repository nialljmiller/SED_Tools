"""W5 (Nill's decisions 2026-10-26 item 8): data fixes on the Cirrus server checkout.

Run from /media/data3/MESA/SED_Tools inside the shared flock:  python3 apply_w5.py PLAN.json [--dry-run]
Nothing is deleted. Every changed curve is archived first; data rows are verified byte-identical.
Parts: (1) BOK/90prime R -> R_Bessell rename; (2) curated band fields and notes; (3) DetectorType headers
(GOES/ABI1-4, JPSS/VIIRS-J1/J2 assumed -> original; SPHERE/IRDIS original -> inferred); (4) legacy header gaps.
"""
import csv, datetime, glob, hashlib, json, os, shutil, sys
from pathlib import Path
from sed_tools.filter_metadata import read_filter_header, released_mesa_reads

DRY = "--dry-run" in sys.argv
PLAN = json.load(open(sys.argv[1]))
NOW = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
F = Path("data/filters")
CUR = Path("filter_info/curated/instruments")
EV = Path("source_evidence/filters")
SVO_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="
ESC = {c: f"%{ord(c):02X}" for c in '% \t\r\n,;"#'}
log = []


def esc(v):
    return "".join(ESC.get(c, c) for c in str(v))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def archive(path, topic):
    dest = EV / topic / Path(path).relative_to(F)
    if not DRY:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != sha(path):
            dest = dest.with_name(dest.stem + f".{NOW}" + dest.suffix)
        if not dest.exists():
            shutil.copy2(path, dest)
    return str(dest)


def edit_header(path, topic, add=None, replace=None, why=""):
    """Add fields (FilterID first, the rest appended) and/or replace 'old'->'new' substrings on line 1 only."""
    raw = Path(path).read_bytes()
    nl = raw.find(b"\n")
    first, rest = raw[:nl], raw[nl:]
    cr = first.endswith(b"\r")
    line = first.rstrip(b"\r").decode()
    assert line.startswith("Wavelength,Transmission,"), (path, line[:60])
    before = read_filter_header(path)
    new = line
    for old, rep in (replace or {}).items():
        assert old in new, (path, old)
        new = new.replace(old, rep, 1)
    add = {k: v for k, v in (add or {}).items() if v and k not in before}
    added = dict(add)
    if "FilterID" in add:
        new = new.replace("Wavelength,Transmission,", f"Wavelength,Transmission,FilterID={esc(add.pop('FilterID'))};", 1)
    for k, v in add.items():
        new += f";{k}={esc(v)}"
    if new == line:
        return None
    out = new.encode() + (b"\r" if cr else b"") + rest
    rec = {"path": str(path), "raw_sha256": hashlib.sha256(raw).hexdigest(), "operation": why,
           "header_before": line.split(",", 2)[2], "header_after": new.split(",", 2)[2]}
    if not DRY:
        rec["raw_copy"] = archive(path, topic)
        tmp = Path(str(path) + ".w5tmp")
        tmp.write_bytes(out)
        shutil.copymode(path, tmp)
        after = read_filter_header(tmp)
        assert tmp.read_bytes()[len(out) - len(rest):] == rest, path          # data rows identical
        replaced = {o.split("=", 1)[0] for o in (replace or {})}
        assert all(after.get(k) == v for k, v in before.items() if k not in replaced), path
        assert all(after.get(k) == str(v) for k, v in added.items()), (path, after)
        assert released_mesa_reads(tmp), path
        os.replace(tmp, path)
        rec["new_sha256"] = sha(path)
    rec["data_rows_identical"] = True
    return rec


def load_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, d):
    if not DRY:
        Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- (1) BOK/90prime R -> R_Bessell
bok = F / "BOK/90prime"
rename_rec = None
if (bok / "R.dat").exists() and not (bok / "R_Bessell.dat").exists():
    topic = "2026-10-26-bok-rename"
    if not DRY:
        shutil.copytree(bok, Path("backups") / f"bok-rename-{NOW}" / "BOK/90prime")
        a = archive(bok / "R.dat", topic)
        man = {"date": "2026-10-26", "reason": "R and r collide case-insensitively; R renamed R_Bessell (DECISIONS-20261026 item 8)",
               "files": [{"path": a, "original_path": str(bok / "R.dat"), "sha256": sha(a)},
                         {"path": str(EV / topic / "BOK/90prime/90prime"), "original_path": str(bok / "90prime")}]}
        shutil.copy2(bok / "90prime", EV / topic / "BOK/90prime/90prime")
        man["files"][1]["sha256"] = sha(EV / topic / "BOK/90prime/90prime")
        (EV / topic / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
        old_sha = sha(bok / "R.dat")
        os.rename(bok / "R.dat", bok / "R_Bessell.dat")            # target checked absent above
        assert sha(bok / "R_Bessell.dat") == old_sha
        idx = (bok / "90prime").read_text().splitlines()
        assert idx.count("R.dat") == 1 and "R_Bessell.dat" not in idx
        (bok / "90prime").write_text("\n".join("R_Bessell.dat" if x == "R.dat" else x for x in idx) + "\n")
        with open("filter_expansion/renamed_filters_20261026_bok.csv", "x", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["old_path", "new_path"])
            w.writerow(["BOK/90prime/R.dat", "BOK/90prime/R_Bessell.dat"])
    rename_rec = {"old": "BOK/90prime/R.dat", "new": "BOK/90prime/R_Bessell.dat", "backup": f"backups/bok-rename-{NOW}"}
print("rename:", rename_rec)

# ---------------------------------------------------------------- (2) curated band fields
curated_changed = set()


def curated(inst):
    p = CUR / (inst.replace("/", "__") + ".json")
    return p, load_json(p)


for inst in sorted(set(PLAN["bands"]) | set(PLAN["comments_append"])):
    p, d = curated(inst)
    info = load_json(F / inst / "info.json")
    bands = d.setdefault("bands", {})
    changed = False
    for band, fields in PLAN["bands"].get(inst, {}).items():
        b = bands.setdefault(band, {})
        for k, v in fields.items():
            if b.get(k, "<absent>") != v:
                b[k] = v
                changed = True
    for band, text in PLAN["comments_append"].get(inst, {}).items():
        b = bands.setdefault(band, {})
        ib = info["bands"].get("R" if band == "R_Bessell" else band, {})
        base = b.get("Comments") or ib.get("Comments") or ""
        if text not in base:
            b["Comments"] = (base.rstrip() + " " + text).strip()
            changed = True
    if changed:
        write_json(p, d)
        curated_changed.add(str(p))
print("curated band edits:", len(curated_changed))

# ---------------------------------------------------------------- (3) DetectorType headers
dt_changes = []
VIIRS = ("DetectorTypeSource changed from assumed to original (2026-10-26, header only, curves unchanged): NOAA's VIIRS "
         "Radiometric Calibration ATBD (474-00027 Rev. E, effective 2022-07-22, "
         "https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/D0001-M01-S01-003_JPSS_ATBD_VIIRS-SDR_E.pdf, §2.3 General Radiometric "
         "Equations, Eq. 9) states 'The Relative Spectral Response (RSR) is defined as the normalized product of the quantum efficiency, "
         "wavelength, and optical transmittance. It is normalized by the maximum value of this product.', and band-averages radiance as "
         "∫RSR(λ)·L(λ)dλ / ∫RSR(λ)dλ (Eq. 12, 14). The wavelength factor is folded into the RSR, so it integrates directly over F_λ: "
         "energy in SED_Tools' convention (as for 2MASS).")
ABI = ("DetectorTypeSource changed from assumed to original (2026-10-26, header only, curves unchanged): the GOES-R Calibration "
       "Working Group (Efremova, Pearlman, Padula & Wu 2016, Proc. SPIE 9972, 99720S, doi:10.1117/12.2238090; NOAA repository "
       "https://repository.library.noaa.gov/view/noaa/62008) writes that the detector output 'is proportional to the source spectral "
       "radiance L(λ) weighted by the detector specific spectral response' (Eq. 1, dn = C∫L(λ)SRF(λ)dλ) and computes the band-average "
       "calibration-source radiance as ∫L(λ)SRF_AV(λ)dλ / ∫SRF_AV(λ)dλ (Eq. 2), where 'a band-average SRF (SRF_AV) is used "
       "operationally to represent the spectral channel'. The SRF thus integrates directly over spectral radiance: energy in "
       "SED_Tools' convention.")
SPH = ("SPHERE/IRDIS headers (2026-10-26): these 24 curves are numerically identical (same 2301-point grid, ratio 1.00000) to "
       "Paranal/SPHERE, which carries ESO's SPHERE_IRDIS_*.dat files "
       "(https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters.html). FilterID, WavelengthUnit, Components=filter "
       "(original) and the ESO Sources URL were copied from the Paranal/SPHERE counterpart, and DetectorTypeSource original was "
       "changed to inferred, matching Paranal/SPHERE: no owner statement of the integration convention exists, the photon label rests "
       "on the filter-only curve and the IRDIS Hawaii II RG detector (block B review, phase4/reports/B-detectortype.md).")


def manifest_sources(inst, bands, old, new):
    mp = F / inst / "manifest.json"
    if not mp.exists():
        return
    m = load_json(mp)
    src = m.get("detector_type_sources") or {}
    hit = [b for b in bands if src.get(b) == old]
    if not hit:
        return
    raw_sha = sha(mp)
    copy = archive(mp, "2026-10-26-detectortype2") if not DRY else None
    for b in hit:
        src[b] = new
    m["notes"] = (m.get("notes", "") + f" [2026-10-26: DetectorTypeSource {old} -> {new}; evidence in the curated notes.]").strip()
    write_json(mp, m)
    dt_changes.append({"path": str(mp), "raw_copy": copy, "raw_sha256": raw_sha, "new_sha256": None if DRY else sha(mp),
                       "operation": f"manifest.json: detector_type_sources {old} -> {new} for {len(hit)} bands; note appended"})


for inst, why in [("GOES/ABI1", ABI), ("GOES/ABI2", ABI), ("GOES/ABI3", ABI), ("GOES/ABI4", ABI),
                  ("JPSS/VIIRS-J1", VIIRS), ("JPSS/VIIRS-J2", VIIRS)]:
    done = []
    for path in sorted((F / inst).glob("*.dat")):
        h = read_filter_header(path)
        if h.get("DetectorType") == "energy" and h.get("DetectorTypeSource") == "assumed":
            r = edit_header(path, "2026-10-26-detectortype2", replace={"DetectorTypeSource=assumed": "DetectorTypeSource=original"},
                            why="header row: DetectorTypeSource=assumed -> original; DetectorType energy and curve unchanged. " + why)
            if r:
                dt_changes.append(r)
                done.append(path.stem)
    manifest_sources(inst, done, "assumed", "original")
    p, d = curated(inst)
    if why not in d.get("notes", ""):
        d["notes"] = (d.get("notes", "").rstrip() + " " + why).strip()
        write_json(p, d)
        curated_changed.add(str(p))
    print(inst, len(done), "headers -> original")

# SPHERE/IRDIS: copy the ESO provenance of the identical Paranal/SPHERE curves
sph_done = []
for path in sorted((F / "SPHERE/IRDIS").glob("*.dat")):
    twin = F / "Paranal/SPHERE" / path.name
    if not twin.exists():
        continue
    a = [l.split(",")[:2] for l in path.read_text().splitlines()[1:]]
    b = [l.split(",")[:2] for l in twin.read_text().splitlines()[1:]]
    if len(a) != len(b) or any(abs(float(x[0]) - float(y[0])) > 1e-6 or abs(float(x[1]) - float(y[1])) > 1e-9 * max(1, abs(float(y[1])))
                               for x, y in zip(a, b)):
        print("SPHERE/IRDIS differs from twin:", path.name)
        continue
    th = read_filter_header(twin)
    h = read_filter_header(path)
    rep = {"DetectorTypeSource=original": "DetectorTypeSource=inferred"} if h.get("DetectorTypeSource") == "original" else {}
    r = edit_header(path, "2026-10-26-legacy-headers",
                    add={"FilterID": f"SPHERE/IRDIS.{path.stem}", "WavelengthUnit": th.get("WavelengthUnit"),
                         "Components": th.get("Components"), "ComponentsSource": th.get("ComponentsSource"),
                         "Sources": th.get("Sources")},
                    replace=rep, why="header only: " + SPH)
    if r:
        (dt_changes if rep else []).append(r)
        sph_done.append(r)
if sph_done:
    p, d = curated("SPHERE/IRDIS")
    if SPH not in d.get("notes", ""):
        d["notes"] = (d.get("notes", "").rstrip() + " " + SPH).strip()
        write_json(p, d)
        curated_changed.add(str(p))
print("SPHERE/IRDIS", len(sph_done))

# ---------------------------------------------------------------- (4) legacy header gaps
SKIP_DIRS = {"JWST/NIRCam", "Roman/WFI", "Generic/Johnson", "Sentinel/MSI-2A", "Sentinel/MSI-2B", "Sentinel/MSI-2C", "JPSS/VIIRS",
             "SPHERE/IRDIS"}
FID = set(PLAN.get("filterid_add", []))
legacy = []
for d in sorted(glob.glob(str(F / "*/*/"))):
    inst = os.path.relpath(d, F)
    if inst in SKIP_DIRS:
        continue
    ip = Path(d) / "info.json"
    info = load_json(ip) if ip.exists() else {"bands": {}}
    for path in sorted(Path(d).glob("*.dat")):
        band = path.stem
        if band.endswith("_svo") or band.endswith("_old"):
            continue
        h = read_filter_header(path)
        add, why = {}, []
        s = (info["bands"].get(band) or {}).get("svo") or {}
        fid = h.get("FilterID")
        if (fid and s.get("curve") == "identical" and s.get("matched_by") == "FilterID" and s.get("filterID") == fid
                and h.get("ComponentsSource") in (None, "svo")):
            v = (s.get("captures") or {}).get("votable") or {}
            ev = f"array identical (8 s.f.) to SVO's {fid} VOTable captured {v.get('timestamp')} ({v.get('url')}, sha256 {v.get('sha256')})"
            if "Sources" not in h:
                add["Sources"] = SVO_URL + fid
                why.append("Sources = the SVO FPS record: " + ev)
            if "Components" not in h and s.get("components"):
                from sed_tools.filter_metadata import svo_components_header
                comp = svo_components_header(s["components"])
                if comp:
                    add.update(comp)
                    why.append(f"Components from that SVO record's components '{s['components']}'")
        elif fid and inst.startswith("HST/") and "Sources" not in h:
            add["Sources"] = SVO_URL + fid
            why.append("Sources = the SVO FPS record: prepare_hst_filters.py (server repo, commit 9667e82) 'Download and install the "
                       "complete SVO HST passband collection' from https://svo2.cab.inta-csic.es/theory/fps/fps.php")
        if not fid and f"{inst}/{band}" in FID:
            add["FilterID"] = f"{inst}.{band}"
            why.append("FilterID = NJM's <Facility>/<Instrument>.<band> identifier (filter-expansion default, tools/fxlib.py); not an "
                       "SVO ID (checked against the SVO ID lists)")
        if add:
            r = edit_header(path, "2026-10-26-legacy-headers", add=add, why="header only, curve unchanged: " + "; ".join(why))
            if r:
                legacy.append(r)
print("legacy header edits:", len(legacy))
if DRY:
    Path("/tmp/w5/dry_legacy.json").write_text(json.dumps(legacy, indent=1, ensure_ascii=False))

# ---------------------------------------------------------------- records
stamp = {"date": "2026-10-26", "worker": "W5 (room request b80ec7c4023ccc09af15fb66)", "decisions": "DECISIONS-20261026.md item 8"}
if not DRY:
    for name, changes, arch in (("detectortype2_changes.json", dt_changes, "2026-10-26-detectortype2"),
                                ("legacy_header_changes.json", legacy + sph_done,
                                 "2026-10-26-legacy-headers")):
        doc = dict(stamp, archive=str(EV / arch), curated_notes_updated=sorted(curated_changed) if "detector" in name else [],
                   changes=changes)
        Path("filter_expansion/phase4", name).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        (EV / arch).mkdir(parents=True, exist_ok=True)
        (EV / arch / "changes.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
if not DRY:
    Path("/tmp/w5/curated.json").write_text(json.dumps(sorted(curated_changed)))
json.dump({"rename": rename_rec, "curated": sorted(curated_changed), "dt": len(dt_changes), "sphere": len(sph_done),
           "legacy": len(legacy)}, sys.stdout)
print("\nDRY RUN" if DRY else "\ndone")
