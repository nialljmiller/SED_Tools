"""Classify parsed archive IDs against NJM and stage the missing curves.
Inputs: work/parsed.json, work/curves.npz, njm_headers.tsv, njm_props/, njm_index_files/ (existing index files),
scripts/detector_research.json (per SVO instrument), scripts/ground.json (facility -> bool for unknowns).
Outputs: stage/filters/..., stage/provenance/..., work/classification.json, work/summary.json"""
import json, re, collections, pathlib, sys, shutil
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import fxlib
from fxlib import Band, build_instrument, safe_band, pivot_wavelength, sha256
from sed_tools.filter_metadata import write_filter_file, released_mesa_reads
from sed_tools.filter_components import parse_components, recipe_name, ComponentError
from sed_tools.facility_sites import is_ground_based
STAGE = HERE / "stage"
CREDIT = "SVO Filter Profile Service (Rodrigo et al. 2012, 2020; Rodrigo & Solano 2024), recovered from the Internet Archive capture {ts}"
parsed = json.load(open(HERE / "work/parsed.json"))
cz = np.load(HERE / "work/curves.npz")
research = json.load(open(HERE / "scripts/detector_research.json")) if (HERE / "scripts/detector_research.json").exists() else {}
ground = json.load(open(HERE / "scripts/ground.json")) if (HERE / "scripts/ground.json").exists() else {}
# NJM state
njm_id, inst_map, dir_stems, dir_case, njm_det = {}, collections.defaultdict(collections.Counter), collections.defaultdict(dict), {}, collections.defaultdict(collections.Counter)
for line in open(HERE / "njm_headers.tsv"):
    p, h = line.rstrip("\n").split("\t", 1); p = p[2:]
    d = p.rsplit("/", 1)[0]; dir_case[d.lower()] = d
    dir_stems[d.lower()][pathlib.Path(p).stem.lower()] = p
    m = re.search(r"FilterID=([^;]+)", h)
    if m:
        fid = m.group(1); njm_id[fid.lower()] = p
        if "/" in fid and "." in fid:
            inst_map[fid.split(".", 1)[0].lower()][d] += 1
        dt = re.search(r"DetectorType=(\w+)", h)
        if dt and "/" in fid: njm_det[fid.split(".", 1)[0].lower()][dt.group(1)] += 1
fac_case = {d.split("/")[0].lower(): d.split("/")[0] for d in dir_case.values()}
def njm_pivot(d, stem):
    f = HERE / "njm_props" / d / "properties.json"
    if not f.exists(): return None
    b = json.load(open(f))["bands"]
    for k, v in b.items():
        if k.lower() == stem.lower(): return v.get("WavelengthPivot")
def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_+-]", "_", s.replace(".", "p")).strip("_")
# archive sibling detector types
arch_det = collections.defaultdict(collections.Counter)
for lid, r in parsed.items():
    dt = r["params"].get("DetectorType")
    if dt in ("0", "1"): arch_det[r["id"].split(".", 1)[0].lower()]["energy" if dt == "0" else "photon"] += 1
cls, plan = {}, collections.defaultdict(list)
for lid, r in sorted(parsed.items()):
    fid = r["id"]; c = {"id": fid}
    cls[lid] = c
    if r.get("njm"): c["status"] = "on_njm_filterid"; c["njm"] = r["njm"]; continue
    if r.get("problem"): c["status"] = "invalid"; c["why"] = r["problem"]; continue
    if "/" not in fid or "." not in fid.split("/", 1)[1]: c["status"] = "invalid"; c["why"] = "unparseable id"; continue
    fac, rest = fid.split("/", 1); inst, band = rest.split(".", 1)
    ikey = f"{fac}/{inst}".lower()
    if ikey in inst_map:
        d = inst_map[ikey].most_common(1)[0][0]; existing = True
        if len(inst_map[ikey]) > 1: c["note"] = f"SVO instrument spread over NJM dirs {dict(inst_map[ikey])}"
    else:
        d = f"{fac_case.get(fac.lower(), safe_name(fac))}/{safe_name(inst)}"
        existing = d.lower() in dir_case
        if existing:
            d = dir_case[d.lower()]
    bname = safe_band(band)
    c.update(dir=d, band=bname, existing_dir=existing)
    wl, tr = cz[lid.replace("/", "|")]
    adt = {"0": "energy", "1": "photon"}.get(r["params"].get("DetectorType"), "photon")
    apv = pivot_wavelength(wl, tr, adt)
    if existing and bname.lower() in dir_stems[d.lower()]:
        nd = njm_pivot(d, bname)
        shift = 100 * (apv - nd) / nd if nd else None
        c["collides_with"] = dir_stems[d.lower()][bname.lower()]; c["pivot_shift_pct"] = None if shift is None else round(shift, 3)
        if shift is not None and abs(shift) <= 0.2:
            c["status"] = "on_njm_same_band"; c["njm"] = c["collides_with"]; continue
        bname = safe_band(f"{inst}_{band}"); c["band"] = bname
        c["note"] = (c.get("note", "") + f" renamed: {c['collides_with']} is a different curve (pivot shift {c['pivot_shift_pct']} %)").strip()
        if bname.lower() in dir_stems[d.lower()]:
            c["status"] = "held"; c["why"] = f"band name collision in {d} even with instrument prefix"; continue
    else:
        F0 = d.split("/")[0].lower(); hit = None
        for dd, stems in dir_stems.items():
            if dd.split("/")[0] == F0 and dd != d.lower() and bname.lower() in stems:
                nd = njm_pivot(dir_case[dd], bname)
                if nd and abs(100 * (apv - nd) / nd) <= 0.2: hit = (stems[bname.lower()], round(100 * (apv - nd) / nd, 3))
        if hit:
            c["status"] = "on_njm_alias"; c["njm"], c["pivot_shift_pct"] = hit; continue
    # detector type
    dtp = r["params"].get("DetectorType")
    sk = fid.split(".", 1)[0].lower()
    if dtp in ("0", "1"):
        det, dsrc, dnote = ("energy" if dtp == "0" else "photon"), "svo-archive", "DetectorType PARAM in archived fps.php VOTable"
    elif len(arch_det[sk]) == 1:
        det = next(iter(arch_det[sk])); dsrc = "svo-archive-instrument"; dnote = f"archived VOTables of {sum(arch_det[sk].values())} sibling bands agree"
    elif sk in research:
        det, dsrc, dnote = research[sk]["type"], research[sk]["source"], research[sk]["note"]
    elif len(njm_det[sk]) == 1:
        det = next(iter(njm_det[sk])); dsrc = "svo-instrument"; dnote = "same SVO instrument's bands already on NJM (SVO DetectorType) agree"
    else:
        c["status"] = "held"; c["why"] = "detector type unknown"; continue
    # components
    comp = r["params"].get("components")
    try:
        roles = parse_components(comp) if comp else ()
    except ComponentError:
        roles = ()
    if roles: components, csrc = recipe_name(roles), "svo-archive"
    else: components, csrc = "filter", "assumed"; c["components_note"] = f"SVO components {comp!r} not stated/unrecognised; filter assumed"
    c.update(status="ingest", detector=det, detector_source=dsrc, detector_note=dnote, components=components, components_source=csrc)
    plan[d].append((lid, c))
# collisions among new bands in one dir
for d, items in plan.items():
    seen = collections.Counter(c["band"].lower() for _, c in items)
    for lid, c in items:
        if seen[c["band"].lower()] > 1: c["status"] = "held"; c["why"] = f"band name collision in {d}"
summary = collections.Counter(c["status"] for c in cls.values())
if "--stage" in sys.argv:
    for d, items in sorted(plan.items()):
        items = [(l, c) for l, c in items if c["status"] == "ingest"]
        if not items: continue
        F, I = d.split("/")
        bands, raws, waybacks = [], [], {}
        for lid, c in items:
            r = parsed[lid]; cap = r["captures"][r["used"]]
            wl, tr = cz[lid.replace("/", "|")]
            srcs = " | ".join(x for x in (cap["wayback_url"], CREDIT.format(ts=cap["capture_timestamp"]),
                                           r["params"].get("ProfileReference", "")) if x)
            pv = r["params"].get("WavelengthPivot")
            b = Band(c["band"], wl, tr, c["detector"], c["detector_source"], c["components"], c["components_source"], srcs,
                     filter_id=r["id"], svo_id=r["id"], clean_note=r.get("clean_note", ""),
                     ref_pivot=(float(pv), f"SVO archived VOTable {cap['capture_timestamp']}") if pv and r["used"] == "fps_votable" else None)
            bands.append(b)
            for k, v in r["captures"].items():
                raws.append(HERE / v["file"])
                waybacks[f"{c['band']}:{k}"] = {kk: v[kk] for kk in ("wayback_url", "original_url", "capture_timestamp", "sha256", "size")}
        extra = {"svo_credit": CREDIT.split(", recovered")[0], "wayback": waybacks,
                 "detector_evidence": {b.name: c["detector_note"] for b, (_, c) in zip(bands, items)},
                 "components_notes": {b.name: c.get("components_note") for b, (_, c) in zip(bands, items) if c.get("components_note")},
                 "recovered_by": "phase3/ia (filter-expansion), raw captures in source_evidence/filters/2026-10-25-svo-archive/"}
        notes = "SVO curves recovered from Internet Archive captures of the SVO Filter Profile Service (svo2.cab.inta-csic.es); see provenance."
        if not items[0][1]["existing_dir"]:
            gb = is_ground_based(F)
            if gb is None: gb = ground.get(F)
            if gb is None:
                for _, c in items: c["status"] = "held"; c["why"] = f"ground_based unknown for {F}"
                continue
            build_instrument(F, I, bands, notes=notes, raw_files=raws, extra_provenance=extra, root=STAGE, ground_based=gb)
        else:
            inst_dir = STAGE / "filters" / F / I
            inst_dir.mkdir(parents=True, exist_ok=True)
            for b in bands:
                write_filter_file(inst_dir / f"{b.name}.dat", b.wavelength, b.transmission, detector_type=b.detector_type,
                                  source=b.detector_source, filter_id=b.filter_id, wavelength_unit="Angstrom",
                                  extra={"Components": b.components, "ComponentsSource": b.components_source, "Sources": b.sources})
                assert released_mesa_reads(inst_dir / f"{b.name}.dat")
            idx = (HERE / "njm_index_files" / F / I / I)
            old = [l.strip() for l in idx.read_text().splitlines() if l.strip()]
            new = [f"{b.name}.dat" for b in sorted(bands, key=lambda b: b.name)]
            (inst_dir / I).write_text("".join(f"{n}\n" for n in old + [n for n in new if n not in old]))
            prov = {"facility": F, "instrument": I, "notes": notes + " Adds bands to an existing NJM instrument; existing files untouched.",
                    "existing_instrument": True,
                    "bands": {b.name: {"sources": b.sources, "svo_id": b.svo_id, "detector_type": b.detector_type,
                                       "detector_source": b.detector_source, "components": b.components,
                                       "components_source": b.components_source, "cleaning": b.clean_note,
                                       "ref_pivot": list(b.ref_pivot) if b.ref_pivot else None, "n": int(len(b.wavelength)),
                                       "pivot_A": round(pivot_wavelength(b.wavelength, b.transmission, b.detector_type), 2)} for b in bands},
                    "raw_files": {str(p.relative_to(HERE)): sha256(p) for p in raws}, **extra}
            (STAGE / "provenance").mkdir(parents=True, exist_ok=True)
            (STAGE / "provenance" / f"{F}__{I}.json").write_text(json.dumps(prov, indent=2) + "\n")
    summary = collections.Counter(c["status"] for c in cls.values())
json.dump(cls, open(HERE / "work/classification.json", "w"), indent=1)
dirs = collections.Counter((c["dir"], c["existing_dir"]) for c in cls.values() if c["status"] == "ingest")
json.dump({"status": summary, "dirs_new": sorted(d for (d, e) in dirs if not e), "dirs_existing": sorted(d for (d, e) in dirs if e)},
          open(HERE / "work/summary.json", "w"), indent=1)
print(dict(summary)); print("dirs new", sum(1 for (d, e) in dirs if not e), "existing", sum(1 for (d, e) in dirs if e))
print(collections.Counter(c.get("why") for c in cls.values() if c["status"] in ("held", "invalid")).most_common(10))
