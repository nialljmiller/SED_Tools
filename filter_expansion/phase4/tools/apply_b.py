"""Block B: apply the reviewed DetectorType header changes on the Cirrus server checkout.

Run on Cirrus from /media/data3/MESA/SED_Tools (piped over ssh). Archives every file it
changes under source_evidence/filters/2026-10-26-detectortype/ first, edits only the
curve header row and manifest.json detector_type_sources, appends the evidence sentence to
the curated instrument notes, and writes filter_expansion/phase4/detectortype_changes.json.
Nothing is deleted. Pass --dry-run as argv[1] to report without writing.
"""
import hashlib, json, shutil, sys
from pathlib import Path

DRY = "--dry-run" in sys.argv
STAMP = "2026-10-26-detectortype"
ROOT = Path(".")
ARCHIVE = ROOT / "source_evidence/filters" / STAMP
CUR = ROOT / "filter_info/curated/instruments"
REPORT = "phase4/reports/B-detectortype.md (request cf1ebced)"

INFERRED = {
    "Paranal/NACO": "ESO publishes filter-only transmissions with no QE or wavelength factor (NACO manual VLT-MAN-ESO-14200-2761 §8), and CONICA's InSb Aladdin array counts photons (manual Table 5-1), so the curves integrate as photon counters. SVO's energy label sits on arrays identical to these (14 bands compared, ratio 1) and gives no basis.",
    "Paranal/HAWKI": "The ESO .dat files are filter curves only ('BrGamma filter curve @ T=90K', Wavelength (nm), T (%)), and HAWK-I's detectors are 'a mosaic of four Hawaii 2RG 2048 * 2048 pixels detectors' (https://www.eso.org/sci/facilities/paranal/instruments/hawki/overview.html), so the curves integrate as photon counters.",
    "Paranal/SPHERE": "The IRDIS .dat files are filter transmissions ('SPHERE/IRDIS BB B_H filter transmission'), and the IRDIS detector is a 'Hawaii II RG 2048 x 1024' (https://www.eso.org/sci/facilities/paranal/instruments/sphere/overview.html), so the curves integrate as photon counters.",
    "IRTF/NSFCam": "The filter page gives 'absolute transmissions (on a scale of 0 to 1)' and 'only the transmission of the filter itself' (https://irtfweb.ifa.hawaii.edu/~nsfcam/filters.html), and NSFCam has 'a 256x256 InSb detector' (https://irtfweb.ifa.hawaii.edu/~nsfcam/), so the curves integrate as photon counters.",
    "WHT/INGRID": "The ING files are filter %T scans ('Filter Brg Temp=77K', https://www.ing.iac.es/Astronomy/instruments/ingrid_new/filter/brg.cold.trans.html), and INGRID 'uses a 1k x 1k HAWAII detector' (https://www.ing.iac.es/Astronomy/instruments/ingrid_new/), so the curves integrate as photon counters.",
    "Gemini/NIFS": "The Gemini .dat files are witness scans of the blocking filters (HK.dat '#ED052-W H-K filter'), and the NIFS detector is a 'Rockwell HAWAII-2RG (H2RG)' HgCdTe device (https://www.gemini.edu/instrumentation/nifs/components), so the curves integrate as photon counters. SVO's energy label for HK and ZJ sits on identical arrays and gives no basis.",
    "Liverpool/IOO": "These are the filter columns of the IO:O efficiency-model table (doi:10.6084/m9.figshare.4659421), whose Technical Note 1 measures 'the absolute photon efficiency' with each filter 'multiplied by the CCD QE'; the detector is an e2v CCD 231 (https://telescope.livjm.ac.uk/TelInst/Inst/IOO/). The curves therefore integrate as photon counters.",
}
PREFIX = "DetectorType photon, DetectorTypeSource inferred (2026-10-26): no owner states the integration convention for the curve file, but "
SUFFIX = " The source is 'inferred' because this rests on the published curve content and the documented detector, not on an explicit owner statement."
TWOMASS = ("DetectorType changed to energy, DetectorTypeSource original (2026-10-26): Cohen, Wheaton & Megeath (2003) §3, quoted on the "
           "Explanatory Supplement VI.4a page, say these RSRs 'are designed to be integrated directly over stellar spectra in Fλ form' "
           "because the wavelength factor is already folded in ('The QE-based component was converted to yield photon-counting RSRs by "
           "multiplying by wavelength and renormalized'). In SED_Tools' convention that is an energy integration; NJM's earlier photon "
           "label came from SVO's 2025 relabelling of 2MASS.J and is not used.")
OLD_2MASS = "; DetectorType left unchanged."


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def archive(path):
    dest = ARCHIVE / Path(path).relative_to("data/filters")
    if not DRY:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(path, dest)
    return str(dest)


changes = []


def record(path, raw_copy, raw_sha, operation):
    changes.append({"path": str(path), "raw_copy": raw_copy, "raw_sha256": raw_sha,
                    "curated_sha256": None if DRY else sha(path), "operation": operation})


def edit_header(path, old, new, why):
    lines = Path(path).read_text().split("\n")
    if old not in lines[0]:
        return False
    raw_sha, copy = sha(path), archive(path)
    lines[0] = lines[0].replace(old, new)
    if not DRY:
        Path(path).write_text("\n".join(lines))
    record(path, copy, raw_sha, f"header row: {old} -> {new}; curve unchanged. {why}")
    return True


def edit_json(path, fn, op):
    raw_sha = sha(path)
    data = json.loads(Path(path).read_text())
    if not fn(data):
        return
    copy = archive(path)
    if not DRY:
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    record(path, copy, raw_sha, op)


curated_edits = []
for inst, why in INFERRED.items():
    d = ROOT / "data/filters" / inst
    bands = [p.stem for p in sorted(d.glob("*.dat"))
             if edit_header(p, "DetectorTypeSource=assumed", "DetectorTypeSource=inferred", "Evidence: " + why)]

    def man(m, bands=bands):
        src = m.get("detector_type_sources") or {}
        hit = [b for b in bands if src.get(b) == "assumed"]
        for b in hit:
            src[b] = "inferred"
        return bool(hit)
    if (d / "manifest.json").exists():
        edit_json(d / "manifest.json", man, f"manifest.json: detector_type_sources assumed -> inferred for {len(bands)} bands; nothing else changed")
    note = PREFIX + why[0].lower() + why[1:] + SUFFIX
    cf = CUR / (inst.replace("/", "__") + ".json")
    c = json.loads(cf.read_text())
    if note not in c.get("notes", ""):
        c["notes"] = (c.get("notes", "").rstrip() + " " + note).strip()
        curated_edits.append(str(cf))
        if not DRY:
            cf.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n")
    print(inst, len(bands), "bands")

d = ROOT / "data/filters/2MASS/2MASS"
n = sum(edit_header(p, "DetectorType=photon;DetectorTypeSource=svo", "DetectorType=energy;DetectorTypeSource=original",
                    "Evidence: " + TWOMASS) for p in sorted(d.glob("*.dat")))
cf = CUR / "2MASS__2MASS.json"
c = json.loads(cf.read_text())
if TWOMASS not in c["notes"]:
    c["notes"] = c["notes"].replace(OLD_2MASS, ".") + " " + TWOMASS
    curated_edits.append(str(cf))
    if not DRY:
        cf.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n")
print("2MASS/2MASS", n, "bands")

out = ROOT / "filter_expansion/phase4/detectortype_changes.json"
doc = {"date": "2026-10-26", "evidence": REPORT, "archive": str(ARCHIVE),
       "curated_notes_updated": curated_edits, "changes": changes}
if not DRY:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    (ARCHIVE / "changes.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
print(len(changes), "file changes;", len(curated_edits), "curated notes", "(dry run)" if DRY else "")
