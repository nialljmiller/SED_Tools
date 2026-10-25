import sys, json, shutil, hashlib, csv
from pathlib import Path
import numpy as np
TOP = Path("/home/nill/SED_Workspace/filter-expansion")
sys.path.insert(0, str(TOP / "tools"))
import fxlib
FIX = TOP / "phase2/fix"; CUR = FIX / "current"; STG = FIX / "stage/filters"
INSTR = ["AAO/AAO","AlSat1/SLIM6","Astrosat/UVIT","BOK/90prime","CAHA/CAHA","CMP_SAI/SPP","CSTAR/CSTAR",
         "CTIO/CTIO","CTIO/ISPI","CTIO/SMARTS","CTIO/Spartan","DOT/TIRCAM2","Gemini/AcqCam_S"]
flags = {}
for r in csv.DictReader(open(FIX / "audit_flags.csv")):
    fi = "/".join(r["path"].split("/")[:2])
    if fi in INSTR: flags.setdefault(fi, []).append(r["path"])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(x): return repr(float(x)) if float(x) != int(x) or abs(x) < 1e15 and False else repr(float(x))
summary = []
for fi in INSTR:
    src, dst = CUR / fi, STG / fi
    shutil.copytree(src, dst, dirs_exist_ok=True)
    changes = []
    for rel in flags[fi]:
        p = CUR / rel
        lines = p.read_text().splitlines()
        header, body = lines[0], [l for l in lines[1:] if l.strip()]
        assert len(lines) - 1 == len(body), rel
        rows = [l.split(",") for l in body]
        assert all(len(r) == 2 for r in rows), rel
        wl = np.array([float(r[0]) for r in rows]); tr = np.array([float(r[1]) for r in rows])
        assert np.all(np.diff(wl) >= 0), rel  # only duplicates, no decreasing segments
        cw, ct, rep = fxlib.clean_curve(wl, tr)
        det = "photon" if "DetectorType=photon" in header else "energy"
        piv0 = fxlib.pivot_wavelength(wl, tr, det); piv1 = fxlib.pivot_wavelength(cw, ct, det)
        # rebuild: keep original text for untouched single samples
        out = [header]; i = 0
        for w, t in zip(cw, ct):
            grp = [k for k in range(i, len(wl)) if wl[k] == w]; i = grp[-1] + 1
            if len(grp) == 1 and tr[grp[0]] >= 0:
                out.append(body[grp[0]])
            else:
                out.append(f"{rows[grp[0]][0]},{format(float(t), '.12g') if t != 0 else '0.0'}")
        q = dst / Path(rel).name
        q.write_text("\n".join(out) + "\n")
        # verify
        chk = np.loadtxt(q, delimiter=",", skiprows=1)
        assert np.allclose(chk[:, 0], cw) and np.allclose(chk[:, 1], ct, rtol=1e-11, atol=1e-15)
        ops = []
        if rep.merged_duplicates:
            ops.append(f"merged {rep.merged_duplicates} duplicate wavelengths (exact repeated Wavelength values; transmissions averaged; no decreasing segments present)")
        if rep.clipped_negative:
            ops.append(f"clipped {rep.clipped_negative} negative transmissions (min {rep.min_negative:.6g}) to 0")
        ops.append(f"fxlib.clean_curve minimal curation; rows {rep.n_in}->{rep.n_out}; pivot {piv0:.3f}->{piv1:.3f} A ({(piv1/piv0-1)*100:+.4f}%); header line unchanged; untouched rows byte-identical")
        changes.append({"path": f"data/filters/{rel}", "raw_copy": f"source_evidence/filters/2026-10-25-fixes/{rel}",
                        "operation": "; ".join(ops), "raw_sha256": sha(p), "curated_sha256": sha(q)})
        summary.append((rel, rep.merged_duplicates, rep.clipped_negative, rep.min_negative, (piv1/piv0-1)*100))
    # other files byte-identical
    for f in src.iterdir():
        if f"{fi}/{f.name}" not in flags[fi]:
            assert sha(f) == sha(dst / f.name), f
    (FIX / "changes" / (fi.replace("/", "__") + ".json")).write_text(json.dumps(changes, indent=1) + "\n")
for s in summary: print(*s)
