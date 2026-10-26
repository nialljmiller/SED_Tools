"""Merge verified block-A files into the server checkout's curated instruments, refresh, commit, push.

Usage: python3 integrate_a.py [--dry-run] -m "commit subject" NAME [NAME ...]
NAME is <Facility>__<Instrument>; the input is phase4/research/verified/NAME.json (verify_a.py output).
Merge rules with the existing curated file: existing description/facility_description/notes/licence win
(a new licence is added only if none exists, or the existing one is 'unknown'); references, links (by url)
and acknowledgements (by 'for') are unioned; band fields are added only where the band lacks them.
Merged files are staged in phase4/research/merged/ and are what gets copied. Nothing is deleted.
"""
import json, pathlib, subprocess, sys

P4 = pathlib.Path("/home/nill/SED_Workspace/filter-expansion/phase4")
SSH = ["ssh", "-q", "-F", "/home/nill/.ssh/config.agent-network", "cirrus"]
REPO = "/media/data3/MESA/SED_Tools"
CUR = "filter_info/curated/instruments"


def existing(name):
    r = subprocess.run(SSH + [f"cat {REPO}/{CUR}/{name}.json 2>/dev/null"], capture_output=True, text=True)
    return json.loads(r.stdout) if r.stdout.strip() else {}


def merge(old, new):
    m = dict(old)
    for k in ("description", "facility_description", "notes"):
        if not m.get(k) and new.get(k):
            m[k] = new[k]
    if new.get("licence") and (not m.get("licence") or m["licence"].get("redistribution") == "unknown"):
        m["licence"] = new["licence"]
    m["references"] = list(dict.fromkeys(m.get("references", []) + new.get("references", [])))
    urls = {l["url"] for l in m.get("links", [])}
    m["links"] = m.get("links", []) + [l for l in new.get("links", []) if l["url"] not in urls]
    fors = {a["for"] for a in m.get("acknowledgements", [])}
    m["acknowledgements"] = m.get("acknowledgements", []) + [a for a in new.get("acknowledgements", [])
                                                             if a["for"] not in fors]
    bands = {b: dict(v) for b, v in (m.get("bands") or {}).items()}
    for b, fields in (new.get("bands") or {}).items():
        tgt = bands.setdefault(b, {})
        for f, v in fields.items():
            tgt.setdefault(f, v)
    m["bands"] = bands
    return {k: v for k, v in m.items() if v not in ([], {}, "", None)}


def main():
    args = sys.argv[1:]
    dry = "--dry-run" in args; args = [a for a in args if a != "--dry-run"]
    subject = args[args.index("-m") + 1]; del args[args.index("-m"):args.index("-m") + 2]
    stage = P4 / "research/merged"; stage.mkdir(parents=True, exist_ok=True)
    for name in args:
        new = json.loads((P4 / "research/verified" / f"{name}.json").read_text())
        (stage / f"{name}.json").write_text(json.dumps(merge(existing(name), new), indent=2, ensure_ascii=False) + "\n")
        print("staged", name)
    if dry:
        return
    files = [str(stage / f"{n}.json") for n in args]
    subprocess.run(["rsync", "-a", "-e", " ".join(SSH[:-1])] + files + [f"cirrus:{REPO}/{CUR}/"], check=True)
    body = f"Curated by research per BRIEF-4 block A; checked with phase4/tools/verify_a.py (references resolve, links load, acknowledgement and licence text found verbatim on the provider's page, band fields carry evidence). Instruments: {', '.join(n.replace('__', '/') for n in args)}."
    script = f"""set -eu
cd {REPO}
exec 9>/tmp/sed-filter-info.lock; flock 9
bash filter_info/refresh.sh 2>&1 | tail -5
git add {' '.join(f'{CUR}/{n}.json' for n in args)} filter_info/references.json filter_info/curated/references.json
git -c "user.name=Niall Miller" -c user.email=niall.j.miller@gmail.com commit -q -m {json.dumps(subject)} -m {json.dumps(body)}
git log -1 --format='%h %an <%ae> %s'
echo "unresolved: $(wc -l < filter_info/unresolved.txt)"
git push -q origin HEAD:server && echo pushed
"""
    subprocess.run(SSH + ["bash", "-s"], input=script, text=True, check=True)


if __name__ == "__main__":
    main()
