"""Check one area's new test against the old suite's baseline, line for line and id for id.

usage: area_check.py AREA NEW.json [INVENTORY.tsv]
  AREA           a key of areas.json
  NEW.json       lines.py output of a run_cov.sh run of the new area test (alone, or with the whole suite)
  INVENTORY.tsv  old_id <TAB> new_test <TAB> row <TAB> note  (one line per old id and row; header optional)

Fails (exit 1) when
  - a line the old area tests ran is not run by the new area test (outside the nondeterministic lines);
  - a line the old area tests ran inside a test now runs only at collection (the proof counts it lost);
  - with an inventory: an old id of the area is missing, or names a row that did not run.
Reports, without failing, each old id whose own lines its rows do not run (the area may run them elsewhere).
"""
import collections, csv, gzip, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "baseline" / "lines.json.gz"
FLAKY = {"sed_tools/ml/common.py:261", "sed_tools/ml/common.py:266"}  # optuna's unseeded sampler


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def main(area, new_path, inventory=None):
    spec = json.load(open(HERE / "areas.json"))[area]
    files = {f"tests/{name}" for name in spec["old"]}
    old, new = load(BASELINE), load(new_path)
    per_old = collections.defaultdict(set)
    for ctx, lines in old["per"].items():
        test_id = ctx.split("|")[0]
        if ctx and test_id.split("::")[0] in files:
            per_old[test_id].update(lines)
    collection_old = set(old["per"].get("", []))
    old_lines = set().union(*per_old.values()) - FLAKY
    test = spec["test"]
    in_test = set().union(*(set(v) for c, v in new["per"].items() if c.split("|")[0] == test))
    rows = collections.defaultdict(set)
    for c, v in new["per"].items():
        if c.startswith(test + "|row="):
            rows[c[len(test) + 5:]].update(v)
    lost = old_lines - set(new["all"])
    moved = (old_lines & set(new["per"].get("", []))) - collection_old
    print(f"{area}: {len(per_old)} old ids ran {len(old_lines)} lines; the new test runs {len(in_test)} "
          f"({len(rows)} rows); lost {len(lost)}; moved to collection {len(moved)}")
    for line in sorted(lost)[:40]:
        print("  LOST", line)
    for line in sorted(moved)[:40]:
        print("  MOVED-TO-COLLECTION", line)
    bad = bool(lost or moved)
    if inventory:
        mapped = collections.defaultdict(set)
        for rec in csv.reader(open(inventory), delimiter="\t"):
            if rec and rec[0] != "old_id" and rec[0].split("::")[0] in files:
                mapped[rec[0]].add(rec[2])
        ids = HERE.parent / "baseline" / "ids.txt"
        lines = (open(ids) if ids.exists() else gzip.open(f"{ids}.gz", "rt")).read().split()
        old_ids = {l for l in lines if l.split("::")[0] in files}
        missing = sorted(old_ids - set(mapped))
        not_run = sorted({r for rs in mapped.values() for r in rs} - set(rows))
        print(f"  inventory: {len(mapped)} of {len(old_ids)} old ids mapped; {len(missing)} missing; "
              f"{len(not_run)} rows named but not run")
        for i in missing[:40]:
            print("  UNMAPPED", i)
        for r in not_run[:40]:
            print("  ROW-NOT-RUN", r)
        bad |= bool(missing or not_run)
        for old_id in sorted(mapped):
            short = (per_old.get(old_id, set()) - FLAKY) - set().union(*(rows[r] for r in mapped[old_id]))
            if short:
                print(f"  note: {old_id}: {len(short)} of its lines run outside its rows, e.g. {sorted(short)[:3]}")
    print("AREA", area, "FAILED" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
