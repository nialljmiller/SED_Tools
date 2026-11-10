"""Which of the current test ids are 0.4.1 or fixed-bug checks.

Sources: the trim's kept_because (TEST_TRIM_ids.tsv: every reason but 'lines'), and the declutter fix
commits (277f358..origin/declutter) that added or changed a test. Writes id, kind, why.
usage: checks_list.py ids.txt TEST_TRIM_ids.tsv declutter_touched.json > checks.tsv
"""
import csv, json, re, sys
ids = [l.strip() for l in open(sys.argv[1]) if "::" in l]
trim = {}
for row in csv.DictReader(open(sys.argv[2]), delimiter="\t"):
    if row["status"] != "removed":
        trim[row["now_id"]] = row["kept_because"]
# declutter renamed four trim ids
RENAMED = {
    "tests/test_builders_grabbers.py::test_probe_finds_a_contiguous_grid":
        "tests/test_builders_grabbers.py::test_probe_finds_a_contiguous_grid",
    "tests/test_cli_menus_and_options.py::test_composition_asks_again_after_a_bad_recipe":
        "tests/test_cli_menus_and_options.py::test_composition_asks_again_after_a_bad_recipe_or_detector_type",
    "tests/test_cube_build.py::test_other_lookup_table_file_is_not_used":
        "tests/test_cube_build.py::test_other_lookup_table_file_is_used_with_a_warning",
}
FEATURE = {"1e459b7b", "4d581f88", "16dfe178", "42a991d8"}  # detector edit feature, golden re-record, test plumbing
# d7308f81 and e2f8fc36 added cases to module-level tables of test_cli_mains.py
TABLE_CASES = {
    "tests/test_cli_mains.py::test_help_prints_the_usage_and_exits_0[svo_regen_spectra_lookup]": "d7308f81",
    "tests/test_cli_mains.py::test_a_bad_argument_prints_the_usage_and_exits_2[svo_regen_spectra_lookup]": "d7308f81",
    "tests/test_cli_mains.py::test_a_bad_argument_prints_the_usage_and_exits_2[svo_filter_grabber]": "e2f8fc36",
}
touched = json.load(open(sys.argv[3]))
fixes = {}
for sha, c in touched.items():
    if sha not in FEATURE:
        for t in c["tests"]:
            fixes.setdefault(t, []).append(f"{sha} {c['subject']}")
print("id\tkind\twhy")
for i in ids:
    func = i.split("[")[0]
    why = []
    base = trim.get(i) or trim.get(RENAMED.get(func, "")) or trim.get(func)
    if base is None:  # a parametrised id: its function's reason
        base = next((v for k, v in trim.items() if k.split("[")[0] == func and v != "lines"), None)
    if base and base != "lines":
        why.append(base)
    if i in TABLE_CASES:
        sha = TABLE_CASES[i]
        why.append(f"declutter fix {sha} {touched[sha]['subject'] if sha in touched else 'svo_regen_spectra_lookup: parse arguments, so --help changes nothing'}")
    for f in fixes.get(func, []):
        why.append(f"declutter fix {f}")
    kind = "always" if base and base.startswith("always") else ("0.4.1/bug" if why else "lines")
    print(f"{i}\t{kind}\t{'; '.join(why) or (base or 'new in declutter (detector edit feature)')}")
