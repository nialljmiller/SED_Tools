import gzip, json, sys
def load(p): return json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
a = load(sys.argv[1]); b = load(sys.argv[2])
A, B = set(a["all"]), set(b["all"])
def tu(d): return set().union(*(set(v) for c, v in d["per"].items() if c))
TA, TB = tu(a), tu(b)
EA = set(a["per"].get("", []))
# Hit or missed from run to run by the optuna test (its sampler is unseeded; CI skips it).
FLAKY = {"sed_tools/ml/common.py:261", "sed_tools/ml/common.py:266"}
for X in (A, B, TA, TB): X -= FLAKY
lost_all, gained_all = A - B, B - A
lost_t, gained_t = TA - TB, TB - TA
print("excluded as nondeterministic:", sorted(FLAKY))
print(f"all lines: before {len(A)} after {len(B)} lost {len(lost_all)} gained {len(gained_all)}")
print(f"test-context lines: before {len(TA)} after {len(TB)} lost {len(lost_t)} gained {len(gained_t)}; gained within baseline collection-only lines: {gained_t <= EA}")
for l in sorted(lost_all)[:50]: print("  LOST", l)
for l in sorted(lost_t)[:50]: print("  LOST-from-tests", l)
for l in sorted(gained_t - EA)[:50]: print("  GAINED-not-in-collection", l)
ok = not lost_all and not gained_all and not lost_t and gained_t <= EA
print("PROOF", "OK" if ok else "FAILED")
CI_SKIPPED = ("tests/test_golden.py::test_case_reproduces_recorded_outputs",
              "tests/test_ml_engines_device_and_auto.py::test_train_complete_and_generate_run_on_mps",
              "tests/test_ml_engines_device_and_auto.py::test_auto_train_generator_saves_a_model_sedgenerator_and_the_api_load",
              "tests/test_ml_engines_device_and_auto.py::test_auto_trainer_stops_on_a_trial_with_no_usable_epoch",
              "tests/test_cli_optional_packages.py::test_ml_generator_auto_without_torch_says_what_to_install")
def ci(d): return set().union(*(set(v) for c, v in d["per"].items() if c and not c.split("|")[0].split("[")[0] in CI_SKIPPED)) - FLAKY
CA, CB = ci(a), ci(b)
print(f"lines run by tests that also run on Linux CI: before {len(CA)} after {len(CB)} lost {len(CA - CB)}; gained within baseline collection-only lines: {(CB - CA) <= EA}")
for l in sorted(CA - CB)[:30]: print("  LOST-on-CI", l)
