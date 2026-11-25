"""Throwaway diagnostic: run every golden case and print, per file, how it compares with the recording."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_golden as g  # noqa: E402
from record import environment, run  # noqa: E402

g.RTOL = 1e-16
print("environment", environment(), "recorded", g.RECORDED)
for name in sorted(g.CASES):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            out = run(name, Path(tmp))
        except Exception as exc:
            print(f"{name}: ERROR {exc!r}")
            continue
        expected_dir = g.EXPECTED / name
        exp, act = g._files(expected_dir), g._files(out)
        if exp != act:
            print(f"{name}: FILE LIST differs: missing {sorted(set(exp)-set(act))} extra {sorted(set(act)-set(exp))}")
        for p in exp:
            if not (out / p).exists():
                continue
            if (expected_dir / p).read_bytes() == (out / p).read_bytes():
                print(f"{name}/{p}: SAME")
                continue
            try:
                units, what = g._difference(expected_dir / p, out / p)
            except Exception as exc:
                units, what = float("nan"), repr(exc)
            print(f"{name}/{p}: DIFF rel~{units * 1e-16:.3e} ({what})")
            if units == float("inf"):
                import difflib
                a = (expected_dir / p).read_text(errors="replace").splitlines()
                b = (out / p).read_text(errors="replace").splitlines()
                for line in list(difflib.unified_diff(a, b, lineterm="", n=0))[:12]:
                    print("    " + line[:200])
