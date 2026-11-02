"""Record the reference outputs: ``python tests/golden/record.py [case ...]``.

Only the manager runs this, on the commit the cleanup is measured against.
It runs every case twice and refuses to record a case whose two runs differ.
"""

from __future__ import annotations

import filecmp
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import sed_tools  # noqa: E402
from cases import CASES  # noqa: E402

EXPECTED = HERE / "expected"


def environment():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "system": platform.system(), "machine": platform.machine()}


def run(name, root):
    work, out = root / "work", root / "out"
    work.mkdir(parents=True)
    out.mkdir(parents=True)
    CASES[name](work, out)
    return out


def same_tree(a, b):
    left = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    right = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    return left == right and all(filecmp.cmp(a / p, b / p, shallow=False) for p in left)


def main(names):
    if Path(sed_tools.__file__).resolve().parent != HERE.parent.parent.resolve() / "sed_tools":
        sys.exit(f"sed_tools imported from {sed_tools.__file__}, not from this checkout")
    for name in names or sorted(CASES):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = run(name, Path(first)), run(name, Path(second))
            if not same_tree(a, b):
                sys.exit(f"{name}: two runs differ; make the case deterministic before recording")
            target = EXPECTED / name
            if target.exists():
                sys.exit(f"{name}: {target} exists; move it aside yourself before re-recording")
            shutil.copytree(a, target)
            print(f"recorded {name}: {sum(1 for p in target.rglob('*') if p.is_file())} files")
    (EXPECTED / "environment.json").write_text(json.dumps(environment(), indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
