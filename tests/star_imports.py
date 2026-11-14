"""What ``from <module> import *`` gives for every module of sed_tools: names older code may rely on.

    python tests/star_imports.py               # record tests/fixtures/star_imports.json from this checkout
    python tests/star_imports.py --out FILE    # write this checkout's surface to FILE (test_api's row does)

Record only on the commit the surface is measured against (a release, or main), with every optional dependency
installed (``pip install -e '.[full,ml-tune]'``): recording refuses a module that cannot be imported.  Names
may be added freely; test_api's row ``every_module_star_imports_and_keeps_its_recorded_names`` fails on a
recorded name a module no longer gives, and on any module whose star import fails.
"""

from __future__ import annotations

import json
import pkgutil
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "star_imports.json"
sys.path.insert(0, str(HERE.parent))


def surface():
    """{module: the sorted names its star import gives, None when an optional dependency is not installed, or
    the error that stopped it}."""
    import sed_tools

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in sorted(n for _, n, _ in pkgutil.walk_packages(sed_tools.__path__, "sed_tools.")):
            if name.endswith("__main__"):
                continue
            try:
                namespace = {}
                exec(f"from {name} import *", namespace)
                out[name] = sorted(key for key in namespace if key != "__builtins__")
            except ModuleNotFoundError as exc:
                missing = (exc.name or "").split(".")[0]
                out[name] = None if missing and missing != "sed_tools" else f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 - any failure is what the row reports
                out[name] = f"{type(exc).__name__}: {exc}"
    return out


def write(path, found):
    """One module per line, names sorted, so a re-recording's diff shows what changed."""
    lines = [f"  {json.dumps(module)}: {json.dumps(names)}" for module, names in sorted(found.items())]
    Path(path).write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")


if __name__ == "__main__":
    found = surface()
    if sys.argv[1:2] == ["--out"]:
        write(sys.argv[2], found)
    else:
        bad = {module: names for module, names in found.items() if not isinstance(names, list)}
        if bad:
            sys.exit("not recorded; these modules do not import here (install every extra):\n  " + "\n  ".join(
                f"{module}: {names or 'an optional dependency is missing'}" for module, names in bad.items()))
        write(FIXTURE, found)
        print(f"recorded {FIXTURE.relative_to(HERE.parent)}: {len(found)} modules")
