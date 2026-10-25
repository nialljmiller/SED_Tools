#!/usr/bin/env python3
"""Run filter-expansion recipes: build.py RECIPE [RECIPE ...]  (names in tools/recipes/)."""
import importlib
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "recipes"))

failed = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(Path(name).stem).run()
        print(f"built {name}")
    except Exception:  # noqa: BLE001
        failed.append(name)
        print(f"FAILED {name}")
        traceback.print_exc()
sys.exit(1 if failed else 0)
