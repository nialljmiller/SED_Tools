"""Old import path and plain script; the code is in sed_tools/cube/tool.py."""

if __package__:
    from . import _old_path
    from .cube import tool as _module
else:  # run as a plain script: python sed_tools/flux_cube_tool.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sed_tools.cube import tool as _module

if __name__ == "__main__":
    raise SystemExit(_module.main())
else:
    _old_path.forward(__name__, _module)
