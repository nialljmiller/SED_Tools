"""Old import path; the code is in sed_tools/cube/build.py."""

from . import _old_path
from .cube import build as _module

if __name__ == "__main__":
    _module._main()
else:
    _old_path.forward(__name__, _module)
