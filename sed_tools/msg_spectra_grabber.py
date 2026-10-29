"""Old import path; the code is in sed_tools/grids/msg.py."""

from . import _old_path
from .grids import msg as _module

if __name__ == "__main__":
    _module.main()
else:
    _old_path.forward(__name__, _module)
