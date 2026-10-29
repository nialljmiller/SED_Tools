"""Old import path; the code is in sed_tools/filters/njm.py."""

from . import _old_path
from .filters import njm as _module

if __name__ == "__main__":
    import sys
    sys.exit(_module.main())
else:
    _old_path.forward(__name__, _module)
