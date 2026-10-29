"""Old import path; the code is in sed_tools/filters/svo.py."""

from . import _old_path
from .filters import svo as _module

if __name__ == "__main__":
    raise SystemExit(_module.main())
else:
    _old_path.forward(__name__, _module)
