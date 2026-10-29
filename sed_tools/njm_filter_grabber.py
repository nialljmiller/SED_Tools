"""Old import path; the code is in sed_tools/filters/njm.py."""

from . import _old_path
from .filters import njm as _module

# What `from sed_tools.njm_filter_grabber import *` gave in 0.5.0.
__all__ = """
BLOCKED_STATUSES BeautifulSoup Dict InstrumentManifest List NJMFilterGrabber NJM_CONNECT_TIMEOUT
NJM_INDEX_MAX_AGE NJM_PAUSE Optional Path RETRY_STATUSES SOURCE_ASSUMED SVOMetadataClient Set
SetDownload Tuple VERIFIED_CURVES curve_file dataclass field json logger logging main
migrate_filter_file monotonic os plain_reason re requests tag_filter_file threading time urllib3
""".split()

if __name__ == "__main__":
    import sys
    sys.exit(_module.main())
else:
    _old_path.forward(__name__, _module, threading="threading", urllib3="urllib3")
