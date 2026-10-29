"""Old import path; the code is in sed_tools/grids/msg.py."""

from . import _old_path
from .grids import msg as _module

# What `from sed_tools.msg_spectra_grabber import *` gave in 0.5.0.
__all__ = """
AXIS_ROLE_OVERRIDES BeautifulSoup DISCOVERY_TIMEOUT HTTPAdapter MSGSpectraGrabber Retry
ThreadPoolExecutor as_completed csv discovery_failure h5py logger logging main mount_discovery np os re
requests tqdm urljoin
""".split()

if __name__ == "__main__":
    _module.main()
else:
    _old_path.forward(__name__, _module, HTTPAdapter="requests.adapters:HTTPAdapter", csv="csv")
