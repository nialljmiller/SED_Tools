"""Old import path; the code is in sed_tools/grids/svo.py."""

from . import _old_path
from .grids import svo as _module

# What `from sed_tools.svo_spectra_grabber import *` gave in 0.5.0.
__all__ = """
BeautifulSoup DISCOVERY_TIMEOUT HTTPAdapter Retry SVOSpectraGrabber ThreadPoolExecutor as_completed csv
discovery_failure io json logger logging mount_discovery os parse_header re requests urllib
""".split()

_old_path.forward(__name__, _module, HTTPAdapter="requests.adapters:HTTPAdapter", csv="csv")
