"""Old import path; the code is in sed_tools/grids/njm.py."""

from . import _old_path
from .grids import njm as _module

# What `from sed_tools.njm_spectra_grabber import *` gave in 0.5.0.
__all__ = """
BeautifulSoup Dict List NJMFilterGrabber NJMSpectraGrabber NJM_CONNECT_TIMEOUT Optional
ThreadPoolExecutor Tuple as_completed csv logger logging np os plain_reason requests tqdm urljoin
urllib3
""".split()

_old_path.forward(__name__, _module, urllib3="urllib3")
