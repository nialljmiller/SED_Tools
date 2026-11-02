"""Old import path; the code is in sed_tools/filters/njm_catalogue.py."""

from . import _old_path
from .filters import njm_catalogue as _module

# What `from sed_tools.filter_catalogue import *` gave in 0.5.0.
__all__ = """
Any Callable CatalogueError Dict FilterCatalogue INDEX_MAX_AGE INFO Iterable Iterator KEY_COLUMNS
LEADING_COLUMNS List MAX_CONCURRENT Mapping Optional PROPERTIES Path Range RequestException Sequence
ThreadPoolExecutor Tuple Union VEGA_URL annotations cache_root cached_property collect_citations
format_bibtex format_citations formatdate json logger logging normalize_detector_type os parse_target pd
quote re read_filter_header sys threading time
""".split()

_old_path.forward(__name__, _module, os="os", re="re")
