"""Old import path; the code is in sed_tools/filters/library.py."""

from . import _old_path
from .filters import library as _module

# What `from sed_tools.filter_library import *` gave in 0.5.0.
__all__ = """
ATMOSPHERE_CHECK_AGE AtmosphereGrid AtmosphereSpec BUILDERS_FOR ComponentError Dict FETCH_ORDER FILTER
FilterLibrary FilterRow INSTRUMENT_CATALOGUE InstrumentManifest List NJMFilterGrabber NJM_UNAVAILABLE
OPTICS PUBLISHED Path QE RequestException SVOFilterBrowser SVOUnreachableError annotations
atmosphere_file build_product builder_for cache_root cached_property component_availability curve_file
dataclass find_atmosphere get_builder index_listing instrument_catalogue json load_instrument_catalogue
logger logging match_name migrate_filter_tree name_key os parse_components recipe_name registered_name
replace resolve_atmosphere svo_components_header svo_unavailable textwrap time
""".split()

_old_path.forward(__name__, _module)
