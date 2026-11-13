"""Old import path; the code is in sed_tools/filters/svo.py."""

from . import _old_path
from .filters import svo as _module

# What `from sed_tools.svo_filter_grabber import *` gave in 0.5.0.
__all__ = """
BeautifulSoup DEFAULT_BASE_DIR Dict ENERGY Facility FilterRow Instrument InstrumentManifest Iterable
List Mapping Optional Path SOURCE_ASSUMED SOURCE_SVO SVOFilterBrowser SVOMetadataClient
SVOUnreachableError SVO_ALTERNATIVES SVO_PAUSE SVO_READ_TIMEOUT Sequence Tuple annotations dataclass
html monotonic os parse_multi_selection prompt_choice re requests
svo_components_header svo_fps_url svo_get svo_unavailable sys textwrap threading urllib
write_filter_file
""".split()

_old_path.forward(
    __name__, _module, parse_multi_selection="sed_tools.ui_utils:parse_multi_selection",
    prompt_choice="sed_tools.ui_utils:prompt_choice", sys="sys", textwrap="textwrap")
