"""Old import path; the code is in sed_tools/filters/svo.py."""

from . import _old_path
from .filters import svo as _module

# What `from sed_tools.svo_filter_grabber import *` gave in 0.5.0.
__all__ = """
BeautifulSoup DEFAULT_BASE_DIR Dict ENERGY Facility FilterRow Instrument InstrumentManifest Iterable
List Mapping Optional Path SOURCE_ASSUMED SOURCE_SVO SVOFilterBrowser SVOMetadataClient
SVOUnreachableError SVO_ALTERNATIVES SVO_PAUSE SVO_READ_TIMEOUT Sequence Tuple annotations dataclass
html main monotonic os parse_multi_selection prompt_choice re requests run_interactive
svo_components_header svo_fps_url svo_get svo_unavailable sys textwrap threading urllib
write_filter_file
""".split()

if __name__ == "__main__":
    raise SystemExit(_module.main())
else:
    _old_path.forward(__name__, _module)
