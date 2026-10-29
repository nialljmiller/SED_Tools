"""Old import path; the code is in sed_tools/filters/curve_metadata.py."""

from . import _old_path
from .filters import curve_metadata as _module

# What `from sed_tools.filter_metadata import *` gave in 0.5.0.
__all__ = """
COLUMN_ROW Callable DETECTOR_KEYS DETECTOR_SOURCE_PRECEDENCE DETECTOR_TYPES DetectorTypeResult Dict
ENERGY HEADER_KEYS Iterable List Mapping Optional PHOTON Path PathLike SOURCE_ASSUMED SOURCE_INFERRED
SOURCE_ORIGINAL SOURCE_SVO SOURCE_SVO_MATCH SVOFilterRecord SVOMetadataClient SVOUnreachableError
SVO_ALTERNATIVES SVO_CONNECT_TIMEOUT SVO_FPS_URL SVO_FPS_URL_ENV Sequence Tuple Union annotations
copy_filter_file dataclass detector_source_may_replace field format_metadata_field has_column_row io
iter_filter_files logger logging migrate_filter_file migrate_filter_tree normalize_detector_type np os
re read_filter_header released_mesa_reads requests resolve_detector_type shutil svo_components_header
svo_fps_url svo_get tag_filter_file tag_filter_tree unquote write_filter_file write_filter_header
""".split()

_old_path.forward(__name__, _module)
