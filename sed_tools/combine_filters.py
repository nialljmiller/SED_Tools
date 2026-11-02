"""Old import path; the code is in sed_tools/filters/combine.py."""

from . import _old_path
from .filters import combine as _module

# What `from sed_tools.combine_filters import *` gave in 0.5.0.
__all__ = """
COMPONENTS_DIR FILTER_DIR_DEFAULT List Optional Path PathLike Sequence Union annotations archived_bands
argparse combine_filter_sets copy_filter_file find_filter_sets index_listing is_auxiliary_path logger
logging main migrate_filter_file os resolve_filter_sources resolve_output_path unique_filter_target
write_filter_index
""".split()

if __name__ == "__main__":
    raise SystemExit(_module.main())
else:
    _old_path.forward(__name__, _module)
