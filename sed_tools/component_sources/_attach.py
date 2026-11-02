"""Old import path; the code is in sed_tools/component_sources/_common.py."""

from .. import _old_path
from . import _common as _module

# What `from sed_tools.component_sources._attach import *` gave in 0.5.0.
__all__ = """
EARLIER InstrumentManifest Iterable Path add_component annotations common_product fresh_manifest np
published_bands register with_earlier_note write_unregistered
""".split()

_old_path.forward(__name__, _module)
