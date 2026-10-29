"""Old import path; the code is in sed_tools/cube/collision.py."""

from . import _old_path
from .cube import collision as _module

# What `from sed_tools.collision_config import *` gave in 0.5.0.
__all__ = """
Any CollisionConfig Dict List MESA_AXES Optional Path Set Tuple VALID_STRATEGIES annotations
copy_global_config_to_model discover_extra_axes load_config load_defaults_config load_toml
match_filter_value merge_config re write_default_config
""".split()

_old_path.forward(__name__, _module)
