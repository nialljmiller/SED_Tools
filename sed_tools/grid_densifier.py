"""Old import path; the code is in sed_tools/cube/densify.py."""

from . import _old_path
from .cube import densify as _module

# What `from sed_tools.grid_densifier import *` gave in 0.5.0.
__all__ = """
Optional annotations argparse densify_grid main np os resample_to_grid run_interactive_workflow struct
time
""".split()

if __name__ == "__main__":
    _module.main()
else:
    _old_path.forward(__name__, _module)
