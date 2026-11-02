"""Old import path; the code is in sed_tools/ml/generator.py."""

from . import _old_path
from .ml import generator as _module

# What `from sed_tools.ml_sed_generator import *` gave in 0.5.0.
__all__ = """
Any Counter DEFAULT_WL_MAX DEFAULT_WL_MIN DEFAULT_WL_POINTS Dict List Optional Path SEDGenerator
SEDGeneratorNetwork Tuple Union json logger logging main np os refused_without run_interactive_workflow
struct
""".split()

if __name__ == "__main__":
    _module.main()
else:
    _old_path.forward(__name__, _module, Counter="collections:Counter", struct="struct")
