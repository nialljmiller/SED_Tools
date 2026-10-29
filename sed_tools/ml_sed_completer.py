"""Old import path; the code is in sed_tools/ml/completer.py."""

from . import _old_path
from .ml import completer as _module

# What `from sed_tools.ml_sed_completer import *` gave in 0.5.0.
__all__ = """
Any BOLTZMANN_K Counter DEFAULT_WL_MAX DEFAULT_WL_MIN DEFAULT_WL_POINTS Dict List MIN_WINDOW_COVERAGE
Optional PLANCK_H Path SEDCompleter SEDCompleterNetwork SIGMA SPEED_C Tuple Union cli_complete cli_list
cli_train extend_catalog json load_model logger logging main np os planck_function refused_without
run_interactive_workflow struct train_model
""".split()

if __name__ == "__main__":
    _module.main()
else:
    _old_path.forward(__name__, _module, Counter="collections:Counter", struct="struct")
