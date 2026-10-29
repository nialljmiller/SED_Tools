"""Old import path; the code is in sed_tools/ml/tuning.py."""

from . import _old_path
from .ml import tuning as _module

# What `from sed_tools.ml_optimized import *` gave in 0.5.0.
__all__ = """
AutoTrainer BATCH_SIZE MAX_SAMPLES SEDGenerator SEDGeneratorNetwork TRIAL_EPOCHS data_loaders fit optuna
""".split()

_old_path.forward(__name__, _module)
