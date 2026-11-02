"""Old import path; the code is in sed_tools/ml/common.py."""

from . import _old_path
from .ml import common as _module

# What `from sed_tools.ml_utils import *` gave in 0.5.0.
__all__ = """
DataLoader Dict List TensorDataset Tuple data_loaders fit get_device masked_mse np torch
""".split()

_old_path.forward(
    __name__, _module,
    DataLoader="torch.utils.data:DataLoader", TensorDataset="torch.utils.data:TensorDataset", torch="torch")
