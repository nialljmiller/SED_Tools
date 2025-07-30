# stellar_colors/utils/integration.py

import numpy as np
from scipy.integrate import quad

def adaptive_integration(func, a, b, **kwargs):
    result, _ = quad(func, a, b, **kwargs)
    return result
