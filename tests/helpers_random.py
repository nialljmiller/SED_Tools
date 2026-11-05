"""keep_random_state, an autouse fixture wherever it is imported."""

import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def keep_random_state():
    """Leave the random, numpy and torch (CPU and MPS) generators as the test found them, so later tests draw the
    same numbers as without these files. The other new test files import this fixture."""
    try:
        import torch
    except ImportError:
        torch = None
    mps = torch is not None and torch.backends.mps.is_available()
    saved = random.getstate(), np.random.get_state(), torch and torch.get_rng_state(), mps and torch.mps.get_rng_state()
    yield
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    if torch is not None:
        torch.set_rng_state(saved[2])
    if mps:
        torch.mps.set_rng_state(saved[3])
