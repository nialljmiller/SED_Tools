"""resample_to_grid keeps its teff argument for the interface; it does not change the result.
"""

import numpy as np

from sed_tools._resample import resample_to_grid


def test_teff_argument_does_not_affect_output():
    # Documented in the docstring: teff is retained for interface
    # stability but not used to extrapolate. Confirm that holds.
    wl_src = np.linspace(4000.0, 8000.0, 20)
    fl_src = np.linspace(1.0, 2.0, 20)
    wl_tgt = np.linspace(4500.0, 7500.0, 15)

    low_t = resample_to_grid(wl_src, fl_src, teff=3000.0, wl_tgt=wl_tgt)
    high_t = resample_to_grid(wl_src, fl_src, teff=50000.0, wl_tgt=wl_tgt)

    np.testing.assert_array_equal(low_t, high_t)
