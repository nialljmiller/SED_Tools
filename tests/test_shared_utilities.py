from pathlib import Path

import h5py
import numpy as np
import pytest

from sed_tools.parsing import parse_numeric_range
from sed_tools.spectrum_io import build_h5_bundle, read_text_spectrum


def test_text_spectrum_compatibility_aliases(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("# header\n1, 2\ninvalid\n3 4 extra\ninf 5\n")
    expected = read_text_spectrum(path)
    np.testing.assert_array_equal(expected[0], [1.0, 3.0])
    np.testing.assert_array_equal(expected[1], [2.0, 4.0])

    from sed_tools.cli import load_txt_spectrum
    from sed_tools.precompute_flux_cube import load_sed
    from sed_tools.spectra_cleaner import read_spectrum_data

    for loader in (load_txt_spectrum, load_sed, read_spectrum_data):
        actual = loader(str(path))
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


