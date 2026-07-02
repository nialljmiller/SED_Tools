from pathlib import Path

import h5py
import numpy as np
import pytest

from sed_tools.parsing import parse_multi_selection, parse_numeric_range
from sed_tools.spectrum_io import build_h5_bundle, read_text_spectrum


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3500,8000", (3500.0, 8000.0)),
        ("5:3", (3.0, 5.0)),
        ("-1.0,0.5", (-1.0, 0.5)),
        ("1e3 2e3", (1000.0, 2000.0)),
        ("", None),
        ("bad", None),
    ],
)
def test_parse_numeric_range(raw, expected):
    assert parse_numeric_range(raw) == expected


def test_selection_compatibility_aliases():
    from sed_tools.cli import _parse_multi_selection
    from sed_tools.svo_filter_grabber import parse_multi_selection as svo_parser

    expected = parse_multi_selection("1,3-5,5", 6)
    assert expected == [0, 2, 3, 4]
    assert _parse_multi_selection("1,3-5,5", 6) == expected
    assert svo_parser("1,3-5,5", 6) == expected


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


def test_h5_bundle_schema_and_metadata(tmp_path):
    spectrum = tmp_path / "star.txt"
    spectrum.write_text("# Teff = 5000 K\n# logg = 4.5\n# [M/H] = -0.5\n1 2\n")
    output = tmp_path / "nested" / "bundle.h5"

    assert build_h5_bundle(tmp_path, output)
    with h5py.File(output) as handle:
        item = handle["spectra/star.txt"]
        np.testing.assert_array_equal(item["lambda"][:], [1.0])
        assert item.attrs["teff"] == 5000.0
        assert item.attrs["feh"] == -0.5


def test_h5_bundle_with_no_spectra_is_not_written(tmp_path):
    output = tmp_path / "bundle.h5"
    assert not build_h5_bundle(tmp_path, output)
    assert not output.exists()
