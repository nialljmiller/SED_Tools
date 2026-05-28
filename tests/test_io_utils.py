import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import pytest

from sed_tools.io_utils import build_h5_bundle_from_txt, load_txt_spectrum


def test_load_txt_spectrum_rejects_malformed_numeric_data(tmp_path):
    spectrum = tmp_path / "bad.txt"
    spectrum.write_text("# comment\n4000 1.0\nnot-a-number 2.0\n")

    with pytest.raises(ValueError):
        load_txt_spectrum(spectrum)


def test_build_h5_bundle_propagates_spectrum_parse_errors(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "bad.txt").write_text("4000 1.0\n5000 not-a-number\n")

    with pytest.raises(ValueError):
        build_h5_bundle_from_txt(model_dir, model_dir / "model.h5")


def test_build_h5_bundle_writes_valid_spectra(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "good.txt").write_text(
        "# teff = 5777\n# logg = 4.44\n# metallicity = 0.0\n4000 1.0\n5000 2.0\n"
    )
    out_h5 = model_dir / "model.h5"

    build_h5_bundle_from_txt(model_dir, out_h5)

    with h5py.File(out_h5, "r") as h5:
        group = h5["spectra/good.txt"]
        assert np.allclose(group["lambda"][:], [4000.0, 5000.0])
        assert np.allclose(group["flux"][:], [1.0, 2.0])
        assert group.attrs["teff"] == 5777.0
