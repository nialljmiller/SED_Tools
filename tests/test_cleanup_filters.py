"""Filter handlers that now raise or warn instead of carrying on silently (cleanup of the filters area).

Every network call is mocked; a real connection fails the test.
"""

import logging

import pytest

from sed_tools import filter_metadata as fm


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.delenv(fm.SVO_FPS_URL_ENV, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("test attempted a real network connection")

    monkeypatch.setattr("socket.socket.connect", refuse)


def _curve(path, header="Wavelength,Transmission,WavelengthUnit=Angstrom"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n4000.0,0.1\n5000.0,1.0\n6000.0,0.1\n")
    return path


# ----------------------------------------------------------------------
# filter_colors
# ----------------------------------------------------------------------
def test_instrument_colors_refuses_an_unreadable_curve(tmp_path):
    from sed_tools.filter_colors import instrument_colors

    good = _curve(tmp_path / "g.dat")
    empty = tmp_path / "r.dat"
    empty.write_text("Wavelength,Transmission\n")
    with pytest.raises(ValueError, match="r.dat"):
        instrument_colors([good, empty])
