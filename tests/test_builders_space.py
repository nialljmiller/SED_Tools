"""The space missions, each built end to end from the archive's saved files (tests/fixtures/builders)."""

from __future__ import annotations

import numpy as np
import pytest

from builder_fixtures import Session, check_published, curve, saved
from sed_tools.component_sources import space
from sed_tools.filter_components import InstrumentManifest

# instrument: facility, bands, published components, detector type, component roles, peak-normalised?,
# and the first band's samples, first and last wavelength (A), peak wavelength (A) and peak
EXPECTED = {
    "DR2": ("GAIA", "BP G RP", "filter+optics+qe", "photon", "", False, (16, 3265.0, 7240.0, 4855.0, 0.659394605)),
    "DR2r": ("GAIA", "BP G RP", "filter+optics+qe", "photon", "", False, (16, 3265.0, 7240.0, 4855.0, 0.673483243)),
    "PACS": ("Herschel", "100 160 70", "filter", "energy", "filter", False,
             (49, 25003.0994, 5.0e6, 977994.995, 0.58252603)),
    "SPIRE": ("Herschel", "250 350 500", "filter+optics+qe", "energy", "", True,
              (52, 1666666.667, 1.0e7, 2666666.667, 1.0)),
    "Hipparcos": ("Hipparcos", "Hp", "filter+optics+qe", "photon", "", False, (5, 3900.0, 8900.0, 4100.0, 0.423)),
    "Tycho": ("Hipparcos", "BT VT", "filter+optics+qe", "photon", "", False, (3, 3750.0, 3850.0, 3850.0, 0.556)),
    "IRAS": ("IRAS", "100 12 25 60", "filter+optics+qe", "energy", "filter+optics qe", True,
             (6, 650000.0, 1400000.0, 1100000.0, 1.0)),
    "MSX": ("MSX", "A C D E", "filter+optics+qe", "energy", "", True, (38, 58000.0, 113600.0, 99100.0, 1.0)),
    "HFI": ("Planck", "100 143 217 353 545 857", "filter+optics+qe", "energy", "", True,
            (48, 166666.667, 34676769.2, 33313077.4, 1.0)),
    "LFI": ("Planck", "30 44 70", "filter+optics+qe", "energy", "", True,
            (53, 74948114.5, 140747632.9, 109413305.8, 1.0)),
    "IRAC": ("Spitzer", "I1 I2 I3 I4", "filter+optics+qe", "photon", "", False,
             (38, 30810.6, 40103.8, 37439.9, 0.472956)),
    "MIPS": ("Spitzer", "160 24 70", "filter+qe", "energy", "", True, (6, 1000851.0, 1164018.0, 1164018.0, 1.0)),
    "IRS": ("Spitzer", "PU_blue PU_red", "filter+qe", "photon", "", False, (38, 100104.0, 350363.0, 147312.0, 2.96)),
    "OM": ("XMM", "B U UVM2 UVW1 UVW2 V", "filter+optics+qe", "photon", "", True, (79, 1000.0, 7000.0, 3980.0, 1.0)),
}


def test_every_recipe_is_covered():
    assert sorted(EXPECTED) == sorted(space.RECIPES)


@pytest.mark.parametrize("instrument", sorted(space.RECIPES))
def test_builds_from_the_archive_files(tmp_path, instrument):
    facility, bands, components, detector, roles, normalised, (n, first, last, at, peak) = EXPECTED[instrument]
    session = Session()
    inst_dir = space.build(tmp_path / "filters", instrument=instrument, session=session)
    manifest = check_published(inst_dir, facility, instrument, bands.split(), components, detector)
    assert manifest.ground_based is False and manifest.outside_range == "zero" and manifest.default_site is None
    assert sorted(manifest.components) == roles.split()
    wl, tr = curve(inst_dir / f"{bands.split()[0]}.dat")
    assert (len(wl), wl[0], wl[-1], wl[np.argmax(tr)]) == (n, pytest.approx(first), pytest.approx(last),
                                                          pytest.approx(at))
    assert tr.max() == pytest.approx(peak)
    if normalised:
        assert all(curve(inst_dir / f"{b}.dat")[1].max() == 1.0 for b in bands.split())
    assert len(session.calls) == len(set(session.calls))


def test_gaia_drops_the_undefined_sentinel(tmp_path):
    for inst in ("DR2", "DR2r"):
        directory = space.build(tmp_path, instrument=inst, session=Session())
        for band in ("G", "BP", "RP"):
            assert curve(directory / f"{band}.dat")[1].max() < 1.0  # 99.99 = undefined is never a value
        assert "99.99 = undefined" in InstrumentManifest.load(directory).notes
    assert (tmp_path / "GAIA" / "DR2" / "G.dat").read_text() != (tmp_path / "GAIA" / "DR2r" / "G.dat").read_text()


def test_iras_components_are_the_table_columns(tmp_path):
    inst = space.build(tmp_path, instrument="IRAS", session=Session())
    manifest = InstrumentManifest.load(inst)
    for band in ("12", "25", "60", "100"):
        wl, system = curve(inst / f"{band}.dat")
        _, optics = curve(inst / manifest.component_file("filter+optics", band))
        _, qe = curve(inst / manifest.component_file("qe", band))
        assert len(optics) == len(qe) == len(wl) and qe.max() == 1.0 and system.max() == 1.0
        assert 0 < optics.max() <= 1.0


def test_planck_hfi_drops_the_far_tail(tmp_path):
    inst = space.build(tmp_path, instrument="HFI", session=Session())
    for band in ("100", "857"):
        assert curve(inst / f"{band}.dat")[0].max() <= 1e8  # wavenumber >= 1 /cm


@pytest.mark.parametrize("instrument, corrupt, message", [
    ("PACS", lambda f: f.__setitem__(space.PACS.format("green"), f[space.PACS.format("blue")]),
     "PACS: FITS BAND 'BS', expected 'BL'"),
    ("IRAS", lambda f: f.__setitem__(space.IRAS_URL, f[space.IRAS_URL].replace(b"100 &nbsp;&micro;m Band", b"")),
     "IRAS: unexpected bands"),
])
def test_a_wrong_file_stops_the_build_before_anything_is_written(tmp_path, instrument, corrupt, message):
    files = saved()
    corrupt(files)
    with pytest.raises(ValueError, match=message):
        space.build(tmp_path, instrument=instrument, session=Session(files))
    assert not any(tmp_path.iterdir())
