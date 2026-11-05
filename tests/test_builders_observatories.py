"""The observatory imagers, each built end to end from its team's saved files (tests/fixtures/builders)."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from builder_fixtures import Session, check_published, curve, saved
from sed_tools.component_sources import observatories as obs
from sed_tools.filter_components import InstrumentManifest
from sed_tools.filter_metadata import read_filter_header, released_mesa_reads

# instrument: facility, bands, published components, component roles, a phrase of the notes,
# and the first band's samples, first and last wavelength (A), peak wavelength (A) and peak
EXPECTED = {
    "FORS1": ("Paranal", "B_BESS I_BESS R_BESS U_BESS V_BESS g_GUNN r_GUNN u_GUNN v_GUNN", "filter", "filter",
              "z_GUNN and H_Alpha held", (30, 3300.0, 10800.0, 4200.0, 0.68712199)),
    "FORS2": ("Paranal", "I_BESS R_SPECIAL b_HIGH g_HIGH u_HIGH v_HIGH", "filter", "filter",
              "current FORS2 standard configuration", (34, 5900.2, 10290.6, 7812.6, 0.954332)),
    "OmegaCAM": ("Paranal", "B_JOHN V_JOHN g_SDSS i_SDSS r_SDSS u_SDSS z_SDSS", "filter+qe", "filter",
                 "mean of their 4 segment-centre scans", (21, 2996.45, 12008.38, 4799.16, 0.70318575)),
    "NIRC2": ("Keck", "H J K Kp Ks Lp Ms", "filter", "filter", "filter_passbands.xls (micron, percent -> fraction)",
              (34, 13968.46, 19140.01, 15198.75, 0.8981323)),
    "LRIS": ("Keck", "B I R Rs V g", "filter", "filter", "I is the vendor test report I.xls",
             (34, 3267.93, 8249.45, 4299.77, 0.7164691)),
    "MOSFIRE": ("Keck", "H H1 H2 J J2 J3 K Ks Y", "filter", "filter", "J2, J3, H1 and H2 are the intermediate bands",
                (33, 13575.0, 19075.0, 17050.0, 0.993073)),
    "DEIMOS": ("Keck", "B I R Z", "filter+optics+qe", "", "column 2 (THROUGHPUT) of the synphot tables",
               (33, 3500.0, 10900.0, 4499.97, 0.397678)),
    "Mosaic": ("KPNO", "B I R U V i_SDSS r_SDSS", "filter", "filter", "U=k1044 Us solid U (k1044bp.txt); B=k1002",
               (33, 3600.0, 6000.0, 4418.0, 0.6937)),
    "FourStar": ("Magellan", "H Hlong Hshort J J1 J2 J3 Ks NB118 NB209", "filter+optics", "",
                 "re-derived from the *_cam_optics.txt / *_convolve.txt files",
                 (33, 14300.0, 18200.0, 16520.0, 0.9082)),
    "MegaCam": ("Magellan", "g i r u z", "filter+optics+qe+atmosphere", "filter optics qe",
                "atmosphere at 1.2 airmasses", (102, 1000.0, 29990.0, 5000.0, 0.540648)),
    "ALFOSC": ("NOT", "B R U V gp i_int ip rp up", "filter", "filter", "up=#109 u'_SDSS 353_55",
               (34, 3195.0, 7500.0, 4345.0, 0.684)),
    "SuprimeCam": ("Subaru", "B Ic Rc V Y g i r z", "filter", "filter qe", "Hamamatsu FDCCD QE",
                   (32, 3520.2, 5490.0, 4777.3, 0.90685)),
    "FOCAS": ("Subaru", "B I R U V", "filter", "filter", "percent converted to fractions",
              (35, 3570.0, 5800.0, 4810.0, 0.8426)),
    "ODI": ("WIYN", "NB422 NB695 r u", "filter", "filter", "g, i, z, NB659 and NB746 are HELD",
            (33, 3000.0, 11000.0, 4245.0, 0.974354)),
}


class _Sheet:
    def __init__(self, name, rows):
        self.name, self.rows, self.nrows = name, rows, len(rows)

    def row_values(self, r):
        return self.rows[r]


class _Book:
    def __init__(self, sheets):
        self.sheets = [_Sheet(name, rows) for name, rows in sheets]

    def sheet_by_name(self, name):
        return next(s for s in self.sheets if s.name == name)

    def sheet_by_index(self, i):
        return self.sheets[i]


@pytest.fixture(autouse=True)
def xlrd(monkeypatch):
    """xlrd itself (an optional extra) is not needed: the saved workbooks are their cells as JSON."""
    module = types.SimpleNamespace(open_workbook=lambda file_contents: _Book(json.loads(file_contents)["sheets"]))
    monkeypatch.setitem(sys.modules, "xlrd", module)


def test_every_recipe_is_covered():
    assert sorted(EXPECTED) == sorted(obs.RECIPES)


@pytest.mark.parametrize("instrument", sorted(obs.RECIPES))
def test_builds_from_the_team_files(tmp_path, instrument):
    facility, bands, components, roles, phrase, (n, first, last, at, peak) = EXPECTED[instrument]
    session = Session()
    inst_dir = obs.build(tmp_path / "filters", instrument=instrument, session=session)
    assert inst_dir == tmp_path / "filters" / facility / instrument
    manifest = check_published(inst_dir, facility, instrument, bands.split(), components, "photon")
    assert manifest.ground_based is True and manifest.outside_range == "zero" and phrase in manifest.notes
    assert sorted(manifest.components) == roles.split()
    for files in manifest.components.values():
        assert all(released_mesa_reads(inst_dir / rel) for rel in files.values())
    wl, tr = curve(inst_dir / f"{bands.split()[0]}.dat")
    assert (len(wl), wl[0], wl[-1], wl[np.argmax(tr)]) == (n, pytest.approx(first), pytest.approx(last),
                                                          pytest.approx(at))
    assert tr.max() == pytest.approx(peak)
    assert len(session.calls) == len(set(session.calls))  # each file is downloaded once


def test_omegacam_publishes_filter_times_qe_and_keeps_the_filter_scan(tmp_path):
    inst = obs.build(tmp_path, instrument="OmegaCAM", session=Session())
    manifest = InstrumentManifest.load(inst)
    for band in ("u_SDSS", "B_JOHN"):
        wl, total = curve(inst / f"{band}.dat")
        fw, filt = curve(inst / manifest.component_file("filter", band))
        assert np.array_equal(wl, fw) and np.all(total <= filt + 1e-12) and total.max() < filt.max()
    source = read_filter_header(inst / manifest.component_file("filter", "B_JOHN"))["Sources"]
    assert source.endswith("B-partD_modified_transmission_data (mean of the 4 segment-centre scans)")


def test_megacam_total_is_filter_times_qe_times_optics_times_atmosphere(tmp_path):
    inst = obs.build(tmp_path, instrument="MegaCam", session=Session())
    manifest = InstrumentManifest.load(inst)
    wl, total = curve(inst / "r.dat")
    factors = [curve(inst / manifest.component_file(role, "r")) for role in ("filter", "optics", "qe")]
    product = np.prod([np.interp(wl, w, t) for w, t in factors], axis=0)
    inside = product > 1e-3
    extinction = total[inside] / product[inside]  # the 1.2-airmass atmosphere, which is not a component
    assert np.all((extinction > 0.5) & (extinction <= 1.0))


def test_suprimecam_qe_is_shared_and_in_angstrom(tmp_path):
    inst = obs.build(tmp_path, instrument="SuprimeCam", session=Session())
    wl, qe = curve(inst / InstrumentManifest.load(inst).component_file("qe", "g"))
    assert wl[0] >= 3000 and wl[-1] <= 11000 and 0.5 < qe.max() <= 1.0


def test_keck_workbooks_need_xlrd(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "xlrd", None)  # not installed
    with pytest.raises(RuntimeError, match="needs xlrd"):
        obs.build(tmp_path, instrument="NIRC2", session=Session())
    assert not (tmp_path / "Keck").exists()


def _swap(url, other):
    return lambda files: files.__setitem__(url, files[other])


KECK, KPNO, FS, MC = obs.KECK, obs.KPNO_DOCS, obs.FS, obs.MC


@pytest.mark.parametrize("instrument, corrupt, message", [
    ("Mosaic", _swap(KPNO + "k1002bp_jul04.txt", KPNO + "k1003bp_jul04.txt"), r"k1002bp_jul04.txt: filter line"),
    ("LRIS", _swap(KECK + "lris/data_files/B.dat", KECK + "mosfire/data/filter/mosfire_J.txt"),
     "LRIS B: not Angstrom / fraction"),
    ("MOSFIRE", _swap(KECK + "mosfire/data/filter/mosfire_J.txt", KECK + "lris/data_files/B.dat"),
     "MOSFIRE J: not micron / fraction"),
    ("DEIMOS", _swap(KECK + "deimos/Bfaber13norm.txt", KECK + "deimos/RlrisDredonorm.txt"), "DEIMOS B: header PIVWV"),
    ("FourStar", _swap(FS + "HST_STD/J.txt", FS + "HST_STD/J1.txt"), "FourStar J: HST_STD differs"),
    ("MegaCam", _swap(MC + "magg.tab", MC + "magr.tab"), "MegaCam g: T column differs from magg.tab"),
    ("MegaCam", _swap(MC + "filterthroughput.magr", MC + "filterthroughput.magu"), "MegaCam r: T column differs"),
    ("FOCAS", _swap(obs.FO + "txt/U.txt", obs.SC + "txt/B.txt"), "outside 290-1001"),
])
def test_a_wrong_file_stops_the_build_before_anything_is_written(tmp_path, instrument, corrupt, message):
    files = saved()
    corrupt(files)
    with pytest.raises(ValueError, match=message):
        obs.build(tmp_path, instrument=instrument, session=Session(files))
    assert not any(tmp_path.iterdir())
