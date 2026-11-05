"""Survey and standard photometric systems, each built end to end from saved files (tests/fixtures/builders)."""

from __future__ import annotations

import hashlib
import io
import tarfile

import numpy as np
import pytest

from builder_fixtures import Session, check_published, curve, saved
from sed_tools.component_sources import standards, surveys
from sed_tools.filter_components import AtmosphereGrid, InstrumentManifest
from test_original_sources import _tar

# instrument: module, facility, bands, published components, component roles, default site,
# and the first band's samples, first and last wavelength (A), peak wavelength (A) and peak
EXPECTED = {
    "ATLAS": (surveys, "ATLAS", "c o", "filter+optics+qe+atmosphere", "", None, (57, 4000.0, 6800.0, 6400.0, 0.717)),
    "PS1": (surveys, "PAN-STARRS", "g i r w y z", "filter+optics+qe+atmosphere", "", "haleakala_ps1",
            (73, 3800.0, 11000.0, 5400.0, 0.501)),
    "OGLE-IV": (surveys, "OGLE", "I V", "filter+qe", "filter qe", None, (8, 3101.2, 9475.09, 7619.99, 0.6787273)),
    "WFCAM": (surveys, "UKIRT", "H J K Y Z", "filter+optics+qe+atmosphere", "qe", None,
              (39, 13510.0, 19500.0, 16300.0, 0.266)),
    "Sinistro": (surveys, "LasCumbres", "B I R U V Y gp ip rp up zs", "filter", "filter", None,
                 (39, 2000.0, 13000.0, 4320.0, 0.7274351)),
    "Spectral": (surveys, "LasCumbres", "B I R V Y gp ip rp up zs", "filter", "filter", None,
                 (39, 2000.0, 13000.0, 4320.0, 0.7274351)),
    "Bessell": (standards, "Generic", "B I R U V", "filter+optics+qe", "", None, (4, 4100.0, 4400.0, 4100.0, 1.0)),
    "MKO": (standards, "Generic", "H J K Lp Mp", "filter", "filter", None, (38, 12977.0, 19964.0, 16111.0, 0.90177917)),
    "USNO": (standards, "Generic", "gp ip rp up zp", "filter+optics+qe+atmosphere", "", None,
             (20, 3630.0, 5830.0, 5130.0, 0.38349)),
}


@pytest.fixture(autouse=True)
def lco_fixture_is_the_verified_copy(monkeypatch):
    """The saved LCO zip is cut down, so it is checked against its own SHA-256."""
    sha, lco = hashlib.sha256(saved()[surveys.LCO_URL]).hexdigest(), surveys._lco
    monkeypatch.setattr(surveys, "_lco", lambda instrument, fetch: lco(instrument, fetch, sha256=sha))


def test_every_recipe_is_covered():
    assert sorted(EXPECTED) == sorted([*surveys.RECIPES, *standards.RECIPES])


@pytest.mark.parametrize("instrument", sorted(EXPECTED))
def test_builds_from_the_team_files(tmp_path, instrument):
    module, facility, bands, components, roles, site, (n, first, last, at, peak) = EXPECTED[instrument]
    session = Session()
    inst_dir = module.build(tmp_path / "filters", instrument=instrument, session=session)
    manifest = check_published(inst_dir, facility, instrument, bands.split(), components, "photon")
    assert manifest.ground_based is True and manifest.outside_range == "zero" and manifest.default_site == site
    assert sorted(manifest.components) == roles.split()
    wl, tr = curve(inst_dir / f"{bands.split()[0]}.dat")
    assert (len(wl), wl[0], wl[-1], wl[np.argmax(tr)]) == (n, pytest.approx(first), pytest.approx(last),
                                                          pytest.approx(at))
    assert tr.max() == pytest.approx(peak)


def test_ps1_writes_its_team_atmosphere_site(tmp_path):
    surveys.build(tmp_path / "filters", instrument="PS1", session=Session())
    site = tmp_path / "atmospheres" / "haleakala_ps1"
    grid = AtmosphereGrid.load(site)
    assert grid.kind == "team" and [(e["airmass"], e["pwv"]) for e in grid.entries] == [(1.2, None)]
    wl, tr = curve(site / grid.entries[0]["file"])
    assert (wl[0], wl[-1], len(wl)) == (3800.0, 11000.0, 73) and tr.max() <= 1.0
    assert tr[0] == pytest.approx(0.945 * 0.694 * 0.998)  # table3 at 380 nm: Aero x Ray x Mol
    assert "Aero, Ray and Mol" in grid.entries[0]["derivation"]


def test_wfcam_qe_is_the_paper_table(tmp_path):
    inst = surveys.build(tmp_path, instrument="WFCAM", session=Session())
    wl, qe = curve(inst / InstrumentManifest.load(inst).component_file("qe", "J"))
    np.testing.assert_allclose(wl, np.array(surveys.UKIRT_QE_UM) * 1e4)
    np.testing.assert_allclose(qe, surveys.UKIRT_QE)


def test_atlas_needs_every_table_row(tmp_path):
    files = saved()
    with tarfile.open(fileobj=io.BytesIO(files[surveys.ATLAS_URL])) as t:
        tex = t.extractfile("atlas_defn.tex").read().decode()
    row = next(ln for ln in tex.splitlines() if ln.startswith("450 & "))
    files[surveys.ATLAS_URL] = _tar({"atlas_defn.tex": tex.replace(row + "\n", "")})
    with pytest.raises(ValueError, match="ATLAS c: parsed 54 rows, expected 57"):
        surveys.build(tmp_path, instrument="ATLAS", session=Session(files))
    assert not any(tmp_path.iterdir())


def test_bessell_checks_the_readme(tmp_path):
    files = saved()
    files[standards.B12_URL.replace("table1.dat", "ReadMe")] = b"Byte-by-byte Description of file: table1.dat\n"
    with pytest.raises(ValueError, match="unexpected Bessell & Murphy ReadMe"):
        standards.build(tmp_path, instrument="Bessell", session=Session(files))
    assert not any(tmp_path.iterdir())
