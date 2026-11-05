"""Offline tests for the CFHT, ERIS/NIX, 2MASS and LSST component builders."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from sed_tools.component_sources import catalog, cfht, eris, lsst, twomass
from sed_tools.filter_components import (FILTER, OPTICS, QE, SHARED, AtmosphereGrid,
                                         InstrumentManifest, compose_band, read_curve)
from sed_tools.filter_metadata import ENERGY, read_filter_header, released_mesa_reads, write_filter_file

from helpers_builders_cfht_lsst import _Session, _lsst_files, _pre, _published, _table, _twomass_files


def _cfht_files(codes):
    nm = np.arange(300, 1101, 50)
    files = {
        cfht.PRIMARY: _table(nm, np.full(nm.size, 90.0)),
        cfht.CORRECTOR: _table(nm, np.full(nm.size, 80.0)),
        cfht.QE_MODEL: "lambda (nm)  mean (%)\n" + _table(np.arange(280, 1051, 10), np.full(78, 50.0)),
        cfht.QE_MEASURED: "lambda (nm) mean (%) stdev (%)\n350 48.9 6.2\n1000 4.3 0.2\n",
    }
    for code in codes:
        files[cfht.CURVE.format(code=code)] = "".join(
            f"{w},{t},{t}\n" for w, t in ((800, 0), (850, 90), (900, 90), (950, 0)))
    return files


def test_cfht_bands_get_their_own_generation_and_ignore_mislabelled_sibling(tmp_path):
    inst = tmp_path / "CFHT" / "Megaprime"
    wl = np.linspace(8000, 9500, 16)
    _published(inst, "z", wl, np.ones_like(wl) * 0.3, "filter+optics+qe")
    _published(inst, "z_fil", wl, np.ones_like(wl), "filter")    # SVO's sibling: really the old Z'
    _published(inst, "zS", wl, np.ones_like(wl) * 0.3, "filter+optics+qe")
    before = (inst / "z.dat").read_bytes()
    files = _cfht_files(["9901", "9801"])
    files[cfht.CURVE.format(code="9801")] = "1200.00 94.0\n800.00 50.0\n700.00 0.0\n"

    cfht.build(tmp_path, instrument="Megaprime", session=_Session(files))

    m = InstrumentManifest.load(inst)
    assert (inst / "z.dat").read_bytes() == before
    assert m.component_file(FILTER, "z") == "components/filter/z.dat"
    assert "cfh9901" in read_filter_header(inst / "components/filter/z.dat")["Sources"]
    assert "cfh9801" in read_filter_header(inst / "components/filter/zS.dat")["Sources"]
    assert m.component_file(FILTER, "z_fil") == "z_fil.dat"           # the sibling stays itself
    assert m.component_file(OPTICS, "z") == "components/optics/primary_corrector.dat"
    optics = read_curve(inst / "components/optics/primary_corrector.dat")
    np.testing.assert_allclose(optics.transmission, 0.72)               # one reflection x corrector
    assert m.component_file(QE, "z") == "components/qe/e2v_ccd42-90_model.dat"
    assert (inst / "components/qe/measured_mean.dat").exists()
    assert "measured_mean" not in str(m.components)
    assert m.outside_range == "zero" and m.default_site is None
    # zS (old long-pass to 1200 nm) composes: the model QE ends at 1050 nm and is zero beyond.
    zs = compose_band(inst, "zS", (FILTER, OPTICS, QE))
    assert zs.transmission[zs.wavelength > 10500].max() == 0.0
    for p in inst.rglob("*.dat"):
        assert released_mesa_reads(p)


def test_cfht_needs_published_curves(tmp_path):
    with pytest.raises(FileNotFoundError):
        cfht.build(tmp_path, instrument="MegaCam", session=_Session({}))
    with pytest.raises(ValueError):
        cfht.build(tmp_path, instrument="WIRCam", session=_Session({}))


# ---------------------------------------------------------------------- ERIS
def _nix_zip():
    def metres(lo_um, hi_um, value):
        x = np.linspace(lo_um, hi_um, 50) * 1e-6
        return "# data from ERIS/NIX/x.txt\n" + "".join(f"  {w:.16f}  {value}\n" for w in x)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("nix-filters/cameras1-13masJHK.dat", metres(1.0, 2.59, 0.9))
        zf.writestr("nix-filters/cameras2-27mas-JHK.dat", metres(1.0, 2.69, 0.85))
        zf.writestr("nix-filters/cameras3-13mas-LM.dat", metres(2.93, 5.5, 0.8))
        zf.writestr("nix-filters/Dichroic.txt",
                    "# \n# [1] : Wavelength [nm]\n# [2] : Transmission [%]\n# \n"
                    + _table(np.arange(900, 5401, 100), np.full(46, 50.0)))
        zf.writestr("nix-filters/ND.txt", _table([1000, 5000], [0.19, 1.61]))
    return buf.getvalue()


def test_eris_unknown_band_is_refused(tmp_path):
    inst = tmp_path / "ERIS" / "NIX"
    _published(inst, "Q", np.linspace(11500, 14000, 20), np.full(20, 0.8), "filter")
    with pytest.raises(ValueError, match="camera"):
        eris.build(tmp_path, session=_Session({eris.ZIP_URL: _nix_zip()}))


def test_twomass_table_title_is_checked():
    with pytest.raises(ValueError, match="title"):
        twomass.micron_curve(_pre("SOMETHING ELSE", [1.0, 2.0], [0.5, 0.5]), "u", twomass.TITLES[1])


# ---------------------------------------------------------------------- LSST
class _SVO:
    def get_filter(self, fid):
        raise AssertionError("published curves exist; SVO must not be asked")


def test_lsst_releases_are_separate_instruments(tmp_path):
    filters = tmp_path / "filters"
    old = filters / "LSST" / "LSST"
    wl = np.linspace(5000, 7000, 21)
    for b in lsst.BANDS:
        _published(old, b, wl, np.full(wl.size, 0.4), "filter+instrument+atmosphere")
    site = tmp_path / "atmospheres" / "lsst_std"
    site.mkdir(parents=True)
    AtmosphereGrid(site="lsst_std", entries=[{"airmass": 1.2, "pwv": None, "file": "am1.20.dat"}]).save(site)
    write_filter_file(site / "am1.20.dat", [3000.0, 12000.0], [0.9, 0.9], detector_type=ENERGY,
                      source="component", wavelength_unit="Angstrom")
    site_before = (site / "manifest.json").read_bytes()
    files = {**_lsst_files("1.5"), **_lsst_files("v1.9.2", scale=0.5)}
    session = _Session(files)

    lsst.build(filters, instrument="LSST", session=session, svo_client=_SVO())
    lsst.build(filters, instrument="LSSTv19", session=session)

    m15 = InstrumentManifest.load(old)
    assert m15.component_file(FILTER, "g") == "components/filter/g_v1p5.dat"
    assert m15.component_file(QE, "g") == "components/qe/detector_v1p5.dat"
    # Component sources are recorded per band ("<role>:<band>") since PR #51.
    assert "lsst/throughputs 1.5" in m15.sources[f"{FILTER}:g"]
    assert "X=1.0" in m15.notes and "1.2 airmasses" in m15.notes
    new = filters / "LSST" / "LSSTv19"
    m19 = InstrumentManifest.load(new)
    assert m19.bands == list(lsst.BANDS) and m19.default_site == "lsst_std"
    assert m19.published_bands["r"] == "filter+optics+qe+atmosphere"
    header = read_filter_header(new / "r.dat")
    assert header["FilterID"] == "LSST/LSSTv19.r" and header["DetectorTypeSource"] == "assumed"
    assert (new / "LSSTv19").read_text().split() == [f"{b}.dat" for b in sorted(lsst.BANDS)]
    composed = compose_band(new, "r", (FILTER, OPTICS, QE), atmosphere={"airmass": 1.2},
                            atmosphere_root=tmp_path / "atmospheres")
    published = read_curve(new / "r.dat")
    np.testing.assert_allclose(np.interp(published.wavelength, composed.wavelength, composed.transmission),
                               published.transmission, atol=1e-12)
    assert (site / "manifest.json").read_bytes() == site_before        # existing site is not rewritten
    for p in new.rglob("*.dat"):
        assert released_mesa_reads(p)


def test_earlier_manifest_note_is_carried_once(tmp_path):
    inst = tmp_path / "ERIS" / "NIX"
    _published(inst, "J", np.linspace(11500, 14000, 20), np.full(20, 0.8), "filter")
    m = InstrumentManifest.load(inst)
    m.notes = "One negative sample clipped to 0."
    m.save(inst)
    session = _Session({eris.ZIP_URL: _nix_zip()})
    eris.build(tmp_path, session=session)
    first = InstrumentManifest.load(inst).notes
    eris.build(tmp_path, session=session)
    assert InstrumentManifest.load(inst).notes == first
    assert first.endswith("Earlier note: One negative sample clipped to 0.")
    assert first.count("Earlier note") == 1
