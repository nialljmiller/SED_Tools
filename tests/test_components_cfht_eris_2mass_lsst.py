"""Offline tests for the CFHT, ERIS/NIX, 2MASS and LSST component builders."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from sed_tools.component_sources import catalog, cfht, eris, lsst, twomass
from sed_tools.filter_components import (FILTER, OPTICS, QE, SHARED, AtmosphereGrid,
                                         InstrumentManifest, compose_band, read_curve)
from sed_tools.filter_metadata import (ENERGY, PHOTON, read_filter_header, released_mesa_reads,
                                       write_filter_file)


class _Resp:
    def __init__(self, content, status=200):
        self.content = content if isinstance(content, bytes) else content.encode()
        self.text = self.content.decode("latin-1")
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self, files):
        self.files = files
        self.headers = {}
        self.calls = []

    def get(self, url, timeout=None, **kw):
        self.calls.append(url)
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


def _published(directory, band, wl, tr, components, detector=PHOTON):
    directory.mkdir(parents=True, exist_ok=True)
    write_filter_file(directory / f"{band}.dat", wl, tr, detector_type=detector, source="svo",
                      wavelength_unit="Angstrom", extra={"Components": components, "ComponentsSource": "svo"})


def _table(wl, tr, sep=" "):
    return "".join(f"{w}{sep}{t}\n" for w, t in zip(wl, tr))


# ---------------------------------------------------------------------- CFHT
def test_cfht_scan_averages_position_columns_and_converts_units():
    text = "400,50,60,70\n300,0,-1,1\n"
    wl, tr, note = cfht.scan(text, "u")
    np.testing.assert_allclose(wl, [3000.0, 4000.0])
    np.testing.assert_allclose(tr, [0.0, 0.6])
    assert note.startswith("mean of 3 scans")
    wl, tr, note = cfht.scan("  554.00     80.0\n  552.00     40.0\n", "o")
    np.testing.assert_allclose(tr, [0.4, 0.8])
    assert "mean" not in note


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


def test_cfht_megacam_generations():
    m = cfht.BANDS["MegaCam"]
    assert (m["u"], m["u_1"], m["i_1"], m["i_2"], m["z_1"]) == ("9302", "9301", "9701", "9702", "9801")
    assert all(code in cfht.FILTERS for band in cfht.BANDS.values() for code in band.values())


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


def test_eris_optics_are_dichroic_times_default_camera(tmp_path):
    inst = tmp_path / "ERIS" / "NIX"
    _published(inst, "J", np.linspace(11500, 14000, 20), np.full(20, 0.8), "filter")
    _published(inst, "Lp", np.linspace(34000, 43000, 20), np.full(20, 0.8), "filter")
    eris.build(tmp_path, session=_Session({eris.ZIP_URL: _nix_zip()}))
    m = InstrumentManifest.load(inst)
    assert m.component_file(FILTER, "J") == "J.dat"
    assert m.component_file(OPTICS, "J").endswith("dichroic_camera13mas_JHK.dat")
    assert m.component_file(OPTICS, "Lp").endswith("dichroic_camera13mas_LM.dat")
    j = read_curve(inst / m.component_file(OPTICS, "J"))
    np.testing.assert_allclose(j.transmission, 0.45)                   # 50 % x 0.9
    assert (inst / "components/optics/dichroic_camera27mas_JHK.dat").exists()
    assert m.default_site == "paranal" and m.outside_range == "zero"
    assert compose_band(inst, "Lp", (FILTER, OPTICS)).transmission.max() == pytest.approx(0.32)


def test_eris_unknown_band_is_refused(tmp_path):
    inst = tmp_path / "ERIS" / "NIX"
    _published(inst, "Q", np.linspace(11500, 14000, 20), np.full(20, 0.8), "filter")
    with pytest.raises(ValueError, match="camera"):
        eris.build(tmp_path, session=_Session({eris.ZIP_URL: _nix_zip()}))


# --------------------------------------------------------------------- 2MASS
def _pre(title, um, values):
    rows = "".join(f"  {i + 1:3d}  {w:.3f}  {v:.4f}\n" for i, (w, v) in enumerate(zip(um, values)))
    return (f"<html><head><title>{title}</title></head><body><b><pre>\n\\CHAR TITLE = {title}\n"
            f"|n  |xlam     |yval     |\n|i  |r        |r        |\n{rows}</pre></body></html>")


def _twomass_files():
    um = np.linspace(1.0, 2.5, 31)
    files = {}
    values = {1: 0.9, 2: 0.95, 3: 0.97, 4: 0.99, 6: 0.98, 7: 0.97, 9: 0.96, 11: 0.6}
    for n, title in twomass.TITLES.items():
        if n in values:
            files[twomass.TABLE.format(n=n)] = _pre(title, um, np.full(um.size, values[n]))
    for n, (lo, hi) in ((5, (1.1, 1.4)), (8, (1.5, 1.8)), (10, (2.0, 2.3))):
        x = np.linspace(lo - 0.05, hi + 0.05, 13)
        files[twomass.TABLE.format(n=n)] = _pre(twomass.TITLES[n].replace("-BAND", "K<sub>s</sub>-BAND")
                                                if n == 10 else twomass.TITLES[n], x,
                                                np.where((x > lo) & (x < hi), 0.8, 0.0))
    return files


def test_twomass_optics_follow_each_band_light_path(tmp_path):
    inst = tmp_path / "2MASS" / "2MASS"
    for band, lo in (("J", 10600), ("H", 15000), ("Ks", 19500)):
        wl = np.linspace(lo, lo + 4000, 30)
        _published(inst, band, wl, np.ones_like(wl), "filter+instrument+atmosphere", detector=ENERGY)
    twomass.build(tmp_path, session=_Session(_twomass_files()))
    m = InstrumentManifest.load(inst)
    base = 0.9 ** 2 * 0.95 * 0.97 ** 7
    for band, path in (("J", 0.99), ("H", 0.98 * 0.97), ("Ks", 0.98 * 0.96)):
        assert m.component_file(OPTICS, band) == f"components/optics/{band}.dat"
        np.testing.assert_allclose(read_curve(inst / m.component_file(OPTICS, band)).transmission,
                                   base * path, rtol=1e-12)
    assert m.component_file(QE, SHARED) == "components/qe/nicmos3.dat"
    assert m.outside_range == "zero"
    j = compose_band(inst, "J", (FILTER, OPTICS, QE))
    assert j.transmission.max() == pytest.approx(0.8 * base * 0.99 * 0.6)
    assert read_filter_header(inst / "J.dat")["DetectorType"] == ENERGY  # published RSR untouched


def test_twomass_table_title_is_checked():
    with pytest.raises(ValueError, match="title"):
        twomass.micron_curve(_pre("SOMETHING ELSE", [1.0, 2.0], [0.5, 0.5]), "u", twomass.TITLES[1])


# ---------------------------------------------------------------------- LSST
def _lsst_files(tag, scale=1.0):
    nm = np.arange(300.0, 1151.0, 50.0)
    one = _table(nm, np.full(nm.size, 0.9))
    files = {}
    for name in lsst.OPTICS_FILES + ("detector", "atmos_10", "atmos_std"):
        files[lsst._url(name, tag)] = "# header\n" + one
    for b in lsst.BANDS:
        filt = np.where((nm >= 500) & (nm <= 700), 0.8 * scale, 0.0)
        files[lsst._url(f"filter_{b}", tag)] = _table(nm, filt)
        files[lsst._url(f"total_{b}", tag)] = _table(nm, filt * 0.9 ** 8)
        files[lsst._url(f"hardware_{b}", tag)] = _table(nm, filt * 0.9 ** 7)
    return files


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


def test_catalog_registers_new_builders():
    assert catalog.builder_for("LSST", "LSSTv19").publishes
    assert dict(catalog.builder_for("CFHT", "MegaCam").kwargs) == {"instrument": "MegaCam"}
    assert catalog.builder_for("ERIS", "NIX").name == "eris"
    assert catalog.component_availability("2MASS", "2MASS")["status"] == "supported"


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
