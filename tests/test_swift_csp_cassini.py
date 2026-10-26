"""Offline tests for the Swift/UVOT, LCO CSP/CSP2 and Cassini ISS builders.

Every remote file is assembled in memory with the layout of the original
(FITS extensions and keywords, the CSP tarball members, PDS tables).
"""

from __future__ import annotations

import io
import json
import tarfile

import numpy as np
import pytest

from sed_tools.component_sources import catalog, cassini, csp, swift
from sed_tools.component_sources._original import Fetcher
from sed_tools.filter_components import ComponentCoverageError, InstrumentManifest, compose_band
from sed_tools.filter_metadata import read_filter_header, released_mesa_reads, write_filter_file


class _Resp:
    def __init__(self, content, status=200):
        self.content = content if isinstance(content, bytes) else content.encode("latin-1")
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            import requests

            raise requests.HTTPError(str(self.status_code))


class _Session:
    def __init__(self, files):
        self.files, self.headers, self.calls = files, {}, []

    def get(self, url, timeout=None, **kw):
        self.calls.append(url)
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


# --------------------------------------------------------------------------- Swift
def _hdul(*tables) -> bytes:
    from astropy.io import fits

    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), *tables]).writeto(buf)
    return buf.getvalue()


def _table(name, columns, header):
    from astropy.io import fits

    cols = [fits.Column(name=n, format="E" if n != "TIME" else "D", unit=u, array=np.asarray(a))
            for n, u, a in columns]
    hdu = fits.BinTableHDU.from_columns(cols, name=name)
    for k, v in header.items():
        hdu.header[k] = v
    return hdu


GRID = np.arange(1600.0, 8001.0, 10.0)


def _gauss(centre, width):
    return np.exp(-0.5 * ((GRID - centre) / width) ** 2)


def _uvot_files(root):
    instea = 60.0 * np.exp(-0.5 * ((GRID - 3000.0) / 1500.0) ** 2)
    filt = {f: 0.9 * _gauss(c, 300.0) for f, c in
            (("U", 3500), ("B", 4400), ("V", 5400), ("UVW1", 2600), ("UVW2", 1950), ("UVM2", 2250), ("WHITE", 3800))}
    files = {
        root + swift.FTRANS: _hdul(*[
            _table("TRANSMISSION" + f, [("ENERGY", "eV", 12398.4 / GRID), ("WAVELENGTH", "angstrom", GRID),
                                        ("TRANSMIS", "", t)], {"FILTER": f}) for f, t in filt.items()]),
        root + swift.MIRROR: _hdul(_table("MIRROREFFAREA", [
            ("ENERGY", "eV", 12398.4 / GRID[1:441]), ("WAVELENGTH", "angstrom", GRID[1:441]),
            ("EA", "cm**2", np.full(440, 590.0))], {})),
        root + swift.INSTEA: _hdul(_table("INSTEFFAREA", [
            ("ENERGY", "eV", 12398.4 / GRID), ("WAVELENGTH", "angstrom", GRID), ("EA", "cm**2", instea)], {})),
        root + swift.SENSCORR: _hdul(*[
            _table("SENSCORR" + f, [("TIME", "s", [1.2e8, 7.9e8]), ("OFFSET", "", [0.0, 0.2]),
                                    ("SLOPE", "", [0.01, 0.0])], {"MJDREFI": 51910, "MJDREFF": 7.4287037e-4})
            for f in filt]),
    }
    for band, (f, arf) in swift.BANDS.items():
        scale = 0.9 if band == "white" else 1.0
        files[root + "cpf/arf/" + arf] = _hdul(_table("SPECRESP" + f, [
            ("ENERG_LO", "keV", 12.3984 / (GRID[::-1] + 5)), ("ENERG_HI", "keV", 12.3984 / (GRID[::-1] - 5)),
            ("WAVE_MIN", "angstrom", GRID[::-1] + 5), ("WAVE_MAX", "angstrom", GRID[::-1] - 5),
            ("SPECRESP", "cm**2", (scale * filt[f] * instea)[::-1])], {"FILTER": f, "CONTENT": "SPECRESP"}))
    return files


def test_uvot_falls_back_to_the_mirror_and_keeps_one_absolute_scale(tmp_path):
    session = _Session(_uvot_files(swift.CALDB_ROOTS[1]))  # the primary host serves nothing
    out = swift.build(tmp_path, instrument="UVOT", session=session)
    assert session.calls[0].startswith(swift.CALDB_ROOTS[0])
    m = InstrumentManifest.load(out)
    assert sorted(m.bands) == sorted(swift.BANDS)
    assert m.outside_range == "error" and m.ground_based is False
    assert m.components["optics"] == {"*": "components/optics/mirror.dat"}
    assert m.components["optics+qe"] == {"*": "components/optics+qe/instrument.dat"}
    h = read_filter_header(out / "U.dat")
    assert h["DetectorType"] == "photon" and h["Components"] == "filter+optics+qe"
    assert swift.CALDB_ROOTS[1] in h["Sources"]
    u = np.loadtxt(out / "U.dat", delimiter=",", skiprows=1)
    assert np.all(np.diff(u[:, 0]) > 0) and u[0, 0] == 1600.0
    assert u[:, 1].max() < 60.0 / swift.GEOMETRIC_AREA_CM2 + 1e-6  # cm^2 / 659 cm^2
    # Composition reproduces the published shape; white differs only by a constant.
    for band in ("U", "white"):
        comp = compose_band(out, band, "filter+optics+qe")
        pub = np.loadtxt(out / f"{band}.dat", delimiter=",", skiprows=1)
        ratio = np.interp(pub[:, 0], comp.wavelength, comp.transmission)[pub[:, 1] > 0.01] / pub[:, 1][pub[:, 1] > 0.01]
        assert np.allclose(ratio, 1.0 if band == "U" else 1 / 0.9, rtol=1e-5)
    with pytest.raises(ComponentCoverageError):  # mirror stops at 6000 A
        compose_band(out, "V", "filter+optics")
    assert "1+OFFSET=1.2000" in m.notes and "NOT applied" in m.notes
    assert all(released_mesa_reads(p) for p in out.rglob("*.dat"))


def test_uvot_rejects_a_wrong_filter_extension():
    files = _uvot_files(swift.CALDB_ROOTS[0])
    v = swift.BANDS["V"][1]
    files[swift.CALDB_ROOTS[0] + "cpf/arf/" + v] = files[swift.CALDB_ROOTS[0] + "cpf/arf/" + swift.BANDS["B"][1]]
    with pytest.raises(ValueError, match="FILTER"):
        swift.uvot(Fetcher(_Session(files)))


# --------------------------------------------------------------------------- CSP
def _two(wl, tr):
    return "".join(f" {w:.1f} {t:.8f}\n" for w, t in zip(wl, tr))


def _csp_package():
    members = {"README": "These final fitlers have atmosphere corresponding to 1.2 airmass.\n"}
    opt = {"u": 3550, "g": 4750, "r": 6200, "i": 7600, "B": 4350, "V_LC3014": 5400, "V_LC3009": 5400,
           "V_LC9844": 5400}
    atm_wl = np.array([2900.0, 4000.0, 6000.0, 70000.0])
    atm_tr = np.array([0.3, 0.7, 0.9, 1.0])
    members["atm_ext_ctio.dat"] = _two(atm_wl, atm_tr)
    members["atm_ext_lines.dat"] = _two([2950.0, 70000.0], [1.0, 1.0])
    for band, c in opt.items():
        wl = np.arange(c - 900.0, c + 901.0, 20.0)
        raw = 0.8 * np.exp(-0.5 * ((wl - c) / 120.0) ** 2)
        members[f"{band}_texas_WLcorr.txt"] = _two(wl, raw)
        members[f"{band}_tel_ccd_atm_ext_1.2.dat"] = _two(wl, raw * np.interp(wl, atm_wl, atm_tr) ** 1.2)
    for name, c in (("Y_SWO", 10350), ("Jrc1_SWO", 12400), ("Jrc2_SWO", 12350), ("H_SWO", 16300),
                    ("Y_DUP", 10400), ("J_DUP", 12380), ("H_DUP", 16280)):
        wl = np.arange(c - 1500.0, c + 1501.0, 4.0)
        members[f"{name}_TAM_scan_atm.dat"] = _two(wl, np.exp(-0.5 * ((wl - c) / 300.0) ** 2) - 0.001)
    nm = np.arange(300.0, 800.01, 0.5)
    members["Telescope.txt"] = "#Wavelength (nm)\n" + _two(nm, np.full(nm.size, 0.012))
    nm = np.arange(302.0, 900.01, 1.0)
    members["Telescope+Camera.txt"] = _two(nm, np.full(nm.size, 0.011))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, text in {**members, "._README": "resource fork"}.items():
            data = text.encode()
            info = tarfile.TarInfo("CSP_filter_package/" + name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_csp1_epochs_atmosphere_and_per_band_components(tmp_path):
    session = _Session({csp.PACKAGE: _csp_package()})
    out = csp.build(tmp_path, instrument="CSP", session=session)
    m = InstrumentManifest.load(out)
    assert out == tmp_path / "LCO" / "CSP"
    assert set(m.bands) == set(csp.CSP1_OPTICAL) | set(csp.CSP1_NIR)
    assert {"V_LC3014", "V_LC3009", "V_LC9844", "Jrc1", "Jrc2"} <= set(m.bands)
    h = read_filter_header(out / "Jrc2.dat")
    assert h["Components"] == "filter+optics+qe+atmosphere" and h["Atmosphere"] == "airmass=1.2"
    # Components only on the Swope optical bands, never shared with the NIR bands.
    for entries in m.components.values():
        assert "*" not in entries and not set(entries) & set(csp.CSP1_NIR)
    assert "i" not in m.components["optics"]  # Telescope.txt ends at 800 nm
    assert set(m.components["optics+qe"]) == set(csp.CSP1_OPTICAL)
    assert m.components["optics"]["u"] == "components/optics/telescope.dat"
    assert m.buildable() == []  # no filter-only curve is published
    assert m.default_site == csp.CSP_SITE
    comp = compose_band(out, "g", "filter+optics+qe")
    assert read_filter_header(out / m.components["filter+optics+qe"]["g"])["Components"] == "filter+optics+qe"
    assert comp.transmission.max() == pytest.approx(0.8, rel=1e-6)
    assert "g 1.000 (+-0.000)" in m.notes
    assert np.loadtxt(out / "Y.dat", delimiter=",", skiprows=1)[:, 1].min() == 0.0  # noise clipped
    assert all(released_mesa_reads(p) for p in out.rglob("*.dat"))


def test_csp2_reads_the_linked_files(tmp_path):
    files = {}
    for band, name in csp.CSP2_FILES.items():
        wl = np.arange(3000.0, 20000.0, 50.0)
        files[csp.DATA + name] = _two(wl, np.exp(-0.5 * ((wl - 9000.0) / 3000.0) ** 2))
    out = csp.build(tmp_path, instrument="CSP2", session=_Session(files))
    m = InstrumentManifest.load(out)
    assert sorted(m.bands) == sorted(csp.CSP2_FILES) and m.components == {}
    assert read_filter_header(out / "V.dat")["Atmosphere"] == "airmass=1.2"
    assert "Atmosphere" not in read_filter_header(out / "J.dat")  # NIR airmass not stated
    assert catalog.builder_for("LCO", "CSP2").components is False


# --------------------------------------------------------------------------- Cassini
def _pds(rows):
    return "PDS_VERSION_ID = PDS3\nEND\n\\begindata\n" + "".join(" ".join(f"{x:.6f}" for x in r) + "\n" for r in rows)


def _cassini_files(camera):
    nm = np.arange(200.0, 1057.05, 0.1)
    qe = np.clip(0.5 - np.abs(nm - 650.0) / 1000.0, 0.01, None)
    optics = np.full(nm.size, 0.6)
    grid = np.arange(200.0, 1101.0)
    files = {cassini.QE_URL: _pds(np.c_[grid, np.interp(grid, nm, qe), np.interp(grid, nm, qe)]),
             cassini.EFFICIENCY + cassini.CAMERAS[camera][1]: _pds(np.c_[np.arange(200.0, 1100.1, 2.5),
                                                                          np.full(361, 0.6)])}
    pairs = {}
    for i, band in enumerate(cassini.BANDS[camera]):
        centre = 1000.0 if band == "IR4" else 300.0 + 15.0 * i
        pairs[band] = 0.9 * np.exp(-0.5 * ((nm - centre) / 20.0) ** 2)
        system = qe * optics * pairs[band]
        files[cassini.systrans_url(camera, band)] = _pds(np.c_[nm, system, system])
    return files, nm, pairs


def test_cassini_publishes_pds_curves_and_keeps_svo_copies_apart(tmp_path):
    files, nm, pairs = _cassini_files("ISS_NAC")
    inst = tmp_path / "Cassini" / "ISS_NAC"
    inst.mkdir(parents=True)
    # An SVO copy (a window of the PDS curve) kept as <band>_svo, as on the njm mirror.
    sys_bl1 = np.loadtxt(io.StringIO(files[cassini.systrans_url("ISS_NAC", "BL1")].split("\\begindata\n")[1]))
    window = (sys_bl1[:, 0] > 400) & (sys_bl1[:, 0] < 450)
    write_filter_file(inst / "BL1_svo.dat", sys_bl1[window, 0] * 10, sys_bl1[window, 1], detector_type="photon",
                      source="svo", filter_id="Cassini/ISS_NAC.BL1", wavelength_unit="Angstrom",
                      extra={"Components": "filter+optics+qe+instrument", "ComponentsSource": "svo"})
    svo_before = (inst / "BL1_svo.dat").read_bytes()
    out = cassini.build(tmp_path, instrument="ISS_NAC", session=_Session(files))
    assert (out / "BL1_svo.dat").read_bytes() == svo_before
    assert (out / "ISS_NAC").read_text().split() == [f"{b}.dat" for b in sorted(cassini.BANDS["ISS_NAC"])]
    h = read_filter_header(out / "BL1.dat")
    assert h["FilterID"] == "Cassini/ISS_NAC.BL1" and h["DetectorType"] == "photon"
    assert h["Components"] == "filter+optics+qe" and h["ComponentsSource"] == "original"
    assert cassini.systrans_url("ISS_NAC", "BL1") in h["Sources"]
    pub = np.loadtxt(out / "BL1.dat", delimiter=",", skiprows=1)
    assert np.allclose(pub[:, 0], nm * 10) and np.allclose(pub[:, 1], sys_bl1[:, 1])
    m = json.loads((out / "manifest.json").read_text())
    assert m["bands"] == sorted(cassini.BANDS["ISS_NAC"]) and "BL1_svo" not in m["detector_types"]
    assert m["published"] == {"components": "filter+optics+qe", "source": "original"}
    assert "IR4" not in m["components"]["filter"]  # IR4 extends past 1000 nm
    assert m["components"]["qe"] == {"*": "components/qe/qe.dat"}
    assert read_filter_header(out / "components/filter/BL1.dat")["Sources"].startswith(
        "DERIVED filter pair T(BL1)*T(CL2)")
    f = np.loadtxt(out / "components/filter/BL1.dat", delimiter=",", skiprows=1)
    assert f[-1, 0] == pytest.approx(10000.0)
    assert np.allclose(f[:, 1], np.interp(f[:, 0] / 10, nm, pairs["BL1"]), atol=2e-3)
    comp = compose_band(out, "BL1", "filter+optics+qe")
    assert np.allclose(np.interp(comp.wavelength, pub[:, 0], pub[:, 1]), comp.transmission, rtol=1e-9, atol=1e-15)
    assert "IR4 (IR4xCL2" in m["notes"] and "1 SVO copies equal the PDS curve" in m["notes"]
    assert all(released_mesa_reads(p) for p in out.rglob("*.dat"))


def test_cassini_keeps_existing_published_curves(tmp_path):
    files, _nm, _pairs = _cassini_files("ISS_WAC")
    inst = tmp_path / "Cassini" / "ISS_WAC"
    inst.mkdir(parents=True)
    write_filter_file(inst / "GRN.dat", [5000.0, 5500.0, 6000.0], [0.0, 0.2, 0.0], detector_type="photon",
                      source="svo", filter_id="Cassini/ISS_WAC.GRN")
    before = (inst / "GRN.dat").read_bytes()
    out = cassini.build(tmp_path, instrument="ISS_WAC", session=_Session(files))
    assert (out / "GRN.dat").read_bytes() == before
    assert "(not for GRN)" in json.loads((out / "manifest.json").read_text())["notes"]


def test_cassini_band_names_map_to_filter_pairs():
    assert cassini.filter_pair("ISS_NAC", "Clear") == ("cl1", "cl2")
    assert cassini.filter_pair("ISS_NAC", "BL2") == ("cl1", "bl2")
    assert cassini.filter_pair("ISS_NAC", "RED_GRN") == ("red", "grn")
    assert cassini.filter_pair("ISS_WAC", "VIO") == ("cl1", "vio")
    assert cassini.filter_pair("ISS_WAC", "MT2_IR1") == ("mt2", "ir1")
    with pytest.raises(ValueError):
        cassini.filter_pair("ISS_WAC", "UV3")
    ref = catalog.builder_for("Cassini", "ISS_WAC")
    assert ref.name == "cassini" and ref.publishes and ref.components
