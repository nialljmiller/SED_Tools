"""Offline tests for the instruments rebuilt from their teams' original files.

Fixtures in tests/fixtures/original_sources are a few lines excerpted from each
original file (same layout and header); archives and FITS files are assembled
in memory from them.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sed_tools import filter_library
from sed_tools.component_sources import BUILDERS, get_builder, catalog
from sed_tools.component_sources import csp, observatories, space, standards, surveys, swift
from sed_tools.component_sources._original import (Fetcher, clean_curve, make_band, to_angstrom)
from sed_tools.filter_components import InstrumentManifest
from sed_tools.filter_library import FilterLibrary
from sed_tools.filter_metadata import read_filter_header

FIX = Path(__file__).parent / "fixtures" / "original_sources"
MODULES = {"surveys": surveys, "space": space, "observatories": observatories, "standards": standards,
           "swift": swift, "csp": csp}


def fx(name: str) -> str:
    return (FIX / name).read_text(encoding="latin-1")


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


def _tar(members) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, text in members.items():
            data = text.encode("ascii")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _zip(members) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in members.items():
            z.writestr(name, text)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_every_original_source_instrument_resolves():
    keys = [(fac, inst) for entries in catalog.ORIGINAL_SOURCES.values() for fac, inst, _ in entries]
    assert len(keys) == len(set(keys)) == 40
    for module, entries in catalog.ORIGINAL_SOURCES.items():
        assert module in BUILDERS and callable(get_builder(module))
        recipes = MODULES[module].RECIPES
        assert sorted(recipes) == sorted(inst for _, inst, _ in entries)
        for fac, inst, comps in entries:
            ref = catalog.builder_for(fac, inst)
            assert ref.name == module and dict(ref.kwargs) == {"instrument": inst}
            assert ref.publishes and ref.components is comps and not ref.heavy
            assert recipes[inst][0] == fac


def test_original_sources_do_not_shadow_other_builders():
    original = {(f, i) for entries in catalog.ORIGINAL_SOURCES.values() for f, i, _ in entries}
    others = {k for k, ref in catalog.BUILDERS_FOR.items() if ref.name not in MODULES}
    assert not original & others
    assert len(others) >= 28  # the builders registered before these were added
    assert all(ref.publishes for ref in catalog.BUILDERS_FOR.values() if ref.name == "tabular")


def test_availability_of_total_only_original_sources():
    info = catalog.component_availability("GAIA", "DR2")
    assert info["status"] == catalog.TOTAL_ONLY and info["builder"] is None
    assert info["original_published"] is True
    assert catalog.component_availability("OGLE", "OGLE-IV")["status"] == "supported"


def test_unknown_instrument_and_wrong_facility_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="No original-source recipe"):
        surveys.build(tmp_path, instrument="Nope")
    with pytest.raises(ValueError, match="belongs to"):
        surveys.build(tmp_path, instrument="ATLAS", facility="Keck")


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------
def test_units_and_cleaning():
    np.testing.assert_allclose(to_angstrom([1.0], "micron"), [1e4])
    np.testing.assert_allclose(to_angstrom([1.0], "1/cm"), [1e8])
    np.testing.assert_allclose(to_angstrom([299.792458], "GHz"), [1e7])
    wl, tr, note = clean_curve([3, 1, 2, 2, 4], [0.5, -0.1, 0.2, 0.4, np.nan], normalize=True)
    np.testing.assert_allclose(wl, [1, 2, 3])
    np.testing.assert_allclose(tr, [0.0, 0.6, 1.0])
    assert "1 negative" in note and "1 undefined" in note and "1 duplicate" in note and "scaled" in note
    with pytest.raises(ValueError):
        clean_curve([1, 2], [0.0, -1.0])


def test_fetcher_rejects_html_unless_allowed():
    fetch = Fetcher(_Session({"u": b"<!DOCTYPE html><p>moved</p>"}))
    with pytest.raises(ValueError, match="HTML"):
        fetch.get("u")
    assert fetch.get("u", allow_html=True).startswith(b"<!DOCTYPE")


# ---------------------------------------------------------------------------
# End to end: FilterLibrary.published rebuilds OGLE-IV from the original files
# ---------------------------------------------------------------------------
def _ogle_files():
    return {surveys.OGLE_FILT["V"]: fx("OGLE-IV-V-filter-transmission.dat"),
            surveys.OGLE_FILT["I"]: fx("OGLE-IV-I-filter-transmission.dat"),
            surveys.OGLE_QE: fx("OGLE-IV-QE-curve-from-specification.dat")}


def test_published_rebuilds_from_original_sources_when_njm_is_down(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr(filter_library.NJMFilterGrabber, "_check_availability", lambda self: False)
    session = _Session(_ogle_files())
    monkeypatch.setattr(requests, "Session", lambda: session)
    library = FilterLibrary(tmp_path)
    library.svo = None  # SVO must not be needed
    directory = library.published("OGLE", "OGLE-IV")
    assert directory == tmp_path / "OGLE" / "OGLE-IV"
    assert FilterLibrary.published_complete(directory)
    assert (directory / "OGLE-IV").read_text() == "I.dat\nV.dat\n"
    header = read_filter_header(directory / "V.dat")
    assert header["FilterID"] == "OGLE/OGLE-IV.V"
    assert header["Components"] == "filter+qe" and header["ComponentsSource"] == "original"
    assert header["DetectorType"] == "photon" and header["DetectorTypeSource"] == "original"
    manifest = InstrumentManifest.load(directory)
    assert manifest.ground_based is True and manifest.outside_range == "zero"
    assert set(manifest.components) == {"filter", "qe"}
    assert manifest.sources["published"] == surveys.OGLE_FILT["V"]
    # published = filter x QE interpolated (0 outside the QE range)
    raw = np.loadtxt(FIX / "OGLE-IV-V-filter-transmission.dat", usecols=(0, 1))
    qe = np.loadtxt(FIX / "OGLE-IV-QE-curve-from-specification.dat")
    data = np.loadtxt(directory / "V.dat", delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, 0], raw[:, 0])
    np.testing.assert_allclose(data[:, 1], np.maximum(raw[:, 1], 0) *
                               np.interp(raw[:, 0], qe[:, 0], qe[:, 1], left=0, right=0), rtol=1e-5)
    # A second call is served locally.
    session.calls.clear()
    assert library.published("OGLE", "OGLE-IV") == directory and not session.calls


def test_builder_keeps_existing_published_curves(tmp_path):
    directory = tmp_path / "OGLE" / "OGLE-IV"
    directory.mkdir(parents=True)
    (directory / "V.dat").write_text("Wavelength,Transmission\n4000,0.5\n5000,0.5\n")
    surveys.build(tmp_path, instrument="OGLE-IV", session=_Session(_ogle_files()))
    assert (directory / "V.dat").read_text() == "Wavelength,Transmission\n4000,0.5\n5000,0.5\n"
    assert (directory / "I.dat").is_file() and (directory / "components" / "qe" / "qe.dat").is_file()


def test_failed_download_writes_nothing(tmp_path):
    files = _ogle_files()
    del files[surveys.OGLE_QE]
    with pytest.raises(Exception):
        surveys.build(tmp_path, instrument="OGLE-IV", session=_Session(files))
    assert not (tmp_path / "OGLE").exists()


def test_ogle_header_is_checked(tmp_path):
    files = _ogle_files()
    files[surveys.OGLE_FILT["V"]] = files[surveys.OGLE_FILT["I"]]
    with pytest.raises(ValueError, match="unexpected header"):
        surveys.build(tmp_path, instrument="OGLE-IV", session=_Session(files))


# ---------------------------------------------------------------------------
# Survey parsers
# ---------------------------------------------------------------------------
def test_atlas_table2_from_eprint_tar():
    raw = _tar({"atlas_defn.tex": fx("atlas_defn.tex"), "fig.eps": "x"})
    curves = surveys.atlas_table2(surveys.atlas_tex(raw))
    assert curves["c"].shape == curves["o"].shape == (9, 2)
    assert curves["c"][1].tolist() == [520.0, 0.699]
    assert curves["c"][0].tolist() == [425.0, 0.509] and curves["o"][0].tolist() == [575.0, 0.707]


def test_ps1_fixed_width_tables():
    wl, cols = surveys.ps1_table3(fx("ps1_table3.dat"))
    assert wl.tolist() == [500.0, 620.0, 750.0] and set(cols) == {"g", "r", "i", "z", "y", "w"}
    assert cols["g"][0] > 0 and cols["i"][2] > 0
    assert surveys.ps1_table4(fx("ps1_table4.dat")) == {"g": 4810.0, "r": 6170.0}


def test_ukirt_tar_members_refuse_unsafe_names():
    member = fx("ukirt_Table02_online.dat")
    text = surveys.tar_members(_tar({"Table02_online.dat": member}), {"Table02_online.dat"})
    assert "WFCAM Z filter" in text["Table02_online.dat"]
    with pytest.raises(ValueError, match="refusing"):
        surveys.tar_members(_tar({"../evil": "x", "Table02_online.dat": member}), {"Table02_online.dat"})
    with pytest.raises(ValueError, match="lacks"):
        surveys.tar_members(_tar({"other.dat": member}), {"Table02_online.dat"})


def test_lco_zip_member_and_checksum():
    raw = _zip({"SDSS.up.txt": fx("lco_SDSS.up.txt")})
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        a = surveys.lco_member(zf, "SDSS.up.txt")
    assert a.shape == (5, 2) and a[0, 1] < 0  # "\lambda [nm]" header skipped
    with pytest.raises(ValueError, match="differs from the verified copy"):
        surveys.sinistro(Fetcher(_Session({surveys.LCO_URL: raw})))


# ---------------------------------------------------------------------------
# Space parsers
# ---------------------------------------------------------------------------
def test_gaia_sentinel_is_dropped():
    raw = _zip({"GaiaDR2_Passbands.dat": fx("GaiaDR2_Passbands.dat")})
    a = space.gaia_passbands(raw, "GaiaDR2_Passbands.dat")
    assert a.shape == (4, 7) and a[0, 1] == 99.99
    b = make_band("G", a[:, 0], a[:, 1], unit="nm", detector_type="photon", components="filter+optics+qe",
                  sources="x", sentinel=99.99)
    assert len(b.wavelength) == 3 and b.wavelength[0] == a[1, 0] * 10
    with pytest.raises(ValueError):
        space.gaia_passbands(raw, "../GaiaDR2_Passbands.dat")


def test_herschel_and_planck_fits(tmp_path):
    import gzip

    from astropy.io import fits

    def table(cols, name=None, **header):
        hdu = fits.BinTableHDU.from_columns([fits.Column(name=n, format="D", unit=u, array=np.array(v))
                                             for n, u, v in cols], name=name)
        hdu.header.update(header)
        return fits.HDUList([fits.PrimaryHDU(), hdu])

    def raw(hdul):
        buf = io.BytesIO()
        hdul.writeto(buf)
        return buf.getvalue()

    pacs = raw(table([("wavelength", "um", [60.0, 70.0]), ("transmission", "", [0.2, 0.5])], BAND="BS"))
    wl, tr = space.pacs_curve(gzip.compress(pacs), "BS")
    assert wl.tolist() == [60.0, 70.0] and tr.tolist() == [0.2, 0.5]
    with pytest.raises(ValueError, match="BAND"):
        space.pacs_curve(pacs, "R")
    spire = raw(table([("frequency", "GHz", [1000.0, 1200.0]), ("psw", "", [0.5, 1.0]),
                       ("pmw", "", [0.0, 0.1]), ("plw", "", [0.0, 0.0])], name="RSRF"))
    nu, cols = space.spire_rsrf(spire)
    assert nu.tolist() == [1000.0, 1200.0] and cols["psw"].tolist() == [0.5, 1.0]
    rimo = table([("WAVENUMBER", "1/cm", [0.5, 2.0, 3.0]), ("TRANSMISSION", "", [1e-8, 0.5, 1.0])],
                 name="BANDPASS_F100")
    x, t = space.planck_bandpass(rimo, "BANDPASS_F100", "1/cm")
    assert x.tolist() == [0.5, 2.0, 3.0]
    with pytest.raises(ValueError):
        space.planck_bandpass(rimo, "BANDPASS_F100", "GHz")


def test_hipparcos_fixed_width_columns_end_independently():
    curves = space.fixed_width_pairs(fx("hip_table2.dat"), space.HIP_COLS)
    assert len(curves["Hp"][0]) == 5 and len(curves["BT"][0]) == 3 and len(curves["VT"][0]) == 3


def test_iras_html_table():
    tables = space.iras_tables(fx("iras_tabC5.html"))
    assert sorted(tables) == ["12", "60"]
    assert tables["12"].tolist() == [[7.0, 0.0, 0.3, 0.0], [7.5, 0.01, 0.35, 0.008]]
    assert tables["60"].shape == (2, 4)


def test_msx_ipac_table():
    a = space.msx_table(fx("msx_rsr_a.tbl"))
    assert a.shape == (4, 2) and a[0].tolist() == [5.8, 0.0]
    with pytest.raises(ValueError):
        space.msx_table(fx("msx_rsr_a.tbl").replace("microns", "nm"))


def test_spitzer_tables():
    cols = space.mips_columns(fx("MIPSfiltsumm.txt"))
    assert len(cols["24"][0]) == 6 and len(cols["70"][0]) == 4 and len(cols["160"][0]) == 6
    assert cols["70"][1][0] == -0.00029 and cols["160"][0][-1] == 116.4018
    wl, tr = space.two_col(fx("irac_ch1.txt"), "#")
    assert len(wl) == 3 and 3.0 < wl[0] < 4.5
    wl, tr = space.two_col(fx("irs_bluePUtrans.txt"), ";")
    assert len(wl) == 3


def test_xmm_effective_areas_are_normalised():
    session = _Session({space.XMM_URL: fx("OM_response.txt")})
    inst = space.xmm_om(Fetcher(session))
    assert [b.name for b in inst.bands] == ["V", "B", "U", "UVW1", "UVM2", "UVW2"]
    assert all(b.transmission.max() == 1.0 for b in inst.bands)


# ---------------------------------------------------------------------------
# Observatory parsers
# ---------------------------------------------------------------------------
def test_fors_percent_scan_is_cut_at_1100_nm():
    wl, tr = observatories.fors_curve("u_HIGH", "u_HIGH.txt", fx("fors_u_HIGH.txt"), "nm", True)
    assert wl.max() <= 1100 and len(wl) == 3
    np.testing.assert_allclose(tr, [0.8344275665, 0.8359967804, 0.838660965])


def test_omegacam_centre_scan():
    wl, tr = observatories.ocam_centre(fx("ocam_u_with_ccd.txt"))
    assert wl.tolist() == [479.916, 482.905, 485.894] and tr[0] == 0.000042
    with pytest.raises(ValueError, match="centre"):
        observatories.ocam_centre(fx("ocam_u_with_ccd.txt").replace("# x_pos 0.000000", "# x_pos 2.0"))


class _Sheet:
    def __init__(self, rows):
        self.rows, self.nrows = rows, len(rows)

    def row_values(self, r):
        return self.rows[r]


class _Book:
    def __init__(self, sheets):
        self.sheets = sheets

    def sheet_by_name(self, name):
        return self.sheets[name]

    def sheet_by_index(self, i):
        return list(self.sheets.values())[i]


def test_keck_xls_sheets():
    a = observatories.nirc2_sheet(_Book({"J": _Sheet([["Wavelength", "Transmission"], [1.1, 1.0],
                                                      [1.2, 90.0], [1.3, 2.0]])}), "J")
    assert a[:, 1].max() == 90.0
    with pytest.raises(ValueError, match="percent"):
        observatories.nirc2_sheet(_Book({"J": _Sheet([["Wavelength", "Transmission"], [1.1, 0.9]])}), "J")
    rows = [["W.M. Keck Observatory test report"]] + [[""] * 2] * 5 + [["", "center"]] + [[""] * 2] * 3
    rows += [["wl(nm)", "T%"], [700.0, 10.0], [760.0, 90.0]]
    np.testing.assert_allclose(observatories.lris_i_sheet(_Book({"s": _Sheet(rows)})), [[7000, 0.1], [7600, 0.9]])


def test_keck_width_check():
    wl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    observatories.check_width("x", wl, np.array([0, 1, 1, 1, 0.0]), 3.0, 2.0, tol=0.01)
    with pytest.raises(ValueError, match="50% points"):
        observatories.check_width("x", wl, np.array([0, 1, 1, 1, 0.0]), 3.5, 2.0)


def test_deimos_synphot_table():
    wl, thr = observatories.deimos_table("B", fx("deimos_Bfaber13norm.txt"))
    assert len(wl) == 3 and thr.max() == 0.397678
    with pytest.raises(ValueError, match="PIVWV"):
        observatories.deimos_table("R", fx("deimos_Bfaber13norm.txt"))


def test_kpno_lambda9_scan():
    a = observatories.kpno_scan(fx("kpno_k1002bp_jul04.txt"), r"Filter: K1002\b")
    assert a.shape == (5, 2) and a[:, 1].max() == 69.37
    with pytest.raises(ValueError, match="does not match"):
        observatories.kpno_scan(fx("kpno_k1002bp_jul04.txt"), r"Filter: K1003\b")


def test_fourstar_rederivation():
    linked = np.array([[20000.0, 0.9], [22000.0, 0.8]])
    wl, tr = observatories.fourstar_rederive("Ks", linked)
    np.testing.assert_allclose(tr, [0.9 * (1.714 - 0.76), 0.8 * (1.714 - 0.836)])
    wl, tr = observatories.fourstar_rederive("FS209", np.array([[2090.0, 50.0]]))
    np.testing.assert_allclose(wl, [20900.0])
    np.testing.assert_allclose(tr, [0.5 * (1.714 - 0.38 * 2.09)])


def test_megacam_table():
    t = observatories.megacam_table(fx("megacam_filterthroughput.magg"))
    assert set(t) == set(observatories.MC_COLS) and len(t["WAVE"]) == 3
    np.testing.assert_allclose(t["T"] * t["QE"] * t["EXTINCT"] * t["THROUGHPUT"], t["TOTAL"], atol=1e-5)


def test_not_scan_header_and_scale():
    a = observatories.not_curve(109, fx("not_109.txt"), False, "u' SDSS")
    assert a.shape == (5, 2) and a[:, 1].max() < 1
    with pytest.raises(ValueError, match="does not identify"):
        observatories.not_curve(110, fx("not_109.txt"), False, "r' SDSS")
    with pytest.raises(ValueError, match="scale"):
        observatories.not_curve(109, fx("not_109.txt"), True, "u' SDSS")


def test_subaru_and_odi_tables():
    qe = observatories.subaru_table(fx("subaru_QE_Hamamatsu.txt"), 300, 1100, 0.2, 1.0, "qe")
    assert qe.shape == (5, 2)
    with pytest.raises(ValueError, match="peak"):
        observatories.subaru_table(fx("subaru_QE_Hamamatsu.txt"), 300, 1100, 0.8, 1.0, "qe")
    session = _Session({observatories.ODI + fn: fx("odi_NB_422.txt") for fn, _, _ in observatories.ODI_FILES.values()})
    with pytest.raises(ValueError, match="scale"):   # u/r/NB695 are fractions; this scan is percent
        observatories.odi(Fetcher(session))
    wl = np.array([4000.0, 9000.0, 11000.0, 12000.0])
    assert observatories.odi_leak(wl, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    assert 0.5 < observatories.odi_leak(wl, np.array([0.0, 0.0, 1.0, 1.0])) < 1.0


# ---------------------------------------------------------------------------
# Standard systems
# ---------------------------------------------------------------------------
def test_bessell_and_mko_parsers():
    curves = space.fixed_width_pairs(fx("bessell_table1.dat"), standards.B12_COLS)
    assert list(curves) == ["U", "B", "V", "R", "I"] and all(len(w) == 4 for w, _ in curves.values())
    data = standards.mko_scan(fx("nsfcam_lpmk_trans.dat"))
    b = make_band("Lp", data[:, 0], data[:, 1] / 100, unit="micron", detector_type="photon",
                  components="filter", sources="x")
    assert np.all(np.diff(b.wavelength) > 0) and b.transmission.min() == 0.0
    with pytest.raises(ValueError, match="unexpected header"):
        standards.mko_scan("WL\tT\n1 2\n")


def test_usno_reads_wayback_then_falls_back_to_live_file():
    text = fx("usno_u.res")
    assert standards.usno_table(text, "u").shape == (4, 6)
    with pytest.raises(ValueError, match="unexpected header"):
        standards.usno_table(text, "g")
    live = standards.USNO_ORIG.format("u")
    wb = f"https://web.archive.org/web/{standards.USNO_WB['u']}id_/{live}"
    session = _Session({live: text})
    assert standards._archived(Fetcher(session), wb, live) == text
    assert session.calls == [wb, live]
    session = _Session({})
    with pytest.raises(OSError):
        standards._archived(Fetcher(session), wb, live)


def test_naco_band_names_have_no_inner_dot():
    from sed_tools.component_sources.tabular import NACO

    assert "IB_2p00" in NACO.filters and "NB_3p74" in NACO.filters
    assert all("." not in name for name in NACO.filters)
    assert NACO.filters["IB_2p00"].url.endswith("/IB_2.00.dat")
