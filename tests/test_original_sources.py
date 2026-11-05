"""Offline tests for the instruments rebuilt from their teams' original files.

Fixtures in tests/fixtures/original_sources are a few lines excerpted from each
original file (same layout and header); archives and FITS files are assembled
in memory from them.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from sed_tools import filter_library
from sed_tools.component_sources import (cassini, catalog, csp, observatories, space, standards,
                                         surveys, swift)
from sed_tools.component_sources._original import (Fetcher, clean_curve, make_band, to_angstrom)
from sed_tools.filter_components import InstrumentManifest
from sed_tools.filter_library import FilterLibrary
from sed_tools.filter_metadata import read_filter_header

from builder_fixtures import _tar

FIX = Path(__file__).parent / "fixtures" / "original_sources"
MODULES = {"surveys": surveys, "space": space, "observatories": observatories, "standards": standards,
           "swift": swift, "csp": csp, "cassini": cassini}


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


def _zip(members) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in members.items():
            z.writestr(name, text)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_original_sources_do_not_shadow_other_builders():
    original = {(f, i) for entries in catalog.ORIGINAL_SOURCES.values() for f, i, _ in entries}
    others = {k for k, ref in catalog.BUILDERS_FOR.items() if ref.name not in MODULES}
    assert not original & others
    assert len(others) >= 28  # the builders registered before these were added
    assert all(ref.publishes for ref in catalog.BUILDERS_FOR.values() if ref.name == "tabular")


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


def test_ogle_header_is_checked(tmp_path):
    files = _ogle_files()
    files[surveys.OGLE_FILT["V"]] = files[surveys.OGLE_FILT["I"]]
    with pytest.raises(ValueError, match="unexpected header"):
        surveys.build(tmp_path, instrument="OGLE-IV", session=_Session(files))


# ---------------------------------------------------------------------------
# Survey parsers
# ---------------------------------------------------------------------------
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


def test_iras_html_table():
    page = fx("iras_tabC5.html")
    with pytest.raises(ValueError, match="rows of 4"):  # the excerpt ends inside a row
        space.iras_tables(page)
    tables = space.iras_tables(page.replace("<TD>33.0</TD>", ""))
    assert sorted(tables) == ["12", "60"]
    assert tables["12"].tolist() == [[7.0, 0.0, 0.3, 0.0], [7.5, 0.01, 0.35, 0.008]]
    assert tables["60"].shape == (2, 4)


def test_msx_ipac_table():
    a = space.msx_table(fx("msx_rsr_a.tbl"))
    assert a.shape == (4, 2) and a[0].tolist() == [5.8, 0.0]
    with pytest.raises(ValueError):
        space.msx_table(fx("msx_rsr_a.tbl").replace("microns", "nm"))


# ---------------------------------------------------------------------------
# Observatory parsers
# ---------------------------------------------------------------------------
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


def test_usno_reads_only_the_wayback_snapshots(tmp_path):
    text = fx("usno_u.res")
    assert standards.usno_table(text, "u").shape == (4, 6)
    with pytest.raises(ValueError, match="unexpected header"):
        standards.usno_table(text, "g")
    # The curves cite the snapshots, so an unreachable archive stops the build
    # instead of silently reading the live files.
    session = _Session({standards.USNO_README: "readme", standards.USNO_ORIG.format("u"): text})
    with pytest.raises(OSError):
        standards.build(tmp_path, instrument="USNO", session=session)
    assert session.calls == [standards.USNO_README_WB]
    assert not (tmp_path / "Generic").exists()


def test_naco_band_names_have_no_inner_dot():
    from sed_tools.component_sources.tabular import NACO

    assert "IB_2p00" in NACO.filters and "NB_3p74" in NACO.filters
    assert all("." not in name for name in NACO.filters)
    assert NACO.filters["IB_2p00"].url.endswith("/IB_2.00.dat")
