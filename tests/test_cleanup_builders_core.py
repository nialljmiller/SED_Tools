"""Builders that used to skip a band or file and still report success now stop.

Offline: every download goes through a fake session or client.
"""

from __future__ import annotations

import numpy as np
import pytest

from sed_tools.component_sources._common import SimpleSource, Table, build_simple, ensure_published
from sed_tools.filter_metadata import PHOTON, SVOFilterRecord, write_filter_file


class _Resp:
    def __init__(self, content, status=200):
        self.content = content if isinstance(content, bytes) else content.encode()
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


class _SVO:
    """SVO metadata client that knows only ``records``."""

    def __init__(self, records):
        self.records = records

    def get_filter(self, fid):
        return self.records.get(fid)


def _record(fid):
    return SVOFilterRecord(fid, PHOTON, np.array([4000.0, 5000.0]), np.array([0.5, 0.6]),
                           wavelength_unit="Angstrom", components="filter")


def _published(inst, band):
    inst.mkdir(parents=True, exist_ok=True)
    write_filter_file(inst / f"{band}.dat", [4000.0, 5000.0], [0.5, 0.6], detector_type=PHOTON,
                      source="svo", wavelength_unit="Angstrom", extra={"Components": "filter"})


# ---------------------------------------------------------------------------
# _common.ensure_published / build_simple
# ---------------------------------------------------------------------------
def test_ensure_published_raises_for_ids_svo_lacks_and_writes_no_index(tmp_path):
    inst = tmp_path / "Fac" / "Cam"
    client = _SVO({"Fac/Cam.g": _record("Fac/Cam.g")})
    with pytest.raises(RuntimeError, match="SVO has no Fac/Cam.r"):
        ensure_published(inst, {"g": "Fac/Cam.g", "r": "Fac/Cam.r"}, client)
    assert (inst / "g.dat").is_file() and not (inst / "r.dat").exists()
    assert not (inst / "Cam").exists()  # no MESA index for an incomplete set


def test_ensure_published_writes_index_when_complete(tmp_path):
    inst = tmp_path / "Fac" / "Cam"
    ensure_published(inst, {"g": "Fac/Cam.g"}, _SVO({"Fac/Cam.g": _record("Fac/Cam.g")}))
    assert (inst / "Cam").read_text() == "g.dat\n"


def test_build_simple_refuses_a_filter_band_without_a_published_curve(tmp_path):
    inst = tmp_path / "Fac" / "Cam"
    _published(inst, "g")
    spec = SimpleSource("Fac", "Cam", filters={"g": Table("f"), "r": Table("f")})
    with pytest.raises(FileNotFoundError, match="no published curve for r"):
        build_simple(spec, tmp_path, session=_Session({"f": "4000 0.5\n5000 0.8\n"}), svo_client=_SVO({}))
    assert not (inst / "manifest.json").exists() and not (inst / "components").exists()


def test_build_simple_refuses_a_composite_band_without_a_published_curve(tmp_path):
    inst = tmp_path / "Fac" / "Cam"
    _published(inst, "g")

    def make(session):
        return np.array([4000.0, 5000.0]), np.array([0.5, 0.5])

    spec = SimpleSource("Fac", "Cam", filters={}, composites={"z": ("filter+qe", make)})
    with pytest.raises(FileNotFoundError, match="no published curve for z"):
        build_simple(spec, tmp_path, session=_Session({}), svo_client=_SVO({}))
    assert not (inst / "manifest.json").exists()


# ---------------------------------------------------------------------------
# hst.build
# ---------------------------------------------------------------------------
def _fits_table(wl, tr):
    import io

    from astropy.io import fits

    cols = [fits.Column(name="WAVELENGTH", format="D", array=np.asarray(wl, float)),
            fits.Column(name="THROUGHPUT", format="D", array=np.asarray(tr, float))]
    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(buf)
    return buf.getvalue()


def _hst_tables():
    """ACS/SBC graph: F122M splits into ota x filter x mama; FR388N has no chain."""
    from sed_tools.component_sources import hst

    t = hst.SynphotTables.__new__(hst.SynphotTables)
    t.tmg_name, t.tmc_name = "x_tmg.fits", "x_tmc.fits"
    t.innode, t.outnode = np.array([1, 2, 3, 3]), np.array([2, 3, 4, 4])
    t.keyword = np.array(["acs", "sbc", "f122m", "default"])
    t.compname = np.array(["hst_ota", "acs_sbc_mama", "acs_f122m", "clear"])
    names = ("hst_ota", "acs_sbc_mama", "acs_f122m")
    t.files = {n: f"acscomp${n}_001_syn.fits" for n in names}
    wl = np.linspace(1000.0, 3000.0, 21)
    t._get = {f"{hst.TRDS}/comp/acs/{n}_001_syn.fits": _fits_table(wl, np.full(21, 0.5)) for n in names}.__getitem__
    t._cache = {}
    return t


def _hst_published(root, bands=("F122M", "FR388N")):
    inst = root / "HST" / "ACS_SBC"
    for band in bands:
        _published(inst, band)
    return inst


def test_hst_build_warns_about_bands_it_cannot_decompose(tmp_path, caplog):
    from sed_tools.component_sources import hst
    from sed_tools.filter_components import InstrumentManifest

    inst = _hst_published(tmp_path)
    with caplog.at_level("WARNING", logger="sed_tools.component_sources.hst"):
        assert hst.build(tmp_path, instruments=["ACS_SBC"], tables=_hst_tables()) == [inst]
    assert "1 bands not decomposed" in caplog.text and "FR388N" in caplog.text
    manifest = InstrumentManifest.load(inst)
    assert "Not decomposed: FR388N." in manifest.notes
    assert set(manifest.components["filter"]) == {"F122M"}


def test_hst_build_raises_after_building_the_others_for_instruments_without_curves(tmp_path):
    from sed_tools.component_sources import hst

    inst = _hst_published(tmp_path)
    with pytest.raises(FileNotFoundError, match="no published curves for ACS_WFC.*Built: ACS_SBC"):
        hst.build(tmp_path, instruments=["ACS_SBC", "ACS_WFC"], tables=_hst_tables())
    assert (inst / "manifest.json").is_file()
    assert not (tmp_path / "HST" / "ACS_WFC").exists()  # no empty directory left behind


def test_hst_build_refuses_an_unreadable_published_curve(tmp_path):
    from sed_tools.component_sources import hst

    inst = _hst_published(tmp_path, bands=())
    inst.mkdir(parents=True)
    (inst / "F122M.dat").write_text("Wavelength,Transmission\n1500,0.1\nbad,row\n")
    with pytest.raises(ValueError, match="F122M.dat: cannot read the published curve"):
        hst.build(tmp_path, instruments=["ACS_SBC"], tables=_hst_tables())


# ---------------------------------------------------------------------------
# jwst.build
# ---------------------------------------------------------------------------
def _pandeia_raw(root):
    """Unpacked Pandeia reference data with the NIRCam F200W imaging factors."""
    import json

    from astropy.io import fits

    def table(path, wl, tr):
        cols = [fits.Column(name="wavelength", format="D", array=wl, unit="microns"),
                fits.Column(name="throughput", format="D", array=tr)]
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)

    wl = np.linspace(1.0, 3.0, 11)
    (root / "jwst" / "telescope").mkdir(parents=True)
    (root / "jwst" / "telescope" / "config.json").write_text(json.dumps({"paths": {"otefile": "ote.fits"}}))
    table(root / "jwst" / "telescope" / "ote.fits", wl, np.full(11, 0.9))
    paths = {"f200w": "filters/f200w.fits", "internal": "optical/internal.fits",
             "dbs_sw": "optical/dbs_sw.fits", "qe_sw": "qe/qe_sw.fits"}
    (root / "jwst" / "nircam").mkdir(parents=True)
    (root / "jwst" / "nircam" / "config.json").write_text(json.dumps({"paths": paths}))
    for i, rel in enumerate(paths.values()):
        table(root / "jwst" / "nircam" / rel, wl[i:], np.linspace(0.5, 0.8, 11 - i))
    return root


def test_jwst_build_raises_after_building_the_others_for_instruments_without_curves(tmp_path):
    from sed_tools.component_sources import jwst
    from sed_tools.filter_components import InstrumentManifest

    raw = _pandeia_raw(tmp_path / "pandeia")
    inst = tmp_path / "filters" / "JWST" / "NIRCam"
    _published(inst, "F200W")
    with pytest.raises(FileNotFoundError, match="no published curves for MIRI.*Built: NIRCam"):
        jwst.build(tmp_path / "filters", raw=raw, instruments=["NIRCam", "MIRI"])
    assert set(InstrumentManifest.load(inst).components) == {"filter", "optics", "qe"}


# ---------------------------------------------------------------------------
# tabular.build with a saved research inventory
# ---------------------------------------------------------------------------
def test_tabular_raw_inventory_warns_when_a_file_is_downloaded_instead(tmp_path, caplog):
    from sed_tools.component_sources import tabular

    folder = tmp_path / "raw" / tabular.RAW_GROUPS["PANIC"]
    folder.mkdir(parents=True)
    scan = b"400 50\n500 80\n"
    *saved, (band, missing) = tabular.PANIC_FILES.items()
    for _, filename in saved:
        (folder / filename).write_bytes(scan)
    url = tabular.PANIC.filters[band].url
    session = _Session({url: scan})
    with caplog.at_level("WARNING", logger="sed_tools.component_sources.tabular"):
        tabular.build(tmp_path / "filters", instrument="PANIC", raw=tmp_path / "raw", session=session)
    assert session.calls == [url]
    assert f"{missing} is not in the saved inventory" in caplog.text
    assert (tmp_path / "filters" / "CAHA" / "PANIC" / f"{band}.dat").is_file()
