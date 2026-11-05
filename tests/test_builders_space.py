"""The space missions, each built end to end from the archive's saved files (tests/fixtures/builders)."""

from __future__ import annotations

import hashlib
import io
import tarfile

import numpy as np
import pytest

from sed_tools.component_sources import catalog, space
from sed_tools.filter_components import FILTER, OPTICS, QE, SHARED, InstrumentManifest
from sed_tools.filter_metadata import PHOTON, released_mesa_reads, write_filter_file

from builder_fixtures import Session, check_published, curve, saved

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


@pytest.mark.parametrize("instrument, corrupt, message", [
    ("IRAS", lambda f: f.__setitem__(space.IRAS_URL, f[space.IRAS_URL].replace(b"100 &nbsp;&micro;m Band", b"")),
     "IRAS: unexpected bands"),
])
def test_a_wrong_file_stops_the_build_before_anything_is_written(tmp_path, instrument, corrupt, message):
    files = saved()
    corrupt(files)
    with pytest.raises(ValueError, match=message):
        space.build(tmp_path, instrument=instrument, session=Session(files))
    assert not any(tmp_path.iterdir())


class _Resp:
    def __init__(self, content, status=200):
        self.content = content if isinstance(content, bytes) else content.encode()
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self, files):
        self.files = files
        self.headers = {}

    def get(self, url, timeout=None, **kw):
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


def _published(inst, band, wl, tr, components="filter+optics+qe"):
    inst.mkdir(parents=True, exist_ok=True)
    write_filter_file(inst / f"{band}.dat", wl, tr, detector_type=PHOTON, source="svo",
                      wavelength_unit="Angstrom", extra={"Components": components})


# ---------------------------------------------------------------------------
# Euclid NISP
# ---------------------------------------------------------------------------
def _nisp_table(neg=False):
    wave = np.arange(1000.0, 1011.0)
    filt = np.full(wave.size, 0.9)
    filt[0] = -1e-5 if neg else 1e-5
    tel, nioa, qe = np.full(wave.size, 0.88), np.full(wave.size, 0.95), np.full(wave.size, 0.93)
    total = filt * tel * nioa * qe
    rows = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in zip(wave, total, filt, tel, nioa, qe))
    return "# WAVE T_TOTAL T_FILTER T_TELESCOPE T_NI-OA T_QE\n" + rows + "\n", wave, total


def test_euclid_parse_checks_columns_and_product():
    from sed_tools.component_sources import euclid

    text, wave, total = _nisp_table(neg=True)
    cols = euclid.parse_table(text)
    wl, curves, negative = euclid.factors(cols)
    np.testing.assert_allclose(wl, wave * 10.0)
    assert negative == 1 and curves[FILTER].min() == 0.0
    np.testing.assert_allclose(curves[OPTICS], 0.88 * 0.95)
    with pytest.raises(ValueError, match="columns"):
        euclid.parse_table(text.replace("T_NI-OA", "T_NIOA"))
    bad = text.splitlines()
    bad[3] = bad[3].replace(bad[3].split()[1], "0.5", 1)
    with pytest.raises(ValueError, match="multiply"):
        euclid.parse_table("\n".join(bad))


def test_euclid_build_writes_components_and_compares(tmp_path):
    from sed_tools.component_sources import euclid

    text, wave, total = _nisp_table()
    inst = tmp_path / "Euclid" / "NISP"
    _published(inst, "Y", wave * 10.0, total * 0.6)
    _published(inst, "Blue", wave * 10.0, total, components="filter+instrument")
    (inst / "NISP").write_text("Blue.dat\nY.dat\n")  # as NJM serves the set
    session = _Session({euclid.URL.format(band="Y"): text})
    euclid.build(tmp_path, bands=["Y"], session=session)
    m = InstrumentManifest.load(inst)
    assert {r: m.components[r] for r in (FILTER, OPTICS, QE)} == {
        r: {"Y": f"components/{r}/Y.dat"} for r in (FILTER, OPTICS, QE)}
    assert "Release mismatch" in m.notes and "Blue" in m.notes
    assert all(released_mesa_reads(inst / rel) for rel in (m.components[r]["Y"] for r in (FILTER, OPTICS, QE)))


def test_euclid_build_writes_missing_published_curves(tmp_path):
    from sed_tools.component_sources import euclid
    from sed_tools.filter_metadata import read_filter_header

    text, wave, total = _nisp_table()
    lines = text.splitlines()
    lines[1] = lines[1].replace(lines[1].split()[1], "-1e-07", 1).replace(lines[1].split()[2], "-1e-07", 1)
    text = "\n".join(lines) + "\n"
    inst = tmp_path / "Euclid" / "NISP"
    _published(inst, "Y_svo", wave * 10.0, total * 0.6)
    # NJM keeps SVO's copy beside the set, outside its bands and index file.
    archived = {"Y_svo": {"detector_type": "photon", "detector_type_source": "svo", "published": "filter+optics+qe"}}
    InstrumentManifest("Euclid", "NISP", bands=[], archived_bands=archived).save(inst)
    euclid.build(tmp_path, bands=["Y", "Blue"], session=_Session({euclid.URL.format(band="Y"): text}))
    header = read_filter_header(inst / "Y.dat")
    assert header["FilterID"] == "Euclid/NISP.Y" and header["DetectorType"] == "photon"
    assert header["DetectorTypeSource"] == "original" and header["Components"] == "filter+optics+qe"
    assert released_mesa_reads(inst / "Y.dat")
    assert (inst / "NISP").read_text() == "Y.dat\n"
    m = InstrumentManifest.load(inst)
    assert m.bands == ["Y"] and "reproduces" in m.notes and "Y_svo are SVO's copies" in m.notes
    assert m.archived_bands == archived  # the archive copy never goes back into bands
    assert "negative samples set to 0: Y 1" in m.notes
    with pytest.raises(ValueError, match="photometric band"):  # grisms have no ESA factors
        euclid.build(tmp_path, bands=["Blue"], session=_Session({}))


# ---------------------------------------------------------------------------
# Roman WFI
# ---------------------------------------------------------------------------
def _roman_qe(values):
    head = ["# %ECSV 1.0", "# ---", "# datatype:",
            "# - {name: wavelength, unit: nm, datatype: float64, description: wavelength}"]
    head += [f"# - {{name: SCU {i}, unit: '', datatype: float64, description: Sensor Control Unit Number {i}}}"
             for i in range(1, 19)]
    head += ["# delimiter: ','", "# schema: astropy-2.0",
             "wavelength," + ",".join(f"SCU {i}" for i in range(1, 19))]
    wl = [800.0, 1000.0, 2000.0]
    rows = [f"{w}," + ",".join(f"{v:.4f}" for v in values[k]) for k, w in enumerate(wl)]
    return "\n".join(head + rows) + "\n"


def _roman_summary(below, above):
    return ("# %ECSV 1.0\n# ---\nwavelength range, mean quantum efficiency, median quantum efficiency\n"
            f"<900, {below}, {below}\n900 <= wave <= 2200, {above}, {above}\n")


def test_roman_qe_is_the_detector_mean(tmp_path):
    from sed_tools.component_sources import roman

    rng = np.random.default_rng(1)
    values = np.round(np.clip(np.array([0.7, 0.88, 0.9])[:, None] + rng.normal(0, 0.02, (3, 18)), 0, 1), 4)
    qe_text = _roman_qe(values)
    wl, qe = roman.parse_qe(qe_text)
    assert qe.shape == (18, 3)
    roman.check_summary(_roman_summary(values[0].mean(), values[1:].mean()), wl, qe)
    with pytest.raises(ValueError, match="summary"):
        roman.check_summary(_roman_summary(0.5, 0.5), wl, qe)
    with pytest.raises(ValueError, match="unit"):
        roman.parse_qe(qe_text.replace("unit: nm", "unit: um"))

    inst = tmp_path / "Roman" / "WFI"
    _published(inst, "F062", np.array([4720.0, 6000.0, 7700.0]), np.array([0.1, 0.5, 0.1]))
    session = _Session({roman.QE_URL: qe_text,
                        roman.SUMMARY_URL: _roman_summary(values[0].mean(), values[1:].mean())})
    roman.build(tmp_path, session=session)
    m = InstrumentManifest.load(inst)
    assert m.components == {QE: {SHARED: "components/qe/qe.dat"}}
    got = np.loadtxt(inst / "components/qe/qe.dat", delimiter=",", skiprows=1)
    np.testing.assert_allclose(got[:, 0], [8000.0, 10000.0, 20000.0])
    np.testing.assert_allclose(got[:, 1], values.mean(1))
    assert "F062" in m.notes and "mean of the 18 detectors" in m.notes and m.buildable() == []


# ---------------------------------------------------------------------------
# HST: pinned component releases
# ---------------------------------------------------------------------------
def _fits_table(wl, tr):
    from astropy.io import fits

    cols = [fits.Column(name="WAVELENGTH", format="D", array=wl),
            fits.Column(name="THROUGHPUT", format="D", array=tr)]
    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(buf)
    return buf.getvalue()


def test_hst_pins_replace_only_the_named_component():
    from sed_tools.component_sources import hst

    t = hst.SynphotTables.__new__(hst.SynphotTables)
    t.tmg_name = "x_tmg.fits"
    t.files = {"acs_f115lp_tds_def": "acscomp$acs_f115lp_tds_def_010_syn.fits",
               "acs_sbc_mama": "acscomp$acs_sbc_mama_013_syn.fits"}
    t._cache = {}
    wl = np.array([1150.0, 1500.0, 2000.0])
    blobs = {
        f"{hst.TRDS}/comp/acs/acs_f115lp_tds_def_010_syn.fits": _fits_table(wl, [0.875, 0.875, 0.875]),
        f"{hst.TRDS}/comp/acs/acs_f115lp_tds_def_004_syn.fits": _fits_table(wl, [0.909, 0.909, 0.909]),
        f"{hst.TRDS}/comp/acs/acs_sbc_mama_013_syn.fits": _fits_table(wl, [0.1, -0.01, 0.2]),
    }
    t._get = blobs.__getitem__
    pins = {"acs_f115lp_tds_def": "acs_f115lp_tds_def_004_syn.fits"}
    assert t.throughput("acs_f115lp_tds_def", pins)[1][0] == pytest.approx(0.909)
    assert t.throughput("acs_f115lp_tds_def")[1][0] == pytest.approx(0.875)
    assert t.throughput("acs_sbc_mama", pins)[1][1] == 0.0  # negative noise set to 0
    assert "pinned" in t.source("acs_f115lp_tds_def", pins) and "pinned" not in t.source("acs_sbc_mama", pins)
    note = hst._pin_note(t, pins, ["acs_f115lp_tds_def", "acs_sbc_mama"])
    assert note == "acs_f115lp_tds_def_010_syn.fits = 0.9626 x acs_f115lp_tds_def_004_syn.fits"
    _, prod = hst._product(t, ["acs_f115lp_tds_def", "acs_sbc_mama"], pins)
    assert prod[-1] == pytest.approx(0.909 * 0.2)


# ---------------------------------------------------------------------------
# JWST: current Pandeia reference data
# ---------------------------------------------------------------------------
def _pandeia_tarball():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in (("pandeia_data-2026.7-jwst/jwst/telescope/config.json", b"{}"),
                           ("pandeia_data-2026.7-jwst/jwst/nircam/psfs/big.fits", b"x"),
                           ("pandeia_data-2026.7-jwst/jwst/nircam/filters/f200w.fits", b"y"),
                           ("pandeia_data-2026.7-jwst/devtools/tool.py", b"z")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_jwst_refdata_is_checked_and_filtered(tmp_path):
    from sed_tools.component_sources import jwst

    assert jwst.REFDATA_VERSION == "2026.7" and "v2026p7" in jwst.REFDATA_URL
    blob = _pandeia_tarball()
    session = _Session({"u": blob})
    root = jwst.fetch_refdata(tmp_path, url="u", session=session, sha1=hashlib.sha1(blob).hexdigest())
    assert root == tmp_path / "pandeia_data-2026.7-jwst"
    assert (root / "jwst/nircam/filters/f200w.fits").is_file()
    assert not (root / "jwst/nircam/psfs/big.fits").exists() and not (root / "devtools").exists()
    with pytest.raises(RuntimeError, match="SHA-1"):
        jwst.fetch_refdata(tmp_path / "b", url="u", session=session)


def test_jwst_read_rejects_non_micron_wavelengths(tmp_path):
    from astropy.io import fits
    from sed_tools.component_sources.jwst import _read

    cols = [fits.Column(name="WAVELENGTH", format="D", array=[1.0, 2.0], unit="ANGSTROM"),
            fits.Column(name="THROUGHPUT", format="D", array=[0.5, 0.6])]
    path = tmp_path / "t.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)
    with pytest.raises(ValueError, match="microns"):
        _read(path)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def test_status_of_2mass_and_ps1_and_new_builders():
    info = catalog.component_availability("PAN-STARRS", "PS1")
    assert info["status"] == catalog.NOT_BUILT, info
    assert "Table 3" in catalog.status_for("PAN-STARRS", "PS1")[1]
    # 2MASS's separate tables are built since PR #48 (component_sources.twomass).
    info = catalog.component_availability("2MASS", "2MASS")
    assert info["status"] == "supported" and info["builder"] == "twomass", info
    info = catalog.component_availability("Roman", "WFI")
    assert info["status"] == "supported" and info["builder"] == "roman"
    info = catalog.component_availability("Euclid", "NISP")
    assert info["status"] == "supported" and info["builder"] == "euclid"
    # Pandeia 2026.7 is 21.6 MB: JWST components no longer need allow_heavy.
    assert not any(catalog.builder_for("JWST", i).heavy for i in ("NIRCam", "NIRISS", "MIRI"))
