"""Builders for teams' tabulated factors, team atmospheres and the checks against teams' totals.

DECam, VISTA, the team atmosphere sites and the HST synphot tables read saved files
(tests/fixtures/builders); Cassini, LSST, 2MASS and SkyCalc reuse their suites' fakes.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from sed_tools.component_sources import (cassini, decam, hst, lsst, skycalc, surveys, team_atmospheres as team,
                                         team_atmospheres, twomass, vista)
from sed_tools.component_sources._common import DerivedCurve, write_site
from sed_tools.component_sources._original import Fetcher
from sed_tools.filter_components import (DERIVED, FILTER, OPTICS, QE, SHARED, TEAM, AtmosphereGrid, InstrumentManifest,
                                         compose_band)
from sed_tools.filter_metadata import SVOFilterRecord, read_filter_header, released_mesa_reads

import helpers_builders_cfht_lsst as t_lsst
import helpers_builders_swift_cassini as t_cassini
from builder_fixtures import Session, curve, saved
from helpers_builders_sites import _FormSession, _Resp, _curve, _site


# ---------------------------------------------------------------------------- Cassini
@pytest.mark.parametrize("camera", sorted(cassini.CAMERAS))
def test_cassini_recipes_describe_each_camera(camera):
    files, _nm, _pairs = t_cassini._cassini_files(camera)
    recipe = cassini.RECIPES[camera][1]
    spec = recipe(Fetcher(t_cassini._Session(files)))
    assert (spec.facility, spec.instrument, spec.ground_based) == ("Cassini", camera, False)
    assert spec.outside_range == "error"
    assert [b.name for b in spec.bands] == cassini.BANDS[camera]
    assert all(b.components == "filter+optics+qe" and b.detector_type == "photon" for b in spec.bands)
    roles = [c.role for c in spec.components]
    assert roles[-2:] == ["qe", "optics"] and set(roles[:-2]) == {"filter"}
    assert f"Cassini ISS {camera[-3:]} from the PDS" in spec.notes


# ----------------------------------------------------------------------------- DECam
class _SVO:
    """SVO's DECam curves: the DES band columns, and flat filter-only siblings."""

    def __init__(self, data):
        self.data, self.asked = data, []

    def get_filter(self, fid):
        self.asked.append(fid)
        band = fid.rsplit(".", 1)[1]
        wl = np.asarray(self.data["LAMBDA"], float)
        if band.endswith("_filter"):
            tr = np.where(np.asarray(self.data[band[0]]) > 0, 0.9, 0.0)
            return SVOFilterRecord(fid, "photon", wl, tr, "Angstrom", "filter")
        return SVOFilterRecord(fid, "photon", wl, np.asarray(self.data[band], float), "Angstrom",
                               "filter+instrument+atmosphere")


def _des():
    from astropy.io import fits

    return fits.open(io.BytesIO(saved()[decam.DES_STD]))[1].data


def test_decam_instrument_response_is_the_des_band_over_its_atmosphere(tmp_path):
    svo = _SVO(_des())
    inst = decam.build(tmp_path / "filters", session=Session(), svo_client=svo,
                       atmosphere_root=tmp_path / "atmospheres")
    assert sorted(svo.asked) == sorted(decam.SPEC.svo_ids.values())
    manifest = InstrumentManifest.load(inst)
    assert manifest.default_site == "ctio_des" and manifest.outside_range == "error"
    assert sorted(manifest.components[decam.RECIPE]) == sorted(decam.BANDS)
    grid = AtmosphereGrid.load(tmp_path / "atmospheres" / "ctio_des")
    assert [(e["airmass"], e["file"]) for e in grid.entries] == [(1.2, "am1.20.dat")]
    for band in decam.BANDS:
        composed = compose_band(inst, band, decam.RECIPE, atmosphere={"airmass": 1.2},
                                atmosphere_root=tmp_path / "atmospheres")
        wl, published = curve(inst / f"{band}.dat")
        np.testing.assert_allclose(np.interp(wl, composed.wavelength, composed.transmission), published,
                                   rtol=1e-6, atol=1e-7)
    # The quotient exceeds 1 in i and z (see the module docstring), as in the full table; the notes say so.
    assert manifest.notes.startswith(
        decam.SPEC.notes + " Flag: the instrument-only quotient exceeds 1 in i peaks at 1.021746 at 8500 A "
        "(118 samples above 1 between 7275 and 8510 A); z peaks at 1.036993 at 9060 A (126 samples above 1 "
        "between 8635 and 9260 A). A physical throughput cannot")


# ------------------------------------------------------------------------------ VISTA
class _VistaSVO:
    def get_filter(self, fid):
        wl = np.linspace(8000.0, 25000.0, 50)
        return SVOFilterRecord(fid, "photon", wl, np.full(wl.size, 0.5), "Angstrom", "filter+instrument")


def test_vista_components_come_from_the_eso_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(vista, "_CACHE", {})
    session = Session()
    inst = vista.build(tmp_path / "filters", session=session, svo_client=_VistaSVO(),
                       atmosphere_root=tmp_path / "atmospheres")
    assert session.calls == [vista.BUNDLE]  # one download for every member
    manifest = InstrumentManifest.load(inst)
    assert manifest.default_site == "paranal" and manifest.outside_range == "edge"
    assert not (tmp_path / "atmospheres").exists()  # the SkyCalc site is not a team file
    for band in vista.BROAD + vista.NARROW:
        path = inst / manifest.component_file(FILTER, band)
        wl, tr = curve(path)
        assert released_mesa_reads(path) and 8000 < wl[np.argmax(tr)] < 25000 and 0.5 < tr.max() <= 1.0
    assert "at 80 K" in read_filter_header(inst / manifest.component_file(FILTER, "J"))["Sources"]
    for role in (OPTICS, QE):
        assert manifest.component_file(role, SHARED) == f"components/{role}/{role}.dat"
    wl, optics = curve(inst / manifest.component_file(OPTICS, SHARED))
    assert wl[0] == 3000.0 and wl[-1] == 30000.0 and 0.8 < optics.max() < 1.0  # M1 x M2, nm and % converted
    wl, qe = curve(inst / manifest.component_file(QE, SHARED))
    assert wl[0] == 8060.0 and 0.6 < qe.max() <= 1.0


# ------------------------------------------------------------------ Team atmospheres
SITES = ["haleakala_ps1", "paranal_vircam"]


def test_team_atmospheres_build_writes_archives_and_resumes(tmp_path):
    root, archive = tmp_path / "atmospheres", tmp_path / "evidence"
    session = Session()
    assert team_atmospheres.build(tmp_path / "filters", sites=SITES, session=session, archive_dir=archive) == root
    assert sorted(session.calls) == sorted([team_atmospheres.PS1_TABLE3, team_atmospheres.VIRCAM_BUNDLE])
    vircam = AtmosphereGrid.load(root / "paranal_vircam")
    assert [(e["airmass"], e["pwv"], e["file"]) for e in vircam.entries] == [
        (1.0, 1.0, "am1.00_pwv1.dat"), (2.0, 3.0, "am2.00_pwv3.dat")]
    wl, tr = curve(root / "paranal_vircam" / "am1.00_pwv1.dat")
    assert wl[0] == 9000.0 and tr[0] == pytest.approx(0.973927)  # 0.900000 um, as published
    assert AtmosphereGrid.load(root / "haleakala_ps1").kind == "team"
    record = json.loads((archive / "manifest.json").read_text())["files"]
    served = saved()
    assert {f["url"]: f["sha256"] for f in record} == {
        u: hashlib.sha256(served[u]).hexdigest() for u in (team_atmospheres.PS1_TABLE3, team_atmospheres.VIRCAM_BUNDLE)}

    before = (root / "haleakala_ps1" / "manifest.json").read_bytes()
    again = Session()
    team_atmospheres.build(tmp_path / "filters", sites=SITES, session=again)
    assert again.calls == [] and (root / "haleakala_ps1" / "manifest.json").read_bytes() == before
    team_atmospheres.build(tmp_path / "filters", sites=["haleakala_ps1"], session=again, overwrite=True)
    assert again.calls == [team_atmospheres.PS1_TABLE3]


def test_team_atmospheres_build_refuses_unknown_sites(tmp_path):
    with pytest.raises(ValueError, match="Unknown team atmosphere site.s. nowhere"):
        team_atmospheres.build(tmp_path, sites=["haleakala_ps1", "nowhere"], session=Session())
    assert not any(tmp_path.iterdir())


# ------------------------------------------------------------------- HST synphot tables
def test_hst_tables_take_the_newest_graph_and_walk_it():
    session = Session()
    tables = hst.SynphotTables(session=session)
    assert (tables.tmg_name, tables.tmc_name) == ("99c1930om_tmg.fits", "aa621124m_tmc.fits")
    assert session.calls == [f"{hst.TRDS}/mtab/"] * 2 + [f"{hst.TRDS}/mtab/{n}" for n in
                                                         ("99c1930om_tmg.fits", "aa621124m_tmc.fits")]
    comps, roles = hst._split_chain(tables, "acs,sbc", "F150LP")
    assert comps == ["hst_ota", "acs_hrc_m12", "acs_f150lp", "acs_f150lp_tds_def", "acs_sbc_mama"]
    assert roles == {FILTER: ["acs_f150lp"], OPTICS: ["hst_ota", "acs_hrc_m12"],
                     QE: ["acs_f150lp_tds_def", "acs_sbc_mama"]}
    assert tables.files["acs_sbc_mama"] == "cracscomp$acs_sbc_mama_013_syn.fits"
    assert tables.source("acs_f150lp") == "STScI synphot acs_f150lp_007_syn.fits (99c1930om_tmg.fits)"
    ir = tables.chain("wfc3,ir,f160w")
    assert ir[0] == "wfc3_ir_primary" and ir[-3:] == ["wfc3_ir_win", "wfc3_ir_qe", "wfc3_ir_cor"]
    with pytest.raises(ValueError, match="no branch at graph node"):
        tables.chain("acs,sbc,f999w")


def test_hst_tables_pinned_names_skip_the_listing():
    session = Session()
    hst.SynphotTables(session=session, tmg="99c1930om_tmg.fits", tmc="aa621124m_tmc.fits")
    assert f"{hst.TRDS}/mtab/" not in session.calls


def test_hst_tables_need_a_listed_table():
    files = saved()
    files[f"{hst.TRDS}/mtab/"] = files[f"{hst.TRDS}/mtab/"].replace(b"_tmc.fits", b"_tmx.fits")
    with pytest.raises(RuntimeError, match=r"No \*_tmc.fits found"):
        hst.SynphotTables(session=Session(files))


# ------------------------------------------------------- Checks against the teams' totals
def test_lsst_validate_compares_with_rubins_curves(tmp_path):
    session = t_lsst._Session(t_lsst._lsst_files("v1.9.2", scale=0.5))
    lsst.build(tmp_path / "filters", instrument="LSSTv19", session=session)
    result = lsst.validate(tmp_path / "filters" / "LSST" / "LSSTv19", session=session)
    assert sorted(result) == sorted(lsst.BANDS)
    for values in result.values():  # Rubin's hardware and total curves are these products
        assert values == {k: pytest.approx(0.0, abs=1e-6) for k in
                          ("hardware_dmag", "hardware_maxdiff", "total_dmag", "total_maxdiff")}
    assert lsst._url("hardware_r", "v1.9.2") in session.calls


def test_twomass_validate_compares_with_the_supplement_totals(tmp_path):
    inst = tmp_path / "2MASS" / "2MASS"
    for band, lo in (("J", 10600), ("H", 15000), ("Ks", 19500)):
        wl = np.linspace(lo, lo + 4000, 30)
        t_lsst._published(inst, band, wl, np.ones_like(wl), "filter+instrument+atmosphere", detector="energy")
    files = t_lsst._twomass_files()
    twomass.build(tmp_path, session=t_lsst._Session(files))
    um = np.round(np.linspace(1.0, 2.5, 151), 3)
    for band, n in twomass.TOTAL_TABLE.items():  # J and H as composed, Ks at half
        mine = compose_band(inst, band, (FILTER, OPTICS, QE))
        total = np.interp(um * 1e4, mine.wavelength, mine.transmission, left=0, right=0) * (0.5 if band == "Ks" else 1)
        files[twomass.TABLE.format(n=n)] = t_lsst._pre(f"2MASS {band} TOTAL", um, total)
    result = twomass.validate(inst, session=t_lsst._Session(files))
    assert result["J"]["dmag"] == pytest.approx(0.0, abs=1e-3) and result["H"]["maxdiff"] < 1e-4
    assert result["Ks"]["dmag"] == pytest.approx(-2.5 * np.log10(2.0), abs=1e-3)


def test_skycalc_transmission_is_the_fetched_curve():
    client = skycalc.SkyCalcClient(session=_FormSession(), interface="form", pause=0)
    wl, tr = client.transmission("paranal", 1.5, 2.5, 1000.0, 1100.0, 2000)
    np.testing.assert_allclose(wl, np.geomspace(1000.0, 1100.0, 200))
    np.testing.assert_allclose(tr, 0.8 ** 1.5)
    assert client.session.posts[-1]["SKYMODEL.TARGET.AIRMASS"] == "1.50"


class _Session:
    def __init__(self, files):
        self.files, self.headers, self.calls = files, {}, []

    def get(self, url, timeout=None, **kw):
        self.calls.append(url)
        return _Resp(self.files[url]) if url in self.files else _Resp(b"", 404)


# ---------------------------------------------------------------------------
# Registering a team atmosphere as a site
# ---------------------------------------------------------------------------
def test_write_site_refuses_a_bad_curve_before_writing(tmp_path):
    bad = _site(entries=[(1.2, None, DerivedCurve("u", _curve([0.5, 1.2, 0.9]), "x"))])
    with pytest.raises(ValueError, match="not a transmission"):
        write_site(tmp_path, bad, None)
    assert not (tmp_path / "here_team").exists()


# ---------------------------------------------------------------------------
# Which sites an instrument may use
# ---------------------------------------------------------------------------
def test_every_registered_instrument_site_is_built_by_a_builder():
    from sed_tools.facility_sites import _INSTRUMENT_SITES

    for sites in _INSTRUMENT_SITES.values():
        assert set(sites) <= set(team.SITES)


# ---------------------------------------------------------------------------
# Team recipes (parsers checked on in-memory excerpts)
# ---------------------------------------------------------------------------
PS1_ROWS = (" 500 0.500 0.400 0.000 0.000 0.000 0.000 0.000 0.970 0.880 0.990\n"
            " 501 0.500 0.410 0.000 0.000 0.000 0.000 0.000 0.970 0.881 0.980\n")


def test_ps1_atmosphere_is_the_product_of_its_three_columns():
    wl, tr = team.ps1_atmosphere(PS1_ROWS)
    np.testing.assert_allclose(wl, [5000.0, 5010.0])
    np.testing.assert_allclose(tr, [0.97 * 0.88 * 0.99, 0.97 * 0.881 * 0.98])


def test_ps1_build_registers_its_team_site(tmp_path, monkeypatch):
    table4 = "g_P1|0.4|481|414|551|24.6\n"
    session = _Session({surveys.PS1_URL: PS1_ROWS, surveys.PS1_BASE + "table4.dat": table4})
    monkeypatch.setattr(surveys, "ps1", lambda fetch: surveys.Instrument(
        "PAN-STARRS", "PS1", [surveys.make_band("g", [500.0, 501.0], [0.4, 0.41], unit="nm",
                                                detector_type="photon", components="filter+optics+qe+atmosphere",
                                                sources="x")], "n", ground_based=True,
        sites=[team.HALEAKALA_PS1], default_site="haleakala_ps1"))
    monkeypatch.setitem(surveys.RECIPES, "PS1", ("PAN-STARRS", surveys.ps1))
    build = surveys.make_builder(surveys.RECIPES)
    inst = build(tmp_path / "filters", instrument="PS1", session=session)
    assert InstrumentManifest.load(inst).default_site == "haleakala_ps1"
    grid = AtmosphereGrid.load(tmp_path / "atmospheres" / "haleakala_ps1")
    assert grid.kind == TEAM and grid.entries[0]["airmass"] == 1.2 and "Aero" in grid.entries[0]["derivation"]


def _tar(members, prefix):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_csp_atmosphere_raises_the_one_airmass_transmission_to_1p2():
    raw = _tar({"atm_ext_ctio.dat": b"3000. 0.5\n5000. 0.8\n11000. 1.0\n",
                "atm_ext_lines.dat": b"2950. 1.\n4000. 0.9\n6000. 1.0\n12000. 1.\n"}, "CSP_filter_package")
    wl, tr = team.csp_atmosphere(raw)
    assert wl[0] == 3000.0 and wl[-1] == 11000.0
    np.testing.assert_allclose(tr[wl == 5000.0], 0.8 ** 1.2 * 0.95)


def test_vircam_atmosphere_skips_its_title_lines():
    member = gzip.compress(b"1.0mm H2O, airmass=1.0\n\n0.900000   0.97\n0.900500   0.96\n")
    raw = _tar({"trans_10_10.dat.gz": member}, "Filters_QE_Atm_curves")
    wl, tr = team.vircam_atmosphere(raw, "trans_10_10.dat.gz", "1.0mm H2O, airmass=1.0")
    np.testing.assert_allclose(wl, [9000.0, 9005.0])
    with pytest.raises(ValueError, match="unexpected title"):
        team.vircam_atmosphere(raw, "trans_10_10.dat.gz", "3.0mm H2O, airmass=2.0")


def test_twomass_table_refuses_a_truncated_copy():
    page = ("<html><head><title>ATMOSPHERIC TRANSMISSION AT 2MASS NORTH SITE</title></head><pre>\n"
            "|n  |xlam     |yval     |\n   1    1.0000    0.8948\n   2    1.0001    0.8922\n</pre>")
    wl, tr = team.twomass_table(page.encode(), "ATMOSPHERIC TRANSMISSION AT 2MASS NORTH SITE")
    np.testing.assert_allclose(wl, [10000.0, 10001.0])
    with pytest.raises(ValueError, match="consecutive"):
        team.twomass_table(page.replace("   2 ", "   3 ").encode(), "ATMOSPHERIC TRANSMISSION AT 2MASS NORTH SITE")


def test_gemini_atran_refuses_a_truncated_file():
    full = ".900000 0.99\n3.0 0.5\n5.59998 0.1\n"
    wl, _ = team.gemini_atran(full.encode(), "u")
    assert wl[-1] == pytest.approx(55999.8)
    with pytest.raises(ValueError, match="truncated"):
        team.gemini_atran(b".900000 0.99\n1.2 0.5\n", "u")


def test_sdss_transparency_merges_bands_and_refuses_conflicts():
    head = "# assumed\n# atmospheric\n# extinction\n# through\n# 1.0 airmasses\n"
    files = {url: head + "3000 0 0 0 0.10\n3025 0 0 0 0.12\n" for url in team.SDSS_FILES.values()}
    wl, tr = team.sdss_transparency(_Session(files))
    np.testing.assert_allclose(tr, [0.10, 0.12])
    files[team.SDSS_FILES["z"]] = head + "3000 0 0 0 0.11\n"
    with pytest.raises(ValueError, match="disagrees"):
        team.sdss_transparency(_Session(files))


def test_bass_ratio_keeps_only_well_measured_rows():
    head = "# ... atmospheric extinction at airmass of 1.0.\n"
    files = {team.BASS_FILES["g"]: head + "4000 0.5 0.400 0.300\n4010 0.1 0.050 0.040\n",
             team.BASS_FILES["r"]: head + "6000 0.9 0.800 0.720\n"}
    wl, tr = team.bass_atmosphere(_Session(files))
    np.testing.assert_allclose(wl, [4000.0, 6000.0])
    np.testing.assert_allclose(tr, [0.75, 0.9])


def test_team_sites_are_registered_with_provenance():
    for name, site in team.SITES.items():
        assert site.kind == TEAM and site.citation and site.source, name
        assert name.count("_") >= 1 and name == site.site
    gemini = team.SITES["maunakea_gemini"]
    assert len(gemini.entries) == 12 and "Lord" in gemini.terms
    assert {(a, p) for a, p, _ in gemini.entries} >= {(1.0, 1.6), (2.0, 5.0)}
    assert team.SITES["hopkins_2mass"].entries[0][0] is None  # airmass not stated


def test_caching_session_archives_each_download_once(tmp_path):
    session = team.CachingSession(_Session({"https://x.org/a.dat": b"1 2\n"}), tmp_path / "ev")
    session.get("https://x.org/a.dat")
    session.get("https://x.org/a.dat")
    manifest = json.loads((tmp_path / "ev" / "manifest.json").read_text())
    assert session.session.calls == ["https://x.org/a.dat"]
    (record,) = manifest["files"]
    assert record["url"] == "https://x.org/a.dat" and record["sha256"] == hashlib.sha256(b"1 2\n").hexdigest()
    assert (tmp_path / "ev" / record["file"]).read_bytes() == b"1 2\n"


# ---------------------------------------------------------------------------
# SkyCalc: derived grids, the web form, provenance
# ---------------------------------------------------------------------------
def test_form_fields_follow_skycalc_spelling():
    params = dict(skycalc._BASE_PARAMS, airmass=2.0, pwv=0.5, observatory="2400")
    fields = skycalc._form_fields(params)
    assert fields["SKYMODEL.PWV"] == "0.50" and fields["TEL.SITE.HEIGHT"] == "2400"
    assert fields["SKYMODEL.TARGET.AIRMASS"] == "2.00" and fields["SKYMODEL.TARGET.ALT"] == "30.000"
    assert "SKYMODEL.INCL.MOON" not in fields  # unticked checkbox
    assert skycalc.airmass_altitude(3.0) == 19.5
    with pytest.raises(ValueError):
        skycalc.pwv_option(2.0)


def test_form_client_checks_the_echo():
    client = skycalc.SkyCalcClient(session=_FormSession(), interface="form", pause=0)
    result = client.fetch("paranal", 1.5, 2.5, 1000.0, 1100.0, 2000)
    assert result.interface == "form" and result.version == "2.0.9"
    assert result.url == skycalc.FORM_HOST + "/observing/etc/tmp/abc/skytable.fits"
    bad = skycalc.SkyCalcClient(session=_FormSession({"SKYMODEL.PWV": "3.5"}), interface="form", pause=0)
    with pytest.raises(RuntimeError, match="echoed SKYMODEL.PWV"):
        bad.fetch("paranal", 1.5, 2.5, 1000.0, 1100.0, 2000)


def test_skycalc_grid_is_labelled_derived_and_archived(tmp_path):
    client = skycalc.SkyCalcClient(session=_FormSession(), interface="form", pause=0)
    root = skycalc.build(tmp_path / "filters", sites=["lasilla"], airmasses=[1.0], pwvs=[2.5],
                         fetch_resolution=2000, store_resolution=500, chunks=2, client=client,
                         archive_dir=tmp_path / "evidence")
    grid = AtmosphereGrid.load(root / "lasilla")
    assert grid.kind == DERIVED and grid.model_version == "2.0.9" and "Noll" in grid.citation
    assert grid.parameters["observatory"] == "2400" and grid.parameters["fetch_resolution"] == 2000
    assert grid.parameters["interface"].startswith("form: https://www.eso.org")
    header = read_filter_header(root / "lasilla" / "am1.00_pwv2.5.dat")
    assert header["ComponentsSource"] == "derived" and header["Sources"] == "ESO SkyCalc 2.0.9"
    evidence = json.loads((tmp_path / "evidence" / "lasilla" / "manifest.json").read_text())
    assert len(evidence["files"]) == 2
    record = evidence["files"][0]
    raw = gzip.decompress((tmp_path / "evidence" / "lasilla" / record["file"]).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == record["sha256"] and record["skycalc_version"] == "2.0.9"


def test_skycalc_auto_mode_switches_to_the_form_when_the_api_host_is_down():
    from requests.exceptions import ConnectTimeout

    class Down(_FormSession):
        def post(self, url, data=None, json=None, timeout=None, **kw):
            assert timeout[0] <= 30  # connect fails fast; the model run may take minutes
            if json is not None:
                raise ConnectTimeout("api host down")
            return super().post(url, data=data, timeout=timeout)

    client = skycalc.SkyCalcClient(session=Down(), interface="auto", pause=0)
    assert client.fetch("paranal", 1.0, 2.5, 1000.0, 1100.0, 2000).interface == "form"
    assert client.interface == "form"


def test_skycalc_resume_leaves_a_complete_grid_untouched(tmp_path):
    client = skycalc.SkyCalcClient(session=_FormSession(), interface="form", pause=0)
    kw = dict(sites=["paranal"], airmasses=[1.0], pwvs=[2.5], chunks=2, client=client)
    root = skycalc.build(tmp_path / "filters", fetch_resolution=2000, store_resolution=500, **kw)
    before = (root / "paranal" / "manifest.json").read_bytes()
    # A curator re-running the builder (other settings, nothing to fetch) must not rewrite served files.
    skycalc.build(tmp_path / "filters", client=skycalc.SkyCalcClient(session=_FormSession(), pause=0),
                  sites=["paranal"], airmasses=[1.0], pwvs=[2.5])
    assert (root / "paranal" / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="different settings"):
        skycalc.build(tmp_path / "filters", fetch_resolution=2000, store_resolution=400,
                      **dict(kw, airmasses=[1.5]))


def test_hsc_builder_and_team_registry_write_the_same_site():
    from sed_tools.component_sources import hsc

    assert team.SITES["maunakea_hsc"] is hsc.SPEC.site and hsc.SPEC.site.kind == TEAM
