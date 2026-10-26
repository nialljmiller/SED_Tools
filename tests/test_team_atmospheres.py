"""Team atmospheres registered as named sites, and derived (SkyCalc) grids.

All sources are served from memory: ``*.dat`` fixtures are not tracked.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from sed_tools import filter_library
from sed_tools.component_sources import _common, skycalc, surveys, team_atmospheres as team
from sed_tools.component_sources._common import DerivedCurve, SiteAtmosphere, Table, write_site
from sed_tools.facility_sites import instrument_sites
from sed_tools.filter_components import (ATMOSPHERE, DERIVED, FILTER, TEAM, AtmosphereGrid, AtmosphereSpec,
                                         ComponentCoverageError, ComponentError, InstrumentManifest,
                                         add_component, build_product, compose_band, resolve_atmosphere)
from sed_tools.filter_library import FilterLibrary
from sed_tools.filter_metadata import (ENERGY, PHOTON, read_filter_header, released_mesa_reads,
                                       write_filter_file)

WL = np.linspace(4000.0, 6000.0, 201)


class _Resp:
    def __init__(self, content, status=200, text=None):
        self.content = content if isinstance(content, bytes) else content.encode("latin-1")
        self.status_code = status
        self.headers = {}
        self.text = text if text is not None else self.content.decode("latin-1")

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


def _curve(values):
    return lambda session: (np.array([3000.0, 5000.0, 7000.0]), np.array(values))


def _instrument(root, facility="Test", instrument="Camera", *, default_site=None, sites=(), outside="error"):
    inst = root / facility / instrument
    inst.mkdir(parents=True)
    write_filter_file(inst / "g.dat", WL, np.exp(-0.5 * ((WL - 5000) / 200) ** 2), detector_type=PHOTON,
                      source="svo", wavelength_unit="Angstrom", extra={"Components": "filter"})
    manifest = InstrumentManifest.load(inst)
    manifest.facility, manifest.instrument = facility, instrument
    manifest.ground_based = True
    manifest.default_site = default_site
    manifest.atmosphere_sites = list(sites)
    manifest.outside_range = outside
    manifest.save(inst)
    return inst


def _site(name="here_team", entries=None, **kw):
    entries = entries if entries is not None else [(1.2, None, DerivedCurve("u", _curve([0.5, 0.8, 0.9]), "x"))]
    return SiteAtmosphere(site=name, altitude_m=1000.0, entries=entries, source="test team",
                          kind=TEAM, model="MODTRAN", model_version="not stated", citation="Someone 2001",
                          **kw)


# ---------------------------------------------------------------------------
# Manifests stay byte-compatible
# ---------------------------------------------------------------------------
LEGACY_SITE = {"altitude_m": 2207.0, "entries": [{"airmass": 1.2, "file": "am1.20.dat", "pwv": None}],
               "notes": "", "schema": 1, "site": "ctio_des", "source": "DES standard atmosphere"}


def test_legacy_site_manifest_round_trips_unchanged(tmp_path):
    (tmp_path / "ctio_des").mkdir()
    text = json.dumps(LEGACY_SITE, indent=2, sort_keys=True) + "\n"
    (tmp_path / "ctio_des" / "manifest.json").write_text(text)
    AtmosphereGrid.load(tmp_path / "ctio_des").save(tmp_path / "ctio_des")
    assert (tmp_path / "ctio_des" / "manifest.json").read_text() == text


def test_instrument_manifest_without_extra_sites_has_no_new_key(tmp_path):
    inst = _instrument(tmp_path, default_site="lsst_std")
    assert "atmosphere_sites" not in json.loads((inst / "manifest.json").read_text())


def test_grid_entries_need_a_numeric_airmass():
    with pytest.raises(ComponentError, match="unstated"):
        AtmosphereGrid("s", [{"airmass": None, "pwv": None, "file": "a.dat"}])


# ---------------------------------------------------------------------------
# Registering a team atmosphere as a site
# ---------------------------------------------------------------------------
def test_write_site_records_provenance_and_derivation(tmp_path):
    verbatim = DerivedCurve("https://team/atm10.dat", _curve([0.4, 0.7, 0.8]), derivation="", note="as published")
    site = _site(entries=[(1.0, 1.5, verbatim),
                          (1.2, None, DerivedCurve("https://team/a | https://team/b", _curve([0.5, 0.8, 0.9]),
                                                   "a / b"))],
                 terms="Cite us.", outside_range="error")
    site_dir = write_site(tmp_path, site, None)

    raw = json.loads((site_dir / "manifest.json").read_text())
    assert raw["kind"] == "team" and raw["model"] == "MODTRAN" and raw["terms"] == "Cite us."
    assert raw["entries"][0] == {"airmass": 1.0, "pwv": 1.5, "file": "am1.00_pwv1.5.dat",
                                 "source": "https://team/atm10.dat", "note": "as published"}
    assert raw["entries"][1]["derivation"] == "a / b"
    assert "unstated" not in raw
    original, derived = (read_filter_header(site_dir / e["file"]) for e in raw["entries"])
    assert original["ComponentsSource"] == "original" and derived["ComponentsSource"] == "derived"
    assert derived["Atmosphere"] == "site=here_team airmass=1.2"
    assert all(released_mesa_reads(site_dir / e["file"]) for e in raw["entries"])


def test_write_site_without_provenance_keeps_the_legacy_layout(tmp_path):
    site = SiteAtmosphere("ctio_des", 2207.0, [(1.2, None, DerivedCurve("u", _curve([0.5, 0.8, 0.9]), ""))],
                          "DES standard atmosphere")
    write_site(tmp_path, site, None)
    assert json.loads((tmp_path / "ctio_des" / "manifest.json").read_text()) == LEGACY_SITE
    assert "ComponentsSource" not in read_filter_header(tmp_path / "ctio_des" / "am1.20.dat")


def test_write_site_refuses_a_bad_curve_before_writing(tmp_path):
    bad = _site(entries=[(1.2, None, DerivedCurve("u", _curve([0.5, 1.2, 0.9]), "x"))])
    with pytest.raises(ValueError, match="not a transmission"):
        write_site(tmp_path, bad, None)
    assert not (tmp_path / "here_team").exists()


def test_unstated_airmass_is_listed_apart_and_selected_explicitly(tmp_path):
    site = _site("plexus_team", entries=[(None, None, DerivedCurve("u", _curve([0.5, 0.8, 0.9]), ""))])
    write_site(tmp_path, site, None)
    raw = json.loads((tmp_path / "plexus_team" / "manifest.json").read_text())
    assert raw["entries"] == [] and raw["unstated"][0]["file"] == "am_unstated.dat"
    # Released SED_Tools lists every served site with float(e["airmass"]).
    assert sorted({float(e["airmass"]) for e in raw["entries"]}) == []

    grid = AtmosphereGrid.load(tmp_path / "plexus_team")
    assert grid.select(AtmosphereSpec("plexus_team", None)) == "am_unstated.dat"
    with pytest.raises(ComponentError, match="unstated airmass: airmass=None"):
        grid.select(AtmosphereSpec("plexus_team", 1.0))
    with pytest.raises(ComponentError, match="atmosphere needs at least 'airmass'"):
        AtmosphereSpec.coerce({"site": "plexus_team"})
    assert AtmosphereSpec.coerce({"site": "plexus_team", "airmass": None}).label == "plexus_team_amunstated"
    with pytest.raises(ComponentError, match="no pwv"):
        AtmosphereSpec("plexus_team", None, 2.0)


# ---------------------------------------------------------------------------
# Which sites an instrument may use
# ---------------------------------------------------------------------------
def test_team_site_is_offered_next_to_the_default_site(tmp_path):
    inst = _instrument(tmp_path / "filters", default_site="paranal", sites=["here_team"])
    manifest = InstrumentManifest.load(inst)
    assert manifest.sites() == ["paranal", "here_team"]
    assert resolve_atmosphere(manifest, AtmosphereSpec(None, 1.2, 2.5)).site == "paranal"
    assert resolve_atmosphere(manifest, AtmosphereSpec("here_team", 1.2)).site == "here_team"
    with pytest.raises(ComponentError, match=r"observes from \['paranal', 'here_team'\].*not mixed in"):
        resolve_atmosphere(manifest, AtmosphereSpec("lsst_std", 1.2))
    info = manifest.describe()
    assert info["atmosphere"] is True and info["atmosphere_sites"] == ["paranal", "here_team"]


def test_registered_team_sites_need_no_manifest_change():
    ps1 = InstrumentManifest(facility="PAN-STARRS", instrument="PS1", bands=["g"], ground_based=True)
    assert resolve_atmosphere(ps1, AtmosphereSpec(None, 1.2)).site == "haleakala_ps1"
    twomass = InstrumentManifest(facility="2MASS", instrument="2MASS", bands=["J"], ground_based=True)
    with pytest.raises(ComponentError, match="several atmosphere sites"):
        resolve_atmosphere(twomass, AtmosphereSpec(None, None))
    assert resolve_atmosphere(twomass, AtmosphereSpec("ctio_2mass", None)).site == "ctio_2mass"
    vista = InstrumentManifest(facility="Paranal", instrument="VISTA", bands=["J"], ground_based=True,
                               default_site="paranal")
    assert vista.sites() == ["paranal", "paranal_vircam"]
    assert instrument_sites("Gemini", "GPI") == ()  # moved between sites: deliberately absent


def test_every_registered_instrument_site_is_built_by_a_builder():
    from sed_tools.facility_sites import _INSTRUMENT_SITES

    for sites in _INSTRUMENT_SITES.values():
        assert set(sites) <= set(team.SITES)


def test_existing_single_site_instruments_are_unchanged(tmp_path):
    inst = _instrument(tmp_path / "filters", "LSST", "LSST", default_site="lsst_std")
    manifest = InstrumentManifest.load(inst)
    assert manifest.sites() == ["lsst_std"]
    with pytest.raises(ComponentError, match="observes from 'lsst_std'.*not mixed in"):
        resolve_atmosphere(manifest, AtmosphereSpec("paranal", 1.2, 2.5))


# ---------------------------------------------------------------------------
# Composition with team atmospheres
# ---------------------------------------------------------------------------
def test_compose_with_unstated_team_atmosphere(tmp_path):
    inst = _instrument(tmp_path / "filters", sites=["plexus_team"])
    write_site(tmp_path / "atm", _site("plexus_team", entries=[
        (None, None, DerivedCurve("u", _curve([0.5, 0.5, 0.5]), ""))]), None)
    out = build_product(inst, ["filter"], atmosphere={"site": "plexus_team", "airmass": None},
                        atmosphere_root=tmp_path / "atm")
    assert out.name == "filter+atm_plexus_team_amunstated"
    header = read_filter_header(out / "g.dat")
    assert header["Atmosphere"] == "site=plexus_team airmass=unstated"
    assert header["Components"] == "filter+atmosphere"


def test_site_outside_range_overrides_the_instruments_policy(tmp_path):
    short = DerivedCurve("u", lambda s: (np.array([4500.0, 5500.0]), np.array([0.9, 0.9])), "")
    inst = _instrument(tmp_path / "filters", default_site="short_team", outside="edge")
    write_site(tmp_path / "atm", _site("short_team", entries=[(1.2, None, short)], outside_range="error"), None)
    with pytest.raises(ComponentCoverageError):
        compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=tmp_path / "atm")
    # Unset on the site: the instrument's own convention (here edge) applies, as before.
    write_site(tmp_path / "atm2", _site("short_team", entries=[(1.2, None, short)]), None)
    curve = compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=tmp_path / "atm2")
    assert curve.transmission.max() == pytest.approx(0.9 * 1.0, rel=1e-3)


def test_add_component_points_atmospheres_to_sites(tmp_path):
    with pytest.raises(ComponentError, match="site of its own"):
        add_component(tmp_path, "atmosphere", [4000.0, 5000.0], [0.9, 0.9])


def test_library_rebuilds_a_missing_team_site(tmp_path, monkeypatch):
    monkeypatch.setattr(filter_library.NJMFilterGrabber, "_check_availability", lambda self: False)
    inst = _instrument(tmp_path / "filters", "PAN-STARRS", "PS1")
    library = FilterLibrary(tmp_path / "filters")
    calls = []

    def build(root, **kw):
        calls.append(kw)
        write_site(kw["atmosphere_root"], _site("haleakala_ps1"), None)

    monkeypatch.setattr(filter_library, "get_builder", lambda name: calls.append(name) or build)
    path = library.prepare_atmosphere(InstrumentManifest.load(inst), AtmosphereSpec(None, 1.2))
    assert path.name == "am1.20.dat"
    assert calls[0] == "team_atmospheres" and calls[1]["sites"] == ["haleakala_ps1"]
    listing = library.atmospheres(remote=False)["haleakala_ps1"]
    assert listing["kind"] == "team" and listing["airmass"] == [1.2]


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
def _skytable(fields, lam, trans, version="2.0.9"):
    from astropy.io import fits

    primary = fits.PrimaryHDU()
    primary.header.add_comment(f"SkyCalc version: {version}")
    for key, value in fields.items():
        primary.header.add_comment(f"{key}={value}")
    table = fits.BinTableHDU.from_columns([fits.Column("lam", "D", array=lam), fits.Column("trans", "D", array=trans)])
    buf = io.BytesIO()
    fits.HDUList([primary, table]).writeto(buf)
    return buf.getvalue()


class _FormSession:
    """The SkyCalc form: a POST returns HTML linking skytable.fits, which echoes its inputs."""

    def __init__(self, echo_override=None):
        self.posts, self.echo_override = [], echo_override or {}

    def post(self, url, data=None, timeout=None, **kw):
        self.posts.append(dict(data))
        return _Resp(b"<h4><a href=/observing/etc/tmp/abc/skytable.fits> skytable.fits</a></h4>")

    def get(self, url, timeout=None, **kw):
        fields = {**self.posts[-1], **self.echo_override}
        lo, hi = float(fields["SKYMODEL.WAVELENGTH.MIN"]), float(fields["SKYMODEL.WAVELENGTH.MAX"])
        lam = np.geomspace(lo, hi, 200)
        return _Resp(_skytable(fields, lam, np.full_like(lam, 0.8 ** float(fields["SKYMODEL.TARGET.AIRMASS"]))))


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


@pytest.mark.parametrize("argv,expected", [
    (["--airmass", "unstated", "--atmosphere-site", "ctio_2mass"], {"airmass": None, "pwv": None, "site": "ctio_2mass"}),
    (["--airmass", "1.2", "--pwv", "2.5"], {"airmass": 1.2, "pwv": 2.5}),
])
def test_cli_passes_the_atmosphere_site_and_unstated_airmass(tmp_path, monkeypatch, argv, expected):
    from sed_tools import api, cli

    seen = {}
    monkeypatch.setattr(api.Filters, "compose", classmethod(lambda cls, target, **kw: seen.update(kw)))
    monkeypatch.setattr("sys.argv", ["sed-tools", "filters-compose", "2MASS/2MASS", "--base", str(tmp_path),
                                    "--components", "filter", *argv])
    assert cli.main() in (0, None)
    assert seen["atmosphere"] == expected
