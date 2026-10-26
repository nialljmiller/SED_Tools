"""Server tooling: info.json from archived SVO records, NJM provenance and resolved references."""

import hashlib
import json

from filter_info.compose import InfoSources, _name_forms, compose_info, match_svo
from filter_info.references import canonical, link_duplicates, reference_keys
from filter_info.svo_archive import build, curve_digest, decode, parse_browse, parse_votable

CURVE = [(5000.0, 0.0), (6000.0, 1.0), (7000.0, 0.0)]
VOTABLE = """<VOTABLE><RESOURCE><TABLE>
<PARAM name="FilterProfileService" value="ivo://svo/fps"/>
<PARAM name="filterID" value="Test/Cam.R"/>
<PARAM name="Description" value="Test R"/>
<PARAM name="PhotSystem" value="Test"/>
<PARAM name="DetectorType" value="1"/>
<PARAM name="Band" value="R"/>
<PARAM name="ProfileReference" value="http://example.org/cam.html"/>
<PARAM name="CalibrationReference" value="http://adsabs.harvard.edu/abs/2003AJ....126.1090C"/>
<PARAM name="Description" value="repeated"/>
<PARAM name="WavelengthPivot" value="6000" unit="Angstrom"/>
<PARAM name="MagSys" value="Vega"/>
<DATA><TABLEDATA>
<TR><TD>5000</TD><TD>0</TD></TR>
<TR><TD>6000</TD><TD>1</TD></TR>
<TR><TD>7000</TD><TD>0</TD></TR>
</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>
"""
TIP = ("Tip('&lt;b&gt;{name}&lt;/b&gt;&lt;br&gt;&lt;table&gt;&lt;td class=helptd&gt;{desc}&lt;/td&gt;&lt;/table&gt;"
       "&lt;ul&gt;&lt;li&gt;&lt;a href=&quot;{url}&quot;&gt;{label}&lt;/a&gt;&lt;/ul&gt;',WIDTH,400)")
BROWSE = f"""<html><body>
<a href="javascript:{TIP.format(name='Test', desc='A test facility.', url='http://example.org', label='Home')}">Test</a>
<a href="javascript:{TIP.format(name='Test/Cam', desc='The test camera.', url='http://example.org/cam', label='Cam')}">Cam</a>
<table><tr><td nowrap>Filter ID <a href="#">(?)</a> :</b></td><td>Test/Cam.R</td></tr>
<tr><td>Description <a href="#">(?)</a> :</td><td>Test R</td></tr>
<tr><td>Phot.System <a href="#">(?)</a> :</td><td>Test</td></tr>
<tr><td>Detector Type <a href="#">(?)</a> :</td><td>Energy counter</td></tr>
<tr><td>Comments <a href="#">(?)</a> :</td><td>------</td></tr></table>
<table><tr><td>Property</td><td>Calculated</td><td>Specified</td><td>Unit</td></tr>
<tr><td>&lambda;<sub>ref</sub> <a href="#">(?)</a> :</td><td>6001.5</td><td>6000</td><td>(Angstrom)</td></tr>
<tr><td>F<sub>sun</sub> <a href="#">(?)</a> :</td><td>45.89</td><td>-------</td><td>(erg/cm2/s/A)</td></tr></table>
<table><tr><td>Transmission components: Filter + QE<p>Reference for filter response:
<a href="http://example.org/cam.html">Camera page</a><p>Reference for calibration:
<a href="http://adsabs.harvard.edu/abs/2003AJ....126.1090C">Cohen 2003</a></td></tr></table>
<b>Vega System</b><table><tr><td>Property</td><td>Specified</td><td>Calculated</td><td>Unit</td></tr>
<tr><td>Zero Point <a href="#">(?)</a> :</td><td>3.1e-09</td><td>3.0e-09</td><td>(erg/cm2/s/A)</td></tr>
<tr><td></td><td>3600</td><td>3590</td><td>(Jy)</td></tr>
<tr><td>ZP Type <a href="#">(?)</a> :</td><td>Pogson</td></tr>
<tr><td>PhotCal ID <a href="#">(?)</a> :</td><td>Test/Cam.R/Vega</td></tr></table>
Filter added: 2010-04-17 05:58:55 Last update: 2021-07-27 18:33:54
we would appreciate if you could include the following acknowledgement in your publication:
Thanks, SVO. and we would appreciate if you could include the following references:
https://ui.adsabs.harvard.edu/abs/2012ivoa.rept.1015R/abstract
</body></html>"""


def _curve_text(header_field):
    return f"Wavelength,Transmission,{header_field}\n" + "".join(f"{w},{t}\n" for w, t in CURVE)


def test_votable_keeps_first_repeated_param_and_samples():
    record = parse_votable(VOTABLE)
    assert record["params"]["Description"] == "Test R"
    assert record["params"]["DetectorType"] == "1"
    assert record["samples"] == CURVE
    assert parse_votable("<VOTABLE/>") is None


def test_browse_page_fields_references_values_and_descriptions():
    page = parse_browse(BROWSE)
    assert page["fields"]["filterID"] == "Test/Cam.R"
    assert page["fields"]["DetectorType"] == "Energy counter"
    assert page["fields"]["Comments"] == ""
    assert page["fields"]["components"] == "Filter + QE"
    assert (page["fields"]["added"], page["fields"]["updated"]) == ("2010-04-17 05:58:55", "2021-07-27 18:33:54")
    assert page["references"]["CalibrationReference"] == {
        "url": "http://adsabs.harvard.edu/abs/2003AJ....126.1090C", "label": "Cohen 2003"}
    assert page["specified"] == {"WavelengthRef": 6000.0}
    assert page["calculated"] == {"WavelengthRef": 6001.5, "Fsun": 45.89}
    assert page["zero_points"]["Vega"]["specified"] == {"erg/cm2/s/A": 3.1e-09, "Jy": 3600.0}
    assert page["facility"]["description"] == "A test facility."
    assert page["instrument"]["links"] == [{"url": "http://example.org/cam", "label": "Cam"}]
    assert page["service"] == {"acknowledgement": "Thanks, SVO.", "references": ["2012ivoa.rept.1015R"]}
    assert parse_browse("<html>Redirecting...</html>") is None


def test_reference_keys_normalise_ads_doi_arxiv_and_urls():
    assert reference_keys(
        "https://arxiv.org/e-print/1802.00879v2 (Table 2) | https://ui.adsabs.harvard.edu/abs/2018PASP..130f4505T"
    ) == ["arxiv:1802.00879", "bibcode:2018PASP..130f4505T"]
    assert reference_keys("https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...3R/abstract") == [
        "bibcode:2021A&A...649A...3R"]
    assert reference_keys("see doi:10.1086/376474. and http://example.org/a.html,") == [
        "doi:10.1086/376474", "url:http://example.org/a.html"]


def test_duplicate_works_point_at_the_bibcode_entry():
    cache = {"arxiv:1802.00879": {"doi": "10.1088/x"}, "bibcode:2018PASP..130f4505T": {"doi": "10.1088/x"}}
    link_duplicates(cache)
    assert canonical("arxiv:1802.00879", cache) == "bibcode:2018PASP..130f4505T"


def test_band_name_forms():
    assert "ib200" in _name_forms("IB_2p00") and "ib200" in _name_forms("IB_2.00")
    assert "h" in _name_forms("H_CONICA")
    assert "lp" in _name_forms("L_prime")


def _archive(tmp_path):
    raw = tmp_path / "harvest" / "raw"
    raw.mkdir(parents=True)
    rows = []
    for kind, text, ext in (("fps_votable", VOTABLE, "xml"), ("fps_browse", BROWSE, "html")):
        path = raw / f"Test_Cam.R__{kind}.{ext}"
        path.write_text(text)
        rows.append({"id": "Test/Cam.R", "kind": kind, "capture_timestamp": "20250101000000",
                     "original_url": f"http://svo2/{kind}", "status": "ok", "file": f"raw/{path.name}",
                     "sha256": hashlib.sha256(text.encode()).hexdigest()})
    rows.append({**rows[0], "capture_timestamp": "20240101000000", "sha256": "0" * 64})  # corrupt copy
    manifest = tmp_path / "harvest" / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    svo_dir = tmp_path / "svo"
    assert build(svo_dir, [manifest]) == 1
    return svo_dir


def test_info_json_from_svo_archive_provenance_and_references(tmp_path):
    svo_dir = _archive(tmp_path)
    record = json.loads((svo_dir / "Test.json").read_text())["Test/Cam.R"]
    assert len(record["votable"]) == 1 and record["votable"][0]["record"]["curve_sha256"] == curve_digest(CURVE)
    assert json.loads((svo_dir / "_service.json").read_text())["acknowledgement"] == "Thanks, SVO."

    inst = tmp_path / "data" / "filters" / "Test" / "Cam"
    inst.mkdir(parents=True)
    curve = inst / "R.dat"
    curve.write_text(_curve_text("FilterID=Test/Cam.R;DetectorType=energy;DetectorTypeSource=svo;"
                                 "Sources=https://example.org/r.dat"))
    references = {"bibcode:2003AJ....126.1090C": {"short": "Cohen et al. 2003", "doi": "10.1086/376474"},
                  "bibcode:2012ivoa.rept.1015R": {"short": "Rodrigo et al. 2012"}}
    sources = InfoSources(tmp_path, svo_dir=svo_dir, curated_dir=tmp_path / "curated", references=references)
    info = compose_info(sources, inst, [curve], {"R": {"WavelengthPivot": 6000.0}})

    band = info["bands"]["R"]
    assert band["Description"] == "Test R" and band["MagSys"] == "Vega"
    assert band["CalibrationReference"] == [{"ref": "bibcode:2003AJ....126.1090C", "label": "Cohen 2003"}]
    assert band["specified"]["ZeroPoint"]["Vega"]["Jy"] == 3600.0
    assert band["svo"]["curve"] == "identical" and band["svo"]["matched_by"] == "FilterID"
    # The header row stays authoritative; SVO's disagreement is only reported.
    assert band["svo"]["DetectorType"] == "photon" and "DetectorType_note" in band["svo"]
    assert band["curve"]["DetectorType"] == "energy"
    assert band["curve"]["sources"] == ["url:https://example.org/r.dat"]
    assert info["description"] == "The test camera." and info["facility_description"] == "A test facility."
    assert info["acknowledgements"][0]["references"] == ["bibcode:2012ivoa.rept.1015R"]
    # Data-file URLs are curve provenance, not citations; URLs never need resolving.
    assert "url:https://example.org/r.dat" not in band["references"]
    assert info["acknowledgements"][0]["applies_to"] == "curves and metadata"
    assert set(info["bibliography"]) == set(info["references"])
    assert info["bibliography"]["url:http://example.org/cam.html"] == {
        "type": "web", "url": "http://example.org/cam.html", "short": "Camera page"}
    assert sources.unresolved == set()


def test_name_match_needs_agreeing_pivot_and_a_single_candidate(tmp_path):
    svo_dir = _archive(tmp_path)
    sources = InfoSources(tmp_path, svo_dir=svo_dir, curated_dir=tmp_path / "curated", references={})
    header = {}
    assert match_svo(sources, "Test", "Cam", "R_wide", header, None, 6010.0)[2] == "band name and pivot"
    assert match_svo(sources, "Test", "Cam", "R_wide", header, None, 6300.0)[0] is None
    assert match_svo(sources, "Test", "Cam", "R_wide", header, None, None)[0] is None


def test_original_curves_acknowledge_svo_for_metadata_only(tmp_path):
    svo_dir = _archive(tmp_path)
    inst = tmp_path / "data" / "filters" / "Test" / "Cam"
    inst.mkdir(parents=True)
    curve = inst / "R.dat"
    curve.write_text("Wavelength,Transmission,FilterID=Test/Cam.R;ComponentsSource=original\n5000,0\n6000,0.9\n7000,0\n")
    sources = InfoSources(tmp_path, svo_dir=svo_dir, curated_dir=tmp_path / "curated", references={})
    info = compose_info(sources, inst, [curve], {"R": {"WavelengthPivot": 6000.0}})
    assert info["bands"]["R"]["svo"]["curve"] == "differs"
    assert info["acknowledgements"][0]["applies_to"] == "metadata"
    assert "bibcode:2012ivoa.rept.1015R" not in info["references"]
    assert "bibcode:2012ivoa.rept.1015R" in info["bibliography"]


def test_captures_decode_by_their_declared_charset():
    page = '<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">Ma\u00edz Apell\u00e1niz'
    assert decode(page.encode("latin-1")).endswith("Maíz Apellániz")
    assert decode("Maíz".encode()) == "Maíz"


def test_tidy_removes_crossref_html():
    from filter_info.references import tidy
    entry = tidy({"journal": "A&amp;A", "title": "The <i>Gaia</i> passbands",
                  "bibtex": "@article{x, title={<i>Planck</i> results}, journal={Astronomy &amp; Astrophysics}}"})
    assert entry["journal"] == "A&A" and entry["title"] == "The Gaia passbands"
    assert entry["bibtex"] == r"@article{x, title={\textit{Planck} results}, journal={Astronomy \& Astrophysics}}"


def test_curated_acknowledgements_and_licence_come_through(tmp_path):
    svo_dir = _archive(tmp_path)
    curated = tmp_path / "curated"
    curated.mkdir()
    (curated / "Test__Cam.json").write_text(json.dumps({
        "acknowledgements": [{"for": "Test Survey", "text": "This publication makes use of data from Test.",
                              "url": "https://example.org/ack", "references": ["bibcode:2003AJ....126.1090C"]}],
        "licence": {"redistribution": "attribution", "terms": "Free with credit", "url": "https://example.org/terms"}}))
    inst = tmp_path / "data" / "filters" / "Test" / "Cam"
    inst.mkdir(parents=True)
    curve = inst / "R.dat"
    curve.write_text(_curve_text("FilterID=Test/Cam.R;ComponentsSource=svo"))
    sources = InfoSources(tmp_path, svo_dir=svo_dir, curated_dir=curated, references={})
    info = compose_info(sources, inst, [curve], {"R": {"WavelengthPivot": 6000.0}})
    assert [a["for"] for a in info["acknowledgements"]] == ["Test Survey", "SVO Filter Profile Service"]
    assert info["acknowledgements"][0]["url"] == "https://example.org/ack"
    assert info["licence"]["redistribution"] == "attribution"
    assert "bibcode:2003AJ....126.1090C" in info["references"]


def test_curated_star_band_fields_apply_to_every_band(tmp_path):
    svo_dir = _archive(tmp_path)
    curated = tmp_path / "curated"
    curated.mkdir()
    (curated / "Test__Cam.json").write_text(json.dumps({"bands": {
        "*": {"references": ["bibcode:2003AJ....126.1090C"], "PhotSystem": "Test"},
        "R": {"PhotSystem": "Test R"}}}))
    inst = tmp_path / "data" / "filters" / "Test" / "Cam"
    inst.mkdir(parents=True)
    curves = []
    for band in ("R", "I"):
        curve = inst / f"{band}.dat"
        curve.write_text(_curve_text(f"FilterID=Test/Cam.{band};ComponentsSource=original"))
        curves.append(curve)
    sources = InfoSources(tmp_path, svo_dir=svo_dir, curated_dir=curated, references={})
    info = compose_info(sources, inst, curves, {"R": {"WavelengthPivot": 6000.0}, "I": {"WavelengthPivot": 8000.0}})
    assert "*" not in info["bands"]
    assert info["bands"]["R"]["PhotSystem"] == "Test R"
    assert info["bands"]["I"]["PhotSystem"] == "Test"
    assert "bibcode:2003AJ....126.1090C" in info["bands"]["I"]["references"]


def test_curated_band_can_repoint_or_clear_the_svo_match_and_its_note(tmp_path):
    svo_dir = _archive(tmp_path)
    curated = tmp_path / "curated"
    curated.mkdir()
    inst = tmp_path / "data" / "filters" / "Test" / "Cam"
    inst.mkdir(parents=True)
    (inst / "R.dat").write_text(_curve_text("FilterID=Test/Cam.R;DetectorType=energy;DetectorTypeSource=original"))
    (inst / "V.dat").write_text(_curve_text("FilterID=Test/Cam.V;DetectorType=photon;DetectorTypeSource=original"))
    (curated / "Test__Cam.json").write_text(json.dumps({"bands": {
        "R": {"DetectorType_note": "SVO split this band into R and R_fil; NJM's R is the full curve."},
        "V": {"svo_filterID": "Test/Cam.R"}}}))
    sources = InfoSources(tmp_path, svo_dir=svo_dir, curated_dir=curated, references={})
    props = {"R": {"WavelengthPivot": 6000.0}, "V": {"WavelengthPivot": 6000.0}}
    info = compose_info(sources, inst, sorted(inst.glob("*.dat")), props)
    assert info["bands"]["R"]["svo"]["DetectorType_note"].startswith("SVO split")
    assert "DetectorType_note" not in info["bands"]["R"]  # override consumed, not copied
    assert info["bands"]["V"]["svo"]["matched_by"] == "curated"
    (curated / "Test__Cam.json").write_text(json.dumps({"bands": {"R": {"svo_filterID": None, "DetectorType_note": ""}}}))
    info = compose_info(sources, inst, sorted(inst.glob("*.dat")), props)
    assert "svo" not in info["bands"]["R"] and "R" in info["svo_unmatched"]


def test_single_letter_bands_match_svo_with_their_case(tmp_path):
    svo_dir = _archive(tmp_path)
    sources = InfoSources(tmp_path, svo_dir=svo_dir, curated_dir=tmp_path / "curated", references={})
    assert match_svo(sources, "Test", "Cam", "R", {}, None, 6000.0)[2] == "band name and pivot"
    assert match_svo(sources, "Test", "Cam", "r", {}, None, 6000.0)[0] is None
