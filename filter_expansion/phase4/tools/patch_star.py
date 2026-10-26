p="filter_info/compose.py"; s=open(p).read()
old="""        entry.update((curated.get("bands") or {}).get(band, {}))\n"""
new="""        # Curated "*" fields apply to every band (e.g. the paper that publishes the whole set);
        # a band's own curated fields override them.
        curated_bands = curated.get("bands") or {}
        entry.update(curated_bands.get("*", {}))
        entry.update(curated_bands.get(band, {}))\n"""
assert s.count(old)==1; open(p,"w").write(s.replace(old,new))
p="filter_info/README.md"; s=open(p).read()
old="`description`, `facility_description`, `notes`, `links`, `references`, `bands.<band>.<field>`,"
assert old in s
open(p,"w").write(s.replace(old,"`description`, `facility_description`, `notes`, `links`, `references`, `bands.<band>.<field>`\n     (`bands.*.<field>` applies to every band unless the band sets its own),"))
p="tests/test_filter_info.py"; s=open(p).read()
s+='''

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
'''
open(p,"w").write(s)
