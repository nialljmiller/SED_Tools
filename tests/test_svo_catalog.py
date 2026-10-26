"""The SVO-compatible service catalogue: filterIDs of _svo/_old copies."""
import importlib.util
import json
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parents[1] / "filter_expansion" / "svc" / "build_catalog.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_catalog", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_curve(path, filter_id, rows="4000,0.1\n5000,0.9\n6000,0.1\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Wavelength,Transmission,FilterID={filter_id};DetectorType=photon\n{rows}")


def build(tmp_path, monkeypatch):
    data = tmp_path / "data"
    filters = data / "filters"
    write_curve(filters / "Euclid/NISP/Y.dat", "Euclid/NISP.Y")
    write_curve(filters / "Euclid/NISP/Y_svo.dat", "Euclid/NISP.Y", "4000,0.2\n5000,0.8\n6000,0.2\n")
    write_curve(filters / "JWST/NIRCam/F444W_old.dat", "JWST/NIRCam.F444W")
    write_curve(filters / "JWST/NIRCam/F444W.dat", "JWST/NIRCam.F444W")
    write_curve(filters / "BOK/90prime/g.dat", "BOK/BASS.g")
    write_curve(filters / "BOK/BASS/g.dat", "BOK/BASS.g")
    (filters / "Euclid/NISP/info.json").write_text(json.dumps({"bands": {
        "Y": {"MagSys": "Vega", "PhotCalID": "Euclid/NISP.Y/Vega"},
        "Y_svo": {"MagSys": "Vega", "PhotCalID": "Euclid/NISP.Y/Vega"}}}))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["build_catalog.py", "--data", str(data), "--out", str(out),
                                      "--aliases", ""])
    assert load_builder().main() == 0
    catalog = json.loads((out / "catalog.json").read_text())
    col = {k: catalog["fields"].index(k) for k in catalog["fields"]}
    rows = {row[col["filterID"]]: row for row in catalog["rows"]}
    return catalog, col, rows


def test_copies_get_their_own_path_ids(tmp_path, monkeypatch):
    catalog, col, rows = build(tmp_path, monkeypatch)
    assert sorted(rows) == ["BOK/90prime.g", "BOK/BASS.g", "Euclid/NISP.Y", "Euclid/NISP.Y_svo",
                            "JWST/NIRCam.F444W", "JWST/NIRCam.F444W_old"]
    assert not [w for w in catalog["warnings"] if "duplicate" in w]
    assert rows["Euclid/NISP.Y"][col["NJMID"]] == "Euclid/NISP.Y"
    assert rows["JWST/NIRCam.F444W"][col["NJMID"]] == "JWST/NIRCam.F444W"
    assert rows["BOK/BASS.g"][col["NJMID"]] == "BOK/BASS.g"
    assert rows["Euclid/NISP.Y_svo"][col["Comments"]].startswith(
        "SVO FPS copy of NJM curve Euclid/NISP.Y; SVO serves it as Euclid/NISP.Y")
    assert rows["JWST/NIRCam.F444W_old"][col["Comments"]].startswith(
        "Superseded curve; the current one is JWST/NIRCam.F444W")
    assert rows["BOK/90prime.g"][col["Comments"]].startswith("Same curve as BOK/BASS.g")
    assert rows["Euclid/NISP.Y_svo"][col["Band"]] == "Y_svo"


def test_shared_photcal_id_stays_with_the_plain_curve(tmp_path, monkeypatch):
    catalog, col, rows = build(tmp_path, monkeypatch)
    assert rows["Euclid/NISP.Y_svo"][col["PhotCalID"]] == ""
    assert "Euclid/NISP.Y/Vega" not in catalog["photcal"]
