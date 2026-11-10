"""Correcting or converting the detector type of installed curves, in place."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import requests
from numpy.testing import assert_allclose

from helpers_cli_menus import answer
from sed_tools import Filters
from sed_tools.cli.filters import review_detector_types
from sed_tools.combine_filters import write_filter_index
from sed_tools.filters.properties import curve_properties, filter_properties, load_vega, properties_record
from sed_tools.njm_mirror import cache_root
from sed_tools.filters.compose import OPTICS, QE, InstrumentManifest, add_component, build_product, read_curve
from sed_tools.filter_metadata import ENERGY, PHOTON, read_filter_header, tag_filter_tree, write_filter_file

WL = np.linspace(4000.0, 6000.0, 201)
G = np.exp(-0.5 * ((WL - 4800.0) / 300.0) ** 2) * 0.9
R = np.exp(-0.5 * ((WL - 5300.0) / 300.0) ** 2) * 0.7
I = np.exp(-0.5 * ((WL - 5600.0) / 200.0) ** 2) * 0.8
REFERENCE = Path(__file__).parent / "golden" / "inputs" / "reference"  # NJM's vega_flam.csv and sun_flam.csv
SPECTRA = {"vega_spectrum": "vega/vega_flam.csv", "sun_spectrum": "sun/sun_flam.csv"}


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The filter root, with NJM's Vega and Sun spectra in its catalogue cache as if fetched just now."""
    root = tmp_path / "filters"
    monkeypatch.setattr(Filters, "_filter_root", root)
    for rel in SPECTRA.values():
        cached = cache_root(root) / rel
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REFERENCE / Path(rel).name, cached)
        cached.with_name(cached.name + ".http.json").write_text(json.dumps({"checked": time.time()}))
    return root


def _set(root, *, manifest=True, energy_band=False):
    """SLOAN/SDSS: photon-counting filter curves g and r (and an energy-counting i), with NJM's catalogue files."""
    inst = root / "SLOAN" / "SDSS"
    inst.mkdir(parents=True)
    curves = {"g": (G, PHOTON), "r": (R, PHOTON), **({"i": (I, ENERGY)} if energy_band else {})}
    for band, (tr, detector) in curves.items():
        write_filter_file(inst / f"{band}.dat", WL, tr, detector_type=detector, source="svo",
                          filter_id=f"SLOAN/SDSS.{band}", wavelength_unit="Angstrom",
                          extra={"Components": "filter", "ComponentsSource": "svo",
                                 "Sources": f"https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=SLOAN/SDSS.{band}"})
    write_filter_index(inst, "SDSS")
    record = {band: {"DetectorType": d, "DetectorTypeSource": "svo", "WavelengthPivot": 1.0, "DisplayColor": "#000"}
              for band, (_, d) in curves.items()}
    definitions = {key: {"url": url} for key, url in SPECTRA.items()}
    (inst / "properties.json").write_text(json.dumps({"bands": record, "definitions": definitions}))
    (inst / "info.json").write_text(json.dumps({"bands": {b: {"curve": {"DetectorType": d, "DetectorTypeSource": "svo"}}
                                                          for b, (_, d) in curves.items()}}))
    if manifest:
        m = InstrumentManifest.load(inst)
        add_component(inst, OPTICS, [3000.0, 7000.0], [0.8, 0.8], name="optics", manifest=m)
        add_component(inst, QE, [3000.0, 7000.0], [0.6, 0.6], name="ccd", manifest=m)
        m.save(inst)
    return inst


def _samples(path):
    return path.read_bytes().split(b"\n", 1)[1]


def _mesa_weights(path):
    """What MESA integrates: an energy-counting curve as T/lambda at the same peak, a photon-counting one as is."""
    curve = read_curve(path)
    if read_filter_header(path)["DetectorType"] != ENERGY:
        return curve.transmission
    weights = curve.transmission / curve.wavelength
    return weights * curve.transmission.max() / weights.max()


def test_correct_changes_only_the_label(root):
    inst = _set(root)
    before = _samples(inst / "g.dat")
    rows = Filters.correct_detector_type(inst, "energy", bands=["g"])
    assert [(r["band"], r["previous"], r["changed"]) for r in rows] == [("g", PHOTON, True)]
    assert _samples(inst / "g.dat") == before
    header = read_filter_header(inst / "g.dat")
    assert (header["DetectorType"], header["DetectorTypeSource"], header["DetectorTypeOriginal"]) == (
        ENERGY, "user", PHOTON)
    assert "DetectorTypeConversion" not in header
    assert read_filter_header(inst / "r.dat")["DetectorType"] == PHOTON  # not asked for
    # The records beside the curves follow, and a correction is the detector's type from now on.
    manifest = InstrumentManifest.load(inst)
    assert (manifest.detector_types["g"], manifest.detector_type_sources["g"]) == (ENERGY, "user")
    assert json.loads((inst / "info.json").read_text())["bands"]["g"]["curve"]["DetectorType"] == ENERGY
    # Its properties are recomputed exactly as NJM computes them, Vega and Sun values included; colours are kept.
    props = json.loads((inst / "properties.json").read_text())["bands"]["g"]
    vega, sun = load_vega(REFERENCE / "vega_flam.csv"), load_vega(REFERENCE / "sun_flam.csv")
    assert props == {**properties_record(inst / "g.dat", vega, sun), "DisplayColor": "#000"}
    assert props["DetectorType"] == ENERGY and None not in (props["ZeroPointVega_flam"], props["Fsun"])
    assert read_filter_header(build_product(inst, ["filter", "optics", "qe"], bands=["g"]) / "g.dat")[
        "DetectorType"] == ENERGY


@pytest.mark.parametrize("manifest", [True, False])
def test_convert_to_energy_keeps_mesa_photometry_and_products(root, manifest):
    inst = _set(root, manifest=manifest)
    manifest_before = (inst / "manifest.json").read_bytes() if manifest else None
    first_lines = {band: (inst / f"{band}.dat").read_text().split("\n", 1)[0] for band in "gr"}
    Filters.convert_detector_type(inst, "energy")
    for band, native in (("g", G), ("r", R)):
        # Line 1 keeps all its metadata and gains the conversion record.
        assert (inst / f"{band}.dat").read_text().split("\n", 1)[0] == first_lines[band].replace(
            "DetectorType=photon;DetectorTypeSource=svo", "DetectorType=energy;DetectorTypeSource=user"
        ) + ";DetectorTypeOriginal=photon;DetectorTypeConversion=lambda"
        curve = read_curve(inst / f"{band}.dat").transmission
        assert np.isclose(curve.max(), native.max())
        assert_allclose(curve, native * WL * native.max() / (native * WL).max(), rtol=1e-12)
        assert_allclose(_mesa_weights(inst / f"{band}.dat"), native, rtol=1e-12)
        # SED_Tools' own weighting agrees: the same pivot and AB zero point as the native curve.
        props, native_props = curve_properties(inst / f"{band}.dat"), filter_properties(WL, native, PHOTON)
        for key in ("WavelengthPivot", "ZeroPointAB_flam"):
            assert np.isclose(props[key], native_props[key], rtol=1e-10)
    # The same instrument in the other convention: only the detector fields of NJM's records change.
    assert json.loads((inst / "properties.json").read_text())["bands"]["g"] == {
        "DetectorType": ENERGY, "DetectorTypeSource": "user", "WavelengthPivot": 1.0, "DisplayColor": "#000"}
    assert json.loads((inst / "info.json").read_text())["bands"]["g"]["curve"] == {
        "DetectorType": ENERGY, "DetectorTypeSource": "user", "DetectorTypeOriginal": PHOTON,
        "DetectorTypeConversion": "lambda"}
    if manifest:
        assert (inst / "manifest.json").read_bytes() == manifest_before  # the detector is unchanged
    inferred = InstrumentManifest.load(inst)
    assert inferred.detector_types["g"] == PHOTON
    assert manifest or inferred.detector_type_sources["g"] == "converted"
    # Products composed afterwards are as they were: own numbers, the detector's type.
    recipes = [["published"], ["filter"]] + ([["filter", "optics", "qe"]] if manifest else [])
    for recipe in recipes:
        product = build_product(inst, recipe)
        assert read_filter_header(product / "g.dat")["DetectorType"] == PHOTON
        factor = 0.48 if "qe" in recipe else 1.0
        assert_allclose(read_curve(product / "g.dat").transmission, G * factor, rtol=1e-12)


def test_convert_to_photon_and_back(root):
    inst = _set(root, energy_band=True)
    rows = Filters.convert_detector_type(inst, "photon")
    assert [r["band"] for r in rows if r["changed"]] == ["i"]  # g and r are photon-counting already
    header = read_filter_header(inst / "i.dat")
    assert (header["DetectorType"], header["DetectorTypeOriginal"], header["DetectorTypeConversion"]) == (
        PHOTON, ENERGY, "1/lambda")
    photon = read_curve(inst / "i.dat").transmission
    mesa_native = I / WL * I.max() / (I / WL).max()  # MESA's weights for the energy-counting original
    assert_allclose(photon, mesa_native, rtol=1e-12)
    assert np.isclose(photon.max(), I.max())

    Filters.convert_detector_type(inst / "i.dat", "energy")
    header = read_filter_header(inst / "i.dat")
    assert header["DetectorType"] == ENERGY and "DetectorTypeConversion" not in header
    assert header["DetectorTypeOriginal"] == ENERGY
    assert_allclose(read_curve(inst / "i.dat").transmission, I, rtol=1e-12)


def test_correct_then_convert_records_the_net_factor(root):
    inst = _set(root, manifest=False)
    Filters.correct_detector_type(inst / "g.dat", "energy")
    Filters.convert_detector_type(inst / "g.dat", "photon")
    header = read_filter_header(inst / "g.dat")
    assert (header["DetectorType"], header["DetectorTypeOriginal"], header["DetectorTypeConversion"]) == (
        PHOTON, PHOTON, "1/lambda")
    assert_allclose(read_curve(inst / "g.dat").transmission, G / WL * G.max() / (G / WL).max(), rtol=1e-12)
    # Without a manifest, the headers still say the detector is the user's energy-counting one.
    inferred = InstrumentManifest.load(inst)
    assert (inferred.detector_types["g"], inferred.detector_type_sources["g"]) == (ENERGY, "user")


def test_correct_without_njm_leaves_the_vega_and_sun_values_unknown(tmp_path, monkeypatch, capsys):
    def unreachable(self, method, url, **kwargs):
        raise requests.ConnectionError("NJM is down")

    monkeypatch.setattr(requests.Session, "request", unreachable)
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # the client's retries
    root = tmp_path / "elsewhere"  # nothing cached under this filter root
    inst = _set(root)
    Filters.correct_detector_type(inst, "energy", bands=["g"], filter_root=root)
    props = json.loads((inst / "properties.json").read_text())["bands"]["g"]
    assert props == {**properties_record(inst / "g.dat"), "DisplayColor": "#000"}
    assert props["DetectorType"] == ENERGY and props["ZeroPointVega_flam"] is None and props["Fsun"] is None
    out = capsys.readouterr().out
    assert "vega/vega_flam.csv is not cached and NJM is unreachable: the Vega-based properties" in out


def test_edit_refuses_unknown_bands_types_and_untagged_conversions(root):
    inst = _set(root)
    with pytest.raises(ValueError, match="no band z"):
        Filters.correct_detector_type(inst, "energy", bands=["z"])
    with pytest.raises(FileNotFoundError):
        Filters.correct_detector_type(root / "SLOAN" / "SDS", "energy")
    with pytest.raises(ValueError, match="Unknown detector type"):
        Filters.convert_detector_type(inst, "bolometer")
    (inst / "u.dat").write_text("Wavelength,Transmission\n4000.0,0.1\n5000.0,0.5\n")
    with pytest.raises(ValueError, match="No DetectorType"):
        Filters.convert_detector_type(inst / "u.dat", "energy")
    (inst / "u.dat").write_text("Wavelength,Transmission,DetectorType=photon\n4000.0,0.1\n4500.0,abc\n5000.0,0.5\n")
    with pytest.raises(ValueError, match=r"u\.dat, line 3: expected two numbers, found '4500\.0,abc'"):
        Filters.convert_detector_type(inst / "u.dat", "energy")


def test_retagging_keeps_edited_types(root):
    class SVO:
        def get_filter(self, filter_id):
            return None

        def facility_filter_ids(self, facility):
            return []

    inst = _set(root)
    Filters.convert_detector_type(inst, "energy", bands=["g"])
    Filters.correct_detector_type(inst, "energy", bands=["r"])
    before = {band: (inst / f"{band}.dat").read_bytes() for band in "gr"}
    tag_filter_tree(inst, client=SVO(), force=True)
    assert {band: (inst / f"{band}.dat").read_bytes() for band in "gr"} == before


def test_build_product_detector_type_still_only_relabels(tmp_path):
    out = build_product(_set(tmp_path), ["published"], detector_type="energy")
    assert out.name == "published+detector_energy"
    assert read_filter_header(out / "g.dat")["DetectorType"] == ENERGY
    assert_allclose(read_curve(out / "g.dat").transmission, G, rtol=0, atol=0)


# ---- The interactive review after a download or a compose
QUESTION = ("This filter set is known to be used with photon-counting detectors (svo). "
            "Do you wish to edit/correct this? [y/N] ")


def test_review_default_changes_nothing(root, monkeypatch):
    inst = _set(root)
    before = (inst / "g.dat").read_bytes()
    prompts = answer(monkeypatch, "")
    review_detector_types(inst, root)
    assert prompts == [QUESTION] and (inst / "g.dat").read_bytes() == before


def test_review_correct(root, monkeypatch):
    inst = _set(root)
    before = _samples(inst / "g.dat")
    prompts = answer(monkeypatch, "y", "correct")
    review_detector_types(inst, root)
    assert prompts == [QUESTION, "Correct the label only, or convert the curves? [correct/convert]: "]
    assert read_filter_header(inst / "g.dat")["DetectorType"] == ENERGY and _samples(inst / "g.dat") == before


def test_review_convert(root, monkeypatch, capsys):
    inst = _set(root)
    answer(monkeypatch, "y", "convert")
    review_detector_types(inst, root)
    assert "Converting to energy-counting with T_E(λ) = T_P(λ)·λ, rescaled to the same peak." in capsys.readouterr().out
    assert read_filter_header(inst / "r.dat")["DetectorTypeConversion"] == "lambda"


def test_review_of_a_mixed_set_asks_for_the_type(root, monkeypatch):
    inst = _set(root, energy_band=True)
    prompts = answer(monkeypatch, "y", "convert", "energy")
    review_detector_types(inst, root, "This product")
    assert prompts[0] == ("This product is known to be used with photon-counting detectors (svo) for g, r and "
                          "energy-counting detectors (svo) for i. Do you wish to edit/correct this? [y/N] ")
    assert prompts[2] == "Change them to energy or photon: "
    assert {b: read_filter_header(inst / f"{b}.dat")["DetectorType"] for b in "gri"} == dict.fromkeys("gri", ENERGY)
    assert "DetectorTypeConversion" not in read_filter_header(inst / "i.dat")


def test_filters_detector_correct_and_convert_options(root, monkeypatch, capsys):
    from sed_tools import cli

    inst = _set(root)
    monkeypatch.setattr("sys.argv", ["sed-tools", "filters-detector", str(inst), "--convert", "energy", "--bands", "g"])
    cli.main()
    assert "1 files, 1 changed" in capsys.readouterr().out
    assert read_filter_header(inst / "g.dat")["DetectorTypeConversion"] == "lambda"
    monkeypatch.setattr("sys.argv", ["sed-tools", "filters-detector", str(inst / "r.dat"), "--correct", "energy"])
    cli.main()
    assert read_filter_header(inst / "r.dat")["DetectorType"] == ENERGY


def test_download_flow_reviews_the_detector_type(root, monkeypatch):
    from types import SimpleNamespace

    from sed_tools.cli.filters import _fetch_instrument

    inst = _set(root)
    prompts = answer(monkeypatch, "y", "correct", "")  # then decline the composed version
    failures = []
    _fetch_instrument(SimpleNamespace(published=lambda f, i: inst, root=root), "SLOAN", "SDSS", failures, offer=True)
    assert prompts[0] == QUESTION and prompts[-1].startswith("Build a composed version") and not failures
    assert read_filter_header(inst / "g.dat")["DetectorType"] == ENERGY
