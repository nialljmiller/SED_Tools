"""NJM catalogue access: search, properties/info caching, citations and plots, with mocked HTTP."""

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from sed_tools.api import Filters
from sed_tools.filter_catalogue import CatalogueError, FilterCatalogue, cache_root, format_bibtex
from sed_tools.filter_library import FilterLibrary
from sed_tools.njm_filter_grabber import NJMFilterGrabber

BASE = "https://njm.invalid/sed_tools"


def band(pivot, fwhm, detector="photon", ab=0.1):
    return {"WavelengthPivot": pivot, "FWHM": fwhm,
            "WavelengthMin": pivot - 0.75 * fwhm, "WavelengthMax": pivot + 0.75 * fwhm,
            "DetectorType": detector, "ABminusVega": ab, "DisplayColor": "#123456"}


SVO_ACK = {"for": "SVO Filter Profile Service", "text": "This research has made use of the SVO FPS.",
           "references": ["bibcode:2012ivoa.rept.1015R"], "applies_to": "curves and metadata"}


def catalogue_files(generated="2026-10-26T10:00:00"):
    index = {"generated": generated, "filters": {
        "facilities": ["Fac", "Other"],
        "properties_definitions": {"units": {"WavelengthPivot": "Angstrom", "FWHM": "Angstrom"},
                                   "vega_spectrum": {"url": "vega/vega_flam.csv"}},
        "facilities_detailed": {
            "Fac": {"instruments": {
                "Cam": {"filters": ["g", "r", "i"], "detector_types": {"photon": 3},
                        "properties": "properties.json", "info": "info.json"},
                "Old": {"filters": ["U"], "detector_types": {"energy": 1}, "properties": None, "info": None},
            }},
            "Other": {"instruments": {
                "IR": {"filters": ["J", "K"], "detector_types": {"energy": 2},
                       "properties": "properties.json", "info": "info.json"},
            }},
        }}}
    cam_props = {"bands": {"g": band(4800, 1200), "r": band(6200, 1300, ab=None), "i": band(7600, 1500, ab=0.4)},
                 "definitions": {}}
    ir_props = {"bands": {"J": band(12400, 2100, "energy", 0.9), "K": band(21600, 2800, "energy", 1.85)},
                "definitions": {}}
    cam_info = {
        "schema": 1, "facility": "Fac", "instrument": "Cam", "description": "A camera.",
        "licence": {"redistribution": "public", "terms": "Free to use.", "url": "https://fac.invalid/terms"},
        "references": ["bibcode:2020Cam.....1A", "bibcode:2021gband...2B", "bibcode:2012ivoa.rept.1015R"],
        "acknowledgements": [{"for": "Fac", "text": "We thank   Fac.", "references": [], "applies_to": "all bands"},
                             SVO_ACK],
        "bands": {"g": {"Description": "Cam g", "MagSys": "AB", "references": ["bibcode:2021gband...2B"]},
                  "r": {"Description": "Cam r", "MagSys": "AB", "references": []},
                  "i": {"Description": "Cam i", "MagSys": "AB", "references": []}},
        "bibliography": {
            "bibcode:2020Cam.....1A": {"short": "Author 2020", "title": "The camera", "bibcode": "2020Cam.....1A",
                                       "bibtex": "@article{2020Cam.....1A, title={The camera}}"},
            "bibcode:2021gband...2B": {"short": "Band 2021", "title": "The g band",
                                       "bibtex": "@article{2021gband...2B, title={The g band}}"},
            "bibcode:2012ivoa.rept.1015R": {"short": "Rodrigo et al. 2012", "title": "SVO FPS",
                                            "url": "https://ui.adsabs.harvard.edu/abs/2012ivoa.rept.1015R"},
        }}
    ir_info = {
        "schema": 1, "facility": "Other", "instrument": "IR", "description": "An infrared imager.",
        "licence": {"redistribution": "attribution", "terms": "Cite us."},
        "references": ["arxiv:2001.00001", "bibcode:2012ivoa.rept.1015R"],
        "acknowledgements": [SVO_ACK],
        "bands": {"J": {"references": []}, "K": {"references": []}},
        "bibliography": {
            "arxiv:2001.00001": {"short": "Author 2020", "same_as": "bibcode:2020Cam.....1A"},
            "bibcode:2020Cam.....1A": {"short": "Author 2020", "title": "The camera",
                                       "bibtex": "@article{2020Cam.....1A, title={The camera}}"},
            "bibcode:2012ivoa.rept.1015R": {"short": "Rodrigo et al. 2012", "title": "SVO FPS"},
        }}
    vega = "WAVELENGTH,FLUX\n" + "".join(f"{w:.1f},{1e-9 * (5000.0 / w) ** 2:.6e}\n"
                                          for w in np.arange(3000.0, 30000.0, 100.0))
    return {
        "index.json": json.dumps(index),
        "filters/Fac/Cam/properties.json": json.dumps(cam_props),
        "filters/Fac/Cam/info.json": json.dumps(cam_info),
        "filters/Other/IR/properties.json": json.dumps(ir_props),
        "filters/Other/IR/info.json": json.dumps(ir_info),
        "vega/vega_flam.csv": vega,
    }


class FakeNJM:
    """Serves ``files`` under BASE with ETags; records each request and its headers."""

    def __init__(self, files, delay=0.0):
        self.files = dict(files)
        self.requests = []
        self.delay = delay
        self.fail = False
        self._lock = threading.Lock()
        self.active = self.max_active = 0

    def etag(self, rel):
        return f'"{abs(hash(self.files[rel]))}"'

    def get(self, url, timeout=None, headers=None, **kwargs):
        rel = url[len(BASE) + 1:]
        with self._lock:
            self.requests.append((rel, dict(headers or {})))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise requests.ConnectionError("mirror down")
            if rel not in self.files:
                return self.response(404, b"")
            tag = self.etag(rel)
            if (headers or {}).get("If-None-Match") == tag:
                return self.response(304, b"", {"ETag": tag})
            return self.response(200, self.files[rel].encode(), {"ETag": tag,
                                                                 "Last-Modified": "Mon, 26 Oct 2026 04:45:30 GMT"})
        finally:
            with self._lock:
                self.active -= 1

    @staticmethod
    def response(status, content, headers=None):
        def raise_for_status():
            if status >= 400:
                raise requests.HTTPError(str(status), response=SimpleNamespace(status_code=status, headers={}))
        return SimpleNamespace(status_code=status, content=content, text=content.decode(), headers=headers or {},
                               raise_for_status=raise_for_status, json=lambda: json.loads(content))

    def fetched(self, rel):
        return [h for r, h in self.requests if r == rel]


@pytest.fixture
def njm(monkeypatch):
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.time", SimpleNamespace(sleep=lambda s: None))
    monkeypatch.setattr(Filters, "_catalogues", {})
    return FakeNJM(catalogue_files())


def grabber_for(root, server):
    grabber = NJMFilterGrabber(base_dir=str(root))
    grabber.base_url, grabber.filters_url = BASE, f"{BASE}/filters"
    grabber.session = SimpleNamespace(get=server.get)
    return grabber


@pytest.fixture
def root(tmp_path, njm):
    """A filter root whose Filters catalogue talks to the fake NJM."""
    filters = tmp_path / "filters"
    filters.mkdir()
    key = (str(filters.resolve()), False)
    Filters._catalogues[key] = FilterCatalogue(key[0], mirror=grabber_for(filters, njm))
    return filters


def write_curve(path, centre, width, header="Wavelength,Transmission"):
    path.parent.mkdir(parents=True, exist_ok=True)
    lam = np.linspace(centre - 2 * width, centre + 2 * width, 81)
    t = np.exp(-0.5 * ((lam - centre) / (width / 2.355)) ** 2)
    path.write_text(header + "\n" + "".join(f"{w:.3f},{v:.6f}\n" for w, v in zip(lam, t)))


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------
def test_search_matches_names_in_the_index_before_fetching_properties(root, njm):
    result = Filters.search(facility="other", filter_root=root)
    assert list(result["band"]) == ["J", "K"]
    assert list(result.columns[:5]) == ["facility", "instrument", "band", "WavelengthPivot", "WavelengthMin"]
    assert njm.fetched("filters/Other/IR/properties.json")
    assert not njm.fetched("filters/Fac/Cam/properties.json")

    assert list(Filters.search("cam/r", filter_root=root)["band"]) == ["r"]


def test_search_by_wavelength_pivot_fwhm_detector_and_any_property(root, njm):
    covers = Filters.search(wavelength=6200, filter_root=root)
    assert list(covers["band"]) == ["r"]
    assert list(Filters.search(wavelength=(7200, 13000), filter_root=root)["band"]) == ["i", "J"]
    assert list(Filters.search(pivot=(4000, 7000), filter_root=root)["band"]) == ["g", "r"]
    assert list(Filters.search(fwhm=(None, 1250), filter_root=root)["band"]) == ["g"]
    # A null Vega quantity never matches, and never raises.
    assert list(Filters.search(where={"ABminusVega": (None, 0.5)}, filter_root=root)["band"]) == ["g", "i"]
    assert list(Filters.search(where={"DetectorType": "ENERGY"}, filter_root=root)["band"]) == ["J", "K"]
    # A permanent catalogue gap is reported apart from fetch failures.
    assert covers.attrs["gaps"] == ["Fac/Old: NJM has no properties file"] and covers.attrs["errors"] == []
    assert covers.attrs["units"]["FWHM"] == "Angstrom"

    njm.requests.clear()
    Filters._catalogues.clear()
    key = (str(root.resolve()), False)
    Filters._catalogues[key] = FilterCatalogue(key[0], mirror=grabber_for(root, njm))
    photon = Filters.search(detector_type="photon", refresh=True, filter_root=root)
    assert list(photon["band"]) == ["g", "r", "i"]
    assert not njm.fetched("filters/Other/IR/properties.json")  # excluded by index.json's detector types


def test_search_accepts_quantities_and_names_only(root, njm):
    from astropy import units as u

    assert list(Filters.search(wavelength=620 * u.nm, filter_root=root)["band"]) == ["r"]
    assert list(Filters.search(pivot=(1, 2) * u.um, filter_root=root)["band"]) == ["J"]
    njm.requests.clear()
    names = Filters.search(band="K", properties=False, filter_root=root)
    assert list(names.columns) == ["facility", "instrument", "band"] and list(names["band"]) == ["K"]
    assert not [r for r, _ in njm.requests if r.endswith("properties.json")]
    with pytest.raises(ValueError):
        Filters.search(fwhm=(1, 2), properties=False, filter_root=root)


def test_bulk_fetch_is_capped_at_six_requests_at_a_time(tmp_path, njm):
    files = catalogue_files()
    index = json.loads(files["index.json"])
    many = {f"I{n}": {"filters": ["b"], "detector_types": {"photon": 1}, "properties": "properties.json"}
            for n in range(20)}
    index["filters"]["facilities_detailed"] = {"Big": {"instruments": many}}
    files = {"index.json": json.dumps(index)}
    files.update({f"filters/Big/I{n}/properties.json": json.dumps({"bands": {"b": band(5000 + n, 100)}})
                  for n in range(20)})
    server = FakeNJM(files, delay=0.02)
    catalogue = FilterCatalogue(tmp_path / "filters", mirror=grabber_for(tmp_path / "filters", server),
                                max_workers=50)
    result = catalogue.search(pivot=(5010, 5020))
    assert list(result["instrument"]) == [f"I{n}" for n in range(10, 20)]
    assert 1 < server.max_active <= 6


# ----------------------------------------------------------------------
# properties.json / info.json and the cache
# ----------------------------------------------------------------------
def test_files_are_cached_beside_the_filter_root_and_revalidated_when_the_index_changes(root, njm):
    props = Filters.properties("fac", "CAM", filter_root=root)  # names resolve ignoring case
    assert props["bands"]["g"]["WavelengthPivot"] == 4800
    cached = cache_root(root) / "filters" / "Fac" / "Cam" / "properties.json"
    assert cached.is_file() and not (root / "Fac").exists()

    Filters.properties("Fac", "Cam", filter_root=root)
    assert len(njm.fetched("filters/Fac/Cam/properties.json")) == 1  # same index generation: no request

    njm.files["index.json"] = catalogue_files(generated="2026-10-27T10:00:00")["index.json"]
    Filters.search(facility="nothing", filter_root=root, refresh=True)  # rereads index.json
    Filters.properties("Fac", "Cam", filter_root=root)
    second = njm.fetched("filters/Fac/Cam/properties.json")
    assert len(second) == 2 and second[1]["If-None-Match"] == njm.etag("filters/Fac/Cam/properties.json")
    assert json.loads(cached.read_text())["bands"]["g"]["WavelengthPivot"] == 4800  # 304 kept the copy

    with pytest.raises(CatalogueError, match="no properties file"):
        Filters.properties("Fac", "Old", filter_root=root)
    with pytest.raises(CatalogueError, match="does not serve"):
        Filters.info("Nope", "Cam", filter_root=root)


def test_offline_and_unreachable_mirror_use_cached_copies(root, njm):
    Filters.info("Fac", "Cam", filter_root=root)
    offline = FilterCatalogue(root, offline=True, mirror=lambda: pytest.fail("offline must not build a client"))
    assert offline.info("Fac", "Cam")["description"] == "A camera."
    with pytest.raises(CatalogueError):
        offline.info("Other", "IR")

    njm.fail = True
    catalogue = FilterCatalogue(root, mirror=grabber_for(root, njm), index_max_age=0)
    assert catalogue.info("Fac", "Cam", refresh=True)["description"] == "A camera."
    with pytest.raises(CatalogueError, match="unreachable"):
        catalogue.info("Other", "IR")


def test_refresh_asks_the_mirror_even_when_its_availability_check_failed(tmp_path, njm):
    grabber = grabber_for(tmp_path, njm)
    catalogue = FilterCatalogue(tmp_path, mirror=grabber)
    catalogue.info("Fac", "Cam")
    grabber._available = False  # e.g. NJM's HEAD answered 403 during a burst block
    njm.files["filters/Fac/Cam/info.json"] = njm.files["filters/Fac/Cam/info.json"].replace("A camera.", "New.")
    assert catalogue.info("Fac", "Cam", prefer_installed=False)["description"] == "A camera."
    assert catalogue.info("Fac", "Cam", refresh=True)["description"] == "New."
    assert len(njm.fetched("filters/Fac/Cam/info.json")) == 2


def test_installed_copy_describes_installed_curves(root, njm):
    local = json.loads(catalogue_files()["filters/Fac/Cam/info.json"])
    local["description"] = "As installed."
    (root / "Fac" / "Cam").mkdir(parents=True)
    (root / "Fac" / "Cam" / "info.json").write_text(json.dumps(local))
    assert Filters.info("Fac", "Cam", filter_root=root)["description"] == "As installed."
    assert Filters.info("Fac", "Cam", filter_root=root, refresh=True)["description"] == "A camera."


def test_download_keeps_catalogue_files_beside_curves_only(tmp_path, njm):
    njm.files.update({"filters/Fac/Cam/Cam": "g.dat\nr.dat\ni.dat\n"})
    for name in "gri":
        njm.files[f"filters/Fac/Cam/{name}.dat"] = "Wavelength,Transmission,DetectorType=photon\n4000,1\n6000,1\n"
    grabber = grabber_for(tmp_path, njm)
    assert grabber.download_filters("Fac", "Cam") == 3
    directory = tmp_path / "Fac" / "Cam"
    assert json.loads((directory / "properties.json").read_text())["bands"]["g"]["FWHM"] == 1200
    assert (directory / "info.json").is_file()
    assert (directory / "Cam").read_text().split() == ["g.dat", "r.dat", "i.dat"]  # MESA index: curves only
    assert FilterLibrary.published_complete(directory)

    from sed_tools.combine_filters import find_filter_sets, write_filter_index
    from sed_tools.models import _build_filter_index

    assert find_filter_sets(tmp_path) == [directory]
    assert set(_build_filter_index(tmp_path)) == {"g", "r", "i"}
    out = tmp_path / "copy"
    out.mkdir()
    for p in directory.iterdir():
        (out / p.name).write_bytes(p.read_bytes())
    assert write_filter_index(out, "copy").read_text().split() == ["g.dat", "i.dat", "r.dat"]

    # Curves already installed are kept, and so is the description that came with them.
    (directory / "info.json").write_text('{"bands": {}, "description": "old"}')
    assert grabber.download_filters("Fac", "Cam") == 0
    assert json.loads((directory / "info.json").read_text())["description"] == "old"


# ----------------------------------------------------------------------
# Citations
# ----------------------------------------------------------------------
def test_citations_merge_references_and_acknowledgements(root, njm):
    result = Filters.citations("Fac/Cam", "Other/IR", filter_root=root)
    keys = [r["key"] for r in result["references"]]
    assert keys == ["bibcode:2020Cam.....1A", "bibcode:2021gband...2B", "bibcode:2012ivoa.rept.1015R"]
    camera = result["references"][0]
    assert camera["used_by"] == ["Fac/Cam", "Other/IR"] and camera["aliases"] == ["arxiv:2001.00001"]
    svo = [a for a in result["acknowledgements"] if a["for"] == "SVO Filter Profile Service"]
    assert len(svo) == 1 and svo[0]["used_by"] == ["Fac/Cam", "Other/IR"]

    bib = format_bibtex(result)
    assert bib.count("@article{") == 2
    assert "% No BibTeX served for bibcode:2012ivoa.rept.1015R" in bib
    assert "@misc" not in bib and "@techreport" not in bib  # nothing made up


def test_citations_for_chosen_bands(root, njm):
    r_only = Filters.citations("Fac/Cam/r", filter_root=root)
    assert [r["key"] for r in r_only["references"]] == ["bibcode:2020Cam.....1A", "bibcode:2012ivoa.rept.1015R"]
    g_too = Filters.citations("Fac/Cam/r", ("Fac", "Cam", "g"), filter_root=root)
    assert "bibcode:2021gband...2B" in [r["key"] for r in g_too["references"]]
    missing = Filters.citations("Fac/Cam/z", "Fac/Old", filter_root=root)
    assert any("no band 'z'" in e for e in missing["errors"]) and any("Fac/Old" in e for e in missing["errors"])
    assert not missing["references"]


def test_citations_default_to_installed_instruments(root, njm):
    write_curve(root / "Fac" / "Cam" / "g.dat", 4800, 1200)
    text = Filters.cite(filter_root=root)
    assert "Citations for Fac/Cam" in text and "We thank Fac." in text and "Free to use." in text
    assert "@article{2020Cam.....1A" in text


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def run_cli(monkeypatch, capsys, *argv):
    from sed_tools import cli

    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    code = cli.main()
    return code, capsys.readouterr().out


def test_cli_search_cite_and_info(root, njm, monkeypatch, capsys):
    code, out = run_cli(monkeypatch, capsys, "filters-search", "--pivot", "4000", "-", "--fwhm", "-", "1400",
                        "--where", "ABminusVega=:0.5", "--format", "json", "--base", str(root))
    data = json.loads(out)
    assert code == 0 and [b["band"] for b in data["bands"]] == ["g"]
    # Fac/Old has no properties file: a gap, reported, but not a failure.
    assert data["errors"] == [] and data["gaps"] == ["Fac/Old: NJM has no properties file"]

    code, out = run_cli(monkeypatch, capsys, "filters-search", "--wavelength", "21000", "--base", str(root))
    assert code == 0 and "K" in out and "1 band in 1 instrument" in out
    assert "[filters] Fac/Old: NJM has no properties file" in out

    code, out = run_cli(monkeypatch, capsys, "filters-cite", "Other/IR", "--format", "bibtex", "--base", str(root))
    assert code == 0 and out.startswith("% No BibTeX served") and "@article{2020Cam.....1A" in out

    write_curve(root / "Fac" / "Cam" / "g.dat", 4800, 1200,
                header="Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=svo")
    code, out = run_cli(monkeypatch, capsys, "filters-info", "Fac/Cam", "--base", str(root), "--offline")
    assert code == 0 and "info.json is not cached and working offline" in out  # the catalogue is an extra
    assert not njm.fetched("filters/Fac/Cam/info.json")
    code, out = run_cli(monkeypatch, capsys, "filters-info", "Fac/Cam", "--base", str(root))
    assert code == 0 and "A camera." in out and "Free to use." in out and "Cam g" in out
    code, out = run_cli(monkeypatch, capsys, "filters-info", "Fac/Cam", "--base", str(root), "--offline", "--json")
    assert json.loads(out)["catalogue"]["info"]["description"] == "A camera."


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def test_plot_uses_filter_colors_and_overlays_vega(root, njm):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from sed_tools.filter_colors import display_colors, instrument_colors

    for name, centre in (("g", 4800), ("r", 6200), ("i", 7600)):
        write_curve(root / "Fac" / "Cam" / f"{name}.dat", centre, 1200)
    for name, centre in (("J", 12400), ("K", 21600)):
        write_curve(root / "Other" / "IR" / f"{name}.dat", centre, 2500)

    ax = Filters.plot("Fac/Cam/r", normalize="peak", vega=True, filter_root=root)
    expected = instrument_colors(sorted((root / "Fac" / "Cam").glob("*.dat")))
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert to_hex(lines["r"].get_color()) == expected["r"]["DisplayColor"]
    assert "Vega f$_\\lambda$ (scaled)" in lines
    assert max(lines["r"].get_ydata()) == pytest.approx(1.0)
    plt.close(ax.figure)

    selection = Filters.search(pivot=(6000, 13000), filter_root=root)
    ax = Filters.plot(selection, filter_root=root)
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert set(lines) == {"Fac/Cam r", "Fac/Cam i", "Other/IR J"}
    from sed_tools.flux_cube_tool import load_filter_curve

    union = {}
    for key in ("Fac/Cam/r", "Fac/Cam/i", "Other/IR/J"):
        curve = load_filter_curve(str(root / f"{key}.dat"))
        union[key] = (curve.wavelength, curve.transmission)
    colors = display_colors(union)
    assert to_hex(lines["Other/IR J"].get_color()) == colors["Other/IR/J"]["DisplayColor"]
    plt.close(ax.figure)

    with pytest.raises(FileNotFoundError):
        Filters.plot("Fac/Missing", filter_root=root, fetch=False)
    with pytest.raises(ValueError):
        Filters.plot(filter_root=root)
