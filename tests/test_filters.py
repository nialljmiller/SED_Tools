"""Filters, as one test that walks named rows: the filter API and pickers, NJM's catalogue,
display colours, components and atmosphere sites, detector types, the filter library with NJM or SVO down or
failing, the SVO browser, the curve format released MESA reads, and wavelength units.

Each row is one old test, or one case of a parametrized one, under the old name without ``test_``, in the old
order within its table. ``@row`` adds a scenario as the row named after it; a ``name: data`` table runs through
its check. NJM, SVO and the original sources are stand-ins: no row uses the network.
"""

from __future__ import annotations

import colorsys
import csv
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest
import requests
from astropy.io import ascii
from numpy.testing import assert_allclose

import sed_tools.filter_properties as fp
from sed_tools import cli, filter_library, njm_filter_grabber, svo_filter_grabber
from sed_tools import filter_metadata as fm
from sed_tools.api import Filters
from sed_tools.cli.filters import _fetch_instrument, review_detector_types
from sed_tools.combine_filters import combine_filter_sets, write_filter_index
from sed_tools.component_sources import catalog
from sed_tools.component_sources.catalog import name_key
from sed_tools.component_sources._common import DerivedCurve, SiteAtmosphere, write_site
from sed_tools.filter_catalogue import CatalogueError, FilterCatalogue, cache_root, format_citations
from sed_tools.filter_colors import display_colors, instrument_colors
from sed_tools.filter_components import (ATMOSPHERE, FILTER, OPTICS, QE, AtmosphereGrid, AtmosphereSpec,
                                         ComponentCoverageError, ComponentError, Curve, InstrumentManifest,
                                         add_component, atmosphere_file, build_all_products, build_product,
                                         compose_band, multiply_curves, parse_components, resolve_atmosphere)
from sed_tools.filter_library import FilterLibrary, instrument_catalogue, load_instrument_catalogue
from sed_tools.filter_metadata import (ENERGY, PHOTON, SOURCE_ASSUMED, SOURCE_INFERRED, SOURCE_SVO, SOURCE_SVO_MATCH,
                                       SVOFilterRecord, SVOUnreachableError, format_metadata_field, has_column_row,
                                       migrate_filter_file, migrate_filter_tree, normalize_detector_type,
                                       read_filter_header, released_mesa_reads, tag_filter_file, tag_filter_tree,
                                       write_filter_file, write_filter_header)
from sed_tools.filter_properties import (AB_JY, curve_properties, extinction_law, extinction_ratio, filter_properties,
                                         instrument_properties)
from sed_tools.filters import svo
from sed_tools.filters.compose import read_curve
from sed_tools.filters.manifest import check_sets, set_problems
from sed_tools.filters.njm import drop_unlisted_bands
from sed_tools.filters.properties import load_vega, properties_record
from sed_tools.flux_cube_tool import infer_unit_from_values, load_filter_curve, load_spectrum, load_two_column_file
from sed_tools.njm_filter_grabber import NJMFilterGrabber, SetDownload
from sed_tools.svo_filter_grabber import Facility, FilterRow, Instrument, SVOFilterBrowser, _download_facilities
from sed_tools.ui_utils import prompt_choice

from support import (CURVE, NJM, STALE, WL, Resp, Session, answer, atmosphere_site, derived, gauss, njm_index,
                     published_set, run_cli, scenarios, team_site)

SCENARIOS, row = scenarios()
NJM_OFFLINE, njm_offline_row = scenarios()
DEFAULT_SVO, default_svo_row = scenarios()

def test_filters(rows):
    rows(SCENARIOS)
    rows(SOURCES, _sources)
    rows(NJM_OFFLINE, _offline_then)
    rows(DEFAULT_SVO, _default_svo_then)
    rows(QUERY_LOCAL, _query_local)
    rows(BULK_DOWNLOAD, _bulk_download)
    rows(PICKER, _picker)
    rows(PUBLISHED_WITH_ATMOSPHERE, _published_with_atmosphere)
    rows(CONVERT_TO_ENERGY, _convert_to_energy)
    rows(COVERAGE_ERROR, _coverage_error)
    rows(COMPONENTS_WITHOUT_A_BUILDER, _components_without_a_builder)
    rows(OUTAGE_ANNOUNCED_ONCE, _outage_announced_once)
    rows(FETCH_INFO_AND_COMPOSE, _fetch_info_and_compose)
    rows(CLI_OLD_SPELLING, _cli_old_spelling)
    rows(BUNDLED_AVAILABILITY, _bundled_availability)
    rows(ARCHIVED_COPIES_APART, _archived_copies_apart)
    rows(RELEASED_MESA_CHECK, _released_mesa_check)
    rows(MIRROR_DOWNLOAD, _mirror_download)
    rows(INFER_UNIT, _infer_unit)
    rows(FILTER_CURVE_UNIT, _filter_curve_unit)


# ======================================================================
# What the rows share: files, NJM, SVO, the bundled catalogue
# ======================================================================
def put(directory, files):
    """Write ``files`` {name: text} into ``directory``, made if missing; return the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text)
    return directory


def write_set(directory, bands, *, index=None, manifest_bands=None, archived=None, text=CURVE):
    """An installed set: a curve per band, the MESA index (``index``, or the bands) and a manifest."""
    put(directory, {f"{band}.dat": text for band in bands})
    (directory / directory.name).write_text("".join(f"{b}.dat\n" for b in (index if index is not None else bands)))
    data = asdict(InstrumentManifest(directory.parent.name, directory.name, list(manifest_bands or bands)))
    if archived is not None:
        data["archived_bands"] = archived
    (directory / "manifest.json").write_text(json.dumps(data))
    return directory


def njm_up(t, tags=False):
    """NJM answers, and its grabber neither pauses nor, unless ``tags``, tags the curves it saves."""
    t.mp.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    t.mp.setattr("sed_tools.njm_filter_grabber.time.sleep", lambda seconds: None)
    if not tags:
        t.mp.setattr("sed_tools.njm_filter_grabber.tag_filter_file", lambda *a, **k: None)


def njm_filters(t, root, files, tags=False):
    """An NJMFilterGrabber under ``root`` (NJM up, see njm_up) whose NJM serves ``files``, paths under the mirror."""
    njm_up(t, tags)
    grabber = NJMFilterGrabber(base_dir=str(root))
    grabber.session = Session(files, base=NJM)
    return grabber


#: An NJMFilterGrabber that found NJM down: it asks nothing and lists nothing.
NJM_DOWN = SimpleNamespace(is_available=lambda: False, fetch_manifest=lambda f, i: None,
                           discover_filters=lambda f, i: [])


def mirror(get, available=True):
    """NJM's client as FilterCatalogue uses it: ``get(url, **kwargs)`` answers each request."""
    return SimpleNamespace(base_url=NJM, is_available=lambda: available, _get=get)


def library_with(root, mirror=None, svo=None):
    """A FilterLibrary under ``root`` asking these NJM and SVO clients (its own where None)."""
    library = FilterLibrary(root)
    if mirror is not None:
        library.mirror = mirror
    if svo is not None:
        library.svo = svo
    return library


def raising(exc):
    """A stand-in that raises ``exc`` whatever it is given."""
    def fail(*args, **kwargs):
        raise exc
    return fail


def bundled(t, catalogue=None):
    """Make ``catalogue`` {facility: {instrument: entry}} the instrument catalogue bundled with sed_tools."""
    facilities = {fac: {"instruments": insts} for fac, insts in (catalogue or {}).items()}
    t.mp.setattr(filter_library, "load_instrument_catalogue", lambda: instrument_catalogue(facilities))


DR2 = {"svo_instruments": ["GAIA/GAIA2"], "bands": ["BP", "G", "RP"],
       "svo_bands": {"GAIA/GAIA2": ["G", "Gbp=BP", "Grp=RP"]}}
GAIA = {"DR2": {"description": "Gaia DR2 passbands (Evans et al. 2018)", **DR2},
        "GAIA": {"description": "Gaia DR3 passbands", "svo_instruments": ["GAIA/GAIA3"],
                 "bands": ["G", "Gbp", "Gbp_bright", "Gbp_faint", "Grp", "Grvs"],
                 "svo_bands": {"GAIA/GAIA3": ["G", "Gbp", "Grp", "Grvs"]}},
        "GAIA0": {"svo_instruments": ["GAIA/GAIA0"], "bands": ["G"], "svo_bands": {"GAIA/GAIA0": ["G"]}}}


def svo_stub(facilities, instruments, *, down=False):
    """An SVO stand-in: ``facilities`` {key: text}; ``instruments`` {facility: {key: text}}."""
    def check():
        if down:
            raise RuntimeError("Failed to fetch SVO index: connection refused")

    def list_facilities():
        check()
        return [Facility(key, key, text) for key, text in facilities.items()]

    def list_instruments(key):
        check()
        return [Instrument(key, name, name, text) for name, text in instruments.get(key, {}).items()]

    return SimpleNamespace(list_facilities=list_facilities, list_instruments=list_instruments)


def svo_browser(root, filters, *, detector="photon"):
    """A real SVOFilterBrowser whose listings and curves come from ``filters`` {(facility, instrument): [band]}."""
    browser = SVOFilterBrowser(base_dir=str(root))
    browser.list_facilities = lambda: [Facility(f, f) for f in sorted({f for f, _ in filters})]
    browser.list_instruments = lambda fac: [Instrument(fac, i, i) for f, i in filters if f == fac]
    browser.list_filters = lambda fac, inst: [FilterRow(f"{fac}/{inst}.{b}", b, fac, inst, "")
                                              for b in filters[(fac, inst)]]
    curve = SimpleNamespace(wavelength=np.array([4000.0, 5000.0, 6000.0]), transmission=np.array([0.0, 1.0, 0.0]),
                            detector_type=detector, wavelength_unit="Angstrom", components="Filter")
    browser.saved = []
    browser.metadata_client = SimpleNamespace(get_filter=lambda fid: browser.saved.append(fid) or curve)
    return browser


def svo_record(filter_id, detector_type, wl, tr, components=None):
    return SVOFilterRecord(filter_id, detector_type, np.asarray(wl), np.asarray(tr), "Angstrom", components)


class SVOClient:
    """An offline SVOMetadataClient: ``records`` {filter ID: record}, ``fail`` {filter ID: exception to raise},
    and ``listing``, what every facility lists."""

    def __init__(self, records=None, fail=None, listing=()):
        self.records, self.fail, self.listing = records or {}, fail or {}, list(listing)

    def get_filter(self, filter_id):
        if filter_id in self.fail:
            raise self.fail[filter_id]
        return self.records.get(filter_id)

    def facility_filter_ids(self, facility):
        return self.listing


class SVOSession:
    """A requests session for SVOMetadataClient: every answer is ``content`` with ``status``, or ``error`` raised.
    ``calls`` lists (url, keyword arguments)."""

    def __init__(self, content=b"", status=200, error=None):
        self.content, self.status, self.error, self.headers, self.calls = content, status, error, {}, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        return Resp(self.content, self.status)


def labels(result, *more):
    """Each picker entry's label and sources, and the ``more`` fields named."""
    return [(e["label"], e["sources"], *(e[key] for key in more)) for e in result["entries"]]


def components_camera(root, ground_based=False):
    """Test/Camera: published g and r (filter+optics+qe), a filter component per band and a shared QE."""
    directory = root / "Test" / "Camera"
    directory.mkdir(parents=True)
    for band in ("g", "r"):
        write_filter_file(directory / f"{band}.dat", [4000., 5000., 6000.], [0.2, 0.8, 0.2],
                          detector_type=PHOTON, source="svo", wavelength_unit="Angstrom",
                          extra={"Components": "filter+optics+qe"})
    manifest = InstrumentManifest.load(directory)
    if ground_based:
        manifest.ground_based = True
    for band in ("g", "r"):
        add_component(directory, "filter", [4000., 5000., 6000.], [0.1, 1., 0.1], band=band, manifest=manifest)
    add_component(directory, "qe", [3000., 7000.], [0.8, 0.6], manifest=manifest)
    manifest.save(directory)
    return directory


# ======================================================================
# The filter API: installed sets and combined sets (test_filter_api)
# ======================================================================
QUERY_LOCAL = {
    "filters_query_local_ignores_empty_filter_dirs": ({"WISE": "W1.dat\nW2.dat\n"}, []),
    "filters_query_local_reports_dat_filter_dirs": (
        {"W1.dat": "Wavelength,Transmission\n1,1\n", "W2.dat": "Wavelength,Transmission\n2,1\n"},
        [{"facility": "WISE", "instrument": "WISE", "n_filters": 2, "is_local": True}]),
}


def _query_local(t, files, expected):
    """An installed set is a folder of curves; an index file alone is none."""
    put(t.tmp / "WISE" / "WISE", files)
    t.mp.setattr(Filters, "_filter_root", t.tmp)
    assert Filters.query(include_local=True, include_remote=False) == expected


@row
def combine_renamed_band_has_no_dot_before_dat(t):
    # MESA names a band after the text before the first dot of its file name:
    # Old_Sony.A200_V.dat would become the column "Old_Sony".
    for facility in ("Cameras", "Old"):
        put(t.tmp / facility / "Sony.A200", {"V.dat": f"Wavelength,Transmission\n{len(facility)},1\n"})
    out = Filters.combine("Combined/Sony", t.tmp / "Cameras/Sony.A200", t.tmp / "Old/Sony.A200", filter_root=t.tmp)
    names = sorted(p.name for p in out.glob("*.dat"))
    assert names == ["Old_SonypA200_V.dat", "V.dat"]
    assert all("." not in name[: -len(".dat")] for name in names)
    assert (out / "Sony").read_text().splitlines() == names
    assert (out / "Old_SonypA200_V.dat").read_text() == "Wavelength,Transmission\n3,1\n"


@row
def combine_leaves_out_archived_and_unlisted_copies(t):
    # As NJM serves Euclid/NISP: H_svo is under archived_bands. Y_old is a copy
    # the index does not list and the manifest does not archive.
    nisp = put(t.tmp / "Euclid" / "NISP", {
        **{f"{band}.dat": "Wavelength,Transmission\n1,1\n" for band in ("H", "Y", "H_svo", "Y_old")},
        "NISP": "H.dat\nY.dat\n", "manifest.json": json.dumps(
            {"facility": "Euclid", "instrument": "NISP", "bands": ["H", "Y"], "archived_bands": {"H_svo": {}}})})
    # Without an index file, every curve but the archived ones.
    bessell = put(t.tmp / "Generic" / "Bessell", {
        **{f"{band}.dat": "Wavelength,Transmission\n2,1\n" for band in ("B", "B_svo")}, "manifest.json": json.dumps(
            {"facility": "Generic", "instrument": "Bessell", "bands": ["B"], "archived_bands": ["B_svo"]})})
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.combine_filters"):
        out = Filters.combine("Combined/NISPB", nisp, bessell, filter_root=t.tmp)
    assert sorted(p.name for p in out.glob("*.dat")) == ["B.dat", "H.dat", "Y.dat"]
    assert (out / "NISPB").read_text().splitlines() == ["B.dat", "H.dat", "Y.dat"]
    assert "leaving out Y_old.dat, which its index file NISP does not list" in t.caplog.text
    assert "H_svo" not in t.caplog.text and "B_svo" not in t.caplog.text


@row
def combine_refuses_an_index_listing_a_missing_curve(t):
    inst = put(t.tmp / "Generic" / "Johnson", {"B.dat": "Wavelength,Transmission\n1,1\n", "Johnson": "B.dat\nV.dat\n"})
    with pytest.raises(FileNotFoundError, match="lists V.dat, not found in"):
        Filters.combine("Combined/J", inst, filter_root=t.tmp)


# ======================================================================
# Choosing several facilities, and the picker's input (test_filter_bulk_selection)
# ======================================================================
BULK_DOWNLOAD = {
    "multiple_facilities_download_every_instrument_piped_without_prompts": (False, ["1-3"]),  # as in 0.4.1
    "multiple_facilities_download_every_instrument_terminal_after_one_confirmation": (True, ["1-3", "y"]),
}


def _bulk_download(t, terminal, answers):
    """Several facilities chosen: every instrument of each is downloaded, after one confirmation at most."""
    cameras = [{"key": "Camera1", "label": "Camera1"}, {"key": "Camera2", "label": "Camera2"}]
    t.mp.setattr(FilterLibrary, "facilities", lambda self: {
        "entries": [{"key": name, "label": name} for name in ("A", "B", "C")], "errors": []})
    t.mp.setattr(FilterLibrary, "instruments", lambda self, facility: {"entries": cameras, "errors": []})
    fetched, offered = [], []
    # The browser downloads with its own library.
    t.mp.setattr(FilterLibrary, "published", lambda self, f, i: fetched.append((f, i)) or t.tmp / f / i)
    t.mp.setattr(cli.filters, "_offer_composition", offered.append)
    # What the browser describes before asking to download (Filters.available, NJM's catalogue entry).
    t.mp.setattr(FilterLibrary, "available", lambda self, facility, instrument, remote=True: {
        "facility": facility, "instrument": instrument, "bands": ["V"], "published": "filter",
        "published_bands": {}, "components": [], "partial_components": {}, "component_status": None,
        "errors": [], "atmosphere": False, "platform": "ground", "variants": {}})
    t.mp.setattr(cli.filters, "_catalogue_entry", lambda *a, **k: {"info": None, "properties": None, "errors": []})
    t.mp.setattr("sys.stdin", SimpleNamespace(isatty=lambda: terminal))
    answers = iter(answers)  # Any further confirmation, instrument or component prompt would fail.
    t.mp.setattr("builtins.input", lambda prompt="": next(answers))
    cli.run_filters_flow(str(t.tmp))
    assert next(answers, None) is None  # every answer was asked for
    assert fetched == [(facility, instrument) for facility in ("A", "B", "C") for instrument in ("Camera1", "Camera2")]
    assert offered == []


PICKER = {"picker_reprompts_on_out_of_range_selection": (["1,99", "1,3"], [0, 2]),
          "picker_id_command_returns_a_multi_selection": (["id 2"], [1])}


def _picker(t, replies, chosen):
    answer(t.mp, *replies)
    assert prompt_choice(["A", "B", "C"], "Facilities", multi=True) == chosen


# ======================================================================
# NJM's catalogue: search, the cache, citations, filters-info, plots (test_filter_catalogue)
# ======================================================================
def band(pivot, fwhm, detector="photon", ab=0.1):
    return {"WavelengthPivot": pivot, "FWHM": fwhm,
            "WavelengthMin": pivot - 0.75 * fwhm, "WavelengthMax": pivot + 0.75 * fwhm,
            "DetectorType": detector, "ABminusVega": ab, "DisplayColor": "#123456"}


SVO_ACK = {"for": "SVO Filter Profile Service", "text": "This research has made use of the SVO FPS.",
           "references": ["bibcode:2012ivoa.rept.1015R"], "applies_to": "curves and metadata"}


def catalogue_files(generated="2026-10-26T10:00:00"):
    """NJM's index.json, Fac/Cam's and Other/IR's properties.json and info.json, and Vega; Fac/Old has none."""
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


class EtagNJM:
    """NJM serving ``files`` with ETags (a matching If-None-Match gets 304); ``requests`` lists each (path,
    headers). With ``fail`` set it does not answer."""

    def __init__(self, files):
        self.files, self.requests, self.fail = dict(files), [], False

    def etag(self, rel):
        return f'"{abs(hash(self.files[rel]))}"'

    def get(self, url, timeout=None, headers=None, **kwargs):
        rel, headers = url[len(NJM) + 1:], dict(headers or {})
        self.requests.append((rel, headers))
        if self.fail:
            raise requests.ConnectionError("mirror down")
        if rel not in self.files:
            return Resp(b"", 404)
        tag = self.etag(rel)
        fresh = headers.get("If-None-Match") != tag
        response = Resp(self.files[rel] if fresh else b"", 200 if fresh else 304)
        response.headers = {"ETag": tag, "Last-Modified": "Mon, 26 Oct 2026 04:45:30 GMT"} if fresh else {"ETag": tag}
        return response

    def fetched(self, rel):
        return [h for r, h in self.requests if r == rel]


def grabber_for(root, server):
    grabber = NJMFilterGrabber(base_dir=str(root))
    grabber.session = SimpleNamespace(get=server.get)
    return grabber


def njm_catalogue(t):
    """A filter root whose Filters catalogue asks an EtagNJM serving catalogue_files(); returns (root, njm)."""
    t.mp.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    t.mp.setattr("sed_tools.njm_filter_grabber.time", SimpleNamespace(sleep=lambda s: None))
    t.mp.setattr(Filters, "_catalogues", {})
    root, njm = t.tmp / "filters", EtagNJM(catalogue_files())
    root.mkdir()
    key = (str(root.resolve()), False)
    Filters._catalogues[key] = FilterCatalogue(key[0], mirror=grabber_for(root, njm))
    return root, njm


def write_curve(path, centre, width, header="Wavelength,Transmission"):
    path.parent.mkdir(parents=True, exist_ok=True)
    lam = np.linspace(centre - 2 * width, centre + 2 * width, 81)
    t = np.exp(-0.5 * ((lam - centre) / (width / 2.355)) ** 2)
    path.write_text(header + "\n" + "".join(f"{w:.3f},{v:.6f}\n" for w, v in zip(lam, t)))


@row
def search_matches_names_in_the_index_before_fetching_properties(t):
    root, njm = njm_catalogue(t)
    result = Filters.search(facility="other", filter_root=root)
    assert list(result["band"]) == ["J", "K"]
    assert list(result.columns[:5]) == ["facility", "instrument", "band", "WavelengthPivot", "WavelengthMin"]
    assert njm.fetched("filters/Other/IR/properties.json")
    assert not njm.fetched("filters/Fac/Cam/properties.json")
    assert list(Filters.search("cam/r", filter_root=root)["band"]) == ["r"]


@row
def search_by_wavelength_pivot_fwhm_detector_and_any_property(t):
    root, njm = njm_catalogue(t)
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


@row
def search_accepts_quantities_and_names_only(t):
    from astropy import units as u

    root, njm = njm_catalogue(t)
    assert list(Filters.search(wavelength=620 * u.nm, filter_root=root)["band"]) == ["r"]
    assert list(Filters.search(pivot=(1, 2) * u.um, filter_root=root)["band"]) == ["J"]
    njm.requests.clear()
    names = Filters.search(band="K", properties=False, filter_root=root)
    assert list(names.columns) == ["facility", "instrument", "band"] and list(names["band"]) == ["K"]
    assert not [r for r, _ in njm.requests if r.endswith("properties.json")]


@row
def files_are_cached_beside_the_filter_root_and_revalidated_when_the_index_changes(t):
    root, njm = njm_catalogue(t)
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


@row
def offline_and_unreachable_mirror_use_cached_copies(t):
    root, njm = njm_catalogue(t)
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


@row
def citations_for_chosen_bands(t):
    root, _ = njm_catalogue(t)
    r_only = Filters.citations("Fac/Cam/r", filter_root=root)
    assert [r["key"] for r in r_only["references"]] == ["bibcode:2020Cam.....1A", "bibcode:2012ivoa.rept.1015R"]
    g_too = Filters.citations("Fac/Cam/r", ("Fac", "Cam", "g"), filter_root=root)
    assert "bibcode:2021gband...2B" in [r["key"] for r in g_too["references"]]
    missing = Filters.citations("Fac/Cam/z", "Fac/Old", filter_root=root)
    assert any("no band 'z'" in e for e in missing["errors"]) and any("Fac/Old" in e for e in missing["errors"])
    assert not missing["references"]


@row
def citations_default_to_installed_instruments(t):
    root, _ = njm_catalogue(t)
    write_curve(root / "Fac" / "Cam" / "g.dat", 4800, 1200)
    text = Filters.cite(filter_root=root)
    assert "Citations for Fac/Cam" in text and "We thank Fac." in text and "Free to use." in text
    assert "@article{2020Cam.....1A" in text


@row
def search_and_cite_in_python(t):
    # The filters-search and filters-cite commands were dropped before release; the API stays.
    root, _ = njm_catalogue(t)
    result = Filters.search(pivot=(4000, None), fwhm=(None, 1400), where={"ABminusVega": (None, 0.5)},
                            filter_root=root)
    assert list(result["band"]) == ["g"]
    # Fac/Old has no properties file: a gap, reported, but not a failure.
    assert result.attrs["errors"] == [] and result.attrs["gaps"] == ["Fac/Old: NJM has no properties file"]
    assert list(Filters.search(wavelength=21000, filter_root=root)["band"]) == ["K"]
    bibtex = Filters.cite("Other/IR", format="bibtex", filter_root=root)
    assert bibtex.startswith("% No BibTeX served") and "@article{2020Cam.....1A" in bibtex


@row
def cli_info(t):
    root, njm = njm_catalogue(t)
    Filters.search(filter_root=root, properties=False)  # index.json is cached; no info.json yet
    write_curve(root / "Fac" / "Cam" / "g.dat", 4800, 1200,
                header="Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=svo")
    code, out = run_cli(t.mp, "filters-info", "Fac/Cam", "--base", str(root), "--offline"), t.capsys.readouterr().out
    assert code == 0 and "info.json is not cached and working offline" in out  # the catalogue is an extra
    assert not njm.fetched("filters/Fac/Cam/info.json")
    code, out = run_cli(t.mp, "filters-info", "Fac/Cam", "--base", str(root)), t.capsys.readouterr().out
    assert code == 0 and "A camera." in out and "Free to use." in out and "Cam g" in out
    run_cli(t.mp, "filters-info", "Fac/Cam", "--base", str(root), "--offline", "--json")
    assert json.loads(t.capsys.readouterr().out)["catalogue"]["info"]["description"] == "A camera."


@row
def plot_uses_filter_colors_and_overlays_vega(t):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    root, _ = njm_catalogue(t)
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


# ======================================================================
# Display colours (test_filter_colors)
# ======================================================================
def top_hat(lo, hi, step=10.0):
    """A top-hat curve from lo to hi Angstrom with zero edges."""
    lam = np.arange(lo - step, hi + 2 * step, step)
    return lam, ((lam >= lo) & (lam <= hi)).astype(float)


def hue(color):
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 360.0 * colorsys.rgb_to_hls(r, g, b)[0]


@row
def all_ultraviolet_set_runs_purple_to_sky_blue(t):
    colors = display_colors({"FUV": top_hat(1350, 1750), "NUV": top_hat(1750, 2800)})
    assert hue(colors["NUV"]["DisplayColor"]) < hue(colors["FUV"]["DisplayColor"])
    assert 180 < hue(colors["NUV"]["DisplayColor"]) < 210 and 280 < hue(colors["FUV"]["DisplayColor"]) < 300


# ======================================================================
# Separable components: filter, optics, QE, atmosphere (test_filter_components)
# ======================================================================
ATM = {"airmass": 1.2, "pwv": 2.5}


def svo_published(directory, band, tr, components):
    """SVO's photon-counting curve of ``band`` on WL, saying what it contains; returns the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    write_filter_file(directory / f"{band}.dat", WL, tr, detector_type=PHOTON, source="svo",
                      wavelength_unit="Angstrom", extra={"Components": components, "ComponentsSource": "svo"})
    return directory


def argus_system(root):
    """Argus/System: published b1 and b2, a filter component per band, shared optics and QE components."""
    inst = root / "Argus" / "System"
    svo_published(inst, "b1", gauss(4800, 150) * 0.8 * 0.6, "filter+optics+qe")
    svo_published(inst, "b2", gauss(5300, 150) * 0.8 * 0.6, "filter+optics+qe")
    write_filter_index(inst, "System")
    manifest = InstrumentManifest.load(inst)
    manifest.components = {}
    add_component(inst, FILTER, WL, gauss(4800, 150), band="b1", manifest=manifest)
    add_component(inst, FILTER, WL, gauss(5300, 150), band="b2", manifest=manifest)
    add_component(inst, OPTICS, [3000.0, 7000.0], [0.8, 0.8], name="optics", manifest=manifest)
    add_component(inst, QE, [3000.0, 7000.0], [0.6, 0.6], name="ccd", manifest=manifest)
    manifest.detector_type = ENERGY
    manifest.detector_types = {}  # legacy manifest with one instrument-wide convention
    manifest.save(inst)
    return inst


def with_site(inst, site="paranal"):
    m = InstrumentManifest.load(inst)
    m.default_site = site
    m.save(inst)
    return inst


def paranal(root, airmass=1.2, pwv=2.5, value=0.9):
    """An atmosphere grid of one condition at paranal; returns ``root``."""
    site_dir = root / "paranal"
    site_dir.mkdir(parents=True, exist_ok=True)
    name = f"am{airmass:.2f}_pwv{pwv:g}.dat"
    write_filter_file(site_dir / name, [1000.0, 3000.0, 3001.0, 300000.0], [0.0, 0.0, value, value],
                      detector_type=ENERGY, source="component", wavelength_unit="Angstrom",
                      extra={"Role": ATMOSPHERE, "Components": ATMOSPHERE})
    AtmosphereGrid(site="paranal", entries=[{"airmass": airmass, "pwv": pwv, "file": name}]).save(site_dir)
    return root


@row
def parse_components_of_an_empty_string(t):
    assert parse_components("") == ()


@row
def multiply_curves_refuses_to_extrapolate(t):
    band = Curve(WL, gauss(5000, 300), {})
    short = Curve(np.array([4500.0, 7000.0]), np.array([1.0, 1.0]), {})
    with pytest.raises(ComponentCoverageError):
        multiply_curves([band, short])
    grid, product = multiply_curves([band, short], allow_partial=True)
    assert np.all(product[grid < 4500.0] == 0.0)


PUBLISHED_WITH_ATMOSPHERE = {
    "published_with_atmosphere_is_not_double_counted": (lambda root: with_site(svo_published(
        root / "2MASS" / "2MASS", "J", gauss(5000, 300), "filter+instrument+atmosphere")), "double count"),
    "space_instrument_refuses_atmosphere": (lambda root: svo_published(
        root / "TESS" / "TESS", "Red", gauss(5000, 300), "filter+instrument"), "no atmosphere site of its own"),
    "published_requires_known_contents": (lambda root: with_site(put(
        root / "Generic" / "Mystery", {"X.dat": "Wavelength,Transmission\n4000,1\n5000,1\n"})),
        "does not say what it contains"),
}


def _published_with_atmosphere(t, instrument, refusal):
    """What a published curve holds, and where its instrument observes, decide whether an atmosphere is added."""
    inst = instrument(t.tmp)
    with pytest.raises(ComponentError, match=refusal):
        build_product(inst, ["published"], atmosphere=ATM, atmosphere_root=paranal(t.tmp / "atm"))


@row
def instrument_without_own_site_gets_no_atmosphere(t):
    inst = argus_system(t.tmp / "filters")  # ground-based, no site assigned
    with pytest.raises(ComponentError, match="no atmosphere site of its own"):
        build_product(inst, ["filter"], atmosphere=ATM, atmosphere_root=paranal(t.tmp / "atm"))
    assert InstrumentManifest.load(inst).describe()["atmosphere"] is False


@row
def build_all_products_makes_every_with_and_without_variant(t):
    inst = with_site(argus_system(t.tmp / "filters"))
    built = build_all_products(inst, conditions=[ATM], atmosphere_root=paranal(t.tmp / "atm"))
    assert sorted(p.name for p in built) == sorted([
        "filter", "filter+optics", "filter+qe", "filter+optics+qe",
        "filter+atm_paranal_am1.2_pwv2.5", "filter+optics+atm_paranal_am1.2_pwv2.5",
        "filter+qe+atm_paranal_am1.2_pwv2.5", "filter+optics+qe+atm_paranal_am1.2_pwv2.5",
    ])


@row
def skycalc_build_writes_grid_and_resumes(t):
    from sed_tools.component_sources.skycalc import build

    def transmission(site, airmass, pwv, lo, hi, resolution):
        calls.append(site)
        lam = lo * np.exp(np.arange(0, np.log(hi / lo), 1 / 2000.0))
        return lam, np.full_like(lam, 0.8 ** airmass)

    calls, fake = [], SimpleNamespace(transmission=transmission)
    root = build(t.tmp / "filters", sites=["paranal"], airmasses=[1.0, 2.0], pwvs=[2.5],
                 fetch_resolution=2000, store_resolution=500, chunks=2, client=fake)
    grid = AtmosphereGrid.load(root / "paranal")
    assert [e["file"] for e in grid.entries] == ["am1.00_pwv2.5.dat", "am2.00_pwv2.5.dat"]
    curve = load_filter_curve(str(root / "paranal" / "am2.00_pwv2.5.dat"))
    assert curve.transmission[curve.wavelength > 4000].mean() == pytest.approx(0.64)
    assert curve.transmission[curve.wavelength < 2999].max() == 0.0  # opaque below 300 nm
    asked = len(calls)
    build(t.tmp / "filters", sites=["paranal"], airmasses=[1.0, 2.0], pwvs=[2.5], client=fake)
    assert len(calls) == asked  # nothing refetched


# ======================================================================
# Correcting or converting installed curves' detector type (test_filter_detector_edit)
# ======================================================================
G, R, I = gauss(4800.0, 300.0) * 0.9, gauss(5300.0, 300.0) * 0.7, gauss(5600.0, 200.0) * 0.8  # noqa: E741
REFERENCE = Path(__file__).parent / "golden" / "inputs" / "reference"  # NJM's vega_flam.csv and sun_flam.csv
SPECTRA = {"vega_spectrum": "vega/vega_flam.csv", "sun_spectrum": "sun/sun_flam.csv"}
QUESTION = ("This filter set is known to be used with photon-counting detectors (svo). "
            "Do you wish to edit/correct this? [y/N] ")


def spectra_cached(t):
    """The filter root, with NJM's Vega and Sun spectra in its catalogue cache as if fetched just now."""
    root = t.tmp / "filters"
    t.mp.setattr(Filters, "_filter_root", root)
    for rel in SPECTRA.values():
        cached = cache_root(root) / rel
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REFERENCE / Path(rel).name, cached)
        cached.with_name(cached.name + ".http.json").write_text(json.dumps({"checked": time.time()}))
    return root


def sdss(root, *, manifest=True, energy_band=False):
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


def samples(path):
    return path.read_bytes().split(b"\n", 1)[1]


def mesa_weights(path):
    """What MESA integrates: an energy-counting curve as T/lambda at the same peak, a photon-counting one as is."""
    curve = read_curve(path)
    if read_filter_header(path)["DetectorType"] != ENERGY:
        return curve.transmission
    weights = curve.transmission / curve.wavelength
    return weights * curve.transmission.max() / weights.max()


@row
def correct_changes_only_the_label(t):
    inst = sdss(spectra_cached(t))
    before = samples(inst / "g.dat")
    rows = Filters.correct_detector_type(inst, "energy", bands=["g"])
    assert [(r["band"], r["previous"], r["changed"]) for r in rows] == [("g", PHOTON, True)]
    assert samples(inst / "g.dat") == before
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


CONVERT_TO_ENERGY = {"convert_to_energy_keeps_mesa_photometry_and_products_with_manifest": True,
                     "convert_to_energy_keeps_mesa_photometry_and_products_without_manifest": False}


def _convert_to_energy(t, manifest):
    inst = sdss(spectra_cached(t), manifest=manifest)
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
        assert_allclose(mesa_weights(inst / f"{band}.dat"), native, rtol=1e-12)
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


@row
def convert_to_photon_and_back(t):
    inst = sdss(spectra_cached(t), energy_band=True)
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


@row
def correct_then_convert_records_the_net_factor(t):
    inst = sdss(spectra_cached(t), manifest=False)
    Filters.correct_detector_type(inst / "g.dat", "energy")
    Filters.convert_detector_type(inst / "g.dat", "photon")
    header = read_filter_header(inst / "g.dat")
    assert (header["DetectorType"], header["DetectorTypeOriginal"], header["DetectorTypeConversion"]) == (
        PHOTON, PHOTON, "1/lambda")
    assert_allclose(read_curve(inst / "g.dat").transmission, G / WL * G.max() / (G / WL).max(), rtol=1e-12)
    # Without a manifest, the headers still say the detector is the user's energy-counting one.
    inferred = InstrumentManifest.load(inst)
    assert (inferred.detector_types["g"], inferred.detector_type_sources["g"]) == (ENERGY, "user")


@row
def correct_without_njm_leaves_the_vega_and_sun_values_unknown(t):
    def unreachable(self, method, url, **kwargs):
        raise requests.ConnectionError("NJM is down")

    t.mp.setattr(requests.Session, "request", unreachable)
    t.mp.setattr("time.sleep", lambda seconds: None)  # the client's retries
    root = t.tmp / "elsewhere"  # nothing cached under this filter root
    inst = sdss(root)
    Filters.correct_detector_type(inst, "energy", bands=["g"], filter_root=root)
    props = json.loads((inst / "properties.json").read_text())["bands"]["g"]
    assert props == {**properties_record(inst / "g.dat"), "DisplayColor": "#000"}
    assert props["DetectorType"] == ENERGY and props["ZeroPointVega_flam"] is None and props["Fsun"] is None
    out = t.capsys.readouterr().out
    assert "vega/vega_flam.csv is not cached and NJM is unreachable: the Vega-based properties" in out


@row
def edit_refuses_unknown_bands_types_and_untagged_conversions(t):
    root = spectra_cached(t)
    inst = sdss(root)
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


@row
def retagging_keeps_edited_types(t):
    inst = sdss(spectra_cached(t))
    Filters.convert_detector_type(inst, "energy", bands=["g"])
    Filters.correct_detector_type(inst, "energy", bands=["r"])
    before = {band: (inst / f"{band}.dat").read_bytes() for band in "gr"}
    tag_filter_tree(inst, client=SVOClient(), force=True)
    assert {band: (inst / f"{band}.dat").read_bytes() for band in "gr"} == before


@row
def build_product_detector_type_still_only_relabels(t):
    out = build_product(sdss(t.tmp), ["published"], detector_type="energy")
    assert out.name == "published+detector_energy"
    assert read_filter_header(out / "g.dat")["DetectorType"] == ENERGY
    assert_allclose(read_curve(out / "g.dat").transmission, G, rtol=0, atol=0)


# The interactive review after a download or a compose
@row
def review_default_changes_nothing(t):
    inst = sdss(root := spectra_cached(t))
    before = (inst / "g.dat").read_bytes()
    prompts = answer(t.mp, "")
    review_detector_types(inst, root)
    assert prompts == [QUESTION] and (inst / "g.dat").read_bytes() == before


@row
def review_correct(t):
    inst = sdss(root := spectra_cached(t))
    before = samples(inst / "g.dat")
    prompts = answer(t.mp, "y", "correct")
    review_detector_types(inst, root)
    assert prompts == [QUESTION, "Correct the label only, or convert the curves? [correct/convert]: "]
    assert read_filter_header(inst / "g.dat")["DetectorType"] == ENERGY and samples(inst / "g.dat") == before


@row
def review_convert(t):
    inst = sdss(root := spectra_cached(t))
    answer(t.mp, "y", "convert")
    review_detector_types(inst, root)
    assert ("Converting to energy-counting with T_E(λ) = T_P(λ)·λ, rescaled to the same peak."
            in t.capsys.readouterr().out)
    assert read_filter_header(inst / "r.dat")["DetectorTypeConversion"] == "lambda"


@row
def review_of_a_mixed_set_asks_for_the_type(t):
    inst = sdss(root := spectra_cached(t), energy_band=True)
    prompts = answer(t.mp, "y", "convert", "energy")
    review_detector_types(inst, root, "This product")
    assert prompts[0] == ("This product is known to be used with photon-counting detectors (svo) for g, r and "
                          "energy-counting detectors (svo) for i. Do you wish to edit/correct this? [y/N] ")
    assert prompts[2] == "Change them to energy or photon: "
    assert {b: read_filter_header(inst / f"{b}.dat")["DetectorType"] for b in "gri"} == dict.fromkeys("gri", ENERGY)
    assert "DetectorTypeConversion" not in read_filter_header(inst / "i.dat")


@row
def filters_detector_correct_and_convert_options(t):
    inst = sdss(spectra_cached(t))
    run_cli(t.mp, "filters-detector", str(inst), "--convert", "energy", "--bands", "g")
    assert "1 files, 1 changed" in t.capsys.readouterr().out
    assert read_filter_header(inst / "g.dat")["DetectorTypeConversion"] == "lambda"
    run_cli(t.mp, "filters-detector", str(inst / "r.dat"), "--correct", "energy")
    assert read_filter_header(inst / "r.dat")["DetectorType"] == ENERGY


@row
def download_flow_reviews_the_detector_type(t):
    inst = sdss(root := spectra_cached(t))
    prompts = answer(t.mp, "y", "correct", "")  # then decline the composed version
    failures = []
    _fetch_instrument(SimpleNamespace(published=lambda f, i: inst, root=root), "SLOAN", "SDSS", failures, offer=True)
    assert prompts[0] == QUESTION and prompts[-1].startswith("Build a composed version") and not failures
    assert read_filter_header(inst / "g.dat")["DetectorType"] == ENERGY


# ======================================================================
# Photon- and energy-counting curves, and how a curve's type is resolved (test_filter_detector_type)
# ======================================================================
NACO_URL = "https://www.eso.org/sci/facilities/paranal/instruments/naco/inst/filters/H_CONICA.dat"


def write_plain(path, wl, tr):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Wavelength,Transmission\n" + "".join(f"{float(w)!r},{float(t)!r}\n" for w, t in zip(wl, tr)))


def naco_h(root, wl, tr, detector_type, source):
    """NACO's H_CONICA, tagged the way the NJM mirror tags evidence-based bands; returns its bytes and path."""
    path = root / "Paranal" / "NACO" / "H_CONICA.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_filter_file(path, wl, tr, detector_type=detector_type, source=source, wavelength_unit="Angstrom",
                      extra={"Components": "filter", "ComponentsSource": "original", "Sources": NACO_URL})
    return path.read_bytes(), path


@row
def normalize_detector_type_of_an_unknown_code(t):
    assert normalize_detector_type(2) is None


@row
def tag_resolves_by_filter_id(t):
    wl, tr = [5000.0, 6000.0], [0.5, 0.6]
    path = t.tmp / "GAIA" / "GAIA" / "G.dat"
    path.parent.mkdir(parents=True)
    write_filter_file(path, wl, tr, detector_type=ENERGY, source=SOURCE_ASSUMED, filter_id="GAIA/GAIA3.G")
    result = tag_filter_file(path, client=SVOClient({"GAIA/GAIA3.G": svo_record("GAIA/GAIA3.G", PHOTON, wl, tr)}))
    assert (result.detector_type, result.source, result.changed) == (PHOTON, SOURCE_SVO, True)
    assert read_filter_header(path)["DetectorType"] == PHOTON
    assert load_filter_curve(str(path)).detector_type == PHOTON


@row
def tag_matches_svo_id_embedded_in_filename(t):
    wl, tr = [4000.0, 5000.0, 6000.0], [0.1, 0.9, 0.2]
    write_plain(t.tmp / "Argus" / "raw" / "SLOAN_SDSS.g.dat", wl, tr)
    client = SVOClient({"SLOAN/SDSS.g": svo_record("SLOAN/SDSS.g", PHOTON, wl, tr)})
    result = tag_filter_tree(t.tmp, client=client)[0]
    assert (result.detector_type, result.filter_id) == (PHOTON, "SLOAN/SDSS.g")


@row
def curve_mismatch_is_not_accepted(t):
    path = t.tmp / "GAIA" / "GAIA" / "G.dat"
    write_plain(path, [5000.0, 6000.0], [0.5, 0.6])
    client = SVOClient({"GAIA/GAIA.G": svo_record("GAIA/GAIA.G", PHOTON, [5000.0, 6000.0], [0.5, 0.7])})
    assert tag_filter_file(path, client=client).source == SOURCE_ASSUMED


# Source precedence: evidence outranks SVO, with or without --force
@row
def force_does_not_replace_user_override(t):
    wl, tr = [4000.0, 5000.0, 6000.0], [0.1, 0.9, 0.2]
    path = t.tmp / "SLOAN" / "SDSS+detector_energy" / "g.dat"
    path.parent.mkdir(parents=True)
    write_filter_file(path, wl, tr, detector_type=ENERGY, source="user", wavelength_unit="Angstrom",
                      extra={"DetectorTypeOriginal": PHOTON})
    before = path.read_bytes()
    client = SVOClient({"SLOAN/SDSS+detector_energy.g": svo_record("SLOAN/SDSS+detector_energy.g", PHOTON, wl, tr)})
    result = tag_filter_tree(t.tmp, client=client, force=True)[0]
    assert (result.detector_type, result.source, result.kept) == (ENERGY, "user", True)
    assert path.read_bytes() == before


@row
def force_keeps_assumed_type_when_nothing_matches(t):
    before, path = naco_h(t.tmp, [15000.0, 16000.0, 17000.0], [0.1, 0.9, 0.1], PHOTON, SOURCE_ASSUMED)
    result = tag_filter_tree(t.tmp, client=SVOClient(), force=True)[0]
    assert (result.detector_type, result.source, result.changed, result.kept) == (
        PHOTON, SOURCE_ASSUMED, False, False)
    assert path.read_bytes() == before


@row
def cli_reports_kept_types(t):
    wl, tr = [15000.0, 16000.0, 17000.0, 18000.0], [0.0, 0.9, 0.95, 0.0]
    before, path = naco_h(t.tmp, wl, tr, PHOTON, SOURCE_INFERRED)
    client = SVOClient({"Paranal/NACO.H_CONICA": svo_record("Paranal/NACO.H_CONICA", ENERGY, wl, tr)})
    t.mp.setattr("sed_tools.filter_metadata.SVOMetadataClient", lambda *a, **k: client)
    run_cli(t.mp, "filters-detector", str(t.tmp), "--force")
    out = t.capsys.readouterr().out
    assert "1 files, 0 changed" in out
    assert "1 kept their DetectorType" in out and "1 of them resolve to the other type" in out
    assert "kept photon [inferred]: inferred outranks svo-curve-match; svo-curve-match gives energy" in out
    assert path.read_bytes() == before


# ======================================================================
# The filter library offline: compose, atmospheres, SVO, NJM's burst block (test_filter_library)
# ======================================================================
REAL_AVAILABILITY_CHECK = NJMFilterGrabber._check_availability
TEST_INDEX = njm_index({"Test": {"Camera": {"filters": ["g", "r"], "detector_types": {"photon": 2},
                                            "properties": "properties.json", "info": "info.json"}}})


def njm_offline(t):
    """Every NJM client of the filter library finds NJM down (test_filter_library's autouse fixture)."""
    t.mp.setattr(filter_library.NJMFilterGrabber, "_check_availability", lambda self: False)


def _offline_then(t, scenario):
    """The rows of test_filter_library: its autouse fixture took NJM offline."""
    njm_offline(t)
    scenario(t)


def skycalc_paranal(directory):
    """What SkyCalc builds for paranal: one energy-counting grid point, airmass 1.2 and PWV 2.5."""
    directory.mkdir(parents=True)
    write_filter_file(directory / "a.dat", [3000., 7000.], [0.8, 0.8], detector_type="energy",
                      source="component", wavelength_unit="Angstrom", extra={"Components": "atmosphere"})
    AtmosphereGrid("paranal", [{"airmass": 1.2, "pwv": 2.5, "file": "a.dat"}]).save(directory)


def new_camera_svo(**more):
    """SVO listing NewFacility/NewCamera, with band g."""
    return SimpleNamespace(list_facilities=lambda: [Facility("NewFacility", "New facility")],
                           list_instruments=lambda facility: [Instrument(facility, "NewCamera", "New camera")],
                           list_filters=lambda f, i: [FilterRow("NewFacility/NewCamera.g", "g", f, i, "")], **more)


def flat_record(filter_id):
    return svo_record(filter_id, PHOTON, [4000., 6000.], np.ones(2), "Filter")


class BurstBlockedNJM:
    """NJM answering the first ``blocked`` requests with 403, as Apache's burst block does; ``calls`` lists
    (method, path)."""

    def __init__(self, files, blocked=0, head_error=None, head_status=301):
        self.files, self.blocked, self.head_error, self.head_status = files, blocked, head_error, head_status
        self.headers, self.verify, self.calls = {}, True, []

    def _answer(self, method, url, status, text=""):
        self.calls.append((method, url[len(NJM):]))
        if self.blocked:
            self.blocked -= 1
            status, text = 403, "Forbidden"
        return Resp(text, status)

    def head(self, url, timeout=None, verify=None):
        if self.head_error is not None:
            self.calls.append(("HEAD", url[len(NJM):]))
            raise self.head_error
        return self._answer("HEAD", url, self.head_status)

    def get(self, url, timeout=None, headers=None, **kwargs):
        rel = url[len(NJM) + 1:]
        return self._answer("GET", url, 200 if rel in self.files else 404, self.files.get(rel, ""))


def blocked_grabber(t, session):
    """An NJMFilterGrabber on ``session`` whose waits are recorded, not waited; returns it and the waits."""
    delays = []
    t.mp.setattr("sed_tools.njm_filter_grabber.time", SimpleNamespace(sleep=delays.append))
    grabber = NJMFilterGrabber(base_dir=str(t.tmp))  # its availability is patched off by the row's setup
    grabber.session = session
    return grabber, delays


@njm_offline_row
def weighting_override_accepts_unknown_published_contents(t):
    directory = t.tmp / "Mystery" / "Camera"
    directory.mkdir(parents=True)
    write_filter_file(directory / "g.dat", [4000., 6000.], [1., 1.],
                      detector_type=PHOTON, source="svo", wavelength_unit="Angstrom")
    out = Filters.compose(directory, "published", detector_type="energy", filter_root=t.tmp)
    assert read_filter_header(out / "g.dat")["DetectorType"] == "energy"


@njm_offline_row
def unresolved_instrument_rejects_extra_detector_or_optics(t):
    instrument = components_camera(t.tmp / "filters", ground_based=True)
    manifest = InstrumentManifest.load(instrument)
    manifest.published_bands["g"] = "filter+instrument"
    manifest.save(instrument)
    with pytest.raises(ComponentError, match="unresolved"):
        build_product(instrument, "published+qe", bands=["g"])


@njm_offline_row
def manifest_role_must_match_component_header(t):
    instrument = components_camera(t.tmp / "filters", ground_based=True)
    path = instrument / InstrumentManifest.load(instrument).component_file("filter", "g")
    path.write_text(path.read_text().replace("Components=filter", "Components=filter+qe"))
    with pytest.raises(ComponentError, match="disagree"):
        build_product(instrument, "filter", bands=["g"])


@njm_offline_row
def atmosphere_requires_pwv_when_grid_is_ambiguous(t):
    grid = AtmosphereGrid("paranal", [{"airmass": 1.2, "pwv": 1., "file": "a.dat"},
                                      {"airmass": 1.2, "pwv": 3., "file": "b.dat"}])
    with pytest.raises(ComponentError, match="choose pwv"):
        grid.select(AtmosphereSpec("paranal", 1.2))
    assert grid.select(AtmosphereSpec("paranal", 1.2, 3)) == "b.dat"


@njm_offline_row
def skycalc_fallback_requests_only_the_instruments_site(t):
    instrument = with_site(components_camera(t.tmp / "filters", ground_based=True))
    library, calls = FilterLibrary(instrument.parent.parent), []
    t.mp.setattr(filter_library, "get_builder", lambda name: lambda root, **kw: calls.append(kw) or skycalc_paranal(
        library.atmosphere_root / "paranal"))
    library.compose(instrument, "filter", atmosphere={"airmass": 1.2, "pwv": 2.5})
    assert calls[0]["sites"] == ["paranal"]
    assert calls[0]["airmasses"] == [1.2]
    assert calls[0]["pwvs"] == [2.5]
    library.compose(instrument, "filter", atmosphere={"airmass": 1.2, "pwv": 2.5})
    assert len(calls) == 1  # installed grid needs no source requests


@njm_offline_row
def svo_availability_works_without_njm_or_downloads(t):
    library = FilterLibrary(t.tmp)
    library.svo = new_camera_svo(metadata_client=SimpleNamespace(get_filter=flat_record))
    info = library.available("NewFacility", "NewCamera")
    assert info["availability_source"] == "svo"
    assert info["buildable"] == ["filter"]
    assert info["atmosphere"] is False
    assert not list(t.tmp.rglob("*.dat"))


@njm_offline_row
def undeclared_partial_download_is_not_accepted(t):
    # A legacy index may have been built after failures.
    directory = put(t.tmp / "Test" / "Camera", {"g.dat": "4000 1\n6000 1\n", "Camera": "g.dat\n"})
    assert not FilterLibrary.published_complete(directory)


@njm_offline_row
def burst_block_403_is_not_reported_as_an_outage(t):
    session = BurstBlockedNJM({"index.json": TEST_INDEX}, blocked=2)  # HEAD and the first index GET
    grabber, delays = blocked_grabber(t, session)
    assert REAL_AVAILABILITY_CHECK(grabber) is True
    assert session.calls == [("HEAD", ""), ("GET", "/index.json"), ("GET", "/index.json")]
    assert delays == [1]  # the grabber's usual wait before retrying a 403

    grabber._available = True
    assert grabber.discover_facilities() == ["Test"]  # reuses the index fetched by the check
    assert len(session.calls) == 3


@njm_offline_row
def outages_and_refusals_are_still_reported_as_down(t):
    down = BurstBlockedNJM({}, head_error=requests.ConnectionError("no route to host"))
    grabber, delays = blocked_grabber(t, down)
    assert REAL_AVAILABILITY_CHECK(grabber) is False
    assert down.calls == [("HEAD", "")] and delays == []

    njm_filter_grabber._STATE.reset()  # each case as in a new process: an outage pauses NJM process-wide
    missing = BurstBlockedNJM({"index.json": TEST_INDEX}, head_status=404)
    grabber, delays = blocked_grabber(t, missing)
    assert REAL_AVAILABILITY_CHECK(grabber) is False and missing.calls == [("HEAD", "")]

    njm_filter_grabber._STATE.reset()
    blocked = BurstBlockedNJM({"index.json": TEST_INDEX}, blocked=10)  # still blocked after the retries
    grabber, delays = blocked_grabber(t, blocked)
    assert REAL_AVAILABILITY_CHECK(grabber) is False
    assert delays == [1, 60] and len(blocked.calls) == 4


@njm_offline_row
def svo_fallback_tracks_and_recovers_partial_downloads(t):
    browser = SVOFilterBrowser(base_dir=str(t.tmp))
    listing = new_camera_svo()
    browser.list_facilities, browser.list_instruments = listing.list_facilities, listing.list_instruments
    browser.list_filters = lambda f, i: [FilterRow(f"NewFacility/NewCamera.{b}", b, "NewFacility", "NewCamera", "")
                                         for b in ("g", "r")]
    browser.metadata_client.get_filter = lambda fid: None if fid.endswith(".r") else flat_record(fid)
    library = FilterLibrary(t.tmp)
    library.svo = browser
    with pytest.raises(ComponentError, match="Incomplete passband"):
        library.published("NewFacility", "NewCamera")
    directory = t.tmp / "NewFacility" / "NewCamera"
    assert InstrumentManifest.load(directory).missing_files(directory) == ["r.dat"]
    browser.metadata_client.get_filter = flat_record
    assert library.published("NewFacility", "NewCamera") == directory
    assert FilterLibrary.published_complete(directory)


@njm_offline_row
def original_source_failure_falls_back_to_svo(t):
    def download_svo(rows):
        write_filter_file(t.tmp / "Paranal" / "HAWKI" / "J.dat", [4000., 5000.], [0.5, 0.9],
                          detector_type=PHOTON, source="svo", extra={"Components": "filter"})
        InstrumentManifest("Paranal", "HAWKI", ["J"]).save(t.tmp / "Paranal" / "HAWKI")

    library = FilterLibrary(t.tmp)
    library.svo = SimpleNamespace(
        list_facilities=lambda: [Facility("Paranal", "Paranal")],
        list_instruments=lambda f: [Instrument("Paranal", "HAWKI", "HAWKI")],
        list_filters=lambda f, i: [FilterRow("Paranal/HAWKI.J", "J", "Paranal", "HAWKI", "")],
        download_filters=download_svo)
    t.mp.setattr(filter_library, "get_builder", lambda name: raising(requests.ConnectionError("ESO offline")))
    directory = t.tmp / "Paranal" / "HAWKI"
    directory.mkdir(parents=True)
    assert library.published("Paranal", "HAWKI") == directory
    assert read_filter_header(directory / "J.dat")["DetectorTypeSource"] == "svo"


@njm_offline_row
def inventory_offline_cli_emits_clean_json(t):
    assert run_cli(t.mp, "filters-inventory", "--offline", "--json", "--base", str(t.tmp)) == 0
    report = json.loads(t.capsys.readouterr().out)
    assert report["instruments"]
    assert report["errors"] == []


# ======================================================================
# Filter properties: definitions, Vega and Sun quantities, A_f/A_V (test_filter_properties)
# ======================================================================
def box(lo=4000.0, hi=6000.0, ramp=1e-3, n=4001):
    """A top-hat filter with near-vertical edges (zero just outside lo..hi)."""
    inner = np.linspace(lo, hi, n)
    return np.concatenate([[lo - ramp], inner, [hi + ramp]]), np.concatenate([[0.0], np.ones(n), [0.0]])


def gaussian(center=5500.0, sigma=300.0):
    lam = np.linspace(center - 6 * sigma, center + 6 * sigma, 6001)
    return lam, np.exp(-0.5 * ((lam - center) / sigma) ** 2)


@row
def box_filter_definitions(t):
    lo, hi = 4000.0, 6000.0
    p = filter_properties(*box(lo, hi), detector_type="photon")
    assert p["WavelengthMean"] == pytest.approx((lo + hi) / 2, rel=1e-6)
    assert p["WidthEff"] == pytest.approx(hi - lo, rel=1e-5)
    assert p["FWHM"] == pytest.approx(hi - lo, rel=1e-5)
    assert p["WavelengthCen"] == pytest.approx((lo + hi) / 2, rel=1e-6)
    assert (p["WavelengthMin"], p["WavelengthMax"]) == (pytest.approx(lo), pytest.approx(hi))
    # photon-counting pivot: λp² = ∫ λ T dλ / ∫ T/λ dλ
    assert p["WavelengthPivot"] == pytest.approx(math.sqrt((hi**2 - lo**2) / 2 / math.log(hi / lo)), rel=1e-6)


@row
def vega_quantities_need_coverage_where_the_filter_transmits(t):
    lam, tr = gaussian(50000.0, 3000.0)
    p = filter_properties(lam, tr, "photon", vega=(np.array([900.0, 30000.0]), np.array([1e-9, 1e-12])))
    assert p["ZeroPointVega_Jy"] is None and p["WavelengthEff"] is None
    assert p["ABminusVega"] is None and p["STminusVega"] is None
    assert p["ZeroPointAB_Jy"] == AB_JY
    # A negligible tail beyond the spectrum counts as zero flux, as in SED_Tools photometry.
    vega = (np.linspace(900.0, 7400.0, 2000), np.full(2000, 1e-9))
    lam, tr = gaussian(5500.0, 300.0)  # tabulated out to 7300 A; above 1 % of peak only to ~6400 A
    tail = np.concatenate([lam, [8000.0]]), np.concatenate([tr, [0.0]])
    assert filter_properties(*tail, "photon", vega=vega)["ZeroPointVega_flam"] == pytest.approx(1e-9, rel=1e-6)


@row
def rejects_unusable_curves(t):
    with pytest.raises(ValueError):
        filter_properties([5000.0], [1.0])
    with pytest.raises(ValueError):
        filter_properties([4000.0, 5000.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        filter_properties([4000.0, 5000.0], [1.0, 1.0], detector_type="quantum")


@row
def afav_is_none_outside_the_laws_range(t):
    pytest.importorskip("dust_extinction")
    from dust_extinction.parameter_averages import F99

    lo, hi = F99.x_range  # inverse microns: 1000 A to 33333 A
    assert extinction_ratio(1e4 / hi * 1.001) is not None
    assert extinction_ratio(1e4 / hi * 0.999) is None
    assert extinction_ratio(1e4 / lo * 0.999) is not None
    assert extinction_ratio(1e4 / lo * 1.001) is None
    assert filter_properties(*gaussian(50000.0, 3000.0), "photon")["AfAv"] is None  # WISE W2-like
    law = extinction_law()
    assert law["valid_range_angstrom"] == pytest.approx([1000.0, 1e5 / 3])
    assert law["dust_extinction_version"] and law["R_V"] == 3.1


@row
def afav_is_none_without_dust_extinction(t):
    t.mp.setitem(sys.modules, "dust_extinction", None)  # makes the import fail
    t.mp.setitem(sys.modules, "dust_extinction.parameter_averages", None)
    t.mp.setattr(fp, "_f99_model", None)  # forget a model built by an earlier row
    with pytest.warns(UserWarning, match="dust_extinction"):
        p = filter_properties(*gaussian(5500.0, 300.0), "photon")
    assert p["AfAv"] is None and p["WavelengthPivot"] > 0
    assert extinction_ratio(5500.0) is None
    assert "valid_range_angstrom" not in extinction_law()


@row
def helpers_pass_the_sun_through(t):
    inst = t.tmp / "Test" / "Camera"
    inst.mkdir(parents=True)
    write_filter_file(inst / "V.dat", *gaussian(5500.0, 300.0), detector_type="photon", source="svo")
    sun = (np.linspace(1000.0, 30000.0, 100), np.full(100, 185.0))
    assert curve_properties(inst / "V.dat", sun=sun)["Fsun"] == pytest.approx(185.0)
    assert instrument_properties(inst, sun=sun)["V"]["Fsun"] == pytest.approx(185.0)


# ======================================================================
# Atmosphere sites: legacy and team sites, provenance, unstated airmass, and errors that say what to choose
# (test_filters_atmosphere_sites)
# ======================================================================
LEGACY_SITE = {"altitude_m": 2207.0, "entries": [{"airmass": 1.2, "file": "am1.20.dat", "pwv": None}],
               "notes": "", "schema": 1, "site": "ctio_des", "source": "DES standard atmosphere"}


def site_camera(root, facility="Test", instrument="Camera", *, default_site=None, sites=(), outside="error"):
    """A ground-based instrument with one band g (a Gaussian at 5000 Å) and the given atmosphere sites."""
    inst = root / facility / instrument
    inst.mkdir(parents=True)
    write_filter_file(inst / "g.dat", WL, gauss(5000, 200), detector_type=PHOTON,
                      source="svo", wavelength_unit="Angstrom", extra={"Components": "filter"})
    manifest = InstrumentManifest.load(inst)
    manifest.facility, manifest.instrument = facility, instrument
    manifest.ground_based = True
    manifest.default_site = default_site
    manifest.atmosphere_sites = list(sites)
    manifest.outside_range = outside
    manifest.save(inst)
    return inst


def team_curve(wavelength, transmission):
    """An atmosphere table as a team publishes it."""
    return DerivedCurve("u", lambda session: (np.array(wavelength), np.array(transmission)), "")


def team_atmosphere(root, name, entries, **kw):
    """A team atmosphere site; ``entries`` are (airmass, pwv, (wavelengths, transmission))."""
    write_site(root, team_site(name, [(am, pwv, team_curve(*curve)) for am, pwv, curve in entries], **kw), None)


@row
def legacy_site_manifest_round_trips_unchanged(t):
    text = json.dumps(LEGACY_SITE, indent=2, sort_keys=True) + "\n"
    put(t.tmp / "ctio_des", {"manifest.json": text})
    AtmosphereGrid.load(t.tmp / "ctio_des").save(t.tmp / "ctio_des")
    assert (t.tmp / "ctio_des" / "manifest.json").read_text() == text


@row
def instrument_manifest_without_extra_sites_has_no_new_key(t):
    inst = site_camera(t.tmp, default_site="lsst_std")
    assert "atmosphere_sites" not in json.loads((inst / "manifest.json").read_text())


@row
def grid_entries_need_a_numeric_airmass(t):
    with pytest.raises(ComponentError, match="unstated"):
        AtmosphereGrid("s", [{"airmass": None, "pwv": None, "file": "a.dat"}])


# Registering a team atmosphere as a site
@row
def write_site_records_provenance_and_derivation(t):
    verbatim = DerivedCurve("https://team/atm10.dat", derived([0.4, 0.7, 0.8]), derivation="", note="as published")
    site = team_site(entries=[(1.0, 1.5, verbatim),
                              (1.2, None, DerivedCurve("https://team/a | https://team/b", derived([0.5, 0.8, 0.9]),
                                                       "a / b"))],
                     terms="Cite us.", outside_range="error")
    site_dir = write_site(t.tmp, site, None)

    raw = json.loads((site_dir / "manifest.json").read_text())
    assert raw["kind"] == "team" and raw["model"] == "MODTRAN" and raw["terms"] == "Cite us."
    assert raw["entries"][0] == {"airmass": 1.0, "pwv": 1.5, "file": "am1.00_pwv1.5.dat",
                                 "source": "https://team/atm10.dat", "note": "as published"}
    assert raw["entries"][1]["derivation"] == "a / b"
    assert "unstated" not in raw
    original, computed = (read_filter_header(site_dir / e["file"]) for e in raw["entries"])
    assert original["ComponentsSource"] == "original" and computed["ComponentsSource"] == "derived"
    assert computed["Atmosphere"] == "site=here_team airmass=1.2"
    assert all(released_mesa_reads(site_dir / e["file"]) for e in raw["entries"])


@row
def write_site_without_provenance_keeps_the_legacy_layout(t):
    site = SiteAtmosphere("ctio_des", 2207.0, [(1.2, None, DerivedCurve("u", derived([0.5, 0.8, 0.9]), ""))],
                          "DES standard atmosphere")
    write_site(t.tmp, site, None)
    assert json.loads((t.tmp / "ctio_des" / "manifest.json").read_text()) == LEGACY_SITE
    assert "ComponentsSource" not in read_filter_header(t.tmp / "ctio_des" / "am1.20.dat")


@row
def unstated_airmass_is_listed_apart_and_selected_explicitly(t):
    write_site(t.tmp, team_site("plexus_team", entries=[(None, None, DerivedCurve("u", derived([0.5, 0.8, 0.9]), ""))]),
               None)
    raw = json.loads((t.tmp / "plexus_team" / "manifest.json").read_text())
    assert raw["entries"] == [] and raw["unstated"][0]["file"] == "am_unstated.dat"
    # Released SED_Tools lists every served site with float(e["airmass"]).
    assert sorted({float(e["airmass"]) for e in raw["entries"]}) == []

    grid = AtmosphereGrid.load(t.tmp / "plexus_team")
    assert grid.select(AtmosphereSpec("plexus_team", None)) == "am_unstated.dat"
    with pytest.raises(ComponentError, match="unstated airmass: airmass=None"):
        grid.select(AtmosphereSpec("plexus_team", 1.0))
    assert AtmosphereSpec.coerce({"site": "plexus_team", "airmass": None}).label == "plexus_team_amunstated"
    with pytest.raises(ComponentError, match="no pwv"):
        AtmosphereSpec("plexus_team", None, 2.0)


# Which sites an instrument may use
@row
def existing_single_site_instruments_are_unchanged(t):
    manifest = InstrumentManifest.load(site_camera(t.tmp / "filters", "LSST", "LSST", default_site="lsst_std"))
    assert manifest.sites() == ["lsst_std"]
    with pytest.raises(ComponentError,
                       match=r"observes from lsst_std; atmospheres from other sites \(.*\) are not mixed in"):
        resolve_atmosphere(manifest, AtmosphereSpec("paranal", 1.2, 2.5))


# Composition with team atmospheres
@row
def site_outside_range_overrides_the_instruments_policy(t):
    short = team_curve([4500.0, 5500.0], [0.9, 0.9])
    inst = site_camera(t.tmp / "filters", default_site="short_team", outside="edge")
    write_site(t.tmp / "atm", team_site("short_team", entries=[(1.2, None, short)], outside_range="error"), None)
    with pytest.raises(ComponentCoverageError):
        compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=t.tmp / "atm")
    # Unset on the site: the instrument's own convention (here edge) applies, as before.
    write_site(t.tmp / "atm2", team_site("short_team", entries=[(1.2, None, short)]), None)
    curve = compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=t.tmp / "atm2")
    assert curve.transmission.max() == pytest.approx(0.9 * 1.0, rel=1e-3)


@row
def add_component_points_atmospheres_to_sites(t):
    with pytest.raises(ComponentError, match="site of its own"):
        add_component(t.tmp, "atmosphere", [4000.0, 5000.0], [0.9, 0.9])


@row
def library_rebuilds_a_missing_team_site(t):
    njm_offline(t)
    inst = site_camera(t.tmp / "filters", "PAN-STARRS", "PS1")
    library, calls = FilterLibrary(t.tmp / "filters"), []

    def build(root, **kw):
        calls.append(kw)
        write_site(kw["atmosphere_root"], team_site("haleakala_ps1"), None)

    t.mp.setattr(filter_library, "get_builder", lambda name: calls.append(name) or build)
    path = library.prepare_atmosphere(InstrumentManifest.load(inst), AtmosphereSpec(None, 1.2))
    assert path.name == "am1.20.dat"
    assert calls[0] == "team_atmospheres" and calls[1]["sites"] == ["haleakala_ps1"]
    listing = library.atmospheres(remote=False)["haleakala_ps1"]
    assert listing["kind"] == "team" and listing["airmass"] == [1.2]


@row
def coverage_tolerance_allows_faint_wings_only(t):
    # Band: a Gaussian at 5000 A (sigma 200); atmosphere table from 4500 A.
    inst = site_camera(t.tmp / "filters", default_site="short_team")
    for name, tolerance in (("loose", 1e-2), ("tight", 1e-4)):
        team_atmosphere(t.tmp / name, "short_team", [(1.2, None, ([4500.0, 7000.0], [0.8, 0.8]))],
                        outside_range="error", coverage_tolerance=tolerance)
    curve = compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=t.tmp / "loose")
    assert curve.transmission[curve.wavelength < 4500].max() > 0  # the wing holds the edge value
    with pytest.raises(ComponentCoverageError, match="lies beyond it"):
        compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=t.tmp / "tight")


# Atmosphere and coverage errors say what to choose, not Python syntax
COVERAGE_ERROR = {
    "coverage_error_names_the_band_and_its_span": (
        {}, r"^Test/Camera band g: .* covers 4500-7000 A but the band spans"),
    "coverage_error_names_the_band_beyond_the_tolerance": (
        {"outside_range": "error", "coverage_tolerance": 1e-4}, r"^Test/Camera band g: .* lies beyond it"),
}


def _coverage_error(t, site_kw, match):
    # Used to be "am1.00_pwv1.dat covers 9000-56000 A but 70.25% of the band lies beyond it".
    inst = site_camera(t.tmp / "filters", default_site="short_team")
    team_atmosphere(t.tmp / "atm", "short_team", [(1.2, None, ([4500.0, 7000.0], [0.8, 0.8]))], **site_kw)
    with pytest.raises(ComponentCoverageError, match=match):
        compose_band(inst, "g", [FILTER], atmosphere={"airmass": 1.2}, atmosphere_root=t.tmp / "atm")


@row
def several_sites_error_lists_the_sites_to_choose_from(t):
    # Used to be "... sites ['hopkins_2mass', 'ctio_2mass']; choose one with atmosphere={'site': ..., ...}".
    twomass = InstrumentManifest(facility="2MASS", instrument="2MASS", bands=["J"], ground_based=True)
    with pytest.raises(ComponentError) as excinfo:
        resolve_atmosphere(twomass, AtmosphereSpec(None, None))
    assert str(excinfo.value) == "2MASS/2MASS has several atmosphere sites; choose a site: hopkins_2mass, ctio_2mass"


@row
def missing_conditions_error_says_what_the_site_has(t):
    # Used to be "No cached atmosphere for hopkins_2mass_am1.2; fetch or generate the requested conditions".
    flat = ([3000.0, 7000.0], [0.9, 0.9])
    team_atmosphere(t.tmp, "plexus_team", [(None, None, flat)])
    team_atmosphere(t.tmp, "grid_team", [(1.0, 2.5, flat), (1.2, 2.5, flat), (1.2, 5.0, flat)])
    messages = [
        (AtmosphereSpec("plexus_team", 1.2),
         "Site plexus_team has no atmosphere for airmass 1.2; it has unstated airmass (the atmosphere as published)"),
        (AtmosphereSpec("grid_team", 1.5, 2.5),
         "Site grid_team has no atmosphere for airmass 1.5 and PWV 2.5 mm; it has airmass 1, 1.2 with PWV 2.5, 5 mm"),
        (AtmosphereSpec("nowhere", 1.2), "No atmosphere of site nowhere is cached; fetch or generate the site first"),
    ]
    for spec, message in messages:
        with pytest.raises(ComponentError) as excinfo:
            atmosphere_file(t.tmp, spec)
        assert str(excinfo.value) == message


# ======================================================================
# A catalogue gap apart from a fetch failure (test_filters_catalogue_gaps)
# ======================================================================
@row
def search_reports_gaps_apart_from_fetch_failures(t):
    def instrument(bands, properties):
        return {"filters": bands, "detector_types": {}, "properties": properties, "info": None}

    # Fac/Cam is served whole; Fac/Old has no properties file (a gap); Other/IR's is missing (a failure).
    index = {"generated": "2026-10-27T00:00:00", "filters": {"facilities_detailed": {
        "Fac": {"instruments": {"Cam": instrument(["g"], "properties.json"), "Old": instrument(["U"], None)}},
        "Other": {"instruments": {"IR": instrument(["K"], "properties.json")}},
    }}}
    njm = Session({"index.json": json.dumps(index),
                   "filters/Fac/Cam/properties.json": json.dumps({"bands": {"g": {"WavelengthPivot": 4800.0}}}),
                   "filters/Other/IR/properties.json": requests.HTTPError("404", response=Resp(b"", 404))}, base=NJM)
    result = FilterCatalogue(t.tmp / "filters", mirror=mirror(njm.get)).search()
    assert list(result["band"]) == ["g", "U", "K"]
    assert result.attrs["gaps"] == ["Fac/Old: NJM has no properties file"]
    assert result.attrs["errors"] == ["Other/IR: NJM does not serve filters/Other/IR/properties.json"]


# ======================================================================
# Component variants, and a component's files from NJM (test_filters_component_variants)
# ======================================================================
def qe_variants(t):
    """Fac/Cam: band g, and QE variants a (the default) and b (from 2017)."""
    inst = t.tmp / "Fac" / "Cam"
    published_set(inst, "g", WL, gauss(5000, 300), "filter")
    m = InstrumentManifest.load(inst)
    add_component(inst, QE, WL, np.full(WL.size, 0.9), name="qe_a", source="a", manifest=m)
    add_component(inst, QE, WL, np.full(WL.size, 0.5), name="qe_b", source="b", manifest=m,
                  variant="b", variant_info={"epoch": "from 2017", "description": "B"})
    m.variants["qe"]["a"] = {"files": {"*": "components/qe/qe_a.dat"}, "default": True}
    m.save(inst)
    return inst


@row
def variant_is_selected_only_on_request(t):
    inst = qe_variants(t)
    default = compose_band(inst, "g", "filter+qe")
    chosen = compose_band(inst, "g", "filter+qe", variants={"qe": "b"})
    assert default.transmission.max() == pytest.approx(0.9)
    assert chosen.transmission.max() == pytest.approx(0.5)
    assert "[qe=b]" in chosen.header["Recipe"]


@row
def variants_round_trip_and_are_listed(t):
    inst = qe_variants(t)
    m = InstrumentManifest.load(inst)
    assert m.variants["qe"]["b"]["epoch"] == "from 2017"
    assert m.variants["qe"]["b"]["sources"] == {"*": "b"}
    assert "components/qe/qe_b.dat" in m.variant_files()
    assert m.describe(inst)["variants"]["qe"]["b"]["bands"] == ["*"]
    (inst / "components/qe/qe_b.dat").unlink()
    assert m.missing_files(inst) == ["components/qe/qe_b.dat"]
    assert m.local(inst).variants["qe"]["b"]["files"] == {}


@row
def manifest_without_variants_is_written_unchanged(t):
    inst = t.tmp / "Fac" / "Cam"
    published_set(inst, "g", WL, gauss(5000, 300), "filter")
    InstrumentManifest.load(inst).save(inst)
    assert "variants" not in json.loads((inst / "manifest.json").read_text())


@row
def variant_paths_must_stay_inside_the_instrument(t):
    with pytest.raises(ComponentError):
        InstrumentManifest("F", "I", ["g"], variants={"qe": {"x": {"files": {"*": "../other/qe.dat"}}}})
    with pytest.raises(ComponentError):
        InstrumentManifest("F", "I", ["g"], variants={"qe": {"bad/name": {"files": {}}}})


@row
def njm_downloads_variant_files(t):
    manifest = {"facility": "F", "instrument": "I", "bands": ["g"], "components": {},
                "variants": {"qe": {"a": {"files": {"*": "components/qe/qe_a.dat"}}}}}
    njm = njm_filters(t, t.tmp, {"filters/F/I/manifest.json": json.dumps(manifest),
                                 "filters/F/I/components/qe/qe_a.dat": "Wavelength,Transmission\n4000,0.5\n5000,0.5\n"})
    assert njm.download_components("F", "I") == 1
    assert (t.tmp / "F/I/components/qe/qe_a.dat").exists()


# ======================================================================
# Failures are raised or warned about, never passed over silently (test_filters_failures)
# ======================================================================
MIRROR_CURVE = "Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=svo\n4000,0.1\n5000,1\n6000,0.1\n"


def no_svo_url(t):
    """SVO is asked at its own address (the autouse fixture of test_filters_failures and test_svo_fail_fast)."""
    t.mp.delenv(fm.SVO_FPS_URL_ENV, raising=False)


def _default_svo_then(t, scenario):
    """The rows of test_filters_failures and test_svo_fail_fast: their autouse fixture unset SVO's address."""
    no_svo_url(t)
    scenario(t)


def three_samples(path, header="Wavelength,Transmission,WavelengthUnit=Angstrom"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n4000.0,0.1\n5000.0,1.0\n6000.0,0.1\n")
    return path


def flat_published(directory):
    """``directory`` holding SVO's flat g, filter+optics+qe, without its components; returns the directory."""
    directory.mkdir(parents=True)
    write_filter_file(directory / "g.dat", [4000.0, 6000.0], [1.0, 1.0], detector_type=PHOTON, source="svo",
                      wavelength_unit="Angstrom", extra={"Components": "filter+optics+qe"})
    return directory


@default_svo_row
def instrument_colors_refuses_an_unreadable_curve(t):
    good = three_samples(t.tmp / "g.dat")
    empty = t.tmp / "r.dat"
    empty.write_text("Wavelength,Transmission\n")
    with pytest.raises(ValueError, match="r.dat"):
        instrument_colors([good, empty])


@default_svo_row
def unreadable_installed_catalogue_file_is_refused(t):
    root = t.tmp / "filters"
    put(root / "Fac" / "Cam", {"g.dat": "Wavelength,Transmission\n4000,1\n6000,1\n",  # an installed copy holds curves
                               "properties.json": "{not json"})
    catalogue = FilterCatalogue(root, mirror=mirror(raising(requests.ConnectionError("mirror down")), available=False))
    with pytest.raises(CatalogueError, match="properties.json is unreadable"):
        catalogue.properties("Fac", "Cam")


@default_svo_row
def unknown_components_header_is_reported(t):
    headers = {"g": {"Components": "filter+warp"}, "r": {"Components": "filter"}}
    with t.caplog.at_level(logging.WARNING):
        manifest = InstrumentManifest.from_headers("Fac", "Cam", headers)
    assert manifest.published_bands == {"r": "filter"}
    assert "Fac/Cam band g" in t.caplog.text and "'warp'" in t.caplog.text


@default_svo_row
def svo_bulk_download_finishes_then_reports_failed_instruments(t):
    downloaded = []

    def download(filters):
        downloaded.append(filters[0])
        if filters[0] == "A/one.g":
            raise RuntimeError("1 of 1 SVO filters could not be saved")

    browser = SimpleNamespace(
        list_instruments=lambda key: [Instrument(key, "one", "One"), Instrument(key, "two", "Two")],
        list_filters=lambda fac, inst: [f"{fac}/{inst}.g"], download_filters=download)
    with pytest.raises(RuntimeError, match="Bulk download incomplete:\nA/One: 1 of 1"):
        _download_facilities(browser, [Facility("A", "A"), Facility("B", "B")])
    assert downloaded == ["A/one.g", "A/two.g", "B/one.g", "B/two.g"]


# njm_filter_grabber, on a stand-in session
@default_svo_row
def directory_listing_is_empty_only_when_nothing_is_there(t):
    grabber = njm_filters(t, t.tmp, {"filters/Fac/": '<a href="Cam/">Cam/</a>', "filters/Down/": 500}, tags=True)
    assert grabber.discover_instruments("Fac") == ["Cam"]
    assert grabber.discover_instruments("Nope") == []
    with pytest.raises(requests.HTTPError, match="500"):
        grabber.discover_instruments("Down")


@default_svo_row
def unreadable_index_falls_back_to_the_listing_with_a_warning(t):
    grabber = njm_filters(t, t.tmp, {"index.json": "{broken", "filters": '<a href="Fac/">Fac/</a>',
                                     "filters/Fac/Cam/Cam": 500, "filters/Fac/Cam/": '<a href="g.dat">g.dat</a>'},
                          tags=True)
    with t.caplog.at_level(logging.WARNING):
        assert grabber.discover_facilities() == ["Fac"]
        assert grabber.discover_filters("Fac", "Cam") == ["g.dat"]  # as an index file lists it (it gave "g")
    assert "Cannot read NJM's index.json" in t.caplog.text
    assert ("Cannot read NJM's index of Fac/Cam (500 for filters/Fac/Cam/Cam); listing its directory instead"
            in t.caplog.text)


@default_svo_row
def catalogue_files_skip_unserved_and_report_unreadable(t):
    grabber = njm_filters(t, t.tmp, {"filters/Fac/Cam/info.json": "{not json"}, tags=True)
    with t.caplog.at_level(logging.WARNING):
        assert grabber.download_catalogue_files("Fac", "Cam") == []
    assert "info.json for Fac/Cam was not saved" in t.caplog.text
    assert "properties.json" not in t.caplog.text  # not served (404): skipped quietly


@default_svo_row
def manifest_is_none_only_when_not_served(t):
    manifest = {"facility": "Fac", "instrument": "Cam", "bands": ["g"]}
    grabber = njm_filters(t, t.tmp, {"filters/Fac/Cam/manifest.json": json.dumps(manifest),
                                     "filters/Fac/Other/manifest.json": json.dumps(manifest),
                                     "filters/Fac/Bad/manifest.json": '{"facility": "Fac"}',
                                     "filters/Fac/Down/manifest.json": 503}, tags=True)
    assert grabber.fetch_manifest("Fac", "Cam") == manifest
    assert grabber.fetch_manifest("Fac", "Nope") is None
    with pytest.raises(ValueError, match="describes Fac/Cam"):
        grabber.fetch_manifest("Fac", "Other")
    with pytest.raises(ValueError, match="unreadable manifest.json for Fac/Bad"):
        grabber.fetch_manifest("Fac", "Bad")
    with pytest.raises(requests.HTTPError, match="503"):
        grabber.fetch_manifest("Fac", "Down")


@default_svo_row
def atmosphere_is_false_only_when_not_served(t):
    grabber = njm_filters(t, t.tmp, {"atmospheres/down/manifest.json": 500}, tags=True)
    assert grabber.download_atmosphere(str(t.tmp), "nowhere") is False
    with pytest.raises(requests.HTTPError, match="500"):
        grabber.download_atmosphere(str(t.tmp), "down")


@default_svo_row
def download_all_filters_finishes_then_lists_incomplete_instruments(t):
    grabber, attempted = njm_filters(t, t.tmp, {}, tags=True), []

    def instruments(facility):
        if facility == "Down":
            raise requests.HTTPError("500 for filters/Down/")
        return ["A", "B", "C"]

    def download(facility, instrument):
        attempted.append(instrument)
        if instrument == "A":
            raise RuntimeError("NJM download of Fac/A is incomplete")
        if instrument == "B":
            raise requests.HTTPError("502 for filters/Fac/B/")
        return 2

    grabber.discover_facilities, grabber.discover_instruments = lambda: ["Down", "Fac"], instruments
    grabber.download_filters = download
    with pytest.raises(RuntimeError, match="3 NJM downloads are incomplete:\nDown: 500 .*\n"
                                           "Fac/A: NJM download of Fac/A is incomplete\nFac/B: 502"):
        grabber.download_all_filters()
    assert attempted == ["A", "B", "C"]


@default_svo_row
def svo_failure_while_checking_a_mirror_curve_keeps_the_curve(t):
    grabber = njm_filters(t, t.tmp, {"filters/Fac/Cam/Cam": "g.dat\n", "filters/Fac/Cam/g.dat": MIRROR_CURVE},
                          tags=True)
    t.mp.setattr("sed_tools.njm_filter_grabber.tag_filter_file",
                 raising(fm.SVOUnreachableError("SVO Filter Profile Service unreachable")))
    with t.caplog.at_level(logging.WARNING):
        assert grabber.download_filters("Fac", "Cam") == 1
    assert (t.tmp / "Fac/Cam/g.dat").read_text() == MIRROR_CURVE
    assert ("kept the mirror's headers of g.dat, which could not be checked against SVO "
            "(SVO Filter Profile Service unreachable)") in t.caplog.text


# filter_library with NJM failures
@default_svo_row
def njm_components_failure_does_not_discard_complete_curves(t):
    manifest = {"facility": "Fac", "instrument": "Cam", "bands": ["g"],
                "components": {"qe": {"*": "components/qe/ccd.dat"}}}
    library = library_with(t.tmp, njm_filters(t, t.tmp, {
        "filters/Fac/Cam/Cam": "g.dat\n", "filters/Fac/Cam/g.dat": MIRROR_CURVE,
        "filters/Fac/Cam/manifest.json": json.dumps(manifest), "filters/Fac/Cam/components/qe/ccd.dat": 500},
        tags=True))
    with t.caplog.at_level(logging.WARNING):
        assert library.published("Fac", "Cam") == t.tmp / "Fac" / "Cam"
    assert "NJM provided every curve of Fac/Cam, but" in t.caplog.text and "components/qe/ccd.dat" in t.caplog.text


@default_svo_row
def availability_and_catalogues_record_njm_failures(t):
    down = raising(requests.ConnectionError("NJM timed out"))
    library = library_with(t.tmp, SimpleNamespace(is_available=lambda: True, fetch_manifest=down,
                                                  discover_facilities=down),
                           SimpleNamespace(list_facilities=lambda: [Facility("SvoOnly", "SvoOnly")],
                                           list_instruments=lambda key: []))
    info = library.available("Fac", "Cam")
    assert info["availability_source"] == "unknown" and "NJM: NJM timed out" in info["errors"]
    assert "NJM: NJM timed out" in library.facilities()["errors"]
    assert "NJM SvoOnly: NJM timed out" in library.instruments("SvoOnly")["errors"]

    library.mirror = SimpleNamespace(is_available=lambda: False)
    assert "NJM unavailable" in library.instruments("SvoOnly")["errors"]


@default_svo_row
def njm_component_failure_is_reported_before_the_original_source(t):
    directory = flat_published(t.tmp / "Fac" / "Cam")
    library = library_with(t.tmp, SimpleNamespace(is_available=lambda: True, download_components=raising(
        RuntimeError("1 component files of Fac/Cam were not downloaded"))))
    t.mp.setitem(catalog.BUILDERS_FOR, ("Fac", "Cam"), catalog.BuilderRef("test"))
    t.mp.setattr(filter_library, "get_builder", lambda name: lambda root, **kwargs: None)
    with t.caplog.at_level(logging.WARNING):
        assert not library.prepare_components(directory, components="filter+qe")
    assert ("NJM cannot provide the components of Fac/Cam (1 component files of Fac/Cam were not downloaded); "
            "trying their original source") in t.caplog.text


@default_svo_row
def njm_atmosphere_failure_is_reported_before_its_source(t):
    library = library_with(t.tmp / "filters", SimpleNamespace(download_atmosphere=raising(
        requests.HTTPError("502 for atmospheres/paranal/manifest.json"))))
    t.mp.setattr(filter_library, "get_builder", lambda name: lambda root, **kwargs: skycalc_paranal(
        library.atmosphere_root / "paranal"))
    manifest = InstrumentManifest("Paranal", "Cam", ["g"], ground_based=True, default_site="paranal")
    with t.caplog.at_level(logging.WARNING):
        path = library.prepare_atmosphere(manifest, AtmosphereSpec(None, 1.2, 2.5))
    assert path.name == "a.dat"
    assert "NJM cannot provide the paranal_am1.2_pwv2.5 atmosphere (502" in t.caplog.text


# filter_metadata: SVO failures are no longer taken for "SVO has no answer"
@default_svo_row
def svo_error_is_raised_and_nothing_is_written(t):
    path = three_samples(t.tmp / "GAIA" / "GAIA" / "G.dat", header="Wavelength,Transmission,FilterID=GAIA/GAIA3.G")
    before = path.read_bytes()
    client = SVOClient(fail={"GAIA/GAIA3.G": requests.HTTPError("500 Server Error")})
    with pytest.raises(requests.HTTPError, match="500"):
        fm.tag_filter_file(path, client=client)
    assert path.read_bytes() == before  # no "energy, assumed" placeholder


@default_svo_row
def svo_timeout_during_curve_matching_is_raised(t):
    path = three_samples(t.tmp / "GAIA" / "GAIA" / "G.dat", header="Wavelength,Transmission")
    before = path.read_bytes()
    client = fm.SVOMetadataClient(session=SVOSession(error=requests.ReadTimeout("read timed out")))
    with pytest.raises(requests.ReadTimeout):
        fm.tag_filter_file(path, client=client)
    assert path.read_bytes() == before


@default_svo_row
def unreadable_svo_answer_is_raised(t):
    client = fm.SVOMetadataClient(session=SVOSession(content=b"<html><body>Service down</body></html>"))
    with pytest.raises(ValueError, match="VOTABLE"):
        client.facility_filter_ids("GAIA")
    with pytest.raises(ValueError, match="VOTABLE"):
        client.get_filter("GAIA/GAIA3.G")


@default_svo_row
def failed_facility_listing_is_raised_and_not_cached_as_empty(t):
    session = SVOSession(status=500)
    client = fm.SVOMetadataClient(session=session)
    with pytest.raises(requests.HTTPError):
        client.facility_filter_ids("GAIA")
    session.status, session.content = 200, (
        b'<?xml version="1.0"?><VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
        b'<RESOURCE><INFO name="QUERY_STATUS" value="ERROR">none</INFO></RESOURCE></VOTABLE>')
    assert client.facility_filter_ids("GAIA") == []


@default_svo_row
def tag_tree_tags_what_it_can_then_lists_the_failures(t):
    instrument = t.tmp / "Fac" / "Cam"
    for band in "ab":
        three_samples(instrument / f"{band}.dat", header=f"Wavelength,Transmission,FilterID=Fac/Cam.{band}")
    before = (instrument / "a.dat").read_bytes()
    record = svo_record("Fac/Cam.b", fm.PHOTON, [4000.0, 5000.0, 6000.0], [0.1, 1.0, 0.1], "Filter")
    client = SVOClient({"Fac/Cam.b": record}, fail={"Fac/Cam.a": requests.HTTPError("502 Bad Gateway")})
    with pytest.raises(RuntimeError, match=r"(?s)1 of 2 filter files could not be tagged.*a\.dat: 502 Bad Gateway"):
        fm.tag_filter_tree(t.tmp, client=client)
    assert (instrument / "a.dat").read_bytes() == before
    assert fm.read_filter_header(instrument / "b.dat")["DetectorType"] == fm.PHOTON


@default_svo_row
def tag_tree_stops_at_once_when_svo_is_unreachable(t):
    for band in "ab":
        three_samples(t.tmp / "Fac" / "Cam" / f"{band}.dat", header=f"Wavelength,Transmission,FilterID=Fac/Cam.{band}")
    down = fm.SVOUnreachableError("SVO Filter Profile Service unreachable")
    with pytest.raises(fm.SVOUnreachableError):
        fm.tag_filter_tree(t.tmp, client=SVOClient(fail={"Fac/Cam.a": down, "Fac/Cam.b": down}))


@default_svo_row
def unverified_ssl_fallback_is_a_warning(t):
    def head(url, timeout=None, verify=True):
        if verify:
            raise requests.exceptions.SSLError("certificate expired")
        return SimpleNamespace(status_code=200)

    grabber = NJMFilterGrabber.__new__(NJMFilterGrabber)
    grabber.base_url, grabber.session = NJM, SimpleNamespace(head=head, verify=True)
    with t.caplog.at_level(logging.WARNING):
        assert grabber._check_availability() is True
    assert grabber.session.verify is False
    assert "NJM's SSL certificate could not be verified" in t.caplog.text


@default_svo_row
def unreadable_served_catalogue_file_is_refused(t):
    catalogue = FilterCatalogue(t.tmp / "filters", mirror=mirror(lambda url, **kwargs: Resp(b"\xff{")))
    with pytest.raises(CatalogueError, match="NJM served an unreadable index.json"):
        catalogue.index()


@default_svo_row
def published_carries_every_source_reason(t):
    library = library_with(t.tmp, SimpleNamespace(is_available=lambda: True, _download_set=raising(
        RuntimeError("NJM download of Fac/Cam is incomplete"))), SimpleNamespace(
        list_facilities=lambda: [Facility("Fac", "Fac")],
        list_instruments=lambda key: [Instrument("Fac", "Cam", "Cam")],
        list_filters=lambda fac, inst: [FilterRow("Fac/Cam.g", "g", "Fac", "Cam", "")],
        download_filters=raising(fm.SVOUnreachableError("SVO Filter Profile Service unreachable"))))
    t.mp.setitem(catalog.BUILDERS_FOR, ("Fac", "Cam"), catalog.BuilderRef("test", publishes=True))
    t.mp.setattr(filter_library, "get_builder", lambda name: lambda root, **kwargs: None)
    with t.caplog.at_level(logging.WARNING), pytest.raises(ComponentError) as info:
        library.published("Fac", "Cam")
    assert str(info.value) == ("No source can provide Fac/Cam: NJM: NJM download of Fac/Cam is incomplete; "
                               "original source: the builder left an incomplete set; "
                               "SVO: SVO Filter Profile Service unreachable")
    assert ("The original source cannot provide Fac/Cam (the builder left an incomplete set); trying SVO"
            in t.caplog.text)


@default_svo_row
def svo_error_on_a_listed_candidate_is_raised(t):
    path = three_samples(t.tmp / "GAIA" / "GAIA" / "G.dat", header="Wavelength,Transmission")
    client = SVOClient(listing=["GAIA/GAIA3.G"], fail={"GAIA/GAIA3.G": requests.HTTPError("503 Service Unavailable")})
    with pytest.raises(requests.HTTPError, match="503"):
        fm.resolve_detector_type(path, client=client)


# ======================================================================
# Filters with NJM or SVO down: NJM's names, one folder per product, outages announced once
# (test_filters_njm_down)
# ======================================================================
def gaia3_from_svo(root):
    """SVO, holding 4 of NJM's 6 GAIA/GAIA bands under GAIA/GAIA3."""
    return svo_browser(root, {("GAIA", "GAIA3"): ["G", "Gbp", "Grp", "Grvs"]})


def gaia_partly_from_svo(t, curve=CURVE):
    """GAIA/GAIA installed from SVO while NJM was down, 4 of its 6 bands; returns the filter root and the files
    NJM serves for it, every band as ``curve``."""
    bundled(t, {"GAIA": GAIA})
    bands, root = GAIA["GAIA"]["bands"], t.tmp / "filters"
    library_with(root, NJM_DOWN, gaia3_from_svo(root)).published("GAIA", "GAIA")  # with a warning
    return root, {"index.json": njm_index({"GAIA": {"GAIA": {"filters": bands, "size_bytes": 1}}}),
                  "filters/GAIA/GAIA/GAIA": "".join(f"{b}.dat\n" for b in bands),
                  **{f"filters/GAIA/GAIA/{b}.dat": curve for b in bands}}


def gaia_from_svo(t):
    """Every FilterLibrary sees NJM down and an SVO holding 4 of GAIA/GAIA's 6 bands; returns the filter root."""
    bundled(t, {"GAIA": GAIA})
    root = t.tmp / "filters"
    browser = gaia3_from_svo(root)
    t.mp.setattr(FilterLibrary, "mirror", NJM_DOWN)
    t.mp.setattr(FilterLibrary, "svo", property(lambda self: browser))
    t.mp.setenv("COLUMNS", "80")
    return root


def hfi_catalogue(t, info, properties, *, filter_ids):
    """A catalogue client for NJM's Planck/HFI beside 0.4.1's PLANCK/HFI from SVO (bands 100GHz and 857GHz,
    their curves naming SVO's filter IDs when ``filter_ids``)."""
    header = "Wavelength,Transmission" + (",FilterID=PLANCK/HFI.{}" if filter_ids else "") + "\n1,1\n2,1\n"
    put(t.tmp / "filters" / "PLANCK" / "HFI", {**{f"{b}.dat": header.format(b) for b in ("100GHz", "857GHz")},
                                              "HFI": "100GHz.dat\n857GHz.dat\n"})
    njm = Session({"index.json": njm_index({"Planck": {"HFI": {
                       "filters": ["100", "100_svo", "857", "857_svo"], "properties": "properties.json",
                       "info": "info.json"}}}),
                   "filters/Planck/HFI/HFI": "100.dat\n857.dat\n",
                   "filters/Planck/HFI/info.json": json.dumps(info),
                   "filters/Planck/HFI/properties.json": json.dumps(properties)}, base=NJM)
    return FilterCatalogue(t.tmp / "filters", mirror=mirror(njm.get))


def njm_no_answer(*args, **kwargs):
    """What requests raises when NJM does not answer, as with the network down (plain_reason: could not connect)."""
    raise requests.ConnectionError("HTTPSConnectionPool(host='data.nialljmiller.com', port=443): Max retries "
                                   "exceeded (Caused by NewConnectionError('Failed to establish a new connection'))")


# The pickers name SVO's copies after NJM while NJM is down
@row
def an_installed_copy_under_svos_name_stays_its_own_entry(t):
    bundled(t, {"GAIA": GAIA})
    root = t.tmp / "filters"
    write_set(root / "GAIA" / "GAIA3", ["G"])  # downloaded from SVO by 0.4.1
    library = library_with(root, NJM_DOWN, svo_stub({"GAIA": ""}, {"GAIA": {"GAIA3": "eDR3"}}))
    assert labels(library.instruments("GAIA")) == [("DR2", ["original"]), ("DR2r", ["original"]),
                                                  ("GAIA", ["svo"]), ("GAIA3", ["local"])]


# NJM's products from SVO, under NJM's names
@row
def an_svo_curve_that_differs_from_njms_is_used_with_a_warning(t):
    bundled(t, {"Roman": {"WFI": {"svo_instruments": ["Roman/WFI"], "bands": ["F062", "F087"],
                                  "svo_bands": {"Roman/WFI": ["F087"]}}}})
    root = t.tmp / "filters"
    library = library_with(root, NJM_DOWN, svo_browser(root, {("Roman", "WFI"): ["F062", "F087"]}))
    with t.caplog.at_level(logging.WARNING):
        directory = library.published("Roman", "WFI")
    assert sorted(p.name for p in directory.glob("*.dat")) == ["F062.dat", "F087.dat"]
    assert "SVO's F062 of Roman/WFI differ from NJM's current curves" in t.caplog.text


@row
def fetch_from_njm_later_replaces_svo_curves_that_differ_and_leaves_no_duplicates(t):
    root, files = gaia_partly_from_svo(t)
    assert not FilterLibrary.published_complete(root / "GAIA" / "GAIA")
    bands, library = GAIA["GAIA"]["bands"], library_with(root, njm_filters(t, root, files))
    with t.caplog.at_level(logging.WARNING):
        assert library.published("GAIA", "GAIA") == root / "GAIA" / "GAIA"
    directory = root / "GAIA" / "GAIA"
    assert sorted(p.stem for p in directory.glob("*.dat")) == bands
    assert t.caplog.text.count("with NJM's copies, which differ") == 1  # said once, not silent
    assert "replaced G.dat, Gbp.dat, Grp.dat, Grvs.dat with NJM's copies" in t.caplog.text
    assert FilterLibrary.published_complete(directory)
    verified = json.loads((root.parent / "cache" / "njm" / "filters" / "GAIA" / "GAIA" /
                           "verified_curves.json").read_text())
    assert sorted(verified["files"]) == [f"{b}.dat" for b in bands]


# One installed folder per product, whatever its spelling
@row
def njm_download_completes_a_folder_installed_under_another_spelling(t):
    bundled(t, {"Gemini": {"GMOS_N": {"svo_instruments": ["Gemini/GMOS-N"]}}})
    root = t.tmp / "filters"
    legacy = write_set(root / "Gemini" / "GMOS-N", ["g"], manifest_bands=["g", "r"])  # incomplete
    library = library_with(root, njm_filters(t, root, {
        "index.json": njm_index({"Gemini": {"GMOS_N": {}}}), "filters/Gemini/GMOS_N/GMOS_N": "g.dat\nr.dat\n",
        "filters/Gemini/GMOS_N/g.dat": CURVE, "filters/Gemini/GMOS_N/r.dat": CURVE}))
    assert library.installed_directory("Gemini", "GMOS_N") == legacy
    assert library.published("Gemini", "GMOS_N") == legacy
    assert sorted(p.name for p in (root / "Gemini").iterdir()) == ["GMOS-N"]  # no second copy beside it
    assert (legacy / "GMOS-N").read_text() == "g.dat\nr.dat\n"  # the index MESA looks for in that folder
    assert library.available("Gemini", "GMOS_N")["availability_source"] == "local"


@row
def a_complete_folder_under_another_spelling_is_used_as_it_is(t):
    bundled(t)
    root = t.tmp / "filters"
    legacy = write_set(root / "PLANCK" / "HFI", ["100GHz"])
    assert library_with(root, NJM_DOWN).published("Planck", "HFI") == legacy


@row
def a_builder_that_would_write_beside_an_installed_spelling_is_not_run(t):
    bundled(t)
    root = t.tmp / "filters"
    legacy = write_set(root / "Herschel" / "Pacs_", ["blue"], manifest_bands=["blue", "red"])  # incomplete
    library = library_with(root, NJM_DOWN, svo_stub({}, {}, down=True))
    t.mp.setattr(filter_library, "get_builder", lambda name: pytest.fail("the builder was run"))
    with t.caplog.at_level(logging.WARNING), pytest.raises(ComponentError) as raised:
        library.published("Herschel", "PACS")
    assert f"its builder writes Herschel/PACS, which would be a second copy beside {legacy}" in str(raised.value)
    assert "trying SVO (the original source is not used: its builder writes Herschel/PACS" in t.caplog.text
    assert sorted(p.name for p in (root / "Herschel").iterdir()) == ["Pacs_"]


@row
def two_installed_spellings_stop_unless_one_is_exact(t):
    root = t.tmp / "filters"
    write_set(root / "Gemini" / "GMOS-N", ["g"])
    write_set(root / "Gemini" / "GMOS N", ["g"])
    library = FilterLibrary(root)
    with pytest.raises(ComponentError, match="installed more than once.*GMOS N.*GMOS-N"):
        library.published("Gemini", "GMOS_N")
    assert library.installed_directory("Gemini", "GMOS-N") == root / "Gemini" / "GMOS-N"  # exact wins


# SVO's availability is known process-wide
@row
def svo_that_does_not_answer_is_waited_for_once_per_process(t):
    t.mp.setattr(requests.Session, "send", njm_no_answer)  # the old test asked NJM itself, which did not answer
    calls = []

    def get(session, url, **kwargs):
        calls.append(url)
        raise requests.ReadTimeout("read timed out")

    t.mp.setattr(svo_filter_grabber, "svo_get", get)
    first = FilterLibrary(t.tmp / "a")
    errors = first.facilities()["errors"] + first.instruments("GAIA")["errors"]
    second = FilterLibrary(t.tmp / "b")  # another library, as Filters.fetch makes
    errors += second.facilities()["errors"] + second.instruments("GAIA")["errors"]
    assert len(calls) == 1
    svo_errors = {e for e in errors if e.startswith("SVO")}
    assert len(svo_errors) == 1  # the same words each time, so the CLI prints them once
    [error] = svo_errors
    assert error.startswith("SVO: SVO Filter Profile Service did not answer") and "(ReadTimeout)" in error
    assert error.endswith("SVO is not asked again for 5 minutes.")
    with pytest.raises(SVOUnreachableError):
        SVOFilterBrowser(base_dir=str(t.tmp / "c")).list_facilities()
    assert len(calls) == 1


@row
def svo_is_asked_again_after_its_pause(t):
    clock = [1000.0]
    t.mp.setattr(svo_filter_grabber, "monotonic", lambda: clock[0])
    t.mp.setattr(svo_filter_grabber, "svo_get", raising(SVOUnreachableError("SVO Filter Profile Service unreachable.")))
    with pytest.raises(SVOUnreachableError):
        SVOFilterBrowser(base_dir=str(t.tmp)).list_facilities()
    assert svo_filter_grabber.svo_unavailable()
    clock[0] += svo_filter_grabber.SVO_PAUSE
    assert svo_filter_grabber.svo_unavailable() is None


# filters-info and filters-cite
@row
def citation_titles_are_one_line_and_url_keys_are_plain(t):
    text = format_citations({"instruments": [], "acknowledgements": [], "errors": [], "references": [
        {"key": "bibcode:2016A&A...595A...1G", "short": "Prusti et al. 2016", "title": "The\n          Gaia\n  mission",
         "doi": "10.1051/0004-6361/201629272", "used_by": ["GAIA/DR2"]},
        {"key": "url:https://www.cosmos.esa.int/web/gaia/iow_20180316/",
         "url": "https://www.cosmos.esa.int/web/gaia/iow_20180316/", "used_by": ["GAIA/DR2"]}]})
    assert '[1] Prusti et al. 2016. "The Gaia mission". doi:10.1051/0004-6361/201629272  <- GAIA/DR2' in text
    assert "[2] https://www.cosmos.esa.int/web/gaia/iow_20180316/  <- GAIA/DR2" in text


# What available() says without NJM
@row
def available_from_svo_describes_njms_product_and_its_missing_bands(t):
    bundled(t, {"GAIA": GAIA})
    root = t.tmp / "filters"
    info = library_with(root, NJM_DOWN, gaia3_from_svo(root)).available("GAIA", "GAIA")
    assert info["availability_source"] == "svo" and info["bands"] == ["G", "Gbp", "Grp", "Grvs"]
    assert info["notes"] == ["SVO has no curve that is NJM's Gbp_bright, Gbp_faint"]


# Installed atmosphere sites follow NJM's manifest
SERVED = dict(STALE, citation="lsst/throughputs v1.9.2", terms="No licence.",
              entries=[{"airmass": 1.0, "pwv": None, "file": "am1.00.dat"},
                       {"airmass": 1.2, "pwv": None, "file": "am1.20.dat"}])


@row
def a_stale_installed_atmosphere_takes_njms_manifest_and_keeps_its_grids(t):
    root = t.tmp / "data"
    site = atmosphere_site(root, "lsst_std", STALE, ["am1.20.dat"])
    grid = (site / "am1.20.dat").read_bytes()
    served = json.dumps(SERVED, indent=2, sort_keys=True) + "\n"
    library = library_with(root / "filters", njm_filters(t, root / "filters", {
        "atmospheres/": '<a href="lsst_std/">lsst_std/</a>', "atmospheres/lsst_std/manifest.json": served}))
    with t.caplog.at_level(logging.WARNING):
        listed = library.atmospheres()["lsst_std"]
    assert listed["citation"] == "lsst/throughputs v1.9.2" and listed["airmass"] == [1.0, 1.2]
    assert (site / "manifest.json").read_text() == served and (site / "am1.20.dat").read_bytes() == grid
    assert t.caplog.text == ""  # a routine update: NJM's citation and terms, a new grid point
    requested = len(library.mirror.session.calls)
    library.atmospheres()  # checked today: only the site list is asked for
    assert library.mirror.session.calls[requested:] == ["atmospheres/"]


@row
def points_built_here_and_installed_files_njm_describes_otherwise_are_kept(t):
    root = t.tmp / "data"
    local = dict(STALE, entries=[{"airmass": 1.3, "pwv": None, "file": "am1.20.dat"},
                                 {"airmass": 1.5, "pwv": 2.5, "file": "skycalc_am1.50.dat"},
                                 {"airmass": 2.0, "pwv": None, "file": "dropped_by_njm.dat"}])  # never fetched
    site = atmosphere_site(root, "lsst_std", local, ["am1.20.dat", "skycalc_am1.50.dat"])
    library = library_with(root / "filters", njm_filters(t, root / "filters", {
        "atmospheres/": '<a href="lsst_std/">lsst_std/</a>', "atmospheres/lsst_std/manifest.json": json.dumps(SERVED)}))
    with t.caplog.at_level(logging.WARNING):
        library.atmospheres()
    entries = json.loads((site / "manifest.json").read_text())["entries"]
    assert {e["file"]: e["airmass"] for e in entries} == {"am1.00.dat": 1.0, "am1.20.dat": 1.3,
                                                          "skycalc_am1.50.dat": 1.5}
    assert "kept the installed description of am1.20.dat" in t.caplog.text


@row
def installed_atmospheres_are_listed_while_njm_is_down(t):
    root = t.tmp / "data"
    atmosphere_site(root, "lsst_std", STALE)
    library = library_with(root / "filters", SimpleNamespace(list_atmosphere_sites=raising(
        requests.ConnectionError("NJM is unreachable"))))
    with t.caplog.at_level(logging.WARNING):
        assert list(library.atmospheres()) == ["lsst_std"]
    assert "the installed ones are used" in t.caplog.text
    library.mirror = SimpleNamespace(list_atmosphere_sites=lambda: [])  # what an unavailable grabber lists
    assert list(library.atmospheres()) == ["lsst_std"]


# A partial set from SVO, NJM down: installed, warned about, a success
@row
def filters_info_and_the_picker_summary_say_the_set_is_incomplete(t):
    root = gaia_from_svo(t)
    Filters.fetch("GAIA", "GAIA", filter_root=root)
    t.capsys.readouterr()
    assert run_cli(t.mp, "filters-info", "GAIA/GAIA", "--base", str(root), "--offline") == 0
    out = " ".join(t.capsys.readouterr().out.split())  # wrapped at 80 columns
    assert ("Incomplete : 4 of 6 bands installed; missing Gbp_bright, Gbp_faint; fetch it again when NJM is "
            "reachable to complete it") in out

    info = Filters.available("GAIA", "GAIA", filter_root=root, include_remote=False)
    lines = cli._pick_summary("GAIA/GAIA", {}, info, {"info": None, "properties": None, "errors": []}, "SVO")
    assert "  Incomplete : 4 of 6 bands installed; missing Gbp_bright, Gbp_faint" in lines
    assert all(len(line) <= 80 for line in lines)


@row
def a_complete_set_has_no_incomplete_line(t):
    root = t.tmp / "filters"
    write_set(root / "Fac" / "Cam", ["B", "V"])
    # An older manifest listing NJM's archived copy among its bands, without the copy installed
    write_set(root / "Fac" / "Old", ["H"], manifest_bands=["H", "H_svo"], archived={"H_svo": {}})
    for target in ("Fac/Cam", "Fac/Old"):
        assert run_cli(t.mp, "filters-info", target, "--base", str(root), "--offline") == 0
        assert "Incomplete" not in t.capsys.readouterr().out
    assert FilterLibrary.published_complete(root / "Fac" / "Old")  # complete for the library too


@row
def an_svo_curve_that_could_not_be_saved_still_fails(t):
    root, original = gaia_from_svo(t), SVOFilterBrowser._save_filter

    def refuse(self, filter_row, out_path):
        if filter_row.band == "Grvs":
            raise ValueError("SVO returned no transmission data")
        original(self, filter_row, out_path)

    t.mp.setattr(SVOFilterBrowser, "_save_filter", refuse)
    with pytest.raises(ComponentError) as raised:
        Filters.fetch("GAIA", "GAIA", filter_root=root)
    message = str(raised.value)
    assert "GAIA/GAIA is only partly installed" in message
    assert "3 of NJM's 6 bands came from SVO's GAIA/GAIA3" in message
    assert "GAIA/GAIA3.Grvs: SVO returned no transmission data" in message


# Messages and folders
@row
def filters_info_describes_curves_installed_under_other_band_names(t):
    # 0.4.1's PLANCK/HFI from SVO (100GHz ...) is what NJM keeps as 100_svo ...: its records describe it.
    info = {"bands": {"100": {"Description": "HFI 100", "curve": {"FilterID": "Planck/HFI.100"}},
                      "100_svo": {"Description": "PLANCK HFI 100GH", "curve": {"FilterID": "PLANCK/HFI.100GHz"}},
                      "857": {"Description": "HFI 857", "curve": {"FilterID": "Planck/HFI.857"}},
                      "857_svo": {"Description": "PLANCK HFI 857GHz", "curve": {"FilterID": "PLANCK/HFI.857GHz"}}}}
    properties = {"bands": {"100": {"WavelengthPivot": 30013450.0}, "100_svo": {"WavelengthPivot": 30035700.0},
                            "857": {"WavelengthPivot": 3525232.0}, "857_svo": {"WavelengthPivot": 3527564.0}}}
    catalogue = hfi_catalogue(t, info, properties, filter_ids=True)
    described = catalogue.info("PLANCK", "HFI")
    assert {band: record["Description"] for band, record in described["bands"].items()} == {
        "100GHz": "PLANCK HFI 100GH", "857GHz": "PLANCK HFI 857GHz"}
    assert sorted(described["archived_bands"]) == ["100", "857"]  # NJM's own curves, not installed
    pivots = catalogue.properties("PLANCK", "HFI")["bands"]
    assert {band: record["WavelengthPivot"] for band, record in pivots.items()} == {
        "100GHz": 30035700.0, "857GHz": 3527564.0}


@row
def a_set_njm_completes_gets_the_manifest_of_a_fresh_download(t):
    curve = CURVE.replace("original\n", "original;Components=filter+instrument;ComponentsSource=original\n")
    root, files = gaia_partly_from_svo(t, curve)
    library_with(root, njm_filters(t, root, files)).published("GAIA", "GAIA")
    fresh = t.tmp / "fresh"
    library_with(fresh, njm_filters(t, fresh, files)).published("GAIA", "GAIA")
    bands, completed = GAIA["GAIA"]["bands"], json.loads((root / "GAIA" / "GAIA" / "manifest.json").read_text())
    assert sorted(completed["detector_types"]) == sorted(completed["published_bands"]) == bands
    assert completed == json.loads((fresh / "GAIA" / "GAIA" / "manifest.json").read_text())


@row
def a_curve_without_a_column_row_is_reported_once_by_its_caller(t):
    bare = t.tmp / "bare.dat"
    bare.write_text("4000,0.1\n5000,0.9\n")
    with t.caplog.at_level(logging.WARNING):
        assert write_filter_header(bare, {"DetectorType": "photon"}) is False  # the caller reports it
    assert t.caplog.text == ""
    legacy = t.tmp / "legacy.dat"
    legacy.write_text("# DetectorType: photon\n4000,0.1\n5000,0.9\n")  # tagged by SED_Tools 0.3.0
    with t.caplog.at_level(logging.WARNING):
        assert migrate_filter_file(legacy) is True
    assert t.caplog.text.count("has no column-name row to hold SED_Tools metadata") == 1
    assert legacy.read_text() == "4000,0.1\n5000,0.9\n"


COMPONENTS_WITHOUT_A_BUILDER = {
    "components_njm_says_it_does_not_have_are_not_announced_without_a_builder_none_served": (None, False),
    "components_njm_says_it_does_not_have_are_not_announced_without_a_builder_download_failed": (
        RuntimeError("2 component files failed"), True),
}


def _components_without_a_builder(t, njm_answer, warned):
    directory = flat_published(t.tmp / "NoBuilderFac" / "Cam")
    download = raising(njm_answer) if njm_answer else lambda facility, instrument: 0  # NJM serves none for it
    library = library_with(t.tmp, SimpleNamespace(is_available=lambda: True, download_components=download))
    with t.caplog.at_level(logging.WARNING):
        assert library.prepare_components(directory, components="filter+qe") is False
    assert ("NJM cannot provide the components of NoBuilderFac/Cam (2 component files failed)"
            in t.caplog.text) == warned
    assert bool(t.caplog.text) == warned


@row
def an_unreadable_verification_cache_means_every_curve_is_checked(t):
    grabber = NJMFilterGrabber.__new__(NJMFilterGrabber)
    damaged = t.tmp / "verified_curves.json"
    damaged.write_text("{not json")
    assert grabber._saved_verified(damaged, "record") == {}
    locked = t.tmp / "locked.json"
    locked.write_text('{"njm_record": "record", "files": {"g.dat": 10}}')
    locked.chmod(0)
    try:
        assert grabber._saved_verified(locked, "record") == {}  # not NJM's fault: never a fallback to SVO
    finally:
        locked.chmod(0o600)
    damaged.write_text('{"njm_record": "old", "files": {"g.dat": 10}}')
    assert grabber._saved_verified(damaged, "record") == {}  # NJM's set changed since
    assert grabber._saved_verified(damaged, "old") == {"g.dat": 10}


@row
def a_manifest_answer_without_a_body_is_unreadable_not_absent(t):
    grabber = NJMFilterGrabber.__new__(NJMFilterGrabber)
    grabber.base_url, grabber.filters_url, grabber._available = NJM, f"{NJM}/filters", True
    grabber.session = SimpleNamespace(get=lambda url, **kwargs: Resp(b"", 204), headers={}, verify=True)
    with pytest.raises(ValueError, match="unreadable manifest.json for Fac/Cam"):
        grabber.fetch_manifest("Fac", "Cam")


OUTAGE_ANNOUNCED_ONCE = {"an_njm_outage_is_not_announced_again_for_each_fetch_njm_down": False,
                         "an_njm_outage_is_not_announced_again_for_each_fetch_njm_up": True}


def _outage_announced_once(t, up):
    root = t.tmp / "filters"
    library = library_with(root, SimpleNamespace(is_available=lambda: up,
                                                 _download_set=lambda f, i: SetDownload(listed=[])),
                           svo_browser(root, {("Fac", "Cam"): ["g", "r"]}))
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.filter_library"):
        assert library.published("Fac", "Cam") == root / "Fac" / "Cam"
    # Down: the NJM client said so once for the process.  Up but without the set: this fetch says so.
    assert ("NJM cannot provide Fac/Cam (it does not serve this instrument); trying SVO" in t.caplog.text) == up
    assert ("NJM cannot provide" in t.caplog.text) == up


@row
def a_set_whose_njm_index_cannot_be_saved_is_never_returned(t):
    root = t.tmp / "filters"
    library = library_with(root, njm_filters(t, root, {
        "filters/Fac/Cam/Cam": 500, "filters/Fac/Cam/": '<a href="g.dat">g.dat</a>', "filters/Fac/Cam/g.dat": CURVE}))
    with pytest.raises(ComponentError, match=r"NJM's index of Fac/Cam could not be saved in .*Fac/Cam") as raised:
        library.published("Fac", "Cam")
    assert "run again" in str(raised.value)
    assert (root / "Fac" / "Cam" / "g.dat").is_file() and not (root / "Fac" / "Cam" / "Cam").exists()


@row
def the_catalogue_finds_an_installed_copy_as_fetch_does(t):
    put(t.tmp / "filters" / "Gemini" / "GMOS-N", {  # NJM spells it GMOS_N
        "g.dat": CURVE, "info.json": json.dumps({"description": "as installed", "bands": {"g": {}}})})
    catalogue = FilterCatalogue(t.tmp / "filters", offline=True)
    assert catalogue.info("Gemini", "GMOS_N")["description"] == "as installed"
    assert catalogue.origin("Gemini", "GMOS_N", "info")["source"] == "installed"


# ======================================================================
# Which source supplies a set: picker entries, spelling, completeness, NJM optional (test_filters_sources)
# ======================================================================
SVO_CURVE = "Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=svo\n4100,0\n5000,1\n5900,0\n"
GAIA_SVO = {"GAIA0": "Gaia pre-launch filters",
            "GAIA2": "Original passbands used for the magnitudes published in Gaia DR2.",
            "Gaia2m": "Revised passbands for Gaia DR2.", "GAIA2r": "Revised passbands for Gaia DR2.",
            "GAIA3": "GAIA eDR3 release."}
GAIA_NJM = {"DR2": {"description": "Gaia DR2 passbands (Evans et al. 2018)", "svo_instruments": ["GAIA/GAIA2"]},
            "DR2r": {"description": "Gaia DR2 revised passbands", "svo_instruments": ["GAIA/GAIA2r"]},
            "GAIA": {"description": "Gaia EDR3 passbands (Riello et al. 2021)", "svo_instruments": ["GAIA/GAIA3"]},
            "GAIA0": {"svo_instruments": ["GAIA/GAIA0"]}}


def fac_cam_svo(download):
    """SVO listing Fac/Cam with bands g and r; ``download(rows)`` saves them."""
    return SimpleNamespace(list_facilities=lambda: [Facility("Fac", "Fac")],
                           list_instruments=lambda key: [Instrument("Fac", "Cam", "Cam")],
                           list_filters=lambda f, i: [FilterRow(f"Fac/Cam.{b}", b, "Fac", "Cam", "") for b in "gr"],
                           download_filters=download)


@row
def gaia_picker_shows_each_product_once_with_its_source_and_description(t):
    bundled(t)
    root = t.tmp / "filters"
    library = library_with(root, njm_filters(t, root, {"index.json": njm_index({"GAIA": GAIA_NJM})}),
                           svo_stub({"GAIA": "Gaia mission."}, {"GAIA": GAIA_SVO}))
    result = library.instruments("GAIA")
    assert labels(result, "description") == [
        ("DR2", ["original", "njm"], "Gaia DR2 passbands (Evans et al. 2018)"),
        ("DR2r", ["original", "njm"], "Gaia DR2 revised passbands"),
        ("GAIA", ["njm"], "Gaia EDR3 passbands (Riello et al. 2021)"),
        ("GAIA0", ["njm"], "Gaia pre-launch filters"),  # NJM gives none: SVO's text for the same product
        ("Gaia2m", ["svo"], "Revised passbands for Gaia DR2."),
    ]
    assert all(e["fetch_key"] == e["key"] == e["label"] for e in result["entries"])
    assert result["errors"] == []
    assert library.mirror.session.calls == ["index.json"]  # instruments come from index.json, not listings


@row
def svo_and_njm_down_leave_installed_and_original_sources(t):
    bundled(t, {"GAIA": GAIA_NJM, "CTIO": {"S_PLUS": {}}})
    root = t.tmp / "filters"
    put(root / "CTIO" / "S_PLUS", {"g.dat": CURVE})
    library = library_with(root, NJM_DOWN, svo_stub({}, {}, down=True))
    result = library.facilities()
    entries = {e["label"]: e for e in result["entries"]}
    # CTIO's only builder (DECam) makes components, not curves, so it is no source of CTIO's sets.
    assert entries["CTIO"]["sources"] == ["local"] and entries["GAIA"]["sources"] == ["original"]
    assert result["errors"] == ["SVO: Failed to fetch SVO index: connection refused", "NJM unavailable"]
    assert labels(library.instruments("CTIO"), "description") == [("S_PLUS", ["local"], "")]
    assert labels(library.instruments("GAIA"), "description")[0] == (
        "DR2", ["original"], "Gaia DR2 passbands (Evans et al. 2018)")


# Spelling
@row
def a_same_named_svo_product_njm_does_not_claim_keeps_its_own_name_and_text(t):
    bundled(t, {"Gemini": {"NIRI": {"description": "NJM's NIRI filter curves", "svo_instruments": []},
                           "GMOS_N": {"description": "NJM's GMOS-N", "svo_instruments": ["Gemini/GMOS-N"]}}})
    library = library_with(t.tmp / "filters", NJM_DOWN, svo_stub(
        {"Gemini": ""}, {"Gemini": {"Niri": "SVO's NIRI", "GMOS-N": "SVO's GMOS-N"}}))
    shown = {label: rest for label, *rest in labels(library.instruments("Gemini"), "description")}
    assert shown["GMOS_N"] == [["svo"], "NJM's GMOS-N"]  # NJM's product: NJM's name and text
    assert shown["Niri"] == [["svo"], "SVO's NIRI"] and "NIRI" not in shown  # another product: SVO's own


# NJM fails fast
@row
def svo_may_complete_a_set_njm_wrote_nothing_into(t):
    bundled(t)
    root = t.tmp / "filters"
    directory = write_set(root / "Fac" / "Cam", ["g"], manifest_bands=["g", "r"], text=SVO_CURVE)  # SVO, partial
    down = requests.ConnectTimeout("connect timed out")

    def download(rows):
        (directory / "r.dat").write_text(SVO_CURVE)

    library = library_with(root, njm_filters(t, root, {"filters/Fac/Cam/Cam": "g.dat\nr.dat\n",
                                                       "filters/Fac/Cam/g.dat": down, "filters/Fac/Cam/r.dat": down}),
                           fac_cam_svo(download))
    assert library.published("Fac", "Cam") == directory
    assert (directory / "g.dat").read_text() == (directory / "r.dat").read_text() == SVO_CURVE


# Compose and availability
@row
def njm_instrument_without_manifest_is_described_not_an_error(t):
    bundled(t)
    info = {"bands": {"G": {"curve": {"Components": "filter+qe+instrument", "ComponentsSource": "svo",
                                      "DetectorType": "photon", "DetectorTypeSource": "svo-curve-match"}},
                      "Grvs": {"curve": {"Components": "filter+qe+instrument", "DetectorType": "photon"}}}}
    root = t.tmp / "filters"
    library = library_with(root, njm_filters(t, root, {"filters/GAIA/GAIA/GAIA": "G.dat\nGrvs.dat\n",
                                                       "filters/GAIA/GAIA/info.json": json.dumps(info)}),
                           SimpleNamespace(list_facilities=lambda: pytest.fail("SVO was asked")))
    result = library.available("GAIA", "GAIA")
    assert result["availability_source"] == "njm" and result["errors"] == []
    assert result["bands"] == ["G", "Grvs"] and result["detector_type"] == "photon"
    assert result["notes"][0].startswith("no component manifest")


# ======================================================================
# The library and NJM's mirror: outages checked once per process, downloads of only what changed, old
# spellings, NJM's archived copies, compose, and where the catalogue comes from
# (test_filters_spelling_and_archives)
# ======================================================================
class CountingSession:
    """A requests session whose HEAD answers ``head`` (a status, or an exception raised) and whose GETs fail;
    ``calls`` lists (method, url, verify)."""

    def __init__(self, head):
        self.head_answer, self.calls, self.headers, self.verify = head, [], {}, True

    def head(self, url, **kwargs):
        self.calls.append(("HEAD", url, kwargs.get("verify", self.verify)))
        if isinstance(self.head_answer, Exception):
            raise self.head_answer
        return SimpleNamespace(status_code=self.head_answer)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs.get("verify", self.verify)))
        raise requests.ConnectionError("not served in this test")


def grabbers_on_one_session(t, head, n=3):
    """``n`` grabbers made one after another, all on one CountingSession; returns (grabbers, session)."""
    session = CountingSession(head)
    t.mp.setattr(njm_filter_grabber.requests, "Session", lambda: session)
    return [NJMFilterGrabber(base_dir=str(t.tmp)) for _ in range(n)], session


def fac_cam_set():
    """What NJM serves for Fac/Cam: bands g and r."""
    index = {"generated": "2026-10-28", "filters": {"facilities": ["Fac"], "facilities_detailed": {
        "Fac": {"instruments": {"Cam": {"filters": ["g", "r"], "size_bytes": 150}}}}}}
    return {"index.json": json.dumps(index), "filters/Fac/Cam/Cam": "g.dat\nr.dat\n",
            "filters/Fac/Cam/g.dat": CURVE, "filters/Fac/Cam/r.dat": CURVE}


def curves_requested(session):
    return [rel.rsplit("/", 1)[1] for rel in session.calls if rel.endswith(".dat")]


# NJM's availability and pause are shared by the whole process
@row
def a_silent_outage_is_checked_and_reported_once_per_process(t):
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.njm_filter_grabber"):
        grabbers, session = grabbers_on_one_session(t, requests.ConnectTimeout("timed out"))
    assert [g.is_available() for g in grabbers] == [False, False, False]
    assert session.calls == [("HEAD", "https://data.nialljmiller.com/sed_tools", True)]  # one 5 s wait, not three
    assert t.caplog.text.count("is unavailable (timed out)") == 1
    with pytest.raises(requests.ConnectionError, match="not asked again for 5 minutes"):
        grabbers[2]._get("https://data.nialljmiller.com/sed_tools/index.json")
    assert len(session.calls) == 1


@row
def the_outage_is_checked_again_once_the_pause_expires(t):
    clock = [1000.0]
    t.mp.setattr("sed_tools.njm_mirror.monotonic", lambda: clock[0])
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.njm_filter_grabber"):
        _, session = grabbers_on_one_session(t, requests.ConnectionError("refused"), n=1)
        clock[0] += njm_filter_grabber.NJM_PAUSE + 1
        again = NJMFilterGrabber(base_dir=str(t.tmp))
    assert not again.is_available() and len(session.calls) == 2
    assert t.caplog.text.count("is unavailable (refused)") == 1  # still the same outage


@row
def a_reachable_mirror_is_checked_once_and_its_ssl_fallback_is_kept(t):
    calls, sessions = [], []

    def head(url, verify=True, **kwargs):
        calls.append(("HEAD", verify))
        if verify:
            raise requests.exceptions.SSLError("certificate expired")
        return SimpleNamespace(status_code=200)

    def new_session():
        sessions.append(SimpleNamespace(headers={}, verify=True, head=head))
        return sessions[-1]

    t.mp.setattr(njm_filter_grabber.requests, "Session", new_session)
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.njm_filter_grabber"):
        first = NJMFilterGrabber(base_dir=str(t.tmp))
        second = NJMFilterGrabber(base_dir=str(t.tmp))
    assert first.is_available() and second.is_available()
    assert calls == [("HEAD", True), ("HEAD", False)]  # the second grabber asked nothing
    assert second.session.verify is False  # and skips the certificate check as the first does
    assert t.caplog.text.count("could not be verified") == 1


@row
def an_error_answer_to_one_catalogue_is_not_an_outage_for_the_next(t):
    # Only the NJM client pauses NJM (no answer, or a block that does not lift); an error answer does not.
    asked = []

    def get(url, **kwargs):
        asked.append(url)
        raise requests.HTTPError("503", response=SimpleNamespace(status_code=503))

    for _ in range(2):
        with pytest.raises(CatalogueError, match="Cannot fetch index.json"):
            FilterCatalogue(t.tmp / "filters", mirror=mirror(get)).index()
    assert len(asked) == 2


# Installed curves are checked against NJM only when there is a reason
@row
def a_retry_after_a_partial_download_fetches_only_what_is_missing(t):
    partial = fac_cam_set()
    partial["filters/Fac/Cam/r.dat"] = 404
    partial["filters/Fac/Cam/Cam"] = "g.dat\n#g_old.dat\nr.dat\n"  # a '#' line of NJM's index is not a band
    first = njm_filters(t, t.tmp / "filters", partial)
    assert first._download_set("Fac", "Cam").curve_failures == ["r.dat: 404 for filters/Fac/Cam/r.dat"]
    assert InstrumentManifest.load(f"{first.base_dir}/Fac/Cam").bands == ["g", "r"]
    retry = njm_filters(t, t.tmp / "filters", fac_cam_set())
    assert retry._download_set("Fac", "Cam").downloaded == 1
    assert curves_requested(retry.session) == ["r.dat"]


# Every old spelling still works as input to the API and the CLI
#: row name: (spelling given, NJM's spelling): SVO's, 0.4.1's and NJM's own.
LEGACY = {
    "planck_hfi_in_capitals": (("PLANCK", "HFI"), ("Planck", "HFI")),
    "planck_hfi": (("Planck", "HFI"), ("Planck", "HFI")),
    "gemini_gmos_n_with_a_hyphen": (("Gemini", "GMOS-N"), ("Gemini", "GMOS_N")),
    "gemini_gmos_n": (("Gemini", "GMOS_N"), ("Gemini", "GMOS_N")),
    "herschel_pacs": (("Herschel", "Pacs"), ("Herschel", "PACS")),
    "ctio_s_plus_with_a_hyphen": (("CTIO", "S-PLUS"), ("CTIO", "S_PLUS")),
    "bepi_colombo_hric_with_a_hyphen": (("Bepi-Colombo", "HRIC"), ("Bepi_Colombo", "HRIC")),
}
FILTER_CURVE = ("Wavelength,Transmission,FilterID=X.b;DetectorType=photon;DetectorTypeSource=original;"
                "WavelengthUnit=Angstrom;Components=filter;ComponentsSource=original;Sources=https://example.invalid\n"
                "4000,0\n5000,1\n6000,0\n")


def legacy_sources(t):
    """NJM serves every LEGACY instrument under its own spelling, with band b; SVO is down; nothing is built from
    source. Returns the filter root and NJM's session."""
    detailed, files = {}, {}
    for _, (fac, inst) in LEGACY.values():
        detailed.setdefault(fac, {"instruments": {}})["instruments"][inst] = {
            "filters": ["b"], "size_bytes": 200, "info": "info.json", "properties": "properties.json",
            "description": f"{fac} {inst} passbands", "svo_instruments": []}
        files[f"filters/{fac}/{inst}/{inst}"] = "b.dat\n"
        files[f"filters/{fac}/{inst}/b.dat"] = FILTER_CURVE
        files[f"filters/{fac}/{inst}/info.json"] = json.dumps(
            {"description": f"{fac} {inst} passbands", "bands": {"b": {"Description": "band b"}}})
        files[f"filters/{fac}/{inst}/properties.json"] = json.dumps({"bands": {"b": {"WavelengthPivot": 5000.0}}})
    files["index.json"] = json.dumps({"generated": "2026-10-28", "filters": {
        "facilities": sorted(detailed), "facilities_detailed": detailed}})
    served, svo_down = Session(files, base=NJM), raising(RuntimeError("SVO is down (test)"))
    njm_up(t)
    t.mp.setattr(njm_filter_grabber.requests, "Session", lambda: served)
    t.mp.setattr(filter_library, "SVOFilterBrowser", lambda **kwargs: SimpleNamespace(
        list_facilities=svo_down, list_instruments=svo_down, list_filters=svo_down, download_filters=svo_down))
    t.mp.setattr(filter_library, "get_builder", lambda name: pytest.fail(f"builder {name} was run"))
    t.mp.setattr(filter_library, "load_instrument_catalogue", lambda: instrument_catalogue(detailed))
    t.mp.setattr(Filters, "_catalogues", {})
    t.mp.setenv("COLUMNS", "100")
    return t.tmp / "filters", served


def njm_curve_requests(served, njm):
    return [rel for rel in served.calls if rel.endswith("/b.dat")], f"filters/{njm[0]}/{njm[1]}/b.dat"


@row
def the_shipped_catalogue_names_each_instrument_as_njm_does(t):
    shipped = load_instrument_catalogue()
    for _, (fac, inst) in LEGACY.values():
        assert inst in shipped[fac]["instruments"], f"{fac}/{inst}"


FETCH_INFO_AND_COMPOSE = {f"fetch_info_and_compose_take_the_old_spelling_{name}": spellings
                          for name, spellings in LEGACY.items()}


def _fetch_info_and_compose(t, given, njm):
    root, served = legacy_sources(t)
    directory = Filters.fetch(*given, filter_root=root)
    # macOS folds case, so PLANCK/HFI and Planck/HFI may be one directory: compare names, not paths.
    assert (name_key(directory.parent.name), name_key(directory.name)) == (name_key(njm[0]), name_key(njm[1]))
    assert FilterLibrary.published_complete(directory)
    curves, njm_curve = njm_curve_requests(served, njm)
    assert curves == [njm_curve]  # downloaded once, under NJM's spelling

    assert Filters.info(*given, filter_root=root)["bands"]["b"]["Description"] == "band b"
    product = Filters.compose("/".join(given), "filter", filter_root=root)
    assert (product / "b.dat").is_file()
    assert njm_curve_requests(served, njm)[0] == [njm_curve]  # nothing downloaded again


CLI_OLD_SPELLING = {f"the_cli_takes_the_old_spelling_{name}": spellings for name, spellings in LEGACY.items()}


def _cli_old_spelling(t, given, njm):
    root, served = legacy_sources(t)
    target, base = "/".join(given), str(root)
    for argv in (["filters-info", target], ["filters", target], ["filters-info", target],
                 ["filters-compose", target]):
        assert run_cli(t.mp, *argv, "--base", base) == 0, (argv, t.capsys.readouterr())
    out = t.capsys.readouterr().out
    assert "products/filter" in out
    assert njm_curve_requests(served, njm)[0] == [f"filters/{njm[0]}/{njm[1]}/b.dat"]


@row
def a_set_installed_under_an_old_spelling_is_used_as_it_is(t):
    # 0.4.1 saved SVO's Gemini/GMOS-N; NJM's spelling is GMOS_N. The installed set is used, offline.
    root, served = legacy_sources(t)
    old = put(root / "Gemini" / "GMOS-N", {"b.dat": FILTER_CURVE, "GMOS-N": "b.dat\n"})
    filter_library.InstrumentManifest.from_published(old).save(old)
    assert Filters.fetch("Gemini", "GMOS-N", filter_root=root) == old
    assert (Filters.compose("Gemini/GMOS-N", "filter", filter_root=root) / "b.dat").is_file()
    assert served.calls == []


@row
def query_keeps_svo_spelling_beside_the_download_key(t):
    root, _ = legacy_sources(t)
    t.mp.setattr(filter_library, "SVOFilterBrowser", lambda **kwargs: SimpleNamespace(
        list_facilities=lambda: [Facility("PLANCK", "PLANCK", "Planck mission")]))
    t.mp.setattr(Filters, "_filter_root", root)
    for asked in ("PLANCK", "planck", "Planck"):
        rows = [entry for entry in Filters.query(asked, include_local=False) if entry["source"] == "svo"]
        assert [(r["facility_key"], r["svo_facility_key"]) for r in rows] == [("Planck", "PLANCK")], asked


# Archive copies stay in archived_bands
ARCHIVED = {"H_svo": {"detector_type": "photon", "detector_type_source": "svo", "published": "filter"}}


def nisp_like(root):
    """A set as NJM serves Euclid/NISP: H and Y, plus SVO's H_svo kept under archived_bands."""
    return put(root / "Euclid" / "NISP", {
        **{f"{band}.dat": FILTER_CURVE for band in ("H", "Y", "H_svo")}, "NISP": "H.dat\nY.dat\n",
        "manifest.json": json.dumps({"facility": "Euclid", "instrument": "NISP", "bands": ["H", "Y"],
                                     "archived_bands": ARCHIVED})})


@row
def a_manifest_keeps_archived_bands_when_saved(t):
    inst = nisp_like(t.tmp)
    manifest = InstrumentManifest.load(inst)  # what jwst.py and roman.py do: load, change, save
    assert manifest.archived_bands == ARCHIVED and manifest.bands == ["H", "Y"]
    manifest.notes = "rebuilt"
    manifest.save(inst)
    assert json.loads((inst / "manifest.json").read_text())["archived_bands"] == ARCHIVED
    InstrumentManifest("Fac", "Cam", ["g"]).save(t.tmp)
    assert "archived_bands" not in json.loads((t.tmp / "manifest.json").read_text())  # unchanged format


@row
def described_from_its_curves_a_set_leaves_archive_copies_out(t):
    inst = nisp_like(t.tmp)
    described = InstrumentManifest.from_published(inst)  # what euclid, hsc, gmos and _common do
    assert described.bands == ["H", "Y"] and described.archived_bands == ARCHIVED
    (inst / "NISP").unlink()
    assert InstrumentManifest.from_published(inst).bands == ["H", "Y"]  # no index file: the manifest decides


@row
def only_index_listed_curves_are_bands_and_the_index_never_lists_an_archive_copy(t):
    inst = nisp_like(t.tmp)
    (inst / "stray.dat").write_text(FILTER_CURVE)  # not in the MESA index: MESA never loads it
    assert InstrumentManifest.from_published(inst).bands == ["H", "Y"]
    write_filter_index(inst, "NISP")  # a builder rewriting the index (hsc, lsst, argus, _common)
    assert (inst / "NISP").read_text().split() == ["H.dat", "Y.dat", "stray.dat"]


@row
def dropping_bands_njm_does_not_list_keeps_archived_bands(t):
    inst = nisp_like(t.tmp)
    data = json.loads((inst / "manifest.json").read_text())
    data["bands"] = ["H", "J", "Y"]  # J: declared, but NJM's index does not list it and it is not installed
    (inst / "manifest.json").write_text(json.dumps(data))
    drop_unlisted_bands(inst, ["H.dat", "Y.dat"])
    saved = json.loads((inst / "manifest.json").read_text())
    assert saved["bands"] == ["H", "Y"] and saved["archived_bands"] == ARCHIVED
    assert FilterLibrary.published_complete(inst)


# A set's records agree: one release per set, a MESA index named after the folder, a manifest naming it
def curve_of(filter_id):
    return FILTER_CURVE.replace("FilterID=X.b", f"FilterID={filter_id}")


def gaia_as_served(root):
    """GAIA/GAIA as NJM served it to 11 Nov 2026: GAIA3's G and Grvs with Gaia2m's Gbp_bright, all indexed."""
    return put(root / "GAIA" / "GAIA", {"G.dat": curve_of("GAIA/GAIA3.G"), "Grvs.dat": curve_of("GAIA/GAIA3.Grvs"),
                                       "Gbp_bright.dat": curve_of("GAIA/Gaia2m.Gbp_bright"),
                                       "GAIA": "G.dat\nGbp_bright.dat\nGrvs.dat\n"})


#: Which source instruments each set is made from, as NJM's svo_instruments.json says.
GAIA_CLAIMS = {"GAIA/GAIA": ["GAIA/GAIA3"], "GAIA/Gaia2m": ["GAIA/Gaia2m"]}


@row
def set_problems_names_each_way_a_sets_records_drift(t):
    def a_set(name, files):
        return put(t.tmp / "Fac" / name, {"b.dat": FILTER_CURVE, **files})

    named = json.dumps({"facility": "Fac", "instrument": "Stale", "bands": ["b", "c"]})
    expected = {  # the old tarball's MIRI and NIRISS had no index, so MESA could not read them
        a_set("NoIndex", {}): "Fac/NoIndex has no MESA index file NoIndex: MESA opens the index named after the folder",
        a_set("Empty", {"Empty": ""}): "Fac/Empty holds b.dat, which its index does not list",
        a_set("Gone", {"Gone": "b.dat\nc.dat\n"}): "Fac/Gone/Gone lists c.dat, not in the folder",
        a_set("Stray", {"Stray": "b.dat\n", "c.dat": FILTER_CURVE}): "Stray holds c.dat, which its index does not list",
        a_set("Unnamed", {"Unnamed": "b.dat\n", "manifest.json": '{"bands": ["b"]}'}): "manifest.json names no set",
        a_set("Broken", {"Broken": "b.dat\n", "manifest.json": "{"}): "Broken/manifest.json cannot be read",
        a_set("Stale", {"Stale": "b.dat\n", "manifest.json": named}): (
            "Stale/manifest.json declares c, which its index does not list (an incomplete set, or a stale manifest)"),
    }
    write_set(t.tmp / "Fac" / "Good", ["b"])
    found = check_sets(t.tmp)
    assert sorted(found) == sorted(str(d) for d in expected)  # Fac/Good agrees with itself
    for directory, line in expected.items():
        assert len(found[str(directory)]) == 1 and line in found[str(directory)][0], (directory, found[str(directory)])
    assert set_problems(put(t.tmp / "Fac" / "NoCurves", {})) == []


@row
def one_set_never_holds_another_release_of_its_instrument(t):
    gaia = gaia_as_served(t.tmp)
    assert set_problems(gaia) == []  # its records agree: only what each curve is shows the mix
    assert set_problems(gaia, GAIA_CLAIMS) == [
        f"{gaia}/Gbp_bright.dat is GAIA/Gaia2m.Gbp_bright, a curve of GAIA/Gaia2m, not of this set"]
    # NJM moves Gbp_bright out: downloading the set again leaves the installed copy beside it, not in it
    InstrumentManifest.from_published(gaia).save(gaia)
    served = {f"filters/GAIA/GAIA/{name}": (gaia / name).read_text() for name in ("G.dat", "Grvs.dat")}
    grabber = njm_filters(t, t.tmp, {**served, "filters/GAIA/GAIA/GAIA": "G.dat\nGrvs.dat\n"})
    with t.caplog.at_level(logging.WARNING):
        grabber.download_filters("GAIA", "GAIA")
    assert (gaia / "GAIA").read_text().split() == ["G.dat", "Grvs.dat"]
    saved = json.loads((gaia / "manifest.json").read_text())
    assert saved["bands"] == ["G", "Grvs"] and saved["archived_bands"] == {"Gbp_bright": {}}
    assert "does not list Gbp_bright; Gbp_bright kept in the folder under archived_bands" in t.caplog.text
    assert InstrumentManifest.from_published(gaia).bands == ["G", "Grvs"] and set_problems(gaia) == []
    assert len(set_problems(gaia, GAIA_CLAIMS)) == 1  # still another release's curve in the folder
    (gaia / "Gbp_bright.dat").unlink()
    assert set_problems(gaia, GAIA_CLAIMS) == []
    mine = put(t.tmp / "GAIA" / "Mine", {"G.dat": curve_of("GAIA/GAIA3.G"), "Mine": "G.dat\n"})
    assert set_problems(mine, GAIA_CLAIMS) == []  # a user's own set: no claim says what it is made from
    drop_unlisted_bands(mine, [])  # no manifest: nothing declared, nothing changed
    assert sorted(p.name for p in mine.iterdir()) == ["G.dat", "Mine"]


@row
def svo_never_saves_two_instruments_curves_into_one_set(t):
    browser = svo_browser(t.tmp, {})

    def rows(release, bands, folder="GAIA"):  # SVO labels every Gaia release "GAIA": 0.2.0 filed them all there
        return [FilterRow(f"GAIA/{release}.{band}", band, "GAIA", folder, "") for band in bands]

    browser.download_filters(rows("GAIA3", ["G", "Grvs"]))
    gaia = t.tmp / "GAIA" / "GAIA"
    with pytest.raises(ComponentError, match=r"GAIA holds G\.dat, Grvs\.dat of GAIA/GAIA3\. Saving GAIA/Gaia2m there "
                                             r"would make one set of two instruments' curves"):
        browser.download_filters(rows("Gaia2m", ["G", "Gbp_bright"]))
    with pytest.raises(ComponentError, match="Other would receive the curves of 2 instruments"):
        browser.download_filters(rows("GAIA0", ["G"], "Other") + rows("GAIA2", ["G"], "Other"))
    assert browser.saved == ["GAIA/GAIA3.G", "GAIA/GAIA3.Grvs"]  # nothing of the others was fetched or written
    assert (gaia / "GAIA").read_text().split() == ["G.dat", "Grvs.dat"] and not (t.tmp / "GAIA" / "Other").exists()
    # Completing one of NJM's products from the SVO instruments it claims: its own bands, whatever their source
    (gaia / "Gbp_bright.dat").write_text(curve_of("GAIA/Gaia2m.Gbp_bright"))
    grp = {"Grp": FilterRow("GAIA/GAIA3.Grp", "Grp", "GAIA", "GAIA3", "")}
    with pytest.raises(ComponentError, match="holds Gbp_bright.dat of GAIA/Gaia2m"):
        browser.download_into(gaia, grp, expected=["G", "Grp", "Grvs"])
    browser.download_into(gaia, grp, expected=["G", "Gbp_bright", "Grp", "Grvs"])
    assert (gaia / "Grp.dat").is_file()


@row
def the_mesa_index_is_named_after_its_folder(t):
    cam = put(t.tmp / "F" / "Cam", {"b.dat": FILTER_CURVE})
    assert write_filter_index(cam) == cam / "Cam"  # where MESA opens it
    out = combine_filter_sets(t.tmp / "Mine", [cam], filter_root=t.tmp, instrument="Other")  # 0.4.1 wrote Mine/Other
    assert (out / "Mine").read_text().split() == ["b.dat"] and not (out / "Other").exists()


@row
def a_renamed_or_copied_set_is_not_described_by_the_manifest_it_carried(t):
    sdss = write_set(t.tmp / "SLOAN" / "SDSS", ["g", "r"])
    copy = t.tmp / "SLOAN" / "SDSS_E"
    shutil.copytree(sdss, copy)
    (copy / "SDSS").rename(copy / "SDSS_E")  # the MESA index renamed by hand, as MESA needs
    with pytest.raises(ComponentError, match=r"SDSS_E/manifest.json describes SLOAN/SDSS, not SLOAN/SDSS_E: the "
                                             r"folder was renamed or copied"):
        Filters.available("SLOAN", "SDSS_E", filter_root=t.tmp, include_remote=False)
    assert set_problems(copy) == [f"{copy}/manifest.json describes SLOAN/SDSS, not SLOAN/SDSS_E"]
    data = json.loads((copy / "manifest.json").read_text())
    (copy / "manifest.json").write_text(json.dumps(dict(data, instrument="SDSS_E")))  # what the error says to do
    assert set_problems(copy) == []
    assert Filters.available("SLOAN", "SDSS_E", filter_root=t.tmp, include_remote=False)["instrument"] == "SDSS_E"
    gmos = write_set(t.tmp / "Gemini" / "GMOS-N", ["g"])  # NJM's Gemini/GMOS_N installed under another spelling
    (gmos / "manifest.json").write_text(json.dumps(dict(json.loads((gmos / "manifest.json").read_text()),
                                                        instrument="GMOS_N")))
    assert set_problems(gmos) == [] and InstrumentManifest.load(gmos).instrument == "GMOS_N"


# Compose of the bands that have the components asks the sources for them
def nisp_with_declared_components(t):
    """Euclid/NISP as NJM's manifest declares it: filter, optics and qe for H and Y; grisms Blue and Red bare."""
    curve = FILTER_CURVE.replace("Components=filter", "Components=filter+optics+qe")
    inst = put(t.tmp / "filters" / "Euclid" / "NISP", {**{f"{band}.dat": curve for band in ("Blue", "H", "Red", "Y")},
                                                       "NISP": "Blue.dat\nH.dat\nRed.dat\nY.dat\n"})
    components = {role: {band: f"components/{role}/{band}.dat" for band in ("H", "Y")}
                  for role in ("filter", "optics", "qe")}  # declared; not downloaded yet
    InstrumentManifest("Euclid", "NISP", ["Blue", "H", "Red", "Y"], components=components).save(inst)
    return inst


@row
def compose_of_the_covered_bands_goes_on_to_the_sources(t):
    nisp_with_declared_components(t)
    library, asked = FilterLibrary(t.tmp / "filters"), []
    library.prepare_components = lambda directory, **kwargs: asked.append(kwargs["bands"]) or False
    with pytest.raises(ComponentError):  # nothing downloaded in this row, so the build itself fails
        library.compose("Euclid/NISP", "filter,optics,qe", bands=["H", "Y"])
    assert asked == [["H", "Y"]]


# available() says which source failures were only fallbacks
@row
def available_reports_failures_by_source_and_which_were_fallbacks(t):
    library = library_with(t.tmp / "filters", SimpleNamespace(is_available=lambda: True, fetch_manifest=raising(
        requests.ConnectionError("Connection refused"))), SimpleNamespace(metadata_client=SimpleNamespace(
            get_filter=lambda fid: None if fid.endswith(".r") else SimpleNamespace(components=None,
                                                                                   detector_type="photon"))))
    library._njm_spelling = lambda f, i: None
    library.svo_filters = lambda f, i: [FilterRow("Fac/Cam.g", "g", "Fac", "Cam", ""),
                                        FilterRow("Fac/Cam.r", "r", "Fac", "Cam", "")]
    info = library.available("Fac", "Cam")
    assert info["availability_source"] == "svo"
    assert info["errors"] == ["NJM: Connection refused", "SVO: no metadata for Fac/Cam.r"]  # as before
    assert info["sources_failed"] == {"njm": ["Connection refused"], "svo": ["no metadata for Fac/Cam.r"]}
    assert info["fallback_from"] == ["njm"]
    assert cli._availability_errors(info) == ["no metadata for Fac/Cam.r"]  # NJM's failure left nothing out


@row
def compose_leaves_variants_to_the_sources_that_register_them(t):
    # A set whose manifest has no variants yet (from SVO, or installed by 0.4.1): the builder or
    # NJM's manifest adds the variant while the components are prepared, so it is not refused first.
    inst = nisp_with_declared_components(t)
    library = FilterLibrary(t.tmp / "filters")

    def prepare(directory, **kwargs):
        manifest = InstrumentManifest.load(inst)
        manifest.variants = {"qe": {"cold": {"files": {"*": "components/qe/cold.dat"}}}}
        manifest.save(inst)
        raise ComponentError("stop here: the variant was registered")

    library.prepare_components = prepare
    with pytest.raises(ComponentError, match="the variant was registered"):
        library.compose("Euclid/NISP", "filter,optics,qe", bands=["H", "Y"], variants={"qe": "cold"})


@row
def no_index_json_is_fetched_for_the_record_while_njm_is_paused(t):
    made = njm_filters(t, t.tmp / "filters", fac_cam_set())
    njm_filter_grabber.NJMFilterGrabber._pause("test outage", "NJM is down (test)")
    with t.caplog.at_level(logging.WARNING):
        assert made._njm_record("Fac", "Cam") is None
    assert made.session.calls == [] and "Cannot read NJM's index.json" not in t.caplog.text


# filters-info before anything is installed and with no source answering: the detector is not known, never the
# energy default
BUNDLED_AVAILABILITY = {"bundled_availability_leaves_the_detector_unknown_with_a_builder": True,
                        "bundled_availability_leaves_the_detector_unknown_without_a_builder": False}


def _bundled_availability(t, builder):
    bundled(t, {"GAIA": {"DR2": DR2}})
    if not builder:
        t.mp.delitem(filter_library.BUILDERS_FOR, ("GAIA", "DR2"))
    info = library_with(t.tmp / "filters", NJM_DOWN, svo_stub({}, {}, down=True)).available("GAIA", "DR2")
    assert info["availability_source"] == ("original" if builder else "unknown")
    # GAIA/DR2's builder writes photon-counting curves; this used to say energy, "unverified"
    assert info["detector_type"] is None
    assert info["detector_type_sources"] == {"BP": "unknown", "G": "unknown", "RP": "unknown"}
    cli.print_filter_availability(info)
    out = t.capsys.readouterr().out
    assert "  Detector                 : not known until the curves are installed\n" in out
    assert "energy" not in out and "unverified" not in out


# filters-info: NJM's archived copies (*_svo, *_old) stay apart from the set, also beside an installed copy
# whose band names differ from NJM's
HFI_INFO = {"bands": {"100": {"Description": "HFI 100"}, "100_svo": {"Description": "PLANCK HFI 100GH"},
                      "857": {"Description": "HFI 857"}, "857_svo": {"Description": "PLANCK HFI 857GHz"}}}


ARCHIVED_COPIES_APART = {
    f"archived_copies_stay_apart_beside_an_installed_copy_under_other_band_names_as_{who}_spells_it": spelling
    for who, spelling in (("njm", ("Planck", "HFI")), ("svo", ("PLANCK", "HFI")))}


def _archived_copies_apart(t, *spelling):
    catalogue = hfi_catalogue(t, HFI_INFO, HFI_INFO, filter_ids=False)
    for kind in ("info", "properties"):
        data = getattr(catalogue, kind)(*spelling)
        assert list(data["bands"]) == ["100", "857"]  # the *_svo rows used to be listed with them
        assert list(data["archived_bands"]) == ["100_svo", "857_svo"]


# ======================================================================
# The SVO filter browser at a terminal, answered through input() (test_filters_svo_interactive)
# ======================================================================
def alpha_beta(t):
    """The browser run_interactive makes: SVO lists Alpha (cameras Cam, with bands g and r, and Dead, with none)
    and Beta (camera IR, with band J); ``saved`` lists the filter IDs whose curves it fetched."""
    made = svo_browser(t.tmp, {("Alpha", "Cam"): ["g", "r"], ("Alpha", "Dead"): [], ("Beta", "IR"): ["J"]})
    t.mp.setattr(svo, "SVOFilterBrowser", lambda base_dir: made)
    return made


@row
def one_instrument_is_downloaded_then_the_browser_is_left(t):
    browser = alpha_beta(t)
    # single-facility mode, Alpha, Cam, download, no other camera; then single-facility mode and quit
    answer(t.mp, "", "1", "1", "", "", "", "q")
    svo.run_interactive(str(t.tmp))
    assert browser.saved == ["Alpha/Cam.g", "Alpha/Cam.r"]
    assert (t.tmp / "Alpha" / "Cam" / "Cam").read_text().split() == ["g.dat", "r.dat"]
    out = t.capsys.readouterr().out
    assert "Instrument: Cam (Cam)" in out and "Filters  : 2 available" in out
    assert out.rstrip().endswith("Exiting.")


@row
def instruments_can_be_declined_skipped_when_empty_and_left(t):
    browser = alpha_beta(t)
    answer(t.mp, "", "1",
           "2",            # Dead has no filters
           "1", "n",       # Cam, but do not download
           "1", "y", "y",  # Cam, downloaded; another from Alpha
           "b",            # back to the facilities
           "", "2", "q")   # Beta, then quit at its instruments
    svo.run_interactive(str(t.tmp))
    assert browser.saved == ["Alpha/Cam.g", "Alpha/Cam.r"]
    out = t.capsys.readouterr().out
    assert "No filters found for Dead." in out
    assert "Instruments for Beta" in out and out.rstrip().endswith("Exiting.")


@row
def bulk_mode_downloads_every_instrument_of_the_facilities_chosen(t):
    browser = alpha_beta(t)
    answer(t.mp, "9",          # not a facility
           ",",                # none chosen
           "1,2", "n",         # chosen, then not confirmed
           "1-2", "", "y",     # downloaded; another set
           "2", "y", "")       # Beta again, then done
    svo.run_interactive(str(t.tmp))
    assert browser.saved == ["Alpha/Cam.g", "Alpha/Cam.r", "Beta/IR.J", "Beta/IR.J"]
    out = t.capsys.readouterr().out
    assert "Invalid selection: Selection out of bounds: 9" in out and "No facilities selected." in out
    assert "  [skip] Dead: no filters found" in out
    assert "Facility: Alpha (2 instruments)" in out and "Downloading 1 filters for IR" in out


@row
def with_nothing_listed_the_browser_says_so(t):
    browser = alpha_beta(t)
    browser.list_facilities = lambda: [Facility("Empty", "Empty")]
    browser.list_instruments = lambda facility: []
    answer(t.mp, "", "1", "", "q")
    svo.run_interactive(str(t.tmp))
    assert "No instruments found for Empty." in t.capsys.readouterr().out
    browser.list_facilities = lambda: []
    svo.run_interactive(str(t.tmp))
    assert t.capsys.readouterr().out == "No facilities discovered.\n"


@row
def main_browses_into_its_base_dir_and_exits_1_when_a_curve_fails(t):
    browser = alpha_beta(t)
    answer(t.mp, "", "q")
    assert svo.main([str(t.tmp)]) == 0
    browser.metadata_client = SimpleNamespace(get_filter=lambda filter_id: None)  # SVO serves no curve
    answer(t.mp, "", "2", "1", "")
    assert svo.main([str(t.tmp)]) == 1
    assert "1 of 1 SVO filters could not be saved" in t.capsys.readouterr().out


# ======================================================================
# SVO unreachable: one clear error, at once (test_svo_fail_fast)
# ======================================================================
@default_svo_row
def connect_timeout_raises_one_clear_error_and_is_not_retried(t):
    session = SVOSession(error=requests.ConnectTimeout("connect timed out"))
    client = fm.SVOMetadataClient(session=session)
    with pytest.raises(fm.SVOUnreachableError) as info:
        client.get_filter("Generic/Johnson.V")
    message = str(info.value)
    assert "SVO Filter Profile Service unreachable" in message
    assert fm.SVO_FPS_URL in message
    assert "NJM" in message and "original-source builders" in message and "SED_SVO_FPS_URL" in message
    assert session.calls[0][1]["timeout"] == (fm.SVO_CONNECT_TIMEOUT, 120.0)
    assert fm.SVO_CONNECT_TIMEOUT <= 10
    # Existing handlers for either exception family still catch it.
    assert isinstance(info.value, requests.RequestException) and isinstance(info.value, RuntimeError)

    with pytest.raises(fm.SVOUnreachableError):
        client.get_filter("Generic/Johnson.B")
    assert len(session.calls) == 1


@default_svo_row
def download_stops_at_first_unreachable_filter(t):
    session = SVOSession(error=requests.ConnectTimeout("connect timed out"))
    browser = SVOFilterBrowser(base_dir=str(t.tmp), session=session)
    rows = [FilterRow(f"Generic/Johnson.{b}", b, "Generic", "Johnson", "") for b in "UBV"]
    with pytest.raises(fm.SVOUnreachableError):
        browser.download_filters(rows)
    assert len(session.calls) == 1


# ======================================================================
# Curves stay readable by released MESA colors, 25.12.1 and 26.4.1 (test_mesa_legacy_format)
#
# Released MESA's ``load_filter`` discards the first line, sizes its arrays from every remaining line, then skips
# leading ``#`` lines plus the first other line before reading samples. Any extra line therefore leaves an
# uninitialised sample at the end of the curve. The column-name row is the one free line, so SED_Tools metadata
# lives there as a third, whitespace-free CSV field.
# ======================================================================
# A curve exactly as SED_Tools 0.3.0 and the NJM mirror published it.
TAGGED_030 = (
    "# FilterID: GAIA/GAIA3.G\n"
    "# DetectorType: photon\n"
    "# DetectorTypeSource: svo-curve-match\n"
    "# WavelengthUnit: Angstrom\n"
    "# Components: filter+qe+instrument\n"
    "# ComponentsSource: svo\n"
    "Wavelength,Transmission\n"
    "3200.0,2.37366962e-08\n"
    "6000.0,0.75\n"
    "10500.0,1.1e-05\n"
)
BODY_030 = TAGGED_030.split("Wavelength,Transmission\n")[1]
ROW_G = ("Wavelength,Transmission,FilterID=GAIA/GAIA3.G;DetectorType=photon;DetectorTypeSource=svo-curve-match;"
         "WavelengthUnit=Angstrom;Components=filter+qe+instrument;ComponentsSource=svo")


def released_mesa_load_filter(path):
    """Transcription of ``load_filter`` from MESA 26.4.1 ``colors_utils.f90``.

    Returns ``(allocated, samples)``: the array size MESA allocates and the samples it actually reads into it.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    allocated = len(lines) - 1  # line 1 discarded, every later line counted
    i = 0
    while i < len(lines) and lines[i][:1] == "#":
        i += 1
    i += 1  # the first non-'#' line is consumed as well
    samples = []
    for line in lines[i:]:
        try:
            wl, tr = (float(v) for v in line.replace(",", " ").split()[:2])
        except ValueError:
            break
        samples.append((wl, tr))
    return allocated, samples


def mesa_plot_unit_test_load(path):
    """Loop of ``load_filters`` in MESA 26.4.1 ``colors/test/python_helpers/plot_unit_test.py``."""
    wav = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                wav.append(float(parts[0]))
    return wav


def assert_legacy_readers(path):
    """Every known consumer of SED_Tools curves reads all samples of ``path``."""
    allocated, samples = released_mesa_load_filter(path)
    assert allocated == len(samples), f"{path}: MESA would allocate {allocated} for {len(samples)} samples"
    expected = np.array(samples).T
    mesa_plot_unit_test_load(path)  # whitespace-split: must not raise on the column row
    np.testing.assert_allclose(load_two_column_file(str(path))[:2], expected)  # SED_Tools <= 0.3.0
    np.testing.assert_allclose(np.loadtxt(path, delimiter=",", skiprows=1).T, expected)
    np.testing.assert_allclose([pd.read_csv(path)["Wavelength"], pd.read_csv(path)["Transmission"]], expected)
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    np.testing.assert_allclose([[float(r["Wavelength"]) for r in rows],
                                [float(r["Transmission"]) for r in rows]], expected)
    table = ascii.read(path, format="csv")
    np.testing.assert_allclose([table["Wavelength"], table["Transmission"]], expected)


@row
def emulation_detects_the_030_defect(t):
    path = t.tmp / "G.dat"
    path.write_text(TAGGED_030)
    allocated, samples = released_mesa_load_filter(path)
    assert (allocated, len(samples)) == (9, 3)


@row
def written_curve_carries_metadata_on_its_column_row(t):
    path = t.tmp / "G.dat"
    write_filter_file(path, [4000.0, 5000.0], [0.1, 0.9], detector_type=PHOTON, source=SOURCE_SVO,
                      filter_id="GAIA/GAIA3.G", wavelength_unit="Angstrom")
    assert path.read_text() == (
        "Wavelength,Transmission,FilterID=GAIA/GAIA3.G;DetectorType=photon;DetectorTypeSource=svo;"
        "WavelengthUnit=Angstrom\n4000.0,0.1\n5000.0,0.9\n")
    assert read_filter_header(path) == {"FilterID": "GAIA/GAIA3.G", "DetectorType": "photon",
                                        "DetectorTypeSource": "svo", "WavelengthUnit": "Angstrom"}
    assert_legacy_readers(path)


@row
def curve_without_metadata_is_byte_identical_to_020(t):
    legacy = b"Wavelength,Transmission\n4000.0,0.1\n5000.0,0.9\n"
    path = t.tmp / "V.dat"
    path.write_bytes(legacy)
    assert read_filter_header(path) == {}
    write_filter_header(path, {"DetectorType": "energy"})
    assert path.read_bytes() == b"Wavelength,Transmission,DetectorType=energy\n4000.0,0.1\n5000.0,0.9\n"
    write_filter_header(path, {})
    assert path.read_bytes() == legacy


@row
def values_are_escaped_onto_one_whitespace_free_field(t):
    path = t.tmp / "A.dat"
    fields = {"DetectorType": "energy", "Atmosphere": "site=paranal airmass=1.2 pwv=2.5",
              "Sources": 'ESO SkyCalc; Noll et al., 2012 "v2" 100% #3', "Recipe": "filter+qe @ Å\tx"}
    write_filter_file(path, [1.0, 2.0], [0.5, 0.5], detector_type="energy", source="component",
                      extra={k: v for k, v in fields.items() if k != "DetectorType"})
    row = path.read_text(encoding="utf-8").splitlines()[0]
    assert len(row.split()) == 1 and len(row.split(",")) == 3 and "#" not in row
    assert "Atmosphere=site=paranal%20airmass=1.2%20pwv=2.5" in row
    assert read_filter_header(path) == {**fields, "DetectorTypeSource": "component"}
    assert_legacy_readers(path)


@row
def migration_moves_fields_and_keeps_samples_byte_for_byte(t):
    path = t.tmp / "G.dat"
    path.write_text(TAGGED_030)
    before = read_filter_header(path)

    assert migrate_filter_file(path, dry_run=True)
    assert path.read_text() == TAGGED_030
    assert migrate_filter_file(path)

    assert path.read_text() == ROW_G + "\n" + BODY_030
    assert read_filter_header(path) == before
    assert not migrate_filter_file(path)  # idempotent
    assert_legacy_readers(path)


@row
def migration_keeps_foreign_lines_and_crlf(t):
    path = t.tmp / "X.dat"
    path.write_bytes(b"# DetectorType: energy\r\n# measured by hand\r\nWavelength,Transmission\r\n1,2\r\n")
    assert not released_mesa_reads(path, after_migration=True)
    migrate_filter_file(path)
    assert path.read_bytes() == b"# measured by hand\r\nWavelength,Transmission,DetectorType=energy\r\n1,2\r\n"
    # The foreign comment is the user's to remove; the migration must not call the curve fixed.
    assert not released_mesa_reads(path)
    assert "released MESA miscounts" in t.caplog.text


RELEASED_MESA_CHECK = {f"released_mesa_check_matches_the_released_reader_{name}": case for name, case in {
    "with_metadata": ("Wavelength,Transmission,DetectorType=photon\n1,2\n3,4\n", True),
    "without_a_final_newline": ("Wavelength,Transmission\n1,2\n3,4", True),
    "without_a_column_row": ("1,2\n3,4\n", False),  # MESA allocates correctly but skips the first sample
    "with_a_blank_line_for_a_column_row": ("\n1,2\n3,4\n", True),  # MESA skips it and reads every sample
    "with_a_comment_before_the_column_row": ("# c\nWavelength,Transmission\n1,2\n", False),
    "with_a_blank_line_before_the_column_row": ("\nWavelength,Transmission\n1,2\n", False),
    "with_a_blank_line_at_the_end": ("Wavelength,Transmission\n1,2\n\n", False),
    "with_a_blank_line_between_samples": ("Wavelength,Transmission\n1,2\n\n3,4\n", False),
}.items()}


def _released_mesa_check(t, text, ok):
    path = t.tmp / "C.dat"
    path.write_text(text)
    allocated, samples = released_mesa_load_filter(path)
    # Every sample row of the file, found independently of the reader.
    rows = [tuple(float(v) for v in line.split(",")) for line in text.splitlines() if line[:1].isdigit()]
    assert released_mesa_reads(path) is ok is (allocated == len(samples) and samples == rows)


@row
def released_mesa_check_after_migration_sees_the_lost_first_sample(t):
    # A 0.3.0-tagged curve that had no column row is restored to "1,2\n3,4\n",
    # whose first sample released MESA skips as the column row.
    path = t.tmp / "C.dat"
    path.write_text("# DetectorType: photon\n1,2\n3,4\n")
    assert not released_mesa_reads(path, after_migration=True)
    migrate_filter_file(path)
    assert path.read_text() == "1,2\n3,4\n"
    assert not released_mesa_reads(path)


@row
def svo_labels_hold_only_what_the_record_states(t):
    from sed_tools.filters.curve_metadata import svo_components_header, svo_detector

    # No DetectorType at SVO: the writer's default is an assumption, never SVO's value (the LSST and
    # SVO-published builders wrote it as "svo").
    assert svo_detector(svo_record("X/Y.b", None, [1, 2], [1, 1]), PHOTON) == (PHOTON, SOURCE_ASSUMED)
    assert svo_detector(svo_record("X/Y.b", ENERGY, [1, 2], [1, 1]), PHOTON) == (ENERGY, SOURCE_SVO)
    # SVO's LSST/LSST.g_filter, "LSST u filter (filter only)", lists the total band's components.
    total = "Filter + Instrument + Atmosphere"
    assert svo_components_header(total, "LSST/LSST.g_filter") == {}
    assert "LSST/LSST.g_filter is named as a filter-only curve" in t.caplog.text
    assert svo_components_header("Filter", "LSST/LSST.g_filter") == {"Components": FILTER, "ComponentsSource": "svo"}
    assert svo_components_header(total, "LSST/LSST.g") == {"Components": "filter+instrument+atmosphere",
                                                           "ComponentsSource": "svo"}
    assert svo_components_header("Filter + Lens", "X/Y.b") == {}


#: The Internet Archive ingest's Sources: the capture, a credit naming the record it archives, and a README.
INGESTED = ("https://web.archive.org/web/2025id_/http://svo/fps.php?ID=LSST/LSST.g_filter | SVO Filter Profile "
            "Service (Rodrigo et al. 2012, 2020; Rodrigo & Solano 2024), record LSST/LSST.g_filter recovered from "
            "the Internet Archive capture 2025 of http://svo/fps.php?ID=LSST/LSST.g_filter | https://lsst/README.md")
#: name: (Sources as found, as SED_Tools writes it)
SOURCES = {
    "sources_drop_a_credit_of_a_listed_reference": (
        INGESTED, "https://web.archive.org/web/2025id_/http://svo/fps.php?ID=LSST/LSST.g_filter | https://lsst/README.md"),
    "sources_split_references_run_together": (
        "product of https://a/m.txt (mirror) and https://a/c.txt (corrector), nm; https://a/w.txt",
        "product of https://a/m.txt (mirror) | https://a/c.txt (corrector), nm | https://a/w.txt"),
    "sources_keep_notes_and_a_url_in_parentheses": (
        "https://e/H.dat (T_FILTER; ESA V1.0, https://doi.org/10.5270/x) | Hewett et al. 2006 Table 1",
        "https://e/H.dat (T_FILTER; ESA V1.0, https://doi.org/10.5270/x) | Hewett et al. 2006 Table 1"),
    "sources_drop_empty_and_repeated_items": ("https://a |  | https://a | b", "https://a | b"),
}


def _sources(t, found, written):
    from sed_tools.filters.curve_metadata import normalize_sources

    assert normalize_sources(found) == written and normalize_sources(written) == written
    write_filter_file(t.tmp / "c.dat", [1.0, 2.0], [1.0, 1.0], detector_type=PHOTON, source="user",
                      extra={"Sources": found})
    assert read_filter_header(t.tmp / "c.dat")["Sources"] == written


@row
def label_check_finds_every_broken_rule_and_a_rebuild_undoing_a_correction(t):
    import runpy

    from sed_tools.filters import label_check

    def curve(path, detector=PHOTON, source=SOURCE_SVO, wl=(4000.0, 5000.0, 6000.0), tr=(0.5, 1.0, 0.5), **extra):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_filter_file(path, wl, tr, detector_type=detector, source=source, extra=extra or None)

    new, old = t.tmp / "new", t.tmp / "old"
    ins = new / "Fac" / "Ins"
    curve(ins / "g.dat", Components="filter+qe")
    curve(ins / "g_filter.dat", tr=(0.5, 0.9, 0.5), Components="filter+qe")   # rule 3, and 4: it is the filter
    curve(ins / "components/filter/g.dat", source="component", tr=(0.5, 0.9, 0.5), Components=FILTER)
    curve(ins / "components/qe/flat.dat", source="component", tr=(0.0, 0.0, 0.0))
    curve(ins / "r.dat", detector="both", wl=(4500.0, 5500.0), tr=(1.0, 1.0))   # rule 1; the filter's wings
    curve(ins / "i.dat", source="manual")                                         # rule 1: a word no one ranks
    curve(ins / "z.dat", source="original", Components="filter+lens")             # rule 2
    curve(ins / "u.dat", source="user")
    curve(ins / "y.dat")                                                          # SVO reclassified it
    curve(ins / "w.dat")                                                          # rule 6: the ingest's credit
    (ins / "w.dat").write_text((ins / "w.dat").read_text().replace("\n", f";Sources={quote(INGESTED)}\n", 1))
    (ins / "bad.dat").write_text("Wavelength,Transmission\n4000,-1\n5000,1\n")
    (ins / "manifest.json").write_text(json.dumps({"components": {
        FILTER: {"g": "components/filter/g.dat"}, QE: {"*": "components/qe/flat.dat"},
        "filter+qe": {"g": "g.dat"}}}))
    curve(new / "Fac" / "Plain" / "v.dat", Components=FILTER)   # its own filter component, as from_published says
    (new / "Fac" / "Plain" / "manifest.json").write_text(json.dumps({"components": {FILTER: {"v": "v.dat"}}}))
    clean = t.tmp / "clean"
    curve(clean / "Fac" / "Bare" / "v.dat")
    curve(old / "Fac" / "Ins" / "g.dat", ENERGY, "manual")                    # TESS: photon by hand, then SVO
    curve(old / "Fac" / "Ins" / "r.dat")
    curve(old / "Fac" / "Ins" / "i.dat")
    curve(old / "Fac" / "Ins" / "z.dat", ENERGY)
    curve(old / "Fac" / "Ins" / "u.dat", ENERGY, "original")
    curve(old / "Fac" / "Ins" / "y.dat", ENERGY)
    (old / "Fac" / "Ins" / "g_filter.dat").write_text("Wavelength,Transmission\n4000,1\n")

    found = {(path.name, message.split(" ")[0]) for path, message in label_check.problems(new, old)}
    assert found == {("r.dat", "DetectorType"), ("i.dat", "DetectorTypeSource"), ("z.dat", "Components"),
                     ("g_filter.dat", "named"), ("g_filter.dat", "is"), ("bad.dat", str(ins / "bad.dat") + ":"),
                     ("g.dat", "DetectorType"), ("y.dat", "DetectorType"), ("w.dat", "Sources")}, found
    messages = [m for _, m in label_check.problems(new, old)]
    assert "DetectorType energy (manual) became photon (svo), which does not outrank it" in messages
    assert "DetectorType energy (svo) became photon (svo), which does not outrank it" in messages
    assert "DetectorType photon (svo) became unlabelled (svo), which does not outrank it" in messages
    assert label_check.main([str(new)]) == 1 and "ERROR" in t.capsys.readouterr().err
    assert label_check.main([str(clean)]) == 0 and "keep every rule" in t.capsys.readouterr().out
    with pytest.raises(SystemExit):
        label_check.main([str(t.tmp / "missing")])
    t.mp.setattr(sys, "argv", ["label_check", str(clean)])
    with pytest.raises(SystemExit) as stop:
        runpy.run_module("sed_tools.filters.label_check", run_name="__main__")
    assert stop.value.code == 0


@row
def label_check_finds_the_same_curve_served_with_two_detector_types(t):
    from sed_tools.filters import label_check

    wl, tr = np.array([4000.0, 4500.0, 5000.0, 5500.0]), np.array([0.1, 0.8312345678, 0.5, 0.05])
    for rel, transmission, detector, source in (
            ("SLOAN/SDSS/g.dat", tr, PHOTON, SOURCE_SVO),                   # rule 7: these three are one curve
            ("Misc/UCAC/sdss_g.dat", tr.astype(np.float32), ENERGY, SOURCE_SVO),  # SDSS's values at float32
            ("CTIO/CTIO/g.dat", 100 * tr, ENERGY, SOURCE_SVO_MATCH),        # in percent
            ("LBT/MODS/g.dat", tr, PHOTON, SOURCE_SVO),                     # the same tag: no clash
            ("Misc/APASS/r.dat", tr[::-1], ENERGY, SOURCE_SVO)):            # another curve
        (t.tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        write_filter_file(t.tmp / rel, wl, [float(v) for v in transmission], detector_type=detector, source=source)
    component = t.tmp / "Misc" / "APASS" / "components" / "filter" / "g.dat"   # components keep their own labels
    component.parent.mkdir(parents=True)
    write_filter_file(component, wl, tr, detector_type=ENERGY, source="component")
    [(_, message)] = label_check.problems(t.tmp)
    assert message == ("the same curve carries more than one DetectorType: "
                       "CTIO/CTIO/g.dat (energy, svo-curve-match), LBT/MODS/g.dat (photon, svo), "
                       "Misc/UCAC/sdss_g.dat (energy, svo), SLOAN/SDSS/g.dat (photon, svo)"), message
    for rel in ("CTIO/CTIO/g.dat", "Misc/UCAC/sdss_g.dat"):   # one tag for the curve: no problem left
        write_filter_header(t.tmp / rel, {**read_filter_header(t.tmp / rel), "DetectorType": PHOTON})
    assert list(label_check.problems(t.tmp)) == []


@row
def keys_from_newer_writers_are_kept(t):
    path = t.tmp / "GAIA" / "GAIA" / "G.dat"
    path.parent.mkdir(parents=True)
    path.write_text("Wavelength,Transmission,FilterID=GAIA/GAIA3.G;DetectorType=photon;DetectorTypeSource=svo;"
                    "WavelengthUnit=Angstrom;Components=filter;Future_Key2=a%20b\n3200.0,0.1\n6000.0,0.75\n")
    assert read_filter_header(path)["Future_Key2"] == "a b"
    assert read_filter_header(path)["DetectorType"] == PHOTON
    client = SimpleNamespace(get_filter=lambda fid: pytest.fail("an authoritative tag needs no lookup"))
    assert tag_filter_file(path, client=client).detector_type == PHOTON
    write_filter_header(path, {**read_filter_header(path), "DetectorType": "energy"})
    row = path.read_text().splitlines()[0]
    assert row.endswith(";Components=filter;Future_Key2=a%20b") and len(row.split(",")) == 3
    assert read_filter_header(path)["DetectorType"] == "energy"
    assert_legacy_readers(path)


@row
def invalid_metadata_keys_are_refused(t):
    with pytest.raises(ValueError):
        format_metadata_field({"bad key": "x"})


@row
def curve_without_column_row_is_never_given_one(t):
    raw = b"3830.0 0.000000\n4000.0 0.5\n"  # MESA has always discarded the first sample here
    path = t.tmp / "raw.dat"
    path.write_bytes(raw)
    assert not has_column_row(path)
    assert write_filter_header(path, {"DetectorType": "photon"}) is False
    assert path.read_bytes() == raw

    path.write_bytes(b"# DetectorType: photon\n# DetectorTypeSource: assumed\n" + raw)  # tagged by 0.3.0
    assert migrate_filter_file(path)
    assert path.read_bytes() == raw


@row
def a_real_third_column_is_not_taken_for_metadata(t):
    path = t.tmp / "E.dat"
    path.write_text("Wavelength,Transmission,Error\n1,2,0.1\n")
    assert read_filter_header(path) == {}
    write_filter_header(path, {"DetectorType": "photon"})
    assert path.read_text() == "Wavelength,Transmission,Error,DetectorType=photon\n1,2,0.1\n"
    assert read_filter_header(path) == {"DetectorType": "photon"}


@row
def migrated_photon_curve_still_loads_as_photon(t):
    path = t.tmp / "G.dat"
    path.write_text(TAGGED_030)
    migrate_filter_file(path)
    curve = load_filter_curve(str(path))
    assert curve.detector_type == PHOTON
    assert curve.metadata["filterid"] == "GAIA/GAIA3.G"
    assert curve.metadata["wavelength_unit_source"] == "header"
    np.testing.assert_allclose(curve.wavelength, [3200.0, 6000.0, 10500.0])


@row
def column_row_overrides_030_lines(t):
    path = t.tmp / "G.dat"
    path.write_text(TAGGED_030.replace("Wavelength,Transmission\n", "Wavelength,Transmission,DetectorType=energy\n"))
    assert read_filter_header(path)["DetectorType"] == "energy"
    migrate_filter_file(path)
    assert read_filter_header(path)["DetectorType"] == "energy"
    assert read_filter_header(path)["FilterID"] == "GAIA/GAIA3.G"


@row
def tagging_an_already_tagged_030_curve_migrates_it(t):
    def no_lookup(filter_id):
        raise AssertionError("an authoritative tag needs no lookup")

    path = t.tmp / "GAIA" / "GAIA" / "G.dat"
    path.parent.mkdir(parents=True)
    path.write_text(TAGGED_030)
    result = tag_filter_file(path, client=SimpleNamespace(get_filter=no_lookup))
    assert (result.detector_type, result.source, result.changed, result.migrated) == (
        PHOTON, SOURCE_SVO_MATCH, False, True)
    assert_legacy_readers(path)


@row
def components_and_products_are_legacy(t):
    directory = components_camera(t.tmp / "filters")
    product = build_product(directory, "filter+qe")
    curves = sorted(directory.rglob("*.dat"))
    assert any("components" in p.parts for p in curves) and any("products" in p.parts for p in curves)
    for path in curves:
        assert_legacy_readers(path)
        assert read_filter_header(path)["DetectorType"]
    assert (product / product.name).read_text().split() == ["g.dat", "r.dat"]


@row
def combined_set_repairs_030_inputs_without_touching_them(t):
    root = t.tmp / "filters"
    first = put(root / "GAIA" / "GAIA", {"G.dat": TAGGED_030})
    second = components_camera(t.tmp / "other")

    out = combine_filter_sets("Mixed", [first, second], filter_root=root)

    assert (first / "G.dat").read_text() == TAGGED_030
    for path in out.glob("*.dat"):
        assert_legacy_readers(path)
    assert read_filter_header(out / "G.dat")["FilterID"] == "GAIA/GAIA3.G"
    assert read_filter_header(out / "g.dat")["DetectorType"] == PHOTON
    assert sorted((out / "Mixed").read_text().split()) == sorted(p.name for p in out.glob("*.dat"))


MIRROR_DOWNLOAD = {"mirror_download_is_legacy_either_way_030_mirror": TAGGED_030,
                   "mirror_download_is_legacy_either_way_migrated_mirror": ROW_G + "\n" + BODY_030}


def _mirror_download(t, served):
    def get(url, **kwargs):
        if url.endswith("/G.dat"):
            return SimpleNamespace(text=served)
        if url.endswith("/GAIA"):
            return SimpleNamespace(text="G.dat\n")
        raise requests.HTTPError(response=SimpleNamespace(status_code=404))

    t.mp.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)  # no network
    grabber = NJMFilterGrabber(base_dir=str(t.tmp))
    grabber._available, grabber._get, grabber.download_components = True, get, lambda facility, instrument: 0
    t.mp.setattr("sed_tools.njm_filter_grabber.tag_filter_file", lambda path, **kw: tag_filter_file(
        path, client=SimpleNamespace(get_filter=lambda fid: pytest.fail("no SVO lookup needed"))))

    assert grabber.download_filters("GAIA", "GAIA", ["G"]) == 1
    path = t.tmp / "GAIA" / "GAIA" / "G.dat"
    assert path.read_text() == ROW_G + "\n" + BODY_030
    assert_legacy_readers(path)


@row
def cached_030_set_is_repaired_before_use(t):
    directory = components_camera(t.tmp / "filters")
    original = {}
    for path in directory.rglob("*.dat"):
        original[path] = read_filter_header(path)
        samples = path.read_text().splitlines(keepends=True)[1:]
        legacy = "".join(f"# {key}: {value}\n" for key, value in original[path].items())
        path.write_text(legacy + "Wavelength,Transmission\n" + "".join(samples))

    assert FilterLibrary(t.tmp / "filters").published("Test", "Camera") == directory
    for path, fields in original.items():
        assert_legacy_readers(path)
        assert read_filter_header(path) == fields


@row
def tree_migration_is_offline_and_idempotent(t):
    root = put(t.tmp / "filters" / "GAIA" / "GAIA",
               {"G.dat": TAGGED_030, "Bp.dat": "Wavelength,Transmission\n1,1\n2,1\n"})
    t.mp.setattr(requests.Session, "request", lambda *a, **k: pytest.fail("migration is offline"))

    assert migrate_filter_tree(t.tmp / "filters", dry_run=True) == [root / "G.dat"]
    assert (root / "G.dat").read_text() == TAGGED_030
    assert migrate_filter_tree(t.tmp / "filters") == [root / "G.dat"]
    assert_legacy_readers(root / "G.dat")
    assert migrate_filter_tree(t.tmp / "filters") == []


# ======================================================================
# Wavelength units of filter curves and spectra (test_wavelength_units)
# ======================================================================
INFER_UNIT = {f"infer_unit_from_values_{unit}": (lo, hi, unit) for lo, hi, unit in [
    (3200.0, 10800.0, "angstrom"),  # Gaia G in A (was misread as nm)
    (460.0, 740.0, "nanometer"),  # Johnson V in nm
    (0.3, 28.0, "micrometer"),
    (5e-7, 1e-6, "meter"),
]}


def _infer_unit(t, lo, hi, expected):
    assert infer_unit_from_values(np.array([lo, hi])) == expected


FILTER_CURVE_UNIT = {f"filter_curve_wavelength_unit_{name}": case for name, case in {
    "optical_filter_without_header_in_angstrom": (
        "", None, [(4600.0, 0.0), (5500.0, 1.0), (7400.0, 0.0)], 1.0, "inferred", "declares no wavelength unit"),
    **{f"header_{name}": (f"# WavelengthUnit: {unit}\n", None, [(2.0, 1.0), (3.0, 1.0)], factor, "header", "")
       for name, unit, factor in [("angstrom", "Angstrom", 1.0), ("nm", "nm", 10.0), ("um", "um", 1e4),
                                  ("a_with_a_ring", "Å", 1.0), ("microns", "microns", 1e4)]},
    # A K-band curve in nm is indistinguishable from a UV curve in A by value alone.
    "argument_overrides_inference": ("", "nm", [(1950.0, 1.0), (2400.0, 1.0)], 10.0, "argument", ""),
}.items()}


def _filter_curve_unit(t, header, argument, points, factor, source, warning):
    path = t.tmp / "f.dat"
    path.write_text(header + "Wavelength,Transmission\n" + "".join(f"{w},{tr}\n" for w, tr in points))
    with t.caplog.at_level(logging.WARNING):
        curve = load_filter_curve(str(path), wavelength_unit=argument)
    np.testing.assert_allclose(curve.wavelength, [w * factor for w, _ in points])
    assert source is None or curve.metadata["wavelength_unit_source"] == source
    assert warning in t.caplog.text


@row
def spectrum_accepts_explicit_unit(t):
    path = t.tmp / "s.txt"
    path.write_text("500 1.0\n600 2.0\n")
    spec = load_spectrum(str(path), wavelength_unit="angstrom")
    np.testing.assert_allclose(spec.wavelength, [500.0, 600.0])
    with pytest.raises(ValueError, match="Unknown wavelength unit 'nm \\(air\\)': declare WavelengthUnit as"):
        load_spectrum(str(path), wavelength_unit="nm (air)")
