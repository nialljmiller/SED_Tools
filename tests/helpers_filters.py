"""Stand-ins for NJM and SVO shared by the filter tests: a fake NJM session, the NJM index and
set writers, an SVO stub and browser, bundled curves, and the NJM-down settings.
"""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from sed_tools import filter_library
from sed_tools.filter_components import InstrumentManifest
from sed_tools.filter_library import instrument_catalogue
from sed_tools.njm_filter_grabber import NJMFilterGrabber
from sed_tools.svo_filter_grabber import Facility, FilterRow, Instrument, SVOFilterBrowser


NJM = "https://data.nialljmiller.com/sed_tools"
CURVE = "Wavelength,Transmission,DetectorType=photon;DetectorTypeSource=original\n4000,0\n5000,1\n6000,0\n"


class Served:
    """A fake NJM session: ``files`` maps paths under the mirror to text, an HTTP status, or an exception."""

    def __init__(self, files):
        self.files = dict(files)
        self.headers, self.verify, self.requested = {}, True, []

    def get(self, url, timeout=None, headers=None, **kwargs):
        rel = url[len(NJM) + 1:]
        self.requested.append(rel)
        body = self.files.get(rel, 404)
        if isinstance(body, Exception):
            raise body
        status, text = (body, "") if isinstance(body, int) else (200, body)

        def raise_for_status():
            if status >= 400:
                raise requests.HTTPError(f"{status} for {rel}", response=SimpleNamespace(status_code=status, headers={}))

        return SimpleNamespace(status_code=status, text=text, content=text.encode(), headers={},
                               raise_for_status=raise_for_status, json=lambda: json.loads(text))


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


def njm_index(detailed):
    return json.dumps({"generated": "2026-10-27T00:00:00",
                       "filters": {"facilities": sorted(detailed), "facilities_detailed": {
                           fac: {"instruments": insts} for fac, insts in detailed.items()}}})


def write_set(directory, bands, *, index=None, manifest_bands=None, archived=None, text=CURVE):
    directory.mkdir(parents=True, exist_ok=True)
    for band in bands:
        (directory / f"{band}.dat").write_text(text)
    (directory / directory.name).write_text("".join(f"{b}.dat\n" for b in (index if index is not None else bands)))
    data = asdict(InstrumentManifest(directory.parent.name, directory.name, list(manifest_bands or bands)))
    if archived is not None:
        data["archived_bands"] = archived
    (directory / "manifest.json").write_text(json.dumps(data))
    return directory


GAIA = {"DR2": {"description": "Gaia DR2 passbands (Evans et al. 2018)", "svo_instruments": ["GAIA/GAIA2"],
                "bands": ["BP", "G", "RP"], "svo_bands": {"GAIA/GAIA2": ["G", "Gbp=BP", "Grp=RP"]}},
        "GAIA": {"description": "Gaia DR3 passbands", "svo_instruments": ["GAIA/GAIA3"],
                 "bands": ["G", "Gbp", "Gbp_bright", "Gbp_faint", "Grp", "Grvs"],
                 "svo_bands": {"GAIA/GAIA3": ["G", "Gbp", "Grp", "Grvs"]}},
        "GAIA0": {"svo_instruments": ["GAIA/GAIA0"], "bands": ["G"], "svo_bands": {"GAIA/GAIA0": ["G"]}}}


@pytest.fixture
def bundled(monkeypatch):
    def use(catalogue):
        facilities = {fac: {"instruments": insts} for fac, insts in catalogue.items()}
        monkeypatch.setattr(filter_library, "load_instrument_catalogue", lambda: instrument_catalogue(facilities))
    use({})
    return use


@pytest.fixture
def njm(monkeypatch):
    monkeypatch.setattr(NJMFilterGrabber, "_check_availability", lambda self: True)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.time.sleep", lambda seconds: None)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.tag_filter_file", lambda *a, **k: None)

    def make(root, files):
        grabber = NJMFilterGrabber(base_dir=str(root))
        grabber.session = Served(files)
        return grabber

    return make


#: An NJMFilterGrabber that found NJM down: it asks nothing and lists nothing.
NJM_DOWN = SimpleNamespace(is_available=lambda: False, fetch_manifest=lambda f, i: None,
                           discover_filters=lambda f, i: [])


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


def atmosphere_site(root, site, manifest, grids=()):
    directory = root / "atmospheres" / site
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for name in grids:
        (directory / name).write_text("Wavelength,Transmission\n5000,0.9\n6000,0.95\n")
    return directory


STALE = {"site": "lsst_std", "schema": 1, "source": "Rubin", "altitude_m": 2647.0,
         "entries": [{"airmass": 1.2, "pwv": None, "file": "am1.20.dat"}]}
