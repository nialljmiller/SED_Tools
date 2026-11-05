"""Fakes shared by the CLI menu and picker tests: input() answers that record the prompts, a
terminal stand-in, a Gaia catalogue, availability summaries, and the 80-column terminal.
"""

from types import SimpleNamespace

import pytest

from sed_tools import cli


class _Stdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def answer(monkeypatch, *answers):
    """Feed input() these answers; return the prompts it was given. Running out is a test failure."""
    queue, prompts = list(answers), []

    def fake_input(prompt=""):
        prompts.append(prompt)
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


@pytest.fixture(autouse=True)
def narrow_terminal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.delenv("SED_TOOLS_DEBUG", raising=False)


def _availability(**extra):
    info = {"facility": "Fac", "instrument": "Cam", "bands": ["J", "H"], "published_source": "", "published": "",
            "published_bands": {}, "components": ["filter"], "partial_components": {}, "buildable": ["filter"],
            "component_status": "", "component_availability": {"status": "supported", "builder": None},
            "missing_files": [], "errors": [], "atmosphere": False, "atmosphere_site": None,
            "atmosphere_sites": [], "platform": "ground", "detector_type": "", "detector_type_sources": {},
            "availability_source": "local", "variants": {}}
    info.update(extra)
    return info


def _summary_inputs(monkeypatch, info=None, catalogue=None):
    """Stub what the picker's summary reads (FilterLibrary.available, NJM's catalogue entry); returns the calls."""
    from sed_tools.filter_library import FilterLibrary

    calls = []
    monkeypatch.setattr(FilterLibrary, "available",
                        lambda self, f, i, remote=True: calls.append(f"{f}/{i}") or (info or _availability()))
    monkeypatch.setattr(cli.filters, "_catalogue_entry", lambda *a, **k: catalogue or {
        "info": None, "properties": None, "errors": ["NJM does not serve it"]})
    return calls


GAIA_INSTRUMENTS = [
    {"key": name, "label": name, "fetch_key": name, "sources": sources, "description": description}
    for name, sources, description in [
        ("DR2", ["njm", "original"], "Gaia DR2 passbands (Evans et al. 2018)"),
        ("DR2r", ["njm", "original"], "Gaia DR2 revised passbands (Evans et al. 2018)"),
        ("GAIA", ["njm"], "Gaia DR3 passbands + BP bright/faint (Maíz Apellániz & Weiler 2018)"),
        ("GAIA0", ["njm"], "Gaia pre-launch passbands (Jordi et al. 2010)"),
        ("Gaia2m", ["svo"], "Revised passbands for Gaia DR2.")]]


@pytest.fixture
def gaia(tmp_path, monkeypatch):
    from sed_tools.filter_library import FilterLibrary

    monkeypatch.setattr(FilterLibrary, "facilities", lambda self: {"entries": [
        {"key": "GAIA", "label": "GAIA", "fetch_key": "GAIA", "sources": ["njm", "svo"],
         "description": "Gaia mission."}], "errors": []})
    monkeypatch.setattr(FilterLibrary, "instruments", lambda self, facility: {
        "entries": GAIA_INSTRUMENTS, "errors": []})
    monkeypatch.setattr(FilterLibrary, "mirror", SimpleNamespace(is_available=lambda: True))
    fetched = []
    monkeypatch.setattr(FilterLibrary, "published", lambda self, f, i: fetched.append(f"{f}/{i}") or tmp_path)
    monkeypatch.setattr(cli.filters, "_offer_composition", lambda directory: None)
    return fetched


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    from sed_tools.filter_library import FilterLibrary

    monkeypatch.setattr(FilterLibrary, "facilities", lambda self: {"entries": [
        {"key": "GAIA", "label": "GAIA", "fetch_key": "Gaia", "sources": ["svo"]},
        {"key": "TWOMASS", "label": "2MASS", "sources": ["svo"]}], "errors": []})
    monkeypatch.setattr(FilterLibrary, "instruments", lambda self, facility: {"entries": [
        {"key": "GAIA2", "label": "GAIA2", "fetch_key": "DR2", "sources": ["svo"]},
        {"key": "Cam", "label": "Cam", "sources": ["svo"]}], "errors": []})
    fetched = []
    monkeypatch.setattr(FilterLibrary, "published", lambda self, f, i: fetched.append(f"{f}/{i}") or tmp_path)
    monkeypatch.setattr(cli.filters, "_offer_composition", lambda directory: None)
    _summary_inputs(monkeypatch)
    return fetched
