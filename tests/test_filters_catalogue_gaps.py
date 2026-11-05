"""A catalogue search reports an instrument NJM has no properties file for (a gap) apart
from one whose file it could not fetch (a failure).
"""

import json
from types import SimpleNamespace

import requests

from helpers_api import no_network  # noqa: F401 (an autouse fixture)


# ----------------------------------------------------------------------
# Catalogue gaps vs fetch failures
# ----------------------------------------------------------------------
NJM = "https://njm.invalid/sed_tools"


class FakeMirror:
    """Serves ``files`` under NJM; anything else is a 404."""

    base_url = NJM

    def __init__(self, files):
        self.files = files

    def is_available(self):
        return True

    def _get(self, url, timeout=None, headers=None):
        rel = url[len(NJM) + 1:]
        if rel not in self.files:
            raise requests.HTTPError("404", response=SimpleNamespace(status_code=404, headers={}))
        return SimpleNamespace(status_code=200, content=self.files[rel].encode(), headers={})


def catalogue_files(serve_ir=True):
    """Fac/Cam is served whole; Fac/Old has no properties file (a gap); Other/IR's may be missing (a failure)."""
    def instrument(bands, properties):
        return {"filters": bands, "detector_types": {}, "properties": properties, "info": None}

    index = {"generated": "2026-10-27T00:00:00", "filters": {"facilities_detailed": {
        "Fac": {"instruments": {"Cam": instrument(["g"], "properties.json"), "Old": instrument(["U"], None)}},
        "Other": {"instruments": {"IR": instrument(["K"], "properties.json")}},
    }}}
    files = {"index.json": json.dumps(index),
             "filters/Fac/Cam/properties.json": json.dumps({"bands": {"g": {"WavelengthPivot": 4800.0}}})}
    if serve_ir:
        files["filters/Other/IR/properties.json"] = json.dumps({"bands": {"K": {"WavelengthPivot": 21600.0}}})
    return files


def test_search_reports_gaps_apart_from_fetch_failures(tmp_path):
    from sed_tools.filter_catalogue import FilterCatalogue

    catalogue = FilterCatalogue(tmp_path / "filters", mirror=FakeMirror(catalogue_files(serve_ir=False)))
    result = catalogue.search()
    assert list(result["band"]) == ["g", "U", "K"]
    assert result.attrs["gaps"] == ["Fac/Old: NJM has no properties file"]
    assert result.attrs["errors"] == ["Other/IR: NJM does not serve filters/Other/IR/properties.json"]
