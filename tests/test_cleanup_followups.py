"""Follow-ups to the cleanup: exit statuses, re-prompting menus and warnings that change no numbers."""

import json
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import requests


class NetworkUsed(BaseException):
    """Raised past every ``except Exception`` so a network call cannot be swallowed."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise NetworkUsed("tests must not use the network")

    monkeypatch.setattr(requests.Session, "send", refuse)


def answer(monkeypatch, *replies):
    """Feed ``input()`` these replies in order; a further prompt fails the test."""
    queue = list(replies)

    def fake_input(prompt=""):
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


def run_cli(monkeypatch, capsys, *argv):
    from sed_tools import cli

    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    code = cli.main()
    return code, capsys.readouterr().out


# ----------------------------------------------------------------------
# 1. Catalogue gaps vs fetch failures
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


@pytest.fixture
def njm_root(tmp_path, monkeypatch):
    """A filter root whose catalogue is served by a FakeMirror; returns ``(root, install)``."""
    from sed_tools.api import Filters
    from sed_tools.filter_catalogue import FilterCatalogue

    monkeypatch.setattr(Filters, "_catalogues", {})
    root = tmp_path / "filters"
    root.mkdir()

    def install(files):
        key = (str(root.resolve()), False)
        Filters._catalogues[key] = FilterCatalogue(key[0], mirror=FakeMirror(files))

    return root, install


def test_search_reports_gaps_apart_from_fetch_failures(tmp_path):
    from sed_tools.filter_catalogue import FilterCatalogue

    catalogue = FilterCatalogue(tmp_path / "filters", mirror=FakeMirror(catalogue_files(serve_ir=False)))
    result = catalogue.search()
    assert list(result["band"]) == ["g", "U", "K"]
    assert result.attrs["gaps"] == ["Fac/Old: NJM has no properties file"]
    assert result.attrs["errors"] == ["Other/IR: NJM does not serve filters/Other/IR/properties.json"]


def test_filters_search_exits_1_when_properties_could_not_be_fetched(njm_root, monkeypatch, capsys):
    root, install = njm_root
    install(catalogue_files(serve_ir=False))
    code, out = run_cli(monkeypatch, capsys, "filters-search", "--base", str(root))
    assert code == 1
    assert "[filters] Fac/Old: NJM has no properties file" in out
    assert "[filters] Other/IR: NJM does not serve filters/Other/IR/properties.json" in out


def test_filters_search_exits_0_when_there_are_only_gaps(njm_root, monkeypatch, capsys):
    root, install = njm_root
    install(catalogue_files())
    code, out = run_cli(monkeypatch, capsys, "filters-search", "--base", str(root))
    assert code == 0 and "[filters] Fac/Old: NJM has no properties file" in out

    code, out = run_cli(monkeypatch, capsys, "filters-search", "--base", str(root), "--format", "json")
    data = json.loads(out)
    assert code == 0 and data["errors"] == [] and data["gaps"] == ["Fac/Old: NJM has no properties file"]
