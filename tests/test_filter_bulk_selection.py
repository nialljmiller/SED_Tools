from types import SimpleNamespace

import pytest

from sed_tools import cli
from sed_tools.filter_library import FilterLibrary
from sed_tools.parsing import parse_multi_selection
from sed_tools.ui_utils import prompt_choice


@pytest.fixture
def filter_catalogue(tmp_path, monkeypatch):
    facilities = [{"key": name, "label": name} for name in ("A", "B", "C")]
    monkeypatch.setattr(FilterLibrary, "facilities", lambda self: {"entries": facilities, "errors": []})
    monkeypatch.setattr(FilterLibrary, "instruments", lambda self, facility:
                        {"entries": [{"key": "Camera1", "label": "Camera1"},
                                     {"key": "Camera2", "label": "Camera2"}], "errors": []})
    fetched, offered = [], []

    def fetch(self, facility, instrument):
        fetched.append((facility, instrument))
        return tmp_path / facility / instrument

    monkeypatch.setattr(FilterLibrary, "published", fetch)  # the browser downloads with its own library
    monkeypatch.setattr(cli.filters, "_offer_composition", offered.append)
    # What the browser describes before asking to download (Filters.available, NJM's catalogue entry).
    monkeypatch.setattr(FilterLibrary, "available", lambda self, facility, instrument, remote=True: {
        "facility": facility, "instrument": instrument, "bands": ["V"], "published": "filter",
        "published_bands": {}, "components": [], "partial_components": {}, "component_status": None,
        "errors": [], "atmosphere": False, "platform": "ground", "variants": {}})
    monkeypatch.setattr(cli.filters, "_catalogue_entry", lambda *a, **k: {"info": None, "properties": None, "errors": []})
    return SimpleNamespace(root=tmp_path, fetched=fetched, offered=offered)


@pytest.mark.parametrize("terminal, answers", [
    pytest.param(False, ["1-3"], id="piped_without_prompts"),  # piped or scripted input, as in 0.4.1
    pytest.param(True, ["1-3", "y"], id="terminal_after_one_confirmation"),
])
def test_multiple_facilities_download_every_instrument(filter_catalogue, monkeypatch, terminal, answers):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: terminal))
    answers = iter(answers)  # Any further confirmation, instrument or component prompt would fail.
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cli.run_filters_flow(str(filter_catalogue.root))
    assert next(answers, None) is None  # every answer was asked for
    assert filter_catalogue.fetched == [(facility, instrument) for facility in ("A", "B", "C")
                                        for instrument in ("Camera1", "Camera2")]
    assert filter_catalogue.offered == []


def test_picker_reprompts_on_out_of_range_selection(monkeypatch):
    answers = iter(["1,99", "1,3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert prompt_choice(["A", "B", "C"], "Facilities", multi=True) == [0, 2]


def test_picker_id_command_returns_a_multi_selection(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "id 2")
    assert prompt_choice(["A", "B", "C"], "Facilities", multi=True) == [1]
