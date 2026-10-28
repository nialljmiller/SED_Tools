from types import SimpleNamespace

import pytest

from sed_tools import cli
from sed_tools.filter_library import FilterLibrary
from sed_tools.parsing import parse_multi_selection
from sed_tools.ui_utils import prompt_choice


def test_parse_multi_selection_accepts_ranges_and_ids() -> None:
    assert parse_multi_selection("1,3-5,5", 6) == [0, 2, 3, 4]
    assert parse_multi_selection("2-1,4", 5) == [0, 1, 3]


def test_parse_multi_selection_rejects_out_of_bounds() -> None:
    try:
        parse_multi_selection("7", 6)
    except ValueError as exc:
        assert "out of bounds" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")


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
    monkeypatch.setattr(cli, "_offer_composition", offered.append)
    return SimpleNamespace(root=tmp_path, fetched=fetched, offered=offered)


@pytest.mark.parametrize("selection,facilities", [("1,3", ["A", "C"]), ("1-3", ["A", "B", "C"])])
def test_multiple_facilities_download_every_instrument_after_one_confirmation(filter_catalogue, monkeypatch, selection, facilities):
    answers = iter([selection, "y"])  # Any further confirmation, instrument or component prompt would fail.
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cli.run_filters_flow(str(filter_catalogue.root))
    assert filter_catalogue.fetched == [(facility, instrument) for facility in facilities
                                        for instrument in ("Camera1", "Camera2")]
    assert filter_catalogue.offered == []


def test_single_facility_opens_instrument_browser(filter_catalogue, monkeypatch):
    answers = iter(["2", "1", "y", "n", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cli.run_filters_flow(str(filter_catalogue.root))
    assert filter_catalogue.fetched == [("B", "Camera1")]
    assert filter_catalogue.offered == [filter_catalogue.root / "B" / "Camera1"]


def test_picker_reprompts_on_out_of_range_selection(monkeypatch):
    answers = iter(["1,99", "1,3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert prompt_choice(["A", "B", "C"], "Facilities", multi=True) == [0, 2]


def test_picker_id_command_returns_a_multi_selection(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "id 2")
    assert prompt_choice(["A", "B", "C"], "Facilities", multi=True) == [1]
