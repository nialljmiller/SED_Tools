"""The SVO filter browser at a terminal (python -m sed_tools.svo_filter_grabber), answered through ``input()``.

SVO is a stand-in that lists Alpha (cameras Cam, with bands g and r, and Dead, with none) and Beta (camera IR,
with band J), and serves one curve for every band; the curves are written under tmp_path.
"""

from types import SimpleNamespace

import pytest
import requests

from sed_tools.filters import svo
from sed_tools.svo_filter_grabber import Facility
from test_cli_mains import keep_random_state  # noqa: F401 (an autouse fixture)
from test_release_njmdown import svo_browser


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


@pytest.fixture
def browser(tmp_path, monkeypatch):
    """The browser run_interactive makes; ``browser.saved`` lists the filter IDs whose curves it fetched."""
    made = svo_browser(tmp_path, {("Alpha", "Cam"): ["g", "r"], ("Alpha", "Dead"): [], ("Beta", "IR"): ["J"]})
    monkeypatch.setattr(svo, "SVOFilterBrowser", lambda base_dir: made)
    return made


def test_one_instrument_is_downloaded_then_the_browser_is_left(browser, tmp_path, monkeypatch, capsys):
    # single-facility mode, Alpha, Cam, download, no other camera; then single-facility mode and quit
    answer(monkeypatch, "", "1", "1", "", "", "", "q")
    svo.run_interactive(str(tmp_path))
    assert browser.saved == ["Alpha/Cam.g", "Alpha/Cam.r"]
    assert (tmp_path / "Alpha" / "Cam" / "Cam").read_text().split() == ["g.dat", "r.dat"]
    out = capsys.readouterr().out
    assert "Instrument: Cam (Cam)" in out and "Filters  : 2 available" in out
    assert out.rstrip().endswith("Exiting.")


def test_instruments_can_be_declined_skipped_when_empty_and_left(browser, tmp_path, monkeypatch, capsys):
    answer(monkeypatch, "", "1",
           "2",           # Dead has no filters
           "1", "n",      # Cam, but do not download
           "1", "y", "y", # Cam, downloaded; another from Alpha
           "b",           # back to the facilities
           "", "2", "q")  # Beta, then quit at its instruments
    svo.run_interactive(str(tmp_path))
    assert browser.saved == ["Alpha/Cam.g", "Alpha/Cam.r"]
    out = capsys.readouterr().out
    assert "No filters found for Dead." in out
    assert "Instruments for Beta" in out and out.rstrip().endswith("Exiting.")


def test_bulk_mode_downloads_every_instrument_of_the_facilities_chosen(browser, tmp_path, monkeypatch, capsys):
    answer(monkeypatch, "9",        # not a facility
           ",",                     # none chosen
           "1,2", "n",              # chosen, then not confirmed
           "1-2", "", "y",          # downloaded; another set
           "2", "y", "")            # Beta again, then done
    svo.run_interactive(str(tmp_path))
    assert browser.saved == ["Alpha/Cam.g", "Alpha/Cam.r", "Beta/IR.J", "Beta/IR.J"]
    out = capsys.readouterr().out
    assert "Invalid selection: Selection out of bounds: 9" in out and "No facilities selected." in out
    assert "  [skip] Dead: no filters found" in out
    assert "Facility: Alpha (2 instruments)" in out and "Downloading 1 filters for IR" in out


def test_with_nothing_listed_the_browser_says_so(browser, tmp_path, monkeypatch, capsys):
    browser.list_facilities = lambda: [Facility("Empty", "Empty")]
    browser.list_instruments = lambda facility: []
    answer(monkeypatch, "", "1", "", "q")
    svo.run_interactive(str(tmp_path))
    assert "No instruments found for Empty." in capsys.readouterr().out
    browser.list_facilities = lambda: []
    svo.run_interactive(str(tmp_path))
    assert capsys.readouterr().out == "No facilities discovered.\n"


def test_main_browses_into_its_base_dir_and_exits_1_when_a_curve_fails(browser, tmp_path, monkeypatch, capsys):
    answer(monkeypatch, "", "q")
    assert svo.main([str(tmp_path)]) == 0
    browser.metadata_client = SimpleNamespace(get_filter=lambda filter_id: None)  # SVO serves no curve
    answer(monkeypatch, "", "2", "1", "")
    assert svo.main([str(tmp_path)]) == 1
    assert "1 of 1 SVO filters could not be saved" in capsys.readouterr().out
