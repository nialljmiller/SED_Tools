"""Nill's decisions of 28 October for 0.5.0.

1. With NJM down, a product of NJM's that SVO holds only in part is installed,
   with one warning, and the call succeeds; the set is shown as incomplete.

Every source is mocked; nothing here touches the network.
"""

import logging
import sys

import pytest

from sed_tools import cli
from sed_tools.api import Filters
from sed_tools.filter_components import ComponentError
from sed_tools.filter_library import FilterLibrary
from test_release_njmdown import GAIA, NJM_DOWN, bundled, svo_browser  # noqa: F401 (bundled is a fixture)


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["sed-tools", *argv])
    return cli.main()


# ----------------------------------------------------------------------
# 1. A partial set from SVO, NJM down: installed, warned about, a success
# ----------------------------------------------------------------------
@pytest.fixture
def gaia_from_svo(tmp_path, bundled, monkeypatch):
    """Every FilterLibrary sees NJM down and an SVO holding 4 of GAIA/GAIA's 6 bands; returns the filter root."""
    bundled({"GAIA": GAIA})
    root = tmp_path / "filters"
    browser = svo_browser(root, {("GAIA", "GAIA3"): ["G", "Gbp", "Grp", "Grvs"]})
    monkeypatch.setattr(FilterLibrary, "mirror", NJM_DOWN)
    monkeypatch.setattr(FilterLibrary, "svo", property(lambda self: browser))
    monkeypatch.setenv("COLUMNS", "80")
    return root


def incomplete_warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and "is incomplete" in r.getMessage()]


def test_filters_fetch_returns_the_partial_set_with_one_warning(gaia_from_svo, caplog):
    with caplog.at_level(logging.WARNING):
        directory = Filters.fetch("GAIA", "GAIA", filter_root=gaia_from_svo)
    assert directory == gaia_from_svo / "GAIA" / "GAIA"
    assert len(incomplete_warnings(caplog)) == 1  # one line for the set, not one per band
    assert "missing Gbp_bright, Gbp_faint" in incomplete_warnings(caplog)[0].getMessage()

    info = Filters.available("GAIA", "GAIA", filter_root=gaia_from_svo, include_remote=False)
    assert info["bands"] == ["G", "Gbp", "Grp", "Grvs"]
    assert info["expected_bands"] == GAIA["GAIA"]["bands"]  # how a script tells the set is incomplete
    assert info["missing_files"] == ["Gbp_bright.dat", "Gbp_faint.dat"]


def test_the_filters_command_exits_0_and_warns_once(gaia_from_svo, monkeypatch, capsys, caplog):
    with caplog.at_level(logging.WARNING):
        assert run_cli(monkeypatch, "filters", "GAIA/GAIA", "--base", str(gaia_from_svo)) == 0
    assert capsys.readouterr().out.rstrip().endswith(str(gaia_from_svo / "GAIA" / "GAIA"))
    [warning] = incomplete_warnings(caplog)
    assert "Fetch again when NJM is reachable to complete it" in warning.getMessage()


def test_filters_info_and_the_picker_summary_say_the_set_is_incomplete(gaia_from_svo, monkeypatch, capsys):
    Filters.fetch("GAIA", "GAIA", filter_root=gaia_from_svo)
    capsys.readouterr()
    assert run_cli(monkeypatch, "filters-info", "GAIA/GAIA", "--base", str(gaia_from_svo), "--offline") == 0
    out = " ".join(capsys.readouterr().out.split())  # wrapped at 80 columns
    assert ("Incomplete : 4 of 6 bands installed; missing Gbp_bright, Gbp_faint; fetch it again when NJM is "
            "reachable to complete it") in out

    info = Filters.available("GAIA", "GAIA", filter_root=gaia_from_svo, include_remote=False)
    lines = cli._pick_summary("GAIA/GAIA", {}, info, {"info": None, "properties": None, "errors": []}, "SVO")
    assert "  Incomplete : 4 of 6 bands installed; missing Gbp_bright, Gbp_faint" in lines
    assert all(len(line) <= 80 for line in lines)


def test_a_complete_set_has_no_incomplete_line(tmp_path, monkeypatch, capsys):
    from test_release_library import write_set

    root = tmp_path / "filters"
    write_set(root / "Fac" / "Cam", ["B", "V"])
    assert run_cli(monkeypatch, "filters-info", "Fac/Cam", "--base", str(root), "--offline") == 0
    assert "Incomplete" not in capsys.readouterr().out


def test_an_svo_curve_that_could_not_be_saved_still_fails(gaia_from_svo, monkeypatch):
    def refuse(self, row, out_path):
        if row.band == "Grvs":
            raise ValueError("SVO returned no transmission data")
        original(self, row, out_path)

    from sed_tools.svo_filter_grabber import SVOFilterBrowser

    original = SVOFilterBrowser._save_filter
    monkeypatch.setattr(SVOFilterBrowser, "_save_filter", refuse)
    with pytest.raises(ComponentError) as raised:
        Filters.fetch("GAIA", "GAIA", filter_root=gaia_from_svo)
    message = str(raised.value)
    assert "GAIA/GAIA is only partly installed" in message and "3 of NJM's 6 bands came from SVO's GAIA/GAIA3" in message
    assert "GAIA/GAIA3.Grvs: SVO returned no transmission data" in message
