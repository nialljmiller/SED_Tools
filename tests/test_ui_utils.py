"""The prompts and pickers: numbers asked again (and NaN refused), names only, descriptions once,
full names in columns, long labels cut with an ellipsis, a count before downloading several, and each
listing error printed once.
"""

from __future__ import annotations

import re


from sed_tools import cli
from sed_tools.api import Filters  # noqa: E402
from sed_tools.filter_components import ComponentError

from helpers_cli_menus import GAIA_INSTRUMENTS, _Stdin, answer
from helpers_cli_menus import catalogue, gaia, narrow_terminal  # noqa: F401 (fixtures)


# ----------------------------------------------------------------------
# messages, not tracebacks; the menu survives a failed workflow
# ----------------------------------------------------------------------
def test_number_prompts_ask_again(monkeypatch, capsys):
    answer(monkeypatch, "high", "", "1.5")
    assert cli._ask_number("Airmass: ") == 1.5
    out = capsys.readouterr().out
    assert "'high' is not a number; try again." in out and "Please enter a number." in out
    answer(monkeypatch, "")
    assert cli._ask_number("PWV: ", default=None) is None
    answer(monkeypatch, "UNSTATED")
    assert cli._ask_number("Airmass: ", words=("unstated",)) == "unstated"


# ----------------------------------------------------------------------
# Atmospheres and the composition prompt
# ----------------------------------------------------------------------
def test_composition_offers_the_bands_that_work(tmp_path, monkeypatch, capsys):
    calls = []

    def compose(cls, directory, **kwargs):
        calls.append(kwargs.get("bands"))
        if kwargs.get("bands") is None:
            raise ComponentError("am1.00_pwv1.dat covers 9000-56000 A but 70% of the band lies beyond it")

    monkeypatch.setattr(Filters, "compose", classmethod(compose))
    monkeypatch.setattr(cli.filters, "_composable_bands", lambda d, request: (["J", "H"], {"Z": "covers 9000-56000 A"}))
    answer(monkeypatch, "y")
    request = {"components": ["filter"], "variants": None, "detector_type": None, "allow_heavy": False,
               "atmosphere": {"site": "paranal", "airmass": 1.0, "pwv": 1.0}}
    cli._compose_the_bands_that_work(tmp_path / "Paranal" / "VISTA", request, ComponentError("no"))
    assert calls == [["J", "H"]]
    assert "Z: covers 9000-56000 A" in capsys.readouterr().out


# ----------------------------------------------------------------------
# The picker: 0.4.1's numbered columns of names
# ----------------------------------------------------------------------
def test_the_pickers_show_names_only(gaia, tmp_path, monkeypatch, capsys):
    answer(monkeypatch, "1", "q")
    assert cli.run_filters_flow(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "   1. GAIA" in out and "Gaia mission" not in out  # no facility descriptions are invented
    pickers = out.split("\nGAIA: 5 instruments\n")[0] + out.split("Instruments for GAIA")[1]
    assert not any(text in pickers for text in ("[njm", "[svo", "Tags show", "Evans et al."))
    assert cli._picker_labels(GAIA_INSTRUMENTS + [{"key": "Other"}]) == [
        "DR2", "DR2r", "GAIA", "GAIA0", "Gaia2m", "Other"]


def test_picking_a_facility_lists_its_instruments_with_their_descriptions_once(gaia, tmp_path, monkeypatch,
                                                                               capsys):
    answer(monkeypatch, "1", "/DR2", "q")
    cli.run_filters_flow(str(tmp_path))
    out = capsys.readouterr().out
    block = out.split("\nGAIA: 5 instruments\n")[1].split("\n\nInstruments for GAIA")[0]
    assert block.splitlines() == [
        "  DR2     Gaia DR2 passbands (Evans et al. 2018)",
        "  DR2r    Gaia DR2 revised passbands (Evans et al. 2018)",
        "  GAIA    Gaia DR3 passbands + BP bright/faint (Maíz Apellániz & Weiler 2018)",
        "  GAIA0   Gaia pre-launch passbands (Jordi et al. 2010)",
        "  Gaia2m  Revised passbands for Gaia DR2."]
    assert out.count("GAIA: 5 instruments") == 1  # not again when the list is filtered


def test_the_instrument_picker_shows_full_names_in_columns(gaia, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "80")
    answer(monkeypatch, "1", "q")
    cli.run_filters_flow(str(tmp_path))
    rows = capsys.readouterr().out.split("Instruments for GAIA\n")[1].splitlines()[1:3]
    assert re.fullmatch(r"\s+1\. DR2\s+3\. GAIA\s+5\. Gaia2m\s*", rows[0])  # 0.4.1 cut these to 'D', 'GA', 'Gaia'
    assert re.fullmatch(r"\s+2\. DR2r\s+4\. GAIA0\s*", rows[1])
    assert all(len(row) <= 80 for row in rows)


def test_picker_cuts_long_labels_with_an_ellipsis(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "40")
    answer(monkeypatch, "q")
    cli.prompt_choice(["x" * 50], "Things")
    assert "x" * 23 + "…" in capsys.readouterr().out


def test_downloading_several_facilities_asks_first_with_the_count(catalogue, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _Stdin(True))  # only a terminal is asked; piped input downloads, as in 0.4.1
    prompts = answer(monkeypatch, "1,2", "n")
    assert cli.run_filters_flow(str(tmp_path)) == 0
    assert catalogue == []
    assert prompts[-1] == "Download all 4? [y/N] "
    assert "4 instruments in 2 facilities: GAIA (2), 2MASS (2)" in capsys.readouterr().out


# ----------------------------------------------------------------------
# One re-asking number question for the CLI and the ML and densifier menus
# ----------------------------------------------------------------------
def test_every_number_question_is_the_same_one_and_refuses_nan(monkeypatch, capsys):
    from sed_tools import ml_sed_completer, ml_sed_generator, ui_utils

    assert cli._ask_number is ml_sed_generator._ask_number is ml_sed_completer._ask_number is ui_utils.ask_number
    answers = iter(["nan", "inf", "", "1.3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli._ask_number("Airmass: ", words=("unstated",)) == 1.3  # the CLI's question took nan before
    out = capsys.readouterr().out
    assert "'nan' is not a number or 'unstated'; try again." in out and "Please enter a number or 'unstated'." in out
