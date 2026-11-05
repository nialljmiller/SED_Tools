"""Paths and input helpers shared by the cube, grids and ML tests."""

from pathlib import Path


KURUCZ = Path(__file__).parent / "golden" / "inputs" / "stellar_models" / "Kurucz2003all"


GOLDEN_MODELS = Path(__file__).resolve().parent / "golden" / "inputs" / "stellar_models"


def _feed_input(monkeypatch, answers):
    feed = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(feed))
