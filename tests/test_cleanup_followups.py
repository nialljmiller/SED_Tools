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


# ----------------------------------------------------------------------
# 2. mesa_prepare: every failure exits non-zero
# ----------------------------------------------------------------------
def write_lookup(model_dir, header, rows):
    model_dir.mkdir(parents=True)
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    (model_dir / "lookup_table.csv").write_text("\n".join(lines) + "\n")


@pytest.fixture
def variant_grid(tmp_path, cube_writer):
    """``<base>/Grid`` with two alpha variants, each with its spectra and cube; returns base."""
    base = tmp_path / "stellar"
    grid = base / "Grid"
    write_lookup(grid, ["file_name", "teff", "logg", "metallicity", "alpha"],
                 [("a.txt", 5000, 4.0, 0.0, "0.0"), ("b.txt", 5000, 4.0, 0.0, "0.4")])
    for name in ("a.txt", "b.txt"):
        (grid / name).write_text("4000 1.0\n5000 2.0\n")
    for alpha in ("0.0", "0.4"):
        cube_writer(grid / "fluxcube_library" / f"flux_cube__alpha_{alpha}.bin",
                    [5000.0], [4.0], [0.0], [4000.0, 5000.0], np.ones((1, 1, 1, 2)))
    return base


def run_mesa_prepare(base, *replies, monkeypatch, **kwargs):
    from sed_tools import mesa_prepare

    answer(monkeypatch, *replies)
    with pytest.raises(SystemExit) as excinfo:
        mesa_prepare.run_interactive(base_dir=str(base), **kwargs)
    return str(excinfo.value.code)


def test_mesa_prepare_exits_1_without_spectral_grids(tmp_path, monkeypatch):
    message = run_mesa_prepare(tmp_path, monkeypatch=monkeypatch)
    assert message.startswith("Error: no spectral grids found in")


def test_mesa_prepare_exits_1_on_an_invalid_grid_choice(variant_grid, monkeypatch):
    message = run_mesa_prepare(variant_grid, "9", monkeypatch=monkeypatch)
    assert message.startswith("Error: invalid spectral grid selection '9'")


def test_mesa_prepare_exits_1_when_the_grid_has_no_variants(tmp_path, monkeypatch):
    # The message it used to print ("the existing flux_cube.bin already covers the
    # full grid") was printed even when no flux_cube.bin had been built.
    write_lookup(tmp_path / "Plain", ["file_name", "teff", "logg", "metallicity"], [("a.txt", 5000, 4.0, 0.0)])
    message = run_mesa_prepare(tmp_path, monkeypatch=monkeypatch, model="Plain", output=str(tmp_path / "out"))
    assert "offers a single variant" in message and "nothing was exported" in message
    assert not (tmp_path / "out").exists()


def test_mesa_prepare_exits_1_on_an_invalid_variant_choice(variant_grid, monkeypatch):
    message = run_mesa_prepare(variant_grid, "3", monkeypatch=monkeypatch, model="Grid",
                               output=str(variant_grid / "out"))
    assert message.startswith("Error: invalid variant selection '3'")
    assert not (variant_grid / "out").exists()


def test_mesa_prepare_exits_1_when_overwrite_is_declined(variant_grid, monkeypatch):
    out = variant_grid / "out"
    out.mkdir()
    (out / "keep.txt").write_text("untouched")
    message = run_mesa_prepare(variant_grid, "1", "n", monkeypatch=monkeypatch, model="Grid", output=str(out))
    assert message == f"Error: {out} already exists and was not overwritten; nothing was exported."
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_mesa_prepare_still_exports_the_chosen_variant(variant_grid, monkeypatch, capsys):
    from sed_tools import mesa_prepare

    answer(monkeypatch, "2")
    out = variant_grid / "out"
    mesa_prepare.run_interactive(base_dir=str(variant_grid), model="Grid", output=str(out))
    assert sorted(p.name for p in out.iterdir()) == ["b.txt", "flux_cube.bin", "lookup_table.csv"]


def test_sed_tools_mesa_prepare_exit_status_is_1_on_failure(tmp_path):
    import subprocess
    from pathlib import Path

    write_lookup(tmp_path / "Plain", ["file_name", "teff", "logg", "metallicity"], [("a.txt", 5000, 4.0, 0.0)])
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-m", "sed_tools.cli", "mesa_prepare", "--base", str(tmp_path),
                           "--grid", "Plain", "--output", str(tmp_path / "out")],
                          cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert done.returncode == 1 and "offers a single variant" in done.stderr
