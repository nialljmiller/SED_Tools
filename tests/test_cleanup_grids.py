"""Tests for the grids-area cleanup: every handler that now raises or warns.

Covers config, grid_densifier, grid_coverage, spectra_cleaner and
combine_stellar_atm. No test touches the network or the user's real
~/.sed_tools directory.
"""

import pytest

import sed_tools.config as config


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------

def test_load_toml_missing_file_is_empty_unless_strict(tmp_path):
    missing = tmp_path / "absent.toml"
    assert config.load_toml(missing) == {}
    with pytest.raises(FileNotFoundError):
        config.load_toml(missing, strict=True)


def test_load_toml_malformed_file_raises_naming_it(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('data_dir = "unterminated\n')
    for strict in (False, True):
        with pytest.raises(ValueError, match="bad.toml"):
            config.load_toml(bad, strict=strict)


def test_malformed_user_config_stops_data_dir_lookup(tmp_path, monkeypatch):
    # Used to fall back silently to the default data directory.
    bad = tmp_path / "config.toml"
    bad.write_text("data_dir = \n")
    monkeypatch.setattr(config, "_CONFIG_FILE", bad)
    monkeypatch.delenv("SED_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="config.toml"):
        config.get_data_dir()


def test_move_data_removes_drained_dirs_and_keeps_conflicts(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "filters" / "A").mkdir(parents=True)
    (old / "filters" / "A" / "same.dat").write_text("x")
    (old / "stellar_models" / "B").mkdir(parents=True)
    (old / "stellar_models" / "B" / "clash.txt").write_text("old")
    (new / "filters" / "A").mkdir(parents=True)
    (new / "filters" / "A" / "same.dat").write_text("x")
    (new / "stellar_models" / "B").mkdir(parents=True)
    (new / "stellar_models" / "B" / "clash.txt").write_text("new")

    conflicts = config._move_data(old, new)

    assert not (old / "filters").exists()
    assert (old / "stellar_models" / "B" / "clash.txt").read_text() == "old"
    assert (new / "stellar_models" / "B" / "clash.txt").read_text() == "new"
    assert conflicts == [(str(old / "stellar_models" / "B" / "clash.txt"),
                          str(new / "stellar_models" / "B" / "clash.txt"))]


# ---------------------------------------------------------------------
# grid_coverage
# ---------------------------------------------------------------------

def test_terminal_coverage_reports_missing_representative_spectra(tmp_path, monkeypatch, capsys):
    # A representative spectrum missing from disk used to be dropped silently.
    from sed_tools.grid_coverage import grid_coverage

    (tmp_path / "cool.txt").write_text("# teff = 4000\n# logg = 3.0\n1000 1.0\n2000 2.0\n")
    (tmp_path / "lookup_table.csv").write_text(
        "teff,logg,metallicity,filename\n4000,3.0,0.0,cool.txt\n8000,5.0,0.0,gone.txt\n"
    )
    monkeypatch.setattr(config, "get_ui_setting",
                        lambda name: "always" if name == "terminal_plots" else "never")
    grid_coverage(tmp_path, plot=False)
    output = capsys.readouterr().out
    assert "Representative SEDs" in output
    assert "spectrum file not found" in output
    assert "gone.txt" in output


# ---------------------------------------------------------------------
# grid_densifier
# ---------------------------------------------------------------------

def _write_small_cube(path, teff=(5000.0, 7000.0), nw=40):
    import struct

    import numpy as np

    teff = np.asarray(teff, dtype=np.float64)
    logg, meta = np.array([4.0]), np.array([0.0])
    wl = np.linspace(3000.0, 10000.0, nw)
    flux = np.ones((len(teff), 1, 1, nw))
    with open(path, "wb") as fh:
        fh.write(struct.pack("4i", len(teff), 1, 1, nw))
        for arr in (teff, logg, meta, wl, flux):
            fh.write(arr.astype(np.float64).tobytes())


def test_densify_stops_when_ml_generation_fails(tmp_path, monkeypatch):
    # Used to log a warning and write a blackbody in place of the ML SED.
    import sys
    import types

    from sed_tools.grid_densifier import densify_grid

    class FailingGenerator:
        def __init__(self, path):
            pass

        def generate(self, **kwargs):
            raise ValueError("network produced NaN")

    fake = types.ModuleType("sed_tools.ml_sed_generator")
    fake.SEDGenerator = FailingGenerator
    monkeypatch.setitem(sys.modules, "sed_tools.ml_sed_generator", fake)

    src = tmp_path / "src.bin"
    _write_small_cube(src)
    with pytest.raises(RuntimeError, match=r"Teff=6000 logg=4\.00 meta=0\.00.*network produced NaN"):
        densify_grid(str(src), str(tmp_path / "dense.bin"), teff_spacing=1000.0, ml_model="fake")


@pytest.mark.parametrize("answer", ["0", "-1", "3", "x"])
def test_densifier_menu_rejects_out_of_range_choice(tmp_path, monkeypatch, capsys, answer):
    # "0" and negative numbers used to pick cubes from the end of the list.
    import sed_tools.grid_densifier as gd

    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        _write_small_cube(tmp_path / name / "flux_cube.bin")
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    monkeypatch.setattr(gd, "densify_grid", lambda **kw: pytest.fail("must not run"))
    gd.run_interactive_workflow(str(tmp_path))
    assert "Invalid selection." in capsys.readouterr().out
