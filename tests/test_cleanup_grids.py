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


# ---------------------------------------------------------------------
# spectra_cleaner: HDF5 wavelength recovery (characterisation)
# ---------------------------------------------------------------------

def _h5(path, build):
    import h5py

    with h5py.File(path, "w") as f:
        build(f)
    return path


def test_recovery_reads_direct_dataset_in_spec_group(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    wl = np.linspace(3000.0, 9000.0, 12)
    _h5(tmp_path / "msg.h5", lambda f: f.create_group("g1").create_dataset("lambda", data=wl))
    np.testing.assert_array_equal(recover_wavelengths_from_hdf5(str(tmp_path), "g1", 10), wl[:10])


def test_recovery_reads_range_x_then_segments(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    def build(f):
        rg = f.create_group("g1").create_group("range")
        rg.create_group("range2").create_dataset("wave", data=np.linspace(6000.0, 9000.0, 5))
        rg.create_group("range1").create_dataset("wave", data=np.linspace(3000.0, 5000.0, 5))
        g2 = f.create_group("g2").create_group("range")
        g2.create_dataset("x", data=np.linspace(2000.0, 4000.0, 8))

    _h5(tmp_path / "msg.h5", build)
    seg = recover_wavelengths_from_hdf5(str(tmp_path), "g1", 10)
    np.testing.assert_array_equal(seg, np.concatenate([np.linspace(3000.0, 5000.0, 5),
                                                       np.linspace(6000.0, 9000.0, 5)]))
    np.testing.assert_array_equal(recover_wavelengths_from_hdf5(str(tmp_path), "g2", 8),
                                  np.linspace(2000.0, 4000.0, 8))


def test_recovery_top_level_array_within_two_points(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    wl = np.linspace(3000.0, 9000.0, 11)
    _h5(tmp_path / "msg.h5", lambda f: f.create_dataset("wavelength", data=wl))
    np.testing.assert_array_equal(recover_wavelengths_from_hdf5(str(tmp_path), "absent", 10), wl[:10])
    assert recover_wavelengths_from_hdf5(str(tmp_path), "absent", 20) is None


def test_recovery_global_search_skips_non_numeric_datasets(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    wl = np.geomspace(1000.0, 50000.0, 9)

    def build(f):
        f.create_dataset("labels", data=np.array([b"a", b"b"]))
        f.create_dataset("records", data=np.zeros(9, dtype=[("a", "f8"), ("b", "i4")]))
        f.create_group("deep").create_group("er").create_dataset("grid", data=wl)

    _h5(tmp_path / "msg.h5", build)
    np.testing.assert_array_equal(recover_wavelengths_from_hdf5(str(tmp_path), "absent", 9), wl)


def test_recovery_rejects_index_like_arrays(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    _h5(tmp_path / "msg.h5", lambda f: f.create_group("g1").create_dataset("lambda", data=np.arange(10.0)))
    assert recover_wavelengths_from_hdf5(str(tmp_path), "g1", 10) is None


def test_recovery_warns_about_unreadable_h5(tmp_path, caplog):
    # An unreadable .h5 used to be skipped with a DEBUG message only.
    import logging

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    (tmp_path / "broken.h5").write_bytes(b"not an hdf5 file")
    with caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        assert recover_wavelengths_from_hdf5(str(tmp_path), "g1", 10) is None
    assert "broken.h5" in caplog.text


def test_recovery_still_uses_a_good_h5_beside_a_broken_one(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import recover_wavelengths_from_hdf5

    (tmp_path / "broken.h5").write_bytes(b"not an hdf5 file")
    wl = np.linspace(3000.0, 9000.0, 10)
    _h5(tmp_path / "good.h5", lambda f: f.create_group("g1").create_dataset("lambda", data=wl))
    np.testing.assert_array_equal(recover_wavelengths_from_hdf5(str(tmp_path), "g1", 10), wl)


# ---------------------------------------------------------------------
# spectra_cleaner: clean_model_dir must not report success over
# spectra it left in their original units
# ---------------------------------------------------------------------

def _spectrum(path, teff=5000.0, unit_line="# wavelength_unit = nm\n", wl=None, flux=None):
    import numpy as np

    wl = np.linspace(400.0, 700.0, 30) if wl is None else wl
    flux = np.full(len(wl), 1e-14) if flux is None else flux
    lines = [f"# source = testgrid\n# teff = {teff}\n# logg = 4.0\n# metallicity = 0.0\n", unit_line]
    lines += [f"{w:.6f} {f:.8e}\n" for w, f in zip(wl, flux)]
    path.write_text("".join(lines))


def _is_standardized(path):
    return "units_standardized = True" in path.read_text()


def test_clean_model_dir_raises_listing_unconverted_spectra(tmp_path):
    # Used to return a summary as if the grid were clean, leaving these two
    # files in their original units beside the converted ones.
    import numpy as np

    from sed_tools.spectra_cleaner import clean_model_dir

    for i in range(3):
        _spectrum(tmp_path / f"good_{i}.txt", teff=4000.0 + 1000 * i)
    (tmp_path / "empty.txt").write_text("# teff = 5000\n")
    _spectrum(tmp_path / "index.txt", unit_line="", wl=np.arange(30.0))

    with pytest.raises(RuntimeError) as err:
        clean_model_dir(str(tmp_path), sample_fraction=1.0)

    message = str(err.value)
    assert "2 of 5 spectra" in message
    assert "empty data (1): empty.txt" in message
    assert "index grid without HDF5 recovery (1): index.txt" in message
    assert "lookup_table.csv was rebuilt" in message
    assert all(_is_standardized(tmp_path / f"good_{i}.txt") for i in range(3))
    assert not _is_standardized(tmp_path / "empty.txt")
    assert (tmp_path / "lookup_table.csv").exists()


def test_clean_model_dir_records_unreadable_file_and_finishes_the_rest(tmp_path, monkeypatch):
    import sed_tools.spectra_cleaner as sc

    _spectrum(tmp_path / "a.txt")
    (tmp_path / "b.txt").mkdir()  # matches *.txt but cannot be read as a file
    _spectrum(tmp_path / "c.txt")
    monkeypatch.setattr(sc.random, "sample", lambda files, n: [str(tmp_path / "a.txt")])

    with pytest.raises(RuntimeError, match=r"IsADirectoryError.*\(1\): b\.txt"):
        sc.clean_model_dir(str(tmp_path), rebuild_lookup=False, sample_fraction=0.4)
    assert _is_standardized(tmp_path / "a.txt") and _is_standardized(tmp_path / "c.txt")


def test_clean_model_dir_unreadable_sampled_file_stops_before_any_rewrite(tmp_path):
    # Used to be counted as an invalid sample without saying why.
    from sed_tools.spectra_cleaner import clean_model_dir

    _spectrum(tmp_path / "a.txt")
    (tmp_path / "b.txt").mkdir()

    with pytest.raises(IsADirectoryError, match=r"b\.txt"):
        clean_model_dir(str(tmp_path), sample_fraction=1.0)
    assert not _is_standardized(tmp_path / "a.txt")


def test_clean_model_dir_stops_at_once_when_a_rewrite_fails(tmp_path, monkeypatch):
    # A failed rewrite can leave a header-only file that later runs skip, so
    # the batch must not carry on (it used to record 'error' and continue).
    import sed_tools.spectra_cleaner as sc

    for name in ("a.txt", "b.txt"):
        _spectrum(tmp_path / name)
    calls = []

    def disk_full(filepath, *args):
        calls.append(filepath)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sc, "write_standardized_spectrum", disk_full)
    with pytest.raises(RuntimeError, match=r"a\.txt.*only its new header"):
        sc.clean_model_dir(str(tmp_path), sample_fraction=1.0)
    assert len(calls) == 1


def test_write_standardized_spectrum_length_mismatch_leaves_file_intact(tmp_path):
    import numpy as np

    from sed_tools.spectra_cleaner import SpectrumMeta, UnitInfo, write_standardized_spectrum

    path = tmp_path / "s.txt"
    _spectrum(path)
    before = path.read_text()
    units = UnitInfo("angstrom", 1.0, "flam", 1.0, "high", "header")
    with pytest.raises(ValueError):
        write_standardized_spectrum(str(path), np.arange(1.0, 5.0), np.ones(3), SpectrumMeta(), units)
    assert path.read_text() == before


def test_clean_model_dir_warns_when_units_come_from_each_file(tmp_path, monkeypatch, caplog):
    # The sample happened to hold only standardized files, so the grid-wide
    # units were unknown and the rest were converted one by one, silently.
    import logging

    import sed_tools.spectra_cleaner as sc

    for i in range(9):
        _spectrum(tmp_path / f"raw_{i}.txt")
    done = tmp_path / "done.txt"
    done.write_text("# units_standardized = True\n4000 1.0\n5000 2.0\n")
    monkeypatch.setattr(sc.random, "sample", lambda files, n: [str(done)])

    with caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = sc.clean_model_dir(str(tmp_path), sample_fraction=0.1)
    assert len(summary["converted"]) == 9
    assert "9 spectra were converted using units detected file by file" in caplog.text


# ---------------------------------------------------------------------
# combine_stellar_atm
# ---------------------------------------------------------------------

GOLDEN_MODELS = __import__("pathlib").Path(__file__).parent / "golden" / "inputs" / "stellar_models"


def _copy_grids(tmp_path):
    import shutil

    models = tmp_path / "models"
    for name in ("Kurucz2003all", "Kurucz2003all__alpha_04"):
        shutil.copytree(GOLDEN_MODELS / name, models / name)
    return models


def _append_lookup_row(grid_dir, row):
    with open(grid_dir / "lookup_table.csv", "a") as fh:
        fh.write(row + "\n")


@pytest.mark.parametrize("grid", ["Kurucz2003all", "Kurucz2003all__alpha_04"])
def test_combine_stops_on_a_missing_spectrum(tmp_path, grid):
    # Used to give the spectrum a normalisation factor of 1 (alpha_04) or
    # leave its node empty for nearest-neighbour filling (Kurucz), silently.
    import numpy as np

    from sed_tools.api import SED

    models = _copy_grids(tmp_path)
    _append_lookup_row(models / grid, "missing.txt,0,999.9,2,1.25,4.0,0.0,999.9,999.9,999.9,9000.0,999.9,2,999.9")
    np.random.seed(0)
    with pytest.raises(ValueError, match=r"missing\.txt: fewer than 2 readable"):
        SED.combine(["Kurucz2003all", "Kurucz2003all__alpha_04"], output="combined",
                    model_root=models, visualization=False)
    assert not (models / "combined").exists()


def test_combine_stops_on_a_row_without_parameters(tmp_path):
    # A NaN Teff used to land on grid node [0, ...] and overwrite it.
    import numpy as np

    from sed_tools.api import SED

    models = _copy_grids(tmp_path)
    grid = models / "Kurucz2003all"
    _append_lookup_row(grid, "Kurucz2003all_fid16515.txt,0,999.9,2,1.25,4.0,-0.5,999.9,999.9,999.9,,999.9,2,999.9")
    np.random.seed(0)
    with pytest.raises(ValueError, match=r"non-finite parameters \(teff=nan"):
        SED.combine(["Kurucz2003all", "Kurucz2003all__alpha_04"], output="combined",
                    model_root=models, visualization=False)


def test_combine_raises_after_writing_when_a_copy_fails(tmp_path, monkeypatch):
    # Copy failures used to be printed and the run reported success.
    import shutil

    import numpy as np

    import sed_tools.combine_stellar_atm as csa
    from sed_tools.api import SED

    models = _copy_grids(tmp_path)
    real_copy = shutil.copy2

    def flaky_copy(src, dst):
        if src.endswith("Kurucz2003all_fid16516.txt"):
            raise PermissionError(13, "Permission denied", src)
        return real_copy(src, dst)

    monkeypatch.setattr(csa.shutil, "copy2", flaky_copy)
    np.random.seed(0)
    with pytest.raises(OSError, match=r"(?s)Could not copy 1 spectra.*fid16516"):
        SED.combine(["Kurucz2003all", "Kurucz2003all__alpha_04"], output="combined",
                    model_root=models, visualization=False)
    assert (models / "combined" / "lookup_table.csv").exists()


def test_normalisation_warns_when_spectra_keep_native_scale(tmp_path, caplog):
    import logging

    import numpy as np

    from sed_tools.combine_stellar_atm import compute_normalization_factors, load_model_data

    models = _copy_grids(tmp_path)
    ref = load_model_data(str(models / "Kurucz2003all"))
    target = load_model_data(str(models / "Kurucz2003all__alpha_04"))
    coarse = np.linspace(3000.0, 10000.0, 50)  # fewer than 100 points in any overlap
    with caplog.at_level(logging.WARNING, logger="sed_tools.combine_stellar_atm"):
        factors = compute_normalization_factors(target, ref, coarse)
    np.testing.assert_array_equal(factors, np.ones(len(target["files"])))
    assert f"{len(target['files'])} spectra of Kurucz2003all__alpha_04 keep their native flux scale" in caplog.text


def test_common_wavelength_grid_needs_a_usable_spectrum(tmp_path):
    # Used to fail with OverflowError (int of -inf).
    import numpy as np

    from sed_tools.combine_stellar_atm import create_common_wavelength_grid

    (tmp_path / "far_ir.txt").write_text("".join(f"{w} 1.0\n" for w in np.linspace(3e4, 9e4, 50)))
    data = {"files": np.array(["far_ir.txt"]), "model_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="between 3000 and 25000"):
        create_common_wavelength_grid([data])
