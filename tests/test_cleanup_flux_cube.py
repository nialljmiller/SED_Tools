"""Failures in the flux-cube code raise or warn instead of passing silently.

Each test covers one handler or skip path changed by the flux-cube cleanup;
see the cleanup ledger for the old behaviour.
"""

from __future__ import annotations

import importlib
import io
import logging

import numpy as np
import pytest

from sed_tools import mesa_prepare
from sed_tools._flux import FluxCube

# sed_tools re-exports the function precompute_flux_cube under the module's name.
pcf = importlib.import_module("sed_tools.precompute_flux_cube")

WAVELENGTHS = np.linspace(1000.0, 10000.0, 20)


def _write_grid(model_dir, rows, columns=("file_name", "teff", "logg", "metallicity"),
                *, missing=(), lookup_name="lookup_table.csv"):
    """Write one flat spectrum per row (except those in ``missing``) and a lookup table."""
    model_dir.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for row in rows:
        name = dict(zip(columns, row))["file_name"]
        if name not in missing:
            (model_dir / name).write_text("".join(f"{w:.1f} {1e-10:.6e}\n" for w in WAVELENGTHS))
        lines.append(",".join(str(v) for v in row))
    (model_dir / lookup_name).write_text("\n".join(lines) + "\n")
    return model_dir


def _simple_grid(model_dir, **kwargs):
    return _write_grid(model_dir, [
        ("a.txt", 5000.0, 4.0, 0.0),
        ("b.txt", 6000.0, 4.0, 0.0),
        ("c.txt", 5000.0, 4.5, 0.0),
        ("d.txt", 6000.0, 4.5, 0.0),
    ], **kwargs)


# ---------------------------------------------------------------- lookup values

def test_nan_logg_and_metallicity_raise_instead_of_an_all_zero_cube(tmp_path):
    # regenerate_lookup_table writes "nan" when a header lacks logg/[M/H] (e.g.
    # blackbodies); every spectrum used to be skipped and an all-zero cube written.
    model = _write_grid(tmp_path / "bbody", [
        ("bb1.txt", 5000.0, "nan", "nan"),
        ("bb2.txt", 6000.0, "nan", "nan"),
    ])
    with pytest.raises(ValueError, match=r"logg = 'nan' for bb1\.txt is not a finite number"):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert not (model / "flux_cube.bin").exists()


def test_blank_teff_raises_naming_the_file(tmp_path):
    model = _write_grid(tmp_path / "grid", [
        ("a.txt", 5000.0, 4.0, 0.0),
        ("b.txt", "", 4.0, 0.0),
    ])
    with pytest.raises(ValueError, match=r"teff = '' for b\.txt"):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert not (model / "flux_cube.bin").exists()


def test_non_numeric_value_in_a_grouped_grid_raises(tmp_path):
    # Grids with extra axes used their own parser, which turned bad values into 0.0.
    model = _write_grid(tmp_path / "grid", [
        ("a.txt", 5000.0, 4.0, 0.0, "0.0"),
        ("b.txt", 5000.0, "high", 0.0, "0.4"),
    ], columns=("file_name", "teff", "logg", "metallicity", "alpha"))
    with pytest.raises(ValueError, match=r"logg = 'high' for b\.txt"):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert not list(model.glob("*/flux_cube.bin"))


def test_missing_teff_column_raises(tmp_path):
    model = _write_grid(tmp_path / "grid", [
        ("a.txt", 4.0, 0.0),
        ("b.txt", 4.5, 0.0),
    ], columns=("file_name", "logg", "metallicity"))
    with pytest.raises(ValueError, match="has no Teff column"):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))


def test_missing_logg_and_metallicity_columns_still_read_as_zero(tmp_path):
    # Unchanged convention: MESA's bbody lookup has only file_name and teff.
    model = _write_grid(tmp_path / "bbody", [
        ("bb1.txt", 5000.0),
        ("bb2.txt", 6000.0),
    ], columns=("file_name", "teff"))
    pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    cube = FluxCube.from_file(str(model / "flux_cube.bin"))
    assert list(cube.teff_grid) == [5000.0, 6000.0]
    assert list(cube.logg_grid) == [0.0]
    assert list(cube.meta_grid) == [0.0]
    assert np.all(cube.flux > 0)


# ---------------------------------------------------------------- spectra

def test_missing_spectra_raise_listing_every_file(tmp_path):
    model = _simple_grid(tmp_path / "grid", missing=("b.txt", "d.txt"))
    with pytest.raises(FileNotFoundError, match="2 spectra listed") as excinfo:
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert "b.txt" in str(excinfo.value) and "d.txt" in str(excinfo.value)
    assert not (model / "flux_cube.bin").exists()


def test_empty_spectrum_raises_and_leaves_no_cube_or_temp_file(tmp_path, monkeypatch):
    model = _simple_grid(tmp_path / "grid")
    (model / "c.txt").write_text("# no data\n")
    # Force the disk-backed cube so the temporary memmap path is exercised.
    monkeypatch.setattr(pcf, "_available_ram_bytes", lambda: 1)
    with pytest.raises(ValueError, match="1 spectra hold no numeric data") as excinfo:
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert "c.txt" in str(excinfo.value)
    assert not (model / "flux_cube.bin").exists()
    assert not (model / "flux_cube.bin.tmp.memmap").exists()


# ---------------------------------------------------------------- split strategy

def _two_variant_grid(model_dir):
    return _write_grid(model_dir, [
        ("a0.txt", 5000.0, 4.0, 0.0, "0.0"),
        ("b0.txt", 6000.0, 4.0, 0.0, "0.0"),
        ("a4.txt", 5000.0, 4.0, 0.0, "0.4"),
        ("b4.txt", 6000.0, 4.0, 0.0, "0.4"),
    ], columns=("file_name", "teff", "logg", "metallicity", "alpha"))


def test_non_interactive_split_over_threshold_raises(tmp_path, monkeypatch):
    model = _two_variant_grid(tmp_path / "Grid")
    monkeypatch.setattr(pcf, "SPLIT_THRESHOLD", 1)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    with pytest.raises(RuntimeError, match="exceed the auto-split threshold"):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert not list(model.glob("Grid_*"))


def test_symlink_failure_warns_and_still_builds_the_subgrid_cube(tmp_path, monkeypatch, caplog):
    model = _two_variant_grid(tmp_path / "Grid")

    def refuse(src, dst):
        raise OSError("symlinks not permitted")

    monkeypatch.setattr(pcf.os, "symlink", refuse)
    with caplog.at_level(logging.WARNING, logger=pcf.__name__):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Could not link 2 spectra" in m and "symlinks not permitted" in m for m in messages)
    assert (model / "Grid_alpha_0p00" / "flux_cube.bin").exists()
    assert (model / "Grid_alpha_0p40" / "flux_cube.bin").exists()


# ---------------------------------------------------------------- fallbacks

def test_ram_probe_failure_warns_and_assumes_2_gib(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(pcf, "psutil", None)
    monkeypatch.setattr(pcf, "_MEMINFO", str(tmp_path / "no_meminfo"))
    with caplog.at_level(logging.WARNING, logger=pcf.__name__):
        assert pcf._available_ram_bytes() == 2 * 1024**3
    assert "Could not measure available RAM" in caplog.text
    assert "psutil is not installed" in caplog.text


def test_meminfo_without_memavailable_warns(monkeypatch, caplog, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  1000 kB\n")
    monkeypatch.setattr(pcf, "psutil", None)
    monkeypatch.setattr(pcf, "_MEMINFO", str(meminfo))
    with caplog.at_level(logging.WARNING, logger=pcf.__name__):
        assert pcf._available_ram_bytes() == 2 * 1024**3
    assert "has no MemAvailable line" in caplog.text


def test_meminfo_is_read_when_psutil_is_missing(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  1000 kB\nMemAvailable:  512 kB\n")
    monkeypatch.setattr(pcf, "psutil", None)
    monkeypatch.setattr(pcf, "_MEMINFO", str(meminfo))
    assert pcf._available_ram_bytes() == 512 * 1024


def test_other_lookup_table_file_is_used_with_a_warning(tmp_path, caplog):
    model = _simple_grid(tmp_path / "grid", lookup_name="lookup_table_full.csv")
    with caplog.at_level(logging.WARNING, logger=pcf.__name__):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert "using" in caplog.text and "lookup_table_full.csv" in caplog.text
    assert (model / "flux_cube.bin").exists()


# ---------------------------------------------------------------- mesa_prepare

def test_list_variants_raises_on_a_truncated_flux_cube(tmp_path):
    # An unreadable cube header used to be logged at debug level and shown as n/a.
    model = _simple_grid(tmp_path / "grid")
    (model / "flux_cube.bin").write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="Flux cube header truncated"):
        mesa_prepare.list_variants(model)


def test_list_variants_without_a_built_cube_shows_no_wavelengths(tmp_path):
    # Unchanged: a grid whose cube has not been built yet lists with n/a wavelengths.
    model = _simple_grid(tmp_path / "grid")
    (variant,) = mesa_prepare.list_variants(model)
    assert variant.wavelength_n == 0 and np.isnan(variant.wavelength_min)


def test_export_variant_raises_on_missing_spectra_before_writing(tmp_path):
    model = _simple_grid(tmp_path / "grid", missing=("c.txt",))
    (variant,) = mesa_prepare.list_variants(model)
    out = tmp_path / "export"
    with pytest.raises(FileNotFoundError, match="1 spectra of variant 'full_grid' are missing") as excinfo:
        mesa_prepare.export_variant(variant, out)
    assert "c.txt" in str(excinfo.value)
    assert not out.exists()


def test_run_interactive_exits_non_zero_when_the_grid_has_no_lookup(tmp_path):
    (tmp_path / "Empty").mkdir()
    with pytest.raises(SystemExit) as excinfo:
        mesa_prepare.run_interactive(base_dir=str(tmp_path), model="Empty", output=str(tmp_path / "out"))
    assert "lookup_table.csv not found" in str(excinfo.value.code)
