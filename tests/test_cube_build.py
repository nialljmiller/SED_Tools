"""Building and reading flux cubes. Failures raise or warn instead of passing silently: lookup
columns, missing or empty spectra, the RAM probe, variants, Vega zero points and the flux-cube tool's
exit status. ML-extended and combined grids build one cube.
"""

from __future__ import annotations

import importlib
import io
import logging
import shutil

import numpy as np
import pytest

from sed_tools import _constants, mesa_prepare
from sed_tools._flux import FluxCube
from sed_tools.api import _build_flux_cube, _flux_cube_shortfall
from sed_tools.precompute_flux_cube import spectra_not_in_cube
from sed_tools.svo_regen_spectra_lookup import regenerate_lookup_table

from helpers_grids import GRID, KURUCZ, copy_without_times


# sed_tools re-exports the function precompute_flux_cube under the module's name.
pcf = importlib.import_module("sed_tools.precompute_flux_cube")

WAVELENGTHS = np.linspace(1000.0, 10000.0, 20)
COLUMNS = ("file_name", "teff", "logg", "metallicity")


def _write_grid(model_dir, rows, columns=COLUMNS, *, missing=(), lookup_name="lookup_table.csv"):
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
@pytest.mark.parametrize("rows, columns, message", [
    # regenerate_lookup_table writes "nan" when a header lacks logg/[M/H] (e.g.
    # blackbodies); every spectrum used to be skipped and an all-zero cube written.
    pytest.param([("bb1.txt", 5000.0, "nan", "nan"), ("bb2.txt", 6000.0, "nan", "nan")], COLUMNS,
                 r"logg = 'nan' for bb1\.txt is not a finite number", id="nan_logg_and_metallicity"),
    pytest.param([("a.txt", 5000.0, 4.0, 0.0), ("b.txt", "", 4.0, 0.0)], COLUMNS, r"teff = '' for b\.txt",
                 id="blank_teff"),
    pytest.param([("a.txt", 4.0, 0.0), ("b.txt", 4.5, 0.0)], ("file_name", "logg", "metallicity"),
                 "has no Teff column", id="missing_teff_column"),
])
def test_unusable_lookup_values_raise_and_write_no_cube(tmp_path, rows, columns, message):
    model = _write_grid(tmp_path / "grid", rows, columns)
    with pytest.raises(ValueError, match=message):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert not (model / "flux_cube.bin").exists()


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
# Available RAM, measured by _constants.available_memory_bytes: psutil if installed,
# else MemAvailable in /proc/meminfo, else half the physical RAM, else 2 GiB.
# The golden venv has psutil, so every test sets _constants.psutil itself.

def _without_psutil(monkeypatch, meminfo):
    monkeypatch.setattr(_constants, "psutil", None)
    monkeypatch.setattr(_constants, "_MEMINFO", str(meminfo))


def _fake_sysconf(monkeypatch, page_size, pages):
    values = {"SC_PAGE_SIZE": page_size, "SC_PHYS_PAGES": pages}
    monkeypatch.setattr(_constants.os, "sysconf", values.__getitem__)


@pytest.mark.parametrize("meminfo_text, available", [
    pytest.param("MemTotal:  1000 kB\nMemAvailable:  512 kB\n", 512 * 1024, id="memavailable"),
    pytest.param("MemTotal:  1000 kB\n", 4096 * 1000 // 2, id="half_the_physical_ram_without_memavailable"),
])
def test_meminfo_is_read_when_psutil_is_missing(monkeypatch, tmp_path, caplog, meminfo_text, available):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(meminfo_text)
    _without_psutil(monkeypatch, meminfo)
    _fake_sysconf(monkeypatch, 4096, 1000)
    with caplog.at_level(logging.WARNING, logger=_constants.__name__):
        assert _constants.available_memory_bytes() == available
    assert not caplog.records


def test_ram_probe_failure_warns_and_assumes_2_gib(monkeypatch, caplog, tmp_path):
    _without_psutil(monkeypatch, tmp_path / "no_meminfo")
    _fake_sysconf(monkeypatch, 4096, -1)
    with caplog.at_level(logging.WARNING, logger=_constants.__name__):
        assert _constants.available_memory_bytes() == 2 * 1024**3
    assert "Could not measure available RAM" in caplog.text
    assert "psutil is not installed" in caplog.text
    assert "no_meminfo could not be read" in caplog.text
    assert "SC_PHYS_PAGES is -1" in caplog.text


def test_other_lookup_table_file_is_not_used(tmp_path):
    # Until 0.6.0 any lookup_table*.csv was used, after a warning, when lookup_table.csv was missing.
    model = _simple_grid(tmp_path / "grid", lookup_name="lookup_table_full.csv")
    with pytest.raises(FileNotFoundError, match="lookup_table.csv not found in"):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert not (model / "flux_cube.bin").exists()


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


# ---------------------------------------------------------------- Vega zero points
def _evaluated_sed(tmp_path):
    from sed_tools.models import SEDModel

    model = _simple_grid(tmp_path / "grid")
    pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    return SEDModel(name="grid", flux_cube_path=model / "flux_cube.bin")(5500.0, 4.25, 0.0)


def _filter_file(path, header_lines=()):
    lines = list(header_lines) + [f"{w:.1f} {t}" for w, t in
                                  [(3000.0, 0.0), (4000.0, 1.0), (5000.0, 1.0), (6000.0, 0.0)]]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_non_numeric_vega_zero_point_in_a_filter_file_raises(tmp_path):
    # It used to be skipped, so the magnitude silently came from the Vega spectrum instead.
    sed = _evaluated_sed(tmp_path)
    band = _filter_file(tmp_path / "Band.dat", ["# zp_vega = 3.6e-9 erg/s/cm2/A", "# WavelengthUnit = Angstrom"])
    vega = _filter_file(tmp_path / "vega.dat", ["# WavelengthUnit = Angstrom"])
    with pytest.raises(ValueError, match=r"Filter 'Band': metadata zp_vega = '3.6e-9 erg/s/cm2/A' is not a number"):
        sed.photometry(band, system="Vega", vega_spectrum=vega)


# ---------------------------------------------------------------- flux_cube_tool
def _run_tool(tmp_path, monkeypatch, filters_dir):
    from sed_tools import flux_cube_tool

    model = _simple_grid(tmp_path / "grid")
    pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    monkeypatch.setattr("builtins.input", lambda *args: "")  # no distance scaling
    return flux_cube_tool.main([
        "--flux-cube", str(model / "flux_cube.bin"),
        "--teff", "5500", "--logg", "4.25", "--metallicity", "0",
        "--filters", str(filters_dir), "--filter-system", "all",
        "--photometric-system", "AB", "--plot", str(tmp_path / "sed.png"),
    ])


def test_flux_cube_tool_exits_non_zero_when_a_filter_fails(tmp_path, monkeypatch, capsys):
    # A failing filter used to print an error and the tool still exited 0.
    filters = tmp_path / "filters"
    filters.mkdir()
    _filter_file(filters / "Good.dat", ["# WavelengthUnit = Angstrom"])
    (filters / "Dark.dat").write_text("# WavelengthUnit = Angstrom\n4000.0 0.0\n5000.0 0.0\n")
    assert _run_tool(tmp_path, monkeypatch, filters) == 1
    out = capsys.readouterr().out
    assert "Photometry failed for 1 filter(s):" in out
    assert "Dark: Filter transmission integrates to zero" in out
    assert "Good" in out and "mag=" in out


def _unreadable_rglob(self, pattern):
    raise PermissionError(13, "Permission denied", str(self))


@pytest.mark.parametrize("search, what", [("discover_vega_candidates", "Vega spectra"),
                                          ("discover_flux_cube_files", "flux cubes")])
def test_a_search_warns_when_a_directory_cannot_be_searched(tmp_path, monkeypatch, caplog, search, what):
    from sed_tools import flux_cube_tool

    monkeypatch.setattr(flux_cube_tool.Path, "rglob", _unreadable_rglob)
    with caplog.at_level(logging.WARNING, logger=flux_cube_tool.__name__):
        assert getattr(flux_cube_tool, search)([tmp_path]) == []
    assert f"Could not finish searching {tmp_path.resolve()} for {what}" in caplog.text


ALPHA_04 = KURUCZ.parent / f"{GRID}__alpha_04"


# ----------------------------------------------------------------------
# Out of date by content, not by file times
# ----------------------------------------------------------------------
def variants_grid(folder):
    """The test grid at [alpha/M] 0 and 0.4 in one folder: built as one flux cube per variant."""
    folder.mkdir(parents=True)
    for source in (KURUCZ, ALPHA_04):
        for spectrum in source.glob("*.txt"):
            shutil.copy2(spectrum, folder / spectrum.name)
    regenerate_lookup_table(str(folder))
    assert _build_flux_cube(folder).name == "variants_index.csv"
    return folder


def test_a_variants_grid_is_compared_variant_by_variant(tmp_path):
    folder = variants_grid(tmp_path / "a" / GRID)
    assert _flux_cube_shortfall(folder) is None
    assert _flux_cube_shortfall(copy_without_times(folder, tmp_path / "b" / GRID)) is None

    # One more spectrum, at a Teff no variant's cube has
    text = (KURUCZ / "Kurucz2003all_fid16515.txt").read_text().replace("# teff = 5750.0", "# teff = 6500.0")
    (folder / "Kurucz2003all_fid99999.txt").write_text(text)
    regenerate_lookup_table(str(folder))
    assert _flux_cube_shortfall(folder) == "does not include 1 of the 25 spectra beside it"


def test_a_spectrum_zero_at_every_sampled_wavelength_is_still_found(tmp_path, cube_writer):
    """A spectrum covering only part of the cube's wavelengths is zero elsewhere (resample_to_grid)."""
    wl = np.linspace(1000.0, 20000.0, 20)  # the sampled planes are 2, 4, 6, 8, 11, 13, 15, 17
    flux = np.zeros((3, 1, 1, 20))
    flux[0, 0, 0] = 1.0                    # covers every wavelength
    flux[1, 0, 0, [0, 19]] = 1.0           # only at wavelengths no sampled plane has
    cube_writer(tmp_path / "flux_cube.bin", [5000.0, 6000.0, 7000.0], [4.0], [0.0], wl, flux)
    (tmp_path / "lookup_table.csv").write_text(
        "#file_name,teff,logg,metallicity\na.txt,5000,4.0,0.0\nb.txt,6000,4.0,0.0\nc.txt,7000,4.0,0.0\n")
    names, missing = spectra_not_in_cube(tmp_path / "flux_cube.bin", tmp_path / "lookup_table.csv")
    assert names == ["a.txt", "b.txt", "c.txt"] and missing == ["c.txt"]  # 7000 K: zero throughout


# ----------------------------------------------------------------------
# Messages and folders
# ----------------------------------------------------------------------
def test_the_densifier_line_fits_80_columns(tmp_path, cube_writer, capsys):
    from sed_tools.grid_densifier import densify_grid

    wl = np.linspace(3000.0, 9000.0, 7)
    cube_writer(tmp_path / "src.bin", [4000.0, 8000.0], [4.0], [0.0], wl, np.ones((2, 1, 1, 7)))
    densify_grid(str(tmp_path / "src.bin"), str(tmp_path / "dense.bin"), teff_spacing=1000.0)
    lines = capsys.readouterr().out.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("Generating"))
    assert lines[at:at + 2] == ["Generating 3 synthetic Teff points (0 by ML, 3 blackbody)",
                                "  in 1 wavelength block(s)..."]


# ----------------------------------------------------------------------
# Provenance columns of combined and ML-extended grids are not axes
# ----------------------------------------------------------------------
def _spectrum(path, wavelengths):
    path.write_text("".join(f"{w:.1f} {1e-10:.6e}\n" for w in wavelengths))


def _extended_grid(model_dir):
    """An ML-extended grid: each spectrum was extended to a different longest wavelength."""
    model_dir.mkdir(parents=True)
    rows = []
    for i, (teff, logg) in enumerate([(5000.0, 4.0), (6000.0, 4.0), (5000.0, 4.5), (6000.0, 4.5)]):
        wl_max = 20000.0 + 5000.0 * i
        _spectrum(model_dir / f"s{i}.txt", np.linspace(1000.0, wl_max, 30))
        rows.append(f"s{i}.txt,{teff},{logg},0.0,True,1000.0,{wl_max}")
    (model_dir / "lookup_table.csv").write_text(
        "#file_name,teff,logg,metallicity,extended_by_ml,wl_min,wl_max\n" + "\n".join(rows) + "\n")
    return model_dir


def test_an_ml_extended_grid_with_different_wavelength_ends_is_one_cube(tmp_path, capsys):
    # Each row's wl_max differs, so these columns used to split the grid into
    # one "variant" per spectrum, with no flux_cube.bin in the grid itself.
    grid = _extended_grid(tmp_path / "Kurucz_extended")
    pcf.precompute_flux_cube(str(grid), str(grid / "flux_cube.bin"))
    assert (grid / "flux_cube.bin").is_file()
    assert not (grid / "variants_index.csv").exists()
    assert not [p for p in grid.iterdir() if p.is_dir()]
    assert "No extra axes detected" in capsys.readouterr().out
    # A file column under another name (the first column is taken) gave one subgrid per spectrum until 0.6.0
    (grid / "lookup_table.csv").write_text((grid / "lookup_table.csv").read_text().replace("#file_name,", "#spectrum,"))
    pcf.precompute_flux_cube(str(grid), str(grid / "flux_cube.bin"))
    assert not [p for p in grid.iterdir() if p.is_dir()] and "No extra axes detected" in capsys.readouterr().out
