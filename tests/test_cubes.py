"""Flux cubes: reading and building them, their variants and mesa_prepare, combining and densifying grids, the
combine, densifier and flux-cube-tool prompts, extra-axis collisions, interpolation, and synthetic photometry.

One test walks tables of named rows; each row runs as a test of its own (support.Rows).
"""

from __future__ import annotations

import importlib
import io
import logging
import math
import shutil
import struct
import sys
import types
import warnings

import numpy as np
import pytest

from sed_tools import _constants
from sed_tools.api import _build_flux_cube, _flux_cube_shortfall
from sed_tools.cube import mesa_export, tool
from sed_tools.cube.build import resample_to_grid, spectra_not_in_cube
from sed_tools.cube.collision import CollisionConfig, copy_global_config_to_model, load_config, match_filter_value
from sed_tools.cube.densify import densify_grid
from sed_tools.cube.mesa_export import list_variants
from sed_tools.cube.model import SED, SEDModel, _build_filter_index
from sed_tools.cube.tool import FluxCube, bolometric_magnitude
from sed_tools.cube.variants import _extra_key_to_filename
from sed_tools.grids.clean import clean_model_dir
from sed_tools.grids.lookup import regenerate_lookup_table
from sed_tools.photometry import PC_TO_M, RSUN_TO_M

from support import (GOLDEN_MODELS, GRID, KURUCZ, affine_cube, answer, copy_without_times, scenarios, wide_lines,
                     write_flux_cube, write_lookup)

SCENARIOS, row = scenarios()

# sed_tools re-exports the function precompute_flux_cube under the module's name.
pcf = importlib.import_module("sed_tools.cube.build")
ALPHA_04 = f"{GRID}__alpha_04"
COLUMNS = ("file_name", "teff", "logg", "metallicity")
ANGSTROM = "# WavelengthUnit = Angstrom"


# ----------------------------------------------------------------------
# Grids, cubes and filters the rows build
# ----------------------------------------------------------------------
def _write_grid(model_dir, rows, columns=COLUMNS, *, missing=(), lookup_name="lookup_table.csv"):
    """One flat spectrum per row (except those in ``missing``) and a lookup table."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row[0] not in missing:
            (model_dir / row[0]).write_text("".join(f"{w:.1f} {1e-10:.6e}\n" for w in np.linspace(1000.0, 1e4, 20)))
    (model_dir / lookup_name).write_text("\n".join(",".join(map(str, r)) for r in [columns, *rows]) + "\n")
    return model_dir


def _simple_grid(model_dir, **kwargs):
    """Teff 5000 and 6000 K at log g 4.0 and 4.5."""
    return _write_grid(model_dir, [("a.txt", 5000.0, 4.0, 0.0), ("b.txt", 6000.0, 4.0, 0.0),
                                   ("c.txt", 5000.0, 4.5, 0.0), ("d.txt", 6000.0, 4.5, 0.0)], **kwargs)


def _two_variant_grid(model_dir):
    return _write_grid(model_dir, [("a0.txt", 5000.0, 4.0, 0.0, "0.0"), ("b0.txt", 6000.0, 4.0, 0.0, "0.0"),
                                   ("a4.txt", 5000.0, 4.0, 0.0, "0.4"), ("b4.txt", 6000.0, 4.0, 0.0, "0.4")],
                       (*COLUMNS, "alpha"))


def _build(model_dir, **kwargs):
    """precompute_flux_cube into ``model_dir``/flux_cube.bin."""
    pcf.precompute_flux_cube(str(model_dir), str(model_dir / "flux_cube.bin"), **kwargs)
    return model_dir


def _raw_cube(path, teff, logg, meta, wavelengths, body=()):
    """A flux_cube.bin's header and axes, then ``body``: no flux at all unless given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("4i", len(teff), len(logg), len(meta), len(wavelengths)))
        for values in (teff, logg, meta, wavelengths, body):
            np.asarray(values, dtype=np.float64).tofile(fh)
    return path


def _small_cube(path, teff=(5000.0, 7000.0), nw=40):
    write_flux_cube(path, teff, [4.0], [0.0], np.linspace(3000.0, 10000.0, nw), np.ones((len(teff), 1, 1, nw)))


def _box_filter(path, *header, edges=(3000.0, 4000.0, 5000.0, 6000.0)):
    """A filter file: ``header`` lines, then transmission 1 between the middle two edges and 0 at the outer two."""
    path.write_text("".join(f"{line}\n" for line in header)
                    + "".join(f"{w:.1f} {tr}\n" for w, tr in zip(edges, (0.0, 1.0, 1.0, 0.0))))
    return path


def flat_filter(name="V"):
    wavelength = np.arange(4000.0, 6001.0, 100.0)
    return tool.FilterCurve(name, wavelength, np.ones_like(wavelength), {})


def _kurucz_alpha(folder, build=False):
    """Both golden Kurucz grids ([alpha/M] 0 and 0.4) as one grid; lh and vtur are the same throughout. Built,
    it is split into one subgrid per alpha, listed in variants_index.csv."""
    import pandas as pd

    folder.mkdir(parents=True)
    for name in (GRID, ALPHA_04):
        for path in (GOLDEN_MODELS / name).glob("*.txt"):
            shutil.copy2(path, folder / path.name)
    regenerate_lookup_table(str(folder))
    assert sorted(set(pd.read_csv(folder / "lookup_table.csv", dtype=str)[" vtur"])) == ["2 km/s"]  # a ' ' and '/'
    if build:
        assert _build_flux_cube(folder).name == "variants_index.csv"
        assert not (folder / "flux_cube.bin").exists()
    return folder


def _golden_pair(t):
    """The golden Kurucz grids at [alpha/M] 0 and 0.4, side by side in the row's models folder."""
    for name in (GRID, ALPHA_04):
        shutil.copytree(GOLDEN_MODELS / name, t.tmp / "models" / name)
    return t.tmp / "models"


def _combine(models, output="combined"):
    from sed_tools.api import SED  # the API's, not the models' SED

    SED.combine([GRID, ALPHA_04], output=output, model_root=models, visualization=False)


def _collision_grid(model_dir):
    """Two Teff nodes x two alpha variants, in Jansky over 400-700 nm, cleaned: two real collisions."""
    model_dir.mkdir()
    for name, teff, alpha, flux_jy in (("a1.txt", 5000.0, 0.0, 100.0), ("a2.txt", 5000.0, 0.4, 300.0),
                                       ("b1.txt", 6000.0, 0.0, 400.0), ("b2.txt", 6000.0, 0.4, 600.0)):
        (model_dir / name).write_text(
            f"# source = synthetic\n# teff = {teff}\n# logg = 4.0\n# metallicity = 0.0\n# alpha = {alpha}\n"
            "# wavelength_unit = nm\n# flux in Jansky\n"
            + "".join(f"{w:.6f} {flux_jy:.8e}\n" for w in np.linspace(400.0, 700.0, 8)))
    clean_model_dir(str(model_dir))
    return model_dir


# ----------------------------------------------------------------------
# Reading flux_cube.bin: truncated header or body, trailing bytes
# ----------------------------------------------------------------------
@row
def truncated_header_raises_low_level(t):
    (t.tmp / "cube.bin").write_bytes(b"\x00" * 10)  # the header is 16 bytes (4 int32s)
    with pytest.raises(ValueError, match="truncated"):
        FluxCube.from_file(str(t.tmp / "cube.bin"))


@row
def truncated_body_raises_with_expected_and_found_counts(t):
    bad = _raw_cube(t.tmp / "cube.bin", [4000.0, 6000.0], [3.0], [0.0], [4000.0, 5000.0], body=[1.0, 2.0])
    with pytest.raises(ValueError) as excinfo:  # 2 x 1 x 1 x 2 = 4 values, 2 of them written
        FluxCube.from_file(str(bad))
    message = str(excinfo.value)
    assert "4" in message  # expected count
    assert "2" in message  # actual count found


@row
def trailing_bytes_emit_warning_but_do_not_raise(t):
    cube = affine_cube(t.tmp)
    with cube.path.open("ab") as fh:
        fh.write(b"\x00")
    fc = FluxCube.from_file(str(cube.path))  # does not raise
    np.testing.assert_allclose(fc.flux[0, 0, 0, :], cube.expected(cube.teff[0], cube.logg[0], cube.meta[0]),
                               rtol=1e-10)
    assert "trailing data" in t.capsys.readouterr().out.lower()


# ----------------------------------------------------------------------
# Building a cube: lookup values, spectra, the split strategy, the RAM probe
# ----------------------------------------------------------------------
#: name: (lookup rows, their columns, what the ValueError says); no cube is written.
UNUSABLE_LOOKUP = {
    # regenerate_lookup_table writes "nan" when a header lacks logg/[M/H] (blackbodies); every spectrum used to
    # be skipped and an all-zero cube written.
    "unusable_lookup_values_raise_and_write_no_cube_nan_logg_and_metallicity": (
        [("bb1.txt", 5000.0, "nan", "nan"), ("bb2.txt", 6000.0, "nan", "nan")], COLUMNS,
        r"logg = 'nan' for bb1\.txt is not a finite number"),
    "unusable_lookup_values_raise_and_write_no_cube_blank_teff": (
        [("a.txt", 5000.0, 4.0, 0.0), ("b.txt", "", 4.0, 0.0)], COLUMNS, r"teff = '' for b\.txt"),
    "unusable_lookup_values_raise_and_write_no_cube_missing_teff_column": (
        [("a.txt", 4.0, 0.0), ("b.txt", 4.5, 0.0)], ("file_name", "logg", "metallicity"), "has no Teff column"),
}


def _unusable_lookup(t, rows, columns, message):
    model = _write_grid(t.tmp / "grid", rows, columns)
    with pytest.raises(ValueError, match=message):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert not (model / "flux_cube.bin").exists()


#: name: (/proc/meminfo's text, or None when there is none; sysconf's page count; the RAM measured; the warning's
#: words). Without psutil the RAM is MemAvailable, else half the physical RAM, else 2 GiB with a warning.
RAM = {
    "meminfo_is_read_when_psutil_is_missing_memavailable": (
        "MemTotal:  1000 kB\nMemAvailable:  512 kB\n", 1000, 512 * 1024, ()),
    "meminfo_is_read_when_psutil_is_missing_half_the_physical_ram_without_memavailable": (
        "MemTotal:  1000 kB\n", 1000, 4096 * 1000 // 2, ()),
    "ram_probe_failure_warns_and_assumes_2_gib": (None, -1, 2 * 1024**3, (
        "Could not measure available RAM", "psutil is not installed", "no_meminfo could not be read",
        "SC_PHYS_PAGES is -1")),
}


def _ram(t, meminfo, pages, available, warned):
    path = t.tmp / ("meminfo" if meminfo else "no_meminfo")
    if meminfo:
        path.write_text(meminfo)
    t.mp.setattr(_constants, "psutil", None)  # the golden venv has psutil
    t.mp.setattr(_constants, "_MEMINFO", str(path))
    t.mp.setattr(_constants.os, "sysconf", {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": pages}.__getitem__)
    with t.caplog.at_level(logging.WARNING, logger=_constants.__name__):
        assert _constants.available_memory_bytes() == available
    assert [w for w in warned if w not in t.caplog.text] == [] and bool(t.caplog.records) == bool(warned)


#: name: (the flux cube tool's search, what it says it was searching for)
SEARCHES = {
    "a_search_warns_when_a_directory_cannot_be_searched_for_vega_spectra": ("discover_vega_candidates", "Vega spectra"),
    "a_search_warns_when_a_directory_cannot_be_searched_for_flux_cubes": ("discover_flux_cube_files", "flux cubes"),
}


def _unsearchable(t, search, what):
    def unreadable(self, pattern):
        raise PermissionError(13, "Permission denied", str(self))

    t.mp.setattr(tool.Path, "rglob", unreadable)
    with t.caplog.at_level(logging.WARNING, logger=tool.__name__):
        assert getattr(tool, search)([t.tmp]) == []
    assert f"Could not finish searching {t.tmp.resolve()} for {what}" in t.caplog.text


@row
def missing_logg_and_metallicity_columns_still_read_as_zero(t):
    # Unchanged convention: MESA's bbody lookup has only file_name and teff.
    model = _build(_write_grid(t.tmp / "bbody", [("bb1.txt", 5000.0), ("bb2.txt", 6000.0)], ("file_name", "teff")))
    cube = FluxCube.from_file(str(model / "flux_cube.bin"))
    assert (list(cube.teff_grid), list(cube.logg_grid), list(cube.meta_grid)) == ([5000.0, 6000.0], [0.0], [0.0])
    assert np.all(cube.flux > 0)


@row
def missing_spectra_raise_listing_every_file(t):
    model = _simple_grid(t.tmp / "grid", missing=("b.txt", "d.txt"))
    with pytest.raises(FileNotFoundError, match="2 spectra listed") as excinfo:
        _build(model)
    assert "b.txt" in str(excinfo.value) and "d.txt" in str(excinfo.value)
    assert not (model / "flux_cube.bin").exists()


@row
def empty_spectrum_raises_and_leaves_no_cube_or_temp_file(t):
    model = _simple_grid(t.tmp / "grid")
    (model / "c.txt").write_text("# no data\n")
    t.mp.setattr(pcf, "_available_ram_bytes", lambda: 1)  # the disk-backed cube, with its temporary memmap
    with pytest.raises(ValueError, match="1 spectra hold no numeric data") as excinfo:
        _build(model)
    assert "c.txt" in str(excinfo.value)
    assert not (model / "flux_cube.bin").exists() and not (model / "flux_cube.bin.tmp.memmap").exists()


@row
def non_interactive_split_over_threshold_raises(t):
    model = _two_variant_grid(t.tmp / "Grid")
    t.mp.setattr(pcf, "SPLIT_THRESHOLD", 1)
    t.mp.setattr("sys.stdin", io.StringIO())
    with pytest.raises(RuntimeError, match="exceed the auto-split threshold"):
        _build(model)
    assert not list(model.glob("Grid_*"))


@row
def symlink_failure_warns_and_still_builds_the_subgrid_cube(t):
    model = _two_variant_grid(t.tmp / "Grid")

    def refuse(src, dst):
        raise OSError("symlinks not permitted")

    t.mp.setattr(pcf.os, "symlink", refuse)
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.cube.variants"):
        _build(model)
    messages = [r.getMessage() for r in t.caplog.records if r.levelno == logging.WARNING]
    assert any("Could not link 2 spectra" in m and "symlinks not permitted" in m for m in messages)
    assert all((model / f"Grid_alpha_{a}" / "flux_cube.bin").exists() for a in ("0p00", "0p40"))


@row
def other_lookup_table_file_is_not_used(t):
    # Until 0.6.0 any lookup_table*.csv was used, after a warning, when lookup_table.csv was missing.
    model = _simple_grid(t.tmp / "grid", lookup_name="lookup_table_full.csv")
    with pytest.raises(FileNotFoundError, match=r"lookup_table\.csv"):
        pcf.precompute_flux_cube(str(model), str(model / "flux_cube.bin"))
    assert not (model / "flux_cube.bin").exists()


@row
def list_variants_raises_on_a_truncated_flux_cube(t):
    # An unreadable cube header used to be logged at debug level and shown as n/a.
    model = _simple_grid(t.tmp / "grid")
    (model / "flux_cube.bin").write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="Flux cube header truncated"):
        mesa_export.list_variants(model)


@row
def list_variants_without_a_built_cube_shows_no_wavelengths(t):
    # Unchanged: a grid whose cube has not been built yet lists with n/a wavelengths.
    (variant,) = mesa_export.list_variants(_simple_grid(t.tmp / "grid"))
    assert variant.wavelength_n == 0 and np.isnan(variant.wavelength_min)


@row
def export_variant_raises_on_missing_spectra_before_writing(t):
    (variant,) = mesa_export.list_variants(_simple_grid(t.tmp / "grid", missing=("c.txt",)))
    with pytest.raises(FileNotFoundError, match="1 spectra of variant 'full_grid' are missing") as excinfo:
        mesa_export.export_variant(variant, t.tmp / "export")
    assert "c.txt" in str(excinfo.value)
    assert not (t.tmp / "export").exists()


def _mesa_exits(t, base, *replies, **kwargs):
    """mesa_prepare's menu over ``base``, answered with ``replies``: the message it exits with."""
    answer(t.mp, *replies)
    with pytest.raises(SystemExit) as excinfo:
        mesa_export.run_interactive(base_dir=str(base), **kwargs)
    return str(excinfo.value.code)


@row
def run_interactive_exits_non_zero_when_the_grid_has_no_lookup(t):
    (t.tmp / "Empty").mkdir()
    assert "lookup_table.csv" in _mesa_exits(t, t.tmp, model="Empty", output=str(t.tmp / "out"))


@row
def non_numeric_vega_zero_point_in_a_filter_file_raises(t):
    # It used to be skipped, so the magnitude silently came from the Vega spectrum instead.
    model = _build(_simple_grid(t.tmp / "grid"))
    sed = SEDModel(name="grid", flux_cube_path=model / "flux_cube.bin")(5500.0, 4.25, 0.0)
    band = str(_box_filter(t.tmp / "Band.dat", "# zp_vega = 3.6e-9 erg/s/cm2/A", ANGSTROM))
    vega = str(_box_filter(t.tmp / "vega.dat", ANGSTROM))
    with pytest.raises(ValueError, match=r"Filter 'Band': metadata zp_vega = '3.6e-9 erg/s/cm2/A' is not a number"):
        sed.photometry(band, system="Vega", vega_spectrum=vega)


def _run_tool(tmp, mp, filters_dir):
    model = _build(_simple_grid(tmp / "grid"))
    mp.setattr("builtins.input", lambda *args: "")  # no distance scaling
    return tool.main(["--flux-cube", str(model / "flux_cube.bin"), "--teff", "5500", "--logg", "4.25",
                      "--metallicity", "0", "--filters", str(filters_dir), "--filter-system", "all",
                      "--photometric-system", "AB", "--plot", str(tmp / "sed.png")])


@row
def flux_cube_tool_exits_non_zero_when_a_filter_fails(t):
    # A failing filter used to print an error and the tool still exited 0.
    filters = t.tmp / "filters"
    filters.mkdir()
    _box_filter(filters / "Good.dat", ANGSTROM)
    (filters / "Dark.dat").write_text(f"{ANGSTROM}\n4000.0 0.0\n5000.0 0.0\n")
    assert _run_tool(t.tmp, t.mp, filters) == 1
    out = t.capsys.readouterr().out
    assert "Photometry failed for 1 filter(s):" in out
    assert "Dark: Filter transmission integrates to zero" in out
    assert "Good" in out and "mag=" in out


@row
def a_variants_grid_is_compared_variant_by_variant(t):
    folder = _kurucz_alpha(t.tmp / "a" / GRID, build=True)
    assert _flux_cube_shortfall(folder) is None
    assert _flux_cube_shortfall(copy_without_times(folder, t.tmp / "b" / GRID)) is None
    # One more spectrum, at a Teff no variant's cube has
    text = (KURUCZ / "Kurucz2003all_fid16515.txt").read_text().replace("# teff = 5750.0", "# teff = 6500.0")
    (folder / "Kurucz2003all_fid99999.txt").write_text(text)
    regenerate_lookup_table(str(folder))
    assert _flux_cube_shortfall(folder) == "does not include 1 of the 25 spectra beside it"


@row
def a_split_grid_is_checked_by_its_variants_not_by_a_top_level_copy(t):
    # NJM serves a split grid with a top-level flux_cube.bin, a copy of its default variant's. The check compared
    # that copy with every variant's rows, so Kurucz2003all "lacked" 476 of 8092 spectra and SED.fetch rebuilt it on
    # every call; and downloaded without the variant folders, Husfeld (the same nodes in every variant) passed.
    grid = _write_grid(t.tmp / "G", [("a0.txt", 5000.0, 4.0, 0.0, "0.0"), ("a4.txt", 5000.0, 4.0, 0.0, "0.4"),
                                     ("b4.txt", 6000.0, 4.0, 0.0, "0.4")], (*COLUMNS, "alpha"))
    _build(grid)
    shutil.copy2(grid / "G_alpha_0p00" / "flux_cube.bin", grid / "flux_cube.bin")
    assert _flux_cube_shortfall(grid) is None
    shutil.rmtree(grid / "G_alpha_0p00")
    assert _flux_cube_shortfall(grid).startswith("cannot be compared with the spectra beside it")


@row
def a_grid_built_as_one_cube_after_a_split_drops_the_split(t):
    # A rebuild that no longer split the grid left the earlier variants_index.csv and subgrids, read as its cubes.
    grid = _two_variant_grid(t.tmp / "G")
    _build(grid)
    for name in ("a4.txt", "b4.txt"):
        (grid / name).unlink()
    _write_grid(grid, [("a0.txt", 5000.0, 4.0, 0.0, "0.0"), ("b0.txt", 6000.0, 4.0, 0.0, "0.0")], (*COLUMNS, "alpha"))
    _build(grid)
    assert [variant.label for variant in list_variants(grid)] == ["full_grid"]
    assert not [p for p in grid.iterdir() if p.is_dir()] and _flux_cube_shortfall(grid) is None


@row
def variants_are_told_apart_by_value_and_named_apart(t):
    # Variants were grouped by text and named by number with two decimals: alpha "0" and "0.0" were two variants
    # in one folder, the second overwriting the first (its 4000 K spectrum in no cube), as yhe 0.001 and 0.004 were.
    grid = _write_grid(t.tmp / "G", [("a.txt", 4000.0, 4.0, 0.0, "0", "0.001"),
                                     ("b.txt", 5000.0, 4.0, 0.0, "0.0", "0.001"),
                                     ("c.txt", 4000.0, 4.0, 0.0, "0.4", "0.004")], (*COLUMNS, "alpha", "yhe"))
    _build(grid)
    assert sorted(p.name for p in grid.iterdir() if p.is_dir()) == [
        "G_alpha_0p00_yhe_0p001", "G_alpha_0p40_yhe_0p004"]
    assert _flux_cube_shortfall(grid) is None


@row
def a_parameter_column_under_any_name_lookup_accepts_is_no_extra_axis(t):
    # A Teff column named temperature, or an [M/H] column named Z/Zo, was read as that axis and also split on: one
    # subgrid per Teff. lookup's find_*_column accept these names; the extra-axis list did not.
    grid = _write_grid(t.tmp / "G", [("a.txt", 4000.0, 4.0, 0.0), ("b.txt", 5000.0, 4.0, 0.02),
                                     ("c.txt", 6000.0, 4.5, 0.02)], ("file_name", "temperature", "logg", "Z/Zo"))
    _build(grid)
    assert (grid / "flux_cube.bin").is_file() and not (grid / "variants_index.csv").exists()


@row
def a_spectrum_zero_at_every_sampled_wavelength_is_still_found(t):
    """A spectrum covering only part of the cube's wavelengths is zero elsewhere (resample_to_grid)."""
    flux = np.zeros((3, 1, 1, 20))  # the sampled wavelength planes are 2, 4, 6, 8, 11, 13, 15, 17
    flux[0, 0, 0] = 1.0  # covers every wavelength
    flux[1, 0, 0, [0, 19]] = 1.0  # only at wavelengths no sampled plane has
    write_flux_cube(t.tmp / "flux_cube.bin", [5000.0, 6000.0, 7000.0], [4.0], [0.0], np.linspace(1e3, 2e4, 20), flux)
    (t.tmp / "lookup_table.csv").write_text(
        "#file_name,teff,logg,metallicity\na.txt,5000,4.0,0.0\nb.txt,6000,4.0,0.0\nc.txt,7000,4.0,0.0\n")
    names, missing = spectra_not_in_cube(t.tmp / "flux_cube.bin", t.tmp / "lookup_table.csv")
    assert names == ["a.txt", "b.txt", "c.txt"] and missing == ["c.txt"]  # 7000 K: zero throughout


@row
def the_densifier_line_fits_80_columns(t):
    write_flux_cube(t.tmp / "src.bin", [4000.0, 8000.0], [4.0], [0.0], np.linspace(3000.0, 9000.0, 7),
                    np.ones((2, 1, 1, 7)))
    densify_grid(str(t.tmp / "src.bin"), str(t.tmp / "dense.bin"), teff_spacing=1000.0)
    lines = t.capsys.readouterr().out.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith("Generating"))
    assert lines[at:at + 2] == ["Generating 3 synthetic Teff points (0 by ML, 3 blackbody)",
                                "  in 1 wavelength block(s)..."]


@row
def an_ml_extended_grid_with_different_wavelength_ends_is_one_cube(t):
    # Each row's wl_max differs, so these columns used to split the grid into one "variant" per spectrum, with no
    # flux_cube.bin in the grid itself.
    grid = t.tmp / "Kurucz_extended"
    grid.mkdir()
    rows = []
    for i, (teff, logg) in enumerate([(5000.0, 4.0), (6000.0, 4.0), (5000.0, 4.5), (6000.0, 4.5)]):
        wl_max = 20000.0 + 5000.0 * i
        (grid / f"s{i}.txt").write_text("".join(f"{w:.1f} {1e-10:.6e}\n" for w in np.linspace(1000.0, wl_max, 30)))
        rows.append(f"s{i}.txt,{teff},{logg},0.0,True,1000.0,{wl_max}")
    (grid / "lookup_table.csv").write_text(
        "#file_name,teff,logg,metallicity,extended_by_ml,wl_min,wl_max\n" + "\n".join(rows) + "\n")
    pcf.precompute_flux_cube(str(grid), str(grid / "flux_cube.bin"))
    assert (grid / "flux_cube.bin").is_file()
    assert not (grid / "variants_index.csv").exists()
    assert not [p for p in grid.iterdir() if p.is_dir()]
    assert "No extra axes detected" in t.capsys.readouterr().out
    # A file column under another name (the first column is taken) gave one subgrid per spectrum until 0.6.0
    (grid / "lookup_table.csv").write_text((grid / "lookup_table.csv").read_text().replace("#file_name,", "#spectrum,"))
    pcf.precompute_flux_cube(str(grid), str(grid / "flux_cube.bin"))
    assert not [p for p in grid.iterdir() if p.is_dir()] and "No extra axes detected" in t.capsys.readouterr().out


# ----------------------------------------------------------------------
# Combining grids: stop on spectra that cannot be read or placed
# ----------------------------------------------------------------------
#: name: the grid whose lookup table lists a spectrum that is not there. Its normalisation factor used to be 1
#: (alpha_04), or its node was left empty for nearest-neighbour filling (Kurucz), silently.
COMBINE_MISSING = {"combine_stops_on_a_missing_spectrum_kurucz2003all": GRID,
                   "combine_stops_on_a_missing_spectrum_kurucz2003all_alpha_04": ALPHA_04}


def _combine_missing(t, grid):
    models = _golden_pair(t)
    with open(models / grid / "lookup_table.csv", "a") as fh:
        fh.write("missing.txt,0,999.9,2,1.25,4.0,0.0,999.9,999.9,999.9,9000.0,999.9,2,999.9\n")
    with pytest.raises(ValueError, match=r"No such file or directory: '.*missing\.txt'"):
        _combine(models)
    assert not (models / "combined").exists()


@row
def combine_stops_on_a_row_without_parameters(t):
    # A NaN Teff used to land on grid node [0, ...] and overwrite it.
    models = _golden_pair(t)
    with open(models / GRID / "lookup_table.csv", "a") as fh:
        fh.write("Kurucz2003all_fid16515.txt,0,999.9,2,1.25,4.0,-0.5,999.9,999.9,999.9,,999.9,2,999.9\n")
    with pytest.raises(ValueError, match=r"non-finite parameters \(teff=nan"):
        _combine(models)


@row
def combine_raises_after_writing_when_a_copy_fails(t):
    # Copy failures used to be printed and the run reported success.
    import sed_tools.cube.combine as csa

    models = _golden_pair(t)
    real_copy = shutil.copy2

    def flaky_copy(src, dst):
        if src.endswith("Kurucz2003all_fid16516.txt"):
            raise PermissionError(13, "Permission denied", src)
        return real_copy(src, dst)

    t.mp.setattr(csa.shutil, "copy2", flaky_copy)
    with pytest.raises(OSError, match=r"(?s)Could not copy 1 spectra.*fid16516"):
        _combine(models)
    assert (models / "combined" / "lookup_table.csv").exists()


def _planck_grid(t, name, nodes, scale=1.0, columns=("teff", "logg", "metallicity"), wl=None):
    """A grid of pi*B_lambda spectra, ``scale`` times their own flux in the files, at ``nodes``: one tuple of values per
    spectrum, for the lookup ``columns`` (Teff first), at the wavelengths ``wl`` (1000 Å to 30 µm by default)."""
    folder = t.tmp / "models" / name
    folder.mkdir(parents=True)
    wl = np.geomspace(1000.0, 3.0e5, 2000) if wl is None else wl
    rows = ["#file_name," + ",".join(columns)]
    for i, values in enumerate(nodes):
        np.savetxt(folder / f"s{i}.txt", np.column_stack([wl, scale * _constants.planck_flam(wl, values[0])]))
        rows.append(f"s{i}.txt," + ",".join(str(v) for v in values))
    (folder / "lookup_table.csv").write_text("\n".join(rows) + "\n")


def _combined_cubes(t, grids):
    """Combine ``grids`` into models/combined: {cube folder: {(Teff, log g, [M/H]) of each lookup row: the node's
    integral over that row's spectrum's}}. The combine guard: each folder's lookup table has one row per node a
    spectrum fills, and that node holds the row's spectrum (the combined folder's copy, resampled onto the cube's
    wavelengths as the cube is built) to 1 %, as NJM's check 4 requires of every served grid."""
    from sed_tools.api import SED
    from sed_tools.cube import fileio
    from sed_tools.grids.spectrum_io import read_text_spectrum

    SED.combine(grids, output="combined", model_root=t.tmp / "models", visualization=False)
    cubes = {}
    for folder in (p.parent for p in sorted((t.tmp / "models" / "combined").rglob("flux_cube.bin"))):
        teff, logg, meta, wl, flux = fileio.load(folder / "flux_cube.bin")
        rows = [line.split(",") for line in (folder / "lookup_table.csv").read_text().splitlines()[1:]]
        nodes = [(float(r[1]), float(r[2]), float(r[3])) for r in rows]
        assert len(set(nodes)) == len(nodes), f"{folder}: two lookup rows at one node"
        ratios = {}
        for node, row in zip(nodes, rows):
            it, il, im = (int(np.argmin(np.abs(axis - value))) for axis, value in zip((teff, logg, meta), node))
            own_flux = resample_to_grid(*read_text_spectrum(folder / row[0]), wl)
            ratios[node] = np.trapezoid(flux[:, im, il, it], wl) / np.trapezoid(own_flux, wl)
        assert all(abs(r - 1) <= 0.01 for r in ratios.values()), (folder, ratios)
        cubes[folder.name] = ratios
    return cubes


@row
def combined_cube_nodes_hold_their_own_spectra(t):
    # Combine used to scale every spectrum to the nearest spectrum of a reference grid, however far away: here the
    # 3500 and 3600 K nodes held 0.00055 and 0.00048 of their own flux, scaled to the 800 K blackbody's.
    _planck_grid(t, "cool", [(800.0, 4.5, 0.0), (900.0, 4.5, 0.0)])
    _planck_grid(t, "warm", [(3500.0, 4.5, 0.0), (3600.0, 4.5, 0.0)])
    assert [len(ratios) for ratios in _combined_cubes(t, ["cool", "warm"]).values()] == [4]


@row
def combine_leaves_a_node_no_spectrum_fills_at_zero(t):
    # Every empty node used to hold its nearest filled node's spectrum, unscaled: (3500 K, log g 4.5) held the
    # 900 K star's and (800 K, log g 5.0) the 3500 K star's, which the 1 % guard cannot see (it checks listed nodes).
    from sed_tools.cube import fileio

    _planck_grid(t, "cool", [(800.0, 4.5, 0.0), (900.0, 4.5, 0.0)])
    _planck_grid(t, "warm", [(3500.0, 5.0, 0.0), (3600.0, 5.0, 0.0)])
    _combined_cubes(t, ["cool", "warm"])
    *_, flux = fileio.load(t.tmp / "models" / "combined" / "flux_cube.bin")
    assert flux.any(axis=0)[0].tolist() == [[True, True, False, False], [False, False, True, True]]  # [log g][Teff]


@row
def combine_keeps_axis_values_that_are_close(t):
    # Values within 1 K, 0.01 dex of log g or 0.01 of [M/H] used to be merged into one node, the later spectrum
    # kept: MSG's OSTAR2002 states Z/Zo 0, 0.001 and 0.01, and only the 0.01 spectrum stayed, at node 0.
    from sed_tools.cube import fileio

    _planck_grid(t, "O", [(30000.0, 4.0, z) for z in (0.0, 0.001, 0.01, 0.02)])
    assert [len(ratios) for ratios in _combined_cubes(t, ["O"]).values()] == [4]
    assert fileio.load_axes(t.tmp / "models" / "combined" / "flux_cube.bin")[2].tolist() == [0.0, 0.001, 0.01, 0.02]


@row
def combine_refuses_to_write_into_one_of_its_grids(t):
    # The CLI refused this; SED.combine replaced a's lookup table and cube and copied a's own spectra into it.
    from sed_tools.api import SED

    _planck_grid(t, "a", [(5000.0, 4.5, 0.0)])
    _planck_grid(t, "b", [(6000.0, 4.5, 0.0)])
    before = sorted(p.name for p in (t.tmp / "models" / "a").iterdir())
    with pytest.raises(ValueError, match="The output grid a is also one of the grids to combine"):
        SED.combine(["a", "b"], output="a", model_root=t.tmp / "models", visualization=False)
    assert sorted(p.name for p in (t.tmp / "models" / "a").iterdir()) == before


@row
def combining_again_into_a_folder_replaces_the_earlier_combination(t):
    # A rerun kept what the earlier one wrote: combining [a, b] then [b, a] left a_000002.txt, listed at 4000 K,
    # holding a's 6000 K spectrum, with four stale copies beside it; and an earlier single cube stayed beside a later
    # split, so SED.combine returned the earlier combination.
    _planck_grid(t, "a", [(4000.0, 4.5, 0.0), (5000.0, 4.5, 0.0), (6000.0, 4.5, 0.0)])
    _planck_grid(t, "b", [(7000.0, 4.5, 0.0), (8000.0, 4.5, 0.0)])
    _planck_grid(t, "al", [(4000.0, 4.5, 0.0, 0.0), (4000.0, 4.5, 0.0, 0.4)], columns=("teff", "logg", "meta", "alpha"))
    combined = t.tmp / "models" / "combined"
    _combined_cubes(t, ["a", "b"])
    _combined_cubes(t, ["b", "a"])  # each listed copy holds its own node's spectrum
    assert _flux_cube_shortfall(combined) is None  # no stale copies
    _combined_cubes(t, ["a", "al"])
    assert sorted(p.name for p in combined.iterdir() if not p.name.endswith(".txt")) == [
        "combined_alpha_0p00", "combined_alpha_0p40", "lookup_table.csv", "variants_index.csv"]


@row
def combine_places_a_grid_without_log_g_at_every_log_g(t):
    # A grid without log g or [M/H] (a blackbody) used to be placed at log g 0 and [M/H] 0, as stars of that log g.
    # It holds the same spectrum at every log g, so it fills every value the other grids give the axis, except
    # where a spectrum placed by its own log g is: the stars keep 5000 K, though the blackbodies are named later.
    _planck_grid(t, "stars", [(5000.0, 4.0, 0.0), (5000.0, 4.5, 0.0), (6000.0, 4.0, 0.0), (6000.0, 4.5, 0.0)])
    _planck_grid(t, "bb", [(5000.0,), (7000.0,)], scale=2.0, columns=("teff",))
    [ratios] = _combined_cubes(t, ["stars", "bb"]).values()
    assert sorted(ratios) == [(teff, logg, 0.0) for teff in (5000.0, 6000.0, 7000.0) for logg in (4.0, 4.5)]
    lookup = (t.tmp / "models" / "combined" / "lookup_table.csv").read_text().splitlines()
    assert [line.split(",")[4] for line in lookup[1:]] == ["stars"] * 4 + ["bb"] * 2
    assert lookup[-2:] == ["bb_000005.txt,7000.0,4.0,0.0,bb,s1.txt", "bb_000005.txt,7000.0,4.5,0.0,bb,s1.txt"]
    assert not (t.tmp / "models" / "combined" / "bb_000004.txt").exists()  # its 5000 K spectrum fills no node


@row
def combine_keeps_each_grids_flux_at_shared_nodes(t):
    # The later grid's spectra used to be scaled onto the earlier grid's at their shared nodes, to 0.971 of their own
    # flux here: combine is not a calibration step, so the second grid's 3 % stays. a's extra column has one value,
    # so it is no axis and does not make a's spectra outrank b's.
    _planck_grid(t, "a", [(5000.0, 4.5, 0.0, 2), (5500.0, 4.5, 0.0, 2), (6000.0, 4.5, 0.0, 2)],
                 columns=("teff", "logg", "metallicity", "vtur"))
    _planck_grid(t, "b", [(5000.0, 4.5, 0.0), (5500.0, 4.5, 0.0)], scale=1.03)
    assert [len(ratios) for ratios in _combined_cubes(t, ["a", "b"]).values()] == [3]
    lookup = (t.tmp / "models" / "combined" / "lookup_table.csv").read_text()
    assert "b_000003.txt,5000.0,4.5,0.0,b,s0.txt" in lookup  # the shared nodes are b's, 3 % brighter than a's
    assert sorted(p.name for p in (t.tmp / "models" / "combined").glob("*.txt")) == [
        "a_000002.txt", "b_000003.txt", "b_000004.txt"]  # a's two overwritten spectra are not copied


@row
def combined_wavelengths_are_the_union_of_the_grids_own(t):
    # The axis used to be at most 5000 even steps over 3000-25000 Å, set by spectra drawn at random: no flux in the
    # UV or the mid-IR (a 30,000 K star 2.4 mag too faint), and two runs on the same grids could write two cubes.
    from sed_tools.cube import fileio

    uv, ir = np.geomspace(900.0, 3.0e4, 300), np.geomspace(2.0e4, 3.0e5, 200)
    _planck_grid(t, "uv", [(30000.0, 4.0, 0.0)], wl=uv)
    _planck_grid(t, "ir", [(3000.0, 5.0, 0.0)], wl=ir)
    _combined_cubes(t, ["uv", "ir"])
    assert np.array_equal(fileio.load_axes(t.tmp / "models" / "combined" / "flux_cube.bin")[3], np.union1d(uv, ir))


# ----------------------------------------------------------------------
# The grid densifier: the Kurucz slice's axes, spectra and sigma*T^4; ML failures; its menu; block sizes
# ----------------------------------------------------------------------
_DENSIFIED = {}


def _kurucz_cubes(t):
    """(source, densified) FluxCubes of the golden Kurucz slice (Teff 5750-6250 K) at 125 K spacing: built once,
    as the old module-scoped fixture was, and only read."""
    if not _DENSIFIED:
        work = t.tmp.parent / "densify"
        shutil.copytree(KURUCZ, work / GRID)
        _build(work / GRID)
        densify_grid(str(work / GRID / "flux_cube.bin"), str(work / "dense.bin"), teff_spacing=125.0)
        _DENSIFIED["cubes"] = (FluxCube.from_file(str(work / GRID / "flux_cube.bin")),
                               FluxCube.from_file(str(work / "dense.bin")))
    return _DENSIFIED["cubes"]


@row
def densified_grid_keeps_source_axes_and_adds_teff_nodes(t):
    source, dense = _kurucz_cubes(t)
    np.testing.assert_array_equal(source.teff_grid, [5750.0, 6000.0, 6250.0])
    np.testing.assert_array_equal(dense.teff_grid, [5750.0, 5875.0, 6000.0, 6125.0, 6250.0])
    for axis in ("logg_grid", "meta_grid", "wavelengths"):
        np.testing.assert_array_equal(getattr(dense, axis), getattr(source, axis))


@row
def densified_cube_copies_source_spectra_exactly(t):
    # Before the fix the copied rows differed from the source by up to 2.6e9.
    source, dense = _kurucz_cubes(t)
    for i_src, teff in enumerate(source.teff_grid):
        np.testing.assert_array_equal(dense.flux[int(np.flatnonzero(dense.teff_grid == teff)[0])], source.flux[i_src])


@row
def every_densified_node_integrates_to_sigma_t4(t):
    # Before the fix every node integrated to 64-89 times sigma*T^4.
    for cube in _kurucz_cubes(t):
        for i_t, i_l, i_m in np.ndindex(cube.flux.shape[:3]):
            ratio = np.trapezoid(cube.flux[i_t, i_l, i_m], cube.wavelengths) / (
                _constants.SIGMA * cube.teff_grid[i_t] ** 4)
            assert ratio == pytest.approx(1.0, rel=0.02), (cube.teff_grid[i_t], i_l, i_m)


@row
def densify_leaves_a_hole_empty_and_writes_the_lookup_table_mesa_needs(t):
    # A synthetic Teff beside an empty node got an unscaled blackbody, so a hole became zeros and blackbodies along
    # Teff (4.0e6 at 5000 K, log g 5, here); and the output had no lookup_table.csv, which MESA stops without.
    flux = np.zeros((2, 2, 1, 7))
    flux[:, 0, 0] = _constants.planck_flam(np.linspace(3000.0, 9000.0, 7), np.array([[4000.0], [6000.0]]))
    write_flux_cube(t.tmp / "src" / "flux_cube.bin", [4000.0, 6000.0], [4.0, 5.0], [0.0], np.linspace(3e3, 9e3, 7),
                    flux)
    (t.tmp / "src" / "lookup_table.csv").write_text("#file_name,teff,logg,meta\na.txt,4000,4.0,0\nb.txt,6000,4.0,0\n")
    densify_grid(str(t.tmp / "src" / "flux_cube.bin"), str(t.tmp / "dense" / "flux_cube.bin"), teff_spacing=1000.0)
    dense = FluxCube.from_file(str(t.tmp / "dense" / "flux_cube.bin"))
    assert dense.teff_grid.tolist() == [4000.0, 5000.0, 6000.0]
    assert dense.flux[:, 0, 0].any(axis=1).tolist() == [True] * 3 and not dense.flux[:, 1].any()
    assert (t.tmp / "dense" / "lookup_table.csv").read_text() == (t.tmp / "src" / "lookup_table.csv").read_text()


def _ml_generator(t, generate):
    """sed_tools.ml.generator's SEDGenerator replaced by one whose ``generate`` is this function."""
    fake = types.ModuleType("sed_tools.ml.generator")
    fake.SEDGenerator = type("SEDGenerator", (), {"__init__": lambda self, path: None, "generate": generate})
    t.mp.setitem(sys.modules, "sed_tools.ml.generator", fake)


@row
def densify_stops_when_ml_generation_fails(t):
    # Used to log a warning and write a blackbody in place of the ML SED.
    def generate(self, **kwargs):
        raise ValueError("network produced NaN")

    _ml_generator(t, generate)
    _small_cube(t.tmp / "src.bin")
    with pytest.raises(RuntimeError, match=r"Teff=6000 logg=4\.00 meta=0\.00.*network produced NaN"):
        densify_grid(str(t.tmp / "src.bin"), str(t.tmp / "dense.bin"), teff_spacing=1000.0, ml_model="fake")


#: name: the cube chosen from a menu of two. "0" and negative numbers used to pick cubes from the end of the list.
DENSIFIER_CHOICES = {f"densifier_menu_rejects_out_of_range_choice_{name}": reply
                     for name, reply in (("0", "0"), ("minus_1", "-1"), ("3", "3"), ("x", "x"))}


def _densifier_rejects(t, reply):
    from sed_tools.cube.densify import run_interactive_workflow

    for name in ("a", "b"):
        _small_cube(t.tmp / name / "flux_cube.bin")
    t.mp.setattr("builtins.input", lambda prompt="": reply)
    t.mp.setattr("sed_tools.cube.densify.densify_grid", lambda **kw: pytest.fail("must not run"))
    run_interactive_workflow(str(t.tmp))
    assert "Invalid selection." in t.capsys.readouterr().out


@row
def densified_ml_nodes_do_not_depend_on_the_block_size(t):
    model_wl = np.geomspace(900.0, 60000.0, 200)

    def generate(self, teff, logg, meta, check_bounds=True):
        return model_wl.copy(), 1e-3 * teff * (model_wl / 5000.0) ** (-2.0 - 0.1 * logg + meta)

    def densify(name, ram):
        """Densify src.bin into <name>.bin as if ``ram`` bytes of RAM were available; its bytes."""
        t.mp.setattr("sed_tools.cube.densify._available_ram_bytes", lambda: ram)
        densify_grid(str(t.tmp / "src.bin"), str(t.tmp / f"{name}.bin"), teff_spacing=1000.0, ml_model="fake")
        return (t.tmp / f"{name}.bin").read_bytes()

    _ml_generator(t, generate)
    # 3 Teff x 3 logg x 2 [M/H] x 30 wavelengths, a different spectrum at every node
    teff, wl, rng = [4000.0, 6000.0, 9000.0], np.geomspace(1000.0, 50000.0, 30), np.random.default_rng(3)
    flux = np.array([_constants.planck_flam(wl, te) * rng.uniform(0.5, 1.5, size=(3, 2, 30)) for te in teff])
    write_flux_cube(t.tmp / "src.bin", teff, [3.0, 4.0, 5.0], [-1.0, 0.0], wl, flux)
    # 6 dense Teffs: each wavelength holds (3 + 6) x 3 x 2 doubles = 432 bytes, of which a block may use half the
    # RAM, so 2 x 7 x 432 bytes gives blocks of 7 wavelengths: 7, 7, 7, 7, 2.
    one_block = densify("one", 2**40)
    assert densify("five", 2 * 7 * 432) == one_block
    dense = FluxCube.from_file(str(t.tmp / "five.bin"))
    for i_t, te in ((1, 5000.0), (3, 7000.0), (4, 8000.0)):
        for i_l, logg in enumerate(dense.logg_grid):
            for i_m, meta in enumerate(dense.meta_grid):
                expected = resample_to_grid(*generate(None, te, logg, meta), dense.wavelengths)
                np.testing.assert_array_equal(dense.flux[i_t, i_l, i_m], expected)


# ----------------------------------------------------------------------
# The menus: mesa_prepare exits non-zero whenever nothing was exported; combine and densifier prompts ask again
# or go back; split, mean and combined grids; averaged rows; 80 columns
# ----------------------------------------------------------------------
def _variant_grid(t):
    """``<base>/Grid`` with two alpha variants, split as ``sed-tools rebuild`` splits them: a subgrid per variant
    with its lookup table and cube, listed in variants_index.csv; returns base."""
    grid = t.tmp / "stellar" / "Grid"
    write_lookup(grid, [*COLUMNS, "alpha"], [("a.txt", 5000, 4.0, 0.0, "0.0"), ("b.txt", 5000, 4.0, 0.0, "0.4")])
    index = ["variant_name,alpha,n_spectra,path"]
    for name, alpha in (("a.txt", "0.0"), ("b.txt", "0.4")):
        (grid / name).write_text("4000 1.0\n5000 2.0\n")
        write_lookup(grid / f"alpha_{alpha}", [*COLUMNS, "alpha"], [(name, 5000, 4.0, 0.0, alpha)])
        write_flux_cube(grid / f"alpha_{alpha}" / "flux_cube.bin", [5000.0], [4.0], [0.0], [4000.0, 5000.0],
                        np.ones((1, 1, 1, 2)))
        index.append(f"alpha_{alpha},{alpha},1,alpha_{alpha}")
    (grid / "variants_index.csv").write_text("\n".join(index) + "\n")
    return grid.parent


@row
def mesa_prepare_exits_1_without_spectral_grids(t):
    message = _mesa_exits(t, t.tmp)
    assert message.startswith("Error: no spectral grids found in")


@row
def mesa_prepare_exits_1_on_an_invalid_grid_choice(t):
    message = _mesa_exits(t, _variant_grid(t), "9")
    assert message.startswith("Error: invalid spectral grid selection '9'")


@row
def mesa_prepare_exits_1_for_a_single_grid_without_a_flux_cube(t):
    # It used to print "the existing flux_cube.bin already covers the full grid" and exit 0 even when no
    # flux_cube.bin had been built.
    write_lookup(t.tmp / "Plain", list(COLUMNS), [("a.txt", 5000, 4.0, 0.0)])
    message = _mesa_exits(t, t.tmp, model="Plain", output=str(t.tmp / "out"))
    assert "has no flux_cube.bin" in message and "Nothing was exported" in message
    assert not (t.tmp / "out").exists()


@row
def mesa_prepare_exits_1_on_an_invalid_variant_choice(t):
    base = _variant_grid(t)
    message = _mesa_exits(t, base, "3", model="Grid", output=str(base / "out"))
    assert message.startswith("Error: invalid variant selection '3'")
    assert not (base / "out").exists()


@row
def mesa_prepare_exits_1_when_overwrite_is_declined(t):
    out = _variant_grid(t) / "out"
    out.mkdir()
    (out / "keep.txt").write_text("untouched")
    message = _mesa_exits(t, out.parent, "1", "n", model="Grid", output=str(out))
    assert message == f"Error: {out} already exists and was not overwritten; nothing was exported."
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


@row
def mesa_prepare_still_exports_the_chosen_variant(t):
    out = _variant_grid(t) / "out"
    answer(t.mp, "2")
    mesa_export.run_interactive(base_dir=str(out.parent), model="Grid", output=str(out))
    assert sorted(p.name for p in out.iterdir()) == ["b.txt", "flux_cube.bin", "lookup_table.csv"]


@row
def combine_menu_asks_again_until_the_selection_is_valid(t):
    from sed_tools.cube.combine import select_models_interactive

    grids = [(name, str(t.tmp / name)) for name in ("A", "B", "C")]
    # Empty input and a bad token used to combine every grid; out-of-range numbers were dropped silently. A
    # repeated grid is refused too.
    answer(t.mp, "", "two", "1,", "0", "4", "1,9", "2,2", "3,1")
    assert select_models_interactive(grids) == [grids[2], grids[0]]  # in the order typed
    out = t.capsys.readouterr().out
    for reply in ("''", "'two'", "'1,'", "'0'", "'4'", "'1,9'", "'2,2'"):
        assert f"Invalid selection {reply}: enter distinct numbers from 1 to 3, comma-separated, or 'all'." in out
    assert "Using all" not in out


@row
def combine_menu_all_still_selects_every_grid(t):
    from sed_tools.cube.combine import select_models_interactive

    grids = [(name, str(t.tmp / name)) for name in ("A", "B")]
    answer(t.mp, "ALL")
    assert select_models_interactive(grids) == grids


#: name: the reply that goes back from the combine menu; only Ctrl-C used to leave it.
GOES_BACK = {"combine_grid_selection_goes_back_q": "q", "combine_grid_selection_goes_back_b": "b"}


def _goes_back(t, reply):
    from sed_tools.cube.combine import select_models_interactive

    grids = [(name, str(t.tmp / name)) for name in ("A", "B")]
    answer(t.mp, reply)
    assert select_models_interactive(grids) == []


@row
def model_photometry_warns_once_about_filters_past_the_model(t):
    cube = affine_cube(t.tmp, wavelengths=(3000.0, 4000.0, 5000.0))
    sed = SEDModel(name="affine", flux_cube_path=cube.path)(4500.0, 2.5, -0.3)
    with t.caplog.at_level(logging.WARNING, logger="sed_tools.photometry"):
        sed.photometry(flat_filter("V"), flat_filter("B"), system="AB")
    [warning] = [r.getMessage() for r in t.caplog.records]
    assert warning.endswith("so these bands are biased low: V (47.5%), B (47.5%).")


def _averaged_rows(model_dir, caplog, **override):
    """Build the grid's cube; the warnings that lookup rows were averaged onto one node."""
    with caplog.at_level(logging.WARNING, logger="sed_tools.cube.variants"):
        _build(model_dir, override_dict=override or None)
    return [r.getMessage() for r in caplog.records if "lookup rows were averaged" in r.getMessage()]


@row
def a_column_that_tells_apart_spectra_at_one_node_splits_the_grid(t):
    # BOSZ's am, cm, vturb and atmos were bookkeeping by name ("MAST BOSZ specific"), so rows differing in them were
    # averaged onto one node, with a warning. Only what SED_Tools writes to describe a file is bookkeeping now: a
    # column that tells apart spectra at one node splits the grid, and one that varies only between nodes (atmos
    # here, hres's model code on NJM) labels the spectra and splits nothing.
    grid = _write_grid(t.tmp / "BOSZ", [
        ("a.txt", 5000, 4.0, 0.0, 0.0, 0.0, 1, "ap"), ("b.txt", 5000, 4.0, 0.0, 0.4, 0.0, 2, "ap"),
        ("c.txt", 6000, 4.0, 0.0, 0.0, 0.0, 1, "mp"), ("d.txt", 6000, 4.0, 0.0, 0.4, 0.0, 2, "mp")],
        (*COLUMNS, "am", "cm", "vturb", "atmos"))
    assert _averaged_rows(grid, t.caplog) == []
    assert sorted(p.name for p in grid.iterdir() if p.is_dir()) == [
        "BOSZ_am_0p00_cm_0p00_vturb_1p00", "BOSZ_am_0p40_cm_0p00_vturb_2p00"]


@row
def duplicate_nodes_without_extra_axes_are_reported(t):
    grid = _write_grid(t.tmp / "Plain", [("a.txt", 5000, 4.0, 0.0), ("b.txt", 5000.0, 4.0, 0.0),
                                         ("c.txt", 6000, 4.0, 0.0)])
    (message,) = _averaged_rows(grid, t.caplog)
    assert "2 lookup rows were averaged onto 1 (Teff, logg, [M/H]) node(s)" in message
    assert message.endswith("differ in: no column but the file name (duplicate nodes).")


@row
def a_grid_with_one_row_per_node_does_not_warn(t):
    grid = _write_grid(t.tmp / "Plain", [("a.txt", 5000, 4.0, 0.0), ("b.txt", 6000, 4.0, 0.0)])
    assert _averaged_rows(grid, t.caplog) == []


@row
def split_and_filter_cubes_report_their_own_averaged_rows(t):
    columns = (*COLUMNS, "alpha", "source")
    rows = [("a.txt", 5000, 4.0, 0.0, "0.0", "x"), ("b.txt", 5000, 4.0, 0.0, "0.0", "y"),
            ("c.txt", 5000, 4.0, 0.0, "0.4", "x")]
    split = _write_grid(t.tmp / "Split", rows, columns)
    (message,) = _averaged_rows(split, t.caplog)
    assert message.startswith(f"{split / 'Split_alpha_0p00' / 'flux_cube.bin'}: 2 lookup rows were averaged")
    assert message.endswith("differ in: source.")

    t.caplog.clear()
    filtered = _write_grid(t.tmp / "Filtered", rows, columns)
    strategy = {"on_collision": {"strategy": "filter", "filter": {"alpha": "0.4"}}}
    assert _averaged_rows(filtered, t.caplog, **strategy) == []
    strategy["on_collision"]["filter"]["alpha"] = "0.0"
    (message,) = _averaged_rows(filtered, t.caplog, **strategy)
    assert message.startswith(f"{filtered / 'flux_cube.bin'}: 2 lookup rows were averaged")


@row
def combine_menu_q_returns_to_the_menu_without_failing(t):
    from sed_tools import cli

    for name in ("A", "B"):
        (t.tmp / name).mkdir()
        (t.tmp / name / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\na.txt,5000,4,0\n")
    answer(t.mp, "q")
    assert cli._run_menu_flow(lambda: cli.run_combine_flow(base_dir=str(t.tmp))) is True
    assert "No spectral grids selected." in t.capsys.readouterr().out
    assert not (t.tmp / "combined_models").exists()


@row
def densifier_cube_list_fits_80_columns(t):
    from sed_tools.cube.densify import run_interactive_workflow

    name = "a-rather-long-spectral-grid-name_dense"
    write_flux_cube(t.tmp / name / "flux_cube.bin", [5750, 6000, 6250], [4.0], [0.0], [1000.0, 2000.0],
                    np.ones((3, 1, 1, 2)))
    answer(t.mp, "")
    run_interactive_workflow(base_dir=str(t.tmp))
    out = t.capsys.readouterr().out
    assert f"   1) {name}/flux_cube.bin\n      Teff 5750–6250 K  max gap 250 K  (3 pts)\n" in out
    assert wide_lines(out, str(t.tmp)) == []  # was 92 wide with a short name


@row
def mesa_prepare_invalid_choice_fits_80_columns(t):
    (t.tmp / "Kurucz").mkdir()
    (t.tmp / "Kurucz" / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\na.txt,5000,4,0\n")
    message = _mesa_exits(t, t.tmp, "")
    assert message == ("Error: invalid spectral grid selection ''.\n"
                       "Enter a number from 1 to 1 or a grid name. Nothing was exported.")
    assert wide_lines(message) == []  # was 107 wide


@row
def densifier_spacing_prompt_takes_the_spacing_typed(t):
    from sed_tools.cube.densify import run_interactive_workflow

    write_flux_cube(t.tmp / "grid" / "flux_cube.bin", [5000.0, 7000.0], [4.0], [0.0], [1000.0, 2000.0],
                    np.ones((2, 1, 1, 2)))
    calls = []
    t.mp.setattr("sed_tools.cube.densify.densify_grid", lambda **kw: calls.append(kw))
    answer(t.mp, "1", "250", "", "")
    run_interactive_workflow(str(t.tmp))
    assert [c["teff_spacing"] for c in calls] == [250.0]


def _split_grid(t):
    """KuruczAlpha after a rebuild: one subgrid per alpha and variants_index.csv; what it printed is dropped."""
    split_grid = _kurucz_alpha(t.tmp / "stellar" / "KuruczAlpha", build=True)
    t.capsys.readouterr()
    return split_grid


@row
def list_variants_reads_the_subgrids_rebuild_writes(t):
    # Used to raise "No flux_cube.bin found in .../KuruczAlpha. Run 'sed-tools rebuild' first."
    split_grid = _split_grid(t)
    variants = list_variants(split_grid)
    assert [v.label for v in variants] == ["alpha_0", "alpha_0.4"]
    assert [v.extra_axes for v in variants] == [{"alpha": "0"}, {"alpha": "0.4"}]
    assert [v.n_spectra for v in variants] == [12, 12]
    sub = "KuruczAlpha_alpha_0p40_lh_1p25_vtur_2_km_s"
    assert variants[1].flux_cube_path == split_grid / sub / "flux_cube.bin"
    assert variants[1].source_model_dir == split_grid
    assert variants[1].wavelength_n == 1199
    assert set(variants[1].lookup_rows["alpha"]) == {"0.4"}


@row
def mesa_prepare_menu_exports_a_variant_of_a_split_grid(t):
    split_grid = _split_grid(t)
    answer(t.mp, "2", "")
    mesa_export.run_interactive(base_dir=str(split_grid.parent), model="KuruczAlpha")
    out = t.capsys.readouterr().out
    assert "Sub-variants of KuruczAlpha: 2 variants over alpha" in out
    assert "lh" not in out.split("Sub-variants")[1] and "vtur" not in out
    exported = split_grid.parent / "KuruczAlpha__alpha_0.4"
    assert len(list(exported.glob("*.txt"))) == 12 and (exported / "flux_cube.bin").is_file()


@row
def list_variants_raises_for_a_subgrid_without_its_cube(t):
    sub = _split_grid(t) / "KuruczAlpha_alpha_0p00_lh_1p25_vtur_2_km_s"
    (sub / "flux_cube.bin").rename(sub / "flux_cube.bin.moved")
    with pytest.raises(FileNotFoundError, match="lists the subgrid KuruczAlpha_alpha_0p00_lh_1p25_vtur_2_km_s"):
        list_variants(sub.parent)


@row
def mesa_prepare_on_a_uniform_grid_says_one_thing(t):
    # The golden Kurucz grid has alpha, lh and vtur columns with one value each. It used to print "the master
    # flux_cube.bin averages all extra-axis variants (alpha, lh, vtur)" and then "This spectral grid has no
    # extra-axis variants".
    shutil.copytree(KURUCZ, t.tmp / GRID)
    _build(t.tmp / GRID)
    t.capsys.readouterr()
    mesa_export.run_interactive(base_dir=str(t.tmp), model=GRID)
    out = t.capsys.readouterr().out
    assert "averages" not in out and "Note" not in out
    assert "has one flux_cube.bin for all its spectra and no variants to choose between" in out
    assert "already MESA-ready" in out


@row
def mesa_prepare_notes_a_mean_cube_naming_only_the_varying_axes(t):
    grid = _build(_kurucz_alpha(t.tmp / "KuruczAlpha"), override_dict={"on_collision": {"strategy": "mean"}})
    t.capsys.readouterr()
    (variant,) = mesa_export.list_variants(grid)
    out = t.capsys.readouterr().out
    assert "Note: KuruczAlpha has 2 variants over alpha, but no per-variant cubes" in out
    assert "lh" not in out and "vtur" not in out
    assert variant.label == "full_grid" and variant.flux_cube_path == grid / "flux_cube.bin"


@row
def export_variant_refuses_a_missing_cube_before_writing(t):
    grid = t.tmp / "Plain"
    grid.mkdir()
    (grid / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\na.txt,5000,4.0,0.0\n")
    (grid / "a.txt").write_text("4000 1.0\n5000 2.0\n")
    (variant,) = mesa_export.list_variants(grid)
    with pytest.raises(FileNotFoundError, match="does not exist; nothing was exported"):
        mesa_export.export_variant(variant, t.tmp / "out")
    assert not (t.tmp / "out").exists()


@row
def find_stellar_models_leaves_out_earlier_combine_outputs(t):
    from sed_tools.cube.combine import find_stellar_models

    two_grids = _golden_pair(t)
    _combine(two_grids, "omni")
    # A combine output whose lookup was rebuilt since is still found by its default name.
    shutil.copytree(KURUCZ, two_grids / "combined_models")
    t.capsys.readouterr()
    found = find_stellar_models(str(two_grids))
    assert sorted(name for name, _ in found) == ["Kurucz2003all", "Kurucz2003all__alpha_04"]
    assert ("Leaving out combined_models, omni: written by an earlier combine. "
            "To combine one again, name it explicitly.") in t.capsys.readouterr().out


@row
def non_interactive_combine_does_not_fold_in_the_previous_output(t):
    # The second run used to combine the two grids and combined_models: each spectrum twice.
    import matplotlib
    import pandas as pd

    from sed_tools.cli import run_combine_flow

    matplotlib.use("Agg")
    _planck_grid(t, "a", [(5000.0, 4.5, 0.0), (5500.0, 4.5, 0.0), (6000.0, 4.5, 0.0)])
    _planck_grid(t, "b", [(7000.0, 4.5, 0.0), (7500.0, 4.5, 0.0)])
    two_grids = t.tmp / "models"
    run_combine_flow(str(two_grids), interactive=False)
    run_combine_flow(str(two_grids), output_name="combined2", interactive=False)
    out = t.capsys.readouterr().out
    assert "Selected 2 spectral grids to combine:" in out and "- combined_models" not in out
    lookup = pd.read_csv(two_grids / "combined2" / "lookup_table.csv")
    assert len(lookup) == 5
    assert set(lookup["source_model"]) == {"a", "b"}


@row
def mesa_prepare_takes_a_combined_grid_as_it_is(t):
    # source_model and original_file differ on every row of a combined grid's lookup; they say where each
    # spectrum came from and are not variants of the grid.
    _planck_grid(t, "a", [(5000.0, 4.5, 0.0), (5500.0, 4.5, 0.0)])
    _planck_grid(t, "b", [(6000.0, 4.5, 0.0)])
    _combined_cubes(t, ["a", "b"])
    t.capsys.readouterr()
    mesa_export.run_interactive(base_dir=str(t.tmp / "models"), model="combined")
    out = t.capsys.readouterr().out
    assert "Note" not in out and "variants over" not in out
    assert "has one flux_cube.bin for all its spectra and no variants to choose between" in out


@row
def combine_keeps_every_variant_of_an_extra_axis(t):
    # Extra axes used to be dropped: spectra that differ only in alpha met on one node, and the later overwrote the
    # earlier. The combination is split as the cube builder splits one grid, and a grid without the axis is in
    # every variant ("any value", as for log g).
    _planck_grid(t, "alpha", [(5000.0, 4.5, 0.0, 0.0), (5000.0, 4.5, 0.0, 0.4)],
                 columns=("teff", "logg", "metallicity", "alpha"))
    _planck_grid(t, "plain", [(6000.0, 4.5, 0.0)])
    cubes = _combined_cubes(t, ["alpha", "plain"])
    assert sorted(cubes) == ["combined_alpha_0p00", "combined_alpha_0p40"]
    assert all(sorted(ratios) == [(5000.0, 4.5, 0.0), (6000.0, 4.5, 0.0)] for ratios in cubes.values())
    combined = t.tmp / "models" / "combined"
    for variant, own in (("combined_alpha_0p00", "alpha_000000.txt,5000.0,4.5,0.0,0.0,alpha,s0.txt"),
                         ("combined_alpha_0p40", "alpha_000001.txt,5000.0,4.5,0.0,0.4,alpha,s1.txt")):
        assert (combined / variant / "lookup_table.csv").read_text().splitlines() == [
            "#file_name,teff,logg,meta,alpha,source_model,original_file", own,
            f"plain_000002.txt,6000.0,4.5,0.0,{own.split(',')[4]},plain,s0.txt"]
    assert not (combined / "flux_cube.bin").exists()  # as the cube builder leaves a split grid's folder
    assert [v.label for v in list_variants(combined)] == ["alpha_0.0", "alpha_0.4"]


# ----------------------------------------------------------------------
# photometry() takes a (name, path) tuple as one filter, several such tuples, and plain paths as before
# ----------------------------------------------------------------------
def _evaluated(t):
    """A SED at 5000 K, log g 4.5, [M/H] 0 from a 2 x 2 x 2 cube: one shape, scaled by Teff."""
    teff, logg, meta = [4000.0, 6000.0], [4.0, 5.0], [-0.5, 0.5]
    wave = np.arange(3000.0, 8001.0, 100.0)
    shape = np.exp(-((wave - 5500.0) / 2000.0) ** 2) + 0.1
    flux = np.array([[[te / 5000.0 * shape for _ in meta] for _ in logg] for te in teff])
    write_flux_cube(t.tmp / "flux_cube.bin", teff, logg, meta, wave, flux)
    return SEDModel(name="tiny", flux_cube_path=t.tmp / "flux_cube.bin")(5000.0, 4.5, 0.0)


def _bands(t):
    """Box filters johnson_v (5000-6000 Å) and johnson_b (4000-5000 Å)."""
    return tuple(_box_filter(t.tmp / f"johnson_{band}.dat", ANGSTROM, edges=(lo - 100, lo, lo + 1000, lo + 1100))
                 for band, lo in (("v", 5000.0), ("b", 4000.0)))


def magnitudes(results):
    return {name: result.magnitude for name, result in results.items()}


@row
def photometry_takes_a_single_name_path_tuple_as_one_filter(t):
    evaluated, (v, _) = _evaluated(t), _bands(t)
    plain = magnitudes(evaluated.photometry(v))
    assert magnitudes(evaluated.photometry(("V", v))) == {"V": plain["johnson_v"]}
    assert magnitudes(evaluated.photometry(("V", str(v)))) == {"V": plain["johnson_v"]}


@row
def photometry_takes_several_name_path_tuples(t):
    evaluated, (v, b) = _evaluated(t), _bands(t)
    plain = magnitudes(evaluated.photometry(v, b))
    expected = {"V": plain["johnson_v"], "B": plain["johnson_b"]}
    assert magnitudes(evaluated.photometry(("V", v), ("B", b))) == expected
    assert magnitudes(evaluated.photometry((("V", v), ("B", b)))) == expected  # one tuple of specs


@row
def a_lone_pair_of_names_stays_two_filters(t):
    # ("V", "B") must not become filter "B" labelled "V": only a path makes a (name, path) spec.
    from sed_tools.cube.model import _normalize_filter_sequence

    v, _ = _bands(t)
    assert _normalize_filter_sequence([("V", "B")]) == ["V", "B"]
    assert _normalize_filter_sequence([("V", str(v))]) == [("V", str(v))]
    assert _normalize_filter_sequence([("V", v)]) == [("V", v)]


@row
def photometry_plain_paths_and_sequences_of_them_are_unchanged(t):
    evaluated, (v, b) = _evaluated(t), _bands(t)
    plain = magnitudes(evaluated.photometry(str(v), str(b)))
    assert set(plain) == {"johnson_v", "johnson_b"}
    assert plain["johnson_v"] != plain["johnson_b"]
    assert magnitudes(evaluated.photometry(v, b)) == plain
    assert magnitudes(evaluated.photometry([str(v), str(b)])) == plain
    assert magnitudes(evaluated.photometry((v, b))) == plain  # Paths, so not a (name, path) spec


# ----------------------------------------------------------------------
# The flux cube tool's prompts: which cube, which Vega spectrum, the stellar radius; the whole tool
# ----------------------------------------------------------------------
TOOL_WL = np.linspace(3000.0, 9000.0, 61)


def _searched_only_here(t):
    """The cube and Vega searches see an empty working directory and data directory, and no environment hints."""
    import matplotlib

    matplotlib.use("Agg")
    (t.tmp / "work").mkdir()
    t.mp.chdir(t.tmp / "work")
    t.mp.setattr("sed_tools.cube.model.DATA_DIR_DEFAULT", t.tmp / "data")
    for variable in ("SED_FLUX_CUBE", "SED_FLUX_CUBE_DIR", "SED_VEGA_SPECTRUM"):
        t.mp.delenv(variable, raising=False)


def _tool_grid(t):
    """grids/g/flux_cube.bin (flat spectra at two Teff and two log g), with filters B and V in grids/filters/Sys;
    the searches look only in the row's folder."""
    _searched_only_here(t)
    flux = np.ones((2, 2, 1, TOOL_WL.size)) * np.array([1.0, 2.0])[:, None, None, None]
    write_flux_cube(t.tmp / "grids" / "g" / "flux_cube.bin", [5000.0, 6000.0], [4.0, 4.5], [0.0], TOOL_WL, flux)
    system = t.tmp / "grids" / "filters" / "Sys"
    system.mkdir(parents=True)
    for band, (lo, hi) in {"B": (4000, 5000), "V": (5000, 6000)}.items():
        (system / f"{band}.dat").write_text(f"Wavelength,Transmission\n{lo - 100},0\n{lo},1\n{hi},1\n{hi + 100},0\n")
    return t.tmp / "grids" / "g"


def _write_vega(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([TOOL_WL, np.full(TOOL_WL.size, 3.0)]))
    return path


@row
def several_flux_cubes_are_offered_by_number(t):
    grid = _tool_grid(t)
    other = t.tmp / "grids" / "h" / "flux_cube.bin"
    other.parent.mkdir()
    other.write_bytes((grid / "flux_cube.bin").read_bytes())
    answer(t.mp, "", "3", "2")  # Enter lists them again; 3 is not listed
    assert tool.prompt_for_flux_cube_path([t.tmp / "grids"]) == other.resolve()
    out = t.capsys.readouterr().out
    assert out.count("Discovered flux cube files:") == 3
    assert "Invalid selection. Please choose one of the listed numbers." in out
    answer(t.mp, str(grid))  # or a path: a folder holding one cube
    assert tool.prompt_for_flux_cube_path([t.tmp / "grids"]) == grid / "flux_cube.bin"


@row
def with_no_flux_cube_found_a_path_is_asked_for_until_quit(t):
    grid = _tool_grid(t)
    answer(t.mp, "", "missing", str(grid / "flux_cube.bin"))
    assert tool.prompt_for_flux_cube_path() == grid / "flux_cube.bin"
    out = t.capsys.readouterr().out
    assert "Please provide a path to a flux cube file." in out
    assert "Flux cube path 'missing' does not exist" in out
    answer(t.mp, "q")
    with pytest.raises(SystemExit, match="No flux cube selected; aborting."):
        tool.prompt_for_flux_cube_path()


@row
def a_vega_spectrum_given_or_hinted_is_used(t):
    cube = _tool_grid(t) / "flux_cube.bin"
    vega = _write_vega(t.tmp / "ref" / "my_vega.dat")
    assert tool.resolve_vega_spectrum_path(str(vega), cube) == vega.resolve()
    assert tool.resolve_vega_spectrum_path(None, cube, [vega]) == vega.resolve()
    assert tool.resolve_vega_spectrum_path(None, cube, [vega.parent]) == vega.resolve()  # a folder is searched
    band = vega.parent / "B.dat"
    band.write_text("4000 1\n")
    assert tool.resolve_vega_spectrum_path(None, cube, [band]) == vega.resolve()  # and a filter file's folder
    t.mp.setenv("SED_VEGA_SPECTRUM", str(vega))
    assert tool.resolve_vega_spectrum_path(None, cube) == vega.resolve()


@row
def vega_spectra_near_the_cube_are_found_and_offered_by_number(t):
    grid = _tool_grid(t)
    cube = grid / "flux_cube.bin"
    with pytest.raises(FileNotFoundError, match="Could not locate a Vega reference spectrum."):
        tool.resolve_vega_spectrum_path(None, cube)
    first = _write_vega(grid / "vega.dat")
    assert tool.resolve_vega_spectrum_path(None, cube) == first.resolve()
    second = _write_vega(grid.parent / "alpha_lyr_vega.txt")
    answer(t.mp, "x", "3", "2")  # listed nearest first
    assert tool.resolve_vega_spectrum_path(None, cube) == second.resolve()
    out = t.capsys.readouterr().out
    assert "Discovered potential Vega spectra:" in out
    assert out.count("Please select one of the listed Vega spectra.") == 2


#: name: (what is typed, the radius in metres). A bare number is in R☉, as the prompt says (it was taken as metres).
RADII = {"radius_prompt_takes_a_bare_number_in_solar_radii_1_5": ("1.5", 1.5 * RSUN_TO_M),
         "radius_prompt_takes_a_bare_number_in_solar_radii_7e8_m": ("7e8 m", 7e8)}


def _radius(t, typed, metres):
    _searched_only_here(t)
    answer(t.mp, typed)
    assert tool.prompt_radius_m() == metres


@row
def tool_scales_and_measures_vega_magnitudes_with_the_cube_it_finds(t):
    grid = _tool_grid(t)
    _write_vega(grid / "vega.dat")
    answer(t.mp, "y", "10 pc", "1 Rsun")  # scale for distance; the cube, filters and Vega are found
    status = tool.main(["--model-dir", str(grid), "--teff", "5000", "--logg", "4.0", "--metallicity", "0",
                        "--photometric-system", "Vega", "--plot", str(t.tmp / "sed.png"),
                        "--save-sed", str(t.tmp / "sed.txt")])
    assert status == 0
    out = t.capsys.readouterr().out
    scale = (RSUN_TO_M / (10 * PC_TO_M)) ** 2
    assert f"Using flux cube: {(grid / 'flux_cube.bin').resolve()}" in out
    assert f"[scaling] Applied geometric dilution: (R/d)^2 = {scale:.3e}" in out
    assert out.count("Using Vega spectrum:") == 1  # read once for both filters
    assert "Photometric system: Vega" in out and "(system: Sys)" in out
    mag = -2.5 * np.log10(scale / 3.0)  # a flat SED of flux 1, scaled, against a flat Vega of flux 3
    for band in ("Sys/B", "Sys/V"):
        line = next(line for line in out.splitlines() if line.strip().startswith(band))
        assert float(line.split("mag=")[1]) == pytest.approx(mag, abs=1e-3)
    assert np.loadtxt(t.tmp / "sed.txt")[:, 1] == pytest.approx(np.full(TOOL_WL.size, scale))
    assert (t.tmp / "sed.png").is_file()


# ----------------------------------------------------------------------
# Extra-axis collisions and collision_config
# ----------------------------------------------------------------------
@row
def filter_strategy_no_match_raises_clear_error(t):
    model_dir = _collision_grid(t.tmp / "FilterGridBad")
    with pytest.raises(ValueError, match="No exact match"):
        _build(model_dir, override_dict={"on_collision": {"strategy": "filter", "filter": {"alpha": "0.9999"}}})


@row
def extra_key_to_filename_sanitizes_unsafe_characters(t):
    name = _extra_key_to_filename((("[alpha/Fe]", "0.4"), ("f sed", "2.0")))
    assert name == "alpha_Fe_0.4__f_sed_2.0" and "/" not in name and " " not in name


@row
def mesa_prepare_can_list_variants_after_default_split_build(t):
    # mesa_prepare used to read per-variant cubes only from fluxcube_library/ or a master flux_cube.bin, and the
    # default split strategy writes neither, so list_variants() raised on a grid 'sed-tools rebuild' had split.
    model_dir = _collision_grid(t.tmp / "CollisionGrid")
    _build(model_dir)  # the default strategy: split
    variants = list_variants(model_dir)
    assert {v.label for v in variants} == {"alpha_0.0", "alpha_0.4"}


#: name: (the filter spec, the values on offer, the axis; the value matched, and the words of the error, if any)
MATCHES = {
    "match_filter_value_min_numeric": ("min", [0.4, 0.0, -0.2], "alpha", -0.2, None),
    "match_filter_value_max_numeric": ("max", [0.4, 0.0, -0.2], "alpha", 0.4, None),
    "match_filter_value_min_max_string_fallback": ("min", ["h-rich", "he-rich"], "composition", "h-rich", None),
    "match_filter_value_string_exact_match_case_insensitive": (
        "H-RICH", ["h-rich", "he-rich"], "composition", "h-rich", None),
    "match_filter_value_no_match_reports_available": (
        "bogus", ["h-rich", "he-rich"], "composition", None, ("h-rich", "he-rich")),
    "match_filter_value_empty_available_list": ("x", [], "alpha", None, ("No values found",)),
}


def _match(t, spec, values, axis, matched, words):
    got, err = match_filter_value(spec, values, axis)
    assert got == matched
    assert err is None if words is None else all(w in err for w in words)


@row
def collision_config_rejects_invalid_strategy(t):
    with pytest.raises(ValueError, match="Invalid strategy"):
        CollisionConfig(strategy="bogus")


@row
def resolve_filter_values_reports_unmatched_key(t):
    cfg = CollisionConfig(strategy="filter", filter_specs={"bogus_key": "0.4"})
    resolved, errors = cfg.resolve_filter_values({"alpha": [0.0, 0.4]})
    assert resolved == {} and len(errors) == 1 and "bogus_key" in errors[0]


def _root_and_model(t):
    for name in ("root", "model"):
        (t.tmp / name).mkdir()
    return t.tmp / "root", t.tmp / "model"


@row
def load_config_root_dir_overrides_builtin(t):
    root, model = _root_and_model(t)
    (root / "sed_tools.defaults").write_text('[on_collision]\nstrategy = "mean"\n')
    cfg = load_config(model_dir=model, root_dir=root)
    assert (cfg.strategy, cfg.source) == ("mean", str(root / "sed_tools.defaults"))


@row
def global_config_is_copied_to_the_model(t):
    root, model = _root_and_model(t)
    (root / "sed_tools.defaults").write_text('[on_collision]\nstrategy = "split"\n')
    copy_global_config_to_model(root, model)
    assert (model / "sed_tools.defaults").read_text() == (root / "sed_tools.defaults").read_text()


# ----------------------------------------------------------------------
# Interpolation off the cube, and magnitudes of zero or negative flux or zero point
# ----------------------------------------------------------------------
#: name: (the affine cube's axes, fill_gaps, the parameters asked for); SEDModel raises ValueError. A NaN passes
#: the fill_gaps clamp (every comparison is False), and FluxCube._hermite_interp_axis's bounds check raises.
OFF_THE_CUBE = {
    "singleton_axis_raises_with_fill_gaps_false": ({"logg": (3.0,)}, False, (4700.0, 999.0, -1.0)),
    "nan_parameter_raises_even_with_fill_gaps": ({}, True, (float("nan"), 2.0, -0.3)),
}


def _off_the_cube(t, axes, fill_gaps, params):
    model = SEDModel(name="affine", flux_cube_path=affine_cube(t.tmp, **axes).path, fill_gaps=fill_gaps)
    with pytest.raises(ValueError):
        model(teff=params[0], logg=params[1], metallicity=params[2])


def _same(value, expected):
    """``value`` is ``expected``, NaN included."""
    if math.isnan(expected):
        assert math.isnan(value)
    else:
        assert value == expected


#: name: (the band's flux level, the system, the band's Vega zero point; the magnitude both EvaluatedSED.photometry
#: and flux_cube_tool._band_magnitude give, without a warning)
MAGNITUDES = {
    # The Vega path used to raise ValueError from math.log10(0).
    "magnitude_of_zero_flux_is_infinite_in_both_systems_ab": (0.0, "AB", 2.0e-10, math.inf),
    "magnitude_of_zero_flux_is_infinite_in_both_systems_vega": (0.0, "Vega", 2.0e-10, math.inf),
    # The AB path used to return inf for a negative flux.
    "magnitude_of_negative_flux_is_nan_in_both_systems_ab": (-1e-12, "AB", 2.0e-10, math.nan),
    "magnitude_of_negative_flux_is_nan_in_both_systems_vega": (-1e-12, "Vega", 2.0e-10, math.nan),
    # A non-positive Vega zero point used to raise ValueError in both modules.
    **{f"vega_magnitude_with_a_zero_or_negative_zero_point_is_computed_{case}": data for case, data in {
        "zero_point_0": (1e-12, "Vega", 0.0, -math.inf), "zero_point_0_zero_flux": (0.0, "Vega", 0.0, math.nan),
        "zero_point_negative": (1e-12, "Vega", -2.0e-10, math.nan),
        "zero_point_negative_zero_flux": (0.0, "Vega", -2.0e-10, math.inf)}.items()},
}


def _magnitude(t, flux_level, system, zero_point, expected):
    from sed_tools.cube.tool import FilterCurve, _band_magnitude
    from sed_tools.cube.model import EvaluatedSED

    def band():
        return FilterCurve("Band", np.array([3000.0, 4000.0, 5000.0, 6000.0]), np.array([0.0, 1.0, 1.0, 0.0]),
                           {"zp_vega": zero_point})

    wl = np.linspace(2000.0, 8000.0, 300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sed = EvaluatedSED(wl, np.full(wl.size, flux_level), {"teff": 5000.0, "logg": 4.0, "meta": 0.0},
                           types.SimpleNamespace(_resolve_filters=lambda filters, curve=band(): {"Band": curve}))
        for magnitude in (sed.photometry("Band", system=system)["Band"].magnitude,
                          _band_magnitude(wl, np.full(wl.size, flux_level), band(), system, None)[2]):
            _same(magnitude, expected)


#: name: (the flux's scale, the reference flux, the bolometric magnitude), computed without a warning
BOLOMETRIC = {f"bolometric_magnitude_of_non_positive_flux_or_reference_is_computed_{case}": data for case, data in {
    "zero_flux": (0.0, 1.0, math.inf),  # used to be inf by a special case
    "negative_flux": (-1.0, 1.0, math.nan),  # used to be inf by a special case
    "zero_reference": (1.0, 0.0, -math.inf),  # used to raise ZeroDivisionError
    "negative_reference": (1.0, -1.0, math.nan),  # used to raise ValueError from math.log10
    "zero_flux_and_reference": (0.0, 0.0, math.nan),  # used to be inf by a special case
}.items()}


def _bolometric(t, flux_scale, reference_flux, expected):
    wl = np.linspace(2000.0, 8000.0, 300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, mbol = bolometric_magnitude(wl, flux_scale * 3e-12 * (1.0 + wl / 7000.0), reference_flux=reference_flux)
    _same(mbol, expected)


@row
def a_non_numeric_line_inside_a_filter_curve_raises(t):
    # Bad rows used to be dropped silently, removing points from the curve.
    from sed_tools.cube.tool import load_filter_curve

    (t.tmp / "Band.dat").write_text("Wavelength,Transmission\n4000.0,0.5\n4500.0,O.9\n5000.0,0.5\n")
    with pytest.raises(ValueError, match=r"Band\.dat, line 3: expected two numbers, found '4500\.0,O\.9'"):
        load_filter_curve(str(t.tmp / "Band.dat"))


@row
def header_lines_before_the_data_are_still_skipped(t):
    from sed_tools.photometry import load_two_column_file

    path = t.tmp / "vega.csv"
    path.write_text("Vega reference\nwavelength,flux\n4000.0,1e-9\n5000.0,2e-9\n")
    assert load_two_column_file(str(path))[:2] == ([4000.0, 5000.0], [1e-9, 2e-9])


# ----------------------------------------------------------------------
# Resampling, grid coverage, the config's ui table, and finding atmospheres and filters
# ----------------------------------------------------------------------
@row
def read_grid_nodes_fallback_has_path(t):
    from sed_tools.grids.coverage import read_grid_nodes

    spectrum = t.tmp / "model.txt"
    spectrum.write_text("# Teff = 5000\n# logg = 4.0\n# metallicity = 0.0\n1000 1.0\n2000 2.0\n3000 1.5\n")
    nodes = read_grid_nodes(t.tmp)
    assert nodes.loc[0, "filename"] == "model.txt"
    assert nodes.loc[0, "path"] == str(spectrum.resolve())
    # A lookup table's columns are found as everywhere else (lookup.find_*_column) since 0.6.0
    (t.tmp / "lookup_table.csv").write_text("#file_name,T_eff,log_g,[Fe/H]\nmodel.txt,5000,4.0,-0.5\n")
    assert read_grid_nodes(t.tmp).loc[0, ["teff", "logg", "metallicity"]].tolist() == [5000.0, 4.0, -0.5]


@row
def set_data_dir_preserves_nested_ui(t):
    import sed_tools.config as config

    config_file = t.tmp / "config.toml"
    config_file.write_text('[ui]\nterminal_plots = "always"\nterminal_color = "never"\n')
    t.mp.setattr(config, "_CONFIG_DIR", t.tmp)
    t.mp.setattr(config, "_CONFIG_FILE", config_file)
    config.set_data_dir(str(t.tmp / "data"))
    loaded = config.load_user_config()
    assert loaded["ui"] == {"terminal_plots": "always", "terminal_color": "never"}
    assert loaded["data_dir"] == str(t.tmp / "data")


@row
def find_grids_prefers_full_coverage(t):
    root = t.tmp / "stellar_models"  # header-only cubes: find_grids reads only the axes
    _raw_cube(root / "model_a" / "flux_cube.bin", [4000.0, 5000.0, 6000.0], [1.0, 2.0, 3.0], [-1.0, 0.0, 1.0],
              [4000.0, 5000.0])
    _raw_cube(root / "model_b" / "flux_cube.bin", [4500.0, 5200.0], [1.5, 2.5], [-0.5, 0.2], [4000.0, 5000.0])
    sed = SED(model_root=root)
    ranges = {"teff_range": (4600.0, 5500.0), "logg_range": (1.8, 2.4), "metallicity_range": (-0.3, 0.1)}
    matches = sed.find_grids(**ranges)
    assert [m.name for m in matches] == ["model_a"] and matches[0].covers_range is True
    partial = sed.find_grids(**ranges, allow_partial=True)
    assert {m.name for m in partial} == {"model_a", "model_b"} and any(not m.covers_range for m in partial)


@row
def build_filter_index_accepts_string_root(t):
    (t.tmp / "GAIA" / "GAIA").mkdir(parents=True)
    (t.tmp / "GAIA" / "GAIA" / "Gbp.dat").write_text("Wavelength,Transmission\n5000,1\n")
    index = _build_filter_index(str(t.tmp))
    assert "gbp" in index and index["gbp"][0].name == "Gbp.dat"


@row
def sedmodel_resolves_nested_instrument_directory_from_name(t):
    cube = _raw_cube(t.tmp / "flux_cube.bin", [5000.0], [4.0], [0.0], [5000.0, 6000.0])
    filters = t.tmp / "filters" / "GAIA" / "GAIA"
    filters.mkdir(parents=True)
    for band in ("Gbp", "Grp"):
        (filters / f"{band}.dat").write_text("Wavelength,Transmission\n5000,1\n6000,1\n")
    model = SEDModel(name="test", flux_cube_path=str(cube), filters_dir=str(t.tmp / "filters"))
    assert sorted(p.name for p in model._locate_filter_paths("GAIA")) == ["Gbp.dat", "Grp.dat"]


# ----------------------------------------------------------------------
# Band sampling: coverage --filters names the bands a grid's wavelengths do not sample (MIRI on ATLAS9's step)
# ----------------------------------------------------------------------
#: ATLAS9's sampling in miniature: 40 Å steps to 10,000 Å, then 20,000 and 40,000 Å.
STEP_AXIS = np.concatenate([np.arange(1000.0, 10001.0, 40.0), [20000.0, 40000.0]])
#: name: (the box band's edges in Å, grid wavelengths where it transmits, whether STEP_AXIS samples it)
BANDS = {
    "band_sampling_dense": ((4000.0, 6000.0), 51, True),
    # MIRI F1130W-F1550C and F2550W on Kurucz2003all: MESA's band average is empty, its "no data" placeholder.
    "band_sampling_inside_the_step": ((12000.0, 15000.0), 0, False),
    # F1800W-F2300C and WISE W4: MESA's band average is the one flux sample.
    "band_sampling_one_sample": ((18000.0, 22000.0), 1, False),
    # F1000W and WISE W3: half or more of the band lies in the step.
    "band_sampling_half_in_the_step": ((9000.0, 11000.0), 26, False),
    # A narrow band between two samples of a fine stretch (HST F656N on MESA's bbody grid).
    "band_sampling_between_two_samples": ((4005.0, 4035.0), 0, False),
    "band_sampling_past_the_end": ((30000.0, 50000.0), 1, False),
}


def _box_band(name, lo, hi):
    wavelength = np.linspace(lo, hi, 101)
    return tool.FilterCurve(name, wavelength, np.ones_like(wavelength), {})


def _band_sampling(t, edges, inside, sampled):
    from sed_tools.photometry import band_sampling

    sampling = band_sampling(STEP_AXIS, _box_band("Band", *edges))
    assert (sampling.inside, sampling.sampled) == (inside, sampled), sampling


def _step_grid(root):
    """A grid folder whose flux cube has STEP_AXIS, and a filter folder: one band dense, one inside the step."""
    cube = root / "Step" / "flux_cube.bin"
    write_flux_cube(cube, [5000.0, 6000.0], [4.0], [0.0], STEP_AXIS, np.ones((2, 1, 1, STEP_AXIS.size)))
    (cube.parent / "lookup_table.csv").write_text("file_name,teff,logg,metallicity\na.txt,5000,4,0\nb.txt,6000,4,0\n")
    filters = root / "filters" / "Cam"
    filters.mkdir(parents=True)
    for name, (lo, hi) in (("Dense", (4000.0, 6000.0)), ("Step", (12000.0, 15000.0))):
        (filters / f"{name}.dat").write_text(f"{ANGSTROM}\n" + "".join(f"{w:.1f} 1.0\n" for w in np.linspace(lo, hi, 11)))
    return cube, filters


@row
def coverage_with_filters_exits_1_naming_the_bands_the_grid_does_not_sample(t):
    from support import run_cli

    _, filters = _step_grid(t.tmp)
    assert run_cli(t.mp, "coverage", "--base", str(t.tmp), "--grids", "Step", "--no-plot",
                   "--filters", str(filters)) == 1
    out = t.capsys.readouterr().out
    assert "Cam/Step: 0 grid wavelengths where it transmits, 100% where lambda/delta-lambda < 16, " \
           "as between 10000 and 20000 Å" in out
    assert "Cam/Dense" not in out.split("Band sampling of Step")[1]


@row
def cube_build_says_which_spectra_were_put_onto_its_axis(t):
    # The axis is the union of five sampled spectra (as NextGen2's is); the sixth has one wavelength more.
    model_dir = _write_grid(t.tmp / "Step", [(f"s{i}.txt", 5000.0 + 100 * i, 4.0, 0.0) for i in range(6)])
    for i in range(6):
        wavelengths = STEP_AXIS if i < 5 else np.sort(np.append(STEP_AXIS, 15000.0))
        (model_dir / f"s{i}.txt").write_text("".join(f"{w:.1f} 1.0e-10\n" for w in wavelengths))
    _build(model_dir)
    out = t.capsys.readouterr().out
    assert "1 of them have wavelengths that are not on the cube's axis" in out


def test_cubes(rows):
    rows(SCENARIOS)
    rows(BANDS, _band_sampling)
    rows(UNUSABLE_LOOKUP, _unusable_lookup)
    rows(RAM, _ram)
    rows(SEARCHES, _unsearchable)
    rows(COMBINE_MISSING, _combine_missing)
    rows(DENSIFIER_CHOICES, _densifier_rejects)
    rows(GOES_BACK, _goes_back)
    rows(RADII, _radius)
    rows(MATCHES, _match)
    rows(OFF_THE_CUBE, _off_the_cube)
    rows(MAGNITUDES, _magnitude)
    rows(BOLOMETRIC, _bolometric)
