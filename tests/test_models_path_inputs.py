from pathlib import Path
import struct

import numpy as np

from sed_tools.models import SEDModel, _build_filter_index


def _write_minimal_flux_cube(path: Path) -> None:
    with path.open("wb") as fh:
        fh.write(struct.pack("4i", 1, 1, 1, 2))
        np.asarray([5000.0], dtype=np.float64).tofile(fh)
        np.asarray([4.0], dtype=np.float64).tofile(fh)
        np.asarray([0.0], dtype=np.float64).tofile(fh)
        np.asarray([5000.0, 6000.0], dtype=np.float64).tofile(fh)


def test_build_filter_index_accepts_string_root(tmp_path):
    filt = tmp_path / "GAIA" / "GAIA"
    filt.mkdir(parents=True)
    (filt / "Gbp.dat").write_text("Wavelength,Transmission\n5000,1\n")

    index = _build_filter_index(str(tmp_path))

    assert "gbp" in index
    assert index["gbp"][0].name == "Gbp.dat"


def test_sedmodel_converts_string_filters_dir_to_path(tmp_path):
    cube = tmp_path / "flux_cube.bin"
    _write_minimal_flux_cube(cube)

    model = SEDModel(
        name="test",
        flux_cube_path=str(cube),
        filters_dir=str(tmp_path),
    )

    assert isinstance(model.filters_dir, Path)
    assert model.filters_dir == tmp_path


def test_sedmodel_resolves_nested_instrument_directory_from_name(tmp_path):
    cube = tmp_path / "flux_cube.bin"
    _write_minimal_flux_cube(cube)

    filt = tmp_path / "filters" / "GAIA" / "GAIA"
    filt.mkdir(parents=True)
    (filt / "Gbp.dat").write_text("Wavelength,Transmission\n5000,1\n6000,1\n")
    (filt / "Grp.dat").write_text("Wavelength,Transmission\n5000,1\n6000,1\n")

    model = SEDModel(
        name="test",
        flux_cube_path=str(cube),
        filters_dir=str(tmp_path / "filters"),
    )

    paths = model._locate_filter_paths("GAIA")

    assert sorted(p.name for p in paths) == ["Gbp.dat", "Grp.dat"]


def test_sedmodel_resolves_filter_group_to_multiple_curves(tmp_path):
    cube = tmp_path / "flux_cube.bin"
    _write_minimal_flux_cube(cube)

    filt = tmp_path / "filters" / "GAIA" / "GAIA"
    filt.mkdir(parents=True)
    (filt / "Gbp.dat").write_text("Wavelength,Transmission\n5000,1\n6000,1\n")
    (filt / "Grp.dat").write_text("Wavelength,Transmission\n5000,1\n6000,1\n")

    model = SEDModel(
        name="test",
        flux_cube_path=str(cube),
        filters_dir=str(tmp_path / "filters"),
    )

    resolved = model._resolve_filters(["GAIA"])

    assert sorted(resolved) == ["Gbp", "Grp"]
