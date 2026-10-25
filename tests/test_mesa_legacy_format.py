"""Filter curves stay readable by released MESA colors (25.12.1, 26.4.1).

Released MESA's ``load_filter`` discards the first line, sizes its arrays from
every remaining line, then skips leading ``#`` lines plus the first other line
before reading samples.  Any extra line therefore leaves an uninitialised
sample at the end of the curve.  The column-name row is the one free line, so
SED_Tools metadata lives there as a third, whitespace-free CSV field.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests
from astropy.io import ascii

from sed_tools import cli
from sed_tools._flux import load_filter_curve
from sed_tools.api import Filters
from sed_tools.combine_filters import combine_filter_sets
from sed_tools.filter_components import InstrumentManifest, add_component, build_product
from sed_tools.filter_library import FilterLibrary
from sed_tools.filter_metadata import (PHOTON, SOURCE_SVO, SOURCE_SVO_MATCH, has_column_row,
                                       migrate_filter_file, read_filter_header, tag_filter_file,
                                       write_filter_file, write_filter_header)
from sed_tools.flux_cube_tool import load_two_column_file

# A curve exactly as SED_Tools 0.3.0 and the NJM mirror published it.
TAGGED_030 = (
    "# FilterID: GAIA/GAIA3.G\n"
    "# DetectorType: photon\n"
    "# DetectorTypeSource: svo-curve-match\n"
    "# WavelengthUnit: Angstrom\n"
    "# Components: filter+qe+instrument\n"
    "# ComponentsSource: svo\n"
    "Wavelength,Transmission\n"
    "3200.0,2.37366962e-08\n"
    "6000.0,0.75\n"
    "10500.0,1.1e-05\n"
)
BODY_030 = TAGGED_030.split("Wavelength,Transmission\n")[1]
ROW_G = ("Wavelength,Transmission,FilterID=GAIA/GAIA3.G;DetectorType=photon;DetectorTypeSource=svo-curve-match;"
         "WavelengthUnit=Angstrom;Components=filter+qe+instrument;ComponentsSource=svo")


def released_mesa_load_filter(path):
    """Transcription of ``load_filter`` from MESA 26.4.1 ``colors_utils.f90``.

    Returns ``(allocated, samples)``: the array size MESA allocates and the
    samples it actually reads into it.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    allocated = len(lines) - 1  # line 1 discarded, every later line counted
    i = 0
    while i < len(lines) and lines[i][:1] == "#":
        i += 1
    i += 1  # the first non-'#' line is consumed as well
    samples = []
    for line in lines[i:]:
        try:
            wl, tr = (float(v) for v in line.replace(",", " ").split()[:2])
        except ValueError:
            break
        samples.append((wl, tr))
    return allocated, samples


def mesa_plot_unit_test_load(path):
    """Loop of ``load_filters`` in MESA 26.4.1 ``colors/test/python_helpers/plot_unit_test.py``."""
    wav = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                wav.append(float(parts[0]))
    return wav


def assert_legacy_readers(path):
    """Every known consumer of SED_Tools curves reads all samples of ``path``."""
    allocated, samples = released_mesa_load_filter(path)
    assert allocated == len(samples), f"{path}: MESA would allocate {allocated} for {len(samples)} samples"
    expected = np.array(samples).T
    mesa_plot_unit_test_load(path)  # whitespace-split: must not raise on the column row
    np.testing.assert_allclose(load_two_column_file(str(path))[:2], expected)  # SED_Tools <= 0.3.0
    np.testing.assert_allclose(np.loadtxt(path, delimiter=",", skiprows=1).T, expected)
    np.testing.assert_allclose([pd.read_csv(path)["Wavelength"], pd.read_csv(path)["Transmission"]], expected)
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    np.testing.assert_allclose([[float(r["Wavelength"]) for r in rows],
                                [float(r["Transmission"]) for r in rows]], expected)
    table = ascii.read(path, format="csv")
    np.testing.assert_allclose([table["Wavelength"], table["Transmission"]], expected)


def test_emulation_detects_the_030_defect(tmp_path):
    path = tmp_path / "G.dat"
    path.write_text(TAGGED_030)
    allocated, samples = released_mesa_load_filter(path)
    assert (allocated, len(samples)) == (9, 3)


def test_written_curve_carries_metadata_on_its_column_row(tmp_path):
    path = tmp_path / "G.dat"
    write_filter_file(path, [4000.0, 5000.0], [0.1, 0.9], detector_type=PHOTON, source=SOURCE_SVO,
                      filter_id="GAIA/GAIA3.G", wavelength_unit="Angstrom")
    assert path.read_text() == (
        "Wavelength,Transmission,FilterID=GAIA/GAIA3.G;DetectorType=photon;DetectorTypeSource=svo;"
        "WavelengthUnit=Angstrom\n4000.0,0.1\n5000.0,0.9\n")
    assert read_filter_header(path) == {"FilterID": "GAIA/GAIA3.G", "DetectorType": "photon",
                                        "DetectorTypeSource": "svo", "WavelengthUnit": "Angstrom"}
    assert_legacy_readers(path)


def test_curve_without_metadata_is_byte_identical_to_020(tmp_path):
    legacy = b"Wavelength,Transmission\n4000.0,0.1\n5000.0,0.9\n"
    path = tmp_path / "V.dat"
    path.write_bytes(legacy)
    assert read_filter_header(path) == {}
    write_filter_header(path, {"DetectorType": "energy"})
    assert path.read_bytes() == b"Wavelength,Transmission,DetectorType=energy\n4000.0,0.1\n5000.0,0.9\n"
    write_filter_header(path, {})
    assert path.read_bytes() == legacy


def test_values_are_escaped_onto_one_whitespace_free_field(tmp_path):
    path = tmp_path / "A.dat"
    fields = {"DetectorType": "energy", "Atmosphere": "site=paranal airmass=1.2 pwv=2.5",
              "Sources": 'ESO SkyCalc; Noll et al., 2012 "v2" 100% #3', "Recipe": "filter+qe @ Å\tx"}
    write_filter_file(path, [1.0, 2.0], [0.5, 0.5], detector_type="energy", source="component",
                      extra={k: v for k, v in fields.items() if k != "DetectorType"})
    row = path.read_text(encoding="utf-8").splitlines()[0]
    assert len(row.split()) == 1 and len(row.split(",")) == 3 and "#" not in row
    assert "Atmosphere=site=paranal%20airmass=1.2%20pwv=2.5" in row
    assert read_filter_header(path) == {**fields, "DetectorTypeSource": "component"}
    assert_legacy_readers(path)


def test_migration_moves_fields_and_keeps_samples_byte_for_byte(tmp_path):
    path = tmp_path / "G.dat"
    path.write_text(TAGGED_030)
    before = read_filter_header(path)

    assert migrate_filter_file(path, dry_run=True)
    assert path.read_text() == TAGGED_030
    assert migrate_filter_file(path)

    assert path.read_text() == ROW_G + "\n" + BODY_030
    assert read_filter_header(path) == before
    assert not migrate_filter_file(path)  # idempotent
    assert_legacy_readers(path)


def test_migration_keeps_foreign_lines_and_crlf(tmp_path):
    path = tmp_path / "X.dat"
    path.write_bytes(b"# DetectorType: energy\r\n# measured by hand\r\nWavelength,Transmission\r\n1,2\r\n")
    migrate_filter_file(path)
    assert path.read_bytes() == b"# measured by hand\r\nWavelength,Transmission,DetectorType=energy\r\n1,2\r\n"


def test_curve_without_column_row_is_never_given_one(tmp_path):
    raw = b"3830.0 0.000000\n4000.0 0.5\n"  # MESA has always discarded the first sample here
    path = tmp_path / "raw.dat"
    path.write_bytes(raw)
    assert not has_column_row(path)
    assert write_filter_header(path, {"DetectorType": "photon"}) is False
    assert path.read_bytes() == raw

    path.write_bytes(b"# DetectorType: photon\n# DetectorTypeSource: assumed\n" + raw)  # tagged by 0.3.0
    assert migrate_filter_file(path)
    assert path.read_bytes() == raw


def test_a_real_third_column_is_not_taken_for_metadata(tmp_path):
    path = tmp_path / "E.dat"
    path.write_text("Wavelength,Transmission,Error\n1,2,0.1\n")
    assert read_filter_header(path) == {}
    write_filter_header(path, {"DetectorType": "photon"})
    assert path.read_text() == "Wavelength,Transmission,Error,DetectorType=photon\n1,2,0.1\n"
    assert read_filter_header(path) == {"DetectorType": "photon"}


def test_migrated_photon_curve_still_loads_as_photon(tmp_path):
    path = tmp_path / "G.dat"
    path.write_text(TAGGED_030)
    migrate_filter_file(path)
    curve = load_filter_curve(str(path))
    assert curve.detector_type == PHOTON
    assert curve.metadata["filterid"] == "GAIA/GAIA3.G"
    assert curve.metadata["wavelength_unit_source"] == "header"
    np.testing.assert_allclose(curve.wavelength, [3200.0, 6000.0, 10500.0])


def test_column_row_overrides_030_lines(tmp_path):
    path = tmp_path / "G.dat"
    path.write_text(TAGGED_030.replace("Wavelength,Transmission\n", "Wavelength,Transmission,DetectorType=energy\n"))
    assert read_filter_header(path)["DetectorType"] == "energy"
    migrate_filter_file(path)
    assert read_filter_header(path)["DetectorType"] == "energy"
    assert read_filter_header(path)["FilterID"] == "GAIA/GAIA3.G"


def test_tagging_an_already_tagged_030_curve_migrates_it(tmp_path):
    class NoSVO:
        def get_filter(self, filter_id):
            raise AssertionError("an authoritative tag needs no lookup")

    path = tmp_path / "GAIA" / "GAIA" / "G.dat"
    path.parent.mkdir(parents=True)
    path.write_text(TAGGED_030)
    result = tag_filter_file(path, client=NoSVO())
    assert (result.detector_type, result.source, result.changed, result.migrated) == (
        PHOTON, SOURCE_SVO_MATCH, False, True)
    assert_legacy_readers(path)


def _instrument(root):
    directory = root / "Test" / "Camera"
    directory.mkdir(parents=True)
    for band in ("g", "r"):
        write_filter_file(directory / f"{band}.dat", [4000., 5000., 6000.], [0.2, 0.8, 0.2],
                          detector_type=PHOTON, source="svo", wavelength_unit="Angstrom",
                          extra={"Components": "filter+optics+qe"})
    manifest = InstrumentManifest.load(directory)
    for band in ("g", "r"):
        add_component(directory, "filter", [4000., 5000., 6000.], [0.1, 1., 0.1], band=band, manifest=manifest)
    add_component(directory, "qe", [3000., 7000.], [0.8, 0.6], manifest=manifest)
    manifest.save(directory)
    return directory


def test_components_and_products_are_legacy(tmp_path):
    directory = _instrument(tmp_path / "filters")
    product = build_product(directory, "filter+qe")
    curves = sorted(directory.rglob("*.dat"))
    assert any("components" in p.parts for p in curves) and any("products" in p.parts for p in curves)
    for path in curves:
        assert_legacy_readers(path)
        assert read_filter_header(path)["DetectorType"]
    assert (product / product.name).read_text().split() == ["g.dat", "r.dat"]


def test_combined_set_repairs_030_inputs_without_touching_them(tmp_path):
    root = tmp_path / "filters"
    first = root / "GAIA" / "GAIA"
    first.mkdir(parents=True)
    (first / "G.dat").write_text(TAGGED_030)
    second = _instrument(tmp_path / "other")

    out = combine_filter_sets("Mixed", [first, second], filter_root=root)

    assert (first / "G.dat").read_text() == TAGGED_030
    for path in out.glob("*.dat"):
        assert_legacy_readers(path)
    assert read_filter_header(out / "G.dat")["FilterID"] == "GAIA/GAIA3.G"
    assert read_filter_header(out / "g.dat")["DetectorType"] == PHOTON
    assert sorted((out / "Mixed").read_text().split()) == sorted(p.name for p in out.glob("*.dat"))


@pytest.mark.parametrize("served", [TAGGED_030, ROW_G + "\n" + BODY_030], ids=["0.3.0 mirror", "migrated mirror"])
def test_mirror_download_is_legacy_either_way(tmp_path, monkeypatch, served):
    from sed_tools.njm_filter_grabber import NJMFilterGrabber

    def get(url, **kwargs):
        if url.endswith("/G.dat"):
            return SimpleNamespace(text=served)
        if url.endswith("/GAIA"):
            return SimpleNamespace(text="G.dat\n")
        raise requests.HTTPError(response=SimpleNamespace(status_code=404))

    grabber = NJMFilterGrabber(base_dir=str(tmp_path))
    grabber._available = True
    monkeypatch.setattr(grabber, "_get", get)
    monkeypatch.setattr(grabber, "download_components", lambda facility, instrument: 0)
    monkeypatch.setattr("sed_tools.njm_filter_grabber.tag_filter_file", lambda path, **kw: tag_filter_file(
        path, client=SimpleNamespace(get_filter=lambda fid: pytest.fail("no SVO lookup needed"))))

    assert grabber.download_filters("GAIA", "GAIA", ["G"]) == 1
    path = tmp_path / "GAIA" / "GAIA" / "G.dat"
    assert path.read_text() == ROW_G + "\n" + BODY_030
    assert_legacy_readers(path)


def test_cached_030_set_is_repaired_before_use(tmp_path):
    directory = _instrument(tmp_path / "filters")
    original = {}
    for path in directory.rglob("*.dat"):
        original[path] = read_filter_header(path)
        samples = path.read_text().splitlines(keepends=True)[1:]
        legacy = "".join(f"# {key}: {value}\n" for key, value in original[path].items())
        path.write_text(legacy + "Wavelength,Transmission\n" + "".join(samples))

    assert FilterLibrary(tmp_path / "filters").published("Test", "Camera") == directory
    for path, fields in original.items():
        assert_legacy_readers(path)
        assert read_filter_header(path) == fields


def test_migrate_command_is_offline_and_reports(tmp_path, monkeypatch, capsys):
    root = tmp_path / "filters" / "GAIA" / "GAIA"
    root.mkdir(parents=True)
    (root / "G.dat").write_text(TAGGED_030)
    (root / "Bp.dat").write_text("Wavelength,Transmission\n1,1\n2,1\n")
    (root / "raw.dat").write_text("# DetectorType: photon\n3830 0\n4000 1\n")
    monkeypatch.setattr(requests.Session, "request", lambda *a, **k: pytest.fail("migration is offline"))

    assert Filters.migrate_metadata(tmp_path / "filters", dry_run=True) == [str(root / "G.dat"), str(root / "raw.dat")]
    assert (root / "G.dat").read_text() == TAGGED_030

    monkeypatch.setattr("sys.argv", ["sed-tools", "filters-migrate", str(tmp_path / "filters")])
    assert cli.main() in (0, None)
    out = capsys.readouterr().out
    assert "2 curves: moved 0.3.0 '#' metadata to the column row" in out
    assert "1 have no column-name row" in out
    assert_legacy_readers(root / "G.dat")
    assert (root / "raw.dat").read_text() == "3830 0\n4000 1\n"
    assert Filters.migrate_metadata(tmp_path / "filters") == []


def test_migrate_resolves_paths_against_the_filter_root(tmp_path, monkeypatch):
    root = tmp_path / "filters"
    (root / "GAIA" / "GAIA").mkdir(parents=True)
    (root / "GAIA" / "GAIA" / "G.dat").write_text(TAGGED_030)
    monkeypatch.setattr(Filters, "_filter_root", root)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        Filters.migrate_metadata("GAIA/Missing")
    assert Filters.migrate_metadata("GAIA/GAIA") == [str(root / "GAIA" / "GAIA" / "G.dat")]
