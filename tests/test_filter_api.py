import json
import logging
from pathlib import Path

import pytest

from sed_tools.api import Filters


def test_filters_query_local_ignores_empty_filter_dirs(tmp_path, monkeypatch):
    empty = tmp_path / "WISE" / "WISE"
    empty.mkdir(parents=True)
    (empty / "WISE").write_text("W1.dat\nW2.dat\n")

    monkeypatch.setattr(Filters, "_filter_root", tmp_path)

    assert Filters.query(include_local=True, include_remote=False) == []


def test_filters_query_local_reports_dat_filter_dirs(tmp_path, monkeypatch):
    filt_dir = tmp_path / "WISE" / "WISE"
    filt_dir.mkdir(parents=True)
    (filt_dir / "W1.dat").write_text("Wavelength,Transmission\n1,1\n")
    (filt_dir / "W2.dat").write_text("Wavelength,Transmission\n2,1\n")

    monkeypatch.setattr(Filters, "_filter_root", tmp_path)

    assert Filters.query(include_local=True, include_remote=False) == [
        {
            "facility": "WISE",
            "instrument": "WISE",
            "n_filters": 2,
            "is_local": True,
        }
    ]


def test_filters_combine_creates_mesa_index_file(tmp_path):
    gaia = tmp_path / "GAIA" / "GAIA"
    twomass = tmp_path / "2MASS" / "2MASS"
    gaia.mkdir(parents=True)
    twomass.mkdir(parents=True)
    (gaia / "G.dat").write_text("Wavelength,Transmission\n5000,1\n")
    (twomass / "J.dat").write_text("Wavelength,Transmission\n12000,1\n")

    out = Filters.combine("Gaia2MASS", "GAIA/GAIA", "2MASS/2MASS", filter_root=tmp_path)

    assert out == tmp_path / "Combined" / "Gaia2MASS"
    assert sorted(p.name for p in out.glob("*.dat")) == ["G.dat", "J.dat"]
    assert (out / "Gaia2MASS").read_text().splitlines() == ["G.dat", "J.dat"]


def test_filters_combine_renames_conflicting_filter_names(tmp_path):
    first = tmp_path / "A" / "Inst"
    second = tmp_path / "B" / "Inst"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "V.dat").write_text("Wavelength,Transmission\n1,1\n")
    (second / "V.dat").write_text("Wavelength,Transmission\n2,1\n")

    out = Filters.combine("Combined/AB", first, second, filter_root=tmp_path)

    names = sorted(p.name for p in out.glob("*.dat"))
    assert names == ["B_Inst_V.dat", "V.dat"]
    assert (out / "AB").read_text().splitlines() == names


def test_combine_renamed_band_has_no_dot_before_dat(tmp_path):
    # MESA names a band after the text before the first dot of its file name:
    # Old_Sony.A200_V.dat would become the column "Old_Sony".
    for facility in ("Cameras", "Old"):
        inst = tmp_path / facility / "Sony.A200"
        inst.mkdir(parents=True)
        (inst / "V.dat").write_text(f"Wavelength,Transmission\n{len(facility)},1\n")

    out = Filters.combine("Combined/Sony", tmp_path / "Cameras/Sony.A200", tmp_path / "Old/Sony.A200",
                          filter_root=tmp_path)

    names = sorted(p.name for p in out.glob("*.dat"))
    assert names == ["Old_SonypA200_V.dat", "V.dat"]
    assert all("." not in name[: -len(".dat")] for name in names)
    assert (out / "Sony").read_text().splitlines() == names
    assert (out / "Old_SonypA200_V.dat").read_text() == "Wavelength,Transmission\n3,1\n"


def test_combine_leaves_out_archived_and_unlisted_copies(tmp_path, caplog):
    # As NJM serves Euclid/NISP: H_svo is under archived_bands. Y_old is a copy
    # the index does not list and the manifest does not archive.
    nisp = tmp_path / "Euclid" / "NISP"
    nisp.mkdir(parents=True)
    for band in ("H", "Y", "H_svo", "Y_old"):
        (nisp / f"{band}.dat").write_text("Wavelength,Transmission\n1,1\n")
    (nisp / "NISP").write_text("H.dat\nY.dat\n")
    (nisp / "manifest.json").write_text(json.dumps(
        {"facility": "Euclid", "instrument": "NISP", "bands": ["H", "Y"], "archived_bands": {"H_svo": {}}}))
    # Without an index file, every curve but the archived ones.
    bessell = tmp_path / "Generic" / "Bessell"
    bessell.mkdir(parents=True)
    for band in ("B", "B_svo"):
        (bessell / f"{band}.dat").write_text("Wavelength,Transmission\n2,1\n")
    (bessell / "manifest.json").write_text(json.dumps(
        {"facility": "Generic", "instrument": "Bessell", "bands": ["B"], "archived_bands": ["B_svo"]}))

    with caplog.at_level(logging.WARNING, logger="sed_tools.combine_filters"):
        out = Filters.combine("Combined/NISPB", nisp, bessell, filter_root=tmp_path)

    assert sorted(p.name for p in out.glob("*.dat")) == ["B.dat", "H.dat", "Y.dat"]
    assert (out / "NISPB").read_text().splitlines() == ["B.dat", "H.dat", "Y.dat"]
    assert "leaving out Y_old.dat, which its index file NISP does not list" in caplog.text
    assert "H_svo" not in caplog.text and "B_svo" not in caplog.text


def test_combine_refuses_an_index_listing_a_missing_curve(tmp_path):
    inst = tmp_path / "Generic" / "Johnson"
    inst.mkdir(parents=True)
    (inst / "B.dat").write_text("Wavelength,Transmission\n1,1\n")
    (inst / "Johnson").write_text("B.dat\nV.dat\n")

    with pytest.raises(FileNotFoundError, match="lists V.dat, not found in"):
        Filters.combine("Combined/J", inst, filter_root=tmp_path)


def test_combine_filter_sets_module_function_matches_api_layout(tmp_path):
    from sed_tools.combine_filters import combine_filter_sets

    source = tmp_path / "Generic" / "Johnson"
    source.mkdir(parents=True)
    (source / "B.dat").write_text("Wavelength,Transmission\n1,1\n")

    out = combine_filter_sets("Optical", ["Generic/Johnson"], filter_root=tmp_path)

    assert out == tmp_path / "Combined" / "Optical"
    assert (out / "B.dat").exists()
    assert (out / "Optical").read_text().splitlines() == ["B.dat"]
