from pathlib import Path

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


def test_remove_empty_filter_dir_removes_only_partial_leaf(tmp_path):
    path = tmp_path / "WISE" / "WISE"
    path.mkdir(parents=True)
    (path / "WISE").write_text("W1.dat\n")

    Filters._remove_empty_filter_dir(path)

    assert not path.exists()
    assert not (tmp_path / "WISE").exists()


def test_remove_empty_filter_dir_keeps_real_filter_dir(tmp_path):
    path = tmp_path / "WISE" / "WISE"
    path.mkdir(parents=True)
    (path / "W1.dat").write_text("Wavelength,Transmission\n1,1\n")

    Filters._remove_empty_filter_dir(path)

    assert path.exists()
    assert (path / "W1.dat").exists()


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


def test_combine_filter_sets_module_function_matches_api_layout(tmp_path):
    from sed_tools.combine_filters import combine_filter_sets

    source = tmp_path / "Generic" / "Johnson"
    source.mkdir(parents=True)
    (source / "B.dat").write_text("Wavelength,Transmission\n1,1\n")

    out = combine_filter_sets("Optical", ["Generic/Johnson"], filter_root=tmp_path)

    assert out == tmp_path / "Combined" / "Optical"
    assert (out / "B.dat").exists()
    assert (out / "Optical").read_text().splitlines() == ["B.dat"]
