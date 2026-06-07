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
