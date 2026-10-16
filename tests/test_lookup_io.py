"""Tests for ``sed_tools._lookup_io``: column detection and CSV writing.

All grabbers, ``precompute_flux_cube``, and ``spectra_cleaner`` are
documented to import from this module rather than reimplementing column
detection or CSV writing, so its correctness matters project-wide.
"""

import csv

import pytest

from sed_tools._lookup_io import (
    find_column,
    find_file_column,
    find_logg_column,
    find_metallicity_column,
    find_teff_column,
    write_lookup_csv,
)


# ---------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "keys,expected",
    [
        (["Teff", "logg"], "Teff"),
        (["T_EFF", "x"], "T_EFF"),
        (["temperature"], "temperature"),
        (["temp"], "temp"),
    ],
)
def test_find_teff_column_case_insensitive(keys, expected):
    assert find_teff_column(keys) == expected


@pytest.mark.parametrize(
    "keys,expected",
    [
        (["log(g)"], "log(g)"),
        (["LOG_G"], "LOG_G"),
        (["surface_gravity"], "surface_gravity"),
        (["gravity"], "gravity"),
    ],
)
def test_find_logg_column_case_insensitive(keys, expected):
    assert find_logg_column(keys) == expected


@pytest.mark.parametrize(
    "keys,expected",
    [
        (["[Fe/H]"], "[Fe/H]"),
        (["Z"], "Z"),  # TLUSTY OSTAR/BSTAR convention
        (["metallicity"], "metallicity"),
        (["m_h"], "m_h"),
    ],
)
def test_find_metallicity_column_case_insensitive(keys, expected):
    assert find_metallicity_column(keys) == expected


def test_find_file_column():
    assert find_file_column(["FileName", "teff"]) == "FileName"


def test_find_column_returns_none_when_no_candidate_matches():
    assert find_column(["x", "y"], ["teff"]) is None


def test_find_column_prefers_earlier_candidate_over_later_one():
    # Both "teff" and "temperature" are present; "teff" is earlier in the
    # candidate list and should win.
    keys = ["temperature", "teff"]
    assert find_teff_column(keys) == "teff"


# ---------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------

def _read_hash_prefixed_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as fh:
        header = fh.readline().lstrip("#").strip()
        cols = [c.strip() for c in header.split(",")]
        rows = list(csv.reader(fh))
    return cols, rows


def test_write_lookup_csv_dict_of_lists_round_trip(tmp_path):
    path = tmp_path / "lookup.csv"
    n = write_lookup_csv({"file_name": ["a.txt", "b.txt"], "teff": [4000, 5000]}, str(path))

    assert n == 2
    cols, rows = _read_hash_prefixed_csv(path)
    assert cols == ["file_name", "teff"]
    assert rows == [["a.txt", "4000"], ["b.txt", "5000"]]


def test_write_lookup_csv_list_of_dicts_round_trip(tmp_path):
    path = tmp_path / "lookup.csv"
    records = [{"file_name": "a.txt", "teff": 4000}, {"file_name": "b.txt", "teff": 5000}]
    n = write_lookup_csv(records, str(path))

    assert n == 2
    cols, rows = _read_hash_prefixed_csv(path)
    assert cols == ["file_name", "teff"]
    assert rows == [["a.txt", "4000"], ["b.txt", "5000"]]


def test_write_lookup_csv_explicit_column_order(tmp_path):
    path = tmp_path / "lookup.csv"
    records = [{"file_name": "a.txt", "teff": 4000, "extra_ignored": "z"}]
    write_lookup_csv(records, str(path), columns=["teff", "file_name"])

    cols, rows = _read_hash_prefixed_csv(path)
    assert cols == ["teff", "file_name"]
    assert rows == [["4000", "a.txt"]]


def test_write_lookup_csv_header_is_hash_prefixed(tmp_path):
    path = tmp_path / "lookup.csv"
    write_lookup_csv({"file_name": ["a.txt"]}, str(path))
    first_line = path.read_text().splitlines()[0]
    assert first_line.startswith("#")


def test_write_lookup_csv_empty_records_writes_header_only(tmp_path):
    path = tmp_path / "lookup.csv"
    n = write_lookup_csv({}, str(path))
    assert n == 0
    assert path.read_text() == "#\n"
