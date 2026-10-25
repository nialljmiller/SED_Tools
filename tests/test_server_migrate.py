"""Server tooling: migrate_filters.py converts mirror trees written by SED_Tools 0.3.0."""

import importlib.util
from pathlib import Path

import pytest
import requests

from sed_tools.filter_metadata import read_filter_header, released_mesa_reads

SCRIPT = Path(__file__).resolve().parents[1] / "migrate_filters.py"
spec = importlib.util.spec_from_file_location("migrate_filters", SCRIPT)
migrate_filters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migrate_filters)

TAGGED_030 = ("# FilterID: GAIA/GAIA3.G\n# DetectorType: photon\n# DetectorTypeSource: svo-curve-match\n"
              "Wavelength,Transmission\n3200.0,2.37366962e-08\n6000.0,0.75\n10500.0,1.1e-05\n")


def test_migration_is_offline_idempotent_and_reports(tmp_path, monkeypatch, capsys):
    root = tmp_path / "filters" / "GAIA" / "GAIA"
    root.mkdir(parents=True)
    (root / "G.dat").write_text(TAGGED_030)
    (root / "Bp.dat").write_text("Wavelength,Transmission\n1,1\n2,1\n")
    (root / "raw.dat").write_text("# DetectorType: photon\n3830 0\n4000 1\n")
    monkeypatch.setattr(requests.Session, "request", lambda *a, **k: pytest.fail("migration is offline"))

    assert migrate_filters.main([str(tmp_path / "filters"), "--dry-run"]) == 0
    assert (root / "G.dat").read_text() == TAGGED_030
    assert "2 curves: would move" in capsys.readouterr().out

    assert migrate_filters.main([str(tmp_path / "filters")]) == 0
    out = capsys.readouterr().out
    assert "2 curves: moved" in out and "1 have no column-name row" in out
    assert read_filter_header(root / "G.dat")["DetectorType"] == "photon"
    assert released_mesa_reads(root / "G.dat")
    assert (root / "raw.dat").read_text() == "3830 0\n4000 1\n"
    assert migrate_filters.main([str(tmp_path / "filters")]) == 0
    assert "0 curves: moved" in capsys.readouterr().out


def test_curves_released_mesa_still_misreads_are_listed(tmp_path, capsys):
    root = tmp_path / "filters" / "CAHA" / "PANIC"
    root.mkdir(parents=True)
    (root / "Y.dat").write_text("# DetectorType: photon\n# Curation: averaged duplicates\nWavelength,Transmission\n1,2\n")
    assert migrate_filters.main([str(tmp_path / "filters"), "--dry-run"]) == 1
    assert str(root / "Y.dat") in capsys.readouterr().out


def test_missing_root_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        migrate_filters.main([str(tmp_path / "nowhere")])
