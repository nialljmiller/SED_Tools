"""Photon- vs energy-counting filter curves (Bessell & Murphy 2012, eq. A9)."""

from __future__ import annotations


import numpy as np

from sed_tools._flux import (load_filter_curve)
from sed_tools.filter_metadata import (ENERGY, PHOTON,
                                       SOURCE_ASSUMED, SOURCE_INFERRED,
                                       SOURCE_SVO,
                                       SVOFilterRecord,
                                       normalize_detector_type,
                                       read_filter_header, tag_filter_file,
                                       tag_filter_tree, write_filter_file)


class FakeSVO:
    """Offline stand-in for :class:`SVOMetadataClient`."""

    def __init__(self, records, facilities=None, fail=False):
        self.records = records
        self.facilities = facilities or {}
        self.fail = fail
        self.calls = []

    def get_filter(self, filter_id):
        self.calls.append(filter_id)
        if self.fail:
            raise RuntimeError("network down")
        return self.records.get(filter_id)

    def facility_filter_ids(self, facility):
        return self.facilities.get(facility, [])


def _record(filter_id, detector_type, wl, tr, unit="Angstrom"):
    return SVOFilterRecord(filter_id, detector_type, np.asarray(wl), np.asarray(tr), unit)


def _write_plain(path, wl, tr):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Wavelength,Transmission\n" + "".join(f"{float(w)!r},{float(t)!r}\n" for w, t in zip(wl, tr)))


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def test_normalize_detector_type():
    assert normalize_detector_type(2) is None


# ---------------------------------------------------------------------------
# Resolution / tagging
# ---------------------------------------------------------------------------
def test_tag_resolves_by_filter_id(tmp_path):
    wl, tr = [5000.0, 6000.0], [0.5, 0.6]
    path = tmp_path / "GAIA" / "GAIA" / "G.dat"
    path.parent.mkdir(parents=True)
    write_filter_file(path, wl, tr, detector_type=ENERGY, source=SOURCE_ASSUMED, filter_id="GAIA/GAIA3.G")
    client = FakeSVO({"GAIA/GAIA3.G": _record("GAIA/GAIA3.G", PHOTON, wl, tr)})

    result = tag_filter_file(path, client=client)

    assert (result.detector_type, result.source, result.changed) == (PHOTON, SOURCE_SVO, True)
    assert read_filter_header(path)["DetectorType"] == PHOTON
    assert load_filter_curve(str(path)).detector_type == PHOTON


def test_tag_matches_svo_id_embedded_in_filename(tmp_path):
    wl, tr = [4000.0, 5000.0, 6000.0], [0.1, 0.9, 0.2]
    path = tmp_path / "Argus" / "raw" / "SLOAN_SDSS.g.dat"
    _write_plain(path, wl, tr)
    client = FakeSVO({"SLOAN/SDSS.g": _record("SLOAN/SDSS.g", PHOTON, wl, tr)})

    result = tag_filter_tree(tmp_path, client=client)[0]

    assert (result.detector_type, result.filter_id) == (PHOTON, "SLOAN/SDSS.g")


def test_curve_mismatch_is_not_accepted(tmp_path):
    path = tmp_path / "GAIA" / "GAIA" / "G.dat"
    _write_plain(path, [5000.0, 6000.0], [0.5, 0.6])
    client = FakeSVO({"GAIA/GAIA.G": _record("GAIA/GAIA.G", PHOTON, [5000.0, 6000.0], [0.5, 0.7])})

    result = tag_filter_file(path, client=client)

    assert result.source == SOURCE_ASSUMED


# ---------------------------------------------------------------------------
# Source precedence: evidence outranks SVO, with or without --force
# ---------------------------------------------------------------------------
NACO_URL = "https://www.eso.org/sci/facilities/paranal/instruments/naco/inst/filters/H_CONICA.dat"


def _evidence_file(path, wl, tr, detector_type, source, filter_id=None, components_source="original"):
    """A curve tagged the way the NJM mirror tags evidence-based bands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_filter_file(path, wl, tr, detector_type=detector_type, source=source, filter_id=filter_id,
                      wavelength_unit="Angstrom",
                      extra={"Components": "filter", "ComponentsSource": components_source,
                             "Sources": NACO_URL})
    return path.read_bytes()


def test_force_does_not_replace_user_override(tmp_path):
    wl, tr = [4000.0, 5000.0, 6000.0], [0.1, 0.9, 0.2]
    path = tmp_path / "SLOAN" / "SDSS+detector_energy" / "g.dat"
    path.parent.mkdir(parents=True)
    write_filter_file(path, wl, tr, detector_type=ENERGY, source="user", wavelength_unit="Angstrom",
                      extra={"DetectorTypeOriginal": PHOTON})
    before = path.read_bytes()
    client = FakeSVO({"SLOAN/SDSS+detector_energy.g": _record("SLOAN/SDSS+detector_energy.g", PHOTON, wl, tr)})

    result = tag_filter_tree(tmp_path, client=client, force=True)[0]

    assert (result.detector_type, result.source, result.kept) == (ENERGY, "user", True)
    assert path.read_bytes() == before


def test_force_keeps_assumed_type_when_nothing_matches(tmp_path):
    wl, tr = [15000.0, 16000.0, 17000.0], [0.1, 0.9, 0.1]
    path = tmp_path / "Paranal" / "NACO" / "H_CONICA.dat"
    before = _evidence_file(path, wl, tr, PHOTON, SOURCE_ASSUMED)

    result = tag_filter_tree(tmp_path, client=FakeSVO({}), force=True)[0]

    assert (result.detector_type, result.source, result.changed, result.kept) == (
        PHOTON, SOURCE_ASSUMED, False, False)
    assert path.read_bytes() == before


def test_cli_reports_kept_types(tmp_path, monkeypatch, capsys):
    from sed_tools import cli

    wl, tr = [15000.0, 16000.0, 17000.0, 18000.0], [0.0, 0.9, 0.95, 0.0]
    path = tmp_path / "Paranal" / "NACO" / "H_CONICA.dat"
    before = _evidence_file(path, wl, tr, PHOTON, SOURCE_INFERRED)
    client = FakeSVO({"Paranal/NACO.H_CONICA": _record("Paranal/NACO.H_CONICA", ENERGY, wl, tr)})
    monkeypatch.setattr("sed_tools.filter_metadata.SVOMetadataClient", lambda *a, **k: client)
    monkeypatch.setattr("sys.argv", ["sed-tools", "filters-detector", str(tmp_path), "--force"])

    cli.main()

    out = capsys.readouterr().out
    assert "1 files, 0 changed" in out
    assert "1 kept their DetectorType" in out and "1 of them resolve to the other type" in out
    assert "kept photon [inferred]: inferred outranks svo-curve-match; svo-curve-match gives energy" in out
    assert path.read_bytes() == before
