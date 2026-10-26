"""Photon- vs energy-counting filter curves (Bessell & Murphy 2012, eq. A9)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sed_tools._flux import (FilterCurve, band_average_flux_lambda,
                             band_average_flux_lambda_from_arrays,
                             band_average_flux_nu, load_filter_curve)
from sed_tools.filter_metadata import (DETECTOR_SOURCE_PRECEDENCE, ENERGY, PHOTON,
                                       SOURCE_ASSUMED, SOURCE_INFERRED,
                                       SOURCE_ORIGINAL, SOURCE_SVO,
                                       SOURCE_SVO_MATCH, SVOFilterRecord,
                                       detector_source_may_replace,
                                       normalize_detector_type,
                                       read_filter_header, tag_filter_file,
                                       tag_filter_tree, write_filter_file)

C_ANG = 2.99792458e18


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


def _boxcar(lo=4000.0, hi=8000.0, n=4001):
    wl = np.linspace(lo, hi, n)
    return wl, np.ones_like(wl)


def _record(filter_id, detector_type, wl, tr, unit="Angstrom"):
    return SVOFilterRecord(filter_id, detector_type, np.asarray(wl), np.asarray(tr), unit)


def _write_plain(path, wl, tr):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Wavelength,Transmission\n" + "".join(f"{float(w)!r},{float(t)!r}\n" for w, t in zip(wl, tr)))


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [(0, ENERGY), (1, PHOTON), ("0", ENERGY), ("1", PHOTON), (1.0, PHOTON),
     ("photon", PHOTON), ("Energy", ENERGY), ("", None), (None, None), (2, None)],
)
def test_normalize_detector_type(value, expected):
    assert normalize_detector_type(value) == expected


def test_flat_spectrum_is_independent_of_detector_type():
    wl, tr = _boxcar()
    flux = np.full_like(wl, 3.0e-9)
    for dtype in (ENERGY, PHOTON):
        assert band_average_flux_lambda_from_arrays(wl, flux, tr, dtype) == pytest.approx(3.0e-9)


def test_power_law_matches_analytic_band_averages():
    lo, hi = 4000.0, 8000.0
    wl, tr = _boxcar(lo, hi, 200001)
    flux = wl**-2.0

    energy = band_average_flux_lambda_from_arrays(wl, flux, tr, ENERGY)
    photon = band_average_flux_lambda_from_arrays(wl, flux, tr, PHOTON)

    # <F> = ∫λ^-2 dλ / ∫dλ  vs  ∫λ^-1 dλ / ∫λ dλ
    assert energy == pytest.approx((1 / lo - 1 / hi) / (hi - lo), rel=1e-8)
    assert photon == pytest.approx(math.log(hi / lo) / ((hi**2 - lo**2) / 2), rel=1e-8)


def test_photon_f_nu_uses_lambda_weighting():
    lo, hi = 4000.0, 8000.0
    wl, tr = _boxcar(lo, hi, 200001)
    flux = wl**-2.0
    curve = FilterCurve("box", wl, tr, {}, detector_type=PHOTON)

    # <F_nu> = ∫F_λ λ dλ / ∫(c/λ) dλ for a photon counter
    expected = math.log(hi / lo) / (C_ANG * math.log(hi / lo))
    assert band_average_flux_nu(wl, flux, curve) == pytest.approx(expected, rel=1e-8)


def test_flux_curve_detector_type_changes_red_spectrum_photometry():
    wl, tr = _boxcar()
    flux = wl**-4.0  # steep blue spectrum
    energy = band_average_flux_lambda(wl, flux, FilterCurve("e", wl, tr, {}))
    photon = band_average_flux_lambda(wl, flux, FilterCurve("p", wl, tr, {}, detector_type=PHOTON))
    # Photon weighting up-weights the red (fainter) end
    assert photon < energy


# ---------------------------------------------------------------------------
# File header / loader
# ---------------------------------------------------------------------------
def test_loader_reads_detector_type_and_unit(tmp_path):
    path = tmp_path / "G.dat"
    write_filter_file(path, [5000.0, 6000.0], [0.5, 0.6], detector_type=PHOTON,
                      source=SOURCE_SVO, filter_id="GAIA/GAIA3.G", wavelength_unit="Angstrom")

    curve = load_filter_curve(str(path))

    assert curve.detector_type == PHOTON
    # Without the WavelengthUnit header these values would be guessed as nm.
    np.testing.assert_allclose(curve.wavelength, [5000.0, 6000.0])
    np.testing.assert_allclose(curve.weight, [0.5 * 5000.0, 0.6 * 6000.0])


def test_loader_defaults_untagged_files_to_energy(tmp_path):
    path = tmp_path / "V.dat"
    _write_plain(path, [5000.0, 6000.0], [0.5, 0.6])

    curve = load_filter_curve(str(path))

    assert curve.detector_type == ENERGY
    assert curve.metadata["detectortypesource"] == SOURCE_ASSUMED


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


def test_tag_matches_curve_and_distinguishes_versions(tmp_path):
    """GAIA2.G (energy) and GAIA3.G (photon) both land at GAIA/GAIA/G.dat."""
    wl = np.linspace(3300.0, 10500.0, 50)
    tr_dr2 = np.exp(-((wl - 6400) / 1500) ** 2)
    tr_dr3 = np.exp(-((wl - 6200) / 1600) ** 2)
    path = tmp_path / "GAIA" / "GAIA" / "G.dat"
    _write_plain(path, wl, tr_dr3)
    client = FakeSVO(
        {
            "GAIA/GAIA2.G": _record("GAIA/GAIA2.G", ENERGY, wl, tr_dr2),
            "GAIA/GAIA3.G": _record("GAIA/GAIA3.G", PHOTON, wl, tr_dr3),
        },
        facilities={"GAIA": ["GAIA/GAIA2.G", "GAIA/GAIA3.G", "GAIA/GAIA3.Gbp"]},
    )

    result = tag_filter_file(path, client=client)

    assert (result.detector_type, result.source, result.filter_id) == (PHOTON, SOURCE_SVO_MATCH, "GAIA/GAIA3.G")
    # Curve body is untouched by the header rewrite
    curve = load_filter_curve(str(path))
    np.testing.assert_allclose(curve.transmission, tr_dr3)


def test_tag_matches_svo_id_embedded_in_filename(tmp_path):
    wl, tr = [4000.0, 5000.0, 6000.0], [0.1, 0.9, 0.2]
    path = tmp_path / "Argus" / "raw" / "SLOAN_SDSS.g.dat"
    _write_plain(path, wl, tr)
    client = FakeSVO({"SLOAN/SDSS.g": _record("SLOAN/SDSS.g", PHOTON, wl, tr)})

    result = tag_filter_tree(tmp_path, client=client)[0]

    assert (result.detector_type, result.filter_id) == (PHOTON, "SLOAN/SDSS.g")


def test_unidentifiable_curve_is_written_as_assumed_energy(tmp_path):
    path = tmp_path / "Custom" / "Mine" / "beta.dat"
    _write_plain(path, [5000.0, 6000.0], [0.5, 0.6])

    result = tag_filter_file(path, client=FakeSVO({}))

    assert (result.detector_type, result.source) == (ENERGY, SOURCE_ASSUMED)
    header = read_filter_header(path)
    assert header == {"DetectorType": ENERGY, "DetectorTypeSource": SOURCE_ASSUMED}


def test_curve_mismatch_is_not_accepted(tmp_path):
    path = tmp_path / "GAIA" / "GAIA" / "G.dat"
    _write_plain(path, [5000.0, 6000.0], [0.5, 0.6])
    client = FakeSVO({"GAIA/GAIA.G": _record("GAIA/GAIA.G", PHOTON, [5000.0, 6000.0], [0.5, 0.7])})

    result = tag_filter_file(path, client=client)

    assert result.source == SOURCE_ASSUMED


def test_tag_is_idempotent_and_skips_network_when_complete(tmp_path):
    path = tmp_path / "G.dat"
    write_filter_file(path, [1.0], [1.0], detector_type=PHOTON, source=SOURCE_SVO,
                      filter_id="GAIA/GAIA3.G", wavelength_unit="Angstrom",
                      extra={"Components": "filter+qe+instrument", "ComponentsSource": "svo"})
    client = FakeSVO({})

    result = tag_filter_file(path, client=client)

    assert not result.changed
    assert client.calls == []


def test_tag_backfills_published_components(tmp_path):
    wl, tr = [5000.0, 6000.0], [0.5, 0.6]
    path = tmp_path / "G.dat"
    write_filter_file(path, wl, tr, detector_type=PHOTON, source=SOURCE_SVO,
                      filter_id="GAIA/GAIA3.G", wavelength_unit="Angstrom")
    record = _record("GAIA/GAIA3.G", PHOTON, wl, tr)
    record.components = "Filter + CCD + Instrument"

    result = tag_filter_file(path, client=FakeSVO({"GAIA/GAIA3.G": record}))

    assert result.changed
    header = read_filter_header(path)
    assert header["Components"] == "filter+qe+instrument"
    assert header["ComponentsSource"] == "svo"


def test_failed_lookup_does_not_downgrade_authoritative_tag(tmp_path):
    path = tmp_path / "G.dat"
    write_filter_file(path, [1.0], [1.0], detector_type=PHOTON, source=SOURCE_SVO, filter_id="GAIA/GAIA3.G")

    result = tag_filter_file(path, client=FakeSVO({}, fail=True), force=True)

    assert (result.detector_type, result.changed) == (PHOTON, False)
    assert read_filter_header(path)["DetectorType"] == PHOTON


def test_failed_lookup_keeps_original_source_assumption(tmp_path):
    path = tmp_path / "Gemini" / "GSAOI" / "J.dat"
    path.parent.mkdir(parents=True)
    write_filter_file(path, [10000., 12000.], [0.5, 0.8], detector_type=PHOTON,
                      source=SOURCE_ASSUMED, wavelength_unit="Angstrom",
                      extra={"Components": "filter", "ComponentsSource": "original"})

    client = FakeSVO({}, fail=True)
    result = tag_filter_file(path, client=client)

    assert (result.detector_type, result.source, result.changed) == (PHOTON, SOURCE_ASSUMED, False)
    assert read_filter_header(path)["DetectorType"] == PHOTON
    assert client.calls == []


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


def test_precedence_order():
    assert DETECTOR_SOURCE_PRECEDENCE == (SOURCE_ORIGINAL, SOURCE_INFERRED, SOURCE_SVO,
                                          SOURCE_SVO_MATCH, SOURCE_ASSUMED)


@pytest.mark.parametrize(
    "existing,new,allowed",
    [(SOURCE_ORIGINAL, SOURCE_SVO, False), (SOURCE_ORIGINAL, SOURCE_SVO_MATCH, False),
     (SOURCE_INFERRED, SOURCE_SVO, False), (SOURCE_INFERRED, SOURCE_SVO_MATCH, False),
     (SOURCE_SVO, SOURCE_SVO_MATCH, False), (SOURCE_SVO, SOURCE_SVO, True),
     (SOURCE_SVO_MATCH, SOURCE_SVO_MATCH, True), (SOURCE_SVO_MATCH, SOURCE_SVO, True),
     (SOURCE_ASSUMED, SOURCE_SVO_MATCH, True), (SOURCE_ASSUMED, SOURCE_SVO, True),
     (SOURCE_INFERRED, SOURCE_ORIGINAL, True), ("", SOURCE_SVO_MATCH, True), (None, SOURCE_SVO, True),
     ("user", SOURCE_SVO, False), ("composed", SOURCE_SVO_MATCH, False), ("component", SOURCE_SVO, False)],
)
def test_detector_source_may_replace(existing, new, allowed):
    assert detector_source_may_replace(existing, new) is allowed


def test_force_does_not_replace_inferred_with_svo_curve_match(tmp_path):
    """NACO: NJM's curve is SVO's array point for point, but SVO labels it energy."""
    wl, tr = [15000.0, 16000.0, 17000.0, 18000.0], [0.0, 0.9, 0.95, 0.0]
    path = tmp_path / "Paranal" / "NACO" / "H_CONICA.dat"
    before = _evidence_file(path, wl, tr, PHOTON, SOURCE_INFERRED)
    client = FakeSVO({"Paranal/NACO.H_CONICA": _record("Paranal/NACO.H_CONICA", ENERGY, wl, tr)})

    result = tag_filter_tree(tmp_path, client=client, force=True)[0]

    assert (result.detector_type, result.source, result.changed) == (PHOTON, SOURCE_INFERRED, False)
    assert result.kept
    assert (result.resolved_type, result.resolved_source) == (ENERGY, SOURCE_SVO_MATCH)
    assert "inferred outranks svo-curve-match" in result.note
    assert "Paranal/NACO.H_CONICA" in result.note
    assert path.read_bytes() == before  # no FilterID or other field taken from the rejected match


@pytest.mark.parametrize("svo_type", [ENERGY, PHOTON])
def test_force_does_not_replace_original(tmp_path, svo_type):
    """Bessell & Murphy 2012 B is photon; SVO's Generic/Bessell.B says energy. Same type: source kept."""
    wl, tr = [4000.0, 4500.0, 5000.0], [0.2, 1.0, 0.3]
    path = tmp_path / "Generic" / "Bessell" / "B.dat"
    before = _evidence_file(path, wl, tr, PHOTON, SOURCE_ORIGINAL, filter_id="Generic/Bessell.B")
    client = FakeSVO({"Generic/Bessell.B": _record("Generic/Bessell.B", svo_type, wl, [0.1, 1.0, 0.4])})

    result = tag_filter_file(path, client=client, force=True)

    assert (result.detector_type, result.source, result.changed) == (PHOTON, SOURCE_ORIGINAL, False)
    assert result.kept
    assert (result.resolved_type, result.resolved_source) == (svo_type, SOURCE_SVO)
    assert path.read_bytes() == before


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


def test_force_upgrades_assumed_by_curve_match(tmp_path):
    wl, tr = [15000.0, 16000.0, 17000.0, 18000.0], [0.0, 0.9, 0.95, 0.0]
    path = tmp_path / "Paranal" / "NACO" / "H_CONICA.dat"
    _evidence_file(path, wl, tr, PHOTON, SOURCE_ASSUMED)
    client = FakeSVO({"Paranal/NACO.H_CONICA": _record("Paranal/NACO.H_CONICA", ENERGY, wl, tr)})

    # Without force an original-source assumption is not retried.
    assert not tag_filter_tree(tmp_path, client=client)[0].changed
    assert client.calls == []

    result = tag_filter_tree(tmp_path, client=client, force=True)[0]

    assert (result.detector_type, result.source, result.changed, result.kept) == (
        ENERGY, SOURCE_SVO_MATCH, True, False)
    header = read_filter_header(path)
    assert (header["DetectorType"], header["DetectorTypeSource"], header["FilterID"]) == (
        ENERGY, SOURCE_SVO_MATCH, "Paranal/NACO.H_CONICA")


def test_force_upgrades_assumed_by_filter_id(tmp_path):
    wl, tr = [5000.0, 6000.0], [0.5, 0.6]
    path = tmp_path / "GOES" / "ABI1" / "C01.dat"
    _evidence_file(path, wl, tr, ENERGY, SOURCE_ASSUMED, filter_id="GOES/GOES16.C01")
    client = FakeSVO({"GOES/GOES16.C01": _record("GOES/GOES16.C01", PHOTON, wl, tr)})

    result = tag_filter_file(path, client=client, force=True)

    assert (result.detector_type, result.source, result.changed) == (PHOTON, SOURCE_SVO, True)
    assert read_filter_header(path)["DetectorTypeSource"] == SOURCE_SVO


def test_force_keeps_assumed_type_when_nothing_matches(tmp_path):
    wl, tr = [15000.0, 16000.0, 17000.0], [0.1, 0.9, 0.1]
    path = tmp_path / "Paranal" / "NACO" / "H_CONICA.dat"
    before = _evidence_file(path, wl, tr, PHOTON, SOURCE_ASSUMED)

    result = tag_filter_tree(tmp_path, client=FakeSVO({}), force=True)[0]

    assert (result.detector_type, result.source, result.changed, result.kept) == (
        PHOTON, SOURCE_ASSUMED, False, False)
    assert path.read_bytes() == before


def test_force_replaces_svo_with_current_svo(tmp_path):
    """Equal precedence: a forced run still refreshes an svo tag from SVO."""
    wl, tr = [5000.0, 6000.0], [0.5, 0.6]
    path = tmp_path / "G.dat"
    write_filter_file(path, wl, tr, detector_type=ENERGY, source=SOURCE_SVO, filter_id="GAIA/GAIA3.G",
                      wavelength_unit="Angstrom",
                      extra={"Components": "filter+qe+instrument", "ComponentsSource": "svo"})

    result = tag_filter_file(path, client=FakeSVO({"GAIA/GAIA3.G": _record("GAIA/GAIA3.G", PHOTON, wl, tr)}),
                             force=True)

    assert (result.detector_type, result.source, result.changed, result.kept) == (
        PHOTON, SOURCE_SVO, True, False)


@pytest.mark.parametrize("source", [SOURCE_ORIGINAL, SOURCE_INFERRED])
def test_default_run_does_not_query_svo_for_evidence(tmp_path, source):
    wl, tr = [15000.0, 16000.0, 17000.0], [0.1, 0.9, 0.1]
    path = tmp_path / "Paranal" / "NACO" / "H_CONICA.dat"
    before = _evidence_file(path, wl, tr, PHOTON, source)
    client = FakeSVO({"Paranal/NACO.H_CONICA": _record("Paranal/NACO.H_CONICA", ENERGY, wl, tr)})

    result = tag_filter_tree(tmp_path, client=client)[0]

    assert (result.detector_type, result.source, result.changed, result.kept) == (PHOTON, source, False, False)
    assert client.calls == []
    assert path.read_bytes() == before


def test_incomplete_original_header_is_not_overwritten(tmp_path):
    """A FilterID without Components is re-resolved by default; the original type still stands."""
    wl, tr = [10000.0, 12000.0, 14000.0], [0.1, 0.8, 0.1]
    path = tmp_path / "2MASS" / "2MASS" / "J.dat"
    path.parent.mkdir(parents=True)
    write_filter_file(path, wl, tr, detector_type=ENERGY, source=SOURCE_ORIGINAL, filter_id="2MASS/2MASS.J",
                      wavelength_unit="Angstrom")
    before = path.read_bytes()
    record = _record("2MASS/2MASS.J", PHOTON, wl, tr)
    record.components = "Filter + Instrument + Atmosphere"

    result = tag_filter_file(path, client=FakeSVO({"2MASS/2MASS.J": record}))

    assert (result.detector_type, result.source, result.kept, result.resolved_type) == (
        ENERGY, SOURCE_ORIGINAL, True, PHOTON)
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


def test_dry_run_leaves_file_untouched(tmp_path):
    path = tmp_path / "X" / "Y" / "beta.dat"
    _write_plain(path, [5000.0], [1.0])
    before = path.read_text()

    result = tag_filter_file(path, client=FakeSVO({}), dry_run=True)

    assert result.changed
    assert path.read_text() == before


# ---------------------------------------------------------------------------
# Download path
# ---------------------------------------------------------------------------
def test_svo_download_writes_detector_header(tmp_path):
    from sed_tools.svo_filter_grabber import FilterRow, SVOFilterBrowser

    browser = SVOFilterBrowser(base_dir=str(tmp_path))
    browser.metadata_client = FakeSVO(
        {"GAIA/GAIA3.G": _record("GAIA/GAIA3.G", PHOTON, [5000.0, 6000.0], [0.5, 0.6])}
    )

    browser.download_filters([FilterRow("GAIA/GAIA3.G", "G", "GAIA", "GAIA", "")])

    path = tmp_path / "GAIA" / "GAIA" / "G.dat"
    assert read_filter_header(path) == {
        "FilterID": "GAIA/GAIA3.G",
        "DetectorType": PHOTON,
        "DetectorTypeSource": SOURCE_SVO,
        "WavelengthUnit": "Angstrom",
    }
    assert (tmp_path / "GAIA" / "GAIA" / "GAIA").read_text() == "G.dat\n"
