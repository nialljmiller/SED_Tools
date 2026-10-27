"""Follow-ups to decisions 18 and 19.

1. BOSZ (MAST) spectra hold the Eddington flux H_λ, which no flux unit of
   spectra_cleaner describes, so cleaning them stops and asks for the unit.
2. AB and bolometric magnitudes return what the formula gives: inf for zero
   flux and NaN for negative flux, as the Vega ones already did.
3. Flux units declared in spectrum headers: W/m²/nm and W/m²/µm had the
   factor for W/m²/Å, and some header text was read as the wrong unit.

No test uses the network: the autouse fixture fails any real HTTP request.
"""

import gzip
import math
import os
import warnings

import numpy as np
import pytest
import requests

from sed_tools import mast_spectra_grabber as mast
from sed_tools._flux import AB_ZERO_FLUX
from sed_tools.flux_cube_tool import FilterCurve, _band_magnitude, bolometric_magnitude
from sed_tools.models import EvaluatedSED
from sed_tools.spectra_cleaner import (FLUX_UNITS, check_flux_unit, clean_model_dir,
                                       detect_units_from_header)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("test tried to use the network")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", refuse)


class FakeResponse:
    def __init__(self, body):
        self.status_code = 200
        self.content = body if isinstance(body, bytes) else body.encode()
        self.text = self.content.decode("utf-8", "ignore")

    def raise_for_status(self):
        pass


class FakeSession:
    """Answers from a {url: response} table; any other URL is a test error."""

    def __init__(self, routes):
        self.routes = routes

    def get(self, url, **kwargs):
        return self.routes[url]


# ---------------------------------------------------------------------
# 1. BOSZ spectra: the flux is H_λ, so cleaning stops and asks
# ---------------------------------------------------------------------

# Twelve rows (1, 824-826, 2001, 3001, 3501, 4001, 5001, 6001, 7001 and 7500 of
# 7500) of a real BOSZ 2024 R=500 spectrum and of its wavelength grid, as MAST
# serves them, retrieved 2026-10-27:
#   https://archive.stsci.edu/hlsps/bosz/bosz2024/r500/m+0.00/<BOSZ_NAME>
#   https://archive.stsci.edu/hlsps/bosz/bosz2024/wavelength_grids/bosz2024_wave_r500.txt
# The spectrum's columns are the Eddington flux H_λ in erg/s/cm²/Å and the
# continuum; the surface flux is F_λ = 4πH_λ (Mészáros et al. 2024, eq. 2).
# Where the flux is below 1e-6 the file holds 0 and 1.
BOSZ_NAME = "bosz2024_mp_t5000_g+5.0_m+0.00_a+0.00_c+0.00_v0_r500_resam.txt.gz"
BOSZ_WAVE_LINES = [
    "5.0000000e+02", "1.0160182e+03", "1.0168939e+03", "1.0177704e+03",
    "2.8008166e+03", "6.6289080e+03", "1.0198133e+04", "1.5689147e+04",
    "3.7132711e+04", "8.7884843e+04", "2.0800382e+05", "3.1972443e+05",
]
BOSZ_SPECTRUM_LINES = [
    "0.0000e+00 1.0000e+00", "0.0000e+00 1.0000e+00", "7.9405e-07 1.0064e-06", "8.1437e-07 1.0373e-06",
    "4.0757e+03 5.8027e+04", "3.1197e+05 3.2022e+05", "1.6892e+05 1.7022e+05", "7.7956e+04 8.1788e+04",
    "3.6002e+03 3.6386e+03", "1.2803e+02 1.2810e+02", "4.2380e+00 4.2434e+00", "7.6432e-01 7.6514e-01",
]


def _download_bosz_fixture(tmp_path):
    """The fixture spectrum as MASTSpectraGrabber writes it; returns the grid directory."""
    url = f"{mast.DATA_BASE}/r500/m+0.00/{BOSZ_NAME}"
    script = f"{mast.SCRIPTS_DIR}/hlsp_bosz_bosz2024_sim_r500_m%2B0.00_v1_bulkdl.sh"
    wave_url = f"{mast.DATA_BASE}/wavelength_grids/bosz2024_wave_r500.txt"
    session = FakeSession({
        script: FakeResponse(f'curl -f --create-dirs --output "r500/m+0.00/{BOSZ_NAME}" "{url}"\n'),
        url: FakeResponse(gzip.compress(("\n".join(BOSZ_SPECTRUM_LINES) + "\n").encode())),
        wave_url: FakeResponse("\n".join(BOSZ_WAVE_LINES) + "\n"),
    })
    grabber = mast.MASTSpectraGrabber(base_dir=str(tmp_path), max_workers=1, session=session)
    assert grabber.download_model_spectra("BOSZ-2024-r500", {"reskey": "r500", "metals": ["m+0.00"]}) == 1
    return tmp_path / "BOSZ-2024-r500"


def _snapshot(directory):
    return {name: (directory / name).read_bytes() for name in sorted(os.listdir(directory))}


def test_bosz_spectrum_is_written_as_served_and_declares_no_flux_unit(tmp_path):
    model_dir = _download_bosz_fixture(tmp_path)
    text = (model_dir / mast._sanitize_filename(BOSZ_NAME.replace(".txt.gz", ".txt"))).read_text()
    header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))

    data = np.loadtxt(text.splitlines(), comments="#")
    np.testing.assert_array_equal(data[:, 0], [float(w) for w in BOSZ_WAVE_LINES])
    np.testing.assert_array_equal(data[:, 1:], [[float(v) for v in line.split()] for line in BOSZ_SPECTRUM_LINES])
    assert detect_units_from_header(header)["flux_unit"] is None


def test_cleaning_a_bosz_grid_stops_and_asks_for_the_flux_unit(tmp_path):
    model_dir = _download_bosz_fixture(tmp_path)
    before = _snapshot(model_dir)
    txt_files = sorted(str(p) for p in model_dir.glob("*.txt"))

    with pytest.raises(ValueError, match=r"is not declared; best guess"):
        check_flux_unit("BOSZ-2024-r500", txt_files)
    with pytest.raises(ValueError, match=r"Flux unit of .*BOSZ-2024-r500 is not declared; best guess"):
        clean_model_dir(str(model_dir), sample_fraction=1.0)
    assert _snapshot(model_dir) == before


# ---------------------------------------------------------------------
# 2. AB, Vega and bolometric magnitudes: the computed value, no error
# ---------------------------------------------------------------------
# Zero and negative band flux in AB and Vega are tested in test_fix_decisions.py.

VEGA_ZERO_POINT = 2.0e-10
WL = np.linspace(2000.0, 8000.0, 300)


class _OneFilter:
    """Stands in for the SEDModel behind an EvaluatedSED: every filter resolves to one curve."""

    def __init__(self, curve):
        self.curve = curve

    def _resolve_filters(self, filters):
        return {"Band": self.curve}


def _band():
    return FilterCurve("Band", np.array([3000.0, 4000.0, 5000.0, 6000.0]), np.array([0.0, 1.0, 1.0, 0.0]),
                       {"zp_vega": VEGA_ZERO_POINT})


def _sloped_flux():
    return 3e-12 * (1.0 + WL / 7000.0)


@pytest.mark.parametrize("system, zero_point", [("AB", AB_ZERO_FLUX), ("Vega", VEGA_ZERO_POINT)])
def test_magnitude_of_positive_flux_is_the_formula_exactly(system, zero_point):
    sed = EvaluatedSED(WL, _sloped_flux(), {"teff": 5000.0, "logg": 4.0, "meta": 0.0}, _OneFilter(_band()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = sed.photometry("Band", system=system)["Band"]
        _, flux_density, magnitude = _band_magnitude(WL, _sloped_flux(), _band(), system, None)

    assert flux_density == result.flux_density > 0
    for value in (result.magnitude, magnitude):
        assert type(value) is float
        assert value == -2.5 * math.log10(flux_density / zero_point)


def test_bolometric_magnitude_of_positive_flux_is_the_formula_exactly():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fbol, mbol = bolometric_magnitude(WL, _sloped_flux(), reference_flux=2.5e-8, reference_magnitude=-26.8)

    assert fbol > 0
    assert type(mbol) is float
    assert mbol == -26.8 - 2.5 * math.log10(fbol / 2.5e-8)


@pytest.mark.parametrize("flux_scale, reference_flux, expected", [
    (0.0, 1.0, math.inf),     # used to be inf by a special case
    (-1.0, 1.0, math.nan),    # used to be inf by a special case
    (1.0, 0.0, -math.inf),    # used to raise ZeroDivisionError
    (1.0, -1.0, math.nan),    # used to raise ValueError from math.log10
    (0.0, 0.0, math.nan),     # used to be inf by a special case
])
def test_bolometric_magnitude_of_non_positive_flux_or_reference_is_computed(flux_scale, reference_flux, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, mbol = bolometric_magnitude(WL, flux_scale * _sloped_flux(), reference_flux=reference_flux)

    if math.isnan(expected):
        assert math.isnan(mbol)
    else:
        assert mbol == expected


# ---------------------------------------------------------------------
# 3. Declared flux units: SI factors and header patterns
# ---------------------------------------------------------------------

LAMBDA_A = np.linspace(3000.0, 10000.0, 41)
FLAM = 1e-5 * (LAMBDA_A / 5000.0) ** -2          # the known F_λ, erg/s/cm²/Å
C_A_PER_S = 2.99792458e18                        # speed of light, Å/s
FNU = FLAM * LAMBDA_A ** 2 / C_A_PER_S           # F_ν = F_λ λ²/c, erg/s/cm²/Hz

# The known spectrum in each declared unit, by hand. 1 erg/s = 1e-7 W and
# 1 cm² = 1e-4 m², so 1 erg/s/cm² = 1e-3 W/m²; an nm holds 10 Å and a µm 1e4 Å.
DECLARED = {
    "erg/s/cm2/A": FLAM,
    "erg/s/cm2/Hz": FNU,
    "Jy": FNU / 1e-23,
    "W/m2/A": FLAM * 1e-3,
    "W/m2/nm": FLAM * 1e-3 * 10.0,
    "W/m2/um": FLAM * 1e-3 * 1e4,
}


def _declared_spectrum(path, flux_header, flux):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{w:.17g} {f:.17g}\n" for w, f in zip(LAMBDA_A, flux))
    path.write_text(f"# wavelength_unit = Angstrom\n# flux_unit = {flux_header}\n{rows}")


@pytest.mark.parametrize("flux_header", list(DECLARED))
def test_a_declared_flux_unit_is_converted_to_erg_s_cm2_a(tmp_path, flux_header):
    grid = tmp_path / "Grid"
    _declared_spectrum(grid / "s0.txt", flux_header, DECLARED[flux_header])

    summary = clean_model_dir(str(grid), sample_fraction=1.0, rebuild_lookup=False)

    assert summary["converted"] == ["s0.txt"]
    data = np.loadtxt(grid / "s0.txt", comments="#")
    np.testing.assert_allclose(data[:, 0], LAMBDA_A, rtol=0, atol=5e-7)
    np.testing.assert_allclose(data[:, 1], FLAM, rtol=1e-8, atol=0)  # written with 9 significant digits


def test_si_flux_units_have_the_factor_of_their_wavelength_unit():
    assert FLUX_UNITS["flam_si"][:2] == ("flam", 1e3)
    assert FLUX_UNITS["flam_si_nm"][:2] == ("flam", 1e2)
    assert FLUX_UNITS["flam_si_um"][:2] == ("flam", 0.1)


@pytest.mark.parametrize("header, expected", [
    # W/m² per nm and per µm used to be read as W/m²/Å
    ("# flux_unit = W/m2/nm", "flam_si_nm"),
    ("# flux_unit = W/m^2/um", "flam_si_um"),
    ("# flux [W/m2/A]", "flam_si"),
    ("# columns = wavelength_A flux_W/m2/A", "flam_si"),
    # a unit next to a quantity name is read as the unit, not the name
    ("# F_nu (Jy)", "fnu_jy"),
    ("# Flambda (W/m2/um)", "flam_si_um"),
    ("# F_lambda (W/m^2/nm)", "flam_si_nm"),
    # words that used to be read as a unit
    ("# BT-Settl low mass models", None),
    ("# a new man", None),
    ("# flux erg/s/cm2/arcsec2", None),
    # spellings read as before
    ("# column 2: FLUX (ERG/CM2/S/A)", "flam"),
    ("# flux_unit = erg/cm2/s/A", "flam"),
    ("# flux (erg/s/cm^2/Angstrom)", "flam"),
    ("# flux (erg/s/cm2/Angstroms)", "flam"),
    ("# erg/s/cm2/AA", "flam"),
    ("# erg/s/cm2/Ang", "flam"),
    ("# erg/s/cm2/Å", "flam"),
    ("# fnu (erg/s/cm2/Hz)", "fnu"),
    ("# units: F_nu", "fnu"),
    ("# flux in Jansky", "fnu_jy"),
    ("# normalized flux", "normalized"),
])
def test_flux_unit_header_patterns(header, expected):
    assert detect_units_from_header(header + "\n")["flux_unit"] == expected


def test_a_grid_declaring_w_m2_nm_in_some_spectra_names_that_unit_as_its_guess(tmp_path):
    # The guess is the unit of the first sampled spectrum; W/m²/nm used to be
    # read with the W/m²/Å factor, which a FLUX_UNITS name had to match.
    grid = tmp_path / "Mixed"
    _declared_spectrum(grid / "a.txt", "W/m2/nm", DECLARED["W/m2/nm"])
    (grid / "b.txt").write_text("".join(f"{w:.17g} {f:.17g}\n" for w, f in zip(LAMBDA_A, FLAM)))

    with pytest.raises(ValueError, match=r"is not declared; best guess: 'flam_si_nm', W/m²/nm"):
        clean_model_dir(str(grid), sample_fraction=1.0)
