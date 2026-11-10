"""Flux units of spectral grids: cleaning stops on an undeclared unit and the CLI asks
for one; declared units are converted to erg/s/cm2/A; BOSZ spectra declare none; header patterns.

No test uses the network: the autouse fixture fails any real HTTP request.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pytest

from sed_tools._constants import planck_flam
from sed_tools.spectra_cleaner import (FLUX_UNITS, clean_model_dir,
                                       detect_units_from_header)

from helpers_grids import no_network  # noqa: F401 (an autouse fixture)


# ---------------------------------------------------------------------
# Undeclared flux units: stop and ask, offering the detector's best guess
# ---------------------------------------------------------------------
def _surface_flux_grid(directory, flux_header=None, teffs=(3500.0, 3600.0, 3700.0)):
    """Spectra of pi*B_lambda, a surface flux in erg/s/cm2/A, with or without a flux-unit header."""
    directory.mkdir(parents=True, exist_ok=True)
    wl = np.linspace(3000.0, 10000.0, 200)
    for i, teff in enumerate(teffs):
        lines = [f"# teff = {teff}\n", "# logg = 4.5\n", "# metallicity = 0.0\n", "# wavelength_unit = Angstrom\n"]
        if flux_header:
            lines.append(f"# flux_unit = {flux_header}\n")
        lines += [f"{w:.6f} {f:.8e}\n" for w, f in zip(wl, planck_flam(wl, teff))]
        (directory / f"s{i}.txt").write_text("".join(lines))
    return directory


def _contents(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.txt"))}


def test_clean_model_dir_stops_on_an_undeclared_flux_unit_naming_the_best_guess(tmp_path):
    from sed_tools.spectra_cleaner import clean_model_dir, undeclared_flux_unit

    grid = _surface_flux_grid(tmp_path / "Cool")
    before = _contents(grid)
    # The range detector reads these surface fluxes (1e5-1e6) as Jansky at Earth.
    assert undeclared_flux_unit(sorted(str(p) for p in grid.glob("*.txt")), 1.0) == "fnu_jy"

    with pytest.raises(ValueError) as err:
        clean_model_dir(str(grid), sample_fraction=1.0)
    message = str(err.value)
    assert f"Flux unit of {grid} is not declared; best guess: 'fnu_jy', Jansky" in message
    assert "Pass the unit as flux_unit='fnu_jy', or as another of flam, fnu, fnu_jy, flam_si, flam_si_nm, flam_si_um, normalized" in message
    assert _contents(grid) == before
    assert not (grid / "lookup_table.csv").exists()


def test_clean_model_dir_uses_the_flux_unit_it_is_given(tmp_path):
    from sed_tools.spectra_cleaner import clean_model_dir
    from sed_tools.spectrum_io import read_text_spectrum

    grid = _surface_flux_grid(tmp_path / "Cool")
    (grid / "s2.txt").rename(grid / "s2.TXT")  # until 0.6.0 left uncleaned, yet bundled and in the cube
    summary = clean_model_dir(str(grid), sample_fraction=1.0, flux_unit="flam")

    assert len(summary["converted"]) == 3 and summary["catalog_units"]["flux"] == "flam"
    wl, flux = read_text_spectrum(str(grid / "s0.txt"))
    np.testing.assert_allclose(flux, planck_flam(wl, 3500.0), rtol=1e-8)
    assert "# original_flux_unit = flam" in (grid / "s0.txt").read_text()


def test_declared_flux_units_clean_as_before(tmp_path):
    # A declared unit is used without asking, and gives the same files as passing it.
    from sed_tools.spectra_cleaner import clean_model_dir

    declared = _surface_flux_grid(tmp_path / "Declared", flux_header="erg/cm2/s/A")
    passed = _surface_flux_grid(tmp_path / "Passed")
    clean_model_dir(str(declared), sample_fraction=1.0)
    clean_model_dir(str(passed), sample_fraction=1.0, flux_unit="flam")
    assert _contents(declared) == _contents(passed)


def test_an_unknown_flux_unit_name_is_refused_before_anything_changes(tmp_path):
    from sed_tools.spectra_cleaner import clean_model_dir

    grid = _surface_flux_grid(tmp_path / "Cool")
    before = _contents(grid)
    with pytest.raises(ValueError, match=r"flux_unit='Jy' is not one of flam, fnu, fnu_jy, flam_si, flam_si_nm, flam_si_um, normalized"):
        clean_model_dir(str(grid), flux_unit="Jy")
    assert _contents(grid) == before


def test_a_spectrum_cleaned_on_its_own_stops_on_an_undeclared_flux_unit(tmp_path):
    from sed_tools.spectra_cleaner import clean_spectrum_file

    grid = _surface_flux_grid(tmp_path / "Cool")
    before = (grid / "s0.txt").read_bytes()
    with pytest.raises(ValueError, match=r"Flux unit of this spectrum is not declared; best guess: 'fnu_jy'"):
        clean_spectrum_file(str(grid / "s0.txt"))
    assert (grid / "s0.txt").read_bytes() == before


def test_files_cleaned_one_by_one_without_a_declared_flux_unit_are_listed(tmp_path, monkeypatch):
    # The sample held only a standardized file, so there are no grid-wide units.
    import sed_tools.spectra_cleaner as sc

    grid = _surface_flux_grid(tmp_path / "Cool")
    done = grid / "done.txt"
    done.write_text("# units_standardized = True\n4000 1.0\n5000 2.0\n")
    monkeypatch.setattr(sc.random, "sample", lambda files, n: [str(done)])

    with pytest.raises(RuntimeError, match=r"(?s)3 of 4 spectra .*Flux unit of this spectrum is not declared; "
                                           r"best guess: 'fnu_jy'.* \(3\): s0.txt, s1.txt, s2.txt"):
        sc.clean_model_dir(str(grid), sample_fraction=0.1)
    summary = sc.clean_model_dir(str(grid), sample_fraction=0.1, flux_unit="flam")
    assert summary["converted"] == ["s0.txt", "s1.txt", "s2.txt"]


class _Stdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def _answers(monkeypatch, *answers):
    """Feed input() these answers on a terminal; return the prompts it was given."""
    prompts, queue = [], list(answers)
    monkeypatch.setattr("sys.stdin", _Stdin(True))
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or queue.pop(0))
    return prompts


@pytest.mark.parametrize("answers, unit", [(["", "fnu_jy"], "fnu_jy"), (["y", "flam"], "flam"), (["flam"], "flam")])
def test_cli_offers_the_best_guess_and_takes_a_typed_unit(tmp_path, monkeypatch, capsys, answers, unit):
    from sed_tools import cli

    grid = _surface_flux_grid(tmp_path / "Cool")
    prompts = _answers(monkeypatch, *answers)
    assert cli._ask_flux_unit("Cool", sorted(str(p) for p in grid.glob("*.txt"))) == unit
    # Enter or 'y' does not take the guess: the unit must be typed.
    assert prompts == ["Flux unit of Cool (type its name, e.g. fnu_jy), or n to leave it uncleaned: "] * len(answers)
    out = capsys.readouterr().out
    assert "flam        erg/s/cm²/Å (F_λ)" in out and "fnu_jy      Jansky (F_ν)" in out
    assert ("Best guess: fnu_jy, Jansky (F_ν); guessed from the size of the flux values, "
            "assuming flux received at Earth.") in " ".join(out.split())


def test_cli_asks_again_after_an_unknown_unit_and_stops_on_no(tmp_path, monkeypatch, capsys):
    from sed_tools import cli

    grid = _surface_flux_grid(tmp_path / "Cool")
    prompts = _answers(monkeypatch, "Jy", "n")
    with pytest.raises(ValueError, match="No flux unit chosen for Cool; it was not cleaned"):
        cli._ask_flux_unit("Cool", sorted(str(p) for p in grid.glob("*.txt")))
    assert len(prompts) == 2
    assert "'Jy' is not one of: flam, fnu, fnu_jy, flam_si, flam_si_nm, flam_si_um, normalized" in capsys.readouterr().out


def test_cli_does_not_ask_when_the_flux_unit_is_declared(tmp_path, monkeypatch):
    from sed_tools import cli

    grid = _surface_flux_grid(tmp_path / "Declared", flux_header="erg/cm2/s/A")
    prompts = _answers(monkeypatch)
    assert cli._ask_flux_unit("Declared", sorted(str(p) for p in grid.glob("*.txt"))) is None
    assert prompts == []


def test_cli_rebuild_asks_once_per_grid_and_cleans_with_the_answer(tmp_path, monkeypatch):
    from sed_tools import cli
    from sed_tools.spectrum_io import read_text_spectrum

    for name in ("A", "B"):
        _surface_flux_grid(tmp_path / name)
    prompts = _answers(monkeypatch, "flam", "fnu_jy")
    cli.run_rebuild_flow(base_dir=str(tmp_path), models=["A", "B"], rebuild_h5=False, rebuild_flux_cube=False)

    assert [p.split(" (")[0] for p in prompts] == ["Flux unit of A", "Flux unit of B"]
    wl, flux = read_text_spectrum(str(tmp_path / "A" / "s0.txt"))
    np.testing.assert_allclose(flux, planck_flam(wl, 3500.0), rtol=1e-8)
    assert "# original_flux_unit = fnu" in (tmp_path / "B" / "s0.txt").read_text()


def test_cli_without_a_terminal_stops_naming_the_grid_and_the_guess(tmp_path, monkeypatch):
    from sed_tools import cli

    grid = _surface_flux_grid(tmp_path / "Cool")
    before = _contents(grid)
    monkeypatch.setattr("sys.stdin", _Stdin(False))
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("asked without a terminal"))

    with pytest.raises(RuntimeError) as err:
        cli.run_rebuild_flow(base_dir=str(tmp_path), models=["Cool"], rebuild_h5=False, rebuild_flux_cube=False)
    message = str(err.value)
    assert f"Cool: ValueError: Flux unit of {grid} is not declared; best guess: 'fnu_jy'" in message
    assert "--flux-unit fnu_jy" in message and "sed-tools asks for it when run in a terminal" in message
    assert _contents(grid) == before


def test_import_grid_stops_before_moving_files_and_imports_with_a_unit(tmp_path):
    from sed_tools.api import SED

    src = _surface_flux_grid(tmp_path / "mine")
    with pytest.raises(ValueError, match=rf"Flux unit of {src} is not declared; best guess: 'fnu_jy'"):
        SED.import_grid(src, model_root=tmp_path / "models", move=True, build_cube=False, build_h5=False)
    assert len(list(src.glob("*.txt"))) == 3
    assert not (tmp_path / "models" / "mine").exists()

    SED.import_grid(src, model_root=tmp_path / "models", move=True, build_cube=False, build_h5=False,
                    flux_unit="flam")
    assert "# original_flux_unit = flam" in (tmp_path / "models" / "mine" / "s0.txt").read_text()


def test_cli_import_asks_for_the_flux_unit_and_passes_it_on(tmp_path, monkeypatch):
    from sed_tools import cli

    src = _surface_flux_grid(tmp_path / "mine")
    calls = []
    monkeypatch.setattr("sed_tools.api.SED.import_grid", lambda *a, **k: calls.append(k))
    _answers(monkeypatch, str(src), "", "n", "flam")
    cli.run_import_flow(base_dir=str(tmp_path / "models"))
    assert calls[0]["flux_unit"] == "flam"


def test_cli_import_command_asks_on_a_terminal_and_not_for_a_dry_run(tmp_path, monkeypatch):
    from sed_tools import cli

    src = _surface_flux_grid(tmp_path / "mine")
    calls = []
    monkeypatch.setattr("sed_tools.api.SED.import_grid", lambda *a, **k: calls.append(k))
    prompts = _answers(monkeypatch, "fnu")
    for extra in ([], ["--dry-run"]):
        monkeypatch.setattr(sys, "argv", ["sed-tools", "import", "--path", str(src),
                                          "--base", str(tmp_path / "models")] + extra)
        assert cli.main() == 0
    assert [c["flux_unit"] for c in calls] == ["fnu", None]
    assert len(prompts) == 1


def test_fetch_stops_on_an_undeclared_flux_unit_and_cleans_with_the_one_given(tmp_path, monkeypatch):
    from sed_tools.api import SED

    def download(cls, catalog, sources, failures, model_dir, workers, cuts):
        _surface_flux_grid(model_dir)
        return "svo"

    monkeypatch.setattr(SED, "_download_from_first_source", classmethod(download))
    with pytest.raises(ValueError, match=r"Flux unit of .*Cool is not declared; best guess: 'fnu_jy'"):
        SED.fetch("Cool", source="svo", model_root=tmp_path)
    sed = SED.fetch("Cool", source="svo", model_root=tmp_path, flux_unit="flam")
    assert len(sed.grid) == 3
    assert "# original_flux_unit = flam" in (tmp_path / "Cool" / "s0.txt").read_text()


# ---------------------------------------------------------------------
# BOSZ spectra: the flux is H_λ, so cleaning stops and asks
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
# ---------------------------------------------------------------------
# Declared flux units: SI factors and header patterns
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


# ----------------------------------------------------------------------
# flux_unit= never replaces a declared flux unit, as --flux-unit does not
# ----------------------------------------------------------------------
def _flam_grid(directory, flux_header):
    directory.mkdir(parents=True)
    wl = np.linspace(3000.0, 10000.0, 50)
    for i, teff in enumerate((3500.0, 3600.0, 3700.0)):
        header = [f"# teff = {teff}\n", "# logg = 4.5\n", "# metallicity = 0.0\n", "# wavelength_unit = Angstrom\n"]
        if flux_header:
            header.append(f"# flux_unit = {flux_header}\n")
        flux = 1e5 * (wl / 5000.0) ** -2
        (directory / f"s{i}.txt").write_text("".join(header) + "".join(f"{w:.4f} {f:.6e}\n" for w, f in zip(wl, flux)))
    return directory


def test_a_declared_flux_unit_is_used_and_a_different_flux_unit_is_reported_unused(tmp_path, caplog):
    from sed_tools.spectra_cleaner import clean_model_dir

    declared = _flam_grid(tmp_path / "Declared", "erg/cm2/s/A")
    reference = _flam_grid(tmp_path / "Reference", "erg/cm2/s/A")
    with caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = clean_model_dir(str(declared), sample_fraction=1.0, flux_unit="fnu")
    clean_model_dir(str(reference), sample_fraction=1.0)
    assert summary["catalog_units"]["flux"] == "flam"  # the header's unit, not fnu
    assert [p.read_text() for p in sorted(declared.glob("s*.txt"))] == \
        [p.read_text() for p in sorted(reference.glob("s*.txt"))]
    assert "the spectra declare their flux unit (flam); flux_unit='fnu' is not used" in caplog.text


def test_flux_unit_is_used_for_spectra_that_declare_none(tmp_path, caplog):
    from sed_tools.spectra_cleaner import clean_model_dir

    grid = _flam_grid(tmp_path / "Undeclared", None)
    with caplog.at_level(logging.WARNING, logger="sed_tools.spectra_cleaner"):
        summary = clean_model_dir(str(grid), sample_fraction=1.0, flux_unit="flam")
    assert summary["catalog_units"]["flux"] == "flam" and "is not used" not in caplog.text
