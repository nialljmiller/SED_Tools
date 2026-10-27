"""Reference-output cases for the SED_Tools cleanup.

Each case runs one science path on the fixed inputs in ``inputs/`` and writes
everything it produces into ``out``. ``record.py`` stored the outputs of
integration/0.4.2 (d622dcc) in ``expected/``; ``tests/test_golden.py`` reruns
every case and requires byte-identical files.

A difference means the code now gives a different answer. Explain it; never
re-record to make it pass. Only the manager re-records.

Inputs:
- stellar_models/Kurucz2003all and Kurucz2003all__alpha_04: 12 spectra each
  (Teff 5750-6250 K, log g 4.0-4.5, [M/H] -0.5-0.0) with their lookup tables;
- filters/: Generic/Johnson, 2MASS/2MASS, LSST/LSST and Palomar/ZTF as served
  by NJM on 2026-10-27, with LSST's and ZTF's component curves;
- atmospheres/lsst_std: Rubin's standard atmosphere at airmass 1.0 and 1.2;
- reference/: NJM's Vega (CALSPEC alpha_lyr_mod_004) and Sun spectra.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import random
import shutil
from pathlib import Path

import numpy as np

INPUTS = Path(__file__).parent / "inputs"
MODELS = INPUTS / "stellar_models"
FILTERS = INPUTS / "filters"
INSTRUMENTS = [FILTERS / "Generic/Johnson", FILTERS / "2MASS/2MASS", FILTERS / "LSST/LSST", FILTERS / "Palomar/ZTF"]
GRID_POINTS = [(5800.0, 4.1, -0.3), (6000.0, 4.5, 0.0), (6200.0, 4.25, -0.45)]

CASES = {}


def case(function):
    CASES[function.__name__] = function
    return function


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(data), indent=1, sort_keys=True) + "\n")


def _plain(value):
    """JSON-ready copy: numpy scalars and arrays become Python floats and lists."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.name
    return value


def copy_tree(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def save_tree(src, out, *, work):
    """Copy every file under ``src`` into ``out``, with the work path replaced in text files."""
    for path in sorted(p for p in Path(src).rglob("*") if p.is_file()):
        target = out / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        try:
            data = data.decode("utf-8").replace(str(work), "<WORK>").encode("utf-8")
        except UnicodeDecodeError:
            pass
        target.write_bytes(data)


def curves(instrument):
    return sorted(p for p in instrument.glob("*.dat"))


# ---------------------------------------------------------------- filters

@case
def filter_headers(work, out):
    from sed_tools.filter_metadata import read_filter_header, released_mesa_reads

    rows = {str(p.relative_to(FILTERS)): {"header": read_filter_header(p), "released_mesa_reads": released_mesa_reads(p)}
            for p in sorted(FILTERS.rglob("*.dat"))}
    write_json(out / "headers.json", rows)


@case
def filter_properties(work, out):
    from sed_tools.filter_properties import instrument_properties, load_sun, load_vega

    vega = load_vega(INPUTS / "reference/vega_flam.csv")
    sun = load_sun(INPUTS / "reference/sun_flam.csv")
    for instrument in INSTRUMENTS:
        write_json(out / f"{instrument.parent.name}_{instrument.name}.json", instrument_properties(instrument, vega, sun))


@case
def filter_colors(work, out):
    from sed_tools.filter_colors import instrument_colors

    for instrument in INSTRUMENTS:
        write_json(out / f"{instrument.parent.name}_{instrument.name}.json", instrument_colors(curves(instrument)))


@case
def compose_lsst_with_atmosphere(work, out):
    from sed_tools.filter_components import build_product

    root = copy_tree(FILTERS, work / "filters")
    atmospheres = copy_tree(INPUTS / "atmospheres", work / "atmospheres")
    product = build_product(root / "LSST/LSST", "filter+optics+qe+atmosphere",
                            atmosphere={"site": "lsst_std", "airmass": 1.2}, atmosphere_root=atmospheres)
    save_tree(product, out / product.name, work=work)


@case
def compose_ztf_qe_variants(work, out):
    from sed_tools.filter_components import build_product

    root = copy_tree(FILTERS, work / "filters")
    for variant in ("single_layer", "multi_layer"):
        product = build_product(root / "Palomar/ZTF", "filter+qe", variants={"qe": variant})
        save_tree(product, out / variant / product.name, work=work)


@case
def combine_filter_sets(work, out):
    from sed_tools.combine_filters import combine_filter_sets as combine

    root = copy_tree(FILTERS, work / "filters")
    combined = combine(work / "combined" / "JohnsonTwoMass", [root / "Generic/Johnson", root / "2MASS/2MASS"],
                       facility="Combined", instrument="JohnsonTwoMass")
    save_tree(combined, out, work=work)


# ---------------------------------------------------------------- spectra

@case
def clean_model_dir(work, out):
    from sed_tools.spectra_cleaner import clean_model_dir as clean

    model = copy_tree(MODELS / "Kurucz2003all", work / "Kurucz2003all")
    random.seed(0)
    report = clean(str(model), try_h5_recovery=False, rebuild_lookup=True, sample_fraction=1.0)
    save_tree(model, out / "model", work=work)
    write_json(out / "report.json", json.loads(json.dumps(report, default=str).replace(str(work), "<WORK>")))


@case
def precompute_flux_cube(work, out):
    from sed_tools.precompute_flux_cube import precompute_flux_cube as precompute

    model = copy_tree(MODELS / "Kurucz2003all", work / "Kurucz2003all")
    precompute(str(model), str(out / "flux_cube.bin"))


@case
def mesa_export(work, out):
    """The MESA-ready folder (spectra, lookup table, flux cube) for each grid variant."""
    from sed_tools.mesa_prepare import export_variant, list_variants
    from sed_tools.precompute_flux_cube import precompute_flux_cube as precompute

    model = copy_tree(MODELS / "Kurucz2003all", work / "Kurucz2003all")
    precompute(str(model), str(model / "flux_cube.bin"))
    for variant in list_variants(model):
        exported = export_variant(variant, work / "export" / variant.label)
        save_tree(exported, out / variant.label, work=work)


@case
def regenerate_lookup_table(work, out):
    from sed_tools.svo_regen_spectra_lookup import regenerate_lookup_table as regenerate

    model = copy_tree(MODELS / "Kurucz2003all", work / "Kurucz2003all")
    files = sorted(str(p) for p in model.glob("*.txt"))
    (work / "lookup").mkdir()
    regenerate(files=files, output_dir=str(work / "lookup"))
    save_tree(work / "lookup", out, work=work)


@case
def interpolation_and_photometry(work, out):
    from sed_tools.models import SEDModel
    from sed_tools.precompute_flux_cube import precompute_flux_cube as precompute

    model = copy_tree(MODELS / "Kurucz2003all", work / "Kurucz2003all")
    precompute(str(model), str(model / "flux_cube.bin"))
    sed = SEDModel(name="Kurucz2003all", flux_cube_path=model / "flux_cube.bin", filters_dir=FILTERS, fill_gaps=True)
    results = {}
    for teff, logg, meta in GRID_POINTS:
        evaluated = sed(teff, logg, meta)
        key = f"{teff}_{logg}_{meta}"
        np.save(out / f"sed_{key}.npy", np.vstack([evaluated.wl, evaluated.fl]))
        bands = [str(p) for i in INSTRUMENTS for p in curves(i)]
        ab = evaluated.photometry(*bands, system="AB")
        vega = evaluated.photometry(*bands, system="Vega", vega_spectrum=INPUTS / "reference/vega_flam.csv")
        results[key] = {"AB": {k: vars(v) for k, v in ab.items()}, "Vega": {k: vars(v) for k, v in vega.items()}}
    write_json(out / "photometry.json", results)


@case
def flux_cube_tool_band_averages(work, out):
    from sed_tools import flux_cube_tool as fct
    from sed_tools.precompute_flux_cube import precompute_flux_cube as precompute

    model = copy_tree(MODELS / "Kurucz2003all", work / "Kurucz2003all")
    precompute(str(model), str(model / "flux_cube.bin"))
    cube = fct.FluxCube.from_file(str(model / "flux_cube.bin"))
    results = {}
    for teff, logg, meta in GRID_POINTS:
        wavelength, flux = cube.interpolate_spectrum(teff, logg, meta)
        row = {"bolometric": fct.bolometric_magnitude(wavelength, flux)}
        for path in (p for i in INSTRUMENTS for p in curves(i)):
            curve = fct.load_filter_curve(str(path))
            name = str(path.relative_to(FILTERS))
            row[name] = {"flam": fct.band_average_flux_lambda(wavelength, flux, curve),
                         "fnu": fct.band_average_flux_nu(wavelength, flux, curve)}
        results[f"{teff}_{logg}_{meta}"] = row
    write_json(out / "band_averages.json", results)


@case
def grid_filter_and_write(work, out):
    from sed_tools.api import SED
    from sed_tools.precompute_flux_cube import precompute_flux_cube as precompute

    models = work / "models"
    model = copy_tree(MODELS / "Kurucz2003all", models / "Kurucz2003all")
    precompute(str(model), str(model / "flux_cube.bin"))
    sed = SED.local("Kurucz2003all", model_root=models, filter_root=FILTERS)
    subset = sed.grid.filter(teff_min=5900, logg_min=4.2)
    written = subset.write(work / "subset", build_flux_cube=True, build_h5=False)
    save_tree(written, out, work=work)


@case
def densify_grid(work, out):
    from sed_tools.grid_densifier import densify_grid as densify
    from sed_tools.precompute_flux_cube import precompute_flux_cube as precompute

    model = copy_tree(MODELS / "Kurucz2003all", work / "Kurucz2003all")
    precompute(str(model), str(model / "flux_cube.bin"))
    densify(str(model / "flux_cube.bin"), str(out / "flux_cube_dense.bin"), teff_spacing=125.0)


@case
def combine_models(work, out):
    from sed_tools.api import SED

    models = work / "models"
    for name in ("Kurucz2003all", "Kurucz2003all__alpha_04"):
        copy_tree(MODELS / name, models / name)
    np.random.seed(0)
    SED.combine(["Kurucz2003all", "Kurucz2003all__alpha_04"], output="combined", model_root=models, visualization=False)
    save_tree(models / "combined", out, work=work)


# ---------------------------------------------------------------- interface

@case
def public_interface(work, out):
    """Public names, call signatures and CLI help: the cleanup must not change them."""
    import sed_tools
    from sed_tools import api, cli

    signatures = {}
    for name in sorted(sed_tools.__all__):
        obj = getattr(sed_tools, name)
        signatures[name] = _signature(obj)
        if inspect.isclass(obj):
            for attr in sorted(vars(obj)):
                if not attr.startswith("_") and callable(getattr(obj, attr)):
                    signatures[f"{name}.{attr}"] = _signature(getattr(obj, attr))
    for cls in (api.Filters, api.SED, api.SpectralGrid, api.Spectrum, api.MLCompleter, api.MLGenerator):
        for attr in sorted(vars(cls)):
            if not attr.startswith("_"):
                signatures[f"api.{cls.__name__}.{attr}"] = _signature(getattr(cls, attr))
    write_json(out / "signatures.json", signatures)

    with _cli_parser(cli.main) as parser:
        helps = {"sed-tools": parser.format_help()}
        for name, sub in sorted(_subcommands(parser).items()):
            helps[name] = sub.format_help()
    write_json(out / "cli_help.json", helps)


def _signature(obj):
    try:
        return str(inspect.signature(obj)).replace(str(Path.home()), "<HOME>")
    except (TypeError, ValueError):
        return type(obj).__name__


class _ParserCaptured(Exception):
    pass


@contextlib.contextmanager
def _cli_parser(main):
    """The ArgumentParser that ``main`` builds, captured when it starts parsing.

    Usage lines take the program name from sys.argv[0] and wrap at $COLUMNS,
    so both stay fixed while the parser is built and its help is formatted.
    """
    import argparse
    import os
    import sys

    captured = []

    def capture(self, *args, **kwargs):
        captured.append(self)
        raise _ParserCaptured

    original = argparse.ArgumentParser.parse_args
    saved_argv, saved_columns = sys.argv, os.environ.get("COLUMNS")
    argparse.ArgumentParser.parse_args = capture
    sys.argv, os.environ["COLUMNS"] = ["sed-tools"], "100"
    try:
        try:
            main()
        except _ParserCaptured:
            pass
        argparse.ArgumentParser.parse_args = original
        yield captured[0]
    finally:
        argparse.ArgumentParser.parse_args = original
        sys.argv = saved_argv
        if saved_columns is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = saved_columns


def _subcommands(parser):
    import argparse

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}
