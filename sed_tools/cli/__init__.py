"""The ``sed-tools`` command: the argument parser, the subcommand handlers, the menu and main().

The menu flows are in cli/grids.py (spectral grids, ML and tools) and cli/filters.py (filter sets);
this module re-exports them.
"""

import argparse
import glob  # noqa: F401  (kept: `from sed_tools.cli import *` gives what 0.5.0 gave)
import inspect
import json
import logging
import os
import re
import shlex  # noqa: F401  (kept, as glob)
import sys
import textwrap  # noqa: F401  (kept, as glob)
import time  # noqa: F401  (kept, as glob)
import traceback
from collections import Counter
from pathlib import Path  # noqa: F401  (kept, as glob)
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: F401  (Sequence kept, as glob)

import numpy as np
from requests import RequestException

# Several names below are imported only so that they stay importable from sed_tools.cli.
from ..api import SED, Filters
from ..component_sources import AUTOMATIC_BUILDERS, BUILDERS, get_builder
from ..config import set_data_dir, show_config
from ..cube.model import FILTER_DIR_DEFAULT, STELLAR_DIR_DEFAULT
from ..filters.compose import ComponentError
from ..filters.njm_catalogue import CatalogueError
from ..grids.clean import FLUX_GUESS_WHY, FLUX_UNITS, clean_model_dir, undeclared_flux_unit
from ..grids.lookup import regenerate_lookup_table
from ..grids.mast import MASTSpectraGrabber
from ..grids.msg import MSGSpectraGrabber
from ..grids.njm import NJMSpectraGrabber
from ..grids.pipeline import _build_data_products, _rebuild_command, build_h5_bundle_from_txt, local_grids
from ..grids.spectrum_io import build_h5_bundle, read_text_spectrum
from ..grids.svo import SVOSpectraGrabber
from ..terminal_plots import terminal_color_enabled, terminal_width
from ..ui_utils import (OPTIONAL_EXTRAS, _ansi, ask_number as _ask_number, missing_package_error, missing_packages,
                        parse_multi_selection, parse_numeric_range, prompt_choice, refused_without)
from .filters import (
    _FilterSetOption, _UNSTATED, _ask_conditions, _atmosphere_sites, _availability_errors, _browse_instruments,
    _catalogue_entry, _composable_bands, _compose_the_bands_that_work, _default_site, _download_origin,
    _fetch_instrument, _installed_availability, _offer_composition, _offer_download, _pick_summary, _picked_names,
    _picker_labels, _print_filter_set, _print_instrument_descriptions, print_catalogue_entry,
    print_filter_availability, run_filter_combine_flow, run_filter_compose_flow, run_filter_info_flow,
    run_filters_flow,
)
from .grids import (
    _ask_flux_unit, _ask_import_flux_unit, _download_cuts, _download_from_first_source, _print_cleaning_report,
    _report_import_flux_unit, ensure_dir, prompt_njm_axis_cuts, run_combine_flow, run_coverage_flow,
    run_grid_densifier_flow, run_import_flow, run_mesa_prepare_flow, run_ml_completer_flow, run_ml_generator_flow,
    run_rebuild_flow, run_spectra_flow,
)

logger = logging.getLogger(__name__)

#: The errors a workflow raises for a request it cannot meet. main() and the
#: menu print their message; anything else is a bug and keeps its traceback.
EXPECTED_ERRORS = (RuntimeError, ValueError, OSError, ComponentError, CatalogueError,
                   RequestException, EOFError)


def load_txt_spectrum(txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return read_text_spectrum(txt_path)


def run_config_flow() -> None:
    show_config()
    print("\nEnter new data directory path, or press Enter (or q) to keep current:")
    raw = input("> ").strip()
    # q and b leave the setting alone, as in the pickers; a directory with that name is ./q
    if not raw or raw.lower() in ("q", "b"):
        return
    # set_data_dir handles the "move existing data?" prompt itself.
    set_data_dir(raw, interactive=True)
    print("Restart sed-tools for the change to take effect.")


#: The menu's sections, each a heading and its (command, label) entries, numbered from 1 in this order.
_MENU = (
    ("Filters", (("filters", "Download Filters (NJM / SVO)"),
                 ("filters_combine", "Combine filter sets"),
                 ("filter_info", "Filter set information"),
                 ("filter_compose", "Build a filter curve from its parts (filter, optics, camera, atmosphere)"))),
    ("Spectral grids", (("grids", "Download spectral grids (NJM / SVO / MSG / MAST)"),
                        ("rebuild", "Rebuild (lookup + HDF5 + flux cube)"),
                        ("combine", "Combine grids into omni grid"))),
    ("ML", (("ml_completer", "ML SED Completer (train/extend incomplete SEDs)"),
            ("ml_generator", "ML SED Generator (generate SEDs from parameters)"),
            ("grid_densifier", "Grid Densifier (fill coarse Teff gaps)"))),
    ("Tools", (("coverage", "Coverage (parameter-space summary + plot)"),
               ("import", "Import a local spectral grid"),
               ("mesa_prepare", "Prepare a spectral grid for MESA"),
               ("config", "Config (data directory)"))),
)


def menu() -> str:
    BOLD, DIM, CYAN, YELL, RED, RESET = _ansi(terminal_color_enabled("auto"),
                                              "bold", "dim", "cyan", "yellow", "red", "reset")
    print(BOLD + "=" * 50 + RESET)
    mapping = dict.fromkeys(("q", "Q", "quit", "QUIT", "Quit", "0"), "quit")
    number = 0
    for heading, entries in _MENU:
        print("\n" + YELL + f"-- {heading} ".ljust(50, "-") + RESET)
        for command, label in entries:
            number += 1
            mapping[str(number)] = command
            print(" " * (3 - len(str(number))) + f"{CYAN}{number}){RESET} {label}")
    print("\n" + DIM + "-" * 50 + RESET)
    print(f"  {RED}0/q){RESET} Quit")
    print(BOLD + "=" * 50 + RESET)
    return mapping.get(input("> ").strip(), "")

# ------------ Command-line parser ------------

def _build_parser() -> argparse.ArgumentParser:
    """The ``sed-tools`` parser, subcommands in --help order; each subcommand's parser is its
    ``command_parser`` default, so its handler's usage errors show that subcommand's usage line."""
    parser = argparse.ArgumentParser(description="SED Tools — spectral grids and photometric filters")
    sub = parser.add_subparsers(dest="cmd", required=False)
    _add_grid_download_parser(sub)
    _add_filter_download_parsers(sub)
    _add_filter_catalogue_parsers(sub)
    _add_filter_component_parsers(sub)
    _add_grid_build_parsers(sub)
    _add_grid_tool_parsers(sub)
    for command_parser in sub.choices.values():
        command_parser.set_defaults(command_parser=command_parser)
    return parser


def _add_base_option(parser, default, help: str) -> None:
    parser.add_argument("--base", default=str(default), help=help)


def _add_flux_unit_option(parser) -> None:
    parser.add_argument("--flux-unit", choices=list(FLUX_UNITS),
                        help="Flux unit of spectra whose headers do not declare one (otherwise sed-tools "
                             "asks in a terminal, and stops without one); spectra that declare theirs keep it")


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {text!r}")
    return value


def _airmass_arg(text):
    """argparse type: an airmass, or 'unstated' for an atmosphere published without one."""
    if str(text).strip().lower() == _UNSTATED:
        return _UNSTATED
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or 'unstated', got {text!r}") from None


def _add_filter_product_options(parser, *, components=None):
    """Keep filter fetch and composition options identical."""
    _add_base_option(parser, FILTER_DIR_DEFAULT, "Base filter directory")
    parser.add_argument("--components", default=components,
                        help="Roles separated by commas or '+': filter, optics, qe, published")
    parser.add_argument("--bands", nargs="+", help="Compose only these bands")
    parser.add_argument("--detector-type", choices=["energy", "photon"],
                        help="Override integration weighting in a derived copy; preserves published curves")
    parser.add_argument("--airmass", type=_airmass_arg,
                        help="Add the instrument's own-site atmosphere at this airmass ('unstated' for a "
                             "team atmosphere whose source gives none)")
    parser.add_argument("--pwv", type=float, help="Atmospheric precipitable water vapour (mm)")
    parser.add_argument("--atmosphere-site",
                        help="One of the instrument's own atmosphere sites (default: its default site)")
    parser.add_argument("--allow-heavy", action="store_true", help="Permit large original-source downloads")
    parser.add_argument("--allow-partial", action="store_true", help="Use zero outside component wavelength coverage")


def _add_grid_download_parser(sub) -> None:
    sp = sub.add_parser("grids", aliases=["spectra"], help="Download and build spectral grids")
    sp.add_argument("--source",
                    choices=["njm", "svo", "msg", "mast", "both", "all"],
                    default="all",
                    help="Which provider(s) to use. NJM mirror is checked first when available.")
    sp.add_argument("--grids", "--models", dest="models", nargs="*", default=None,
                    help="Spectral grid names to process, or 'all'")
    _add_base_option(sp, STELLAR_DIR_DEFAULT, "Output directory for spectral grids")
    sp.add_argument("--workers", type=int, default=5, help="Parallel workers")
    sp.add_argument("--no-h5", action="store_true", help="Do not force-create HDF5 bundle")
    sp.add_argument("--no-cube", action="store_true", help="Skip flux cube build")
    _add_flux_unit_option(sp)
    cut = ("; NJM, MSG and MAST cut downloads, SVO refuses cuts. Without any cut option an NJM "
           "download asks for cuts in a terminal and takes the full grid otherwise")
    sp.add_argument("--teff", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Download only spectra with Teff in this range (K)" + cut)
    sp.add_argument("--logg", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Download only spectra with log g in this range")
    sp.add_argument("--meta", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Download only spectra with [M/H] in this range")
    sp.add_argument("--wavelength", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Keep only this wavelength range (Å)")


def _add_filter_download_parsers(sub) -> None:
    fp = sub.add_parser("filters", help="Download instrument passbands or browse interactively")
    fp.add_argument("target", nargs="?", help="Facility/Instrument; omit to browse")
    _add_filter_product_options(fp)

    fcp = sub.add_parser("filters-combine", help="Combine filter sets into one MESA-compatible instrument")
    fcp.add_argument("output", help="Output name or Facility/Instrument path for the combined filter set")
    fcp.add_argument("inputs", nargs="+", help="Filter-set directories or .dat files to combine")
    _add_base_option(fcp, FILTER_DIR_DEFAULT, "Base filter directory")
    fcp.add_argument("--facility", default=None, help="Output facility label (default: Combined)")
    fcp.add_argument("--instrument", default=None, help="Output instrument/index-file name")
    fcp.add_argument("--on-conflict", choices=["rename", "overwrite", "error"], default="rename",
                     help="How duplicate filter filenames are handled")

    fdp = sub.add_parser("filters-detector",
                         help="Tag filter files as photon- or energy-counting (DetectorType header)")
    fdp.add_argument("path", nargs="?", default=str(FILTER_DIR_DEFAULT),
                     help="Filter root, instrument directory or single .dat file")
    fdp.add_argument("--facility", default=None,
                     help="SVO facility name when PATH is a single instrument directory or file")
    fdp.add_argument("--force", action="store_true",
                     help="Re-resolve files that are already tagged; a result never replaces a type "
                          "from a stronger source (original > inferred > svo > svo-curve-match > assumed)")
    fdp.add_argument("--dry-run", action="store_true",
                     help="Report without modifying files")


def _add_filter_catalogue_parsers(sub) -> None:
    fip = sub.add_parser("filters-info",
                         help="Show what a filter set contains and which components can be requested")
    fip.add_argument("target", help="Facility/Instrument, e.g. SLOAN/SDSS")
    _add_base_option(fip, FILTER_DIR_DEFAULT, "Base filter directory")

    fip.add_argument("--offline", action="store_true", help="Inspect local data without contacting sources")
    fip.add_argument("--json", action="store_true", help="Emit availability as JSON")


def _add_filter_component_parsers(sub) -> None:
    fcop = sub.add_parser("filters-compose", help="Compose optical-path components")
    fcop.add_argument("target", help="Facility/Instrument or installed instrument directory")
    _add_filter_product_options(fcop, components="filter")

    finv = sub.add_parser("filters-inventory", help="Inventory all SVO, NJM, local and registered instruments")
    _add_base_option(finv, FILTER_DIR_DEFAULT, "Base filter directory")
    finv.add_argument("--facility", help="Limit discovery to this facility")
    finv.add_argument("--offline", action="store_true", help="List local and registered instruments only")
    finv.add_argument("--json", action="store_true", help="Emit inventory and source errors as JSON")

    fbp = sub.add_parser("filters-build-components",
                         help="(Mirror curation) populate components/atmospheres from a published source")
    fbp.add_argument("source", help=f"Component source: {', '.join(sorted(BUILDERS))}; or 'all' for "
                                    f"those that need no input ({', '.join(AUTOMATIC_BUILDERS)})")
    _add_base_option(fbp, FILTER_DIR_DEFAULT, "Base filter directory")
    fbp.add_argument("--raw", default=None, help="Raw input directory (source specific)")
    fbp.add_argument("--instrument", help="Instrument to build, for a source that holds several "
                                          "(e.g. tabular, surveys, space); a source that needs one says so")
    fbp.add_argument("--sites", nargs="+", default=None, help="SkyCalc or team atmosphere sites")
    fbp.add_argument("--archive", default=None,
                     help="Keep every raw download here with its URL, time and SHA-256 (skycalc, team_atmospheres)")
    fbp.add_argument("--airmass", nargs="+", type=float, default=None, help="SkyCalc airmass grid")
    fbp.add_argument("--pwv", nargs="+", type=float, default=None, help="SkyCalc PWV grid (mm)")


# densify accepts --ml-gap-threshold only at this default, since densify_grid has no such threshold.
_ML_GAP_THRESHOLD_DEFAULT = 5000.0


def _add_grid_build_parsers(sub) -> None:
    rp = sub.add_parser("rebuild", help="Rebuild lookup table + HDF5 + flux cube")
    _add_base_option(rp, STELLAR_DIR_DEFAULT, "Base spectral grid dir")
    rp.add_argument("--grids", "--models", dest="models", nargs="*", default=None,
                    help="Names of local spectral grids to rebuild")
    rp.add_argument("--no-h5", action="store_true", help="Skip HDF5 bundle")
    rp.add_argument("--no-cube", action="store_true", help="Skip flux cube")
    _add_flux_unit_option(rp)

    cp = sub.add_parser("combine", help="Combine multiple grids into omni grid")
    _add_base_option(cp, STELLAR_DIR_DEFAULT, "Base spectral grid directory")
    cp.add_argument("--output", default="combined_models",
                    help="Name for the combined spectral grid")
    cp.add_argument("--non-interactive", action="store_true",
                    help="Combine without prompting: the --grids given, else every local spectral grid "
                         "except earlier combined ones")
    cp.add_argument("--grids", "--models", dest="models", nargs="+", default=None,
                    help="Names of the local spectral grids to combine (no prompts)")

    mp = sub.add_parser("ml_completer", help="Train/use ML SED completer")
    _add_base_option(mp, STELLAR_DIR_DEFAULT, "Base spectral grid directory")
    mp.add_argument("--models-dir", default="models",
                    help="Directory for trained ML models")

    gp = sub.add_parser("ml_generator", help="Train/use ML SED generator")
    _add_base_option(gp, STELLAR_DIR_DEFAULT, "Base spectral grid directory")
    gp.add_argument("--models-dir", default="models",
                    help="Directory for trained ML models")
    gp.add_argument("--auto", action="store_true",
                    help="Run optimized auto-training with hyperparameter search")
    gp.add_argument("--library", help="Library path for auto-training")
    gp.add_argument("--output", help="Output directory for auto-training")
    gp.add_argument("--trials", type=_positive_int, default=None,
                    help="With --auto: number of hyperparameter search trials (default 20)")
    gp.add_argument("--epochs", type=_positive_int, default=None,
                    help="With --auto: maximum epochs of the final training (default 100)")

    p_densify = sub.add_parser("densify", help="Fill coarse Teff gaps in a flux_cube.bin")
    p_densify.add_argument("--flux-cube", required=True, help="The flux_cube.bin to densify")
    p_densify.add_argument("--output", required=True, help="Where to write the densified flux cube")
    p_densify.add_argument("--teff-spacing", type=float, default=1000.0, help="Target Teff spacing in K")
    p_densify.add_argument("--method", default="auto",
                           choices=["auto", "interp", "ml", "blackbody"],
                           help="auto: fill gaps inside the grid's Teff range with --ml-model if given, and "
                                "elsewhere with a blackbody scaled to the nearest spectrum; ml: the same, "
                                "--ml-model required; blackbody: blackbodies only; interp: not implemented, refused")
    p_densify.add_argument("--ml-model", default=None,
                           help="Trained ML generator for gaps inside the grid's Teff range")
    p_densify.add_argument("--ml-gap-threshold", type=float, default=_ML_GAP_THRESHOLD_DEFAULT,
                           help="Not implemented: refused unless left at its default")
    p_densify.add_argument("--no-lookup", action="store_true",
                           help="No effect: densify writes only the flux cube")


def _add_grid_tool_parsers(sub) -> None:
    pp = sub.add_parser("mesa_prepare",
        help="Export a spectral grid variant as a clean MESA-ready folder")
    _add_base_option(pp, STELLAR_DIR_DEFAULT, "Base spectral grid directory")
    pp.add_argument("--grid", "--model", dest="model", default=None,
                    help="Spectral grid name (e.g. Kurucz2003all); prompted if omitted")
    pp.add_argument("--output", default=None,
                    help="Output directory for the exported variant; prompted if omitted")

    cfg_p = sub.add_parser("config", help="Show or set the data directory")
    cfg_p.add_argument("--set", metavar="PATH", default=None,
                       help="Set data directory to PATH")
    cfg_p.add_argument("--move", action="store_true",
                       help="With --set: move existing data to the new path")

    covp = sub.add_parser("coverage",
        help="Report parameter-space coverage of a local grid")
    _add_base_option(covp, STELLAR_DIR_DEFAULT, "Base spectral grid dir")
    covp.add_argument("--grids", "--models", dest="models", nargs="*", default=None,
                      help="Spectral grid folder name(s); default: every local grid")
    covp.add_argument("--no-plot", action="store_true",
                      help="Skip the coverage plot")
    covp.add_argument("--out", default=None,
                      help="Output path for the plot PNG (single spectral grid only)")

    imp = sub.add_parser("import",
        help="Ingest a local grid of .txt spectra into the pipeline")
    imp.add_argument("--path", required=True,
                     help="Directory of .txt spectra (or a single .txt file)")
    imp.add_argument("--name", default=None,
                     help="Spectral grid name (default: source folder name)")
    _add_base_option(imp, STELLAR_DIR_DEFAULT, "Base spectral grid dir")
    imp.add_argument("--move", action="store_true",
                     help="Move files instead of copying")
    imp.add_argument("--no-h5", action="store_true", help="Skip HDF5 bundle")
    imp.add_argument("--no-cube", action="store_true", help="Skip flux cube")
    imp.add_argument("--dry-run", action="store_true",
                     help="Report parseable headers and whether the flux unit is declared, without importing")
    _add_flux_unit_option(imp)


# ------------ Command handlers ------------
# Each takes the parsed arguments and its subcommand's parser (for .error)
# and returns the process exit status: 0 done, 1 failed, 2 usage error.

def _range_option(args, name: str, parser) -> Optional[Tuple[float, float]]:
    """A ``--NAME LOW HIGH`` cut as ``(low, high)``, or None when not given."""
    values = getattr(args, name)
    if values is None:
        return None
    low, high = values
    if low > high:
        parser.error(f"--{name} LOW HIGH: {low:g} is above {high:g}")
    return low, high


def _cmd_grids(args, parser):
    run_spectra_flow(
        source=args.source,
        base_dir=args.base,
        models=args.models,
        workers=args.workers,
        force_bundle_h5=not args.no_h5,
        build_flux_cube=not args.no_cube,
        flux_unit=args.flux_unit,
        teff_range=_range_option(args, "teff", parser),
        logg_range=_range_option(args, "logg", parser),
        meta_range=_range_option(args, "meta", parser),
        wl_range=_range_option(args, "wavelength", parser),
    )
    return 0


def _facility_instrument(target: str, parser) -> List[str]:
    parts = target.strip("/").split("/")
    if len(parts) != 2:
        parser.error("target must be Facility/Instrument")
    return parts


def _atmosphere_request(args, parser) -> Optional[Dict]:
    """The ``atmosphere`` argument of Filters.fetch/compose, from --airmass, --pwv and --atmosphere-site."""
    if args.pwv is not None and args.airmass is None:
        parser.error("--pwv needs --airmass")
    if args.atmosphere_site and args.airmass is None:
        parser.error("--atmosphere-site needs --airmass (a number, or 'unstated')")
    if "atmosphere" in re.split(r"[\s,+]+", (args.components or "").lower()) and args.airmass is None:
        parser.error("--components atmosphere needs --airmass (a number, or 'unstated')")
    if args.airmass is None:
        return None
    atmosphere = {"airmass": None if args.airmass == _UNSTATED else args.airmass, "pwv": args.pwv}
    if args.atmosphere_site:
        atmosphere["site"] = args.atmosphere_site
    return atmosphere


def _cmd_filters(args, parser):
    """``filters`` and ``filters-compose``: download (or browse) an instrument, optionally composing it."""
    if not args.target:
        if any((args.components, args.bands, args.detector_type, args.airmass is not None,
                args.pwv is not None, args.atmosphere_site, args.allow_heavy, args.allow_partial)):
            parser.error("component options require a Facility/Instrument target")
        return run_filters_flow(base_dir=args.base)
    options = dict(filter_root=args.base, components=args.components, atmosphere=_atmosphere_request(args, parser),
                   bands=args.bands, detector_type=args.detector_type,
                   allow_heavy=args.allow_heavy, allow_partial=args.allow_partial)
    try:
        if args.cmd == "filters-compose":
            Filters.compose(args.target, **options)
        else:
            print(Filters.fetch(*_facility_instrument(args.target, parser), **options))
    except (ComponentError, RequestException, OSError, RuntimeError) as exc:
        print(f"[filters] {_atmosphere_error(exc, args) if options['atmosphere'] else exc}")
        return 1
    return 0


def _atmosphere_error(exc: Exception, args) -> str:
    """A failed composition's error, in command-line terms (--atmosphere-site) for an instrument with
    several sites and no default; a site's own errors already say which conditions it has."""
    from ..filters.library import FilterLibrary

    if args.atmosphere_site:
        return str(exc)
    directory = FilterLibrary(args.base).directory(args.target)
    try:
        info = _installed_availability(directory)
    except (ComponentError, OSError, RuntimeError) as lookup_error:
        return f"{exc} (its atmosphere sites could not be listed: {lookup_error})"
    sites = _atmosphere_sites(info)
    if len(sites) > 1 and _default_site(info, sites) is None:
        return (f"{directory.parent.name}/{directory.name} has several atmosphere sites "
                f"({', '.join(sites)}) and no default; choose one with --atmosphere-site")
    return str(exc)


def _cmd_filters_combine(args, parser):
    from ..filters.combine import combine_filter_sets

    out = combine_filter_sets(
        args.output,
        args.inputs,
        filter_root=args.base,
        facility=args.facility,
        instrument=args.instrument,
        on_conflict=args.on_conflict,
    )
    print(f"[filters] Combined filter set written to {out}")
    return 0


def _cmd_filters_detector(args, parser):
    # Each file is printed as it is tagged, so on failure main() has only the error left to say.
    rows = Filters.tag_detector_types(
        args.path, facility=args.facility, force=args.force, dry_run=args.dry_run
    )
    counts = Counter((r["detector_type"], r["source"]) for r in rows)
    changed = sum(r["changed"] for r in rows)
    verb = "would change" if args.dry_run else "changed"
    migrated = sum(r["migrated"] for r in rows)
    moved = "would move" if args.dry_run else "moved"
    print(f"\n[filters] {len(rows)} files, {changed} {verb}, {migrated} {moved} 0.3.0 '#' metadata to the column row")
    for (dtype, source), n in sorted(counts.items()):
        print(f"  {dtype:6s} [{source}]: {n}")
    kept = [r for r in rows if r["kept"]]
    if kept:
        differ = sum(r["resolved_type"] != r["detector_type"] for r in kept)
        print(f"[filters] {len(kept)} kept their DetectorType: the new resolution does not outrank "
              f"their DetectorTypeSource and was not written ({differ} of them resolve to the other type)")
        for r in kept:
            print(f"  {r['path']}: {r['note']}")
    return 0


def _cmd_filters_info(args, parser):
    target = _facility_instrument(args.target, parser)
    info = Filters.available(*target, filter_root=args.base, include_remote=not args.offline)
    catalogue = _catalogue_entry(*target, args.base, offline=args.offline)
    if args.json:
        print(json.dumps(dict(info, catalogue=catalogue), indent=2))
    else:
        _print_filter_set("/".join(target), info, catalogue)
    # NJM's catalogue (description, licence, band properties) is an extra: without it the
    # availability report still stands, so only an incomplete availability report fails.
    return 1 if _availability_errors(info) else 0


def _cmd_filters_inventory(args, parser):
    result = Filters.inventory(filter_root=args.base, include_remote=not args.offline, facility=args.facility)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for row in result["instruments"]:
            status = row["component_availability"]["status"]
            print(f"{row['facility']}/{row['instrument']}: {status} [{', '.join(row['sources'])}]")
        for error in result["errors"]:
            print(f"[filters] {error}")
    return 1 if result["errors"] else 0


def _cmd_filters_build_components(args, parser):
    if args.source == "all":
        return _build_all_component_sources(args.base, args.sites)
    if args.source.lower() not in BUILDERS:
        parser.error(f"unknown source {args.source!r}; choose one of: {', '.join(sorted(BUILDERS))}, or all")
    builder = get_builder(args.source)
    needs = inspect.signature(builder).parameters.get("instrument")
    if not args.instrument and needs is not None and needs.default is inspect.Parameter.empty:
        parser.error(f"{args.source} builds several instruments; choose one with --instrument")
    if args.source.lower() == "tabular":
        from ..component_sources.tabular import SOURCES

        if args.instrument not in SOURCES:
            parser.error(f"tabular has no instrument {args.instrument!r}; choose one of: {', '.join(SOURCES)}")
    options = (("raw", args.raw), ("instrument", args.instrument), ("sites", args.sites),
               ("airmasses", args.airmass), ("pwvs", args.pwv), ("archive_dir", args.archive))
    kwargs = {key: value for key, value in options if value}
    print(f"[filters] Built {builder(args.base, **kwargs)}")
    return 0


def _build_all_component_sources(base: str, sites: Optional[List[str]]) -> int:
    """Run every automatic builder; report each failure and return 1 if any failed."""
    failed = []
    for name in AUTOMATIC_BUILDERS:
        kw = {"sites": sites or ["paranal", "lasilla"]} if name == "skycalc" else {}
        print(f"[filters] === {name} ===")
        try:
            get_builder(name)(base, **kw)
        except (ValueError, RequestException, OSError, RuntimeError) as exc:
            failed.append(f"{name}: {exc}")
    for line in failed:
        print(f"[filters] FAILED {line}")
    return 1 if failed else 0


def _cmd_rebuild(args, parser):
    run_rebuild_flow(
        base_dir=args.base,
        models=args.models,
        rebuild_h5=not args.no_h5,
        rebuild_flux_cube=not args.no_cube,
        flux_unit=args.flux_unit,
    )
    return 0


def _cmd_combine(args, parser):
    run_combine_flow(
        base_dir=args.base,
        output_name=args.output,
        interactive=not (args.non_interactive or args.models),
        models=args.models,
    )
    return 0


def _cmd_ml_completer(args, parser):
    return run_ml_completer_flow(
        base_dir=args.base,
        models_dir=args.models_dir
    ) or 0


def _cmd_ml_generator(args, parser):
    limits = {key: value for key, value in (("n_trials", args.trials), ("epochs", args.epochs)) if value}
    if not args.auto:
        if limits:
            parser.error("--trials and --epochs apply to --auto training only")
        return run_ml_generator_flow(
            base_dir=args.base,
            models_dir=args.models_dir
        ) or 0
    if not args.library or not args.output:
        parser.error("--auto needs --library and --output")
    if refused_without("torch", "sklearn", "optuna"):
        return 1
    from ..ml import auto_train_generator
    auto_train_generator(library=args.library, output=args.output, **limits)
    return 0


def _cmd_densify(args, parser):
    """densify_grid fills each Teff gap with --ml-model inside the source grid's Teff range, and
    with a blackbody scaled to the nearest real spectrum otherwise; options it cannot honour are refused."""
    from ..cube.densify import densify_grid

    if args.method == "interp":
        parser.error("--method interp is not implemented: densify fills gaps with --ml-model or blackbodies")
    if args.method == "ml" and not args.ml_model:
        parser.error("--method ml needs --ml-model")
    if args.method == "blackbody" and args.ml_model:
        parser.error("--method blackbody does not use --ml-model; leave one of them out")
    if args.ml_gap_threshold != _ML_GAP_THRESHOLD_DEFAULT:
        parser.error("--ml-gap-threshold is not implemented: with --ml-model, every gap inside the source "
                     "grid's Teff range is filled by the model")
    # --no-lookup needs nothing: densify_grid writes only the flux cube.
    densify_grid(
        src=args.flux_cube,
        dst=args.output,
        teff_spacing=args.teff_spacing,
        ml_model=args.ml_model,
    )
    return 0


def _cmd_mesa_prepare(args, parser):
    run_mesa_prepare_flow(
        base_dir=args.base,
        model=args.model,
        output=args.output,
    )
    return 0


def _cmd_config(args, parser):
    if args.set:
        set_data_dir(args.set, move=args.move or None)
    else:
        show_config()
    return 0


def _cmd_coverage(args, parser):
    names = args.models or local_grids(args.base)
    if not names:
        print(f"No local grids found under {args.base}")
        return 1
    failed = []
    for nm in names:
        try:
            SED.coverage(
                nm,
                model_root=args.base,
                plot=not args.no_plot,
                out_path=args.out if len(names) == 1 else None,
            )
        except (ValueError, OSError, RuntimeError) as exc:
            failed.append(f"{nm}: {exc}")
    for line in failed:
        print(f"[coverage] FAILED {line}")
    return 1 if failed else 0


def _cmd_import(args, parser):
    flux_unit = None if args.dry_run else _ask_import_flux_unit(args.path, args.name, args.flux_unit)
    SED.import_grid(
        args.path,
        name=args.name,
        model_root=args.base,
        move=args.move,
        build_h5=not args.no_h5,
        build_cube=not args.no_cube,
        dry_run=args.dry_run,
        flux_unit=flux_unit,
    )
    if args.dry_run:
        _report_import_flux_unit(args.path, args.name, args.flux_unit)
    return 0


_COMMANDS = {
    "grids": _cmd_grids,
    "spectra": _cmd_grids,
    "filters": _cmd_filters,
    "filters-compose": _cmd_filters,
    "filters-combine": _cmd_filters_combine,
    "filters-detector": _cmd_filters_detector,
    "filters-info": _cmd_filters_info,
    "filters-inventory": _cmd_filters_inventory,
    "filters-build-components": _cmd_filters_build_components,
    "rebuild": _cmd_rebuild,
    "combine": _cmd_combine,
    "ml_completer": _cmd_ml_completer,
    "ml_generator": _cmd_ml_generator,
    "densify": _cmd_densify,
    "mesa_prepare": _cmd_mesa_prepare,
    "config": _cmd_config,
    "coverage": _cmd_coverage,
    "import": _cmd_import,
}


# ------------ Interactive mode ------------

def _print_banner() -> None:
    use_color = terminal_color_enabled("auto")
    BOLD, RESET = _ansi(use_color, "bold", "reset")
    banner = ("▄▖▄▖▄   ▄▖    ▜   ",
              "▚ ▙▖▌▌  ▐ ▛▌▛▌▐ ▛▘",
              "▄▌▙▖▙▘▄▖▐ ▙▌▙▌▐▖▄▌")
    print()
    print("\n" + BOLD + "=" * 50 + RESET)
    for line, color in zip(banner, _ansi(use_color, "red", "green", "blue")):
        print(f"{BOLD}{color}{line}{RESET}")


def _run_menu() -> int:
    """Run the workflows chosen from the menu until the user quits; 1 if any failed (each says why), else 0."""
    _print_banner()
    flows = {
        "grids": lambda: run_spectra_flow(source="all"),
        "filters": run_filters_flow,
        "filters_combine": run_filter_combine_flow,
        "filter_info": run_filter_info_flow,
        "filter_compose": run_filter_compose_flow,
        "rebuild": run_rebuild_flow,
        "combine": run_combine_flow,
        "ml_completer": run_ml_completer_flow,
        "ml_generator": run_ml_generator_flow,
        "grid_densifier": run_grid_densifier_flow,
        "mesa_prepare": run_mesa_prepare_flow,
        "config": run_config_flow,
        "coverage": run_coverage_flow,
        "import": run_import_flow,
    }
    failed = False
    while True:
        try:
            choice = menu()
        except (EOFError, KeyboardInterrupt):  # input closed, or Ctrl-C at the menu: quit
            print()
            return int(failed)
        if choice == "quit":
            return int(failed)
        if choice not in flows:
            print("Unknown option: enter a number from the menu, or q to quit.")
            continue
        if not _run_menu_flow(flows[choice]):
            failed = True


def _run_menu_flow(flow) -> bool:
    """Run one menu workflow; say why if it fails, and return whether it succeeded."""
    try:
        status = flow()
    except EXPECTED_ERRORS as exc:
        _print_error(exc)
        return False
    except ImportError as exc:  # an optional package a step needs (scipy, torch, ...): say what to install
        missing = missing_package_error(exc)
        if missing is None:
            raise
        print(missing, file=sys.stderr)
        return False
    except KeyboardInterrupt:
        print("\nInterrupted; back to the menu.")
        return True
    except SystemExit as exc:  # mesa_prepare stops with SystemExit(message)
        if exc.code in (None, 0):
            return True
        print(exc.code if isinstance(exc.code, str) else f"Stopped with status {exc.code}.", file=sys.stderr)
        return False
    return status in (None, 0)


def _print_error(exc: BaseException) -> None:
    """``sed-tools: error: <message>`` on stderr; with SED_TOOLS_DEBUG=1 the traceback first."""
    if os.environ.get("SED_TOOLS_DEBUG") == "1":
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    if isinstance(exc, EOFError):
        message = ("input ended before a question was answered; run it in a terminal, "
                   "or see the command's --help for options that give the answers")
    else:
        message = str(exc) or type(exc).__name__
    print(f"sed-tools: error: {message}", file=sys.stderr)


# ------------ Main CLI ------------

def main():
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.cmd is None:
            return _run_menu()
        return _COMMANDS[args.cmd](args, args.command_parser)
    except EXPECTED_ERRORS as exc:
        _print_error(exc)
        return 1
    except ImportError as exc:  # an optional package a step needs (scipy, torch, ...): say what to install
        missing = missing_package_error(exc)
        if missing is None:
            raise
        print(missing, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nsed-tools: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
