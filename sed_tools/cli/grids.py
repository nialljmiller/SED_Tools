"""The spectral-grid flows of ``sed-tools``: download, rebuild, combine, import and coverage, and the
ML, densifier and MESA forwarders. The cli package re-exports them."""

import glob
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from requests import RequestException

from ..api import SED
from ..cube.model import STELLAR_DIR_DEFAULT
from ..grids.clean import FLUX_GUESS_WHY, FLUX_UNITS, clean_model_dir, undeclared_flux_unit
from ..grids.lookup import regenerate_lookup_table
from ..grids.pipeline import (_build_combined_grid, _build_data_products, _build_flux_cube, _directory_state,
                              _download_count, _import_sources, build_h5_bundle_from_txt, download_from_first_source,
                              local_grids, rebuildable_grids, spectra_grabber)
from ..ui_utils import ask_range, parse_multi_selection, prompt_choice, wrap

logger = logging.getLogger("sed_tools.cli")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def prompt_njm_axis_cuts(
    name: str,
    grabber,  # NJMSpectraGrabber instance
    model_url: Optional[str] = None,
) -> Dict:
    """Ask for axis cuts on an NJM download, after showing the grid's ranges from its remote lookup table.

    Returns a dict with keys teff_range, logg_range, meta_range and wl_range, each a (min, max) tuple or None.
    """
    print(f"\n  ── Axis Cuts for {name} ──")
    print(f"  You can restrict the download to a subset of the grid.")
    print(f"  Leave blank to download everything.\n")

    info = grabber.get_grid_info(name, model_url=model_url)  # None after a warning saying why
    if info:
        print(f"  Available grid ({info['n_spectra']} spectra):")
        if 'teff_min' in info:
            print(f"    Teff:  {info['teff_min']:.0f} – {info['teff_max']:.0f} K  ({info['teff_unique']} values)")
        if 'logg_min' in info:
            print(f"    logg:  {info['logg_min']:.2f} – {info['logg_max']:.2f}    ({info['logg_unique']} values)")
        if 'meta_min' in info:
            print(f"    [M/H]: {info['meta_min']:+.2f} – {info['meta_max']:+.2f}    ({info['meta_unique']} values)")
        print()

    cuts = {'teff_range': ask_range("  Teff range (e.g. '3500,8000') or blank for all: "),
            'logg_range': ask_range("  logg range (e.g. '3.5,5.0') or blank for all: "),
            'meta_range': ask_range("  [M/H] range (e.g. '-1.0,0.5') or blank for all: "),
            'wl_range': ask_range("  Wavelength range in Å (e.g. '3000,10000') or blank for all: ")}

    # The grabber prints the cuts as it applies them.
    if all(v is None for v in cuts.values()):
        print("\n  No cuts — downloading full grid.\n")

    return cuts


def run_rebuild_flow(
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models: Optional[List[str]] = None,
    rebuild_h5: bool = True,
    rebuild_flux_cube: bool = True,
    *,
    flux_unit: Optional[str] = None,
) -> None:
    """Clean each chosen local grid and rebuild its lookup table, HDF5 bundle and flux cube.

    ``flux_unit`` (a ``spectra_cleaner.FLUX_UNITS`` name) is used for grids
    whose spectra do not declare their flux unit; without it sed-tools asks.
    Every grid that can be rebuilt is; then RuntimeError lists each grid that
    could not be, and why.
    """
    ensure_dir(base_dir)

    cand = rebuildable_grids(base_dir)
    if not cand:
        raise FileNotFoundError(f"No local spectral grids found under {base_dir}")

    selected = models if models is not None else _ask_rebuild_selection(cand)
    if not selected:
        print("Nothing selected.")
        return

    failures = []
    for model_name in selected:
        print("\n" + "=" * 64)
        print(f"[rebuild] {model_name}")
        if model_name not in cand:
            print(f"[rebuild] {model_name} is not a local spectral grid under {base_dir}")
            failures.append(f"{model_name}: not a local spectral grid under {base_dir}")
            continue
        try:
            problems = _rebuild_grid(os.path.join(base_dir, model_name), model_name,
                                     rebuild_h5, rebuild_flux_cube, flux_unit)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"[rebuild] {model_name} failed: {exc}")
            problems = [f"{type(exc).__name__}: {exc}"]
        failures.extend(f"{model_name}: {problem}" for problem in problems)

    if failures:
        raise RuntimeError(f"Spectral grids not rebuilt ({len(failures)}): " + "; ".join(failures))
    print("\nRebuild complete.")


def _ask_rebuild_selection(cand: List[str]) -> List[str]:
    """Ask which grids to rebuild ('all', or numbers and ranges like 1,3-6) until the answer parses."""
    print("\nLocal spectral grids available for rebuild:")
    print("-" * 64)
    for i, m in enumerate(cand, 1):
        print(f"{i:3d}. {m}")
    print("\nSelect indices (comma / ranges like 3-6) or 'all':")
    while True:
        raw = input("> ").strip().lower()
        if raw == "all":
            return cand
        try:
            return [cand[i] for i in parse_multi_selection(raw, len(cand))]
        except ValueError as exc:
            print(f"  {exc}")


def _ask_flux_unit(grid: str, txt_files: List[str], flux_unit: Optional[str] = None) -> Optional[str]:
    """``flux_unit`` if given, else the unit typed in a terminal for undeclared spectra (Enter never takes
    the guess; 'n' raises ValueError), else None, leaving cleaning to stop with its best guess."""
    if flux_unit is not None or not sys.stdin.isatty():
        return flux_unit
    guess = undeclared_flux_unit(txt_files)
    if guess is None:
        return None
    print(f"\nFlux unit of {grid} is not declared. Accepted flux units:")
    for name, (_, _, description) in FLUX_UNITS.items():
        print(f"  {name:11s} {description}")
    print(wrap(f"Best guess: {guess}, {FLUX_UNITS[guess][2]}; {FLUX_GUESS_WHY}.", indent="  ", hanging="    "))
    question = f"Flux unit of {grid} (type its name, e.g. {guess}), or n to leave it uncleaned: "
    while True:
        answer = input(question).strip()
        if answer in FLUX_UNITS:
            return answer
        if answer.lower() in ("n", "no"):
            raise ValueError(f"No flux unit chosen for {grid}; it was not cleaned")
        if answer.lower() in ("", "y", "yes"):
            print(f"  Type the unit's name; the guess ({guess}) is not taken without it.")
        else:
            print(f"  {answer!r} is not one of: {', '.join(FLUX_UNITS)}")


def _txt_files(model_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(model_dir, "*.txt")))


def _clean_grid(name: str, model_dir: str, flux_unit: Optional[str]) -> Dict:
    """clean_model_dir on a grid's folder, with ``flux_unit`` or the unit asked for (_ask_flux_unit)."""
    flux_unit = _ask_flux_unit(name, _txt_files(model_dir), flux_unit)
    return clean_model_dir(model_dir, try_h5_recovery=True, rebuild_lookup=True, flux_unit=flux_unit)


def _ask_import_flux_unit(path: str, name: Optional[str], flux_unit: Optional[str] = None) -> Optional[str]:
    """_ask_flux_unit for the spectra SED.import_grid would import from path."""
    files, src_dir = _import_sources(Path(path).expanduser())
    return _ask_flux_unit(name or src_dir.name, [str(f) for f in files], flux_unit)


def _report_import_flux_unit(path: str, name: Optional[str], flux_unit: Optional[str]) -> None:
    """For ``import --dry-run``: whether the spectra declare their flux unit, and the guess when not."""
    files, src_dir = _import_sources(Path(path).expanduser())
    guess = undeclared_flux_unit([str(f) for f in files])
    grid = name or src_dir.name
    if guess is None:
        print(f"[import] {grid}: the spectra declare their flux unit"
              + (f"; --flux-unit {flux_unit} would not be used" if flux_unit else ""))
    elif flux_unit:
        print(f"[import] {grid}: flux unit not declared; --flux-unit {flux_unit} would be used "
              f"(best guess: {guess})")
    else:
        print(f"[import] {grid}: flux unit not declared; best guess: {guess}, {FLUX_UNITS[guess][2]} "
              f"({FLUX_GUESS_WHY}). Importing needs --flux-unit, or an answer in a terminal.")


def _rebuild_grid(model_dir: str, model_name: str, rebuild_h5: bool, rebuild_flux_cube: bool,
                  flux_unit: Optional[str] = None) -> List[str]:
    """Clean one local grid and rebuild its products; returns what was not done (empty when complete)."""
    summary = _clean_grid(model_name, model_dir, flux_unit)
    print(f"[clean] {model_name}: total={summary['total']}")

    if not _txt_files(model_dir):
        print(f"[rebuild] no spectra (.txt) present after cleaning; skipping {model_name}.")
        return ["no spectra (.txt) present after cleaning"]

    regenerate_lookup_table(model_dir)

    if rebuild_h5:
        out_h5 = os.path.join(model_dir, f"{model_name}.h5")
        build_h5_bundle_from_txt(model_dir, out_h5)

    if rebuild_flux_cube:
        _build_flux_cube(model_dir)
    return []


class _ModelOption:
    """Display wrapper for a discoverable spectral grid, used by run_spectra_flow."""
    def __init__(self, name: str, sources: List[str]) -> None:
        self.name = name
        self.sources = sources
        self.label = f"{name} {''.join(f'[{s}]' for s in sources)}"


def _print_cleaning_report(summary: Dict) -> None:
    """Print the unit-detection and per-file outcome report after clean_model_dir."""
    n_total    = summary['total']
    det_stats  = summary.get('detection_stats', {})
    cat_units  = summary.get('catalog_units', {}) or {}
    det_status = det_stats.get('status', '')
    sample_size = det_stats.get('sample_size', '?')
    detected_wl   = cat_units.get('wavelength', 'unknown')
    detected_flux = cat_units.get('flux', 'unknown')

    if det_status == 'already_standardized':
        print(f" Method    : Sampled {sample_size}/{n_total} files")
        print(f" Result    : All samples already standardized")
        print(f" Wavelength: angstrom (Å)")
        print(f" Flux      : F_lambda (erg/s/cm²/Å)")
    elif cat_units:
        print(f" Method    : Spectral grid consensus ({sample_size}/{n_total} files sampled)")
        print(f" Wavelength: {detected_wl} ({det_stats.get('wavelength_agreement', '?')} agreement)")
        print(f" Flux      : {detected_flux} ({det_stats.get('flux_agreement', '?')} agreement)")
        print(f" Confidence: {cat_units.get('confidence', 'unknown')}")
    else:
        print(f" Status    : {det_status or 'No unit detection performed'}")
        print(f" Wavelength: {detected_wl}")
        print(f" Flux      : {detected_flux}")

    n_converted, n_tagged = len(summary['converted']), 0
    n_recovered = len(summary['recovered'])
    n_skipped   = len(summary['skipped_already'])

    # Files "converted" from units that were already Å and F_λ were only cleaned and tagged.
    if n_converted > 0 and detected_wl == 'angstrom' and detected_flux == 'flam':
        n_tagged, n_converted = n_converted, 0

    print(f" Total files: {n_total}")
    if n_converted > 0:
        print(f"Converted     : {n_converted} files")
        if detected_wl != 'angstrom':
            print(f"   λ: {detected_wl} → angstrom")
        if detected_flux != 'flam':
            print(f"   F: {detected_flux} → F_lambda")
    if n_tagged    > 0: print(f"Cleaned       : {n_tagged} files  (units already Å + F_λ)")
    if n_recovered > 0: print(f"Recovered     : {n_recovered} files  (wavelengths from HDF5)")
    if n_skipped   > 0: print(f"Already done  : {n_skipped} files  (units_standardized header)")

    # clean_model_dir raises instead of returning when any file is left invalid, index-only or failed.
    print()
    print(f"All {n_total} files ready for use")


def run_spectra_flow(
    source: str,
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models: Optional[List[str]] = None,
    workers: int = 5,
    force_bundle_h5: bool = True,
    build_flux_cube: bool = True,
    *,
    flux_unit: Optional[str] = None,
    teff_range: Optional[Tuple[float, float]] = None,
    logg_range: Optional[Tuple[float, float]] = None,
    meta_range: Optional[Tuple[float, float]] = None,
    wl_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Discover spectral grids, then download, clean and build each chosen one.

    ``flux_unit`` is used for grids whose spectra do not declare their flux
    unit (see run_rebuild_flow).  The ``*_range`` cuts, each ``(min, max)``,
    are passed to every source (SVO refuses them); without any, an NJM
    download asks for cuts in a terminal and takes the full grid otherwise.
    A grid that cannot be completed is reported and the rest are processed;
    then RuntimeError lists every grid that was not completed.
    """
    ranges = {"teff_range": teff_range, "logg_range": logg_range, "meta_range": meta_range, "wl_range": wl_range}
    ensure_dir(base_dir)

    src_list = _spectra_sources(source)
    grabs = _spectra_grabbers(src_list, base_dir, workers)
    model_sources = _discover_spectral_grids(src_list, grabs)
    if not model_sources:
        unreachable = "; njm: mirror unreachable" if "njm" in src_list and "njm" not in grabs else ""
        raise RuntimeError("No spectral grids discovered" + unreachable)

    all_models = [_ModelOption(n, srcs) for n, srcs in sorted(model_sources.items())]
    if models is not None:
        chosen, failures = _requested_spectral_grids(models, all_models)
    elif not sys.stdin.isatty():
        raise ValueError(f"No terminal to choose from the {len(all_models)} spectral grids found; "
                         "name them with --grids NAME ... (or --grids all)")
    else:
        idxs = prompt_choice(all_models, label="Spectral grids", allow_back=True, multi=True)
        if idxs is None or idxs == -1:
            return
        chosen, failures = [(all_models[i].sources, all_models[i].name) for i in idxs], []

    for sources, name in chosen:
        try:
            problems = _download_and_build_grid(name, sources, grabs, base_dir, force_bundle_h5, build_flux_cube,
                                                flux_unit=flux_unit, ranges=ranges)
        except (ValueError, RequestException, OSError, RuntimeError) as exc:
            print(f"[{name}] failed: {exc}")
            problems = [f"{type(exc).__name__}: {exc}"]
        failures.extend(f"{name}: {problem}" for problem in problems)

    print("\n" + "─" * 64)
    if failures:
        raise RuntimeError(f"Spectral grids not completed ({len(failures)}): " + "; ".join(failures))
    print("Done.")


def _spectra_sources(source: str) -> List[str]:
    """Source names from --source: one name, a comma-separated list, 'both' or 'all'."""
    if source.lower() == "all":
        return ["njm", "svo", "msg", "mast"]
    if source.lower() == "both":
        return ["njm", "svo", "msg"]
    if "," in source:
        return [s.strip().lower() for s in source.split(",")]
    return [source.lower()]


def _spectra_grabbers(src_list: List[str], base_dir: str, workers: int) -> Dict:
    """A grabber per requested source; NJM is left out when the mirror is unavailable (the grabber warns)."""
    grabs: Dict = {}
    for src in [s for s in ("njm", "svo", "msg", "mast") if s in src_list]:
        grabber = spectra_grabber(src, base_dir, max_workers=workers)
        if src != "njm" or grabber.is_available():
            grabs[src] = grabber
    return grabs


def _discover_spectral_grids(src_list: List[str], grabs: Dict) -> Dict[str, List[str]]:
    """Each grid name with the sources listing it, in --source order (a source that cannot be asked lists none)."""
    model_sources: Dict[str, List[str]] = {}
    for s in src_list:
        if s in grabs:
            for model_name in grabs[s].discover_models():
                model_sources.setdefault(model_name, []).append(s)
    return model_sources


def _requested_spectral_grids(models: List[str], all_models: List[_ModelOption]):
    """``(sources, name)`` for each name in ``models`` ('all', NAME or SOURCE:NAME), and the names not found."""
    if len(models) == 1 and models[0].lower() == "all":
        return [(opt.sources, opt.name) for opt in all_models], []
    chosen, not_found = [], []
    for m in models:
        if ":" in m:
            s, n = m.split(":", 1)
            chosen.append(([s.strip().lower()], n.strip()))
            continue
        matching = [opt for opt in all_models if opt.name == m.strip()]
        if matching:
            chosen.append((matching[0].sources, matching[0].name))
        else:
            print(f"[skip] Spectral grid '{m.strip()}' not found")
            not_found.append(f"{m.strip()}: not listed by any source")
    return chosen, not_found


def _download_from_first_source(name: str, sources: List[str], grabs: Dict, model_dir: str,
                                ranges: Optional[Dict] = None):
    """``(source, metadata)`` of the first source that downloads ``name`` (pipeline.download_from_first_source).

    A source without a grabber is listed as not available; RuntimeError gives every source's error.
    """
    tried = [f"{src}: not available" for src in sources if src not in grabs]
    metadata = {}

    def attempt(src: str) -> Optional[str]:
        meta, error = _download_from(name, src, grabs[src], model_dir, ranges)
        metadata[src] = meta
        return error

    def warn(src: str, error: str) -> None:
        logger.warning("  [%s] %s — trying next source...", src, error)

    src = download_from_first_source([s for s in sources if s in grabs], attempt, model_dir, tried, warn)
    if src is None:
        raise RuntimeError(f"no source could supply it. Tried: {'; '.join(tried)}")
    return src, metadata[src]


def _download_from(name: str, src: str, g, model_dir: str, ranges: Optional[Dict] = None):
    """Download ``name`` from one source: ``(metadata, None)``, or ``(metadata, what went wrong)``."""
    meta = None
    try:
        meta = g.get_model_metadata(name)  # raises when the source has no such grid
        print(f"  Using source: {src}")
        cuts = _download_cuts(name, src, g, meta, ranges or {})
        before = _directory_state(model_dir)
        n_written = g.download_model_spectra(name, meta, **cuts)
    except (ValueError, RequestException, OSError, RuntimeError) as exc:
        return meta, str(exc)
    print(f"{_download_count(n_written, before, _directory_state(model_dir))} → {model_dir}")
    if not n_written and not _is_preprocessed(meta):
        return meta, "no spectra downloaded"
    return meta, None


def _download_cuts(name: str, src: str, g, meta, ranges: Dict) -> Dict:
    """The ``*_range`` cuts for one download: those given, else asked for an NJM grid in a terminal, else none."""
    if any(value is not None for value in ranges.values()):
        return dict(ranges)
    if src != "njm":
        return {}
    if not sys.stdin.isatty():
        print("  No terminal to ask for axis cuts: downloading the full grid "
              "(--teff, --logg, --meta and --wavelength cut it)")
        return {}
    return prompt_njm_axis_cuts(name, g, model_url=meta.get("model_url"))


def _is_preprocessed(meta) -> bool:
    return isinstance(meta, dict) and meta.get("pre_processed", False)


def _download_and_build_grid(name: str, sources: List[str], grabs: Dict, base_dir: str,
                             force_bundle_h5: bool, build_flux_cube: bool, *,
                             flux_unit: Optional[str] = None, ranges: Optional[Dict] = None) -> List[str]:
    """Download one grid from its first source that has it, clean it and build its products; returns what
    was not done."""
    print("\n" + "=" * 64)
    print(f"{''.join(f'[{s}]' for s in sources)} {name}")
    print("=" * 64)

    model_dir = os.path.join(base_dir, name)
    ensure_dir(model_dir)

    src, meta = _download_from_first_source(name, sources, grabs, model_dir, ranges)

    if _is_preprocessed(meta):
        print("Pre-processed data (skipping cleaning)")
        essential = ["flux_cube.bin", "lookup_table.csv"]
        missing = [f for f in essential if not os.path.exists(os.path.join(model_dir, f))]
        print(f"Missing: {', '.join(missing)}" if missing else "All essential files present")
        return [f"missing {', '.join(missing)}"] if missing else []

    _print_cleaning_report(_clean_grid(name, model_dir, flux_unit))

    if not _txt_files(model_dir):
        print("\n  No spectra remaining after cleaning")
        return ["no spectra remaining after cleaning"]

    return _build_data_products(name, model_dir, src, force_bundle_h5, build_flux_cube)


def run_ml_generator_flow(
    base_dir: str = STELLAR_DIR_DEFAULT,
    models_dir: str = "models"
) -> Optional[int]:
    """Run ML SED Generator interactive workflow; 1 when the chosen action's packages are missing."""
    from ..ml.generator import run_interactive_workflow
    return run_interactive_workflow(base_dir=base_dir, models_dir=models_dir)


def run_grid_densifier_flow(base_dir: str = STELLAR_DIR_DEFAULT) -> Optional[int]:
    """Run Grid Densifier interactive workflow; 1 when an ML model is given and torch is missing."""
    from ..cube.densify import run_interactive_workflow
    return run_interactive_workflow(base_dir=base_dir)


def run_mesa_prepare_flow(
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    model: Optional[str] = None,
    output: Optional[str] = None,
) -> None:
    """Export one extra-axis variant of a spectral grid as a MESA-ready folder, asking for what is not given."""
    from ..cube.mesa_export import run_interactive
    run_interactive(base_dir=base_dir, model=model, output=output)


def run_combine_flow(
    base_dir: str = STELLAR_DIR_DEFAULT,
    output_name: str = "combined_models",
    interactive: bool = True,
    *,
    models: Optional[List[str]] = None,
) -> None:
    """Combine stellar spectral grids into one omni grid, with one flux cube spanning all their parameters.

    Args:
        base_dir: Base directory containing stellar_models/ subdirectories
        output_name: Name for the combined spectral grid directory
        interactive: If True, prompt user to select spectral grids
        models: Names of the grids to combine, instead of choosing them.
            Without it and without prompting, every local grid except the
            output of an earlier combine is used.

    A grid is never combined into itself: ValueError when ``output_name`` is
    one of the inputs.
    """
    from ..cube.combine import _grids_with_lookup, find_stellar_models, select_models_interactive

    ensure_dir(base_dir)

    # The menu offers no earlier combine outputs; --grids may name one, and
    # --non-interactive says which it leaves out.
    model_dirs = find_stellar_models(base_dir) if interactive and models is None else _grids_with_lookup(base_dir)
    if not model_dirs:
        raise FileNotFoundError(f"No spectral grids found in {base_dir}")

    if models is not None:
        selected_models = _named_grids(models, model_dirs, base_dir)
    elif interactive:
        selected_models = select_models_interactive(model_dirs)
    else:
        selected_models = _grids_to_combine_by_default(model_dirs, output_name)
    if not selected_models:
        print("No spectral grids selected.")
        return

    output_name = _combined_grid_name(output_name, [name for name, _ in selected_models], interactive)
    print("\nSpectral grid will be saved as:", output_name, "\n")

    print(f"\nSelected {len(selected_models)} spectral grids to combine:")
    for name, _ in selected_models:
        print(f"- {name}")

    output_dir = os.path.join(base_dir, output_name)
    _build_combined_grid(selected_models, output_dir)

    print(f"\nSuccessfully combined {len(selected_models)} stellar spectral grids!")
    print(f"Output saved to: {output_dir}")
    print("You can now use this combined spectral grid in MESA by setting:")
    print(f"stellar_atm = '{output_dir}/'")


def _combined_grid_name(output_name: str, inputs: List[str], interactive: bool) -> str:
    """The output grid's name, asked for when interactive; never one of the ``inputs``."""
    if interactive:
        output_name = input(f"What should the combined spectral grid be called? [{output_name}] ").strip() or output_name
        while output_name in inputs:
            output_name = (input(f"{output_name} is one of the grids being combined; choose another name: ").strip()
                           or output_name)
    elif output_name in inputs:
        raise ValueError(f"The output grid {output_name} is also one of the grids to combine; "
                         "choose another --output or leave it out of --grids")
    return output_name


def _named_grids(models: List[str], model_dirs: List[Tuple[str, str]], base_dir: str) -> List[Tuple[str, str]]:
    """The ``(name, path)`` of each grid in ``models``, in that order; ValueError names any not found."""
    found = dict(model_dirs)
    missing = [m for m in models if m not in found]
    if missing:
        raise ValueError(f"No spectral grid with a lookup table named {', '.join(missing)} in {base_dir}; "
                         f"available: {', '.join(sorted(found))}")
    return [(m, found[m]) for m in models]


def _grids_to_combine_by_default(model_dirs: List[Tuple[str, str]], output_name: str) -> List[Tuple[str, str]]:
    """Every grid except earlier combined outputs and ``output_name``; says which were left out."""
    from ..cube.combine import _is_combined_output

    keep = []
    for name, path in model_dirs:
        if name == output_name:
            print(f"Leaving out {name}: it is the output grid")
        elif _is_combined_output(name, os.path.join(path, "lookup_table.csv")):
            print(f"Leaving out {name}: an earlier combined grid (name it with --grids to include it)")
        else:
            keep.append((name, path))
    return keep


def run_ml_completer_flow(
    base_dir: str = str(STELLAR_DIR_DEFAULT),
    models_dir: str = "models"
) -> Optional[int]:
    """Run the ML SED Completer interactive workflow; 1, saying what to install, when the chosen action's
    packages are missing."""
    from ..ml.completer import run_interactive_workflow
    return run_interactive_workflow(base_dir, models_dir)


def run_coverage_flow(base_dir: str = str(STELLAR_DIR_DEFAULT)) -> None:
    """Interactive parameter-space coverage report for a local grid."""
    cands = local_grids(base_dir)
    if not cands:
        print(f"No local grids found under {base_dir}")
        return
    idx = prompt_choice(cands, label="Local grids", allow_back=True)
    if idx is None or idx == -1:
        return
    SED.coverage(cands[idx], model_root=base_dir, plot=True)


def run_import_flow(base_dir: str = str(STELLAR_DIR_DEFAULT)) -> None:
    """Interactive ingest of a local grid into the pipeline."""
    path = input("Path to your grid directory (or a .txt file): ").strip()
    if not path:
        print("No path given.")
        return
    name = input("Spectral grid name [blank = source folder name]: ").strip() or None
    mv = input("Move files instead of copying? [y/N] ").strip().lower().startswith("y")
    flux_unit = _ask_import_flux_unit(path, name)
    SED.import_grid(path, name=name, model_root=base_dir, move=mv, flux_unit=flux_unit)
