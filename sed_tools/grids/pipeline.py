"""A spectral grid's folder from download to products, shared by the API (SED.fetch, import_grid, combine)
and the sed-tools grid flows: the grabber for each source, the first-source download rule, the HDF5
bundle, lookup table and flux cube builds, the import helpers and the local-grid scanners.
"""

import math
import os
import shlex
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..cube.check import _bundle_shortfall, _flux_cube_file, _flux_cube_shortfall, _variant_paths
from ..cube.model import STELLAR_DIR_DEFAULT
from ..ui_utils import wrap
from . import mast, msg, njm, svo
from .header_parser import parse_header
from .lookup import regenerate_lookup_table
from .spectrum_io import _bundle_path, build_h5_bundle


def spectra_grabber(src: str, base_dir: str, **kwargs):
    """A new grabber for ``src``: njm, svo, msg or mast.

    The class is looked up in its module at call time, so a patched grabber class is used.
    """
    grabbers = {"njm": njm.NJMSpectraGrabber, "svo": svo.SVOSpectraGrabber,
                "msg": msg.MSGSpectraGrabber, "mast": mast.MASTSpectraGrabber}
    return grabbers[src](base_dir=base_dir, **kwargs)


def _directory_state(directory: Path) -> Dict[str, int]:
    """Name and modification time of each entry in ``directory`` ({} if it does not exist)."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}


def download_from_first_source(sources: List[str], attempt: Callable[[str], Optional[str]], model_dir,
                               tried: List[str], warn: Callable[[str, str], None]) -> Optional[str]:
    """The first of ``sources`` whose ``attempt(src)`` returns None (no error); None when all fail.

    Each failure is added to ``tried`` and passed to ``warn(src, error)`` before the next source is
    tried, unless the failed attempt changed ``model_dir``: the next source's spectra would then be
    mixed with its files, so RuntimeError is raised instead.
    """
    for src in sources:
        before = _directory_state(model_dir)
        error = attempt(src)
        if error is None:
            return src
        tried.append(f"{src}: {error}")
        if _directory_state(model_dir) != before:
            raise RuntimeError(f"{src} failed after changing {model_dir}, so another source's spectra would be "
                               f"mixed with its files; no further source tried. Tried: {'; '.join(tried)}")
        warn(src, error)
    return None


def _download_count(n_spectra: int, before: Dict[str, int], after: Dict[str, int]) -> str:
    """``Downloaded N spectra``, or ``K already present, N downloaded`` when the grabber's count
    (spectra downloaded or found in place) includes spectra it did not write."""
    written = sum(1 for name, mtime in after.items() if name.lower().endswith(".txt") and before.get(name) != mtime)
    present = n_spectra - written
    return f"{present} already present, {written} downloaded" if present > 0 else f"Downloaded {n_spectra} spectra"


def _parameter_cuts(teff_min, teff_max, logg_min, logg_max, metallicity_min,
                    metallicity_max) -> Dict[str, Tuple[float, float]]:
    """The grabbers' ``teff_range``, ``logg_range`` and ``meta_range`` for the bounds given.

    A bound of 0 is a bound; an axis bounded on one side only is open on the other.
    """
    cuts = {}
    for name, low, high in (("teff_range", teff_min, teff_max), ("logg_range", logg_min, logg_max),
                            ("meta_range", metallicity_min, metallicity_max)):
        if low is not None or high is not None:
            cuts[name] = (-math.inf if low is None else float(low), math.inf if high is None else float(high))
    return cuts


def build_h5_bundle_from_txt(model_dir: str, out_h5: str) -> None:
    """Create an HDF5 file bundling all .txt spectra."""
    if not build_h5_bundle(model_dir, out_h5):
        print(f"[H5 bundle] No .txt spectra found in {model_dir}; skipping.")
        return
    print(f"[H5 bundle] Wrote {out_h5}")


def _build_flux_cube(model_dir: Path) -> Path:
    """Run precompute_flux_cube on ``model_dir``; returns ``flux_cube.bin``, or ``variants_index.csv``
    for a grid split into one cube per variant. RuntimeError when this call wrote neither (it said why)."""
    from ..cube.build import precompute_flux_cube  # at call time, so a patched builder is the one run

    model_dir = Path(model_dir)
    outputs = (model_dir / "flux_cube.bin", model_dir / "variants_index.csv")
    before = {path: path.stat().st_mtime_ns for path in outputs if path.exists()}
    precompute_flux_cube(str(model_dir), str(outputs[0]))
    for path in outputs:
        if path.exists() and path.stat().st_mtime_ns != before.get(path):
            return path
    raise RuntimeError(f"No flux cube was built in {model_dir}; see the flux cube messages above")


def _build_data_products(name: str, model_dir: str, src: str, force_bundle_h5: bool,
                         build_flux_cube: bool) -> List[str]:
    """Build the HDF5 bundle, lookup table and flux cube of a downloaded grid; returns what was not done.

    A bundle or cube lacking spectra beside it is rebuilt; with ``build_flux_cube`` False such a cube is a problem.
    """
    out_h5 = str(_bundle_path(model_dir, name))
    fresh = force_bundle_h5 or not os.path.exists(out_h5)
    stale = None if fresh else _bundle_shortfall(Path(out_h5), Path(model_dir))
    if fresh or stale:
        if stale:
            print(wrap(f"{os.path.basename(out_h5)} {stale}; rebuilding it",
                       indent="HDF5 bundle   : ", hanging=" " * 16))
        build_h5_bundle_from_txt(model_dir, out_h5)
        print(f"HDF5 bundle   : {os.path.basename(out_h5)}")
    else:
        print(f"HDF5 bundle   : {os.path.basename(out_h5)} (exists)")

    regenerate_lookup_table(model_dir)
    print(f"Lookup table  : lookup_table.csv")

    if not build_flux_cube:
        shortfall = _flux_cube_shortfall(Path(model_dir))
        if shortfall:
            return [f"{_flux_cube_file(Path(model_dir))} {shortfall}; it was not rebuilt (--no-cube). "
                    f"Rebuild it with: {_rebuild_command(name, model_dir)}"]
        return []

    written = _build_flux_cube(model_dir)
    if written.name == "variants_index.csv":
        _print_subgrids(model_dir, written)
    else:
        print("Flux cube     : flux_cube.bin")
    return []


def _print_subgrids(model_dir: str, variants_index: Path) -> None:
    """Say which folders MESA can read when a grid is split: one per variant, each with its flux cube."""
    subgrids = _variant_paths(variants_index)
    print(f"Flux cubes    : {len(subgrids)} MESA-ready subgrid(s) created")
    for subgrid in subgrids:
        print(f"  -> {os.path.join(model_dir, subgrid)}/")
    print("\n  Point MESA Colors to one of the above subdirectories.")
    print(f"  See {variants_index} for the full inventory.")


def _rebuild_command(name: str, model_dir: str) -> str:
    """``sed-tools rebuild --grids NAME``, with ``--base`` when the grid is not under the default directory."""
    command = f"sed-tools rebuild --grids {shlex.quote(name)}"
    base = os.path.dirname(os.path.abspath(model_dir))
    if base != os.path.abspath(str(STELLAR_DIR_DEFAULT)):
        command += f" --base {shlex.quote(base)}"
    return command


def _build_combined_grid(selected_models: List[Tuple[str, str]], output_dir: str, *,
                         visualization: bool = True) -> None:
    """Merge the ``(name, path)`` grids onto common parameter and wavelength grids and save to ``output_dir``
    (:func:`~sed_tools.cube.combine.write_combined_grid`). ValueError, before anything is read or written, when
    ``output_dir`` is one of the grids: the API and the CLI share this rule."""
    from ..cube.combine import load_model_data, write_combined_grid

    if os.path.realpath(output_dir) in {os.path.realpath(path) for _, path in selected_models}:
        raise ValueError(f"The output grid {os.path.basename(os.path.normpath(output_dir))} is also one of the "
                         "grids to combine; nothing was written")
    print("\nLoading spectral grid data...")
    all_models_data = []
    for name, path in selected_models:
        print(f"  Loading {name}...")
        all_models_data.append(load_model_data(path))

    print("\nBuilding combined flux cube...")
    write_combined_grid(all_models_data, output_dir, visualization=visualization)


def _import_sources(src: Path) -> Tuple[List[Path], Path]:
    """The .txt spectra to import from a directory or a single .txt file, and their directory."""
    if src.is_dir():
        files, src_dir = sorted(src.glob("*.[tT][xX][tT]")), src
    else:
        files, src_dir = ([src] if src.suffix.lower() == ".txt" else []), src.parent

    if not files:
        raise RuntimeError(f"No .txt spectra found in {src}")
    return files, src_dir


def _report_import_headers(files: List[Path], src_dir: Path) -> List[str]:
    """Print how many files carry Teff, logg and [M/H]; return the names of those that do not."""
    missing: List[str] = []
    for f in files:
        h = parse_header(str(f))
        if any(math.isnan(h[k]) for k in ("teff", "logg", "metallicity")):
            missing.append(f.name)

    print(f"[import] {len(files)} .txt files in {src_dir}")
    print(f"[import] parseable Teff+logg+[M/H] headers: {len(files) - len(missing)}/{len(files)}")
    if missing:
        print(f"[import] missing one or more parameters: {len(missing)}")
        for b in missing[:10]:
            print(f"           {b}")
        if len(missing) > 10:
            print(f"           ... and {len(missing) - 10} more")
        print("[import] If a key is merely unrecognised, add it to ALIASES in header_parser.py.")
    return missing


def _grid_folders(base_dir, wanted: Callable[[str, List[str]], bool]) -> List[str]:
    """Sorted names of the folders under ``base_dir`` for which ``wanted(path, lower-cased file names)``."""
    base_dir = str(base_dir)
    if not os.path.isdir(base_dir):
        return []
    names = []
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and wanted(path, [fn.lower() for fn in os.listdir(path)]):
            names.append(name)
    return names


def rebuildable_grids(base_dir) -> List[str]:
    """Folders under ``base_dir`` holding .txt spectra or an .h5 file."""
    return _grid_folders(base_dir, lambda path, files: any(fn.endswith((".txt", ".h5")) for fn in files))


def local_grids(base_dir) -> List[str]:
    """Local spectral grid folders that have a lookup table or at least one .txt."""
    return _grid_folders(base_dir, lambda path, files: os.path.exists(os.path.join(path, "lookup_table.csv"))
                         or any(fn.endswith(".txt") for fn in files))
