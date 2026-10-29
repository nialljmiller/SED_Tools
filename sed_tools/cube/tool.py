"""FluxCube, a flux_cube.bin in memory with cubic Hermite interpolation in Teff, log g and [M/H] (the format is in
cube/fileio.py), and the flux cube tool (python -m sed_tools.flux_cube_tool --flux-cube PATH): it prompts for
what is missing, evaluates the SED, its bolometric magnitude and its AB or Vega magnitude in every filter it finds
under filters/ (or --filters), and saves a plot to plots/. It re-exports the photometry names of photometry.py.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .fileio import read_axes
from ..filters.curve_metadata import ENERGY, PHOTON, normalize_detector_type, read_filter_header  # noqa: F401
# Re-exported: these names are imported from sed_tools.flux_cube_tool.
from ..photometry import (  # noqa: F401
    AB_ZERO_FLUX, FILTER_EXTENSIONS, PC_TO_M, RSUN_TO_M, SPEED_OF_LIGHT_ANG_PER_S, UNIT_TO_ANGSTROM, VEGA_ZP_KEYS,
    FilterCurve, Spectrum, _band_magnitude, _standardize_file_wavelengths, _warn_outside_sed,
    band_average_flux_lambda, band_average_flux_lambda_from_arrays, band_average_flux_nu, bolometric_magnitude,
    filter_weight, get_metadata_float, infer_unit_from_values, integrate_flux, load_filter_curve, load_spectrum,
    load_two_column_file, normalize_metadata_key, parse_metadata_value, standardize_wavelength_array)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - only plot_sed needs matplotlib
    plt = None

logger = logging.getLogger("sed_tools.flux_cube_tool")

@dataclass
class FluxCube:
    """In-memory representation of a pre-computed flux cube."""

    teff_grid: np.ndarray
    logg_grid: np.ndarray
    meta_grid: np.ndarray
    wavelengths: np.ndarray
    flux: np.ndarray  # shape (nt, nl, nm, nw)

    @classmethod
    def from_file(cls, path: str) -> "FluxCube":
        """Load a :class:`FluxCube` from a binary file."""
        with open(path, "rb") as fh:
            header = fh.read(16)
            if len(header) != 16:
                raise ValueError("Flux cube header truncated")
            nt, nl, nm, nw = struct.unpack("4i", header)
            teff, logg, meta, wavelengths = read_axes(fh, nt, nl, nm, nw)
            flux_expected = nt * nl * nm * nw
            flux_flat = np.fromfile(fh, dtype=np.float64, count=flux_expected)
            leftover = fh.read(1) if flux_flat.size == flux_expected else b""

        if flux_flat.size != flux_expected:
            raise ValueError(
                f"Flux cube body truncated: expected {flux_expected} values,"
                f" found {flux_flat.size}"
            )

        if leftover:
            print("Warning: trailing data detected in flux cube; ignoring extra bytes.")

        # The body is stored in Fortran order (Teff varies fastest) for MESA,
        # so read it as (nw, nm, nl, nt) and transpose back.
        flux = flux_flat.reshape((nw, nm, nl, nt)).transpose(3, 2, 1, 0)
        return cls(teff, logg, meta, wavelengths, flux)

    def _hermite_interp_axis(
        self, values: np.ndarray, grid: np.ndarray, x: float, axis: int
    ) -> np.ndarray:
        """Cubic Hermite interpolation along one axis."""
        if values.shape[axis] < 2 or len(grid) < 2:
            # Degenerate axis: return the lone slice without interpolation.
            return np.take(values, 0, axis=axis)

        if not (grid[0] <= x <= grid[-1]):
            raise ValueError(
                f"Requested value {x} is outside interpolation range"
                f" [{grid[0]}, {grid[-1]}]"
            )

        idx = int(np.searchsorted(grid, x) - 1)
        idx = max(0, min(idx, len(grid) - 2))

        x0 = grid[idx]
        h = grid[idx + 1] - x0
        t = (x - x0) / h if h != 0 else 0.0

        f0 = np.take(values, idx, axis=axis)
        f1 = np.take(values, idx + 1, axis=axis)

        m0 = self._axis_derivative(values, grid, idx, axis)
        m1 = self._axis_derivative(values, grid, idx + 1, axis)

        h00 = (2 * t**3) - (3 * t**2) + 1
        h10 = (t**3) - (2 * t**2) + t
        h01 = (-2 * t**3) + (3 * t**2)
        h11 = (t**3) - (t**2)

        return h00 * f0 + h10 * h * m0 + h01 * f1 + h11 * h * m1

    def _axis_derivative(
        self, values: np.ndarray, grid: np.ndarray, idx: int, axis: int
    ) -> np.ndarray:
        n = len(grid)
        if idx <= 0:
            lo, hi = 0, 1
        elif idx >= n - 1:
            lo, hi = n - 2, n - 1
        else:
            lo, hi = idx - 1, idx + 1
        return (np.take(values, hi, axis=axis) - np.take(values, lo, axis=axis)) / (grid[hi] - grid[lo])

    def interpolate_spectrum(self, teff: float, logg: float, meta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(wavelength, flux)`` at the requested parameter point."""
        tmp = self._hermite_interp_axis(self.flux, self.teff_grid, teff, axis=0)
        tmp = self._hermite_interp_axis(tmp, self.logg_grid, logg, axis=0)
        flux = self._hermite_interp_axis(tmp, self.meta_grid, meta, axis=0)
        return self.wavelengths.copy(), flux


def prompt_scale_choice() -> bool:
    resp = input("\nScale flux for distance (apparent magnitudes)? [y/N]: ").strip().lower()
    return resp in ("y", "yes")

def prompt_distance_m() -> float:
    val = input("Enter distance (append unit 'pc' or 'm', e.g. '10 pc' or '3.1e17 m'): ").strip()
    parts = val.split()
    d = float(parts[0])
    unit = parts[1].lower() if len(parts) > 1 else "pc"
    return d * PC_TO_M if unit == "pc" else d

def prompt_radius_m() -> float:
    val = input("Enter stellar radius (append unit 'Rsun' or 'm', e.g. '1 Rsun' or '7e8 m'): ").strip()
    parts = val.split()
    r = float(parts[0])
    unit = parts[1].lower() if len(parts) > 1 else "Rsun"
    return r * RSUN_TO_M if unit in ("rsun", "r☉", "r_sun") else r


def plot_sed(
    wavelength: np.ndarray,
    flux: np.ndarray,
    filters: Dict[str, FilterCurve] | None = None,
    out_path: str | None = None,
    *,
    xlim: tuple[float, float] | None = None,   # manual override
    clip: str = "auto",                        # "auto" | "flux" | "filters" | "none"
    rel_cutoff: float = 1e-4,                  # keep λ where flux > rel_cutoff * peak
    margin_frac: float = 0.03,                 # pad the chosen window by this fraction
) -> None:
    """Plot the interpolated SED and optional filter curves, with sane x-limits."""
    if plt is None:
        raise ImportError("matplotlib is required for plotting SEDs")

    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, flux, label="Interpolated SED", color="C0")

    peak = np.nanmax(flux) if flux.size else 1.0

    if filters:
        for name, curve in filters.items():
            if curve.transmission.size == 0:
                continue
            scale = np.max(curve.transmission)
            scaled = curve.transmission / scale * peak if scale and scale > 0 else curve.transmission
            plt.plot(curve.wavelength, scaled, label=f"Filter {name}")

    if xlim is None and clip != "none":
        candidates: list[tuple[float, float]] = []

        if clip in ("auto", "flux") and flux.size:
            mask = np.isfinite(flux) & (flux > rel_cutoff * peak)
            if np.any(mask):
                wl_lo = np.nanmin(wavelength[mask])
                wl_hi = np.nanmax(wavelength[mask])
                candidates.append((wl_lo, wl_hi))

        if clip in ("auto", "filters") and filters:
            f_los = [np.min(fc.wavelength) for fc in filters.values() if fc.wavelength.size]
            f_his = [np.max(fc.wavelength) for fc in filters.values() if fc.wavelength.size]
            if f_los and f_his:
                candidates.append((min(f_los), max(f_his)))

        if candidates:
            lo = min(c[0] for c in candidates)
            hi = max(c[1] for c in candidates)
            span = hi - lo if hi > lo else 1.0
            lo -= span * margin_frac
            hi += span * margin_frac
            plt.xlim(lo, hi)

    if xlim is not None:
        plt.xlim(*xlim)

    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.title("Interpolated Spectral Energy Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if out_path:
        ensure_dir(os.path.dirname(out_path) or ".")
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close()


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_sed(path: str, wavelength: np.ndarray, flux: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.savetxt(path, np.column_stack((wavelength, flux)), header="wavelength flux")


def prompt_for_parameter(name: str, units: str, grid: np.ndarray, provided: float | None) -> float:
    """Return the requested parameter, prompting if ``provided`` is ``None``."""
    if provided is not None:
        return provided

    lower = float(np.min(grid))
    upper = float(np.max(grid))
    prompt = f"Enter target {name} in {units} (range: {lower:g}–{upper:g} {units}): "

    while True:
        try:
            value_str = input(prompt).strip()
        except EOFError:
            raise SystemExit(f"Missing {name}; aborting.")

        try:
            value = float(value_str)
        except ValueError:
            print(f"Could not parse '{value_str}'. Please enter a numeric value.")
            continue

        if not (lower <= value <= upper):
            print(f"Value {value} is outside the supported range {lower:g}–{upper:g} {units}.")
            continue

        return value


def infer_filter_directories(flux_cube_path: str) -> List[Path]:
    """Return likely filter directories based on the flux cube location."""
    cube_path = Path(flux_cube_path).expanduser().resolve()
    candidates = [parent / "filters" for parent in [cube_path.parent, *cube_path.parents]]
    return [candidate for candidate in candidates if candidate.is_dir()]


def _is_auxiliary(path: Path, root: Path) -> bool:
    """Components/products sub-directories hold derived curves, not published ones."""
    return any(part in ("components", "products") for part in path.relative_to(root).parts)


def _unique_resolved(paths: Iterable[Path]) -> List[Path]:
    """Resolved *paths* in order, without duplicates."""
    return list(dict.fromkeys(path.resolve() for path in paths))


def collect_filter_files(
    entries: Sequence[str] | None, flux_cube_path: str
) -> Tuple[List[Tuple[str, Path]], List[Path]]:
    """Discover filter files from explicit paths or inferred directories."""
    discovered: List[Path] = []
    search_roots: List[Path] = []

    def add_file(file_path: Path) -> None:
        if file_path.suffix.lower() in FILTER_EXTENSIONS and file_path.is_file():
            discovered.append(file_path)

    def add_directory(directory: Path) -> None:
        search_roots.append(directory)
        for file_path in sorted(directory.rglob("*")):
            if not _is_auxiliary(file_path, directory):
                add_file(file_path)

    if entries:
        for item in entries:
            path = Path(item).expanduser().resolve()
            if path.is_dir():
                add_directory(path)
            elif path.is_file():
                add_file(path)
            else:
                raise FileNotFoundError(f"Filter path '{item}' does not exist")
    else:
        for directory in infer_filter_directories(flux_cube_path):
            add_directory(directory)

    unique_files = list(dict.fromkeys(discovered))
    if not unique_files:
        return [], _unique_resolved(search_roots)

    if not search_roots:
        common_root = Path(os.path.commonpath([str(p.parent) for p in unique_files])).resolve()
        search_roots = [common_root]
    search_roots = _unique_resolved(search_roots)

    filter_entries: List[Tuple[str, Path]] = []
    existing_names: set[str] = set()
    name_counts: Dict[str, int] = {}

    for file_path in unique_files:
        # Name each filter by its path below the first search root holding it.
        name = file_path.stem
        for root in search_roots:
            if file_path.is_relative_to(root):
                name = file_path.relative_to(root).with_suffix("").as_posix()
                break

        base_name = name
        counter = name_counts.get(base_name, 0)
        while name in existing_names:
            counter += 1
            name = f"{base_name}_{counter}"
        name_counts[base_name] = counter

        filter_entries.append((name, file_path))
        existing_names.add(name)

    return filter_entries, search_roots


def extract_filter_system(filter_name: str) -> str:
    separator = "/" if "/" in filter_name else "\\"
    return filter_name.split(separator, 1)[0]


def prompt_for_filter_system(
    filter_entries: Sequence[Tuple[str, Path]], provided: str | None
) -> Tuple[str | None, Sequence[Tuple[str, Path]]]:
    systems: Dict[str, List[Tuple[str, Path]]] = {}
    for name, path in filter_entries:
        system = extract_filter_system(name)
        systems.setdefault(system, []).append((name, path))

    if provided:
        provided_key = provided.strip()
        if provided_key.lower() in {"all", "*"}:
            return None, filter_entries
        if provided_key in systems:
            return provided_key, systems[provided_key]
        raise SystemExit(
            f"Unknown filter system '{provided}'. Available systems: {', '.join(sorted(systems))}"
        )

    if len(systems) <= 1:
        return next(iter(systems), None), filter_entries

    print("\nDiscovered filter systems:")
    ordered = sorted((system, len(entries)) for system, entries in systems.items())
    for idx, (system, count) in enumerate(ordered, start=1):
        print(f"  {idx}) {system} ({count} filters)")
    print("  0) All systems")

    while True:
        response = input("Select a filter system by number (or '0' for all): ").strip()
        if response.lower() in {"", "0", "all", "a", "*"}:
            return None, filter_entries
        if response.isdigit():
            index = int(response)
            if 1 <= index <= len(ordered):
                system = ordered[index - 1][0]
                return system, systems[system]
        print("Please enter a valid selection from the list above.")


def prompt_for_photometric_system(provided: str | None) -> str:
    if provided:
        normalized = provided.strip().lower()
        if normalized in {"ab", "vega"}:
            return "AB" if normalized == "ab" else "Vega"
        raise SystemExit("Photometric system must be either 'AB' or 'Vega'.")

    answers = {"": "AB", "ab": "AB", "a": "AB", "vega": "Vega", "v": "Vega"}
    while True:
        response = input("Compute magnitudes in the AB or Vega system? [AB/Vega]: ").strip().lower()
        if response in answers:
            return answers[response]
        print("Please answer 'AB' or 'Vega'.")


def discover_vega_candidates(search_roots: Sequence[Path]) -> List[Path]:
    candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        root = root.resolve()
        for suffix in (".dat", ".txt", ".csv", ".sed"):
            try:
                for path in root.rglob(f"*vega*{suffix}"):
                    if path.is_file():
                        candidates.append(path)
            except OSError as exc:
                logger.warning(
                    "Could not finish searching %s for Vega spectra (%s); "
                    "spectra there may be missing from the candidates.", root, exc,
                )
    return _unique_resolved(candidates)


def resolve_vega_spectrum_path(
    provided: str | None, flux_cube_path: Path, hints: Iterable[str | os.PathLike[str]] = ()
) -> Path:
    hint_paths = [Path(hint).expanduser() for hint in hints]
    env_hint = os.environ.get("SED_VEGA_SPECTRUM")
    if env_hint:
        hint_paths.append(Path(env_hint).expanduser())

    if provided:
        candidate = Path(provided).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Vega spectrum '{provided}' does not exist")

    for hint in hint_paths:
        if hint.is_file() and "vega" in hint.name.lower():
            return hint.resolve()

    cube_dir = flux_cube_path.parent
    search_roots = [cube_dir, cube_dir.parent, Path.cwd(), Path.cwd() / "data"]
    for hint in hint_paths:
        if hint.is_dir():
            search_roots.append(hint)
        elif hint.is_file():
            search_roots.append(hint.parent)

    candidates = discover_vega_candidates(search_roots)
    if not candidates:
        raise FileNotFoundError("Could not locate a Vega reference spectrum.")
    if len(candidates) == 1:
        return candidates[0]

    print("\nDiscovered potential Vega spectra:")
    for idx, path in enumerate(candidates, start=1):
        print(f"  {idx}) {path}")

    while True:
        choice = input("Select Vega spectrum by number: ").strip()
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
        print("Please select one of the listed Vega spectra.")


def discover_flux_cube_files(hints: Sequence[str | os.PathLike[str]] | None = None) -> List[Path]:
    """Discover ``flux_cube.bin`` files from common locations and hints."""
    from .model import DATA_DIR_DEFAULT

    hints = list(hints or [])
    for variable in ("SED_FLUX_CUBE", "SED_FLUX_CUBE_DIR"):
        if os.environ.get(variable):
            hints.append(os.environ[variable])

    found: set[Path] = set()
    search_dirs: List[Path] = []
    for hint in hints:
        path = Path(hint).expanduser()
        if path.is_file():
            found.add(path.resolve())
        elif path.is_dir():
            search_dirs.append(path)

    this_dir = Path(__file__).resolve().parents[1]
    default_roots = [
        Path.cwd(),
        Path.cwd() / "data",
        Path.cwd() / "data" / "stellar_models",
        DATA_DIR_DEFAULT / "stellar_models",
        this_dir,
        this_dir / "data",
        this_dir / "data" / "stellar_models",
    ]
    for root in default_roots:
        if root not in search_dirs:
            search_dirs.append(root)

    for directory in search_dirs:
        directory = directory.expanduser().resolve()
        if not directory.exists():
            continue
        if directory.is_file():
            found.add(directory)
            continue
        try:
            for match in directory.rglob("flux_cube.bin"):
                if match.is_file():
                    found.add(match.resolve())
        except OSError as exc:
            logger.warning(
                "Could not finish searching %s for flux cubes (%s); "
                "cubes there may be missing from the candidates.", directory, exc,
            )
    return sorted(found)


def resolve_flux_cube_path(path: str) -> Path:
    """Return a concrete flux cube file path, from the file or a directory holding exactly one cube."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        matches = sorted(p for p in candidate.rglob("flux_cube.bin") if p.is_file())
        if not matches:
            raise FileNotFoundError(
                f"No flux_cube.bin found under directory '{candidate}'. "
                "Provide a directory containing the cube or the explicit file path."
            )
        if len(matches) > 1:
            formatted = "\n  ".join(str(m) for m in matches[:10])
            more = "\n  ..." if len(matches) > 10 else ""
            raise FileNotFoundError(
                "Multiple flux_cube.bin files found; please specify one explicitly:\n  "
                f"{formatted}{more}"
            )
        return matches[0]

    raise FileNotFoundError(f"Flux cube path '{path}' does not exist")


def prompt_for_flux_cube_path(hints: Sequence[str | os.PathLike[str]] | None = None) -> Path:
    """Interactively determine which flux cube file to use."""
    hints = list(hints or [])

    while True:
        candidates = discover_flux_cube_files(hints)
        if len(candidates) == 1:
            chosen = candidates[0]
            print(f"Using flux cube: {chosen}")
            return chosen
        if len(candidates) > 1:
            print("Discovered flux cube files:")
            for idx, candidate in enumerate(candidates, start=1):
                print(f"  {idx}. {candidate}")
            response = input(
                "Select a flux cube by number or enter a path (or 'q' to quit): "
            ).strip()
            if not response:
                continue
            if response.lower() in {"q", "quit", "exit"}:
                raise SystemExit("No flux cube selected; aborting.")
            if response.isdigit():
                index = int(response)
                if 1 <= index <= len(candidates):
                    return candidates[index - 1]
                print("Invalid selection. Please choose one of the listed numbers.")
                continue
            try:
                return resolve_flux_cube_path(response)
            except FileNotFoundError as exc:
                print(exc)
                continue

        response = input("Enter path to flux_cube.bin (or 'q' to quit): ").strip()
        if not response:
            print("Please provide a path to a flux cube file.")
            continue
        if response.lower() in {"q", "quit", "exit"}:
            raise SystemExit("No flux cube selected; aborting.")
        try:
            return resolve_flux_cube_path(response)
        except FileNotFoundError as exc:
            print(exc)
            hints.append(response)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and interpolate a flux cube")
    parser.add_argument("--flux-cube", help="Path to flux_cube.bin")
    parser.add_argument("--model-dir", help="Model directory containing a flux cube", default=None)
    parser.add_argument("--teff", type=float, help="Target effective temperature (K)")
    parser.add_argument("--logg", type=float, help="Target log g (dex)")
    parser.add_argument("--metallicity", type=float, help="Target metallicity [M/H] (dex)")
    parser.add_argument("--bolometric-reference-flux", type=float, default=1.0,
                        help="Reference flux for bolometric magnitude (default: 1)")
    parser.add_argument("--bolometric-reference-mag", type=float, default=0.0,
                        help="Reference magnitude for bolometric magnitude (default: 0)")
    parser.add_argument("--filters", nargs="*", help="Filter files or directories (all filters discovered are used)")
    parser.add_argument("--filter-system", help="Restrict photometry to a specific filter system")
    parser.add_argument("--photometric-system", choices=["AB", "Vega", "ab", "vega"],
                        help="Photometric system for synthetic magnitudes")
    parser.add_argument("--vega-sed", help="Path to a Vega reference spectrum (required for Vega magnitudes if no zeropoint metadata is provided)")
    parser.add_argument("--plot", help="Write a diagnostic plot to this path (defaults to plots/)")
    parser.add_argument("--save-sed", help="Write interpolated SED (wavelength, flux) to a text file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    hints: List[str | os.PathLike[str]] = [args.model_dir] if args.model_dir else []

    if args.flux_cube:
        flux_cube_path = resolve_flux_cube_path(args.flux_cube)
    else:
        flux_cube_path = prompt_for_flux_cube_path(hints)

    cube = FluxCube.from_file(str(flux_cube_path))

    teff = prompt_for_parameter("effective temperature", "K", cube.teff_grid, args.teff)
    logg = prompt_for_parameter("surface gravity (log g)", "dex", cube.logg_grid, args.logg)
    metallicity = prompt_for_parameter("metallicity [M/H]", "dex", cube.meta_grid, args.metallicity)

    wl, flux = cube.interpolate_spectrum(teff, logg, metallicity)

    if prompt_scale_choice():
        d_m = prompt_distance_m()
        r_m = prompt_radius_m()
        scale = (r_m / d_m) ** 2
        flux = flux * scale
        print(f"[scaling] Applied geometric dilution: (R/d)^2 = {scale:.3e}")

    fbol, mbol = bolometric_magnitude(
        wl, flux,
        reference_flux=args.bolometric_reference_flux,
        reference_magnitude=args.bolometric_reference_mag,
    )

    print(f"Interpolated SED at Teff={teff}, logg={logg}, [M/H]={metallicity}")
    print(f"Bolometric flux: {fbol:.6e}")
    if math.isfinite(mbol):
        print(f"Bolometric magnitude: {mbol:.4f}")
    else:
        print("Bolometric magnitude: undefined (non-positive flux or reference flux)")

    filters: Dict[str, FilterCurve] = {}
    failures: List[str] = []
    filter_entries, filter_roots = collect_filter_files(args.filters, str(flux_cube_path))
    if filter_entries:
        filters, failures = _print_filter_photometry(
            args, flux_cube_path, filter_entries, filter_roots, wl, flux
        )
    else:
        print("\nNo filter files found; skipping filter photometry.")

    if args.save_sed:
        save_sed(args.save_sed, wl, flux)
        print(f"Saved SED to {args.save_sed}")

    plot_path = args.plot or os.path.join("plots", f"sed_T{teff:.0f}_g{logg:.2f}_M{metallicity:.2f}.png")

    plot_sed(wl, flux, filters, out_path=plot_path)
    print(f"Saved plot to {plot_path}")

    if failures:
        print(f"\nPhotometry failed for {len(failures)} filter(s):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


def _print_filter_photometry(
    args: argparse.Namespace,
    flux_cube_path: Path,
    filter_entries: Sequence[Tuple[str, Path]],
    filter_roots: Sequence[Path],
    wl: np.ndarray,
    flux: np.ndarray,
) -> Tuple[Dict[str, FilterCurve], List[str]]:
    """Print a magnitude for each selected filter; return the curves and "name: error" failures."""
    system_name, selected_entries = prompt_for_filter_system(filter_entries, args.filter_system)
    photometric_system = prompt_for_photometric_system(args.photometric_system)

    vega_spec: Spectrum | None = None

    def ensure_vega_spec() -> Spectrum:
        nonlocal vega_spec
        if vega_spec is None:
            try:
                spectrum_path = resolve_vega_spectrum_path(
                    args.vega_sed, flux_cube_path, args.filters or []
                )
            except FileNotFoundError as exc:
                raise SystemExit(
                    f"{exc} Provide a Vega spectrum via --vega-sed or place one near the flux cube."
                )
            print(f"Using Vega spectrum: {spectrum_path}")
            vega_spec = load_spectrum(str(spectrum_path))
        return vega_spec

    source_msg = ", ".join(str(path) for path in filter_roots)
    if not source_msg:
        source_msg = "provided paths" if args.filters else "inferred filter directories"
    if system_name:
        source_msg = f"{source_msg} (system: {system_name})"

    print(
        f"\nFilter photometry (using {len(selected_entries)} filters from {source_msg})\n"
        f"Photometric system: {photometric_system}"
    )

    filters: Dict[str, FilterCurve] = {}
    failures: List[str] = []
    for name, fpath in selected_entries:
        curve = load_filter_curve(str(fpath), name)
        filters[name] = curve
        try:
            flux_label, flux_density, magnitude = _band_magnitude(
                wl, flux, curve, photometric_system, ensure_vega_spec
            )
        except ValueError as exc:
            print(f"  {name:30s}  error: {exc}")
            failures.append(f"{name}: {exc}")
            continue

        mag_text = f"{magnitude:.4f}" if math.isfinite(magnitude) else "undefined"
        print(f"  {name:30s}  {flux_label}={flux_density:.6e}  mag={mag_text}")
    _warn_outside_sed(wl, filters.values())
    return filters, failures


if __name__ == "__main__":
    raise SystemExit(main())

