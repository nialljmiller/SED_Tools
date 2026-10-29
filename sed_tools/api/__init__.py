"""
Python API for SED_Tools - Programmatic interface to stellar spectral grids.

This module provides a Pythonic interface that mirrors and extends the CLI functionality.
Users can query available spectral grids, fetch filtered data, and construct pipelines.

Example usage::

    from sed_tools.api import SED
    
    # Query available remote spectral grids
    grids = SED.query()
    grids = SED.query(teff_min=5000, teff_max=7000)  # Filter by coverage
    
    # Fetch data from a specific spectral grid
    sed = SED.fetch('Kurucz2003all', teff_min=4000, teff_max=7000, logg_min=3.0)
    
    # Access the spectral grid data
    print(sed.grid)
    print(sed.grid.spectra)
    print(sed.grid.parameters)
    
    # Save to disk (same structure as CLI)
    sed.grid.write()
    sed.grid.write('/custom/path')
    
    # Work with local data
    sed = SED.local('Kurucz2003all')
    spectrum = sed(5777, 4.44, 0.0)  # Interpolate
    
    # Create ensemble models
    ensemble = SED.combine(['Kurucz2003all', 'PHOENIX'], output='combined_grid')
    
    # ML SED completion
    completer = SED.ml_completer()
    completer.train(grid='combined_grid')
    completer.extend('sparse_model', wavelength_range=(100, 100000))
"""

from __future__ import annotations

import csv
import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from requests import RequestException

from ..grids._lookup_io import find_teff_column, find_logg_column, find_metallicity_column, find_file_column
from ..cube.model import (
    FILTER_DIR_DEFAULT,
    STELLAR_DIR_DEFAULT,
    EvaluatedSED,
    ModelMatch,
    SEDModel,
    SED as _SEDCore,
)
from .filters import Filters, _listing_failures  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

Number = Union[int, float, np.floating]

# Remote spectral grid sources, in the order SED.fetch tries them.
_SPECTRA_SOURCES = ("njm", "svo", "msg", "mast")

# The sources SED.fetch can ask for only the spectra within its Teff, log g and [M/H] cuts. (MSG
# downloads its grid's one HDF5 file whole and extracts the spectra within them; SVO serves whole grids.)
_CUTTING_SOURCES = ("njm", "msg")

# What a remote source raises when it is unreachable or serves something unexpected.
_SOURCE_ERRORS = (RequestException, OSError, ValueError, RuntimeError)


def _directory_state(directory: Path) -> Dict[str, int]:
    """Name and modification time of each entry in ``directory`` ({} if it does not exist)."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}


def _mixed_sources_error(src: str, model_dir, tried: List[str]) -> RuntimeError:
    """The error when ``src`` failed after writing into ``model_dir``, so no further source is tried."""
    return RuntimeError(f"{src} failed after changing {model_dir}, so another source's spectra would be mixed "
                        f"with its files; no further source tried. Tried: {'; '.join(tried)}")


def _download_count(n_spectra: int, before: Dict[str, int], after: Dict[str, int]) -> str:
    """``Downloaded N spectra``, or ``K already present, N downloaded`` when the grabber's
    ``n_spectra`` (the spectra it downloaded or found in place) includes spectra it did not write.

    ``before`` and ``after`` are :func:`_directory_state` around the download.
    """
    written = sum(1 for name, mtime in after.items()
                  if name.lower().endswith(".txt") and before.get(name) != mtime)
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


def _flux_cube_file(model_dir: Path) -> Optional[Path]:
    """The grid's flux cube: ``flux_cube.bin``, else ``variants_index.csv`` (one cube per variant), else None."""
    for name in ("flux_cube.bin", "variants_index.csv"):
        if (Path(model_dir) / name).is_file():
            return Path(model_dir) / name
    return None


def _flux_cube_shortfall(model_dir: Path) -> Optional[str]:
    """What the grid's flux cube lacks of the spectra in ``model_dir``, or None when it holds them all.

    None too when there is no cube, or no spectrum.  The spectra are the
    ``.txt`` files there, at the (Teff, logg, [M/H]) its ``lookup_table.csv``
    gives them, and the cube holds one when it has a point there that holds
    flux (:func:`~sed_tools.precompute_flux_cube.spectra_not_in_cube`).  A
    cube built from a selection, before the rest was downloaded beside it,
    lacks the rest.  With ``variants_index.csv``, each variant's cube is
    compared with its own lookup table, and between them they must list every
    spectrum.  The files' contents decide, not their times, so a grid copied
    without them (``cp -R``, ``scp``) is as complete as the original.
    """
    from ..cube.build import spectra_not_in_cube
    from ..grids.spectrum_io import list_text_spectra

    model_dir = Path(model_dir)
    cube = _flux_cube_file(model_dir)
    present = {path.name for path in list_text_spectra(model_dir)} if cube is not None else set()
    if not present:
        return None
    try:
        variants = [model_dir] if cube.name == "flux_cube.bin" else [model_dir / v for v in _variant_paths(cube)]
        listed, lacking = set(), set()
        for directory in variants:
            names, missing = spectra_not_in_cube(directory / "flux_cube.bin", directory / "lookup_table.csv")
            listed.update(names)
            lacking.update(missing)
    except (OSError, ValueError, KeyError, TypeError) as exc:  # unreadable or malformed files
        return f"cannot be compared with the spectra beside it ({exc})"
    lacking = (lacking | (present - listed)) & present
    if not lacking:
        return None
    return f"does not include {len(lacking)} of the {len(present)} spectra beside it"


def _variant_paths(variants_index: Path) -> List[str]:
    """The folder of each variant ``variants_index.csv`` lists, relative to the grid's folder."""
    with open(variants_index, newline="", encoding="utf-8") as fh:
        fields = [name.lstrip("#").strip() for name in fh.readline().strip().split(",")]
        return [row["path"] for row in csv.DictReader(fh, fieldnames=fields)]


def _bundle_shortfall(bundle: Path, model_dir: Path) -> Optional[str]:
    """How an HDF5 bundle's spectra differ from the ``.txt`` spectra in ``model_dir``; None when they are the same.

    The bundle records each spectrum under its file name
    (:func:`~sed_tools.spectrum_io.build_h5_bundle`); names are compared, not
    file times.
    """
    from ..grids.spectrum_io import bundled_spectra, list_text_spectra

    present = {path.name for path in list_text_spectra(model_dir)}
    try:
        held = set(bundled_spectra(bundle))
    except (OSError, ValueError) as exc:
        return f"cannot be compared with the spectra beside it ({exc})"
    if held == present:
        return None
    extra = len(held - present)
    return (f"holds {len(held & present)} of the {len(present)} spectra beside it"
            + (f" and {extra} that are not there" if extra else ""))


def _build_flux_cube(model_dir: Path) -> Path:
    """Run precompute_flux_cube on ``model_dir`` and return what it wrote.

    That is ``flux_cube.bin``, or ``variants_index.csv`` when a grid with extra
    axes is split into one flux cube per variant.  precompute_flux_cube can
    also stop without writing either (it then prints why), so RuntimeError is
    raised unless one of them was written by this call.
    """
    from ..cube.build import precompute_flux_cube

    model_dir = Path(model_dir)
    outputs = (model_dir / "flux_cube.bin", model_dir / "variants_index.csv")
    before = {path: path.stat().st_mtime_ns for path in outputs if path.exists()}
    precompute_flux_cube(str(model_dir), str(outputs[0]))
    for path in outputs:
        if path.exists() and path.stat().st_mtime_ns != before.get(path):
            return path
    raise RuntimeError(f"No flux cube was built in {model_dir}; see the flux cube messages above")


def _build_combined_grid(selected_models: List[Tuple[str, str]], output_dir: str, *,
                         visualization: bool = True) -> None:
    """Merge the ``(name, path)`` grids onto common parameter and wavelength grids and save to ``output_dir``."""
    from ..cube.combine import (
        build_combined_flux_cube,
        create_common_wavelength_grid,
        create_unified_grid,
        load_model_data,
        save_combined_data,
        visualize_parameter_space,
    )

    print("\nLoading spectral grid data...")
    all_models_data = []
    for name, path in selected_models:
        print(f"  Loading {name}...")
        all_models_data.append(load_model_data(path))

    print("\nCreating unified parameter grids...")
    teff_grid, logg_grid, meta_grid = create_unified_grid(all_models_data)
    wavelength_grid = create_common_wavelength_grid(all_models_data)

    print("\nBuilding combined flux cube...")
    flux_cube, source_map = build_combined_flux_cube(
        all_models_data, teff_grid, logg_grid, meta_grid, wavelength_grid
    )

    save_combined_data(
        output_dir,
        teff_grid,
        logg_grid,
        meta_grid,
        wavelength_grid,
        flux_cube,
        all_models_data,
    )

    if visualization:
        visualize_parameter_space(
            teff_grid, logg_grid, meta_grid, source_map, all_models_data, output_dir,
            wavelength_grid=wavelength_grid, flux_cube=flux_cube)


# =============================================================================
# Data Classes for API Results
# =============================================================================

@dataclass
class SpectralGridInfo:
    """Information about an available stellar spectral grid."""
    
    name: str
    source: str  # 'svo', 'msg', 'mast', 'njm', 'local'
    teff_range: Optional[Tuple[float, float]] = None
    logg_range: Optional[Tuple[float, float]] = None
    metallicity_range: Optional[Tuple[float, float]] = None
    n_spectra: Optional[int] = None
    wavelength_range: Optional[Tuple[float, float]] = None
    description: str = ""
    url: Optional[str] = None
    is_local: bool = False
    
    def covers(
        self,
        teff: Optional[float] = None,
        logg: Optional[float] = None,
        metallicity: Optional[float] = None,
    ) -> bool:
        """Check if this spectral grid covers a specific point in parameter space."""
        if teff is not None and self.teff_range:
            if not (self.teff_range[0] <= teff <= self.teff_range[1]):
                return False
        if logg is not None and self.logg_range:
            if not (self.logg_range[0] <= logg <= self.logg_range[1]):
                return False
        if metallicity is not None and self.metallicity_range:
            if not (self.metallicity_range[0] <= metallicity <= self.metallicity_range[1]):
                return False
        return True
    
    def covers_range(
        self,
        teff_min: Optional[float] = None,
        teff_max: Optional[float] = None,
        logg_min: Optional[float] = None,
        logg_max: Optional[float] = None,
        metallicity_min: Optional[float] = None,
        metallicity_max: Optional[float] = None,
    ) -> bool:
        """Check if this spectral grid covers a parameter range (at least partial overlap)."""
        if self.teff_range:
            if teff_min is not None and teff_min > self.teff_range[1]:
                return False
            if teff_max is not None and teff_max < self.teff_range[0]:
                return False
        if self.logg_range:
            if logg_min is not None and logg_min > self.logg_range[1]:
                return False
            if logg_max is not None and logg_max < self.logg_range[0]:
                return False
        if self.metallicity_range:
            if metallicity_min is not None and metallicity_min > self.metallicity_range[1]:
                return False
            if metallicity_max is not None and metallicity_max < self.metallicity_range[0]:
                return False
        return True
    
    def __repr__(self) -> str:
        parts = [f"SpectralGridInfo('{self.name}', source='{self.source}'"]
        if self.teff_range:
            parts.append(f"teff={self.teff_range}")
        if self.logg_range:
            parts.append(f"logg={self.logg_range}")
        if self.n_spectra:
            parts.append(f"n_spectra={self.n_spectra}")
        return ", ".join(parts) + ")"


@dataclass
class Spectrum:
    """An individual model spectrum with wavelength, flux and parameters."""
    
    wavelength: np.ndarray  # Angstroms
    flux: np.ndarray        # erg/cm²/s/Å
    teff: float
    logg: float
    metallicity: float
    filename: Optional[str] = None
    source_catalog: Optional[str] = None
    
    @property
    def wl(self) -> np.ndarray:
        """Alias for wavelength."""
        return self.wavelength
    
    @property
    def fl(self) -> np.ndarray:
        """Alias for flux."""
        return self.flux
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'wavelength': self.wavelength,
            'flux': self.flux,
            'teff': self.teff,
            'logg': self.logg,
            'metallicity': self.metallicity,
            'filename': self.filename,
        }
    
    def save(self, path: Union[str, Path], format: str = 'txt') -> None:
        """Save spectrum to file."""
        path = Path(path)
        
        if format == 'txt':
            with open(path, 'w') as f:
                f.write(f"# teff = {self.teff}\n")
                f.write(f"# logg = {self.logg}\n")
                f.write(f"# metallicity = {self.metallicity}\n")
                f.write("# wavelength_unit = angstrom\n")
                f.write("# flux_unit = erg/cm2/s/A\n")
                f.write("# units_standardized = True\n")
                for w, fl in zip(self.wavelength, self.flux):
                    f.write(f"{w:.6f} {fl:.8e}\n")
        else:
            raise ValueError(f"Unsupported format: {format}")


@dataclass  
class SpectralGrid:
    """
    A spectral grid of model spectra and their parameters.
    
    This is the main data container returned by SED.fetch().
    """
    
    name: str
    source: str
    spectra: List[Spectrum] = field(default_factory=list)
    _base_dir: Path = field(default_factory=lambda: STELLAR_DIR_DEFAULT)
    _parameters_df: Optional[pd.DataFrame] = None
    
    @property
    def parameters(self) -> pd.DataFrame:
        """Return a DataFrame of all spectrum parameters."""
        if self._parameters_df is None:
            rows = []
            for spec in self.spectra:
                rows.append({
                    'file_name': spec.filename or f"teff{spec.teff}_logg{spec.logg}_meta{spec.metallicity}.txt",
                    'teff': spec.teff,
                    'logg': spec.logg,
                    'metallicity': spec.metallicity,
                })
            self._parameters_df = pd.DataFrame(rows)
        return self._parameters_df
    
    @property
    def teff_grid(self) -> np.ndarray:
        """Unique Teff values in the spectral grid."""
        return np.unique([s.teff for s in self.spectra])
    
    @property
    def logg_grid(self) -> np.ndarray:
        """Unique logg values in the spectral grid."""
        return np.unique([s.logg for s in self.spectra])
    
    @property
    def metallicity_grid(self) -> np.ndarray:
        """Unique metallicity values in the spectral grid."""
        return np.unique([s.metallicity for s in self.spectra])
    
    def filter(
        self,
        teff_min: Optional[float] = None,
        teff_max: Optional[float] = None,
        logg_min: Optional[float] = None,
        logg_max: Optional[float] = None,
        metallicity_min: Optional[float] = None,
        metallicity_max: Optional[float] = None,
    ) -> 'SpectralGrid':
        """Return a new SpectralGrid with filtered spectra."""
        filtered = []
        for spec in self.spectra:
            if teff_min is not None and spec.teff < teff_min:
                continue
            if teff_max is not None and spec.teff > teff_max:
                continue
            if logg_min is not None and spec.logg < logg_min:
                continue
            if logg_max is not None and spec.logg > logg_max:
                continue
            if metallicity_min is not None and spec.metallicity < metallicity_min:
                continue
            if metallicity_max is not None and spec.metallicity > metallicity_max:
                continue
            filtered.append(spec)
        
        return SpectralGrid(
            name=self.name,
            source=self.source,
            spectra=filtered,
            _base_dir=self._base_dir,
        )
    
    def __len__(self) -> int:
        return len(self.spectra)
    
    def __iter__(self) -> Iterator[Spectrum]:
        return iter(self.spectra)
    
    def __getitem__(self, idx: int) -> Spectrum:
        return self.spectra[idx]
    
    def write(
        self,
        path: Optional[Union[str, Path]] = None,
        build_flux_cube: bool = True,
        build_h5: bool = True,
    ) -> Path:
        """
        Write the spectral grid to disk in the standard SED_Tools structure.
        
        Parameters
        ----------
        path : str or Path, optional
            Output directory. Defaults to {base_dir}/{catalog_name}
        build_flux_cube : bool
            Whether to build the MESA-compatible flux cube
        build_h5 : bool
            Whether to build the HDF5 bundle
            
        Returns
        -------
        Path
            The output directory path
        """
        if path is None:
            path = self._base_dir / self.name
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Write individual spectra
        for spec in self.spectra:
            fname = spec.filename or f"teff{spec.teff:.0f}_logg{spec.logg:.2f}_meta{spec.metallicity:.2f}.txt"
            spec.save(path / fname)
        
        # Write lookup table
        self.parameters.to_csv(path / "lookup_table.csv", index=False)
        print(f"[write] Wrote {len(self.spectra)} spectra to {path}")
        
        # Build flux cube
        if build_flux_cube and len(self.spectra) > 0:
            written = _build_flux_cube(path)
            if written.name == "flux_cube.bin":
                print("[write] Built flux_cube.bin")
            else:
                print(f"[write] Built one flux cube per extra-axis variant, listed in {written}")

        # Build H5 bundle
        if build_h5 and len(self.spectra) > 0:
            from .. import build_h5_bundle_from_txt
            if not build_h5_bundle_from_txt(str(path), str(path / f"{self.name}.h5")):
                raise RuntimeError(f"No .txt spectra in {path} to bundle into {self.name}.h5")
            print(f"[write] Built {self.name}.h5")

        return path


# Existing imports continue to refer to the same classes.
Catalog = SpectralGrid
CatalogInfo = SpectralGridInfo


def _read_listed_spectra(lookup_path: Path, grid_name: str) -> List[Spectrum]:
    """The spectra a lookup table lists, for rows whose file exists beside it.

    Raises ValueError naming the file when a listed spectrum cannot be read.
    """
    model_dir = lookup_path.parent
    df = pd.read_csv(lookup_path)
    df.columns = [c.lstrip('#').strip().lower() for c in df.columns]
    file_col = find_file_column(df.columns)
    teff_col = find_teff_column(df.columns)
    logg_col = find_logg_column(df.columns)
    meta_col = find_metallicity_column(df.columns)

    spectra = []
    for _, row in df.iterrows():
        filename = row[file_col] if file_col else None
        if not filename or not (model_dir / filename).exists():
            continue
        try:
            wl, fl = np.loadtxt(model_dir / filename, unpack=True, comments='#')
            spectra.append(Spectrum(
                wavelength=wl,
                flux=fl,
                teff=float(row.get(teff_col, np.nan)) if teff_col else np.nan,
                logg=float(row.get(logg_col, np.nan)) if logg_col else np.nan,
                metallicity=float(row.get(meta_col, np.nan)) if meta_col else np.nan,
                filename=filename,
                source_catalog=grid_name,
            ))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read spectrum {model_dir / filename} listed in {lookup_path}: {exc}") from exc
    return spectra


def _import_sources(src: Path) -> Tuple[List[Path], Path]:
    """The .txt spectra to import from a directory or a single file, and their directory."""
    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")

    if src.is_dir():
        files = sorted(src.glob("*.txt"))
        src_dir = src
    elif src.is_file() and src.suffix.lower() == ".txt":
        files = [src]
        src_dir = src.parent
    else:
        raise ValueError(f"Source must be a directory or a .txt file: {src}")

    if not files:
        raise RuntimeError(f"No .txt spectra found in {src}")
    return files, src_dir


def _report_import_headers(files: List[Path], src_dir: Path) -> List[str]:
    """Print how many files carry Teff, logg and [M/H]; return the names of those that do not."""
    from ..grids.header_parser import parse_header

    missing: List[str] = []
    for f in files:
        h = parse_header(str(f))
        vals = [float(h.get(k, float("nan"))) for k in ("teff", "logg", "metallicity")]
        if any(math.isnan(v) for v in vals):
            missing.append(f.name)

    print(f"[import] {len(files)} .txt files in {src_dir}")
    print(f"[import] parseable Teff+logg+[M/H] headers: {len(files) - len(missing)}/{len(files)}")
    if missing:
        print(f"[import] missing one or more parameters: {len(missing)}")
        for b in missing[:10]:
            print(f"           {b}")
        if len(missing) > 10:
            print(f"           ... and {len(missing) - 10} more")
        print("[import] If a key is merely unrecognised, add it to ALIASES "
              "in header_parser.py.")
    return missing


# =============================================================================
# Main SED API Class
# =============================================================================

class SED:
    """
    Main API class for SED_Tools.
    
    This class provides both class methods for discovery/fetching and instance
    methods for working with loaded data.
    
    Class methods (discovery):
        SED.query() - List available spectral grids
        SED.fetch() - Download and return a spectral grid
        SED.local() - Load a local spectral grid
        SED.combine() - Create ensemble grids
        SED.ml_completer() - Access ML completion tools
        SED.ml_generator() - Access ML generation tools
    
    Instance methods (loaded data):
        sed(teff, logg, metallicity) - Interpolate a spectrum
        sed.model() - Access the underlying SEDModel
        sed.photometry() - Compute synthetic photometry
    
    Examples
    --------
    >>> # Query what's available
    >>> grids = SED.query()
    >>> grids = SED.query(teff_min=5000)
    
    >>> # Fetch remote data
    >>> sed = SED.fetch('Kurucz2003all', teff_min=4000, teff_max=8000)
    >>> sed.grid.write()
    
    >>> # Work with local data
    >>> sed = SED.local('Kurucz2003all')
    >>> spectrum = sed(5777, 4.44, 0.0)
    
    >>> # Generate SEDs from parameters (no input spectrum needed)
    >>> generator = SED.ml_generator()
    >>> generator.load('sed_generator_Kurucz2003all')
    >>> wl, flux = generator.generate(teff=5777, logg=4.44, metallicity=0.0)
    """
    
    _model_root: Path = STELLAR_DIR_DEFAULT
    _filter_root: Path = FILTER_DIR_DEFAULT
    
    def __init__(
        self,
        catalog: Optional[SpectralGrid] = None,
        model: Optional[SEDModel] = None,
        model_root: Optional[Union[str, Path]] = None,
        filter_root: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize SED instance.
        
        Typically created via class methods like SED.fetch() or SED.local().
        """
        self.cat = catalog
        self._model = model
        self._core = _SEDCore(
            model_root=model_root or self._model_root,
            filter_root=filter_root or self._filter_root,
        )
        
        if model_root:
            self._model_root = Path(model_root)
        if filter_root:
            self._filter_root = Path(filter_root)
    
    @property
    def grid(self) -> Optional[SpectralGrid]:
        """The loaded spectral grid; also available through the legacy ``cat`` attribute."""
        return self.cat

    # =========================================================================
    # Class Methods - Discovery & Fetching
    # =========================================================================
    
    @classmethod
    def query(
        cls,
        source: Optional[str] = None,
        teff_min: Optional[float] = None,
        teff_max: Optional[float] = None,
        logg_min: Optional[float] = None,
        logg_max: Optional[float] = None,
        metallicity_min: Optional[float] = None,
        metallicity_max: Optional[float] = None,
        include_local: bool = True,
        include_remote: bool = True,
    ) -> List[SpectralGridInfo]:
        """
        Query available spectral grids.
        
        Parameters
        ----------
        source : str, optional
            Filter by source: 'svo', 'msg', 'mast', 'njm', 'local', or 'all'
        teff_min, teff_max : float, optional
            Filter spectral grids that cover this Teff range
        logg_min, logg_max : float, optional
            Filter spectral grids that cover this logg range
        metallicity_min, metallicity_max : float, optional
            Filter spectral grids that cover this metallicity range
        include_local : bool
            Include locally installed spectral grids
        include_remote : bool
            Query remote sources (SVO, MSG, MAST, NJM)
            
        Returns
        -------
        List[SpectralGridInfo]
            Available spectral grids matching the criteria

        A remote source that cannot be asked lists nothing, after a warning
        from its client; RuntimeError if NJM is asked for alone and is
        unreachable.
        """
        if source not in (None, 'local', 'all', *_SPECTRA_SOURCES):
            raise ValueError(f"Unknown source: {source}")
        catalogs = []
        if include_local and source in (None, 'local', 'all'):
            catalogs.extend(cls._query_local())
        if include_remote:
            catalogs.extend(cls._query_remote(source))

        # Filter by parameter coverage; a bound of 0 is a bound
        bounds = (teff_min, teff_max, logg_min, logg_max, metallicity_min, metallicity_max)
        if any(b is not None for b in bounds):
            catalogs = [
                c for c in catalogs
                if c.covers_range(
                    teff_min=teff_min,
                    teff_max=teff_max,
                    logg_min=logg_min,
                    logg_max=logg_max,
                    metallicity_min=metallicity_min,
                    metallicity_max=metallicity_max,
                )
            ]
        
        return catalogs
    
    @classmethod
    def _query_local(cls) -> List[SpectralGridInfo]:
        """Query locally installed spectral grids."""
        catalogs = []
        
        if not cls._model_root.exists():
            return catalogs
        
        for entry in sorted(cls._model_root.iterdir()):
            if not entry.is_dir():
                continue
            
            cube = entry / "flux_cube.bin"
            lookup = entry / "lookup_table.csv"
            
            if not cube.is_file():
                continue
            
            info = SpectralGridInfo(
                name=entry.name,
                source='local',
                is_local=True,
            )
            
            # Parameter ranges from the flux cube header
            try:
                from ..cube.model import _read_flux_cube_header
                meta = _read_flux_cube_header(cube)
                info.teff_range = (float(meta['teff'][0]), float(meta['teff'][-1]))
                info.logg_range = (float(meta['logg'][0]), float(meta['logg'][-1]))
                info.metallicity_range = (float(meta['meta'][0]), float(meta['meta'][-1]))
                info.wavelength_range = (float(meta['wavelengths'][0]), float(meta['wavelengths'][-1]))
            except (OSError, ValueError, IndexError) as exc:
                raise ValueError(f"Cannot read the flux cube header of local grid '{entry.name}' ({cube}): "
                                 f"{exc}. Rebuild it with: sed-tools rebuild --grids {entry.name}") from exc

            # Count spectra from lookup table
            if lookup.exists():
                try:
                    info.n_spectra = len(pd.read_csv(lookup))
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Cannot read {lookup} of local grid '{entry.name}': {exc}") from exc

            catalogs.append(info)
        
        return catalogs
    
    @classmethod
    def _query_remote(cls, source: Optional[str]) -> List[SpectralGridInfo]:
        """Grids from each requested remote source.  SVO, MSG or MAST list none when they cannot be
        asked, and their clients warn why; an unreachable NJM is left out (its client has warned),
        or raises when it is the source asked for."""
        catalogs = []
        for name, query in (('svo', cls._query_svo), ('msg', cls._query_msg), ('mast', cls._query_mast)):
            if source in (None, 'all', name):
                catalogs.extend(query())
        if source in (None, 'all', 'njm'):
            try:
                catalogs.extend(cls._query_njm())
            except RuntimeError:
                if source == 'njm':
                    raise
        return catalogs

    @classmethod
    def _query_svo(cls) -> List[SpectralGridInfo]:
        """Query SVO remote spectral grids."""
        from ..grids.svo import SVOSpectraGrabber
        grabber = SVOSpectraGrabber(base_dir=str(cls._model_root))
        return [
            SpectralGridInfo(name=m, source='svo', url="http://svo2.cab.inta-csic.es/theory/newov2/")
            for m in grabber.discover_models()
        ]

    @classmethod
    def _query_msg(cls) -> List[SpectralGridInfo]:
        """Query MSG (Townsend) remote spectral grids."""
        from ..grids.msg import MSGSpectraGrabber
        grabber = MSGSpectraGrabber(base_dir=str(cls._model_root))
        return [
            SpectralGridInfo(name=m, source='msg', description="MSG HDF5 format")
            for m in grabber.discover_models()
        ]

    @classmethod
    def _query_mast(cls) -> List[SpectralGridInfo]:
        """Query MAST (BOSZ) remote spectral grids."""
        from ..grids.mast import MASTSpectraGrabber
        grabber = MASTSpectraGrabber(base_dir=str(cls._model_root))
        return [
            SpectralGridInfo(name=m, source='mast', description="BOSZ spectral grids")
            for m in grabber.discover_models()
        ]

    @classmethod
    def _query_njm(cls) -> List[SpectralGridInfo]:
        """Query NJM mirror remote spectral grids; RuntimeError, without a second warning, if the
        mirror is unreachable (the NJM client warns once per outage)."""
        from ..grids.njm import NJMSpectraGrabber
        grabber = NJMSpectraGrabber(base_dir=str(cls._model_root))
        if not grabber.is_available():
            raise RuntimeError(f"NJM mirror {grabber.base_url} is unreachable")
        return [
            SpectralGridInfo(name=m, source='njm', description="NJM pre-processed mirror")
            for m in grabber.discover_models()
        ]

    @classmethod
    def fetch(
        cls,
        catalog: str,
        source: Optional[str] = None,
        teff_min: Optional[float] = None,
        teff_max: Optional[float] = None,
        logg_min: Optional[float] = None,
        logg_max: Optional[float] = None,
        metallicity_min: Optional[float] = None,
        metallicity_max: Optional[float] = None,
        workers: int = 5,
        clean: bool = True,
        model_root: Optional[Union[str, Path]] = None,
        flux_unit: Optional[str] = None,
    ) -> 'SED':
        """
        Fetch and download a stellar spectral grid.
        
        Parameters
        ----------
        catalog : str
            Name of the spectral grid to fetch (e.g., 'Kurucz2003all')
        source : str, optional
            Force a specific source: 'svo', 'msg', 'mast', 'njm'
            If None, tries NJM first (pre-processed), then others
        teff_min, teff_max : float, optional
            Download and return only spectra within this Teff range (see Notes)
        logg_min, logg_max : float, optional
            Download and return only spectra within this logg range (see Notes)
        metallicity_min, metallicity_max : float, optional
            Download and return only spectra within this metallicity range (see Notes)
        workers : int
            Number of parallel download workers
        clean : bool
            Whether to run unit standardization after download
        model_root : str or Path, optional
            Base directory for spectral grids
        flux_unit : str, optional
            The spectra's flux unit for cleaning (see spectra_cleaner.clean_model_dir);
            required when their headers do not declare one, and never used
            in place of a declared unit.

        Notes
        -----
        With any Teff, logg or metallicity bound (a bound of 0 is a bound; a
        single bound leaves the other side open), only the spectra within the
        bounds are downloaded, as ``sed-tools grids --teff ... --logg ...
        --meta ...`` does, and the lookup table, HDF5 bundle and flux cube in
        ``model_root/catalog`` are built from them, so ``SED.local`` and MESA
        use that selection. NJM's flux cube and HDF5 bundle of the whole grid
        are not downloaded. MSG downloads its grid's HDF5 file whole and
        extracts only the selected spectra. SVO cannot select spectra, so from
        SVO the whole grid is downloaded, with a warning, then cut.

        A folder that already holds every selected spectrum with a flux cube
        holding every spectrum there (the whole grid, or a wider selection)
        is cut as it stands: nothing is downloaded or rewritten. A later fetch
        without bounds, or with wider ones, downloads the rest; a flux cube
        that then lacks spectra beside it is rebuilt from every spectrum in
        the folder, with the lookup table and HDF5 bundle. Whether the cube
        holds them is read from the cube and the lookup table, not from file
        times, so a grid copied without its file times counts as complete.
        Bounds that no spectrum meets leave no new folder behind.

        Returns
        -------
        SED
            SED instance with the fetched spectral grid in sed.grid

        Raises
        ------
        ValueError
            If no source could supply the grid; the message gives each source's error.
            Also if the spectra do not declare their flux unit and flux_unit is not
            given; the message gives the best guess.
        RuntimeError
            If a failed download left files behind (another source is then
            not tried), or cleaning left spectra unusable.
        """
        base_dir = Path(model_root) if model_root else cls._model_root
        base_dir.mkdir(parents=True, exist_ok=True)

        if source is None:
            sources_to_try, failures = cls._get_sources_for_catalog(catalog)
        elif source in _SPECTRA_SOURCES:
            sources_to_try, failures = [source], []
        else:
            raise ValueError(f"Unknown source: {source}")
        model_dir = base_dir / catalog
        cuts = _parameter_cuts(teff_min, teff_max, logg_min, logg_max, metallicity_min, metallicity_max)
        created = not model_dir.exists()
        before = _directory_state(model_dir)
        used_source = cls._download_from_first_source(catalog, sources_to_try, failures, model_dir, workers,
                                                      cuts)
        selected = bool(cuts) and used_source in _CUTTING_SOURCES
        cube = _flux_cube_file(model_dir)

        if (selected and cube is not None and _directory_state(model_dir) == before
                and _flux_cube_shortfall(model_dir) is None):
            # The whole grid, or a wider selection, installed with its flux cube: cleaning would
            # rewrite its lookup table, so it is cut as it stands.
            print(f"[fetch] {model_dir} already holds every spectrum of {catalog} within the bounds, with "
                  "its flux cube; nothing was downloaded or rewritten")
        elif created and selected and not (model_dir.is_dir() and any(model_dir.iterdir())):
            if model_dir.is_dir():
                model_dir.rmdir()  # no spectrum was within the bounds: remove the empty folder this call made
        else:
            if clean:
                cls._clean_fetched(model_dir, flux_unit)  # leaves the flux cube files alone
            shortfall = _flux_cube_shortfall(model_dir)  # None without a cube
            if ((selected and cube is None) or shortfall) and any(model_dir.glob("*.txt")):
                # As sed-tools grids does: a selection gets a lookup table, HDF5 bundle and flux cube of
                # its own, and a cube that lacks spectra beside it (a selection, then the rest) is
                # rebuilt from every spectrum present.
                if shortfall:
                    print(f"[fetch] {cube.name} {shortfall}:\n        rebuilding the lookup table, HDF5 "
                          "bundle and flux cube from all of them")
                from ..cli import _build_data_products
                _build_data_products(catalog, str(model_dir), used_source, True, True)

        # Load spectra into SpectralGrid object
        cat = cls._load_catalog_from_dir(model_dir, catalog, used_source)
        
        # Apply parameter filters; a bound of 0 is a bound
        if cuts:
            cat = cat.filter(
                teff_min=teff_min,
                teff_max=teff_max,
                logg_min=logg_min,
                logg_max=logg_max,
                metallicity_min=metallicity_min,
                metallicity_max=metallicity_max,
            )
            print(f"[fetch] Filtered to {len(cat)} spectra")
        
        return cls(catalog=cat, model_root=base_dir)

    @staticmethod
    def _clean_fetched(model_dir: Path, flux_unit: Optional[str]) -> None:
        """Standardize the units of a fetched grid; clean_model_dir raises RuntimeError for any spectrum
        it leaves unusable."""
        from ..grids.clean import clean_model_dir
        summary = clean_model_dir(str(model_dir), try_h5_recovery=True, rebuild_lookup=True,
                                  flux_unit=flux_unit)
        print(f"[fetch] Cleaned: {summary['total']} total, {len(summary['converted'])} converted")

    @classmethod
    def _get_sources_for_catalog(cls, catalog: str) -> Tuple[List[str], List[str]]:
        """Sources whose index lists this spectral grid (NJM first, then SVO as a fallback), and the
        entry for an unreachable NJM, which goes in the fetch's error (its client has warned).

        SVO, MSG or MAST list nothing when they cannot be asked; their clients warn why.
        """
        sources, failures = [], []
        try:
            if catalog in [info.name for info in cls._query_njm()]:
                sources.append('njm')
        except RuntimeError as exc:
            failures.append(f"njm: discovery failed: {exc}")
        for name, query in (('svo', cls._query_svo), ('msg', cls._query_msg), ('mast', cls._query_mast)):
            if catalog in [info.name for info in query()]:
                sources.append(name)

        # Always include SVO as fallback if not already there
        if 'svo' not in sources:
            sources.append('svo')
        return sources, failures

    @classmethod
    def _download_from_first_source(cls, catalog: str, sources: List[str], failures: List[str],
                                    model_dir: Path, workers: int,
                                    cuts: Optional[Dict[str, Tuple[float, float]]] = None) -> str:
        """Download ``catalog`` into ``model_dir`` from the first source that can supply it.

        Returns that source.  A source that fails is recorded in ``failures``
        with a warning and the next one is tried, unless the failed attempt
        changed ``model_dir``: the next source's spectra would then be mixed
        with its files, so RuntimeError is raised instead.  ValueError lists
        every failure when no source succeeds.  ``cuts`` are as for
        :meth:`_download_from`.
        """
        for src in sources:
            print(f"[fetch] Trying {catalog} from {src}...")
            before = _directory_state(model_dir)
            error = cls._download_from(src, catalog, model_dir, workers, cuts)
            if error is None:
                return src
            failures.append(f"{src}: {error}")
            if _directory_state(model_dir) != before:
                raise _mixed_sources_error(src, model_dir, failures)
            logger.warning("[fetch] %s: %s, trying next...", src, error)
        raise ValueError(f"Spectral grid '{catalog}' could not be fetched. Tried: {'; '.join(failures)}")

    @classmethod
    def _download_from(cls, src: str, catalog: str, model_dir: Path, workers: int,
                       cuts: Optional[Dict[str, Tuple[float, float]]] = None) -> Optional[str]:
        """Download ``catalog`` from one source; returns None on success, else what went wrong.

        ``cuts`` (:func:`_parameter_cuts`) select the spectra a source in
        ``_CUTTING_SOURCES`` downloads; none of them lying within the cuts is a
        success.  Any other source downloads the whole grid, with a warning.
        """
        try:
            grabber = cls._get_grabber(src, str(model_dir.parent), workers)
            metadata = grabber.get_model_metadata(catalog)  # raises when the source has no such grid
            if cuts and src not in _CUTTING_SOURCES:
                logger.warning("[fetch] %s cannot select spectra by Teff, logg or metallicity, so the whole of %s "
                               "is downloaded, then cut", src, catalog)
                cuts = None
            print(f"[fetch] Downloading {catalog} from {src}...")
            before = _directory_state(model_dir)
            n_downloaded = grabber.download_model_spectra(catalog, metadata, **(cuts or {}))
        except _SOURCE_ERRORS as exc:
            return str(exc)
        print(f"[fetch] {_download_count(n_downloaded, before, _directory_state(model_dir))}")
        if cuts and not n_downloaded:
            print(f"[fetch] {src} has no spectrum of {catalog} within the bounds")
            return None
        return None if n_downloaded else "no spectra downloaded"

    @classmethod
    def _get_grabber(cls, source: str, base_dir: str, workers: int):
        """Get the appropriate grabber instance."""
        if source == 'svo':
            from ..grids.svo import SVOSpectraGrabber
            return SVOSpectraGrabber(base_dir=base_dir, max_workers=workers)
        elif source == 'msg':
            from ..grids.msg import MSGSpectraGrabber
            grabber = MSGSpectraGrabber(base_dir=base_dir, max_workers=workers)
            grabber.discover_models()  # get_model_metadata finds grids in the index this reads
            return grabber
        elif source == 'mast':
            # The MAST grabber asks on the terminal which metallicities to download.
            raise RuntimeError("MAST (BOSZ) grids are fetched with 'sed-tools grids --source mast', which asks "
                               "which metallicities to download; SED.fetch cannot answer that prompt")
        elif source == 'njm':
            from ..grids.njm import NJMSpectraGrabber
            return NJMSpectraGrabber(base_dir=base_dir)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    @classmethod
    def _load_catalog_from_dir(cls, model_dir: Path, name: str, source: str) -> SpectralGrid:
        """Load the spectra listed in ``lookup_table.csv`` into a SpectralGrid."""
        lookup_path = model_dir / "lookup_table.csv"
        spectra = _read_listed_spectra(lookup_path, name) if lookup_path.exists() else []
        return SpectralGrid(
            name=name,
            source=source,
            spectra=spectra,
            _base_dir=model_dir.parent,
        )
    
    @classmethod
    def local(
        cls,
        catalog: str,
        model_root: Optional[Union[str, Path]] = None,
        filter_root: Optional[Union[str, Path]] = None,
        fill_gaps: bool = True,
    ) -> 'SED':
        """
        Load a local stellar spectral grid.
        
        Parameters
        ----------
        catalog : str
            Name of the spectral grid (directory name under model_root)
        model_root : str or Path, optional
            Base directory for spectral grids
        filter_root : str or Path, optional
            Base directory for filter profiles
        fill_gaps : bool
            Whether to clamp out-of-range parameters to grid edges
            
        Returns
        -------
        SED
            SED instance ready for interpolation
        """
        base_dir = Path(model_root) if model_root else cls._model_root
        model_dir = base_dir / catalog
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Spectral grid '{catalog}' not found at {model_dir}")
        
        cube_path = model_dir / "flux_cube.bin"
        if not cube_path.exists():
            raise FileNotFoundError(f"flux_cube.bin not found in {model_dir}")
        
        # Load the catalog
        cat = cls._load_catalog_from_dir(model_dir, catalog, 'local')
        
        # Create SEDModel for interpolation
        sed_model = SEDModel(
            name=catalog,
            flux_cube_path=cube_path,
            filters_dir=filter_root or cls._filter_root,
            fill_gaps=fill_gaps,
        )
        
        return cls(
            catalog=cat,
            model=sed_model,
            model_root=base_dir,
            filter_root=filter_root,
        )

    @classmethod
    def coverage(
        cls,
        catalog: Union[str, Path],
        model_root: Optional[Union[str, Path]] = None,
        plot: bool = True,
        out_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Report parameter-space coverage of a local grid (downloaded or built).

        Prints per-axis ranges, unique grid values and spacing, the node count
        vs. the full Teff x logg x [M/H] product (fill fraction), and an
        Anderson-Darling normality stat per axis. Optionally writes a
        Teff-logg + 3D coverage plot (defaults to <model_dir>/coverage.png).

        Parameters
        ----------
        catalog : grid name (resolved under model_root) or a model directory.
        model_root : base stellar_models dir for resolving a bare name.
        plot : whether to write the coverage figure.
        out_path : plot output path.

        Returns the summary dict.
        """
        from ..grids.coverage import grid_coverage

        base = Path(model_root) if model_root else cls._model_root
        return grid_coverage(catalog, base_dir=base, plot=plot, out_path=out_path)

    @classmethod
    def import_grid(
        cls,
        src: Union[str, Path],
        name: Optional[str] = None,
        model_root: Optional[Union[str, Path]] = None,
        move: bool = False,
        build_cube: bool = True,
        build_h5: bool = True,
        dry_run: bool = False,
        flux_unit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a local grid of .txt spectra into the SED_Tools pipeline.

        The spectra must carry their parameters in the file header in any form
        recognised by header_parser.parse_header (Teff, logg, [M/H]); if a grid
        uses a header key the parser does not know, add it to ALIASES in
        header_parser.py rather than mapping it here.

        Files are copied (or moved with move=True) into
        <model_root>/<name>/, then run through the standard
        clean -> lookup -> HDF5 -> flux-cube pipeline, after which the grid is
        usable via SED.local(name).

        Parameters
        ----------
        src : a directory of .txt spectra, or a single .txt file.
        name : model name (defaults to the source folder name).
        model_root : base stellar_models dir (defaults to the configured one).
        move : move files instead of copying.
        build_cube : build flux_cube.bin.
        build_h5 : build the HDF5 bundle.
        dry_run : only report how many files have parseable headers; do not copy.
        flux_unit : the spectra's flux unit for cleaning (see
            spectra_cleaner.clean_model_dir); required when their headers do
            not declare one, and never used in place of a declared unit.

        Returns a summary dict.  A spectrum that cleaning leaves unusable
        stops the import (clean_model_dir's RuntimeError), and so does a Teff,
        logg or [M/H] that only some headers state (regenerate_lookup_table's
        ValueError); one that no header states (log g of a blackbody grid) gets
        no column. If the spectra do not declare their flux unit
        and flux_unit is not given, ValueError names the best guess before any
        file is copied or moved; so does a flux_unit that is not a known name.
        """
        files, src_dir = _import_sources(Path(src).expanduser())
        if name is None:
            name = src_dir.name
        base = Path(model_root) if model_root else cls._model_root
        dest = Path(base) / name

        missing = _report_import_headers(files, src_dir)
        n_ok = len(files) - len(missing)
        if dry_run:
            return {
                "name": name,
                "n_files": len(files),
                "n_parseable": n_ok,
                "n_missing": len(missing),
                "dest": str(dest),
                "dry_run": True,
            }

        from ..grids.clean import check_flux_unit, clean_model_dir

        check_flux_unit(str(src_dir), [str(f) for f in files], flux_unit)

        # --- stage files into the data dir ---
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = dest / f.name
            if move:
                shutil.move(str(f), str(target))
            else:
                shutil.copy2(str(f), str(target))
        print(f"[import] {'moved' if move else 'copied'} {len(files)} files -> {dest}")

        # --- standard pipeline: clean -> lookup -> h5 -> cube ---
        from ..grids.lookup import regenerate_lookup_table

        summary = clean_model_dir(str(dest), try_h5_recovery=True, rebuild_lookup=True, flux_unit=flux_unit)
        print(f"[import] cleaned: total={summary['total']}")

        regenerate_lookup_table(str(dest))

        if build_h5:
            from .. import build_h5_bundle_from_txt
            build_h5_bundle_from_txt(str(dest), str(dest / f"{name}.h5"))  # dest holds the .txt files copied above

        if build_cube:
            _build_flux_cube(dest)

        same_root = Path(base).expanduser().resolve() == Path(cls._model_root).expanduser().resolve()
        root_arg = "" if same_root else f", model_root={str(base)!r}"
        print(f"[import] done. Load with: SED.local('{name}'{root_arg})")
        return {
            "name": name,
            "n_files": len(files),
            "n_parseable": n_ok,
            "dest": str(dest),
            "dry_run": False,
        }

    @classmethod
    def combine(
        cls,
        catalogs: List[str],
        output: str = "combined_models",
        model_root: Optional[Union[str, Path]] = None,
        visualization: bool = True,
    ) -> 'SED':
        """
        Combine multiple stellar spectral grids into a unified ensemble.
        
        Every grid named in ``catalogs`` is combined, including an earlier
        combine output; only discovery of "all" grids leaves those out
        (see :mod:`sed_tools.combine_stellar_atm`).
        
        Parameters
        ----------
        catalogs : list of str
            Names of spectral grids to combine
        output : str
            Name for the combined spectral grid
        model_root : str or Path, optional
            Base directory for spectral grids
        visualization : bool
            Whether to generate parameter space visualization
            
        Returns
        -------
        SED
            SED instance with the combined spectral grid
        """
        base_dir = Path(model_root) if model_root else cls._model_root

        selected_models = []
        for cat in catalogs:
            cat_path = base_dir / cat
            if not cat_path.exists():
                raise FileNotFoundError(f"SpectralGrid directory not found: {cat_path}")
            selected_models.append((cat, str(cat_path)))

        print(f"\nCombining {len(selected_models)} spectral grids:")
        for name, _ in selected_models:
            print(f"  - {name}")

        output_dir = str(base_dir / output)
        _build_combined_grid(selected_models, output_dir, visualization=visualization)

        print(f"\nSuccessfully combined {len(selected_models)} spectral grids!")
        print(f"Output: {output_dir}")
        
        # Load the result
        return cls.local(output, model_root=base_dir)
    
    @classmethod
    def ml_completer(
        cls,
        model_root: Optional[Union[str, Path]] = None,
        models_dir: str = "models",
    ) -> 'MLCompleter':
        """
        Get the ML SED completer for extending incomplete grids.
        
        Returns
        -------
        MLCompleter
            ML completer instance with train() and extend() methods
        """
        base_dir = Path(model_root) if model_root else cls._model_root
        return MLCompleter(base_dir=base_dir, models_dir=models_dir)
    
    @classmethod
    def ml_generator(
        cls,
        model_root: Optional[Union[str, Path]] = None,
        models_dir: str = "models",
    ) -> 'MLGenerator':
        """
        Get the ML SED generator for creating SEDs from stellar parameters.
        
        Unlike the completer (which extends existing spectra), the generator
        creates complete SEDs from scratch using only Teff, logg, and [M/H].
        
        Returns
        -------
        MLGenerator
            ML generator instance with train() and generate() methods
            
        Example
        -------
        >>> generator = SED.ml_generator()
        >>> generator.train(grid='Kurucz2003all', epochs=200)
        >>> wl, flux = generator.generate(teff=5777, logg=4.44, metallicity=0.0)
        """
        base_dir = Path(model_root) if model_root else cls._model_root
        return MLGenerator(base_dir=base_dir, models_dir=models_dir)
    
    # =========================================================================
    # Instance Methods - Working with loaded data
    # =========================================================================
    
    def __call__(
        self,
        teff: Number,
        logg: Number,
        metallicity: Number,
    ) -> EvaluatedSED:
        """
        Interpolate a spectrum at the given stellar parameters.
        
        Parameters
        ----------
        teff : float
            Effective temperature (K)
        logg : float
            Surface gravity (log g)
        metallicity : float
            Metallicity [M/H]
            
        Returns
        -------
        EvaluatedSED
            Interpolated spectrum with wavelength and flux arrays
        """
        if self._model is None:
            raise RuntimeError(
                "No spectral grid loaded for interpolation. "
                "Use SED.local() to load a spectral grid with a flux cube."
            )
        return self._model(teff, logg, metallicity)
    
    def model(self, name: Optional[str] = None) -> SEDModel:
        """Get or set the active SEDModel."""
        if name is not None:
            self._model = self._core.model(name)
        if self._model is None:
            raise RuntimeError("No model loaded")
        return self._model
    
    def available_grids(self) -> List[str]:
        """List locally available spectral grids."""
        return self._core.available_grids()

    available_models = available_grids
    
    def find_model(
        self,
        teff: Number,
        logg: Number,
        metallicity: Number,
        limit: Optional[int] = None,
    ) -> List[ModelMatch]:
        """Find spectral grids covering a specific point in parameter space."""
        return self._core.find_model(teff, logg, metallicity, limit=limit)
    
    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get the parameter ranges of the active model."""
        if self._model is None:
            raise RuntimeError("No model loaded")
        return self._model.parameter_ranges()


# =============================================================================
# ML Completer API
# =============================================================================

def _training_grid(base_dir: Path, grid: str) -> Path:
    """The folder of a grid to train on; FileNotFoundError, before anything is written, without its flux cube."""
    grid_dir = Path(base_dir) / grid
    if not (grid_dir / "flux_cube.bin").is_file():
        raise FileNotFoundError(f"No flux_cube.bin found in {grid_dir}. Run 'sed-tools rebuild' first.")
    return grid_dir


class MLCompleter:
    """
    ML-based SED completion for extending incomplete stellar grids.
    
    Usage::
    
        completer = SED.ml_completer()
        completer.train(grid='combined_grid')
        completer.extend('sparse_model', wavelength_range=(100, 100000))
    """
    
    def __init__(self, base_dir: Path, models_dir: str = "models"):
        self.base_dir = base_dir
        self.models_dir = models_dir
        self._model = None
    
    def train(
        self,
        grid: str,
        epochs: int = 100,
        batch_size: int = 32,
        save_name: Optional[str] = None,
    ) -> 'MLCompleter':
        """
        Train the ML model on a stellar spectral grid.
        
        Parameters
        ----------
        grid : str
            Name of the grid to train on
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        save_name : str, optional
            Name for the saved model
        """
        from ..ml.completer import train_model

        grid_dir = _training_grid(self.base_dir, grid)
        save_name = save_name or f"{grid}_completer"
        
        self._model = train_model(
            grid_dir=str(grid_dir),
            save_path=str(self.base_dir / self.models_dir / save_name),
            epochs=epochs,
            batch_size=batch_size,
        )
        
        return self
    
    def extend(
        self,
        catalog: str,
        wavelength_range: Optional[Tuple[float, float]] = None,
        output: Optional[str] = None,
    ) -> SpectralGrid:
        """
        Extend an incomplete spectral grid using the trained model.
        
        Parameters
        ----------
        catalog : str
            Name of the spectral grid to extend
        wavelength_range : tuple, optional
            Target wavelength range (min, max) in Angstroms
        output : str, optional
            Name for the extended output
            
        Returns
        -------
        SpectralGrid
            Extended spectral grid
        """
        from ..ml.completer import extend_catalog
        
        if self._model is None:
            raise RuntimeError("No model trained. Call train() first.")
        
        output = output or f"{catalog}_extended"

        summary = extend_catalog(
            model=self._model,
            catalog_dir=str(self.base_dir / catalog),
            output_dir=str(self.base_dir / output),
            wavelength_range=wavelength_range,
        )
        # extend_catalog itself raises RuntimeError listing any spectrum it could not extend.
        if not summary['total']:
            raise FileNotFoundError(f"No spectra to extend in {self.base_dir / catalog}")

        return SED._load_catalog_from_dir(self.base_dir / output, output, 'ml')
    
    def load(self, model_name: str) -> 'MLCompleter':
        """Load a previously trained model."""
        from ..ml.completer import load_model
        
        model_path = self.base_dir / self.models_dir / model_name
        self._model = load_model(str(model_path))
        return self


# =============================================================================
# ML Generator API
# =============================================================================

class MLGenerator:
    """
    ML-based SED generator for creating complete SEDs from stellar parameters.
    
    Unlike the completer (which extends existing spectra), the generator
    creates complete SEDs from scratch using only Teff, logg, and [M/H].
    
    Usage::
    
        generator = SED.ml_generator()
        generator.train(grid='Kurucz2003all', epochs=200)
        wl, flux = generator.generate(teff=5777, logg=4.44, metallicity=0.0)
        
    Or load a pre-trained model::
    
        generator = SED.ml_generator()
        generator.load('sed_generator_Kurucz2003all')
        wl, flux = generator.generate(teff=5777, logg=4.44, metallicity=0.0)
    """
    
    def __init__(self, base_dir: Path, models_dir: str = "models"):
        self.base_dir = base_dir
        self.models_dir = models_dir
        self._generator = None
        self._model_path: Optional[Path] = None  # set by train() and load()
    
    def train(
        self,
        grid: str,
        epochs: int = 200,
        batch_size: int = 64,
        save_name: Optional[str] = None,
        max_samples: int = 10000,
    ) -> 'MLGenerator':
        """
        Train the ML generator on a stellar spectral grid.
        
        Parameters
        ----------
        grid : str
            Name of the grid to train on (must have flux_cube.bin)
        epochs : int
            Number of training epochs (default: 200)
        batch_size : int
            Batch size for training (default: 64)
        save_name : str, optional
            Name for the saved model. Default: '{grid}_generator'
        max_samples : int
            Maximum training samples to use (default: 10000)
            
        Returns
        -------
        MLGenerator
            Self for method chaining
        """
        from ..ml.generator import SEDGenerator

        grid_dir = _training_grid(self.base_dir, grid)
        save_name = save_name or f"sed_generator_{grid}"
        save_path = self.base_dir / self.models_dir / save_name

        self._generator = SEDGenerator()
        self._generator.train(
            library_path=str(grid_dir),
            output_path=str(save_path),
            epochs=epochs,
            batch_size=batch_size,
            max_samples=max_samples,
        )
        self._model_path = save_path

        return self
    
    def load(self, model_name: str) -> 'MLGenerator':
        """
        Load a previously trained generator model.
        
        Parameters
        ----------
        model_name : str
            Name of the saved model directory
            
        Returns
        -------
        MLGenerator
            Self for method chaining
        """
        from ..ml.generator import SEDGenerator
        
        model_path = self.base_dir / self.models_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self._generator = SEDGenerator(str(model_path))
        self._model_path = model_path
        return self
    
    def generate(
        self,
        teff: float,
        logg: float,
        metallicity: float,
        check_bounds: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete SED from stellar parameters.
        
        Parameters
        ----------
        teff : float
            Effective temperature (K)
        logg : float
            Surface gravity (log cm/s²)
        metallicity : float
            Metallicity [M/H]
        check_bounds : bool
            Warn if parameters are outside training range (default: True)
            
        Returns
        -------
        wavelength : np.ndarray
            Wavelength array in Angstroms
        flux : np.ndarray
            Flux array in erg/s/cm²/Å
        """
        if self._generator is None:
            raise RuntimeError(
                "No model loaded. Call train() or load() first."
            )
        
        return self._generator.generate(
            teff=teff,
            logg=logg,
            meta=metallicity,
            check_bounds=check_bounds,
        )
    
    def generate_with_outputs(
        self,
        teff: float,
        logg: float,
        metallicity: float,
        output_dir: Optional[str] = None,
        check_bounds: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete SED and save data file plus diagnostic figures.
        
        Creates:
            - sed_T{teff}_g{logg}_m{meta}.txt - The SED data
            - sed_T{teff}_g{logg}_m{meta}_spectrum.png - SED plot
            - sed_T{teff}_g{logg}_m{meta}_params.png - Parameter space plot
        
        Parameters
        ----------
        teff : float
            Effective temperature (K)
        logg : float
            Surface gravity (log cm/s²)
        metallicity : float
            Metallicity [M/H]
        output_dir : str, optional
            Directory to save outputs. Default: model_dir/SED/
        check_bounds : bool
            Warn if parameters are outside training range (default: True)
            
        Returns
        -------
        wavelength : np.ndarray
            Wavelength array in Angstroms
        flux : np.ndarray
            Flux array in erg/s/cm²/Å
        """
        if self._generator is None:
            raise RuntimeError(
                "No model loaded. Call train() or load() first."
            )
        
        if output_dir is None:
            output_dir = str(self._model_path / 'SED')  # train() and load() set both
        
        return self._generator.generate_with_outputs(
            teff=teff,
            logg=logg,
            meta=metallicity,
            output_dir=output_dir,
            check_bounds=check_bounds,
        )
    
    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the parameter ranges the model was trained on.
        
        Returns
        -------
        dict
            Dictionary with 'teff', 'logg', 'metallicity' keys
            and (min, max) tuple values
        """
        if self._generator is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")
        
        ranges = self._generator.config['parameter_ranges']
        return {
            'teff': tuple(ranges['teff']),
            'logg': tuple(ranges['logg']),
            'metallicity': tuple(ranges['meta']),
        }
    
    @staticmethod
    def list_models(models_dir: str = "models") -> List[Dict[str, Any]]:
        """
        List available trained generator models.
        
        Parameters
        ----------
        models_dir : str
            Directory containing trained models
            
        Returns
        -------
        list of dict
            List of model info dictionaries with 'name', 'path', 
            'parameter_ranges', and 'architecture' keys
        """
        from ..ml.generator import SEDGenerator
        return SEDGenerator.list_models(models_dir)


# =============================================================================
# Convenience exports
# =============================================================================

def query(*args, **kwargs) -> List[SpectralGridInfo]:
    """Alias for SED.query()"""
    return SED.query(*args, **kwargs)

def fetch(*args, **kwargs) -> SED:
    """Alias for SED.fetch()"""
    return SED.fetch(*args, **kwargs)

def local(*args, **kwargs) -> SED:
    """Alias for SED.local()"""
    return SED.local(*args, **kwargs)
