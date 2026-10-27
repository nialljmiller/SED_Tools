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

import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from requests import RequestException

from ._lookup_io import find_teff_column, find_logg_column, find_metallicity_column, find_file_column
from .models import (
    FILTER_DIR_DEFAULT,
    STELLAR_DIR_DEFAULT,
    EvaluatedSED,
    ModelMatch,
    SEDModel,
    SED as _SEDCore,
)

logger = logging.getLogger(__name__)

Number = Union[int, float, np.floating]

# Remote spectral grid sources, in the order SED.fetch tries them.
_SPECTRA_SOURCES = ("njm", "svo", "msg", "mast")

# What a remote source raises when it is unreachable or serves something unexpected.
_SOURCE_ERRORS = (RequestException, OSError, ValueError, RuntimeError)


def _unusable_after_cleaning(summary: Dict[str, Any]) -> List[str]:
    """Spectra that ``clean_model_dir`` could not make usable: invalid, index-only or failed."""
    return [*summary['skipped_invalid'], *summary['skipped_index'], *summary['error']]


def _directory_state(directory: Path) -> Dict[str, int]:
    """Name and modification time of each entry in ``directory`` ({} if it does not exist)."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {path.name: path.stat().st_mtime_ns for path in directory.iterdir()}


def _build_flux_cube(model_dir: Path) -> Path:
    """Run precompute_flux_cube on ``model_dir`` and return what it wrote.

    That is ``flux_cube.bin``, or ``variants_index.csv`` when a grid with extra
    axes is split into one flux cube per variant.  precompute_flux_cube can
    also stop without writing either (it then prints why), so RuntimeError is
    raised unless one of them was written by this call.
    """
    from .precompute_flux_cube import precompute_flux_cube

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
    from .combine_stellar_atm import (
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
            from . import build_h5_bundle_from_txt
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
    from .header_parser import parse_header

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

        A remote source that cannot be queried is skipped with a warning;
        RuntimeError is raised if every requested remote source fails.
        """
        if source not in (None, 'local', 'all', *_SPECTRA_SOURCES):
            raise ValueError(f"Unknown source: {source}")
        catalogs = []
        if include_local and source in (None, 'local', 'all'):
            catalogs.extend(cls._query_local())
        if include_remote:
            catalogs.extend(cls._query_remote(source))

        # Filter by parameter coverage
        if any([teff_min, teff_max, logg_min, logg_max, metallicity_min, metallicity_max]):
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
                from .models import _read_flux_cube_header
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
        """Grids from each requested remote source; a source that fails is skipped with a warning."""
        queries = {'svo': cls._query_svo, 'msg': cls._query_msg,
                   'mast': cls._query_mast, 'njm': cls._query_njm}
        requested = [name for name in queries if source in (None, 'all', name)]
        catalogs, errors = [], []
        for name in requested:
            try:
                catalogs.extend(queries[name]())
            except _SOURCE_ERRORS as exc:
                errors.append(f"{name}: {exc}")
                logger.warning("[query] %s spectral grid discovery failed: %s", name, exc)
        if requested and len(errors) == len(requested):
            raise RuntimeError("Spectral grid discovery failed for every requested source: "
                               + "; ".join(errors))
        return catalogs

    @classmethod
    def _query_svo(cls) -> List[SpectralGridInfo]:
        """Query SVO remote spectral grids."""
        from .svo_spectra_grabber import SVOSpectraGrabber
        grabber = SVOSpectraGrabber(base_dir=str(cls._model_root))
        return [
            SpectralGridInfo(name=m, source='svo', url="http://svo2.cab.inta-csic.es/theory/newov2/")
            for m in grabber.discover_models()
        ]

    @classmethod
    def _query_msg(cls) -> List[SpectralGridInfo]:
        """Query MSG (Townsend) remote spectral grids."""
        from .msg_spectra_grabber import MSGSpectraGrabber
        grabber = MSGSpectraGrabber(base_dir=str(cls._model_root))
        return [
            SpectralGridInfo(name=m, source='msg', description="MSG HDF5 format")
            for m in grabber.discover_models()
        ]

    @classmethod
    def _query_mast(cls) -> List[SpectralGridInfo]:
        """Query MAST (BOSZ) remote spectral grids."""
        from .mast_spectra_grabber import MASTSpectraGrabber
        grabber = MASTSpectraGrabber(base_dir=str(cls._model_root))
        return [
            SpectralGridInfo(name=m, source='mast', description="BOSZ spectral grids")
            for m in grabber.discover_models()
        ]

    @classmethod
    def _query_njm(cls) -> List[SpectralGridInfo]:
        """Query NJM mirror remote spectral grids; RuntimeError if the mirror is unreachable."""
        from .njm_spectra_grabber import NJMSpectraGrabber
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
            Filter to spectra within this Teff range
        logg_min, logg_max : float, optional
            Filter to spectra within this logg range
        metallicity_min, metallicity_max : float, optional
            Filter to spectra within this metallicity range
        workers : int
            Number of parallel download workers
        clean : bool
            Whether to run unit standardization after download
        model_root : str or Path, optional
            Base directory for spectral grids
            
        Returns
        -------
        SED
            SED instance with the fetched spectral grid in sed.grid

        Raises
        ------
        ValueError
            If no source could supply the grid; the message gives each source's error.
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
        used_source = cls._download_from_first_source(catalog, sources_to_try, failures, model_dir, workers)

        # Clean/standardize units
        if clean:
            from .spectra_cleaner import clean_model_dir
            summary = clean_model_dir(str(model_dir), try_h5_recovery=True, rebuild_lookup=True)
            print(f"[fetch] Cleaned: {summary['total']} total, {len(summary['converted'])} converted")
            unusable = _unusable_after_cleaning(summary)
            if unusable:
                raise RuntimeError(
                    f"{len(unusable)} of {summary['total']} spectra of '{catalog}' in {model_dir} are unusable "
                    f"after cleaning (invalid, index-only or failed; see the [clean] report above): "
                    + ", ".join(unusable))

        # Load spectra into SpectralGrid object
        cat = cls._load_catalog_from_dir(model_dir, catalog, used_source)
        
        # Apply parameter filters
        if any([teff_min, teff_max, logg_min, logg_max, metallicity_min, metallicity_max]):
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
    
    @classmethod
    def _get_sources_for_catalog(cls, catalog: str) -> Tuple[List[str], List[str]]:
        """Sources whose index lists this spectral grid (NJM first), and each discovery failure."""
        queries = {'njm': cls._query_njm, 'svo': cls._query_svo,
                   'msg': cls._query_msg, 'mast': cls._query_mast}
        sources, failures = [], []
        for name, query in queries.items():
            try:
                listed = [info.name for info in query()]
            except _SOURCE_ERRORS as exc:
                failures.append(f"{name}: discovery failed: {exc}")
                logger.warning("[fetch] %s discovery failed: %s; trying the other sources", name, exc)
                continue
            if catalog in listed:
                sources.append(name)

        # Always include SVO as fallback if not already there
        if 'svo' not in sources:
            sources.append('svo')
        return sources, failures

    @classmethod
    def _download_from_first_source(cls, catalog: str, sources: List[str], failures: List[str],
                                    model_dir: Path, workers: int) -> str:
        """Download ``catalog`` into ``model_dir`` from the first source that can supply it.

        Returns that source.  A source that fails is recorded in ``failures``
        with a warning and the next one is tried, unless the failed attempt
        changed ``model_dir``: the next source's spectra would then be mixed
        with its files, so RuntimeError is raised instead.  ValueError lists
        every failure when no source succeeds.
        """
        for src in sources:
            print(f"[fetch] Trying {catalog} from {src}...")
            before = _directory_state(model_dir)
            error = cls._download_from(src, catalog, model_dir, workers)
            if error is None:
                return src
            failures.append(f"{src}: {error}")
            if _directory_state(model_dir) != before:
                raise RuntimeError(
                    f"Fetching '{catalog}' stopped: {src} failed after changing {model_dir}, so another "
                    f"source's spectra would be mixed with its files. Tried: {'; '.join(failures)}")
            logger.warning("[fetch] %s: %s, trying next...", src, error)
        raise ValueError(f"Spectral grid '{catalog}' could not be fetched. Tried: {'; '.join(failures)}")

    @classmethod
    def _download_from(cls, src: str, catalog: str, model_dir: Path, workers: int) -> Optional[str]:
        """Download ``catalog`` from one source; returns None on success, else what went wrong."""
        try:
            grabber = cls._get_grabber(src, str(model_dir.parent), workers)
            metadata = grabber.get_model_metadata(catalog)
            if not metadata:
                return f"no metadata for {catalog}"
            print(f"[fetch] Downloading {catalog} from {src}...")
            n_downloaded = grabber.download_model_spectra(catalog, metadata)
        except _SOURCE_ERRORS as exc:
            return str(exc)
        print(f"[fetch] Downloaded {n_downloaded} spectra")
        return None if n_downloaded else "no spectra downloaded"

    @classmethod
    def _get_grabber(cls, source: str, base_dir: str, workers: int):
        """Get the appropriate grabber instance."""
        if source == 'svo':
            from .svo_spectra_grabber import SVOSpectraGrabber
            return SVOSpectraGrabber(base_dir=base_dir, max_workers=workers)
        elif source == 'msg':
            from .msg_spectra_grabber import MSGSpectraGrabber
            grabber = MSGSpectraGrabber(base_dir=base_dir, max_workers=workers)
            grabber.discover_models()  # get_model_metadata finds grids in the index this reads
            return grabber
        elif source == 'mast':
            # The MAST grabber asks on the terminal which metallicities to download.
            raise RuntimeError("MAST (BOSZ) grids are fetched with 'sed-tools grids --source mast', which asks "
                               "which metallicities to download; SED.fetch cannot answer that prompt")
        elif source == 'njm':
            from .njm_spectra_grabber import NJMSpectraGrabber
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
        from .grid_coverage import grid_coverage

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

        Returns a summary dict.  Files whose headers lack a parameter, or that
        cleaning leaves unusable, are left out of the built grid and then
        listed in a RuntimeError.
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
        from .spectra_cleaner import clean_model_dir
        from .svo_regen_spectra_lookup import regenerate_lookup_table

        summary = clean_model_dir(str(dest), try_h5_recovery=True, rebuild_lookup=True)
        print(f"[import] cleaned: total={summary['total']}")

        regenerate_lookup_table(str(dest))

        if build_h5:
            from . import build_h5_bundle_from_txt
            if not build_h5_bundle_from_txt(str(dest), str(dest / f"{name}.h5")):
                raise RuntimeError(f"No .txt spectra left in {dest} to bundle into {name}.h5")

        if build_cube:
            _build_flux_cube(dest)

        unusable = _unusable_after_cleaning(summary)
        if missing or unusable:
            raise RuntimeError(
                f"Imported '{name}' into {dest} without some spectra. "
                f"Lacking a Teff, logg or [M/H] header ({len(missing)}): {', '.join(missing) or 'none'}. "
                f"Unusable after cleaning ({len(unusable)}): {', '.join(unusable) or 'none'}.")
        print(f"[import] done. Load with: SED.local('{name}')")
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
        from .ml_sed_completer import train_model
        
        grid_dir = self.base_dir / grid
        if not grid_dir.exists():
            raise FileNotFoundError(f"Grid not found: {grid_dir}")
        
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
        from .ml_sed_completer import extend_catalog
        
        if self._model is None:
            raise RuntimeError("No model trained. Call train() first.")
        
        output = output or f"{catalog}_extended"

        summary = extend_catalog(
            model=self._model,
            catalog_dir=str(self.base_dir / catalog),
            output_dir=str(self.base_dir / output),
            wavelength_range=wavelength_range,
        )
        if summary['failed']:
            raise RuntimeError(f"{summary['failed']} of {summary['total']} spectra of '{catalog}' could not be "
                               f"extended; the others were written to {self.base_dir / output}")
        if not summary['total']:
            raise FileNotFoundError(f"No spectra to extend in {self.base_dir / catalog}")

        return SED._load_catalog_from_dir(self.base_dir / output, output, 'ml')
    
    def load(self, model_name: str) -> 'MLCompleter':
        """Load a previously trained model."""
        from .ml_sed_completer import load_model
        
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
        from .ml_sed_generator import SEDGenerator
        
        grid_dir = self.base_dir / grid
        if not grid_dir.exists():
            raise FileNotFoundError(f"Grid not found: {grid_dir}")
        
        flux_cube = grid_dir / "flux_cube.bin"
        if not flux_cube.exists():
            raise FileNotFoundError(
                f"No flux_cube.bin found in {grid_dir}. "
                "Run 'sed-tools rebuild' first."
            )
        
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
        from .ml_sed_generator import SEDGenerator
        
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
            if self._model_path is None:
                raise RuntimeError("Give output_dir: the model was not trained or loaded through this "
                                   "MLGenerator, so its model directory is unknown")
            output_dir = str(self._model_path / 'SED')
        
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
        from .ml_sed_generator import SEDGenerator
        return SEDGenerator.list_models(models_dir)


# =============================================================================
# Filter API
# =============================================================================

class Filters:
    """
    API for working with photometric filter profiles.
    
    Usage::
    
        filters = Filters.query()
        filters = Filters.query(facility='HST')
        Filters.fetch('Generic', 'Johnson')
        Filters.search(wavelength=21500, fwhm=(1000, 5000))   # NJM's catalogue, no SVO
        Filters.info('2MASS', '2MASS'); Filters.properties('2MASS', '2MASS')
        print(Filters.cite('2MASS/2MASS', 'GAIA/GAIA/G'))
        Filters.plot('2MASS/2MASS', normalize='peak', vega=True)
    """
    
    _filter_root: Path = FILTER_DIR_DEFAULT
    
    @classmethod
    def query(
        cls,
        facility: Optional[str] = None,
        include_local: bool = True,
        include_remote: bool = True,
    ) -> List[Dict[str, Any]]:
        """Query available filter profiles."""
        results = []
        
        if include_local:
            results.extend(cls._query_local(facility))
        
        if include_remote:
            results.extend(cls._query_remote(facility))
        
        return results
    
    @classmethod
    def _query_local(cls, facility: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query locally installed filters.

        Empty/incomplete filter directories are ignored. This avoids treating a
        failed partial download, e.g. a directory containing only an index file,
        as a valid local filter set.
        """
        results = []

        if not cls._filter_root.exists():
            return results

        for fac_dir in sorted(cls._filter_root.iterdir()):
            if not fac_dir.is_dir():
                continue
            if facility and fac_dir.name.lower() != facility.lower():
                continue

            for inst_dir in sorted(fac_dir.iterdir()):
                if not inst_dir.is_dir():
                    continue

                filters = sorted(inst_dir.glob("*.dat"))
                if not filters:
                    continue

                results.append({
                    'facility': fac_dir.name,
                    'instrument': inst_dir.name,
                    'n_filters': len(filters),
                    'is_local': True,
                })

        return results

    @classmethod
    def _query_remote(cls, facility: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query remote filter sources.

        SVO is treated as the authoritative discovery catalogue. The NJM mirror
        is also reported when available, but fetch() still validates actual .dat
        files before accepting a download.
        """
        from .filter_library import FilterLibrary

        report = FilterLibrary(cls._filter_root).facilities()
        for error in report["errors"]:
            logger.warning("Filter discovery: %s", error)
        return [
            {"facility": row["label"], "facility_key": row["key"], "source": source, "is_local": False}
            for row in report["entries"]
            if not facility or facility.casefold() in (row["key"].casefold(), row["label"].casefold())
            for source in row["sources"] if source != "local"
        ]

    @classmethod
    def fetch(
        cls,
        facility: str,
        instrument: str,
        filter_root: Optional[Union[str, Path]] = None,
        components: Optional[Union[str, Sequence[str]]] = None,
        atmosphere: Optional[Dict[str, Any]] = None,
        *,
        bands: Optional[Sequence[str]] = None,
        detector_type: Optional[str] = None,
        allow_heavy: bool = False,
        allow_partial: bool = False,
        variants: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Download filter profiles, optionally composing them from components.

        With no ``components``/``atmosphere`` this returns the published curves
        exactly as before (the directory MESA reads).  Otherwise the published
        curves are downloaded, the instrument's components fetched, and a
        product built under ``products/<recipe>/``, whose path is returned.

        Parameters
        ----------
        components : str or list of str, optional
            Any of ``filter``, ``optics``, ``qe``, or ``published`` (the
            published curve as the base), e.g. ``["filter", "optics", "qe"]``
            or ``"filter"`` for the bare bandpass.  See :meth:`available`.
        atmosphere : dict, optional
            Observing conditions, e.g. ``{"airmass": 1.2, "pwv": 2.5}``, to fold
            in the telluric transmission at the instrument's *own* site
            (ground-based instruments with a site only).  Components are never
            taken from another instrument or site.
        bands : sequence of str, optional
            Compose only these bands, useful when components cover part of a set.
        detector_type : str, optional
            Explicit energy/photon integration convention for a derived copy.
            This changes weighting, not the physical QE component.
        allow_heavy : bool
            Permit large original-source downloads such as JWST reference data.
        allow_partial : bool
            Explicitly allow zero transmission outside component coverage.
        variants : dict, optional
            Alternative component curves to use, e.g. ``{"qe": "single_layer"}``
            for a detector coating or epoch; see ``variants`` in
            :meth:`available`.  The choice is part of the product name.
        """
        instrument_dir = cls._fetch_published(facility, instrument, filter_root=filter_root)
        if (components is None and atmosphere is None and detector_type is None and bands is None
                and not variants):
            return instrument_dir
        return cls.compose(instrument_dir, components or ["published"], atmosphere=atmosphere,
                           filter_root=filter_root, bands=bands, detector_type=detector_type,
                           allow_heavy=allow_heavy, allow_partial=allow_partial, variants=variants)

    @classmethod
    def _fetch_published(cls, facility, instrument, filter_root=None) -> Path:
        from .filter_library import FilterLibrary

        return FilterLibrary(filter_root or cls._filter_root).published(facility, instrument)

    @classmethod
    def tag_detector_types(
        cls,
        path: Optional[Union[str, Path]] = None,
        *,
        facility: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Ensure every filter file under ``path`` declares its detector type.

        Each ``.dat`` file gets ``DetectorType=energy|photon`` on its column
        row (plus ``FilterID`` and ``DetectorTypeSource``), resolved from SVO by
        filter ID or by matching the curve against SVO's records. Files with no
        ``DetectorType`` that cannot be identified are written as ``energy``
        with source ``assumed``; files that already have one keep it. An
        existing type is replaced only by a resolution whose source has equal
        or higher precedence (``original`` > ``inferred`` > ``svo`` >
        ``svo-curve-match`` > ``assumed``); a weaker resolution is reported
        (``kept``, ``resolved_type``, ``resolved_source``, ``note``) and not
        written. See :mod:`sed_tools.filter_metadata`.

        Parameters
        ----------
        path : str or Path, optional
            Filter root, a facility/instrument directory, or a single file.
            Defaults to the configured filter root.
        facility : str, optional
            SVO facility name, when ``path`` is a single instrument directory.
        force : bool
            Re-resolve files that are already tagged. The precedence still
            applies, so ``original`` and ``inferred`` types are kept.
        dry_run : bool
            Report what would be written without modifying files.

        Each file is reported as it is tagged.  Files that cannot be tagged are
        left unchanged and listed in a RuntimeError once the others are done;
        an unreachable SVO raises
        :class:`~sed_tools.filter_metadata.SVOUnreachableError` at once.
        """
        from .filter_metadata import (SVOMetadataClient, SVOUnreachableError, iter_filter_files,
                                      tag_filter_file, tag_filter_tree)

        rows: List[Dict[str, Any]] = []

        def record(r) -> None:
            rows.append({
                "path": str(r.path),
                "detector_type": r.detector_type,
                "source": r.source,
                "filter_id": r.filter_id,
                "changed": r.changed,
                "migrated": r.migrated,
                "note": r.note,
                "kept": r.kept,
                "resolved_type": r.resolved_type,
                "resolved_source": r.resolved_source,
            })
            if verbose or r.kept or (r.changed and r.source == "assumed"):
                flag = "*" if r.changed else " "
                note = f"  ({r.note})" if r.note and (r.kept or r.source == "assumed") else ""
                print(f"[filters]{flag} {r.path}: {r.detector_type} [{r.source}]{note}")

        target = Path(path) if path is not None else cls._filter_root
        if facility is None:
            tag_filter_tree(target, force=force, dry_run=dry_run, progress=record)
            return rows

        client = SVOMetadataClient()
        failed = []
        for p in iter_filter_files(target):
            try:
                record(tag_filter_file(p, client=client, facility=facility, force=force, dry_run=dry_run))
            except SVOUnreachableError:
                raise  # every later lookup would fail too
            except (RequestException, ValueError) as exc:
                failed.append(f"{p}: {exc}")
        if failed:
            raise RuntimeError(f"{len(failed)} of {len(failed) + len(rows)} filter files could not be tagged "
                               "and were left unchanged:\n  " + "\n  ".join(failed))
        return rows

    @classmethod
    def available(
        cls,
        facility: str,
        instrument: str,
        filter_root: Optional[Union[str, Path]] = None,
        *,
        include_remote: bool = True,
    ) -> Dict[str, Any]:
        """Describe what can be requested for an instrument.

        Returns the published curve's contents (e.g. ``filter+instrument+atmosphere``),
        the instrument's own separable components, the with/without variants
        that can be built from them, whether (and at which site) a telluric
        component can be added, and the detector type.  Uses the local copy if present,
        otherwise the NJM manifest or SVO metadata. ``include_remote=False``
        inspects installed data and source registration without network access.
        """
        from .filter_library import FilterLibrary

        return FilterLibrary(filter_root or cls._filter_root).available(
            facility, instrument, remote=include_remote)

    @classmethod
    def inventory(cls, *, filter_root=None, include_remote=True, facility=None):
        """List all discovered instruments and source-review status.

        Includes SVO, NJM, local data and registered original-source builders.
        Individual passbands are not downloaded. Discovery errors are returned
        alongside the rows; unreviewed instruments are never called total-only.
        """
        from .filter_library import FilterLibrary

        return FilterLibrary(filter_root or cls._filter_root).inventory(
            remote=include_remote, facility=facility)

    @classmethod
    def atmospheres(cls, filter_root=None, *, include_remote=True):
        """Describe cached and mirrored atmosphere grids, without generating models."""
        from .filter_library import FilterLibrary

        return FilterLibrary(filter_root or cls._filter_root).atmospheres(remote=include_remote)

    @classmethod
    def compose(
        cls,
        instrument: Union[str, Path],
        components: Union[str, Sequence[str]],
        *,
        atmosphere: Optional[Dict[str, Any]] = None,
        filter_root: Optional[Union[str, Path]] = None,
        allow_partial: bool = False,
        bands: Optional[Sequence[str]] = None,
        detector_type: Optional[str] = None,
        allow_heavy: bool = False,
        variants: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Build ``products/<recipe>/`` for a local instrument directory.

        ``instrument`` is a path or ``"Facility/Instrument"`` under the filter
        root.  Raises :class:`sed_tools.filter_components.ComponentError` with
        an explanation when the request cannot be met (component not
        published for this instrument, double counting, space-based
        instrument with an atmosphere, ...).
        """
        from .filter_library import FilterLibrary

        out = FilterLibrary(filter_root or cls._filter_root).compose(
            instrument, components, atmosphere=atmosphere, bands=bands,
            detector_type=detector_type, allow_heavy=allow_heavy, allow_partial=allow_partial,
            variants=variants)
        print(f"[filters] Built {out}")
        return out

    @classmethod
    def combine(
        cls,
        output: Union[str, Path],
        *inputs: Union[str, Path],
        filter_root: Optional[Union[str, Path]] = None,
        facility: Optional[str] = None,
        instrument: Optional[str] = None,
        on_conflict: str = "rename",
    ) -> Path:
        """Combine existing filter sets into one MESA-compatible set.

        This is a convenience wrapper around
        :func:`sed_tools.combine_filters.combine_filter_sets`.
        """
        from .combine_filters import combine_filter_sets

        return combine_filter_sets(
            output,
            inputs,
            filter_root=filter_root or cls._filter_root,
            facility=facility,
            instrument=instrument,
            on_conflict=on_conflict,
        )


    # ------------------------------------------------------------------
    # NJM catalogue: properties, descriptions, search, citations, plots
    # ------------------------------------------------------------------
    _catalogues: Dict[Tuple[str, bool], Any] = {}

    @classmethod
    def _catalogue(cls, filter_root=None, *, offline: bool = False):
        """One catalogue client per filter root, so the mirror is checked once per session."""
        from .filter_catalogue import FilterCatalogue

        key = (str(Path(filter_root or cls._filter_root).expanduser().resolve()), offline)
        if key not in cls._catalogues:
            cls._catalogues[key] = FilterCatalogue(key[0], offline=offline)
        return cls._catalogues[key]

    @classmethod
    def properties(
        cls,
        facility: str,
        instrument: str,
        *,
        filter_root: Optional[Union[str, Path]] = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> Dict[str, Any]:
        """An instrument's ``properties.json`` as NJM serves it.

        ``{"bands": {band: {"WavelengthPivot": ..., "FWHM": ..., "ZeroPointVega_Jy": ...,
        "ABminusVega": ..., "DetectorType": ..., "DisplayColor": ...}}, "definitions": {...}}``,
        computed by :mod:`sed_tools.filter_properties` (units under ``definitions``).

        The copy saved beside installed curves is used first, since it describes
        those curves; otherwise NJM's current file, cached under
        ``<filter root>/../cache/njm/``.  ``refresh=True`` asks NJM again;
        ``offline=True`` uses installed and cached copies only.  Raises
        :class:`sed_tools.filter_catalogue.CatalogueError` when neither has it.
        """
        return cls._catalogue(filter_root, offline=offline).properties(facility, instrument, refresh=refresh)

    @classmethod
    def info(
        cls,
        facility: str,
        instrument: str,
        *,
        filter_root: Optional[Union[str, Path]] = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> Dict[str, Any]:
        """An instrument's ``info.json`` as NJM serves it.

        Descriptions, links, notes, the references to cite (``bibliography``, with
        BibTeX where available), the providers' ``acknowledgements``, the
        ``licence``, and per band the SVO-style fields (``Description``,
        ``PhotSystem``, ``MagSys``, ...), owner-published values and what SVO
        recorded.  Sources are chosen as in :meth:`properties`.
        """
        return cls._catalogue(filter_root, offline=offline).info(facility, instrument, refresh=refresh)

    @classmethod
    def search(
        cls,
        text: Optional[str] = None,
        *,
        facility: Optional[str] = None,
        instrument: Optional[str] = None,
        band: Optional[str] = None,
        wavelength: Any = None,
        pivot: Optional[Sequence[Optional[float]]] = None,
        fwhm: Optional[Sequence[Optional[float]]] = None,
        detector_type: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
        properties: bool = True,
        filter_root: Optional[Union[str, Path]] = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Find bands in NJM's catalogue, one row per band, without SVO.

        Parameters
        ----------
        text : str, optional
            Case-insensitive substring of ``Facility/Instrument/Band``.
        facility, instrument, band : str, optional
            Case-insensitive substrings of each name.
        wavelength : float or (low, high), optional
            Å (or an astropy quantity).  A single value keeps bands that cover it
            (``WavelengthMin <= λ <= WavelengthMax``, where T is at least 1 % of its
            peak); a range keeps bands that overlap it.
        pivot, fwhm : (low, high), optional
            Ranges in Å for ``WavelengthPivot`` and ``FWHM``; ``None`` leaves an end open.
        detector_type : {"energy", "photon"}, optional
        where : dict, optional
            Any other ``properties.json`` column: a ``(low, high)`` range or a value
            to equal (strings ignore case), e.g. ``{"ABminusVega": (None, 0.5)}``.
            A band whose value is missing (Vega quantities where Vega does not
            cover it) never matches a criterion.
        properties : bool
            ``False`` matches names only, from ``index.json``, without fetching
            any ``properties.json``.

        Names are matched in ``index.json`` first, so ``properties.json`` is
        fetched only for instruments that can still match (at most six requests
        at a time; cached).  Returns a DataFrame sorted by facility, instrument and
        pivot, with ``facility``, ``instrument``, ``band`` and the property
        columns; ``.attrs["units"]`` gives units, ``.attrs["errors"]`` lists
        instruments whose properties could not be read, and ``.attrs["gaps"]``
        lists instruments NJM serves no ``properties.json`` for.
        """
        return cls._catalogue(filter_root, offline=offline).search(
            text, facility=facility, instrument=instrument, band=band, wavelength=wavelength,
            pivot=pivot, fwhm=fwhm, detector_type=detector_type, where=where,
            properties=properties, refresh=refresh)

    @classmethod
    def _targets(cls, targets, filter_root) -> List[Tuple[str, str, Optional[List[str]], Optional[Path]]]:
        """Normalise targets to ``(facility, instrument, bands or None, directory or None)``.

        A target is ``"Facility/Instrument"``, ``"Facility/Instrument/Band"``, a
        tuple of those parts, an instrument directory, a ``.dat`` file, or a
        :meth:`search` result.  No targets means every installed instrument.
        """
        from .filter_catalogue import parse_target

        root = Path(filter_root or cls._filter_root)
        items: List[Tuple[str, str, Optional[str], Optional[Path]]] = []
        if not targets:
            for directory in sorted(root.glob("*/*")):
                if directory.is_dir() and any(directory.glob("*.dat")):
                    items.append((directory.parent.name, directory.name, None, directory))
        for target in targets:
            if isinstance(target, pd.DataFrame):
                items.extend((r.facility, r.instrument, r.band, None)
                             for r in target[["facility", "instrument", "band"]].itertuples())
            elif isinstance(target, (tuple, list)):
                fac, inst, *rest = target
                items.append((fac, inst, rest[0] if rest else None, None))
            elif isinstance(target, Path) or (Path(str(target)).exists() and (
                    Path(str(target)).suffix == ".dat" or any(Path(str(target)).glob("*.dat")))):
                path = Path(target)
                if path.is_file():
                    items.append((path.parent.parent.name, path.parent.name, path.stem, path.parent))
                else:
                    items.append((path.parent.name, path.name, None, path))
            else:
                items.append(parse_target(str(target)) + (None,))
        grouped: Dict[Tuple[str, str], List[Any]] = {}
        for fac, inst, band, directory in items:
            entry = grouped.setdefault((fac, inst), [fac, inst, [], directory])
            if band is None:
                entry[2] = None
            elif entry[2] is not None and band not in entry[2]:
                entry[2].append(band)
        return [tuple(v) for v in grouped.values()]

    @classmethod
    def citations(
        cls,
        *targets: Any,
        filter_root: Optional[Union[str, Path]] = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> Dict[str, Any]:
        """What to cite and acknowledge for instruments or bands, from their ``info.json``.

        Targets are as in :meth:`plot`; with none, every installed instrument.
        Returns ``{"instruments", "references", "acknowledgements", "errors"}``:
        references merged across instruments (one per work, following
        ``same_as`` to the bibcode form, each with ``bibtex`` where NJM has it and
        ``used_by``), and the providers' acknowledgement texts, merged.  For a
        subset of bands, the instrument-wide references plus those bands' own.
        """
        from .filter_catalogue import CatalogueError, collect_citations

        catalogue = cls._catalogue(filter_root, offline=offline)
        infos, errors = [], []
        for fac, inst, bands, directory in cls._targets(targets, filter_root):
            try:
                local = directory / "info.json" if directory is not None else None
                if local is not None and local.is_file() and not refresh:
                    info = catalogue._json_object(local.read_text(encoding="utf-8"))
                else:
                    info = catalogue.info(fac, inst, refresh=refresh)
            except (CatalogueError, OSError, ValueError) as exc:
                errors.append(f"{fac}/{inst}: {exc}")
                continue
            if bands is not None:
                unknown = [b for b in bands if b not in info.get("bands", {})]
                errors.extend(f"{fac}/{inst}: no band {b!r} in info.json" for b in unknown)
                bands = [b for b in bands if b not in unknown]
                if not bands:
                    continue
            infos.append((f"{info.get('facility', fac)}/{info.get('instrument', inst)}", info, bands))
        result = collect_citations(infos)
        result["errors"] = errors
        return result

    @classmethod
    def cite(
        cls,
        *targets: Any,
        format: str = "text",
        filter_root: Optional[Union[str, Path]] = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> str:
        """:meth:`citations` as text (``format="text"``: acknowledgements, licences,
        references and BibTeX) or as a ``.bib`` file (``format="bibtex"``).

        References NJM holds no BibTeX for appear in the ``.bib`` output as ``%``
        comments with what is known; nothing is made up.  The text lists any
        target whose ``info.json`` could not be read; the ``.bib`` file cannot,
        so then :class:`~sed_tools.filter_catalogue.CatalogueError` is raised
        (:meth:`citations` returns what was found).
        """
        from .filter_catalogue import CatalogueError, format_bibtex, format_citations

        result = cls.citations(*targets, filter_root=filter_root, refresh=refresh, offline=offline)
        if format == "bibtex":
            if result["errors"]:
                raise CatalogueError("Cannot cite every target: " + "; ".join(result["errors"]))
            return format_bibtex(result)
        if format == "text":
            return format_citations(result)
        raise ValueError("format must be 'text' or 'bibtex'")

    @classmethod
    def plot(
        cls,
        *targets: Any,
        normalize: Union[None, bool, str] = None,
        vega: Any = False,
        ax=None,
        legend: bool = True,
        log_wavelength: Optional[bool] = None,
        filter_root: Optional[Union[str, Path]] = None,
        fetch: bool = True,
    ):
        """Plot filter curves in their :mod:`sed_tools.filter_colors` colours; returns the Axes.

        Targets: ``"Facility/Instrument"``, ``"Facility/Instrument/Band"``, a
        tuple of those parts, an instrument directory or ``.dat`` file, or a
        :meth:`search` result.  Instruments that are not installed are fetched
        (``fetch=False`` refuses instead).  Bands keep their instrument's colours
        (true colours for visible bands, ordered false colours otherwise) when
        they all come from one instrument; a selection across instruments is
        coloured as one set.  ``normalize`` is ``None``, ``"peak"`` or
        ``"area"``.  ``vega=True`` overlays NJM's Vega spectrum
        (``vega/vega_flam.csv``) scaled to the curves; a path or a
        ``(wavelength Å, f_λ)`` pair overlays another spectrum.  Needs matplotlib.
        """
        from .filter_colors import instrument_colors
        from .filter_plot import plot_curves
        from .filter_properties import load_vega
        from .flux_cube_tool import load_filter_curve

        if not targets:
            raise ValueError("nothing to plot: give Facility/Instrument[/Band] targets or a search result")
        catalogue = cls._catalogue(filter_root)
        chosen = cls._plot_bands(targets, filter_root, catalogue, fetch)

        curves, labels = {}, {}
        for fac, inst, band, directory in chosen:
            key = f"{fac}/{inst}/{band}"
            curve = load_filter_curve(str(directory / f"{band}.dat"))
            curves[key] = (curve.wavelength, curve.transmission)
        directories = {directory for *_, directory in chosen}
        colors, title = None, None
        if len(directories) == 1:
            directory = directories.pop()
            served = instrument_colors(sorted(directory.glob("*.dat")))
            colors = {f"{fac}/{inst}/{band}": served[band]["DisplayColor"] for fac, inst, band, _ in chosen}
            labels = {f"{fac}/{inst}/{band}": band for fac, inst, band, _ in chosen}
            title = f"{chosen[0][0]}/{chosen[0][1]}"
        else:
            labels = {f"{fac}/{inst}/{band}": f"{fac}/{inst} {band}" for fac, inst, band, _ in chosen}

        spectrum = None
        if vega is True:
            spectrum = load_vega(catalogue.vega())
        elif isinstance(vega, (str, Path)):
            spectrum = load_vega(vega)
        elif vega is not None and vega is not False:
            spectrum = vega
        return plot_curves(curves, colors=colors, labels=labels, normalize=normalize, vega=spectrum,
                           ax=ax, legend=legend, log_wavelength=log_wavelength, title=title)

    @classmethod
    def _plot_bands(cls, targets, filter_root, catalogue, fetch) -> List[Tuple[str, str, str, Path]]:
        """``(facility, instrument, band, directory)`` for every band the plot targets select."""
        chosen: List[Tuple[str, str, str, Path]] = []
        for fac, inst, bands, directory in cls._targets(targets, filter_root):
            if directory is None:
                fac, inst, directory = cls._installed_or_fetched(fac, inst, filter_root, catalogue, fetch)
            files = {p.stem: p for p in sorted(Path(directory).glob("*.dat"))}
            for band in (files if bands is None else bands):
                if band not in files:
                    raise FileNotFoundError(f"{fac}/{inst} has no band {band!r} in {directory}")
                chosen.append((fac, inst, band, Path(directory)))
        return chosen

    @classmethod
    def _installed_or_fetched(cls, fac, inst, filter_root, catalogue, fetch) -> Tuple[str, str, Path]:
        """``(facility, instrument, directory)`` of an installed instrument (names ignore case),
        or of its published curves fetched under NJM's names for it."""
        from .filter_catalogue import CatalogueError

        root = Path(filter_root or cls._filter_root)
        for d in (root / fac / inst, *root.glob("*/*")):
            if (d.is_dir() and (d.parent.name.casefold(), d.name.casefold()) == (fac.casefold(), inst.casefold())
                    and any(d.glob("*.dat"))):
                return fac, inst, d
        if not fetch:
            raise FileNotFoundError(f"{fac}/{inst} is not installed under {root}")
        try:
            served = catalogue.resolve(fac, inst)
        except CatalogueError as exc:
            logger.warning("%s; fetching %s/%s under the name given", exc, fac, inst)
        else:
            fac, inst = served["facility"], served["instrument"]
        return fac, inst, cls._fetch_published(fac, inst, filter_root=filter_root)


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
