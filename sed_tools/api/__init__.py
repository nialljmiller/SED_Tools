"""The Python API for spectral grids: SED (query, fetch, local, import, combine, ML), SpectralGrid,
Spectrum, SpectralGridInfo, MLCompleter and MLGenerator; ``help()`` on each shows how to use it.

Filters is defined in api/filters.py and re-exported here; the download and build steps are in
grids/pipeline.py and cube/check.py.
"""

from __future__ import annotations

import csv  # noqa: F401  (kept: `from sed_tools.api import *` gives what 0.5.0 gave)
import logging
import math  # noqa: F401  (kept, as csv)
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union  # noqa: F401  (Sequence kept)

import numpy as np
import pandas as pd
from requests import RequestException

from ..grids.lookup import (find_file_column, find_logg_column, find_metallicity_column, find_teff_column,
                           regenerate_lookup_table)
from ..grids.spectrum_io import build_h5_bundle
from ..cube.model import (
    FILTER_DIR_DEFAULT,
    STELLAR_DIR_DEFAULT,
    EvaluatedSED,
    ModelMatch,
    SEDModel,
    SED as _SEDCore,
    _read_flux_cube_header,
)
from ..cube.check import _bundle_shortfall, _flux_cube_file, _flux_cube_shortfall  # noqa: F401
from ..grids.pipeline import (  # noqa: F401  (_build_combined_grid and _parameter_cuts are re-exported)
    _build_combined_grid, _build_data_products, _build_flux_cube, _directory_state, _download_count,
    _import_sources, _parameter_cuts, _report_import_headers, download_from_first_source,
    spectra_grabber,
)
from .filters import Filters, _listing_failures  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

Number = Union[int, float, np.floating]

# Remote spectral grid sources, in the order SED.fetch tries them.
_SPECTRA_SOURCES = ("njm", "svo", "msg", "mast")

# The sources SED.fetch can ask for only the spectra within its Teff, log g and [M/H] cuts. (MSG
# downloads its grid's one HDF5 file whole and extracts the spectra within them; SVO serves whole grids.)
_CUTTING_SOURCES = ("njm", "msg")

# How SED.query describes the grids each remote source lists.
_LISTED_AS = {"svo": {"url": "http://svo2.cab.inta-csic.es/theory/newov2/"}, "msg": {"description": "MSG HDF5 format"},
              "mast": {"description": "BOSZ spectral grids"}, "njm": {"description": "NJM pre-processed mirror"}}

# What a remote source raises when it is unreachable or serves something unexpected.
_SOURCE_ERRORS = (RequestException, OSError, ValueError, RuntimeError)


# ------------ Results ------------

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
        for value, bounds in ((teff, self.teff_range), (logg, self.logg_range),
                              (metallicity, self.metallicity_range)):
            if value is not None and bounds and not (bounds[0] <= value <= bounds[1]):
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
        for low, high, bounds in ((teff_min, teff_max, self.teff_range), (logg_min, logg_max, self.logg_range),
                                  (metallicity_min, metallicity_max, self.metallicity_range)):
            if bounds and ((low is not None and low > bounds[1]) or (high is not None and high < bounds[0])):
                return False
        return True
    
    def __repr__(self) -> str:
        parts = [f"SpectralGridInfo('{self.name}', source='{self.source}'"]
        parts += [f"{label}={value}" for label, value in
                  (("teff", self.teff_range), ("logg", self.logg_range), ("n_spectra", self.n_spectra)) if value]
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
        return {name: getattr(self, name) for name in ('wavelength', 'flux', 'teff', 'logg', 'metallicity', 'filename')}
    
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


def _not_outside(value: float, low: Optional[float], high: Optional[float]) -> bool:
    """False only for a value below ``low`` or above ``high``; a NaN value is never outside."""
    return not ((low is not None and value < low) or (high is not None and value > high))


@dataclass  
class SpectralGrid:
    """A spectral grid of model spectra and their parameters: what SED.fetch() returns in ``sed.grid``."""
    
    name: str
    source: str
    spectra: List[Spectrum] = field(default_factory=list)
    _base_dir: Path = field(default_factory=lambda: STELLAR_DIR_DEFAULT)
    _parameters_df: Optional[pd.DataFrame] = None
    
    @property
    def parameters(self) -> pd.DataFrame:
        """Return a DataFrame of all spectrum parameters."""
        if self._parameters_df is None:
            self._parameters_df = pd.DataFrame([
                {'file_name': spec.filename or f"teff{spec.teff}_logg{spec.logg}_meta{spec.metallicity}.txt",
                 'teff': spec.teff, 'logg': spec.logg, 'metallicity': spec.metallicity}
                for spec in self.spectra])
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
        filtered = [spec for spec in self.spectra
                    if _not_outside(spec.teff, teff_min, teff_max) and _not_outside(spec.logg, logg_min, logg_max)
                    and _not_outside(spec.metallicity, metallicity_min, metallicity_max)]
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
        """Write the spectral grid to disk in the standard SED_Tools structure.

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
        path = self._base_dir / self.name if path is None else Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for spec in self.spectra:
            fname = spec.filename or f"teff{spec.teff:.0f}_logg{spec.logg:.2f}_meta{spec.metallicity:.2f}.txt"
            spec.save(path / fname)
        
        self.parameters.to_csv(path / "lookup_table.csv", index=False)
        print(f"[write] Wrote {len(self.spectra)} spectra to {path}")

        if build_flux_cube and len(self.spectra) > 0:
            written = _build_flux_cube(path)
            if written.name == "flux_cube.bin":
                print("[write] Built flux_cube.bin")
            else:
                print(f"[write] Built one flux cube per extra-axis variant, listed in {written}")

        if build_h5 and len(self.spectra) > 0:
            if not build_h5_bundle(str(path), str(path / f"{self.name}.h5")):
                raise RuntimeError(f"No .txt spectra in {path} to bundle into {self.name}.h5")
            print(f"[write] Built {self.name}.h5")

        return path


# The older names of the two classes.
Catalog = SpectralGrid
CatalogInfo = SpectralGridInfo


def _read_listed_spectra(lookup_path: Path, grid_name: str) -> List[Spectrum]:
    """The spectra a lookup table lists whose file exists beside it; ValueError names one that cannot be read."""
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


class SED:
    """Spectral grids: class methods to find, download and build them, instances to interpolate a loaded one.

    Class methods:
        SED.query() - List available spectral grids
        SED.fetch() - Download and return a spectral grid
        SED.local() - Load a local spectral grid
        SED.import_grid() - Ingest a local grid of .txt spectra
        SED.combine() - Create ensemble grids
        SED.coverage() - Report a local grid's parameter coverage
        SED.ml_completer() - Access ML completion tools
        SED.ml_generator() - Access ML generation tools

    Instance methods (loaded data):
        sed(teff, logg, metallicity) - Interpolate a spectrum
        sed.model() - Access the underlying SEDModel

    Examples
    --------
    >>> grids = SED.query(teff_min=5000)
    >>> sed = SED.fetch('Kurucz2003all', teff_min=4000, teff_max=8000)
    >>> sed.grid.write()
    >>> sed = SED.local('Kurucz2003all')
    >>> spectrum = sed(5777, 4.44, 0.0)
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
        """Usually made by SED.fetch() or SED.local(), not directly."""
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

    # ------------ Discovery and fetching ------------

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
        """List the available spectral grids.

        Parameters
        ----------
        source : str, optional
            Filter by source: 'svo', 'msg', 'mast', 'njm', 'local', or 'all'
        teff_min, teff_max, logg_min, logg_max, metallicity_min, metallicity_max : float, optional
            Keep the spectral grids that overlap these ranges (a bound of 0 is a bound)
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

        bounds = (teff_min, teff_max, logg_min, logg_max, metallicity_min, metallicity_max)
        if any(b is not None for b in bounds):
            catalogs = [c for c in catalogs if c.covers_range(*bounds)]
        return catalogs
    
    @classmethod
    def _query_local(cls) -> List[SpectralGridInfo]:
        """Installed grids with a flux_cube.bin, with their ranges from its header."""
        catalogs = []
        if not cls._model_root.exists():
            return catalogs
        for entry in sorted(cls._model_root.iterdir()):
            cube = entry / "flux_cube.bin"
            lookup = entry / "lookup_table.csv"
            if not entry.is_dir() or not cube.is_file():
                continue
            info = SpectralGridInfo(name=entry.name, source='local', is_local=True)
            try:
                meta = _read_flux_cube_header(cube)
                for attr, key in (("teff_range", "teff"), ("logg_range", "logg"), ("metallicity_range", "meta"),
                                  ("wavelength_range", "wavelengths")):
                    setattr(info, attr, (float(meta[key][0]), float(meta[key][-1])))
            except (OSError, ValueError, IndexError) as exc:
                raise ValueError(f"Cannot read the flux cube header of local grid '{entry.name}' ({cube}): "
                                 f"{exc}. Rebuild it with: sed-tools rebuild --grids {entry.name}") from exc

            if lookup.exists():
                try:
                    info.n_spectra = len(pd.read_csv(lookup))
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Cannot read {lookup} of local grid '{entry.name}': {exc}") from exc

            catalogs.append(info)
        return catalogs
    
    @classmethod
    def _query_remote(cls, source: Optional[str]) -> List[SpectralGridInfo]:
        """Grids from each requested remote source; an unreachable NJM is left out, or raises if asked for alone."""
        catalogs = []
        for name in ('svo', 'msg', 'mast'):
            if source in (None, 'all', name):
                catalogs.extend(cls._query_source(name))
        if source in (None, 'all', 'njm'):
            try:
                catalogs.extend(cls._query_njm())
            except RuntimeError:
                if source == 'njm':
                    raise
        return catalogs

    @classmethod
    def _query_source(cls, source: str) -> List[SpectralGridInfo]:
        """The grids SVO, MSG or MAST lists."""
        grabber = spectra_grabber(source, str(cls._model_root))
        return [SpectralGridInfo(name=m, source=source, **_LISTED_AS[source]) for m in grabber.discover_models()]

    @classmethod
    def _query_njm(cls) -> List[SpectralGridInfo]:
        """The grids NJM lists; RuntimeError if it is unreachable, without a second warning (its client warned)."""
        grabber = spectra_grabber('njm', str(cls._model_root))
        if not grabber.is_available():
            raise RuntimeError(f"NJM mirror {grabber.base_url} is unreachable")
        return [SpectralGridInfo(name=m, source='njm', **_LISTED_AS['njm']) for m in grabber.discover_models()]

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
        """Download a stellar spectral grid and return it loaded.

        Parameters
        ----------
        catalog : str
            Name of the spectral grid to fetch (e.g., 'Kurucz2003all')
        source : str, optional
            Force a specific source: 'svo', 'msg', 'mast', 'njm'.
            If None, tries NJM first (pre-processed), then others
        teff_min, teff_max, logg_min, logg_max, metallicity_min, metallicity_max : float, optional
            Download and return only spectra within these ranges (see Notes)
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
        With any bound (a bound of 0 is a bound; a single bound leaves the
        other side open), only the spectra within the bounds are downloaded,
        as ``sed-tools grids --teff ... --logg ... --meta ...`` does, and the
        lookup table, HDF5 bundle and flux cube in ``model_root/catalog`` are
        built from them, so ``SED.local`` and MESA use that selection. NJM's
        flux cube and HDF5 bundle of the whole grid are not downloaded. MSG
        downloads its grid's HDF5 file whole and extracts the selection; SVO
        cannot select, so the whole grid is downloaded, with a warning, then cut.

        A folder that already holds every selected spectrum, with a flux cube
        holding every spectrum there, is cut as it stands: nothing is
        downloaded or rewritten. A later fetch with no or wider bounds
        downloads the rest, and a flux cube that then lacks spectra beside it
        is rebuilt from all of them, with the lookup table and HDF5 bundle.
        The cube's and lookup table's contents decide this, not file times.
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
            # Cleaning would rewrite the lookup table of a complete installed grid, so it is cut as it stands.
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
                # As in sed-tools grids: a selection gets products of its own, and a cube lacking
                # spectra beside it is rebuilt from all of them.
                if shortfall:
                    print(f"[fetch] {cube.name} {shortfall}:\n        rebuilding the lookup table, HDF5 "
                          "bundle and flux cube from all of them")
                _build_data_products(catalog, str(model_dir), used_source, True, True)

        cat = cls._load_catalog_from_dir(model_dir, catalog, used_source)
        if cuts:
            cat = cat.filter(teff_min=teff_min, teff_max=teff_max, logg_min=logg_min, logg_max=logg_max,
                             metallicity_min=metallicity_min, metallicity_max=metallicity_max)
            print(f"[fetch] Filtered to {len(cat)} spectra")
        
        return cls(catalog=cat, model_root=base_dir)

    @staticmethod
    def _clean_fetched(model_dir: Path, flux_unit: Optional[str]) -> None:
        """Standardize a fetched grid's units; RuntimeError (clean_model_dir) for any spectrum left unusable."""
        from ..grids.clean import clean_model_dir
        summary = clean_model_dir(str(model_dir), try_h5_recovery=True, rebuild_lookup=True,
                                  flux_unit=flux_unit)
        print(f"[fetch] Cleaned: {summary['total']} total, {len(summary['converted'])} converted")

    @classmethod
    def _get_sources_for_catalog(cls, catalog: str) -> Tuple[List[str], List[str]]:
        """The sources listing ``catalog`` (NJM first, SVO always as a fallback), and an unreachable NJM's error."""
        sources, failures = [], []
        try:
            if catalog in [info.name for info in cls._query_njm()]:
                sources.append('njm')
        except RuntimeError as exc:
            failures.append(f"njm: discovery failed: {exc}")
        for name in ('svo', 'msg', 'mast'):
            if catalog in [info.name for info in cls._query_source(name)]:
                sources.append(name)

        if 'svo' not in sources:
            sources.append('svo')
        return sources, failures

    @classmethod
    def _download_from_first_source(cls, catalog: str, sources: List[str], failures: List[str],
                                    model_dir: Path, workers: int,
                                    cuts: Optional[Dict[str, Tuple[float, float]]] = None) -> str:
        """The first source that supplies ``catalog`` (pipeline.download_from_first_source); ValueError lists
        every failure."""
        def attempt(src: str) -> Optional[str]:
            print(f"[fetch] Trying {catalog} from {src}...")
            return cls._download_from(src, catalog, model_dir, workers, cuts)

        def warn(src: str, error: str) -> None:
            logger.warning("[fetch] %s: %s, trying next...", src, error)

        used = download_from_first_source(sources, attempt, model_dir, failures, warn)
        if used is None:
            raise ValueError(f"Spectral grid '{catalog}' could not be fetched. Tried: {'; '.join(failures)}")
        return used

    @classmethod
    def _download_from(cls, src: str, catalog: str, model_dir: Path, workers: int,
                       cuts: Optional[Dict[str, Tuple[float, float]]] = None) -> Optional[str]:
        """Download ``catalog`` from one source: None on success (no spectrum within ``cuts`` is one), else the error.

        Only the ``_CUTTING_SOURCES`` take ``cuts``; any other source downloads the whole grid, with a warning.
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
        """A grabber for one fetch; NJM uses its own worker count, and MAST is refused (it asks on a terminal)."""
        if source == 'mast':
            raise RuntimeError("MAST (BOSZ) grids are fetched with 'sed-tools grids --source mast', which asks "
                               "which metallicities to download; SED.fetch cannot answer that prompt")
        if source == 'njm':
            return spectra_grabber('njm', base_dir)
        grabber = spectra_grabber(source, base_dir, max_workers=workers)
        if source == 'msg':
            grabber.discover_models()  # get_model_metadata finds grids in the index this reads
        return grabber

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
        """Load a local stellar spectral grid (it needs its flux_cube.bin).

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

        cat = cls._load_catalog_from_dir(model_dir, catalog, 'local')
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
        """Report parameter-space coverage of a local grid (downloaded or built).

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
        """Ingest a local grid of .txt spectra into the SED_Tools pipeline.

        The spectra must carry Teff, logg and [M/H] in their headers in a form
        header_parser.parse_header recognises; for an unknown header key, add
        it to ALIASES in header_parser.py. Files are copied (or moved with
        move=True) into <model_root>/<name>/, then cleaned, given a lookup
        table, HDF5 bundle and flux cube, so SED.local(name) can load the grid.

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
        no column. If the spectra do not declare their flux unit and flux_unit
        is not given, ValueError names the best guess before any file is
        copied or moved; so does a flux_unit that is not a known name.
        """
        files, src_dir = _import_sources(Path(src).expanduser())
        if name is None:
            name = src_dir.name
        base = Path(model_root) if model_root else cls._model_root
        dest = Path(base) / name

        missing = _report_import_headers(files, src_dir)
        result = {"name": name, "n_files": len(files), "n_parseable": len(files) - len(missing)}
        if dry_run:
            return {**result, "n_missing": len(missing), "dest": str(dest), "dry_run": True}

        from ..grids.clean import check_flux_unit, clean_model_dir

        check_flux_unit(str(src_dir), [str(f) for f in files], flux_unit)

        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            (shutil.move if move else shutil.copy2)(str(f), str(dest / f.name))
        print(f"[import] {'moved' if move else 'copied'} {len(files)} files -> {dest}")

        summary = clean_model_dir(str(dest), try_h5_recovery=True, rebuild_lookup=True, flux_unit=flux_unit)
        print(f"[import] cleaned: total={summary['total']}")

        regenerate_lookup_table(str(dest))

        if build_h5:
            build_h5_bundle(str(dest), str(dest / f"{name}.h5"))  # dest holds the .txt files copied above

        if build_cube:
            _build_flux_cube(dest)

        same_root = Path(base).expanduser().resolve() == Path(cls._model_root).expanduser().resolve()
        root_arg = "" if same_root else f", model_root={str(base)!r}"
        print(f"[import] done. Load with: SED.local('{name}'{root_arg})")
        return {**result, "dest": str(dest), "dry_run": False}

    @classmethod
    def combine(
        cls,
        catalogs: List[str],
        output: str = "combined_models",
        model_root: Optional[Union[str, Path]] = None,
        visualization: bool = True,
    ) -> 'SED':
        """Combine stellar spectral grids into one ensemble grid, and load it.

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
        return cls.local(output, model_root=base_dir)
    
    @classmethod
    def ml_completer(
        cls,
        model_root: Optional[Union[str, Path]] = None,
        models_dir: str = "models",
    ) -> 'MLCompleter':
        """The ML SED completer for extending incomplete grids: an MLCompleter with train(), extend() and load()."""
        base_dir = Path(model_root) if model_root else cls._model_root
        return MLCompleter(base_dir=base_dir, models_dir=models_dir)
    
    @classmethod
    def ml_generator(
        cls,
        model_root: Optional[Union[str, Path]] = None,
        models_dir: str = "models",
    ) -> 'MLGenerator':
        """The ML SED generator, which makes whole SEDs from Teff, logg and [M/H]: an MLGenerator (see its help)."""
        base_dir = Path(model_root) if model_root else cls._model_root
        return MLGenerator(base_dir=base_dir, models_dir=models_dir)
    
    # ------------ A loaded grid ------------

    def __call__(
        self,
        teff: Number,
        logg: Number,
        metallicity: Number,
    ) -> EvaluatedSED:
        """Interpolate a spectrum at the given stellar parameters.

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


# ------------ ML ------------

def _training_grid(base_dir: Path, grid: str) -> Path:
    """The folder of a grid to train on; FileNotFoundError, before anything is written, without its flux cube."""
    grid_dir = Path(base_dir) / grid
    if not (grid_dir / "flux_cube.bin").is_file():
        raise FileNotFoundError(f"No flux_cube.bin found in {grid_dir}. Run 'sed-tools rebuild' first.")
    return grid_dir


class MLCompleter:
    """ML-based SED completion for extending incomplete stellar grids.

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
        """Train the ML model on a stellar spectral grid; returns self.

        Parameters
        ----------
        grid : str
            Name of the grid to train on (must have flux_cube.bin)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        save_name : str, optional
            Name for the saved model. Default: '{grid}_completer'
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
        """Extend an incomplete spectral grid using the trained model.

        Parameters
        ----------
        catalog : str
            Name of the spectral grid to extend
        wavelength_range : tuple, optional
            Target wavelength range (min, max) in Angstroms
        output : str, optional
            Name for the extended output. Default: '{catalog}_extended'

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


class MLGenerator:
    """ML-based SED generator for creating complete SEDs from stellar parameters.

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
        """Train the ML generator on a stellar spectral grid.

        Parameters
        ----------
        grid : str
            Name of the grid to train on (must have flux_cube.bin)
        epochs : int
            Number of training epochs (default: 200)
        batch_size : int
            Batch size for training (default: 64)
        save_name : str, optional
            Name for the saved model. Default: 'sed_generator_{grid}'
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
        """Load a previously trained generator model (FileNotFoundError if it is not there).

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
        """Generate a complete SED from stellar parameters (RuntimeError before train() or load()).

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
        return self._loaded_generator().generate(
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
        """Generate a complete SED and save its data file and diagnostic figures.

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
        generator = self._loaded_generator()
        if output_dir is None:
            output_dir = str(self._model_path / 'SED')  # train() and load() set both
        return generator.generate_with_outputs(
            teff=teff,
            logg=logg,
            meta=metallicity,
            output_dir=output_dir,
            check_bounds=check_bounds,
        )
    
    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get the parameter ranges the model was trained on.

        Returns
        -------
        dict
            Dictionary with 'teff', 'logg', 'metallicity' keys
            and (min, max) tuple values
        """
        ranges = self._loaded_generator().config['parameter_ranges']
        return {
            'teff': tuple(ranges['teff']),
            'logg': tuple(ranges['logg']),
            'metallicity': tuple(ranges['meta']),
        }
    
    def _loaded_generator(self):
        """The trained or loaded generator; RuntimeError before train() or load()."""
        if self._generator is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")
        return self._generator

    @staticmethod
    def list_models(models_dir: str = "models") -> List[Dict[str, Any]]:
        """List available trained generator models.

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


# ------------ Module-level aliases ------------

def query(*args, **kwargs) -> List[SpectralGridInfo]:
    """Alias for SED.query()"""
    return SED.query(*args, **kwargs)

def fetch(*args, **kwargs) -> SED:
    """Alias for SED.fetch()"""
    return SED.fetch(*args, **kwargs)

def local(*args, **kwargs) -> SED:
    """Alias for SED.local()"""
    return SED.local(*args, **kwargs)
