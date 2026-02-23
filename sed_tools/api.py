"""
Python API for SED_Tools - Programmatic interface to stellar atmosphere models.

This module provides a Pythonic interface that mirrors and extends the CLI functionality.
Users can query available catalogs, fetch filtered data, and construct pipelines.

Example usage::

    from sed_tools.api import SED
    
    # Query available remote catalogs
    catalogs = SED.query()
    catalogs = SED.query(teff_min=5000, teff_max=7000)  # Filter by coverage
    
    # Fetch data from a specific catalog
    sed = SED.fetch('Kurucz2003all', teff_min=4000, teff_max=7000, logg_min=3.0)
    
    # Access the catalog data
    print(sed.cat)
    print(sed.cat.spectra)
    print(sed.cat.parameters)
    
    # Save to disk (same structure as CLI)
    sed.cat.write()
    sed.cat.write('/custom/path')
    
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
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Import existing infrastructure
from .models import (
    DATA_DIR_DEFAULT,
    FILTER_DIR_DEFAULT,
    STELLAR_DIR_DEFAULT,
    EvaluatedSED,
    ModelMatch,
    SEDModel,
    SED as _SEDCore,
)

Number = Union[int, float, np.floating]


# =============================================================================
# Data Classes for API Results
# =============================================================================

@dataclass
class CatalogInfo:
    """Information about an available stellar atmosphere catalog."""
    
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
        """Check if this catalog covers a specific point in parameter space."""
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
        """Check if this catalog covers a parameter range (at least partial overlap)."""
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
        parts = [f"CatalogInfo('{self.name}', source='{self.source}'"]
        if self.teff_range:
            parts.append(f"teff={self.teff_range}")
        if self.logg_range:
            parts.append(f"logg={self.logg_range}")
        if self.n_spectra:
            parts.append(f"n_spectra={self.n_spectra}")
        return ", ".join(parts) + ")"


@dataclass
class Spectrum:
    """A single spectrum with wavelength, flux, and metadata."""
    
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
class Catalog:
    """
    A collection of spectra from a stellar atmosphere model.
    
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
        """Unique Teff values in the catalog."""
        return np.unique([s.teff for s in self.spectra])
    
    @property
    def logg_grid(self) -> np.ndarray:
        """Unique logg values in the catalog."""
        return np.unique([s.logg for s in self.spectra])
    
    @property
    def metallicity_grid(self) -> np.ndarray:
        """Unique metallicity values in the catalog."""
        return np.unique([s.metallicity for s in self.spectra])
    
    def filter(
        self,
        teff_min: Optional[float] = None,
        teff_max: Optional[float] = None,
        logg_min: Optional[float] = None,
        logg_max: Optional[float] = None,
        metallicity_min: Optional[float] = None,
        metallicity_max: Optional[float] = None,
    ) -> 'Catalog':
        """Return a new Catalog with filtered spectra."""
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
        
        return Catalog(
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
        Write the catalog to disk in the standard SED_Tools structure.
        
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
            from .precompute_flux_cube import precompute_flux_cube
            try:
                precompute_flux_cube(str(path), str(path / "flux_cube.bin"))
                print(f"[write] Built flux_cube.bin")
            except Exception as e:
                print(f"[write] Warning: flux cube build failed: {e}")
        
        # Build H5 bundle
        if build_h5 and len(self.spectra) > 0:
            from . import build_h5_bundle_from_txt
            try:
                build_h5_bundle_from_txt(str(path), str(path / f"{self.name}.h5"))
                print(f"[write] Built {self.name}.h5")
            except Exception as e:
                print(f"[write] Warning: H5 bundle build failed: {e}")
        
        return path


# =============================================================================
# Main SED API Class
# =============================================================================

class SED:
    """
    Main API class for SED_Tools.
    
    This class provides both class methods for discovery/fetching and instance
    methods for working with loaded data.
    
    Class methods (discovery):
        SED.query() - List available catalogs
        SED.fetch() - Download and return a catalog
        SED.local() - Load a local catalog
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
    >>> catalogs = SED.query()
    >>> catalogs = SED.query(teff_min=5000)
    
    >>> # Fetch remote data
    >>> sed = SED.fetch('Kurucz2003all', teff_min=4000, teff_max=8000)
    >>> sed.cat.write()
    
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
        catalog: Optional[Catalog] = None,
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
    ) -> List[CatalogInfo]:
        """
        Query available stellar atmosphere catalogs.
        
        Parameters
        ----------
        source : str, optional
            Filter by source: 'svo', 'msg', 'mast', 'njm', 'local', or 'all'
        teff_min, teff_max : float, optional
            Filter catalogs that cover this Teff range
        logg_min, logg_max : float, optional
            Filter catalogs that cover this logg range
        metallicity_min, metallicity_max : float, optional
            Filter catalogs that cover this metallicity range
        include_local : bool
            Include locally installed catalogs
        include_remote : bool
            Query remote sources (SVO, MSG, MAST, NJM)
            
        Returns
        -------
        List[CatalogInfo]
            Available catalogs matching the criteria
        """
        catalogs = []
        
        # Local catalogs
        if include_local and (source is None or source == 'local' or source == 'all'):
            catalogs.extend(cls._query_local())
        
        # Remote sources
        if include_remote:
            if source is None or source in ('svo', 'all'):
                catalogs.extend(cls._query_svo())
            if source is None or source in ('msg', 'all'):
                catalogs.extend(cls._query_msg())
            if source is None or source in ('mast', 'all'):
                catalogs.extend(cls._query_mast())
            if source is None or source in ('njm', 'all'):
                catalogs.extend(cls._query_njm())
        
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
    def _query_local(cls) -> List[CatalogInfo]:
        """Query locally installed catalogs."""
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
            
            info = CatalogInfo(
                name=entry.name,
                source='local',
                is_local=True,
            )
            
            # Try to extract parameter ranges from flux cube
            try:
                from .models import _read_flux_cube_header
                meta = _read_flux_cube_header(cube)
                info.teff_range = (float(meta['teff'][0]), float(meta['teff'][-1]))
                info.logg_range = (float(meta['logg'][0]), float(meta['logg'][-1]))
                info.metallicity_range = (float(meta['meta'][0]), float(meta['meta'][-1]))
                info.wavelength_range = (float(meta['wavelengths'][0]), float(meta['wavelengths'][-1]))
            except Exception:
                pass
            
            # Count spectra from lookup table
            if lookup.exists():
                try:
                    df = pd.read_csv(lookup)
                    info.n_spectra = len(df)
                except Exception:
                    pass
            
            catalogs.append(info)
        
        return catalogs
    
    @classmethod
    def _query_svo(cls) -> List[CatalogInfo]:
        """Query SVO remote catalogs."""
        try:
            from .svo_spectra_grabber import SVOSpectraGrabber
            grabber = SVOSpectraGrabber(base_dir=str(cls._model_root))
            models = grabber.discover_models()
            return [
                CatalogInfo(name=m, source='svo', url="http://svo2.cab.inta-csic.es/theory/newov2/")
                for m in models
            ]
        except Exception as e:
            print(f"[query] SVO discovery failed: {e}")
            return []
    
    @classmethod
    def _query_msg(cls) -> List[CatalogInfo]:
        """Query MSG (Townsend) remote catalogs."""
        try:
            from .msg_spectra_grabber import MSGSpectraGrabber
            grabber = MSGSpectraGrabber(base_dir=str(cls._model_root))
            models = grabber.discover_models()
            return [
                CatalogInfo(name=m, source='msg', description="MSG HDF5 format")
                for m in models
            ]
        except Exception as e:
            print(f"[query] MSG discovery failed: {e}")
            return []
    
    @classmethod
    def _query_mast(cls) -> List[CatalogInfo]:
        """Query MAST (BOSZ) remote catalogs."""
        try:
            from .mast_spectra_grabber import MASTSpectraGrabber
            grabber = MASTSpectraGrabber(base_dir=str(cls._model_root))
            models = grabber.discover_models()
            return [
                CatalogInfo(name=m, source='mast', description="BOSZ models")
                for m in models
            ]
        except Exception as e:
            print(f"[query] MAST discovery failed: {e}")
            return []
    
    @classmethod
    def _query_njm(cls) -> List[CatalogInfo]:
        """Query NJM mirror remote catalogs."""
        try:
            from .njm_spectra_grabber import NJMSpectraGrabber
            grabber = NJMSpectraGrabber(base_dir=str(cls._model_root))
            if not grabber.is_available():
                return []
            models = grabber.discover_models()
            return [
                CatalogInfo(name=m, source='njm', description="NJM pre-processed mirror")
                for m in models
            ]
        except Exception as e:
            print(f"[query] NJM discovery failed: {e}")
            return []
    
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
        Fetch and download a stellar atmosphere catalog.
        
        Parameters
        ----------
        catalog : str
            Name of the catalog to fetch (e.g., 'Kurucz2003all')
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
            Base directory for stellar models
            
        Returns
        -------
        SED
            SED instance with the fetched catalog in sed.cat
        """
        base_dir = Path(model_root) if model_root else cls._model_root
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Build list of sources to try
        if source is not None:
            sources_to_try = [source]
        else:
            # Try sources in order of preference
            sources_to_try = cls._get_sources_for_catalog(catalog)
        
        # Try each source until one works
        metadata = None
        grabber = None
        used_source = None
        
        for src in sources_to_try:
            try:
                print(f"[fetch] Trying {catalog} from {src}...")
                grabber = cls._get_grabber(src, str(base_dir), workers)
                metadata = grabber.get_model_metadata(catalog)
                if metadata:
                    used_source = src
                    break
                else:
                    print(f"[fetch] {src}: no metadata returned, trying next...")
            except Exception as e:
                print(f"[fetch] {src} failed: {e}, trying next...")
                continue
        
        if not metadata or grabber is None:
            tried = ", ".join(sources_to_try)
            raise ValueError(f"Catalog '{catalog}' not found. Tried: {tried}")
        
        print(f"[fetch] Downloading {catalog} from {used_source}...")
        n_downloaded = grabber.download_model_spectra(catalog, metadata)
        print(f"[fetch] Downloaded {n_downloaded} spectra")
        
        # Clean/standardize units
        model_dir = base_dir / catalog
        if clean:
            from .spectra_cleaner import clean_model_dir
            summary = clean_model_dir(str(model_dir), try_h5_recovery=True, backup=True, rebuild_lookup=True)
            n_fixed = len(summary.get('fixed', []))
            n_total = summary.get('total', 0)
            print(f"[fetch] Cleaned: {n_total} total, {n_fixed} fixed")
        
        # Load spectra into Catalog object
        cat = cls._load_catalog_from_dir(model_dir, catalog, source)
        
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
    def _get_sources_for_catalog(cls, catalog: str) -> List[str]:
        """Get ordered list of sources that might have this catalog."""
        sources = []
        
        # Try NJM first (pre-processed)
        try:
            from .njm_spectra_grabber import NJMSpectraGrabber
            njm = NJMSpectraGrabber(base_dir=str(cls._model_root))
            if njm.is_available():
                models = njm.discover_models()
                if catalog in models:
                    sources.append('njm')
        except Exception:
            pass
        
        # Try SVO
        try:
            from .svo_spectra_grabber import SVOSpectraGrabber
            svo = SVOSpectraGrabber(base_dir=str(cls._model_root))
            models = svo.discover_models()
            if catalog in models:
                sources.append('svo')
        except Exception:
            pass
        
        # Try MSG
        try:
            from .msg_spectra_grabber import MSGSpectraGrabber
            msg = MSGSpectraGrabber(base_dir=str(cls._model_root))
            models = msg.discover_models()
            if catalog in models:
                sources.append('msg')
        except Exception:
            pass
        
        # Try MAST
        try:
            from .mast_spectra_grabber import MASTSpectraGrabber
            mast = MASTSpectraGrabber(base_dir=str(cls._model_root))
            models = mast.discover_models()
            if catalog in models:
                sources.append('mast')
        except Exception:
            pass
        
        # Always include SVO as fallback if not already there
        if 'svo' not in sources:
            sources.append('svo')
        
        return sources
    
    @classmethod
    def _detect_best_source(cls, catalog: str) -> str:
        """Detect the best source for a catalog (NJM first, then others)."""
        sources = cls._get_sources_for_catalog(catalog)
        return sources[0] if sources else 'svo'
    
    @classmethod
    def _get_grabber(cls, source: str, base_dir: str, workers: int):
        """Get the appropriate grabber instance."""
        if source == 'svo':
            from .svo_spectra_grabber import SVOSpectraGrabber
            return SVOSpectraGrabber(base_dir=base_dir, max_workers=workers)
        elif source == 'msg':
            from .msg_spectra_grabber import MSGSpectraGrabber
            return MSGSpectraGrabber(base_dir=base_dir, max_workers=workers)
        elif source == 'mast':
            from .mast_spectra_grabber import MASTSpectraGrabber
            return MASTSpectraGrabber(base_dir=base_dir, max_workers=workers)
        elif source == 'njm':
            from .njm_spectra_grabber import NJMSpectraGrabber
            return NJMSpectraGrabber(base_dir=base_dir)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    @classmethod
    def _load_catalog_from_dir(cls, model_dir: Path, name: str, source: str) -> Catalog:
        """Load spectra from a local directory into a Catalog."""
        spectra = []
        
        # Read lookup table
        lookup_path = model_dir / "lookup_table.csv"
        if lookup_path.exists():
            df = pd.read_csv(lookup_path)
            
            # Normalize column names
            df.columns = [c.lstrip('#').strip().lower() for c in df.columns]
            
            # Find column mappings
            file_col = None
            for c in ['file_name', 'filename', 'file']:
                if c in df.columns:
                    file_col = c
                    break
            
            teff_col = next((c for c in df.columns if 'teff' in c), None)
            logg_col = next((c for c in df.columns if 'logg' in c or 'log(g)' in c), None)
            meta_col = next((c for c in df.columns if 'meta' in c or 'feh' in c or 'm/h' in c), None)
            
            for _, row in df.iterrows():
                filename = row[file_col] if file_col else None
                if filename and (model_dir / filename).exists():
                    try:
                        wl, fl = np.loadtxt(model_dir / filename, unpack=True, comments='#')
                        spec = Spectrum(
                            wavelength=wl,
                            flux=fl,
                            teff=float(row.get(teff_col, np.nan)) if teff_col else np.nan,
                            logg=float(row.get(logg_col, np.nan)) if logg_col else np.nan,
                            metallicity=float(row.get(meta_col, np.nan)) if meta_col else np.nan,
                            filename=filename,
                            source_catalog=name,
                        )
                        spectra.append(spec)
                    except Exception:
                        pass
        
        return Catalog(
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
        Load a local stellar atmosphere catalog.
        
        Parameters
        ----------
        catalog : str
            Name of the catalog (directory name under model_root)
        model_root : str or Path, optional
            Base directory for stellar models
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
            raise FileNotFoundError(f"Catalog '{catalog}' not found at {model_dir}")
        
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
    def combine(
        cls,
        catalogs: List[str],
        output: str = "combined_models",
        model_root: Optional[Union[str, Path]] = None,
        visualization: bool = True,
    ) -> 'SED':
        """
        Combine multiple stellar atmosphere grids into a unified ensemble.
        
        Parameters
        ----------
        catalogs : list of str
            Names of catalogs to combine
        output : str
            Name for the combined output catalog
        model_root : str or Path, optional
            Base directory for stellar models
        visualization : bool
            Whether to generate parameter space visualization
            
        Returns
        -------
        SED
            SED instance with the combined catalog
        """
        base_dir = Path(model_root) if model_root else cls._model_root
        
        from .combine_stellar_atm import (
            build_combined_flux_cube,
            create_common_wavelength_grid,
            create_unified_grid,
            load_model_data,
            save_combined_data,
            visualize_parameter_space,
        )
        
        # Build list of (name, path) tuples
        selected_models = []
        for cat in catalogs:
            cat_path = base_dir / cat
            if not cat_path.exists():
                raise FileNotFoundError(f"Catalog directory not found: {cat_path}")
            selected_models.append((cat, str(cat_path)))
        
        print(f"\nCombining {len(selected_models)} models:")
        for name, _ in selected_models:
            print(f"  - {name}")
        
        # Load data
        print("\nLoading model data...")
        all_models_data = []
        for name, path in selected_models:
            print(f"  Loading {name}...")
            all_models_data.append(load_model_data(path))
        
        # Create grids
        print("\nCreating unified parameter grids...")
        teff_grid, logg_grid, meta_grid = create_unified_grid(all_models_data)
        wavelength_grid = create_common_wavelength_grid(all_models_data)
        
        # Build cube
        print("\nBuilding combined flux cube...")
        flux_cube, source_map = build_combined_flux_cube(
            all_models_data, teff_grid, logg_grid, meta_grid, wavelength_grid
        )
        
        # Save
        output_dir = str(base_dir / output)
        save_combined_data(
            output_dir,
            teff_grid,
            logg_grid,
            meta_grid,
            wavelength_grid,
            flux_cube,
            all_models_data,
        )
        
        # Visualize
        if visualization:

            visualize_parameter_space(
                teff_grid, logg_grid, meta_grid, source_map, all_models_data, output_dir,
                wavelength_grid=wavelength_grid, flux_cube=flux_cube)
            
        print(f"\nSuccessfully combined {len(selected_models)} models!")
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
                "No model loaded for interpolation. "
                "Use SED.local() to load a catalog with a flux cube."
            )
        return self._model(teff, logg, metallicity)
    
    def model(self, name: Optional[str] = None) -> SEDModel:
        """Get or set the active SEDModel."""
        if name is not None:
            self._model = self._core.model(name)
        if self._model is None:
            raise RuntimeError("No model loaded")
        return self._model
    
    def available_models(self) -> List[str]:
        """List locally available models."""
        return self._core.available_models()
    
    def find_model(
        self,
        teff: Number,
        logg: Number,
        metallicity: Number,
        limit: Optional[int] = None,
    ) -> List[ModelMatch]:
        """Find models covering a specific point in parameter space."""
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
        Train the ML model on a stellar atmosphere grid.
        
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
    ) -> Catalog:
        """
        Extend an incomplete catalog using the trained model.
        
        Parameters
        ----------
        catalog : str
            Name of the catalog to extend
        wavelength_range : tuple, optional
            Target wavelength range (min, max) in Angstroms
        output : str, optional
            Name for the extended output
            
        Returns
        -------
        Catalog
            Extended catalog
        """
        from .ml_sed_completer import extend_catalog
        
        if self._model is None:
            raise RuntimeError("No model trained. Call train() first.")
        
        output = output or f"{catalog}_extended"
        
        extend_catalog(
            model=self._model,
            catalog_dir=str(self.base_dir / catalog),
            output_dir=str(self.base_dir / output),
            wavelength_range=wavelength_range,
        )
        
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
    
    def train(
        self,
        grid: str,
        epochs: int = 200,
        batch_size: int = 64,
        save_name: Optional[str] = None,
        max_samples: int = 10000,
    ) -> 'MLGenerator':
        """
        Train the ML generator on a stellar atmosphere grid.
        
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
            # Default to SED subdirectory inside model directory
            output_dir = str(Path(self._generator.config.get('_model_path', 'output')) / 'SED')
        
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
        
        ranges = self._generator.config.get('parameter_ranges', {})
        return {
            'teff': tuple(ranges.get('teff', [0, 0])),
            'logg': tuple(ranges.get('logg', [0, 0])),
            'metallicity': tuple(ranges.get('meta', [0, 0])),
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
        """Query locally installed filters."""
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
                
                filters = list(inst_dir.glob("*.dat"))
                results.append({
                    'facility': fac_dir.name,
                    'instrument': inst_dir.name,
                    'n_filters': len(filters),
                    'is_local': True,
                })
        
        return results
    
    @classmethod
    def _query_remote(cls, facility: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query remote filter sources (SVO, NJM)."""
        results = []
        
        try:
            from .svo_filter_grabber import SVOFilterBrowser
            browser = SVOFilterBrowser(base_dir=str(cls._filter_root))
            facilities = browser.list_facilities()
            
            for fac in facilities:
                if facility and fac.label.lower() != facility.lower():
                    continue
                results.append({
                    'facility': fac.label,
                    'source': 'svo',
                    'is_local': False,
                })
        except Exception:
            pass
        
        return results
    
    @classmethod
    def fetch(
        cls,
        facility: str,
        instrument: str,
        filter_root: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Download filter profiles.
        
        Parameters
        ----------
        facility : str
            Facility name (e.g., 'Generic', 'HST')
        instrument : str
            Instrument name (e.g., 'Johnson', 'WFC3')
        filter_root : str or Path, optional
            Base directory for filters
            
        Returns
        -------
        Path
            Directory where filters were saved
        """
        base_dir = Path(filter_root) if filter_root else cls._filter_root
        output_path = base_dir / facility / instrument
        
        # Check if already exists locally
        if output_path.exists():
            existing = list(output_path.glob("*.dat"))
            if existing:
                print(f"[filters] Already have {len(existing)} filters at {output_path}")
                return output_path
        
        # Try NJM first
        try:
            from .njm_filter_grabber import NJMFilterGrabber
            njm = NJMFilterGrabber(base_dir=str(base_dir))
            if njm.is_available():
                njm.download_filters(facility, instrument)
                # Check if files now exist
                if output_path.exists():
                    existing = list(output_path.glob("*.dat"))
                    if existing:
                        print(f"[filters] Got {len(existing)} filters from NJM")
                        return output_path
        except Exception as e:
            print(f"[filters] NJM failed: {e}")
        
        # Fall back to SVO
        try:
            from .svo_filter_grabber import SVOFilterBrowser
            browser = SVOFilterBrowser(base_dir=str(base_dir))
            
            # Navigate and download
            facilities = browser.list_facilities()
            fac = next((f for f in facilities if f.label.lower() == facility.lower()), None)
            if fac:
                instruments = browser.list_instruments(fac)
                inst = next((i for i in instruments if i.label.lower() == instrument.lower()), None)
                if inst:
                    filters = browser.list_filters(inst)
                    browser.download_filters(filters)
                    if output_path.exists():
                        existing = list(output_path.glob("*.dat"))
                        if existing:
                            print(f"[filters] Got {len(existing)} filters from SVO")
                            return output_path
        except Exception as e:
            print(f"[filters] SVO failed: {e}")
        
        raise ValueError(f"Could not download filters for {facility}/{instrument}")


# =============================================================================
# Convenience exports
# =============================================================================

def query(*args, **kwargs) -> List[CatalogInfo]:
    """Alias for SED.query()"""
    return SED.query(*args, **kwargs)

def fetch(*args, **kwargs) -> SED:
    """Alias for SED.fetch()"""
    return SED.fetch(*args, **kwargs)

def local(*args, **kwargs) -> SED:
    """Alias for SED.local()"""
    return SED.local(*args, **kwargs)