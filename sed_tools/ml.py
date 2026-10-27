#!/usr/bin/env python3
"""
Unified ML API for SED Tools
=============================

High-level API for machine learning operations on stellar spectral grids.

This module provides a clean, unified interface for:
- Training SED completion models
- Running inference to extend incomplete SEDs
- Managing multiple trained models

Usage
-----
Training::

    from sed_tools.ml import train_completer
    
    model = train_completer(
        library='combined_models',
        output='models/my_completer',
        epochs=100,
    )

Inference::

    from sed_tools.ml import load_completer, complete_sed
    
    completer = load_completer('models/my_completer')
    wl, flux = complete_sed(completer, wavelength, flux, teff=5500, logg=4.5, meta=0.0)

Or using the class-based API::

    from sed_tools.ml import Completer
    
    completer = Completer()
    completer.train('combined_models', 'models/my_completer')
    wl, flux = completer.extend(wavelength, flux, teff=5500, logg=4.5, meta=0.0)

Model Management::

    from sed_tools.ml import list_models, model_info
    
    models = list_models()
    info = model_info('models/my_completer')
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    # Functions
    'train_completer',
    'load_completer',
    'complete_sed',
    'list_models',
    'model_info',
    'auto_train_generator',
    # Classes
    'Completer',
]


# =============================================================================
# Functional API
# =============================================================================

def train_completer(
    library: str,
    output: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    max_samples: int = 5000,
    hidden_layers: List[int] = None,
    dropout: float = 0.3,
    patience: int = 20,
    verbose: bool = True,
) -> 'SEDCompleter':
    """
    Train a new SED completion model.
    
    Parameters
    ----------
    library : str
        Path to SED library directory (must contain flux_cube.bin)
    output : str
        Output directory for trained model
    epochs : int
        Maximum training epochs
    batch_size : int
        Training batch size
    learning_rate : float
        Initial learning rate
    max_samples : int
        Maximum training samples to use
    hidden_layers : list of int
        Neural network hidden layer sizes. Default: [256, 128, 64]
    dropout : float
        Dropout rate for regularization
    patience : int
        Early stopping patience (epochs without improvement)
    verbose : bool
        Print training progress
    
    Returns
    -------
    SEDCompleter
        Trained model instance
    
    Example
    -------
    >>> model = train_completer(
    ...     library='data/stellar_models/combined_models',
    ...     output='models/combined_completer',
    ...     epochs=150,
    ... )
    """
    from .ml_sed_completer import SEDCompleter
    
    completer = SEDCompleter()
    completer.train(
        library_path=library,
        output_path=output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_samples=max_samples,
        hidden_layers=hidden_layers,
        dropout=dropout,
        patience=patience,
        verbose=verbose,
    )
    
    return completer


def load_completer(model_path: str) -> 'SEDCompleter':
    """
    Load a trained SED completion model.
    
    Parameters
    ----------
    model_path : str
        Path to model directory
    
    Returns
    -------
    SEDCompleter
        Loaded model instance
    
    Example
    -------
    >>> completer = load_completer('models/combined_completer')
    >>> wl, flux = completer.complete_sed(wl_in, flux_in, teff=5500, logg=4.5, meta=0.0)
    """
    from .ml_sed_completer import SEDCompleter
    
    return SEDCompleter(model_path=model_path)


def complete_sed(
    completer: 'SEDCompleter',
    wavelength: np.ndarray,
    flux: np.ndarray,
    teff: float,
    logg: float,
    meta: float,
    extension_range: Tuple[float, float] = (100.0, 100000.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete an incomplete SED using a trained model.
    
    Parameters
    ----------
    completer : SEDCompleter
        Trained model instance
    wavelength : np.ndarray
        Input wavelength array (Angstroms)
    flux : np.ndarray
        Input flux array (erg/s/cm²/Å)
    teff : float
        Effective temperature (K)
    logg : float
        Surface gravity (log cm/s²)
    meta : float
        Metallicity [M/H]
    extension_range : tuple
        Target wavelength range (min, max) in Angstroms
    
    Returns
    -------
    wavelength_ext : np.ndarray
        Extended wavelength array
    flux_ext : np.ndarray
        Extended flux array
    
    Example
    -------
    >>> completer = load_completer('models/my_model')
    >>> wl_ext, flux_ext = complete_sed(
    ...     completer, wl, flux,
    ...     teff=5500, logg=4.5, meta=0.0,
    ...     extension_range=(100, 100000),
    ... )
    """
    return completer.complete_sed(
        wavelength=wavelength,
        flux=flux,
        teff=teff,
        logg=logg,
        meta=meta,
        extension_range=extension_range,
    )


def list_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """
    List all available trained models.
    
    Parameters
    ----------
    models_dir : str
        Directory containing model subdirectories
    
    Returns
    -------
    list of dict
        Model information dictionaries
    
    Example
    -------
    >>> for model in list_models():
    ...     print(f"{model['name']}: {model['framework']}")
    """
    from .ml_sed_completer import SEDCompleter
    
    return SEDCompleter.list_models(models_dir)


def model_info(model_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a trained model.
    
    Parameters
    ----------
    model_path : str
        Path to model directory
    
    Returns
    -------
    dict
        Model configuration and metadata
    
    Example
    -------
    >>> info = model_info('models/my_model')
    >>> print(f"Architecture: {info['architecture']}")
    """
    import json
    from .ml_sed_completer import SEDCompleter
    
    config_file = Path(model_path) / SEDCompleter.CONFIG_FILE
    
    if not config_file.exists():
        raise FileNotFoundError(f"Model config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        return json.load(f)


def auto_train_generator(
    library: str,
    output: str,
    n_trials: int = 20,
    epochs: int = 100,
) -> Any:
    """
    Train a generator using automated hyperparameter optimization and GPU acceleration.
    
    Parameters
    ----------
    library : str
        Path to SED library directory
    output : str
        Output directory for the best model
    n_trials : int
        Number of hyperparameter optimization trials
    epochs : int
        Number of epochs for final training
    """
    from .ml_optimized import AutoTrainer
    import json
    
    os.makedirs(output, exist_ok=True)
    
    trainer = AutoTrainer(library, task="generator")
    model, best_params = trainer.train_best(
        library_path=library,
        output_path=output,
        n_trials=n_trials,
        epochs=epochs
    )
    
    # Save config
    config = {
        "architecture": "ResidualMLP",
        "best_params": best_params,
        "task": "generator",
        "framework": "PyTorch-Optimized"
    }
    with open(os.path.join(output, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    return model


# =============================================================================
# Class-based API
# =============================================================================

class Completer:
    """
    High-level SED completion interface.
    
    Provides a clean API for training and using SED completion models.
    
    Example
    -------
    Training and saving::
    
        completer = Completer()
        completer.train('data/combined_models', 'models/my_model', epochs=100)
    
    Loading and using::
    
        completer = Completer('models/my_model')
        wl, flux = completer.extend(wl_in, flux_in, teff=5500, logg=4.5, meta=0.0)
    
    Batch processing::
    
        completer = Completer('models/my_model')
        results = completer.extend_catalog('data/sphinx_models', 'data/sphinx_extended')
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize completer.
        
        Parameters
        ----------
        model_path : str, optional
            Path to existing model directory. If None, must call train() first.
        """
        from .ml_sed_completer import SEDCompleter
        
        if model_path is not None:
            self._completer = SEDCompleter(model_path)
            self._model_path = model_path
        else:
            self._completer = SEDCompleter()
            self._model_path = None
    
    @property
    def is_trained(self) -> bool:
        """Check if model is loaded/trained."""
        return self._completer.model is not None
    
    @property
    def model_path(self) -> Optional[str]:
        """Get current model path."""
        return self._model_path
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._completer.config
    
    @property
    def wavelength_grid(self) -> Optional[np.ndarray]:
        """Get model wavelength grid."""
        return self._completer.wavelength_grid
    
    def train(
        self,
        library: str,
        output: str,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        **kwargs,
    ) -> 'Completer':
        """
        Train a new completion model.
        
        Parameters
        ----------
        library : str
            Path to SED library
        output : str
            Output directory for model
        epochs : int
            Maximum training epochs
        batch_size : int
            Training batch size
        learning_rate : float
            Initial learning rate
        **kwargs
            Additional arguments passed to SEDCompleter.train()
        
        Returns
        -------
        self
            For method chaining
        """
        self._completer.train(
            library_path=library,
            output_path=output,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        self._model_path = output
        return self
    
    def load(self, model_path: str) -> 'Completer':
        """
        Load a trained model.
        
        Parameters
        ----------
        model_path : str
            Path to model directory
        
        Returns
        -------
        self
            For method chaining
        """
        self._completer.load(model_path)
        self._model_path = model_path
        return self
    
    def extend(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        teff: float,
        logg: float,
        meta: float,
        extension_range: Tuple[float, float] = (100.0, 100000.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extend an incomplete SED.
        
        Parameters
        ----------
        wavelength : np.ndarray
            Input wavelength (Angstroms)
        flux : np.ndarray
            Input flux (erg/s/cm²/Å)
        teff : float
            Effective temperature (K)
        logg : float
            Surface gravity
        meta : float
            Metallicity [M/H]
        extension_range : tuple
            Target wavelength range
        
        Returns
        -------
        wavelength_ext, flux_ext : np.ndarray
            Extended arrays
        """
        if not self.is_trained:
            raise RuntimeError("No model loaded. Call train() or load() first.")
        
        return self._completer.complete_sed(
            wavelength=wavelength,
            flux=flux,
            teff=teff,
            logg=logg,
            meta=meta,
            extension_range=extension_range,
        )
    


    def extend_catalog(
        self,
        input_dir: str,
        output_dir: str,
        extension_range: Tuple[float, float] = (100.0, 100000.0),
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Extend all model spectra in a spectral grid directory.
        
        Each spectrum's file name, Teff, logg and [M/H] come from
        ``input_dir/lookup_table.csv``. Spectra that already cover
        ``extension_range`` (to within 10%) are copied unchanged. The spectra
        and a new lookup table are written to ``output_dir``.
        
        Returns
        -------
        dict
            Counts of spectra 'extended', 'skipped' (copied unchanged),
            'failed' (always 0; see Raises) and 'total', and 'output_dir'.
        
        Raises
        ------
        FileNotFoundError
            If ``input_dir`` has no lookup_table.csv.
        ValueError
            If the lookup table has no file-name, Teff, logg or [M/H] column.
        RuntimeError
            After every other spectrum has been written, if any could not be
            read or extended; the message lists each one and why.
        """
        import pandas as pd
        
        if not self.is_trained:
            raise RuntimeError("No model loaded. Call train() or load() first.")
        
        df, columns = _read_catalog_table(input_dir)
        if verbose:
            print(f"  Total rows to process: {len(df)}")
            print(f"  Using file column: {columns['file']}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        new_rows = []
        examples = []
        failures = []
        extended_count = 0
        skipped_count = 0
        for i, row in df.iterrows():
            try:
                new_row, example = self._extend_one(row, columns, input_dir, output_dir, extension_range)
            except (OSError, ValueError) as exc:
                failures.append(f"row {i} ({row[columns['file']]}): {exc}")
                continue
            
            new_rows.append(new_row)
            if example is None:
                skipped_count += 1
                continue
            
            extended_count += 1
            if len(examples) < 6:
                examples.append(example)
            if verbose and extended_count % 50 == 0:
                print(f"  Extended {extended_count} SEDs...")
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            cols = ['file_name'] + [c for c in new_df.columns if c != 'file_name']
            new_df[cols].to_csv(os.path.join(output_dir, "lookup_table.csv"), index=False)
        
        if examples:
            if verbose:
                print("\nGenerating inference examples plot...")
            self._completer.plot_inference_examples(examples, output_dir)
        
        if verbose:
            print("\nProcessing complete:")
            print(f"  {extended_count} extended")
            print(f"  {skipped_count} skipped (already covered the range)")
            print(f"  {len(failures)} failed")
        
        if failures:
            raise RuntimeError(
                f"{len(failures)} of {len(df)} spectra in {input_dir} could not be extended "
                f"(the other {len(new_rows)} were written to {output_dir}):\n  " + "\n  ".join(failures)
            )
        
        return {
            'extended': extended_count,
            'failed': 0,
            'skipped': skipped_count,
            'total': len(df),
            'output_dir': output_dir,
        }
    
    def _extend_one(self, row, columns, input_dir, output_dir, extension_range):
        """
        Extend, or copy, one catalog spectrum into output_dir.
        
        Returns its row for the new lookup table and, if it was extended, an
        example for the inference plot (None if it was copied). Raises
        ValueError or OSError if the spectrum cannot be used.
        """
        teff, logg, meta = _row_parameters(row, columns)
        
        filename = str(row[columns['file']])
        if filename == 'nan' or not filename:
            raise ValueError("no file name")
        
        data = np.loadtxt(_find_spectrum(input_dir, filename))
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"expected wavelength and flux columns, got an array of shape {data.shape}")
        wavelength = data[:, 0]
        flux = data[:, 1]
        
        output_file = os.path.join(output_dir, filename)
        new_row = row.to_dict()
        
        covers_low = wavelength.min() <= extension_range[0] * 1.1
        covers_high = wavelength.max() >= extension_range[1] * 0.9
        if covers_low and covers_high:
            np.savetxt(output_file, data, header='wavelength_A flux_erg/s/cm2/A', fmt='%.6e')
            new_row['extended_by_ml'] = False
            new_row['file_name'] = filename
            return new_row, None
        
        wl_ext, flux_ext = self.extend(
            wavelength, flux,
            teff=teff, logg=logg, meta=meta,
            extension_range=extension_range,
        )
        np.savetxt(
            output_file,
            np.column_stack([wl_ext, flux_ext]),
            header='wavelength_A flux_erg/s/cm2/A',
            fmt='%.6e',
        )
        new_row['extended_by_ml'] = True
        new_row['file_name'] = filename
        new_row['wl_min'] = float(wl_ext.min())
        new_row['wl_max'] = float(wl_ext.max())
        return new_row, (filename, wavelength, flux, wl_ext, flux_ext)
    
    @staticmethod
    def available_models(models_dir: str = "models") -> List[Dict[str, Any]]:
        """List available trained models."""
        return list_models(models_dir)
    
    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        path = f" @ {self._model_path}" if self._model_path else ""
        return f"<Completer({status}{path})>"


# =============================================================================
# Catalog helpers for Completer.extend_catalog
# =============================================================================

# Column names accepted in a catalog's lookup_table.csv, in order of preference
_CATALOG_COLUMNS = {
    'file': ['file_name', 'file', 'filename', 'sed_file', 'spectrum', 'File', 'Filename'],
    'teff': ['teff', 'Teff', 'TEFF', 'T_eff'],
    'logg': ['logg', 'log_g', 'Logg', 'LOG_G', 'gravity'],
    'meta': ['meta', 'metallicity', '[M/H]', 'M_H', 'feh', '[Fe/H]', 'FeH'],
}


def _read_catalog_table(input_dir: str):
    """Read a catalog's lookup table and find its file-name and parameter columns."""
    import pandas as pd
    
    lookup_path = os.path.join(input_dir, "lookup_table.csv")
    if not os.path.exists(lookup_path):
        raise FileNotFoundError(
            f"lookup_table.csv not found in {input_dir}; extend_catalog needs each "
            "spectrum's file name, Teff, logg and [M/H] from it"
        )
    df = pd.read_csv(lookup_path, comment='#')
    
    columns = {}
    missing = []
    for role, candidates in _CATALOG_COLUMNS.items():
        found = [c for c in candidates if c in df.columns]
        if found:
            columns[role] = found[0]
        else:
            missing.append(f"{role} (one of {', '.join(candidates)})")
    if missing:
        raise ValueError(
            f"{lookup_path} has no column for {'; '.join(missing)}. Columns read: "
            f"{list(df.columns)}. A header line starting with '#' is dropped when "
            "the table is read, so its column names are not seen."
        )
    return df, columns


def _row_number(row, column: str, name: str) -> float:
    value = row[column]
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} {value!r} is not a number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} is {value!r}")
    return number


def _row_parameters(row, columns) -> Tuple[float, float, float]:
    """Teff, logg and [M/H] of one lookup-table row; ValueError if one is unusable."""
    teff = _row_number(row, columns['teff'], 'Teff')
    if teff <= 0:
        raise ValueError(f"Teff {teff} is not positive")
    logg = _row_number(row, columns['logg'], 'logg')
    if logg > 10:
        raise ValueError(f"logg {logg} is above 10")
    meta = _row_number(row, columns['meta'], '[M/H]')
    if abs(meta) > 100:
        raise ValueError(f"[M/H] {meta} is outside -100..100")
    return teff, logg, meta


def _find_spectrum(input_dir: str, filename: str) -> str:
    """Path of a catalog spectrum; the lookup may hold a path, so its basename is tried too."""
    path = os.path.join(input_dir, filename)
    if os.path.exists(path):
        return path
    path = os.path.join(input_dir, os.path.basename(filename))
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"{filename} not found in {input_dir}")
