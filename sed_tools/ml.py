#!/usr/bin/env python3
"""
Unified ML API for SED Tools
=============================

High-level API for machine learning operations on stellar atmosphere models.

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

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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
    import os
    
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
            Extend all SEDs in a catalog directory.
            """
            import os
            import glob
            import pandas as pd
            
            if not self.is_trained:
                raise RuntimeError("No model loaded. Call train() or load() first.")
            
            # Load lookup table
            lookup_path = os.path.join(input_dir, "lookup_table.csv")
            if not os.path.exists(lookup_path):
                # Fallback: create a dummy DF from txt files if lookup doesn't exist
                if verbose:
                    print(f"  Warning: lookup_table.csv not found in {input_dir}")
                txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
                if not txt_files:
                    return {'extended': 0, 'failed': 0, 'skipped': 0, 'total': 0, 'output_dir': output_dir}
                
                # Create dummy dataframe
                df = pd.DataFrame({'file_name': [os.path.basename(f) for f in txt_files]})
                if verbose:
                    print(f"  Found {len(df)} .txt files (no lookup table)")
            else:
                df = pd.read_csv(lookup_path, comment='#')
            
            if verbose:
                print(f"  Total rows to process: {len(df)}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Find the file column - try multiple variants
            file_col = None
            for col_name in ['file_name', 'file', 'filename', 'sed_file', 'spectrum', 'File', 'Filename']:
                if col_name in df.columns:
                    file_col = col_name
                    break
            
            # Find parameter columns
            teff_col = None
            for col in ['teff', 'Teff', 'TEFF', 'T_eff']:
                if col in df.columns: teff_col = col; break
            
            logg_col = None
            for col in ['logg', 'log_g', 'Logg', 'LOG_G', 'gravity']:
                if col in df.columns: logg_col = col; break
            
            meta_col = None
            for col in ['meta', 'metallicity', '[M/H]', 'M_H', 'feh', '[Fe/H]', 'FeH']:
                if col in df.columns: meta_col = col; break
                
            if verbose and file_col:
                print(f"  Using file column: {file_col}")

            examples_to_plot = []
            extended_count = 0
            failed_count = 0
            skipped_count = 0
            extended_rows = []
            
            # Pre-scan for files if file_col is missing
            local_files = []
            if not file_col:
                local_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
                if verbose:
                    print(f"  No file column found. Matching {len(local_files)} local files to {len(df)} rows by index.")

            for i, row in df.iterrows():
                # Get stellar parameters with fallbacks
                try:
                    teff = float(row[teff_col]) if teff_col else 5000.0
                    if not np.isfinite(teff) or teff <= 0: teff = 5000.0
                except (ValueError, TypeError): teff = 5000.0
                
                try:
                    logg = float(row[logg_col]) if logg_col else 4.5
                    if not np.isfinite(logg) or logg > 10: logg = 4.5
                except (ValueError, TypeError): logg = 4.5
                
                try:
                    meta = float(row[meta_col]) if meta_col else 0.0
                    if not np.isfinite(meta) or abs(meta) > 100: meta = 0.0
                except (ValueError, TypeError): meta = 0.0
                
                # Determine filename
                filename = None
                if file_col:
                    filename = str(row[file_col])
                    if filename == 'nan' or not filename:
                        filename = None
                elif i < len(local_files):
                    # Fallback: match by index
                    filename = os.path.basename(local_files[i])
                
                if not filename:
                    if verbose and skipped_count < 3:
                        print(f"  [Skip] Row {i}: No filename determined")
                    skipped_count += 1
                    continue
                
                input_file = os.path.join(input_dir, filename)
                if not os.path.exists(input_file):
                    # Try just the basename in case filename has path components
                    basename = os.path.basename(filename)
                    alt_file = os.path.join(input_dir, basename)
                    if os.path.exists(alt_file):
                        input_file = alt_file
                    else:
                        if verbose and failed_count < 3:
                            print(f"  [Fail] File not found: {filename}")
                        failed_count += 1
                        continue
                
                try:
                    # Load SED
                    data = np.loadtxt(input_file)
                    if data.ndim != 2 or data.shape[1] < 2:
                        if verbose and skipped_count < 3:
                            print(f"  [Skip] {filename}: Invalid data format {data.shape}")
                        skipped_count += 1
                        continue
                        
                    wavelength = data[:, 0]
                    flux = data[:, 1]
                    
                    # Check if extension is actually needed
                    # (Allow 10% buffer)
                    covers_low = wavelength.min() <= extension_range[0] * 1.1
                    covers_high = wavelength.max() >= extension_range[1] * 0.9
                    
                    if covers_low and covers_high:
                        # Already covers range - just copy
                        output_file = os.path.join(output_dir, filename)
                        np.savetxt(output_file, data, header='wavelength_A flux_erg/s/cm2/A', fmt='%.6e')
                        
                        new_row = row.to_dict()
                        new_row['extended_by_ml'] = False
                        new_row['file_name'] = filename # Ensure filename is recorded
                        extended_rows.append(new_row)
                        skipped_count += 1
                        continue
                    
                    # Extend
                    wl_ext, flux_ext = self.extend(
                        wavelength, flux,
                        teff=teff, logg=logg, meta=meta,
                        extension_range=extension_range,
                    )
                    
                    # Save
                    output_file = os.path.join(output_dir, filename)
                    np.savetxt(
                        output_file,
                        np.column_stack([wl_ext, flux_ext]),
                        header='wavelength_A flux_erg/s/cm2/A',
                        fmt='%.6e',
                    )

                    # NEW: Capture example for plotting (limit to 6)
                    if len(examples_to_plot) < 6:
                        examples_to_plot.append((filename, wavelength, flux, wl_ext, flux_ext))

                    
                    # Track
                    new_row = row.to_dict()
                    new_row['extended_by_ml'] = True
                    new_row['file_name'] = filename
                    new_row['wl_min'] = float(wl_ext.min())
                    new_row['wl_max'] = float(wl_ext.max())
                    extended_rows.append(new_row)
                    
                    extended_count += 1
                    
                    if verbose and extended_count % 50 == 0:
                        print(f"  Extended {extended_count} SEDs...")
                    
                except Exception as e:
                    if verbose and failed_count < 5:
                        print(f"  [Error] {filename}: {e}")
                    failed_count += 1
                
            # Save new lookup table
            if extended_rows:
                new_df = pd.DataFrame(extended_rows)
                cols = ['file_name'] + [c for c in new_df.columns if c != 'file_name']
                new_df[cols].to_csv(os.path.join(output_dir, "lookup_table.csv"), index=False)

            # NEW: Generate inference plot
            if examples_to_plot:
                if verbose: print("\nGenerating inference examples plot...")
                self._completer.plot_inference_examples(examples_to_plot, output_dir)


            
            summary = {
                'extended': extended_count,
                'failed': failed_count,
                'skipped': skipped_count,
                'total': len(df),
                'output_dir': output_dir,
            }
            
            if verbose:
                print(f"\nProcessing complete:")
                print(f"  {extended_count} extended")
                print(f"  {skipped_count} skipped (already covered or invalid)")
                print(f"  {failed_count} failed")
            
            return summary
    
    @staticmethod
    def available_models(models_dir: str = "models") -> List[Dict[str, Any]]:
        """List available trained models."""
        return list_models(models_dir)
    
    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        path = f" @ {self._model_path}" if self._model_path else ""
        return f"<Completer({status}{path})>"