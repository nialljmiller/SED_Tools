"""The 0.4.1 SED completer API: load_completer, complete_sed, list_models, model_info, Completer.

These functions call the engines directly: SEDCompleter, and the Completer class over it, in ml/completer.py (old
path sed_tools.ml_sed_completer). SEDGenerator is in ml/generator.py (sed_tools.ml_sed_generator), and its optuna
tuning in ml/tuning.py. ml/common.py is what they share. Models are trained with api.MLCompleter and api.MLGenerator,
the sed-tools menus, or the engines' own train().
"""

import json
import logging
import os  # noqa: F401  (os and Optional are unused here, but sed_tools.ml had them in 0.4.1)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

import numpy as np

from .completer import Completer, SEDCompleter

logger = logging.getLogger(__name__)

__all__ = [
    'load_completer',
    'complete_sed',
    'list_models',
    'model_info',
    'Completer',
]


# Functional API

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
    return completer.complete_sed(wavelength=wavelength, flux=flux, teff=teff, logg=logg, meta=meta,
                                  extension_range=extension_range)


def list_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """
    List the trained SED completer models in ``models_dir``.

    SED generator models, which may share the directory, are not listed
    (see ``MLGenerator.list_models``).
    
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
    with open(Path(model_path) / SEDCompleter.CONFIG_FILE, 'r') as f:
        return json.load(f)
