#!/usr/bin/env python3
"""
ML-Powered SED Completer (PyTorch)
===================================

Extends incomplete SEDs to broader wavelength ranges using:
1. Black body radiation as physical baseline
2. Neural network trained on complete SEDs for refinement

The tool only appends/extends - never modifies existing wavelength coverage.

Model Storage
-------------
Each trained model is stored in its own directory containing:
    model.pt              - PyTorch state dict
    config.json           - Architecture, scaler params, wavelength grid
    training_plots.png    - Loss curves
    prediction_examples.png - Sample predictions vs ground truth

Usage
-----
    # Training
    completer = SEDCompleter()
    completer.train('path/to/library', 'path/to/model_dir', epochs=100)
    
    # Inference
    completer = SEDCompleter('path/to/model_dir')
    wl, flux = completer.complete_sed(wavelength, flux, teff=5500, logg=4.5, meta=0.0)
    
CLI
---
    python -m sed_tools.ml_sed_completer train --library /path/to/lib --output models/my_model
    python -m sed_tools.ml_sed_completer complete --model models/my_model --input sed.txt --output extended.txt
"""

import json
import logging
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import numpy as np
from scipy.integrate import simpson as simps

from ._constants import H as PLANCK_H, C as SPEED_C, K as BOLTZMANN_K, SIGMA
from .ml_sed_generator import _ask_number

# Default wavelength grid parameters
DEFAULT_WL_MIN = 100.0      # Angstroms
DEFAULT_WL_MAX = 100000.0   # Angstroms
DEFAULT_WL_POINTS = 1000
# Training uses only samples whose input window is at least this fraction valid.
MIN_WINDOW_COVERAGE = 0.8


def _warn_dropped(dropped: Counter, n_sampled: int, library_path: str) -> None:
    """Say how many sampled spectra were left out of the training set, and why."""
    if not dropped:
        return
    reasons = ", ".join(f"{count} with {reason}" for reason, count in dropped.items())
    logger.warning("Training data from %s: dropped %d of %d sampled spectra (%s)",
                   library_path, sum(dropped.values()), n_sampled, reasons)


def _warn_window_coverage(window: np.ndarray, wavelength: np.ndarray) -> None:
    """Warn when the SED covers less of the network's input window than training required."""
    covered = (window >= wavelength.min()) & (window <= wavelength.max())
    if covered.sum() >= MIN_WINDOW_COVERAGE * covered.size:
        return
    logger.warning(
        "The input SED (%.0f-%.0f Å) covers %.0f%% of the network's input window (%.0f-%.0f Å); "
        "the other %.0f%% is filled with 1e-20. Training required at least %.0f%% coverage, so "
        "this prediction is outside what the network was trained on.",
        wavelength.min(), wavelength.max(), 100 * covered.mean(), window[0], window[-1],
        100 * (1 - covered.mean()), 100 * MIN_WINDOW_COVERAGE,
    )


def _blend_with_blackbody(wl_out, log_dist, wl_pred, pred, bb_flux):
    """
    Blend the network prediction into the black body away from the data edge.
    
    ``log_dist`` is each output point's distance from the edge of the input
    data in decades of wavelength. The prediction's weight falls as
    exp(-log_dist / 0.5): the network is trusted for about half a decade (a
    factor of ~3), then the black body takes over. Log wavelength because
    stellar spectra behave more uniformly there.
    """
    blend_decades = 0.5
    pred_interp = np.interp(wl_out, wl_pred, pred, left=pred[0], right=pred[-1])
    weight_ml = np.exp(-log_dist / blend_decades)
    weight_ml = np.clip(weight_ml, 0, 1)
    return weight_ml * pred_interp + (1 - weight_ml) * bb_flux


def planck_function(wavelength_angstrom: np.ndarray, temperature: float) -> np.ndarray:
    """
    Calculate black body flux using Planck's law.
    
    Parameters
    ----------
    wavelength_angstrom : np.ndarray
        Wavelength in Angstroms
    temperature : float
        Effective temperature in Kelvin
    
    Returns
    -------
    np.ndarray
        Flux in erg/s/cm²/Å (normalized to σT⁴)
    """
    wavelength_cm = wavelength_angstrom * 1e-8
    
    numerator = 2.0 * PLANCK_H * SPEED_C**2 / wavelength_cm**5
    exponent = PLANCK_H * SPEED_C / (wavelength_cm * BOLTZMANN_K * temperature)
    exponent = np.clip(exponent, 0, 700)
    
    denominator = np.exp(exponent) - 1.0
    denominator = np.where(denominator > 0, denominator, 1e-100)
    
    flux = numerator / denominator * 1e-8  # Convert to per Angstrom
    
    # Normalize to σT⁴
    integrated = simps(flux, wavelength_angstrom)
    if integrated > 0:
        flux = flux * (SIGMA * temperature**4 / integrated)
    
    return flux


class SEDCompleterNetwork:
    """
    PyTorch neural network for SED completion.
    
    Architecture: Input (SED + params) -> Dense layers with BatchNorm + Dropout -> Output (extension)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize the network.
        
        Parameters
        ----------
        input_dim : int
            Input dimension (SED features + 3 stellar params)
        output_dim : int
            Output dimension (extension wavelengths)
        hidden_layers : list of int
            Hidden layer sizes. Default: [256, 128, 64]
        dropout : float
            Dropout rate
        """
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required. Install: pip install torch")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers or [256, 128, 64]
        self.dropout = dropout
        
        # Build network
        layers = []
        prev_dim = input_dim + 3  # +3 for stellar parameters
        
        for i, hidden_dim in enumerate(self.hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            if i < len(self.hidden_layers) - 1:  # No dropout on last hidden layer
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        # The same device SEDCompleter.device picks, so inputs and weights agree
        from .ml_utils import get_device
        self.device = get_device()
        self.model.to(self.device)
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def __call__(self, x):
        return self.forward(x)
    
    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self
    
    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def to(self, device):
        self.model.to(device)
        self.device = device
        return self


class SEDCompleter:
    """
    Complete incomplete SEDs using black body + neural network predictions.
    
    Usage
    -----
    Training::
    
        completer = SEDCompleter()
        completer.train('/path/to/library', '/path/to/output_model')
    
    Inference::
    
        completer = SEDCompleter('/path/to/model')
        wl_ext, flux_ext = completer.complete_sed(wl, flux, teff=5500, logg=4.5, meta=0.0)
    """
    
    # Standard filenames within model directory
    MODEL_FILE = "model.pt"
    CONFIG_FILE = "config.json"
    TRAINING_PLOT = "training_plots.png"
    PREDICTION_PLOT = "prediction_examples.png"
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the SED completer.
        
        Parameters
        ----------
        model_path : str or Path, optional
            Path to model directory. If provided, loads the model.
        """
        self.model: Optional[SEDCompleterNetwork] = None
        self.config: Dict[str, Any] = {}
        self.wavelength_grid: Optional[np.ndarray] = None
        self.scaler_params: Optional[Dict[str, float]] = None
        self._device = None
        
        if model_path is not None:
            self.load(model_path)
    
    @property
    def device(self):
        """Get compute device."""
        if self._device is None:
            from .ml_utils import get_device
            self._device = get_device()
        return self._device
    
    def _create_wavelength_grid(
        self,
        min_wl: float = DEFAULT_WL_MIN,
        max_wl: float = DEFAULT_WL_MAX,
        n_points: int = DEFAULT_WL_POINTS,
    ) -> np.ndarray:
        """Create logarithmic wavelength grid."""
        return np.logspace(np.log10(min_wl), np.log10(max_wl), n_points)
    
    def _normalize_params(self, teff: float, logg: float, meta: float) -> np.ndarray:
        """Normalize stellar parameters using stored scaler."""
        if self.scaler_params is None:
            # Fallback defaults
            return np.array([
                (teff - 5000.0) / 10000.0,
                (logg - 3.0) / 3.0,
                meta / 2.0,
            ])
        
        return np.array([
            (teff - self.scaler_params['teff_mean']) / self.scaler_params['teff_std'],
            (logg - self.scaler_params['logg_mean']) / self.scaler_params['logg_std'],
            (meta - self.scaler_params['meta_mean']) / self.scaler_params['meta_std'],
        ])
    
    def _load_flux_cube(self, library_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load flux cube from binary file.
        
        Returns
        -------
        teff, logg, meta, wavelengths, flux_cube
        """
        cube_path = os.path.join(library_path, "flux_cube.bin")
        if not os.path.exists(cube_path):
            raise FileNotFoundError(f"flux_cube.bin not found in {library_path}")
        
        with open(cube_path, 'rb') as f:
            header = f.read(16)
            nt, nl, nm, nw = struct.unpack('4i', header)
            
            teff = np.fromfile(f, dtype=np.float64, count=nt)
            logg = np.fromfile(f, dtype=np.float64, count=nl)
            meta = np.fromfile(f, dtype=np.float64, count=nm)
            wavelengths = np.fromfile(f, dtype=np.float64, count=nw)
            
            flux_data = np.fromfile(f, dtype=np.float64, count=nt*nl*nm*nw)
            flux_cube = flux_data.reshape((nw, nm, nl, nt))
        
        return teff, logg, meta, wavelengths, flux_cube
    
    def _prepare_training_data(
        self,
        library_path: str,
        max_samples: int = 5000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from flux cube.
        
        Uses randomized "known" regions to train the network to extend from
        any part of the spectrum (UV-only, optical-only, IR-only, etc.).
        
        Returns
        -------
        X_input : np.ndarray
            Input SED (known region, zero-padded to fixed size)
        y_output : np.ndarray
            Output SED (full spectrum for extension)
        params : np.ndarray
            Stellar parameters (teff, logg, meta)
        masks : np.ndarray
            Valid data masks (1 = valid, 0 = invalid/missing)
        """
        print("Loading flux cube...")
        teff, logg, meta, wavelengths, flux_cube = self._load_flux_cube(library_path)
        
        nt, nl, nm = len(teff), len(logg), len(meta)
        total_seds = nt * nl * nm
        n_samples = min(max_samples, total_seds)
        
        print(f"  Grid: {nt} Teff × {nl} logg × {nm} [M/H] = {total_seds} total SEDs")
        print(f"  Sampling {n_samples} SEDs for training")
        
        self.wavelength_grid = self._create_wavelength_grid()
        n_wl = len(self.wavelength_grid)
        
        # The network input is a fixed 40% of the grid. Training moves that
        # window around; inference uses the central one stored here.
        known_width = n_wl * 2 // 5
        self._known_width = known_width
        self._known_start = (n_wl - known_width) // 2
        self._known_end = self._known_start + known_width
        
        X_list = []
        y_list = []
        params_list = []
        masks_list = []
        dropped = Counter()
        
        indices = np.random.permutation(total_seds)[:n_samples]
        for idx in indices:
            i_t = idx // (nl * nm)
            remainder = idx % (nl * nm)
            i_l = remainder // nm
            i_m = remainder % nm
            
            # The cube is stored as [nw, nm, nl, nt]
            reason, sample = self._training_sample(flux_cube[:, i_m, i_l, i_t], wavelengths, known_width)
            if reason:
                dropped[reason] += 1
                continue
            
            X_known, y_full, ext_mask = sample
            X_list.append(X_known)
            y_list.append(y_full)
            params_list.append([teff[i_t], logg[i_l], meta[i_m]])
            masks_list.append(ext_mask)
        
        _warn_dropped(dropped, len(indices), library_path)
        if len(X_list) == 0:
            raise ValueError(f"No valid training samples found in {library_path}")
        
        print(f"  Prepared {len(X_list)} valid samples")
        print(f"  Known region width: {known_width} points (randomized position)")
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        params = np.array(params_list, dtype=np.float32)
        masks = np.array(masks_list, dtype=np.float32)
        
        avg_coverage = masks.mean() * 100
        print(f"  Average extension coverage: {avg_coverage:.1f}%")
        
        return X, y, params, masks

    def _training_sample(
        self,
        flux: np.ndarray,
        wavelengths: np.ndarray,
        known_width: int,
    ) -> Tuple[Optional[str], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Cut one training sample from a grid spectrum at a random input window.
        
        Returns ``(None, (X_known, y_full, ext_mask))``, or ``(reason, None)``
        when the spectrum cannot be used.
        """
        valid_original = (flux > 0) & np.isfinite(flux)
        if valid_original.sum() < 100:
            return "fewer than 100 positive, finite flux points", None
        
        # Resample to the model grid; NaN outside the original coverage
        flux_interp = np.interp(self.wavelength_grid, wavelengths, flux, left=np.nan, right=np.nan)
        valid_mask = (
            (self.wavelength_grid >= wavelengths.min()) &
            (self.wavelength_grid <= wavelengths.max()) &
            np.isfinite(flux_interp) &
            (flux_interp > 0)
        )
        
        # Place the input window at random inside the valid data
        too_narrow = f"coverage narrower than the {known_width}-point input window"
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < known_width:
            return too_narrow, None
        max_start = valid_indices[-1] - known_width
        min_start = valid_indices[0]
        if max_start <= min_start:
            return too_narrow, None
        known_start = np.random.randint(min_start, max_start)
        known_end = known_start + known_width
        
        if valid_mask[known_start:known_end].sum() < known_width * MIN_WINDOW_COVERAGE:
            return "input window less than 80% valid", None
        
        # The loss covers every valid point outside the input window
        ext_mask = valid_mask.copy()
        ext_mask[known_start:known_end] = False
        if ext_mask.sum() < 50:
            return "fewer than 50 valid points outside the input window", None
        
        # The target is the whole spectrum. Invalid points get a small positive
        # flux; the masked loss ignores them.
        X_known = flux_interp[known_start:known_end]
        X_known = np.where(np.isfinite(X_known) & (X_known > 0), X_known, 1e-20)
        y_full = np.where(np.isfinite(flux_interp) & (flux_interp > 0), flux_interp, 1e-20)
        return None, (X_known, y_full, ext_mask.astype(np.float32))
    
    def train(
        self,
        library_path: str,
        output_path: str,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        max_samples: int = 5000,
        hidden_layers: List[int] = None,
        dropout: float = 0.3,
        patience: int = 20,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the neural network on complete SEDs.
        
        Parameters
        ----------
        library_path : str
            Path to SED library with flux_cube.bin
        output_path : str
            Directory to save trained model
        epochs : int
            Maximum training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Initial learning rate
        max_samples : int
            Maximum training samples to use
        hidden_layers : list of int
            Hidden layer sizes
        dropout : float
            Dropout rate
        patience : int
            Early stopping patience
        verbose : bool
            Print progress
        
        Returns
        -------
        history : dict
            Training history with 'train_loss' and 'val_loss'
        """
        import torch
        from sklearn.model_selection import train_test_split
        from .ml_utils import data_loaders, fit
        
        X, y, params, masks = self._prepare_training_data(library_path, max_samples)
        X_combined, y_norm, masks = self._normalise_training_data(X, y, params, masks, library_path)
        
        X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
            X_combined, y_norm, masks, test_size=0.2, random_state=42
        )
        train_loader, val_loader = data_loaders((X_train, y_train, m_train), (X_val, y_val, m_val), batch_size)
        
        input_dim = X.shape[1]
        output_dim = y_norm.shape[1]
        print(f"Building network: {input_dim} + 3 -> {hidden_layers or [256, 128, 64]} -> {output_dim}")
        self.model = SEDCompleterNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        history, best_val_loss = fit(
            self.model, train_loader, val_loader, optimizer, scheduler,
            epochs=epochs, patience=patience, verbose=verbose, device=self.device,
            progress="  Epoch {epoch:3d}: train_loss={train:.6f}, val_loss={val:.6f}, lr={lr:.2e}",
        )
        if not history['train_loss']:
            raise RuntimeError(
                f"Training on {library_path} produced no usable epoch (every training or "
                f"validation batch had a NaN loss); nothing was saved to {output_path}"
            )
        
        print("-" * 60)
        print(f"Training complete. Best val_loss: {best_val_loss:.6f}")
        
        self.save(output_path, hidden_layers=hidden_layers, dropout=dropout)
        
        print("\nGenerating training plots...")
        self._plot_training_results(history, output_path)
        self._plot_prediction_examples(X_val, y_val, output_path, masks=m_val)
        
        return history

    def _normalise_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray,
        masks: np.ndarray,
        library_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit ``self.scaler_params`` and return the network inputs and targets.
        
        Fluxes are scaled in log10, using valid points only. Returns
        ``(X_combined, y_norm, masks)`` as float32, where ``X_combined`` is
        the normalised input window followed by the three parameters.
        """
        print("Normalizing data...")
        
        # Floor at 1e-20 so log10 has no -inf; the floor values are excluded below
        X_log = np.log10(np.maximum(X, 1e-20))
        y_log = np.log10(np.maximum(y, 1e-20))
        
        x_valid = X_log[X_log > -25]
        y_valid = y_log[(masks > 0.5) & (y_log > -25)]
        if len(x_valid) == 0 or len(y_valid) == 0:
            raise ValueError("No valid data points for normalization")
        
        self.scaler_params = {
            'teff_mean': float(np.mean(params[:, 0])),
            'teff_std': float(np.std(params[:, 0]) + 1e-8),
            'logg_mean': float(np.mean(params[:, 1])),
            'logg_std': float(np.std(params[:, 1]) + 1e-8),
            'meta_mean': float(np.mean(params[:, 2])),
            'meta_std': float(np.std(params[:, 2]) + 1e-8),
            'flux_log_mean': float(np.mean(x_valid)),
            'flux_log_std': float(np.std(x_valid) + 1e-8),
            'ext_log_mean': float(np.mean(y_valid)),
            'ext_log_std': float(np.std(y_valid) + 1e-8),
        }
        
        print(f"  X (known) log10 range: [{x_valid.min():.2f}, {x_valid.max():.2f}]")
        print(f"  y (ext) log10 range: [{y_valid.min():.2f}, {y_valid.max():.2f}]")
        
        X_norm = (X_log - self.scaler_params['flux_log_mean']) / self.scaler_params['flux_log_std']
        y_norm = (y_log - self.scaler_params['ext_log_mean']) / self.scaler_params['ext_log_std']
        
        # Invalid points are masked out of the loss, so any finite value will do
        X_norm = np.where(np.isfinite(X_norm), X_norm, 0.0)
        y_norm = np.where(np.isfinite(y_norm), y_norm, 0.0)
        
        params_norm = np.column_stack([
            (params[:, 0] - self.scaler_params['teff_mean']) / self.scaler_params['teff_std'],
            (params[:, 1] - self.scaler_params['logg_mean']) / self.scaler_params['logg_std'],
            (params[:, 2] - self.scaler_params['meta_mean']) / self.scaler_params['meta_std'],
        ])
        
        X_combined = np.concatenate([X_norm, params_norm], axis=1).astype(np.float32)
        y_norm = y_norm.astype(np.float32)
        masks = masks.astype(np.float32)
        
        print(f"  Input shape: {X_combined.shape}")
        print(f"  Output shape: {y_norm.shape}")
        print(f"  Masks shape: {masks.shape}")
        
        # The fluxes were made finite above; non-finite grid parameters were not
        if not np.all(np.isfinite(X_combined)):
            raise ValueError(f"Non-finite Teff, logg or [M/H] in the training grid {library_path}")
        
        return X_combined, y_norm, masks

    def _plot_training_results(self, history: Dict[str, List[float]], output_path: str):
        """Plot and save training loss curves."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('SED Completer Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Mark best epoch
        best_epoch = np.argmin(history['val_loss']) + 1
        best_loss = min(history['val_loss'])
        ax.axvline(best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best: epoch {best_epoch}')
        ax.annotate(f'Best: {best_loss:.4f}', xy=(best_epoch, best_loss),
                   xytext=(best_epoch + 5, best_loss * 1.5),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   fontsize=10, color='green')
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_path, self.TRAINING_PLOT)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {plot_path}")
    
    def _plot_prediction_examples(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_path: str,
        n_examples: int = 6,
        masks: np.ndarray = None,
    ):
        """Plot example predictions vs ground truth."""
        import matplotlib.pyplot as plt
        import torch
        
        self.model.eval()
        
        # Select random examples
        indices = np.random.choice(len(X_val), min(n_examples, len(X_val)), replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            
            X_sample = torch.from_numpy(X_val[idx:idx+1]).to(self.device)
            
            with torch.no_grad():
                y_pred = self.model(X_sample).cpu().numpy()[0]
            
            y_true = y_val[idx]
            mask = masks[idx] if masks is not None else np.ones_like(y_true)
            
            # Denormalize for plotting
            y_true_log = y_true * self.scaler_params['ext_log_std'] + self.scaler_params['ext_log_mean']
            y_pred_log = y_pred * self.scaler_params['ext_log_std'] + self.scaler_params['ext_log_mean']
            
            y_true_flux = 10 ** y_true_log
            y_pred_flux = 10 ** y_pred_log
            
            # Full wavelength grid (output is now full spectrum)
            wl = self.wavelength_grid
            
            # Only plot valid (masked) regions - these are the extension points
            valid = mask > 0.5
            
            if valid.sum() > 0:
                ax.loglog(wl[valid], y_true_flux[valid], 'b-', alpha=0.7, label='True', linewidth=1.5)
                ax.loglog(wl[valid], y_pred_flux[valid], 'r--', alpha=0.7, label='Predicted', linewidth=1.5)
                
                # Calculate errors only on valid points
                rel_error = np.median(np.abs(y_pred[valid] - y_true[valid]) / (np.abs(y_true[valid]) + 1e-8)) * 100
                coverage = valid.sum() / len(valid) * 100
                
                ax.set_title(f'Sample {i+1}: Error={rel_error:.1f}%, Coverage={coverage:.0f}%')
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Sample {i+1}: No valid extension data')
            
            ax.set_xlabel('Wavelength (Å)')
            ax.set_ylabel('Flux (erg/s/cm²/Å)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Prediction Examples: Extension Regions (valid data only)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_path, self.PREDICTION_PLOT)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {plot_path}")

    def plot_inference_examples(
        self,
        examples: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        output_dir: str,
        filename: str = "inference_examples.png",
    ):
        """
        Plot examples of extended SEDs from inference.
        
        Parameters
        ----------
        examples : list of tuples
            (filename, wl_orig, flux_orig, wl_ext, flux_ext)
        output_dir : str
            Directory to save plot
        filename : str
            Name of the output file
        """
        import matplotlib.pyplot as plt
        
        n = len(examples)
        if n == 0:
            return
        
        # Determine grid size (max 3 columns)
        cols = 3
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        # Handle single subplot case
        if n == 1:
            axes = np.array([axes])
        
        # Flatten for easy indexing if multiple rows
        ax_flat = axes.flatten()
        
        for i, (fname, wl_in, flux_in, wl_out, flux_out) in enumerate(examples):
            ax = ax_flat[i]
            
            # Plot extended (background, red)
            ax.loglog(wl_out, flux_out, 'r-', alpha=0.6, linewidth=2, label='Extended')
            
            # Plot original (foreground, blue)
            ax.loglog(wl_in, flux_in, 'b-', alpha=0.8, linewidth=1.5, label='Original')
            
            ax.set_title(fname, fontsize=10, pad=5)
            ax.set_xlabel('Wavelength (Å)', fontsize=8)
            ax.set_ylabel('Flux', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, which='both')
            
            # Only add legend to the first plot to reduce clutter
            if i == 0:
                ax.legend(fontsize=8, loc='best')
        
        # Hide empty subplots
        for j in range(i+1, len(ax_flat)):
            ax_flat[j].axis('off')
            
        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved inference plot: {out_path}")

    def save(
        self,
        path: Union[str, Path],
        hidden_layers: List[int] = None,
        dropout: float = 0.3,
    ):
        """
        Save model to directory.
        
        Creates directory with:
            model.pt      - PyTorch state dict
            config.json   - Architecture and scaler parameters
        
        Parameters
        ----------
        path : str or Path
            Directory to save model
        hidden_layers : list
            Hidden layer sizes (for config)
        dropout : float
            Dropout rate (for config)
        """
        import torch
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        model_file = path / self.MODEL_FILE
        torch.save(self.model.state_dict(), model_file)
        
        # Build config
        self.config = {
            'architecture': {
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'hidden_layers': hidden_layers or [256, 128, 64],
                'dropout': dropout,
            },
            'scaler_params': self.scaler_params,
            'wavelength_grid': self.wavelength_grid.tolist(),
            'regions': {
                'known_start': getattr(self, '_known_start', None),
                'known_end': getattr(self, '_known_end', None),
                'known_width': getattr(self, '_known_width', None),
            },
            'version': '2.0.0',
            'framework': 'pytorch',
        }
        
        config_file = path / self.CONFIG_FILE
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Model saved to: {path}")
        print(f"  - {self.MODEL_FILE}")
        print(f"  - {self.CONFIG_FILE}")
    
    def load(self, path: Union[str, Path]):
        """
        Load model from directory.
        
        Parameters
        ----------
        path : str or Path
            Directory containing model.pt and config.json
        """
        import torch
        
        path = Path(path)
        
        if not path.is_dir():
            raise ValueError(f"Model path must be a directory: {path}")
        
        # Load config
        config_file = path / self.CONFIG_FILE
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.scaler_params = self.config['scaler_params']
        self.wavelength_grid = np.array(self.config['wavelength_grid'])
        
        # Restore region indices if available
        regions = self.config.get('regions', {})
        self._known_start = regions.get('known_start')
        self._known_end = regions.get('known_end')
        self._known_width = regions.get('known_width')
        
        # If not stored, compute from wavelength grid (backwards compatibility)
        if self._known_start is None or self._known_end is None:
            n_wl = len(self.wavelength_grid)
            width = n_wl * 2 // 5  # 40% of grid
            self._known_width = width
            self._known_start = (n_wl - width) // 2
            self._known_end = self._known_start + width
        
        # Rebuild model architecture
        arch = self.config['architecture']
        self.model = SEDCompleterNetwork(
            input_dim=arch['input_dim'],
            output_dim=arch['output_dim'],
            hidden_layers=arch['hidden_layers'],
            dropout=arch['dropout'],
        )
        
        # Load weights
        model_file = path / self.MODEL_FILE
        if not model_file.exists():
            raise FileNotFoundError(f"Model weights not found: {model_file}")
        
        state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"Loaded model from: {path}")
    
    def complete_sed(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        teff: float,
        logg: float,
        meta: float,
        extension_range: Tuple[float, float] = (DEFAULT_WL_MIN, DEFAULT_WL_MAX),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete/extend an incomplete SED.
        
        The SED's own samples are kept unchanged, including any outside
        ``extension_range``. Points are added only where ``extension_range``
        reaches beyond the SED: those points of a 1,000-point logarithmic
        grid over ``extension_range`` that lie below or above its coverage.
        
        Parameters
        ----------
        wavelength : np.ndarray
            Existing wavelength coverage (Angstroms), increasing
        flux : np.ndarray
            Existing flux (erg/s/cm²/Å)
        teff : float
            Effective temperature (K)
        logg : float
            Surface gravity (log cm/s²)
        meta : float
            Metallicity [M/H]
        extension_range : tuple
            (min_wavelength, max_wavelength) the output SED must cover
        
        Returns
        -------
        wavelength_extended : np.ndarray
            The added points below the SED, its own wavelengths, then the added points above it
        flux_extended : np.ndarray
            Flux at those wavelengths
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Train or load a model first.")
        
        self.model.eval()
        
        full_grid = self._create_wavelength_grid(extension_range[0], extension_range[1])
        wl_min, wl_max = wavelength.min(), wavelength.max()
        need_low = full_grid < wl_min
        need_high = full_grid > wl_max
        
        if not (need_low.any() or need_high.any()):
            print("SED already covers requested range")
            return wavelength, flux
        
        # The black body is the baseline wherever the prediction fades out
        bb_flux = planck_function(full_grid, teff)
        
        y_pred_flux = self._predict_grid_flux(wavelength, flux, teff, logg, meta)
        known_start = self._known_start
        known_end = self._known_end
        
        # The network predicts the whole grid; only the parts outside the
        # input window extend the SED (either part may be empty)
        wl_low = full_grid[need_low]
        flux_low = _blend_with_blackbody(
            wl_low, np.log10(wl_min) - np.log10(wl_low),
            self.wavelength_grid[:known_start], y_pred_flux[:known_start], bb_flux[need_low],
        )
        wl_high = full_grid[need_high]
        flux_high = _blend_with_blackbody(
            wl_high, np.log10(wl_high) - np.log10(wl_max),
            self.wavelength_grid[known_end:], y_pred_flux[known_end:], bb_flux[need_high],
        )
        
        wl_out = np.concatenate([wl_low, wavelength, wl_high])
        flux_out = np.concatenate([flux_low, flux, flux_high])
        print(f"Extended SED: {wl_min:.0f}-{wl_max:.0f} Å → {wl_out[0]:.0f}-{wl_out[-1]:.0f} Å "
              f"(its {len(wavelength)} points kept; {len(wl_low)} added below, {len(wl_high)} above)")
        
        return wl_out, flux_out

    def _predict_grid_flux(self, wavelength, flux, teff, logg, meta) -> np.ndarray:
        """
        Run the network on an SED and return its predicted flux on ``self.wavelength_grid``.
        
        The input is always the stored central window of the model grid, with
        1e-20 wherever the SED does not reach; a warning says so when that is
        more than training allowed.
        """
        import torch

        _warn_window_coverage(self.wavelength_grid[self._known_start:self._known_end], wavelength)
        flux_safe = np.maximum(flux, 1e-20)
        flux_log = np.log10(flux_safe)
        flux_on_grid = np.interp(self.wavelength_grid, wavelength, 10**flux_log, left=1e-20, right=1e-20)
        
        flux_known = flux_on_grid[self._known_start:self._known_end]
        flux_known_log = np.log10(np.maximum(flux_known, 1e-20))
        flux_known_norm = (flux_known_log - self.scaler_params['flux_log_mean']) / self.scaler_params['flux_log_std']
        flux_known_norm = np.where(np.isfinite(flux_known_norm), flux_known_norm, 0.0)
        
        params_norm = self._normalize_params(teff, logg, meta)
        X_input = np.concatenate([flux_known_norm, params_norm]).astype(np.float32)
        X_tensor = torch.from_numpy(X_input[np.newaxis, :]).to(self.device)
        
        with torch.no_grad():
            y_pred_norm = self.model(X_tensor).cpu().numpy()[0]
        
        y_pred_log = y_pred_norm * self.scaler_params['ext_log_std'] + self.scaler_params['ext_log_mean']
        return 10 ** y_pred_log
    
    @staticmethod
    def list_models(models_dir: str = "models") -> List[Dict[str, Any]]:
        """
        List available trained models.
        
        Parameters
        ----------
        models_dir : str
            Directory containing model subdirectories
        
        Returns
        -------
        list of dict
            Model info: name, path, version, etc. Generators (SEDGenerator
            models, which may share the directory) are not listed.
        """
        models = []
        models_path = Path(models_dir)
        
        if not models_path.exists():
            return models
        
        for subdir in models_path.iterdir():
            if not subdir.is_dir():
                continue
            
            config_file = subdir / SEDCompleter.CONFIG_FILE
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    # Completer configs have no model_type; SEDGenerator writes 'sed_generator'
                    if 'model_type' in config:
                        continue
                    
                    models.append({
                        'name': subdir.name,
                        'path': str(subdir),
                        'version': config.get('version', 'unknown'),
                        'framework': config.get('framework', 'unknown'),
                        'architecture': config.get('architecture', {}),
                    })
                except (OSError, ValueError) as exc:
                    logger.warning("Skipping %s: could not read its %s: %s", subdir, SEDCompleter.CONFIG_FILE, exc)

        return models


# =============================================================================
# CLI Interface
# =============================================================================

def cli_train(args):
    """CLI handler for training."""
    completer = SEDCompleter()
    completer.train(
        library_path=args.library,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        patience=args.patience,
    )


def cli_complete(args):
    """CLI handler for SED completion."""
    # Load model
    completer = SEDCompleter(args.model)
    
    # Load input SED
    data = np.loadtxt(args.input)
    wavelength = data[:, 0]
    flux = data[:, 1]
    
    # Complete
    wl_ext, flux_ext = completer.complete_sed(
        wavelength, flux,
        teff=args.teff,
        logg=args.logg,
        meta=args.meta,
        extension_range=(args.wl_min, args.wl_max),
    )
    
    # Save
    np.savetxt(
        args.output,
        np.column_stack([wl_ext, flux_ext]),
        header='wavelength_A flux_erg/s/cm2/A',
        fmt='%.6e',
    )
    print(f"Saved completed SED to: {args.output}")


def cli_list(args):
    """CLI handler for listing models."""
    models = SEDCompleter.list_models(args.models_dir)
    
    if not models:
        print(f"No models found in {args.models_dir}/")
        return
    
    print(f"\nAvailable models in {args.models_dir}/:")
    print("-" * 60)
    
    for m in models:
        arch = m.get('architecture', {})
        hidden = arch.get('hidden_layers', [])
        print(f"  {m['name']}")
        print(f"    Path: {m['path']}")
        print(f"    Framework: {m['framework']}, Version: {m['version']}")
        if hidden:
            print(f"    Architecture: {arch.get('input_dim', '?')} -> {hidden} -> {arch.get('output_dim', '?')}")
        print()


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ML-powered SED Completer (PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python -m sed_tools.ml_sed_completer train \\
      --library /path/to/stellar_library \\
      --output models/my_model

  # Complete an SED
  python -m sed_tools.ml_sed_completer complete \\
      --model models/my_model \\
      --input incomplete.txt \\
      --output extended.txt \\
      --teff 5500 --logg 4.5 --meta 0.0

  # List available models
  python -m sed_tools.ml_sed_completer list
        """,
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model on SED library')
    train_parser.add_argument('--library', required=True, help='Path to SED library')
    train_parser.add_argument('--output', required=True, help='Output model directory')
    train_parser.add_argument('--epochs', type=int, default=100, help='Max epochs (default: 100)')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    train_parser.add_argument('--max-samples', type=int, default=5000, help='Max training samples (default: 5000)')
    train_parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (default: 20)')
    train_parser.set_defaults(func=cli_train)
    
    # Complete command
    complete_parser = subparsers.add_parser('complete', help='Complete an incomplete SED')
    complete_parser.add_argument('--model', required=True, help='Model directory')
    complete_parser.add_argument('--input', required=True, help='Input SED file')
    complete_parser.add_argument('--output', required=True, help='Output file')
    complete_parser.add_argument('--teff', type=float, required=True, help='Effective temperature (K)')
    complete_parser.add_argument('--logg', type=float, required=True, help='Surface gravity')
    complete_parser.add_argument('--meta', type=float, required=True, help='Metallicity [M/H]')
    complete_parser.add_argument('--wl-min', type=float, default=100.0, help='Min wavelength (default: 100)')
    complete_parser.add_argument('--wl-max', type=float, default=100000.0, help='Max wavelength (default: 100000)')
    complete_parser.set_defaults(func=cli_complete)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    list_parser.add_argument('--models-dir', default='models', help='Models directory (default: models)')
    list_parser.set_defaults(func=cli_list)
    
    args = parser.parse_args()
    args.func(args)


# =============================================================================
# Interactive Workflow (for CLI integration)
# =============================================================================

def run_interactive_workflow(base_dir: str, models_dir: str = "models") -> None:
    """
    Interactive ML SED Completer workflow.
    
    Allows user to:
    1. Train new models on existing SED libraries
    2. Load trained models and extend incomplete SEDs
    
    Parameters
    ----------
    base_dir : str
        Base directory containing stellar model libraries
    models_dir : str
        Directory to store/load trained models (default: "models")
    """
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ML SED COMPLETER (PyTorch)")
    print("=" * 60)
    print("\nExtend incomplete SEDs using trained neural networks")
    
    print("\nWhat would you like to do?")
    print("  1) Train new model")
    print("  2) Complete/extend SEDs (inference)")
    print("  3) List available models")
    print("  0) Back")
    
    choice = input("> ").strip()
    if choice == "0":
        return
    if choice == "1":
        _train_menu(base_dir, models_dir)
    elif choice == "2":
        _extend_menu(base_dir, models_dir)
    elif choice == "3":
        _list_menu(models_dir)
    else:
        print("Invalid choice")


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _section(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def _menu_choice(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Read a 1-based menu number; print "Invalid selection" and return None for anything else."""
    selection = input("> ").strip()
    try:
        idx = int(selection) - 1
    except ValueError:
        print("Invalid selection")
        return None
    if idx < 0 or idx >= len(items):
        print("Invalid selection")
        return None
    return items[idx]


def _confirmed(prompt: str) -> bool:
    """Ask a [Y/n] question; anything but an empty or y... answer prints "Cancelled"."""
    answer = input(prompt).strip().lower()
    if answer and not answer.startswith('y'):
        print("Cancelled")
        return False
    return True


def _print_libraries(libraries: List[Dict[str, Any]]) -> None:
    for i, library in enumerate(libraries, 1):
        print(f"  {i:2d}) {library['name']:30s} ({library['n_seds']} SEDs)")


def _training_libraries(base_dir: str) -> List[Dict[str, Any]]:
    """Libraries under base_dir with a flux_cube.bin, and their SED counts ("?" if unknown)."""
    import pandas as pd
    
    libraries = []
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "flux_cube.bin")):
            continue
        
        lookup = os.path.join(path, "lookup_table.csv")
        n_seds = "?"
        if os.path.exists(lookup):
            try:
                n_seds = len(pd.read_csv(lookup))
            except (OSError, ValueError) as exc:
                logger.warning("Could not count the SEDs in %s: %s", lookup, exc)
        libraries.append({'name': name, 'path': path, 'n_seds': n_seds})
    return libraries


def _extendable_grids(base_dir: str) -> List[Dict[str, Any]]:
    """Grids under base_dir with a readable lookup_table.csv."""
    import pandas as pd
    
    grids = []
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        lookup = os.path.join(path, "lookup_table.csv")
        if not os.path.isdir(path) or not os.path.exists(lookup):
            continue
        try:
            df = pd.read_csv(lookup)
        except (OSError, ValueError) as exc:
            logger.warning("Not offering %s: could not read %s: %s", name, lookup, exc)
            continue
        grids.append({'name': name, 'path': path, 'n_seds': len(df)})
    return grids


def _train_menu(base_dir: str, models_dir: str) -> None:
    _section("TRAIN NEW MODEL")
    
    libraries = _training_libraries(base_dir)
    if not libraries:
        print(f"\nNo model libraries with flux_cube.bin found in {base_dir}")
        print("Download spectral grids first with 'sed-tools grids'")
        return
    
    print("\nAvailable model libraries for training:")
    print("-" * 60)
    _print_libraries(libraries)
    
    print("\nSelect library to train on (enter number):")
    selected = _menu_choice(libraries)
    if selected is None:
        return
    print(f"\nTraining on: {selected['name']}")
    
    print("\nTraining parameters:")
    epochs = _ask_number("  Epochs [100]: ", 100, whole=True, positive=True)
    batch_size = _ask_number("  Batch size [32]: ", 32, whole=True, positive=True)
    
    default_name = f"sed_model_{selected['name']}"
    model_name = input(f"  Model name [{default_name}]: ").strip() or default_name
    output_path = os.path.join(models_dir, model_name)
    
    print(f"\nModel will be saved to: {output_path}/")
    print("  - model.pt")
    print("  - config.json")
    print("  - training_plots.png")
    print("  - prediction_examples.png")
    if not _confirmed("\nStart training? [Y/n]: "):
        return
    
    _banner("TRAINING...")
    SEDCompleter().train(
        library_path=selected['path'],
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size,
    )
    _banner("TRAINING COMPLETE!")
    print(f"Model saved to: {output_path}/")


def _ask_extension_range(model_grid: np.ndarray) -> Tuple[float, float]:
    """Ask for the wavelength range to extend to; Enter keeps the model's own range."""
    default_lo, default_hi = float(model_grid[0]), float(model_grid[-1])
    print(f"\nWavelength range to extend to (the model covers {default_lo:g}-{default_hi:g} Å).")
    print("Every spectrum keeps all its own points; points are added only beyond its ends.")
    while True:
        lo = _ask_number(f"  Shortest wavelength in Å [{default_lo:g}]: ", default_lo, positive=True)
        hi = _ask_number(f"  Longest wavelength in Å [{default_hi:g}]: ", default_hi, positive=True)
        if lo < hi:
            return lo, hi
        print(f"  {lo:g} Å is not below {hi:g} Å; try again.")


def _extend_menu(base_dir: str, models_dir: str) -> None:
    from .ml import Completer
    
    _section("COMPLETE/EXTEND SEDs")
    
    trained_models = SEDCompleter.list_models(models_dir)
    if not trained_models:
        print(f"\nNo trained models found in {models_dir}/")
        print("Train a model first (option 1)")
        return
    
    print("\nAvailable trained spectral grids:")
    print("-" * 60)
    for i, model in enumerate(trained_models, 1):
        arch = model.get('architecture', {})
        print(f"  {i:2d}) {model['name']}")
        if arch:
            hidden = arch.get('hidden_layers', [])
            print(f"      Architecture: {arch.get('input_dim', '?')} -> {hidden} -> {arch.get('output_dim', '?')}")
    
    print("\nSelect model to use (enter number):")
    selected_model = _menu_choice(trained_models)
    if selected_model is None:
        return
    print(f"\nUsing model: {selected_model['name']}")
    completer = Completer(selected_model['path'])
    print("✓ Model loaded successfully")
    
    _section("SELECT MODEL SET TO EXTEND")
    grids = _extendable_grids(base_dir)
    if not grids:
        print(f"\nNo model sets found in {base_dir}")
        return
    
    print("\nAvailable model sets:")
    print("-" * 60)
    _print_libraries(grids)
    
    print("\nSelect model set to extend (enter number):")
    source = _menu_choice(grids)
    if source is None:
        return
    print(f"\nExtending: {source['name']}")
    extension_range = _ask_extension_range(completer.wavelength_grid)
    
    output_name = source['name'] + "_ml_extended"
    output_dir = os.path.join(base_dir, output_name)
    print(f"Output will be saved to: {output_name}/")
    if not _confirmed("Start extension? [Y/n]: "):
        return
    
    _banner("EXTENDING SEDs...")
    summary = completer.extend_catalog(
        input_dir=source['path'],
        output_dir=output_dir,
        extension_range=extension_range,
        verbose=True,
    )
    
    _banner("EXTENSION COMPLETE!")
    print(f"Extended model: {output_dir}")
    print(f"  {summary['extended']} SEDs extended successfully")
    print(f"  {summary.get('skipped', 0)} SEDs skipped (already covered range)")


def _list_menu(models_dir: str) -> None:
    _section("AVAILABLE TRAINED MODELS")
    
    models = SEDCompleter.list_models(models_dir)
    if not models:
        print(f"\nNo models found in {models_dir}/")
        return
    
    for m in models:
        arch = m.get('architecture', {})
        print(f"\n  {m['name']}")
        print(f"    Path: {m['path']}")
        print(f"    Framework: {m['framework']}, Version: {m['version']}")
        if arch:
            hidden = arch.get('hidden_layers', [])
            print(f"    Architecture: {arch.get('input_dim', '?')} -> {hidden} -> {arch.get('output_dim', '?')}")


# =============================================================================
# API wrappers expected by sed_tools/api.py
# =============================================================================

def train_model(
    grid_dir: str,
    save_path: str,
    epochs: int = 100,
    batch_size: int = 32,
    max_samples: int = 5000,
    **kwargs: Any,
):
    """
    Wrapper expected by MLCompleter in api.py.

    Trains a model using SEDCompleter.train(), saves it under save_path/,
    and returns a model handle suitable for extend_catalog(). Extra keyword
    arguments go to SEDCompleter.train(); one it does not take raises TypeError.
    """
    from .ml import Completer

    save_path = str(save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    completer = SEDCompleter()
    completer.train(
        library_path=str(grid_dir),
        output_path=save_path,
        epochs=int(epochs),
        batch_size=int(batch_size),
        max_samples=int(max_samples),
        **kwargs,
    )
    return Completer(save_path)


def load_model(model_path: str):
    """
    Wrapper expected by MLCompleter.load() in api.py.

    Returns an object that can be passed into extend_catalog().
    """
    from .ml import Completer

    return Completer(str(model_path))


def extend_catalog(
    model: Union[str, Path, Any],
    catalog_dir: str,
    output_dir: str,
    wavelength_range: Optional[Tuple[float, float]] = None,
    verbose: bool = False,
    **kwargs: Any,
):
    """
    Wrapper expected by MLCompleter.extend() in api.py.

    Extends a spectral grid directory (with lookup_table.csv + spectra files) into output_dir.
    """
    extension_range = wavelength_range or (DEFAULT_WL_MIN, DEFAULT_WL_MAX)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # If caller passed a path, load it
    if isinstance(model, (str, Path)):
        model = load_model(str(model))

    # If model provides extend_catalog (the preferred path), use it.
    if hasattr(model, "extend_catalog") and callable(getattr(model, "extend_catalog")):
        return model.extend_catalog(
            input_dir=str(catalog_dir),
            output_dir=str(output_dir),
            extension_range=extension_range,
            verbose=bool(verbose),
            **kwargs,
        )

    raise TypeError(
        "extend_catalog() expected a model with an .extend_catalog() method "
        "(e.g. sed_tools.ml.Completer) or a model directory path."
    )


if __name__ == '__main__':
    main()