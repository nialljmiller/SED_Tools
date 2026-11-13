"""SEDCompleter, the engine that extends an SED beyond its wavelength range: a black body refined by a neural network.

Also here: Completer, the 0.4.1 class over it that loads a model and extends a whole grid (extend_catalog), which
api.MLCompleter uses; and the sed-tools menu.
A model folder: model.pt, config.json (architecture, scalers, grid), training_plots.png and prediction_examples.png.
sed_tools.ml holds the 0.4.1 functions.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .._constants import H as PLANCK_H, C as SPEED_C, K as BOLTZMANN_K, SIGMA
from ..cube.fileio import load as load_cube
from ..ui_utils import (ask_number as _ask_number, banner as _banner, menu_choice as _menu_choice, refused_without,
                       section as _section)
from .common import (DEFAULT_WL_MAX, DEFAULT_WL_MIN, DEFAULT_WL_POINTS, Engine, Network, action_menu, confirmed,
                     data_loaders, fit, log_grid, model_configs, normalised_params, param_scaler, predict,
                     print_libraries, resample, sample_spectra, training_libraries, write_spectrum)
from .common import mlp  # noqa: F401  (kept: `from sed_tools.ml.completer import *` gives mlp, as before)

logger = logging.getLogger(__name__)

# Training uses only samples whose input window is at least this fraction valid.
MIN_WINDOW_COVERAGE = 0.8


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
    """The prediction blended into the black body. Its weight is exp(-log_dist / 0.5), log_dist being the distance
    from the data edge in decades (spectra are more uniform in log wavelength): it fades over about half a decade."""
    pred_interp = np.interp(wl_out, wl_pred, pred, left=pred[0], right=pred[-1])
    weight_ml = np.clip(np.exp(-log_dist / 0.5), 0, 1)
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
    exponent = np.clip(PLANCK_H * SPEED_C / (wavelength_cm * BOLTZMANN_K * temperature), 0, 700)
    denominator = np.exp(exponent) - 1.0
    denominator = np.where(denominator > 0, denominator, 1e-100)
    flux = numerator / denominator * 1e-8  # Convert to per Angstrom

    # Normalize to σT⁴. scipy is only in the [full] extra, so it is imported here, where it is
    # needed: listing or training models works without it.
    from scipy.integrate import simpson as simps
    integrated = simps(flux, wavelength_angstrom)
    if integrated > 0:
        flux = flux * (SIGMA * temperature**4 / integrated)
    return flux


class SEDCompleterNetwork(Network):
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
        self.input_dim = input_dim
        # The input is the SED window followed by the three stellar parameters
        super().__init__(input_dim + 3, hidden_layers or [256, 128, 64], output_dim, dropout)


class SEDCompleter(Engine):
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

    _create_wavelength_grid = staticmethod(log_grid)

    def _prepare_training_data(
        self,
        library_path: str,
        max_samples: int = 5000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(X, y, params, masks), float32: X a randomly placed input window, y the spectrum, masks 1 where y counts."""
        print("Loading flux cube...")
        teff, logg, meta, wavelengths, flux_cube = load_cube(os.path.join(library_path, "flux_cube.bin"))
        nt, nl, nm = len(teff), len(logg), len(meta)
        n_samples = min(max_samples, nt * nl * nm)
        print(f"  Grid: {nt} Teff × {nl} logg × {nm} [M/H] = {nt * nl * nm} total SEDs")
        print(f"  Sampling {n_samples} SEDs for training")

        self.wavelength_grid = self._create_wavelength_grid()
        # Training moves the input window around; inference uses the central one
        self._use_central_window()
        known_width = self._known_width
        params_list, samples = sample_spectra(
            teff, logg, meta, flux_cube, n_samples, lambda flux: self._training_sample(flux, wavelengths, known_width),
            library_path)
        X_list, y_list, masks_list = zip(*samples)
        print(f"  Prepared {len(X_list)} valid samples")
        print(f"  Known region width: {known_width} points (randomized position)")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        params = np.array(params_list, dtype=np.float32)
        masks = np.array(masks_list, dtype=np.float32)
        print(f"  Average extension coverage: {masks.mean() * 100:.1f}%")
        return X, y, params, masks

    def _use_central_window(self) -> None:
        """Make the network's input window the central 40% of the model grid."""
        n_wl = len(self.wavelength_grid)
        self._known_width = n_wl * 2 // 5
        self._known_start = (n_wl - self._known_width) // 2
        self._known_end = self._known_start + self._known_width

    def _training_sample(
        self,
        flux: np.ndarray,
        wavelengths: np.ndarray,
        known_width: int,
    ) -> Tuple[Optional[str], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """(None, (X_known, y_full, ext_mask)) cut from a spectrum at a random input window, or (reason, None)."""
        valid_original = (flux > 0) & np.isfinite(flux)
        if valid_original.sum() < 100:
            return "fewer than 100 positive, finite flux points", None
        flux_interp, valid_mask = resample(self.wavelength_grid, wavelengths, flux)

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
        X, y, params, masks = self._prepare_training_data(library_path, max_samples)
        X_combined, y_norm, masks = self._normalise_training_data(X, y, params, masks, library_path)
        X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
            X_combined, y_norm, masks, test_size=0.2, random_state=42)
        train_loader, val_loader = data_loaders((X_train, y_train, m_train), (X_val, y_val, m_val), batch_size)

        input_dim = X.shape[1]
        output_dim = y_norm.shape[1]
        print(f"Building network: {input_dim} + 3 -> {hidden_layers or [256, 128, 64]} -> {output_dim}")
        self.model = SEDCompleterNetwork(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers,
                                         dropout=dropout)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               min_lr=1e-6)
        history, best_val_loss = fit(
            self.model, train_loader, val_loader, optimizer, scheduler, epochs=epochs, patience=patience,
            verbose=verbose, device=self.device,
            progress="  Epoch {epoch:3d}: train_loss={train:.6f}, val_loss={val:.6f}, lr={lr:.2e}")
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
        """Fit self.scaler_params; (X_combined, y_norm, masks), float32: X_combined is the window, then the params."""
        print("Normalizing data...")
        # Floored at 1e-20, no log10 is below -20: the scalers use every input point, and every extension
        # point (each sample has at least 50).
        X_log = np.log10(np.maximum(X, 1e-20))
        y_log = np.log10(np.maximum(y, 1e-20))
        x_valid = X_log.ravel()
        y_valid = y_log[masks > 0.5]

        self.scaler_params = {
            **param_scaler(params),
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

        params_norm = np.column_stack(normalised_params(self.scaler_params, params[:, 0], params[:, 1], params[:, 2]))
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

        best_epoch = np.argmin(history['val_loss']) + 1
        best_loss = min(history['val_loss'])
        ax.axvline(best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best: epoch {best_epoch}')
        ax.annotate(f'Best: {best_loss:.4f}', xy=(best_epoch, best_loss), xytext=(best_epoch + 5, best_loss * 1.5),
                    arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')
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
        masks: np.ndarray,
        n_examples: int = 6,
    ):
        """Plot example predictions vs ground truth."""
        import matplotlib.pyplot as plt
        self.model.eval()
        indices = np.random.choice(len(X_val), min(n_examples, len(X_val)), replace=False)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (ax, idx) in enumerate(zip(axes, indices)):
            y_pred = predict(self.model, X_val[idx], self.device)
            y_true = y_val[idx]
            y_true_flux = 10 ** (y_true * self.scaler_params['ext_log_std'] + self.scaler_params['ext_log_mean'])
            y_pred_flux = 10 ** (y_pred * self.scaler_params['ext_log_std'] + self.scaler_params['ext_log_mean'])
            wl = self.wavelength_grid
            # The extension points: valid and outside the input window (at least 50 in every training sample)
            valid = masks[idx] > 0.5
            ax.loglog(wl[valid], y_true_flux[valid], 'b-', alpha=0.7, label='True', linewidth=1.5)
            ax.loglog(wl[valid], y_pred_flux[valid], 'r--', alpha=0.7, label='Predicted', linewidth=1.5)
            rel_error = np.median(np.abs(y_pred[valid] - y_true[valid]) / (np.abs(y_true[valid]) + 1e-8)) * 100
            coverage = valid.sum() / len(valid) * 100
            ax.set_title(f'Sample {i+1}: Error={rel_error:.1f}%, Coverage={coverage:.0f}%')
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
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        ax_flat = axes.flatten()

        for i, (fname, wl_in, flux_in, wl_out, flux_out) in enumerate(examples):
            ax = ax_flat[i]
            # The extended SED behind, in red; the original in front, in blue
            ax.loglog(wl_out, flux_out, 'r-', alpha=0.6, linewidth=2, label='Extended')
            ax.loglog(wl_in, flux_in, 'b-', alpha=0.8, linewidth=1.5, label='Original')
            ax.set_title(fname, fontsize=10, pad=5)
            ax.set_xlabel('Wavelength (Å)', fontsize=8)
            ax.set_ylabel('Flux', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, which='both')
            # One legend, on the first plot, to reduce clutter
            if i == 0:
                ax.legend(fontsize=8, loc='best')
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
        torch.save(self.model.state_dict(), path / self.MODEL_FILE)
        self.config = {
            'architecture': {
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'hidden_layers': hidden_layers or [256, 128, 64],
                'dropout': dropout,
            },
            'scaler_params': self.scaler_params,
            'wavelength_grid': self.wavelength_grid.tolist(),
            'regions': {key: getattr(self, f'_{key}', None) for key in ('known_start', 'known_end', 'known_width')},
            'version': '2.0.0',
            'framework': 'pytorch',
        }
        with open(path / self.CONFIG_FILE, 'w') as f:
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
        with open(path / self.CONFIG_FILE, 'r') as f:
            self.config = json.load(f)
        self.scaler_params = self.config['scaler_params']
        self.wavelength_grid = np.array(self.config['wavelength_grid'])

        regions = self.config.get('regions', {})
        self._known_start = regions.get('known_start')
        self._known_end = regions.get('known_end')
        self._known_width = regions.get('known_width')
        # A config without the stored window gets the central one
        if self._known_start is None or self._known_end is None:
            self._use_central_window()

        arch = self.config['architecture']
        self.model = SEDCompleterNetwork(input_dim=arch['input_dim'], output_dim=arch['output_dim'],
                                         hidden_layers=arch['hidden_layers'], dropout=arch['dropout'])
        self.model.load_state_dict(torch.load(path / self.MODEL_FILE, map_location=self.device, weights_only=True))
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
        known_start, known_end = self._known_start, self._known_end
        # The network predicts the whole grid; only the parts outside the
        # input window extend the SED (either part may be empty)
        wl_low = full_grid[need_low]
        flux_low = _blend_with_blackbody(
            wl_low, np.log10(wl_min) - np.log10(wl_low), self.wavelength_grid[:known_start], y_pred_flux[:known_start],
            bb_flux[need_low])
        wl_high = full_grid[need_high]
        flux_high = _blend_with_blackbody(
            wl_high, np.log10(wl_high) - np.log10(wl_max), self.wavelength_grid[known_end:], y_pred_flux[known_end:],
            bb_flux[need_high])

        wl_out = np.concatenate([wl_low, wavelength, wl_high])
        flux_out = np.concatenate([flux_low, flux, flux_high])
        print(f"Extended SED: {wl_min:.0f}-{wl_max:.0f} Å → {wl_out[0]:.0f}-{wl_out[-1]:.0f} Å "
              f"(its {len(wavelength)} points kept; {len(wl_low)} added below, {len(wl_high)} above)")
        return wl_out, flux_out

    def _predict_grid_flux(self, wavelength, flux, teff, logg, meta) -> np.ndarray:
        """The network's flux on self.wavelength_grid for an SED, fed the stored central window (1e-20 beyond it)."""
        _warn_window_coverage(self.wavelength_grid[self._known_start:self._known_end], wavelength)
        flux_on_grid = np.interp(self.wavelength_grid, wavelength, 10**np.log10(np.maximum(flux, 1e-20)),
                                 left=1e-20, right=1e-20)
        flux_known = flux_on_grid[self._known_start:self._known_end]
        flux_known_log = np.log10(np.maximum(flux_known, 1e-20))
        flux_known_norm = (flux_known_log - self.scaler_params['flux_log_mean']) / self.scaler_params['flux_log_std']
        flux_known_norm = np.where(np.isfinite(flux_known_norm), flux_known_norm, 0.0)

        params_norm = np.array(normalised_params(self.scaler_params, teff, logg, meta))
        X_input = np.concatenate([flux_known_norm, params_norm]).astype(np.float32)
        y_pred_norm = predict(self.model, X_input, self.device)
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
        for subdir, config in model_configs(models_dir, SEDCompleter.CONFIG_FILE, logger):
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
        return models


# Completer: the 0.4.1 class API (sed_tools.ml.Completer) over SEDCompleter, and its extend_catalog

class Completer:
    """
    High-level SED completion interface over a trained SEDCompleter model.
    
    Example
    -------
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
            Path to existing model directory. If None, call load() before using it.
        """
        self._completer = SEDCompleter(model_path)
        self._model_path = model_path

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
        return self._completer.complete_sed(wavelength=wavelength, flux=flux, teff=teff, logg=logg, meta=meta,
                                            extension_range=extension_range)

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
        ``input_dir/lookup_table.csv``, whose header line may start with
        '#' as sed_tools writes it. Spectra that already cover
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
            If the lookup table has no file-name, Teff, logg or [M/H] column
            (the names sed_tools.grids.lookup's find_*_column accept).
        RuntimeError
            After every other spectrum has been written, if any could not be
            read or extended; the message lists each one and why.
        """
        import pandas as pd
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
        """Extend, or copy, one catalog spectrum into output_dir: its new lookup row, and a plot example if extended.
        Raises ValueError or OSError, which extend_catalog collects, if the spectrum cannot be used."""
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
            write_spectrum(output_file, *data.T)  # every column of the file, as read
            new_row.update(extended_by_ml=False, file_name=filename)
            return new_row, None

        wl_ext, flux_ext = self.extend(wavelength, flux, teff=teff, logg=logg, meta=meta,
                                       extension_range=extension_range)
        write_spectrum(output_file, wl_ext, flux_ext)
        new_row.update(extended_by_ml=True, file_name=filename,
                       wl_min=float(wl_ext.min()), wl_max=float(wl_ext.max()))
        return new_row, (filename, wavelength, flux, wl_ext, flux_ext)

    @staticmethod
    def available_models(models_dir: str = "models") -> List[Dict[str, Any]]:
        """List the trained SED completer models in ``models_dir`` (not generator models)."""
        return SEDCompleter.list_models(models_dir)

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        path = f" @ {self._model_path}" if self._model_path else ""
        return f"<Completer({status}{path})>"


# Catalog helpers for Completer.extend_catalog

def _read_catalog_table(input_dir: str):
    """Read a catalog's lookup table and find its file-name and parameter columns, as every lookup reader does
    (sed_tools.grids.lookup's find_*_column)."""
    import pandas as pd

    from ..grids.lookup import find_file_column, find_logg_column, find_metallicity_column, find_teff_column
    lookup_path = os.path.join(input_dir, "lookup_table.csv")
    # sed_tools writes the header as '#file_name,teff,...'; comment='#' would drop it
    df = pd.read_csv(lookup_path)
    df.columns = [c.strip().lstrip('#').strip() for c in df.columns]
    columns = {'file': find_file_column(df.columns), 'teff': find_teff_column(df.columns),
               'logg': find_logg_column(df.columns), 'meta': find_metallicity_column(df.columns)}
    missing = [role for role, column in columns.items() if column is None]
    if missing:
        raise ValueError(f"{lookup_path} has no column for {', '.join(missing)}. Columns read: {list(df.columns)}")
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
    for path in (os.path.join(input_dir, filename), os.path.join(input_dir, os.path.basename(filename))):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{filename} not found in {input_dir}")



# Interactive workflow (sed-tools main menu item 8)

def run_interactive_workflow(base_dir: str, models_dir: str = "models") -> Optional[int]:
    """
    Interactive ML SED Completer workflow.
    
    Allows user to:
    1. Train new models on existing SED libraries
    2. Load trained models and extend incomplete SEDs

    Returns 1, saying what to install, when the chosen action needs a package
    that is not installed (training: torch and scikit-learn; extending: torch
    and scipy); listing models needs neither.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing stellar model libraries
    models_dir : str
        Directory to store/load trained models (default: "models")
    """
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    _banner("ML SED COMPLETER (PyTorch)")
    print("\nExtend incomplete SEDs using trained neural networks")
    return action_menu([
        ("Train new model", ("torch", "sklearn"), lambda: _train_menu(base_dir, models_dir)),
        ("Complete/extend SEDs (inference)", ("scipy", "torch"), lambda: _extend_menu(base_dir, models_dir)),
        ("List available models", (), lambda: _list_menu(models_dir)),
    ])


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
    libraries = training_libraries(base_dir)
    if not libraries:
        print(f"\nNo model libraries with flux_cube.bin found in {base_dir}")
        print("Download spectral grids first with 'sed-tools grids'")
        return
    print_libraries(libraries, "Available model libraries for training:")
    print("\nSelect library to train on (enter number):")
    selected = _menu_choice(libraries)
    if selected is None:
        return
    print(f"\nTraining on: {selected['name']}")

    print("\nTraining parameters:")
    epochs = _ask_number("  Epochs [100]: ", 100, whole=True)
    batch_size = _ask_number("  Batch size [32]: ", 32, whole=True)
    default_name = f"sed_model_{selected['name']}"
    model_name = input(f"  Model name [{default_name}]: ").strip() or default_name
    output_path = os.path.join(models_dir, model_name)
    print(f"\nModel will be saved to: {output_path}/")
    for name in ("model.pt", "config.json", "training_plots.png", "prediction_examples.png"):
        print(f"  - {name}")
    if not confirmed("\nStart training? [Y/n]: "):
        return

    _banner("TRAINING...")
    SEDCompleter().train(library_path=selected['path'], output_path=output_path, epochs=epochs, batch_size=batch_size)
    _banner("TRAINING COMPLETE!")
    print(f"Model saved to: {output_path}/")


def _ask_extension_range(model_grid: np.ndarray) -> Tuple[float, float]:
    """Ask for the wavelength range to extend to; Enter keeps the model's own range."""
    default_lo, default_hi = float(model_grid[0]), float(model_grid[-1])
    print(f"\nWavelength range to extend to (the model covers {default_lo:g}-{default_hi:g} Å).")
    print("Every spectrum keeps all its own points; points are added only beyond its ends.")
    while True:
        lo = _ask_number(f"  Shortest wavelength in Å [{default_lo:g}]: ", default_lo)
        hi = _ask_number(f"  Longest wavelength in Å [{default_hi:g}]: ", default_hi)
        if lo < hi:
            return lo, hi
        print(f"  {lo:g} Å is not below {hi:g} Å; try again.")


def _extend_menu(base_dir: str, models_dir: str) -> None:
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
    print_libraries(grids, "Available model sets:")
    print("\nSelect model set to extend (enter number):")
    source = _menu_choice(grids)
    if source is None:
        return
    print(f"\nExtending: {source['name']}")
    extension_range = _ask_extension_range(completer.wavelength_grid)
    output_name = source['name'] + "_ml_extended"
    output_dir = os.path.join(base_dir, output_name)
    print(f"Output will be saved to: {output_name}/")
    if not confirmed("Start extension? [Y/n]: "):
        return

    _banner("EXTENDING SEDs...")
    summary = completer.extend_catalog(input_dir=source['path'], output_dir=output_dir,
                                       extension_range=extension_range, verbose=True)
    _banner("EXTENSION COMPLETE!")
    print(f"Extended model: {output_dir}")
    print(f"  {summary['extended']} SEDs extended successfully")
    print(f"  {summary['skipped']} SEDs skipped (already covered range)")


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
