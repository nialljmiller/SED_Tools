"""SEDGenerator, the engine that makes a whole SED from (Teff, logg, [M/H]) alone, with no input spectrum.

A model folder: model.pt, config.json (model_type 'sed_generator', scalers, grid, parameter ranges) and two plots.
Old import path sed_tools.ml_sed_generator, also run as python -m. The completer engine is ml/completer.py.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..cube.fileio import load as load_cube
from .common import (DEFAULT_WL_MAX, DEFAULT_WL_MIN, DEFAULT_WL_POINTS, action_menu, data_loaders, fit, log_grid, mlp,
                     model_configs, normalised_params, param_scaler, print_libraries, resample, sample_spectra,
                     training_libraries)
from ..ui_utils import (ask_number as _ask_number, ask_numbers as _ask_numbers, banner as _banner,
                       menu_choice as _menu_choice, refused_without, section as _section)

logger = logging.getLogger("sed_tools.ml_sed_generator")


class SEDGeneratorNetwork:
    """
    PyTorch neural network for SED generation.
    
    Maps 3 stellar parameters to full SED flux array.
    """
    
    def __init__(
        self,
        output_dim: int,
        hidden_layers: List[int] = None,
        dropout: float = 0.2,
    ):
        """
        Initialize network.
        
        Parameters
        ----------
        output_dim : int
            Number of wavelength points to output
        hidden_layers : list of int
            Hidden layer sizes. Default: [512, 512, 256, 256]
        dropout : float
            Dropout rate
        """
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers or [512, 512, 256, 256]
        self.dropout = dropout
        # The input is Teff, logg and [M/H]
        self.model = mlp(3, self.hidden_layers, output_dim, dropout)
        # The same device SEDGenerator.device picks, so inputs and weights agree
        from .common import get_device
        self.device = get_device()
        self.model.to(self.device)
    
    def forward(self, x):
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


def _optimizer_and_scheduler(model: SEDGeneratorNetwork, learning_rate: float):
    """The optimizer and learning-rate schedule SEDGenerator.train uses (and AutoTrainer's trials)."""
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    return optimizer, scheduler


class SEDGenerator:
    """
    Generate complete SEDs from stellar parameters using neural networks.
    
    Usage
    -----
    Training::
    
        gen = SEDGenerator()
        gen.train('path/to/library', 'models/my_gen', epochs=200)
    
    Generating::
    
        gen = SEDGenerator('models/my_gen')
        wl, flux = gen.generate(teff=5500, logg=4.5, meta=0.0)
    """
    
    MODEL_FILE = "model.pt"
    CONFIG_FILE = "config.json"
    TRAINING_PLOT = "training_plots.png"
    PREDICTION_PLOT = "prediction_examples.png"
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize generator.
        
        Parameters
        ----------
        model_path : str or Path, optional
            Path to trained model directory. If provided, loads the model.
        """
        self.model: Optional[SEDGeneratorNetwork] = None
        self.config: Dict[str, Any] = {}
        self.wavelength_grid: Optional[np.ndarray] = None
        self.scaler_params: Optional[Dict[str, float]] = None
        self._device = None
        
        if model_path is not None:
            self.load(model_path)
    
    @property
    def device(self):
        if self._device is None:
            from .common import get_device
            self._device = get_device()
        return self._device
    
    _create_wavelength_grid = staticmethod(log_grid)
    
    def _load_flux_cube(self, library_path: str):
        """Load flux cube from binary file."""
        return load_cube(os.path.join(library_path, "flux_cube.bin"))
    
    def _prepare_training_data(
        self,
        library_path: str,
        max_samples: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """params (N, 3: teff, logg, meta), flux and validity masks (N, n_wavelengths), float32, from the flux cube."""
        print("Loading flux cube...")
        teff_grid, logg_grid, meta_grid, wavelengths, flux_cube = self._load_flux_cube(library_path)
        
        nt, nl, nm = len(teff_grid), len(logg_grid), len(meta_grid)
        total_seds = nt * nl * nm
        
        print(f"  Grid: {nt} Teff × {nl} logg × {nm} [M/H] = {total_seds} total SEDs")
        print(f"  Original wavelength: {wavelengths.min():.0f} - {wavelengths.max():.0f} Å")
        
        self.wavelength_grid = self._create_wavelength_grid()
        
        n_samples = min(max_samples, total_seds)
        print(f"  Processing {n_samples} SEDs...")
        params_list, samples = sample_spectra(
            teff_grid, logg_grid, meta_grid, flux_cube, n_samples,
            lambda flux: self._training_sample(flux, wavelengths), library_path,
        )
        flux_list, masks_list = zip(*samples)
        
        params = np.array(params_list, dtype=np.float32)
        flux = np.array(flux_list, dtype=np.float32)
        masks = np.array(masks_list, dtype=np.float32)
        
        print(f"  Prepared {len(params)} valid samples")
        print(f"  Teff: {params[:, 0].min():.0f} - {params[:, 0].max():.0f} K")
        print(f"  logg: {params[:, 1].min():.2f} - {params[:, 1].max():.2f}")
        print(f"  [M/H]: {params[:, 2].min():.2f} - {params[:, 2].max():.2f}")
        print(f"  Average coverage: {masks.mean() * 100:.1f}%")
        
        return params, flux, masks

    def _training_sample(
        self,
        flux: np.ndarray,
        wavelengths: np.ndarray,
    ) -> Tuple[Optional[str], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """(None, (flux, valid_mask)) on the model grid, 1e-30 and 0 outside the coverage; or (reason, None)."""
        if np.any(flux <= 0) or np.any(~np.isfinite(flux)):
            return "non-positive or non-finite flux", None
        
        flux_interp, valid_mask = resample(self.wavelength_grid, wavelengths, flux)
        if valid_mask.sum() < len(self.wavelength_grid) * 0.3:
            return "less than 30% of the model grid covered", None
        
        flux_interp = np.where(valid_mask, flux_interp, 1e-30)
        return None, (flux_interp, valid_mask.astype(np.float32))
    
    def train(
        self,
        library_path: str,
        output_path: str,
        epochs: int = 200,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        max_samples: int = 10000,
        hidden_layers: List[int] = None,
        dropout: float = 0.2,
        patience: int = 30,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the generator on stellar spectral grids.
        
        Parameters
        ----------
        library_path : str
            Path to SED library with flux_cube.bin
        output_path : str
            Directory to save trained model
        epochs : int
            Maximum epochs
        batch_size : int
            Batch size
        learning_rate : float
            Initial learning rate
        max_samples : int
            Max training samples
        hidden_layers : list
            Hidden layer sizes
        dropout : float
            Dropout rate
        patience : int
            Early stopping patience
        verbose : bool
            Print progress
        
        Returns
        -------
        history : dict with 'train_loss' and 'val_loss'
        """
        train_set, val_set = self._training_split(library_path, max_samples)
        p_val, f_val, m_val = val_set
        train_loader, val_loader = data_loaders(train_set, val_set, batch_size)
        
        output_dim = f_val.shape[1]
        hidden = hidden_layers or [512, 512, 256, 256]
        print(f"Building network: 3 → {hidden} → {output_dim}")
        self.model = SEDGeneratorNetwork(
            output_dim=output_dim,
            hidden_layers=hidden,
            dropout=dropout,
        )
        
        optimizer, scheduler = _optimizer_and_scheduler(self.model, learning_rate)
        history, best_val_loss = fit(
            self.model, train_loader, val_loader, optimizer, scheduler,
            epochs=epochs, patience=patience, verbose=verbose, device=self.device,
            progress="  Epoch {epoch:3d}: train={train:.6f}, val={val:.6f}, lr={lr:.2e}",
        )
        
        print("-" * 60)
        print(f"Training complete. Best val_loss: {best_val_loss:.6f}")
        
        self.save(output_path, hidden_layers=hidden, dropout=dropout)
        
        print("\nGenerating plots...")
        self._plot_training(history, output_path)
        self._plot_predictions(p_val, f_val, m_val, output_path)
        
        return history

    def _training_split(
        self,
        library_path: str,
        max_samples: int,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The normalised (params, flux, masks) training and validation sets, split 80/20.
        AutoTrainer (ml/tuning.py) uses it too, so its trials see the data train() does."""
        from sklearn.model_selection import train_test_split

        params, flux, masks = self._prepare_training_data(library_path, max_samples)
        params_norm, flux_norm = self._normalise_training_data(params, flux, masks, library_path)

        # Split indices, so the raw parameters of the training set can be kept
        idx_train, idx_val = train_test_split(np.arange(len(params_norm)), test_size=0.2, random_state=42)
        # save() stores these in config.json for the parameter-space plot
        self._train_params_raw = params[idx_train]

        train_set = (params_norm[idx_train], flux_norm[idx_train], masks[idx_train])
        val_set = (params_norm[idx_val], flux_norm[idx_val], masks[idx_val])
        return train_set, val_set

    def _normalise_training_data(
        self,
        params: np.ndarray,
        flux: np.ndarray,
        masks: np.ndarray,
        library_path: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit self.scaler_params (the flux scaler on valid points only); the normalised parameters and log10 fluxes."""
        print("Normalizing data...")
        flux_log = np.log10(np.maximum(flux, 1e-30))
        
        flux_valid = flux_log[(masks > 0.5) & (flux_log > -25)]
        if len(flux_valid) == 0:
            raise ValueError(f"No valid data points for normalization in {library_path}")
        
        self.scaler_params = {
            **param_scaler(params, with_range=True),
            'flux_log_mean': float(np.mean(flux_valid)),
            'flux_log_std': float(np.std(flux_valid) + 1e-8),
        }
        
        print(f"  Flux log10 range: [{flux_valid.min():.2f}, {flux_valid.max():.2f}]")
        
        params_norm = np.column_stack(
            normalised_params(self.scaler_params, params[:, 0], params[:, 1], params[:, 2])).astype(np.float32)
        
        flux_norm = (flux_log - self.scaler_params['flux_log_mean']) / self.scaler_params['flux_log_std']
        flux_norm = np.where(np.isfinite(flux_norm), flux_norm, 0.0).astype(np.float32)
        
        print(f"  Input: {params_norm.shape}, Output: {flux_norm.shape}")
        return params_norm, flux_norm
    
    def _plot_training(self, history: Dict, output_path: str):
        """Plot training curves."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('SED Generator Training', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        best_epoch = np.argmin(history['val_loss']) + 1
        best_loss = min(history['val_loss'])
        ax.axvline(best_epoch, color='g', linestyle='--', alpha=0.7)
        ax.annotate(f'Best: {best_loss:.4f}', xy=(best_epoch, best_loss),
                   xytext=(best_epoch + 5, best_loss * 1.5),
                   arrowprops=dict(arrowstyle='->'), fontsize=10, color='green')
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, self.TRAINING_PLOT), dpi=150)
        plt.close(fig)
        print(f"  Saved: {self.TRAINING_PLOT}")
    
    def _plot_predictions(self, p_val, f_val, m_val, output_path: str, n=6):
        """Plot prediction examples."""
        import matplotlib.pyplot as plt
        import torch
        
        self.model.eval()
        
        indices = np.random.choice(len(p_val), min(n, len(p_val)), replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for ax, idx in zip(axes, indices):
            with torch.no_grad():
                pred = self.model(torch.from_numpy(p_val[idx:idx+1]).to(self.device)).cpu().numpy()[0]
            
            true = f_val[idx]
            
            true_flux = 10 ** (true * self.scaler_params['flux_log_std'] + self.scaler_params['flux_log_mean'])
            pred_flux = 10 ** (pred * self.scaler_params['flux_log_std'] + self.scaler_params['flux_log_mean'])
            
            teff = p_val[idx, 0] * self.scaler_params['teff_std'] + self.scaler_params['teff_mean']
            logg = p_val[idx, 1] * self.scaler_params['logg_std'] + self.scaler_params['logg_mean']
            meta = p_val[idx, 2] * self.scaler_params['meta_std'] + self.scaler_params['meta_mean']
            
            valid = m_val[idx] > 0.5
            if valid.sum() > 0:
                ax.loglog(self.wavelength_grid[valid], true_flux[valid], 'b-', 
                         alpha=0.7, label='True', linewidth=1.5)
                ax.loglog(self.wavelength_grid[valid], pred_flux[valid], 'r--', 
                         alpha=0.7, label='Generated', linewidth=1.5)
                
                err = np.median(np.abs(pred[valid] - true[valid])) * 100
                ax.set_title(f'T={teff:.0f}K, g={logg:.1f}, [M/H]={meta:.1f}\nErr={err:.1f}%', fontsize=10)
            
            ax.set_xlabel('Wavelength (Å)')
            ax.set_ylabel('Flux')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('SED Generator: Predictions vs Truth', fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, self.PREDICTION_PLOT), dpi=150)
        plt.close(fig)
        print(f"  Saved: {self.PREDICTION_PLOT}")
    
    def save(self, path: Union[str, Path], hidden_layers: List[int] = None, dropout: float = 0.2):
        """Save model to directory."""
        import torch
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), path / self.MODEL_FILE)
        
        # At most 500 training points, for the parameter-space plot of generate_with_outputs
        train_params = getattr(self, '_train_params_raw', None)
        train_params_list = []
        if train_params is not None:
            if len(train_params) > 500:
                train_params = train_params[np.random.choice(len(train_params), 500, replace=False)]
            train_params_list = train_params.tolist()
        
        self.config = {
            'model_type': 'sed_generator',
            'architecture': {
                'output_dim': self.model.output_dim,
                'hidden_layers': hidden_layers or [512, 512, 256, 256],
                'dropout': dropout,
            },
            'scaler_params': self.scaler_params,
            'wavelength_grid': self.wavelength_grid.tolist(),
            'parameter_ranges': {
                name: [self.scaler_params[f'{name}_min'], self.scaler_params[f'{name}_max']]
                for name in ('teff', 'logg', 'meta')
            },
            'training_params': train_params_list,
            'version': '1.0.0',
            'framework': 'pytorch',
        }
        
        with open(path / self.CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Model saved to: {path}")
    
    def load(self, path: Union[str, Path]):
        """Load model from directory."""
        import torch
        
        path = Path(path)
        
        with open(path / self.CONFIG_FILE, 'r') as f:
            self.config = json.load(f)
        
        self.scaler_params = self.config['scaler_params']
        self.wavelength_grid = np.array(self.config['wavelength_grid'])
        
        arch = self.config['architecture']
        self.model = SEDGeneratorNetwork(
            output_dim=arch['output_dim'],
            hidden_layers=arch['hidden_layers'],
            dropout=arch['dropout'],
        )
        
        self.model.load_state_dict(torch.load(path / self.MODEL_FILE, map_location=self.device, weights_only=True))
        self.model.eval()
        
        print(f"Loaded generator from: {path}")
    
    def generate(
        self,
        teff: float,
        logg: float,
        meta: float,
        check_bounds: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate an SED from stellar parameters.
        
        Parameters
        ----------
        teff : float
            Effective temperature (K)
        logg : float
            Surface gravity (log cm/s²)
        meta : float
            Metallicity [M/H]
        check_bounds : bool
            Warn if parameters outside training range
        
        Returns
        -------
        wavelength : array (Angstroms)
        flux : array (erg/s/cm²/Å)
        """
        import torch
        
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        if check_bounds:
            ranges = self.config.get('parameter_ranges', {})
            for key, label, value, spec in (('teff', 'Teff', teff, '.0f'), ('logg', 'logg', logg, '.2f'),
                                            ('meta', '[M/H]', meta, '.2f')):
                if key not in ranges:
                    continue
                low, high = ranges[key][0], ranges[key][1]
                if value < low or value > high:
                    print(f"  Warning: {label}={value:{spec}} outside range [{low:{spec}}, {high:{spec}}]")
        
        self.model.eval()
        
        params_norm = np.array(normalised_params(self.scaler_params, teff, logg, meta), dtype=np.float32)
        
        with torch.no_grad():
            pred = self.model(torch.from_numpy(params_norm[None, :]).to(self.device))
            flux_norm = pred.cpu().numpy()[0]
        
        flux_log = flux_norm * self.scaler_params['flux_log_std'] + self.scaler_params['flux_log_mean']
        return self.wavelength_grid.copy(), 10 ** flux_log
    
    def generate_with_outputs(
        self,
        teff: float,
        logg: float,
        meta: float,
        output_dir: str,
        check_bounds: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate an SED and save data file plus diagnostic figures.
        
        Creates:
            - sed_T{teff}_g{logg}_m{meta}.txt  - The SED data
            - sed_T{teff}_g{logg}_m{meta}_spectrum.png - SED plot
            - sed_T{teff}_g{logg}_m{meta}_params.png - Parameter space plot
        
        Parameters
        ----------
        teff : float
            Effective temperature (K)
        logg : float
            Surface gravity (log cm/s²)
        meta : float
            Metallicity [M/H]
        output_dir : str
            Directory to save outputs
        check_bounds : bool
            Warn if parameters outside training range
        
        Returns
        -------
        wavelength : array (Angstroms)
        flux : array (erg/s/cm²/Å)
        """
        wl, flux = self.generate(teff, logg, meta, check_bounds=check_bounds)
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"sed_T{teff:.0f}_g{logg:.2f}_m{meta:+.2f}"
        
        data_file = os.path.join(output_dir, f"{base_name}.txt")
        np.savetxt(
            data_file,
            np.column_stack([wl, flux]),
            header=f'wavelength_A flux_erg/s/cm2/A\n# Teff={teff} logg={logg} [M/H]={meta}',
            fmt='%.6e',
        )
        print(f"  Saved SED data: {data_file}")
        
        # Wide defaults for models saved without their training ranges
        ranges = self.config.get('parameter_ranges', {})
        teff_range = ranges.get('teff', [3000, 50000])
        logg_range = ranges.get('logg', [0, 5])
        meta_range = ranges.get('meta', [-5, 1])
        outside_range = any(value < bounds[0] or value > bounds[1]
                            for value, bounds in ((teff, teff_range), (logg, logg_range), (meta, meta_range)))
        
        spectrum_file = os.path.join(output_dir, f"{base_name}_spectrum.png")
        _plot_generated_sed(wl, flux, teff, logg, meta, outside_range, spectrum_file)
        print(f"  Saved SED plot: {spectrum_file}")
        
        params_file = os.path.join(output_dir, f"{base_name}_params.png")
        _plot_parameter_space(self.config.get('training_params', []), teff, logg, meta,
                              teff_range, logg_range, meta_range, params_file)
        print(f"  Saved parameter space plot: {params_file}")
        
        return wl, flux
    
    @staticmethod
    def list_models(models_dir: str = "models") -> List[Dict[str, Any]]:
        """List available trained generators."""
        models = []
        for subdir, config in model_configs(models_dir, SEDGenerator.CONFIG_FILE, logger):
            if config.get('model_type') == 'sed_generator':
                models.append({
                    'name': subdir.name,
                    'path': str(subdir),
                    'parameter_ranges': config.get('parameter_ranges', {}),
                    'architecture': config.get('architecture', {}),
                })
        return models


def _plot_generated_sed(wl, flux, teff, logg, meta, outside_range: bool, path: str) -> None:
    """Plot a generated SED over shaded wavelength regions."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.loglog(wl, flux, 'b-', linewidth=1.5, label='Generated SED')
    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Flux (erg/s/cm²/Å)', fontsize=12)
    ax.set_title(f'Generated SED: Teff={teff:.0f} K, logg={logg:.2f}, [M/H]={meta:+.2f}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    if outside_range:
        ax.annotate('⚠ Parameters outside training range', 
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=10, color='red', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.axvspan(100, 912, alpha=0.1, color='purple', label='EUV')
    ax.axvspan(912, 4000, alpha=0.1, color='blue', label='UV')
    ax.axvspan(4000, 7000, alpha=0.1, color='green', label='Optical')
    ax.axvspan(7000, 25000, alpha=0.1, color='red', label='NIR')
    ax.axvspan(25000, 100000, alpha=0.1, color='darkred', label='MIR')
    
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_parameter_space(train_params, teff, logg, meta, teff_range, logg_range, meta_range, path: str) -> None:
    """Plot the generated point against the training parameters: 3-D, Teff-logg and Teff-[M/H]."""
    import matplotlib.pyplot as plt
    
    train_arr = np.array(train_params) if train_params else None
    fig = plt.figure(figsize=(14, 5))
    
    ax3d = fig.add_subplot(131, projection='3d')
    if train_arr is not None:
        ax3d.scatter(train_arr[:, 0], train_arr[:, 1], train_arr[:, 2],
                     c='blue', alpha=0.3, s=10, label='Training data')
    ax3d.scatter([teff], [logg], [meta], c='red', s=200, marker='*', 
                 edgecolors='black', linewidths=1.5, label='Generated', zorder=10)
    ax3d.set_xlabel('Teff (K)')
    ax3d.set_ylabel('logg')
    ax3d.set_zlabel('[M/H]')
    ax3d.set_title('3D Parameter Space')
    ax3d.legend(loc='upper left', fontsize=8)
    
    projections = [(132, 1, logg, logg_range, 'logg', 'Teff vs logg'),
                   (133, 2, meta, meta_range, '[M/H]', 'Teff vs [M/H]')]
    for position, column, value, value_range, ylabel, title in projections:
        ax = fig.add_subplot(position)
        if train_arr is not None:
            ax.scatter(train_arr[:, 0], train_arr[:, column], c='blue', alpha=0.3, s=10, label='Training')
        ax.scatter(teff, value, c='red', s=200, marker='*', edgecolors='black', 
                   linewidths=1.5, label='Generated', zorder=10)
        ax.axvline(teff_range[0], color='gray', linestyle='--', alpha=0.5)
        ax.axvline(teff_range[1], color='gray', linestyle='--', alpha=0.5)
        ax.axhline(value_range[0], color='gray', linestyle='--', alpha=0.5)
        ax.axhline(value_range[1], color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Teff (K)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Parameter Space: Teff={teff:.0f} K, logg={logg:.2f}, [M/H]={meta:+.2f}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# Interactive workflow (sed-tools main menu item 9)

def run_interactive_workflow(base_dir: str, models_dir: str = "models") -> Optional[int]:
    """
    Interactive ML SED Generator workflow.
    
    Called from the CLI menu. Returns 1, saying what to install, when the
    chosen action needs a package that is not installed (training: torch and
    scikit-learn; generating: torch); listing generators needs neither.
    """
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    _banner("ML SED GENERATOR (PyTorch)")
    print("\nGenerate complete SEDs from stellar parameters alone")
    print("Input: (Teff, logg, [M/H]) → Output: Full SED")
    return action_menu([
        ("Train new generator", ("torch", "sklearn"), lambda: _train_menu(base_dir, models_dir)),
        ("Generate SED", ("torch",), lambda: _generate_menu(models_dir)),
        ("Batch generate (grid of parameters)", ("torch",), lambda: _batch_menu(models_dir)),
        ("List available generators", (), lambda: _list_menu(models_dir)),
    ])


def _choose_generator(models_dir: str, show_teff_range: bool) -> Optional[Dict[str, Any]]:
    """List the generators in models_dir and read the user's choice."""
    generators = SEDGenerator.list_models(models_dir)
    if not generators:
        print(f"\nNo generators found in {models_dir}/")
        return None
    
    print("\nAvailable generators:")
    for i, g in enumerate(generators, 1):
        if show_teff_range:
            teff_r = g.get('parameter_ranges', {}).get('teff', [0, 0])
            print(f"  {i}) {g['name']} (Teff: {teff_r[0]:.0f}-{teff_r[1]:.0f} K)")
        else:
            print(f"  {i}) {g['name']}")
    return _menu_choice(generators, "\nSelect generator: > ")


def _ask_output_dir(default: str) -> str:
    print(f"Output directory [{default}]: ", end="")
    return input().strip() or default


def _train_menu(base_dir: str, models_dir: str) -> None:
    _section("TRAIN NEW GENERATOR")
    
    libraries = training_libraries(base_dir)
    if not libraries:
        print(f"\nNo libraries with flux_cube.bin found in {base_dir}")
        return
    
    print_libraries(libraries, "Available libraries:")
    
    selected = _menu_choice(libraries, "\nSelect library: > ")
    if selected is None:
        return
    print(f"\nTraining on: {selected['name']}")
    
    epochs = _ask_number("  Epochs [200]: ", 200, whole=True, positive=True)
    batch_size = _ask_number("  Batch size [64]: ", 64, whole=True, positive=True)
    
    default_name = f"sed_generator_{selected['name']}"
    model_name = input(f"  Model name [{default_name}]: ").strip() or default_name
    output_path = os.path.join(models_dir, model_name)
    
    print(f"\nModel will be saved to: {output_path}/")
    if input("Start training? [Y/n]: ").strip().lower() not in ('', 'y', 'yes'):
        return
    
    _banner("TRAINING...")
    SEDGenerator().train(selected['path'], output_path, epochs=epochs, batch_size=batch_size)
    _banner("TRAINING COMPLETE!")


def _generate_menu(models_dir: str) -> None:
    _section("GENERATE SED")
    
    selected = _choose_generator(models_dir, show_teff_range=True)
    if selected is None:
        return
    gen = SEDGenerator(selected['path'])
    
    print("\nEnter stellar parameters:")
    teff = _ask_number("  Teff (K): ", positive=True)
    logg = _ask_number("  logg: ")
    meta = _ask_number("  [M/H]: ")
    
    print()
    output_dir = _ask_output_dir(os.path.join(selected['path'], "SED"))
    
    print("\nGenerating SED with diagnostics...")
    wl, flux = gen.generate_with_outputs(teff, logg, meta, output_dir)
    
    print(f"\n✓ Generated: {len(wl)} points, {wl.min():.0f}-{wl.max():.0f} Å")
    print(f"✓ All outputs saved to: {output_dir}/")


def _batch_menu(models_dir: str) -> None:
    from itertools import product
    
    _section("BATCH GENERATE (Parameter Grid)")
    
    selected = _choose_generator(models_dir, show_teff_range=False)
    if selected is None:
        return
    gen = SEDGenerator(selected['path'])
    
    print("\nDefine parameter grid (comma-separated values):")
    print("  Example: 5000,5500,6000")
    teff_vals = _ask_numbers("  Teff values: ")
    logg_vals = _ask_numbers("  logg values: ")
    meta_vals = _ask_numbers("  [M/H] values: ")
    
    n_total = len(teff_vals) * len(logg_vals) * len(meta_vals)
    print(f"\nWill generate {n_total} SEDs")
    
    output_dir = _ask_output_dir(os.path.join(selected['path'], "SED", "batch"))
    
    print(f"\nGenerating {n_total} SEDs...")
    for count, (t, g, m) in enumerate(product(teff_vals, logg_vals, meta_vals), 1):
        gen.generate_with_outputs(t, g, m, output_dir)
        if count % 10 == 0:
            print(f"  Generated {count}/{n_total}...")
    
    print(f"\n✓ Generated {n_total} SEDs in {output_dir}/")


def _list_menu(models_dir: str) -> None:
    _section("AVAILABLE GENERATORS")
    
    generators = SEDGenerator.list_models(models_dir)
    if not generators:
        print(f"\nNo generators found in {models_dir}/")
        return
    
    for g in generators:
        r = g.get('parameter_ranges', {})
        arch = g.get('architecture', {})
        
        print(f"\n  {g['name']}")
        print(f"    Path: {g['path']}")
        if r:
            print(f"    Teff: {r.get('teff', [0,0])[0]:.0f} - {r.get('teff', [0,0])[1]:.0f} K")
            print(f"    logg: {r.get('logg', [0,0])[0]:.2f} - {r.get('logg', [0,0])[1]:.2f}")
            print(f"    [M/H]: {r.get('meta', [0,0])[0]:.2f} - {r.get('meta', [0,0])[1]:.2f}")
        if arch:
            print(f"    Architecture: 3 → {arch.get('hidden_layers', [])} → {arch.get('output_dim', '?')}")


# python -m sed_tools.ml_sed_generator

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML SED Generator")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    train_p = subparsers.add_parser('train', help='Train generator')
    train_p.add_argument('--library', required=True)
    train_p.add_argument('--output', required=True)
    train_p.add_argument('--epochs', type=int, default=200)
    train_p.add_argument('--batch-size', type=int, default=64)
    
    gen_p = subparsers.add_parser('generate', help='Generate SED')
    gen_p.add_argument('--model', required=True)
    gen_p.add_argument('--teff', type=float, required=True)
    gen_p.add_argument('--logg', type=float, required=True)
    gen_p.add_argument('--meta', type=float, required=True)
    gen_p.add_argument('--output', required=True)
    
    list_p = subparsers.add_parser('list', help='List generators')
    list_p.add_argument('--models-dir', default='models')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        SEDGenerator().train(args.library, args.output, epochs=args.epochs, batch_size=args.batch_size)
    
    elif args.command == 'generate':
        wl, flux = SEDGenerator(args.model).generate(args.teff, args.logg, args.meta)
        np.savetxt(args.output, np.column_stack([wl, flux]), 
                  header=f'wavelength_A flux_erg/s/cm2/A\n# Teff={args.teff} logg={args.logg} [M/H]={args.meta}',
                  fmt='%.6e')
        print(f"Saved: {args.output}")
    
    elif args.command == 'list':
        for g in SEDGenerator.list_models(args.models_dir):
            print(f"{g['name']}: {g['path']}")


if __name__ == '__main__':
    main()
