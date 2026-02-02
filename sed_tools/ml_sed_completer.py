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
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import simpson as simps

# Physical constants
PLANCK_H = 6.62607015e-27   # erg·s
SPEED_C = 2.99792458e10     # cm/s
BOLTZMANN_K = 1.380649e-16  # erg/K
SIGMA = 5.670374419e-5      # Stefan-Boltzmann constant erg/s/cm²/K⁴

# Default wavelength grid parameters
DEFAULT_WL_MIN = 100.0      # Angstroms
DEFAULT_WL_MAX = 100000.0   # Angstroms
DEFAULT_WL_POINTS = 1000


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
            import torch
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def __call__(self, x):
        return self.forward(x)
    
    def train_mode(self):
        self.model.train()
    
    def eval_mode(self):
        self.model.eval()
    
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
            import torch
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    def _check_combined_grid(self, library_path: str) -> bool:
        """Check if library is a combined grid with varying wavelength coverage."""
        import pandas as pd
        
        lookup_path = os.path.join(library_path, "lookup_table.csv")
        if not os.path.exists(lookup_path):
            return False
        
        try:
            df = pd.read_csv(lookup_path, comment='#', nrows=5)
            return 'source_model' in df.columns or 'original_file' in df.columns
        except Exception:
            return False
    
    def _build_coverage_map(
        self,
        library_path: str,
        teff_grid: np.ndarray,
        logg_grid: np.ndarray,
        meta_grid: np.ndarray,
    ) -> Dict[Tuple[float, float, float], Tuple[float, float]]:
        """
        Build wavelength coverage map for combined grids.
        
        Returns dict mapping (teff, logg, meta) -> (wl_min, wl_max)
        """
        import pandas as pd
        
        coverage = {}
        lookup_path = os.path.join(library_path, "lookup_table.csv")
        
        if not os.path.exists(lookup_path):
            return coverage
        
        try:
            df = pd.read_csv(lookup_path, comment='#')
            
            for _, row in df.iterrows():
                t = float(row.get('teff', row.get('Teff', 0)))
                g = float(row.get('logg', row.get('log_g', 0)))
                m = float(row.get('meta', row.get('metallicity', row.get('[M/H]', 0))))
                
                # Get wavelength range from lookup if available
                wl_min = float(row.get('wl_min', row.get('wavelength_min', 0)))
                wl_max = float(row.get('wl_max', row.get('wavelength_max', 1e6)))
                
                if wl_min > 0 and wl_max > wl_min:
                    coverage[(t, g, m)] = (wl_min, wl_max)
        except Exception:
            pass
        
        return coverage
    
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
        
        # Create wavelength grid
        self.wavelength_grid = self._create_wavelength_grid()
        n_wl = len(self.wavelength_grid)
        
        # Fixed input size for network (40% of grid - but position varies)
        known_width = n_wl * 2 // 5  # 400 points for 1000-point grid
        
        # Store for architecture (fixed size needed for network input)
        self._known_width = known_width
        # These will be set per-sample during training, but we need defaults for inference
        self._known_start = (n_wl - known_width) // 2
        self._known_end = self._known_start + known_width
        
        X_list = []
        y_list = []
        params_list = []
        masks_list = []
        known_regions = []  # Store (start, end) for each sample
        
        # Get original wavelength coverage
        wl_orig_min = wavelengths.min()
        wl_orig_max = wavelengths.max()
        
        # Random sampling
        indices = np.random.permutation(total_seds)[:n_samples]
        
        for idx in indices:
            i_t = idx // (nl * nm)
            remainder = idx % (nl * nm)
            i_l = remainder // nm
            i_m = remainder % nm
            
            # Get flux from cube (shape: [nw, nm, nl, nt])
            flux = flux_cube[:, i_m, i_l, i_t]
            
            # Skip spectra with invalid values in original data
            valid_original = (flux > 0) & np.isfinite(flux)
            if valid_original.sum() < 100:
                continue
            
            # Interpolate to common grid (use NaN for out-of-range)
            flux_interp = np.interp(
                self.wavelength_grid, 
                wavelengths, 
                flux,
                left=np.nan, 
                right=np.nan
            )
            
            # Build validity mask based on original wavelength coverage
            valid_mask = (
                (self.wavelength_grid >= wl_orig_min) & 
                (self.wavelength_grid <= wl_orig_max) &
                np.isfinite(flux_interp) &
                (flux_interp > 0)
            )
            
            # Randomize known region position
            # Ensure we stay within valid data regions
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) < known_width:
                continue
            
            # Random start position within valid range
            max_start = valid_indices[-1] - known_width
            min_start = valid_indices[0]
            if max_start <= min_start:
                continue
            
            known_start = np.random.randint(min_start, max_start)
            known_end = known_start + known_width
            
            # Check known region has enough valid data
            known_valid = valid_mask[known_start:known_end].sum()
            if known_valid < known_width * 0.8:  # Need 80% valid in known region
                continue
            
            # Extension mask: everything outside known region that has valid data
            ext_mask = valid_mask.copy()
            ext_mask[known_start:known_end] = False
            
            if ext_mask.sum() < 50:  # Need at least 50 valid extension points
                continue
            
            # Extract known region
            X_known = flux_interp[known_start:known_end].copy()
            
            # Full spectrum as target (network learns to predict everything, masked loss handles rest)
            y_full = flux_interp.copy()
            
            # Replace NaN/invalid with small positive value (will be masked during training)
            X_known = np.where(np.isfinite(X_known) & (X_known > 0), X_known, 1e-20)
            y_full = np.where(np.isfinite(y_full) & (y_full > 0), y_full, 1e-20)
            
            X_list.append(X_known)
            y_list.append(y_full)
            params_list.append([teff[i_t], logg[i_l], meta[i_m]])
            masks_list.append(ext_mask.astype(np.float32))
            known_regions.append((known_start, known_end))
        
        if len(X_list) == 0:
            raise ValueError("No valid training samples found")
        
        print(f"  Prepared {len(X_list)} valid samples")
        print(f"  Known region width: {known_width} points (randomized position)")
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        params = np.array(params_list, dtype=np.float32)
        masks = np.array(masks_list, dtype=np.float32)
        
        # Store known regions for potential debugging
        self._training_known_regions = known_regions
        
        # Report coverage statistics
        avg_coverage = masks.mean() * 100
        print(f"  Average extension coverage: {avg_coverage:.1f}%")
        
        return X, y, params, masks
    
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
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        
        # Prepare data (now always returns masks)
        X, y, params, masks = self._prepare_training_data(library_path, max_samples)
        
        # Log-scale flux - only on valid (positive) values
        print("Normalizing data...")
        
        # Use a floor value that won't create extreme outliers
        FLUX_FLOOR = 1e-20
        X_safe = np.maximum(X, FLUX_FLOOR)
        y_safe = np.maximum(y, FLUX_FLOOR)
        
        X_log = np.log10(X_safe)
        y_log = np.log10(y_safe)
        
        # Compute scaler parameters only from VALID data points
        # For X (known region), all values should be valid
        x_valid = X_log[X_log > -25]  # Exclude floor values
        
        # For y (extension), use mask to select valid values
        y_valid_mask = (masks > 0.5) & (y_log > -25)
        y_valid = y_log[y_valid_mask]
        
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
        
        # Normalize
        X_norm = (X_log - self.scaler_params['flux_log_mean']) / self.scaler_params['flux_log_std']
        y_norm = (y_log - self.scaler_params['ext_log_mean']) / self.scaler_params['ext_log_std']
        
        # Replace invalid normalized values with 0 (will be masked anyway)
        X_norm = np.where(np.isfinite(X_norm), X_norm, 0.0)
        y_norm = np.where(np.isfinite(y_norm), y_norm, 0.0)
        
        params_norm = np.column_stack([
            (params[:, 0] - self.scaler_params['teff_mean']) / self.scaler_params['teff_std'],
            (params[:, 1] - self.scaler_params['logg_mean']) / self.scaler_params['logg_std'],
            (params[:, 2] - self.scaler_params['meta_mean']) / self.scaler_params['meta_std'],
        ])
        
        # Combine input features
        X_combined = np.concatenate([X_norm, params_norm], axis=1).astype(np.float32)
        y_norm = y_norm.astype(np.float32)
        masks = masks.astype(np.float32)
        
        print(f"  Input shape: {X_combined.shape}")
        print(f"  Output shape: {y_norm.shape}")
        print(f"  Masks shape: {masks.shape}")
        
        # Verify no NaN/inf in data
        assert np.all(np.isfinite(X_combined)), "NaN/inf in X_combined"
        assert np.all(np.isfinite(y_norm)), "NaN/inf in y_norm"
        
        # Train/val split (include masks)
        X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
            X_combined, y_norm, masks, test_size=0.2, random_state=42
        )
        
        # Create PyTorch datasets (include masks)
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(m_train),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
            torch.from_numpy(m_val),
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Build model
        input_dim = X_norm.shape[1]
        output_dim = y_norm.shape[1]
        
        print(f"Building network: {input_dim} + 3 -> {hidden_layers or [256, 128, 64]} -> {output_dim}")
        self.model = SEDCompleterNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        
        # Masked MSE loss function
        def masked_mse_loss(y_pred, y_true, mask):
            """MSE loss that only considers valid (masked) regions."""
            # mask: 1 = valid, 0 = invalid
            sq_error = (y_pred - y_true) ** 2
            masked_error = sq_error * mask
            # Average over valid points only
            n_valid = mask.sum() + 1e-8  # Avoid division by zero
            return masked_error.sum() / n_valid
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        print(f"\nTraining for up to {epochs} epochs...")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Training
            self.model.train_mode()
            train_losses = []
            
            for X_batch, y_batch, m_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                m_batch = m_batch.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = masked_mse_loss(y_pred, y_batch, m_batch)
                
                if torch.isnan(loss):
                    print(f"  Warning: NaN loss at epoch {epoch+1}, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval_mode()
            val_losses = []
            
            with torch.no_grad():
                for X_batch, y_batch, m_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    m_batch = m_batch.to(self.device)
                    
                    y_pred = self.model(X_batch)
                    loss = masked_mse_loss(y_pred, y_batch, m_batch)
                    
                    if not torch.isnan(loss):
                        val_losses.append(loss.item())
            
            if not train_losses or not val_losses:
                print(f"  Epoch {epoch+1}: No valid batches, stopping")
                break
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={lr:.2e}")
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        print("-" * 60)
        print(f"Training complete. Best val_loss: {best_val_loss:.6f}")
        
        # Save model
        self.save(output_path, hidden_layers=hidden_layers, dropout=dropout)
        
        # Generate plots
        print("\nGenerating training plots...")
        self._plot_training_results(history, output_path)
        self._plot_prediction_examples(X_val, y_val, output_path, masks=m_val)
        
        return history
    
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
        
        self.model.eval_mode()
        
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
            filename: str = "inference_examples.png"
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
            import os
            
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
        self.model.eval_mode()
        
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
        
        Parameters
        ----------
        wavelength : np.ndarray
            Existing wavelength coverage (Angstroms)
        flux : np.ndarray
            Existing flux (erg/s/cm²/Å)
        teff : float
            Effective temperature (K)
        logg : float
            Surface gravity (log cm/s²)
        meta : float
            Metallicity [M/H]
        extension_range : tuple
            (min_wavelength, max_wavelength) for output SED
        
        Returns
        -------
        wavelength_extended : np.ndarray
            Extended wavelength array
        flux_extended : np.ndarray
            Extended flux array
        """
        import torch
        
        if self.model is None:
            raise RuntimeError("No model loaded. Train or load a model first.")
        
        self.model.eval_mode()
        
        # Create output wavelength grid
        full_grid = self._create_wavelength_grid(extension_range[0], extension_range[1])
        
        # Check what needs extension
        wl_min, wl_max = wavelength.min(), wavelength.max()
        need_low = full_grid < wl_min
        need_high = full_grid > wl_max
        
        if not (need_low.any() or need_high.any()):
            print("SED already covers requested range")
            return wavelength, flux
        
        # Calculate black body baseline
        bb_flux = planck_function(full_grid, teff)
        
        # Initialize output with black body
        flux_extended = bb_flux.copy()
        
        # Resample input to training grid
        flux_safe = np.maximum(flux, 1e-20)
        flux_log = np.log10(flux_safe)
        flux_on_grid = np.interp(self.wavelength_grid, wavelength, 10**flux_log, left=1e-20, right=1e-20)
        
        # Use stored region indices
        known_start = self._known_start
        known_end = self._known_end
        
        # Prepare model input
        flux_known = flux_on_grid[known_start:known_end]
        flux_known_log = np.log10(np.maximum(flux_known, 1e-20))
        flux_known_norm = (flux_known_log - self.scaler_params['flux_log_mean']) / self.scaler_params['flux_log_std']
        
        # Handle any NaN/inf from normalization
        flux_known_norm = np.where(np.isfinite(flux_known_norm), flux_known_norm, 0.0)
        
        params_norm = self._normalize_params(teff, logg, meta)
        
        X_input = np.concatenate([flux_known_norm, params_norm]).astype(np.float32)
        X_tensor = torch.from_numpy(X_input[np.newaxis, :]).to(self.device)
        
        # Predict extension
        with torch.no_grad():
            y_pred_norm = self.model(X_tensor).cpu().numpy()[0]
        
        # Denormalize
        y_pred_log = y_pred_norm * self.scaler_params['ext_log_std'] + self.scaler_params['ext_log_mean']
        y_pred_flux = 10 ** y_pred_log
        
        # Network now outputs full spectrum - extract UV and IR extension regions
        pred_low = y_pred_flux[:known_start]
        pred_high = y_pred_flux[known_end:]
        
        wl_pred_low = self.wavelength_grid[:known_start]
        wl_pred_high = self.wavelength_grid[known_end:]
        
        # Blend ML predictions with black body using log-wavelength distance
        # This is physically motivated: stellar spectra behave more uniformly in log space
        # blend_decades controls how far (in decades of wavelength) we trust ML before fading to BB
        blend_decades = 0.5  # Trust ML for ~0.5 decades (factor of ~3), then blend toward BB
        
        if need_low.any():
            idx_low = np.where(need_low)[0]
            wl_low = full_grid[idx_low]
            
            # Interpolate prediction to output grid
            pred_interp = np.interp(wl_low, wl_pred_low, pred_low, left=pred_low[0], right=pred_low[-1])
            
            # Log-space distance from data edge (in decades)
            log_dist = np.log10(wl_min) - np.log10(wl_low)
            weight_ml = np.exp(-log_dist / blend_decades)
            weight_ml = np.clip(weight_ml, 0, 1)
            
            flux_extended[idx_low] = weight_ml * pred_interp + (1 - weight_ml) * bb_flux[idx_low]
        
        if need_high.any():
            idx_high = np.where(need_high)[0]
            wl_high = full_grid[idx_high]
            
            pred_interp = np.interp(wl_high, wl_pred_high, pred_high, left=pred_high[0], right=pred_high[-1])
            
            # Log-space distance from data edge (in decades)
            log_dist = np.log10(wl_high) - np.log10(wl_max)
            weight_ml = np.exp(-log_dist / blend_decades)
            weight_ml = np.clip(weight_ml, 0, 1)
            
            flux_extended[idx_high] = weight_ml * pred_interp + (1 - weight_ml) * bb_flux[idx_high]
        
        # Insert original data in overlap region
        overlap_mask = (full_grid >= wl_min) & (full_grid <= wl_max)
        if overlap_mask.any():
            flux_extended[overlap_mask] = np.interp(full_grid[overlap_mask], wavelength, flux)
        
        print(f"Extended SED: {wavelength.min():.0f}-{wavelength.max():.0f} Å → {full_grid.min():.0f}-{full_grid.max():.0f} Å")
        
        return full_grid, flux_extended
    
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
            Model info: name, path, version, etc.
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
                    
                    models.append({
                        'name': subdir.name,
                        'path': str(subdir),
                        'version': config.get('version', 'unknown'),
                        'framework': config.get('framework', 'unknown'),
                        'architecture': config.get('architecture', {}),
                    })
                except Exception:
                    pass
        
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
    import pandas as pd
    
    # Ensure directories exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ML SED COMPLETER (PyTorch)")
    print("=" * 60)
    print("\nExtend incomplete SEDs using trained neural networks")
    
    # Main choice: train or complete
    print("\nWhat would you like to do?")
    print("  1) Train new model")
    print("  2) Complete/extend SEDs (inference)")
    print("  3) List available models")
    print("  0) Back")
    
    choice = input("> ").strip()
    
    if choice == "0":
        return
    
    elif choice == "1":
        # TRAINING MODE
        print("\n" + "-" * 60)
        print("TRAIN NEW MODEL")
        print("-" * 60)
        
        # Discover available model libraries
        available_models = []
        for name in sorted(os.listdir(base_dir)):
            model_path = os.path.join(base_dir, name)
            if not os.path.isdir(model_path):
                continue
            
            # Check for flux_cube.bin (required for training)
            flux_cube = os.path.join(model_path, "flux_cube.bin")
            if os.path.exists(flux_cube):
                # Count SEDs in lookup table
                lookup = os.path.join(model_path, "lookup_table.csv")
                n_seds = "?"
                if os.path.exists(lookup):
                    try:
                        df = pd.read_csv(lookup, comment='#')
                        n_seds = len(df)
                    except Exception:
                        pass
                
                available_models.append({
                    'name': name,
                    'path': model_path,
                    'n_seds': n_seds
                })
        
        if not available_models:
            print(f"\nNo model libraries with flux_cube.bin found in {base_dir}")
            print("Download some models first (option 1 in main menu)")
            return
        
        # Display available models
        print("\nAvailable model libraries for training:")
        print("-" * 60)
        for i, model in enumerate(available_models, 1):
            print(f"  {i:2d}) {model['name']:30s} ({model['n_seds']} SEDs)")
        
        # Select model
        print("\nSelect library to train on (enter number):")
        selection = input("> ").strip()
        
        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(available_models):
                print("Invalid selection")
                return
            selected = available_models[idx]
        except (ValueError, IndexError):
            print("Invalid selection")
            return
        
        print(f"\nTraining on: {selected['name']}")
        
        # Training parameters
        print("\nTraining parameters:")
        epochs_str = input("  Epochs [100]: ").strip() or "100"
        batch_size_str = input("  Batch size [32]: ").strip() or "32"
        
        try:
            epochs = int(epochs_str)
            batch_size = int(batch_size_str)
        except ValueError:
            print("Invalid parameters")
            return
        
        # Model name - now creates a DIRECTORY
        default_name = f"sed_model_{selected['name']}"
        model_name = input(f"  Model name [{default_name}]: ").strip() or default_name
        output_path = os.path.join(models_dir, model_name)
        
        print(f"\nModel will be saved to: {output_path}/")
        print(f"  - model.pt")
        print(f"  - config.json")
        print(f"  - training_plots.png")
        print(f"  - prediction_examples.png")
        
        confirm = input("\nStart training? [Y/n]: ").strip().lower()
        
        if confirm and not confirm.startswith('y'):
            print("Cancelled")
            return
        
        # Train
        print("\n" + "=" * 60)
        print("TRAINING...")
        print("=" * 60)
        
        completer = SEDCompleter()
        history = completer.train(
            library_path=selected['path'],
            output_path=output_path,
            epochs=epochs,
            batch_size=batch_size,
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Model saved to: {output_path}/")
    
    elif choice == "2":
        # INFERENCE/COMPLETION MODE
        print("\n" + "-" * 60)
        print("COMPLETE/EXTEND SEDs")
        print("-" * 60)
        
        # Discover trained models (look for directories with config.json)
        trained_models = SEDCompleter.list_models(models_dir)
        
        if not trained_models:
            print(f"\nNo trained models found in {models_dir}/")
            print("Train a model first (option 1)")
            return
        
        # Select trained model
        print("\nAvailable trained models:")
        print("-" * 60)
        for i, model in enumerate(trained_models, 1):
            arch = model.get('architecture', {})
            print(f"  {i:2d}) {model['name']}")
            if arch:
                hidden = arch.get('hidden_layers', [])
                print(f"      Architecture: {arch.get('input_dim', '?')} -> {hidden} -> {arch.get('output_dim', '?')}")
        
        print("\nSelect model to use (enter number):")
        selection = input("> ").strip()
        
        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(trained_models):
                print("Invalid selection")
                return
            selected_model = trained_models[idx]
        except (ValueError, IndexError):
            print("Invalid selection")
            return
        
        print(f"\nUsing model: {selected_model['name']}")
        
        # Load model
        completer = SEDCompleter(selected_model['path'])
        print("✓ Model loaded successfully")
        
        # Select source model set to complete
        print("\n" + "-" * 60)
        print("SELECT MODEL SET TO EXTEND")
        print("-" * 60)
        
        available_sources = []
        for name in sorted(os.listdir(base_dir)):
            model_path = os.path.join(base_dir, name)
            if not os.path.isdir(model_path):
                continue
            
            # Check for lookup table
            lookup = os.path.join(model_path, "lookup_table.csv")
            if os.path.exists(lookup):
                try:
                    df = pd.read_csv(lookup, comment='#')
                    n_seds = len(df)
                    available_sources.append({
                        'name': name,
                        'path': model_path,
                        'n_seds': n_seds
                    })
                except Exception:
                    pass
        
        if not available_sources:
            print(f"\nNo model sets found in {base_dir}")
            return
        
        # Display models
        print("\nAvailable model sets:")
        print("-" * 60)
        for i, model in enumerate(available_sources, 1):
            print(f"  {i:2d}) {model['name']:30s} ({model['n_seds']} SEDs)")
        
        print("\nSelect model set to extend (enter number):")
        selection = input("> ").strip()
        
        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(available_sources):
                print("Invalid selection")
                return
            source_model = available_sources[idx]
        except (ValueError, IndexError):
            print("Invalid selection")
            return
        
        print(f"\nExtending: {source_model['name']}")
        
        # Output directory
        output_name = source_model['name'] + "_ml_extended"
        output_dir = os.path.join(base_dir, output_name)
        
        print(f"Output will be saved to: {output_name}/")
        confirm = input("Start extension? [Y/n]: ").strip().lower()
        
        if confirm and not confirm.startswith('y'):
            print("Cancelled")
            return
        
        # Run extension
        print("\n" + "=" * 60)
        print("EXTENDING SEDs...")
        print("=" * 60)
        
        # Use the Completer class for batch processing
        from .ml import Completer
        comp = Completer(selected_model['path'])
        
        summary = comp.extend_catalog(
            input_dir=source_model['path'],
            output_dir=output_dir,
            extension_range=(100.0, 100000.0),
            verbose=True,
        )
        
        print("\n" + "=" * 60)
        print("EXTENSION COMPLETE!")
        print("=" * 60)
        print(f"Extended model: {output_dir}")
        print(f"  {summary['extended']} SEDs extended successfully")
        print(f"  {summary.get('skipped', 0)} SEDs skipped (already covered range)")
        if summary.get('failed', 0):
            print(f"  {summary['failed']} SEDs failed")





    elif choice == "3":
        # LIST MODELS
        print("\n" + "-" * 60)
        print("AVAILABLE TRAINED MODELS")
        print("-" * 60)
        
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
    
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()