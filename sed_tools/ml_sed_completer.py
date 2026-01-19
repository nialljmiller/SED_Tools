#!/usr/bin/env python3
"""
ML-Powered SED Completer
=========================

Extends incomplete SEDs to broader wavelength ranges using:
1. Black body radiation as physical baseline
2. Neural network trained on complete SEDs for refinement

The tool only appends/extends - never modifies existing wavelength coverage.

COMBINED GRID SUPPORT
---------------------
The completer now supports training on combined stellar grids created by
combine_stellar_atm.py. When training on a combined grid:

1. Automatically detects combined grids (via source_model in lookup_table.csv)
2. Tracks individual wavelength coverage for each spectrum
3. Uses masked training to only learn from valid data regions
4. Handles varying wavelength ranges and resolutions across libraries

Usage with combined grid:
    # First, create combined grid
    python combine_stellar_atm.py
    # Select models to combine (e.g., Kurucz + PHOENIX + NextGen)
    
    # Then train on the combined grid
    python ml_sed_completer.py train \\
        --library ../data/stellar_models/combined_models \\
        --output models/combined_model
    
The training will automatically:
- Sample from all contributing libraries
- Respect individual wavelength coverage
- Apply proper masking during training
- Create a universal model that works across different datasets
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import simpson as simps
from sklearn.model_selection import train_test_split

# Constants
PLANCK_H = 6.62607015e-27  # erg·s
SPEED_C = 2.99792458e10    # cm/s
BOLTZMANN_K = 1.380649e-16 # erg/K
SIGMA = 5.670374419e-5     # Stefan-Boltzmann constant erg/s/cm²/K⁴


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
    # Convert Angstroms to cm
    wavelength_cm = wavelength_angstrom * 1e-8
    
    # Planck function: B_λ(T) = (2hc²/λ⁵) * 1/(exp(hc/λkT) - 1)
    numerator = 2.0 * PLANCK_H * SPEED_C**2 / wavelength_cm**5
    exponent = PLANCK_H * SPEED_C / (wavelength_cm * BOLTZMANN_K * temperature)
    
    # Avoid overflow
    exponent = np.clip(exponent, 0, 700)
    
    denominator = np.exp(exponent) - 1.0
    denominator = np.where(denominator > 0, denominator, 1e-100)
    
    flux = numerator / denominator
    
    # Convert to per Angstrom
    flux = flux * 1e-8
    
    # Normalize to σT⁴
    integrated = simps(flux, wavelength_angstrom)
    target = SIGMA * temperature**4
    
    if integrated > 0:
        flux = flux * (target / integrated)
    
    return flux


class SEDCompleter:
    """
    Complete incomplete SEDs using black body + neural network predictions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the SED completer.
        
        Parameters
        ----------
        model_path : str, optional
            Path to pre-trained model (without .keras extension). If None, will need to train.
        """
        self.model = None
        self.scaler_params = None
        self.wavelength_grid = None
        
        if model_path:
            # Check if .keras file exists (model_path should be without extension)
            keras_file = model_path + '.keras'
            params_file = model_path + '.params'
            if os.path.exists(keras_file) and os.path.exists(params_file):
                self.load_model(model_path)
            elif os.path.exists(model_path):
                # Legacy: if full path provided, try to use it
                self.load_model(model_path)
            else:
                print(f"Warning: Model files not found at {model_path}")
                print(f"  Expected: {keras_file} and {params_file}")
    
    def _create_wavelength_grid(self, min_wl: float = 100.0, max_wl: float = 100000.0, 
                                n_points: int = 1000) -> np.ndarray:
        """Create logarithmic wavelength grid."""
        return np.logspace(np.log10(min_wl), np.log10(max_wl), n_points)
    
    def _normalize_parameters(self, teff: float, logg: float, meta: float) -> np.ndarray:
        """Normalize stellar parameters for model input."""
        if self.scaler_params is None:
            # Default normalization ranges
            teff_norm = (teff - 5000.0) / 10000.0
            logg_norm = (logg - 3.0) / 3.0
            meta_norm = meta / 2.0
        else:
            teff_norm = (teff - self.scaler_params['teff_mean']) / self.scaler_params['teff_std']
            logg_norm = (logg - self.scaler_params['logg_mean']) / self.scaler_params['logg_std']
            meta_norm = (meta - self.scaler_params['meta_mean']) / self.scaler_params['meta_std']
        
        return np.array([teff_norm, logg_norm, meta_norm])
    
    def train_from_library(self, sed_library_path: str, output_model_path: str,
                          epochs: int = 100, batch_size: int = 32):
        """
        Train the neural network on complete SEDs from library.
        
        Parameters
        ----------
        sed_library_path : str
            Path to directory with flux_cube.bin files
        output_model_path : str
            Where to save trained model
        epochs : int
            Training epochs
        batch_size : int
            Batch size for training
        """
        import tensorflow as tf
        from tensorflow import keras
        
        print("Loading training data from SED library...")
        X_train, y_train, params_train = self._load_training_data(sed_library_path)
        
        if X_train is None or len(X_train) == 0:
            raise ValueError("No training data loaded")
        
        print(f"Loaded {len(X_train)} training samples")
        
        # Get masks if available (for combined grids with variable coverage)
        masks = getattr(self, 'training_masks', None)
        if masks is not None:
            print(f"  Using masked training (variable wavelength coverage)")
        
        # Log-scale flux data to handle many orders of magnitude
        print("Normalizing flux data...")
        X_log = np.log10(np.maximum(X_train, 1e-50))
        y_log = np.log10(np.maximum(y_train, 1e-50))
        
        # Calculate normalization parameters for flux
        self.scaler_params = {
            'teff_mean': np.mean(params_train[:, 0]),
            'teff_std': np.std(params_train[:, 0]) + 1e-8,
            'logg_mean': np.mean(params_train[:, 1]),
            'logg_std': np.std(params_train[:, 1]) + 1e-8,
            'meta_mean': np.mean(params_train[:, 2]),
            'meta_std': np.std(params_train[:, 2]) + 1e-8,
            'flux_log_mean': np.mean(X_log),
            'flux_log_std': np.std(X_log) + 1e-8,
            'ext_log_mean': np.mean(y_log),
            'ext_log_std': np.std(y_log) + 1e-8,
        }
        
        print(f"  Flux log10 range: [{X_log.min():.2f}, {X_log.max():.2f}]")
        print(f"  Extension log10 range: [{y_log.min():.2f}, {y_log.max():.2f}]")
        
        # Normalize flux data
        X_norm = (X_log - self.scaler_params['flux_log_mean']) / self.scaler_params['flux_log_std']
        y_norm = (y_log - self.scaler_params['ext_log_mean']) / self.scaler_params['ext_log_std']
        
        # Normalize parameters
        params_norm = np.column_stack([
            (params_train[:, 0] - self.scaler_params['teff_mean']) / self.scaler_params['teff_std'],
            (params_train[:, 1] - self.scaler_params['logg_mean']) / self.scaler_params['logg_std'],
            (params_train[:, 2] - self.scaler_params['meta_mean']) / self.scaler_params['meta_std'],
        ])
        
        print(f"  Normalized input: mean={X_norm.mean():.3f}, std={X_norm.std():.3f}")
        print(f"  Normalized output: mean={y_norm.mean():.3f}, std={y_norm.std():.3f}")
        
        # Build model architecture
        print("Building neural network...")
        self.model = self._build_model(X_norm.shape[1], y_norm.shape[1])
        
        # Combine SED features with parameters
        X_combined = np.concatenate([X_norm, params_norm], axis=1)
        
        # Train
        print("Training...")
        
        # Add early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Split data for validation manually to keep examples
        from sklearn.model_selection import train_test_split
        
        if masks is not None:
            # Split with masks
            X_train_split, X_val_split, y_train_split, y_val_split, mask_train, mask_val = train_test_split(
                X_combined, y_norm, masks, test_size=0.2, random_state=42
            )
            
            # Create masked loss function
            import tensorflow as tf
            
            def masked_mse(y_true, y_pred):
                """MSE that only considers valid (masked) regions."""
                # mask_val is a TensorFlow constant created from the batch's masks
                # For now, we'll use sample_weight in fit() instead
                return tf.reduce_mean(tf.square(y_true - y_pred))
            
            # Use sample weights to mask invalid regions
            # Create sample weights: 1.0 for valid pixels, 0.0 for invalid
            sample_weights_train = mask_train.astype(np.float32)
            sample_weights_val = mask_val.astype(np.float32)
            
            history = self.model.fit(
                X_train_split, y_train_split,
                sample_weight=sample_weights_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_split, y_val_split, sample_weights_val),
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
        else:
            # Standard training without masks
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_combined, y_norm, test_size=0.2, random_state=42
            )
            
            history = self.model.fit(
                X_train_split, y_train_split,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_split, y_val_split),
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
        
        # Plot training results
        print("\nGenerating training plots...")
        self._plot_training_results(
            history, 
            X_val_split, 
            y_val_split,
            output_model_path
        )
        
        # Save model
        self.save_model(output_model_path)
        print(f"Model saved to {output_model_path}")
        
        return history
    
    def _build_model(self, input_dim: int, output_dim: int) -> 'keras.Model':
        """Build neural network architecture."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim + 3,)),  # SED + parameters
            
            # Smaller, more stable architecture
            layers.Dense(256, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            layers.Dense(128, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            layers.Dense(64, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Dense(output_dim, kernel_initializer='glorot_uniform')
        ])
        
        # Lower learning rate and gradient clipping for stability
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,  # Much lower than default
            clipnorm=1.0           # Clip gradients to prevent explosion
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _load_training_data(self, library_path: str) -> Tuple[Optional[np.ndarray], 
                                                                Optional[np.ndarray],
                                                                Optional[np.ndarray]]:
        """
        Load complete SEDs from library for training.
        
        Handles both single libraries and combined grids with varying wavelength coverage.
        For combined grids, properly masks regions where individual spectra don't have data.
        """
        import struct
        from glob import glob

        import pandas as pd
        
        flux_cubes = glob(library_path + "/flux_cube.bin")
        
        if not flux_cubes:
            print(f"No flux_cube.bin files found in {library_path}")
            return None, None, None
        
        all_seds = []
        all_extensions = []
        all_params = []
        all_masks = []  # Track which regions have real data
        
        # Create target wavelength grid
        self.wavelength_grid = self._create_wavelength_grid()
        
        # Check if this is a combined grid (has lookup_table.csv with source_model column)
        is_combined = False
        lookup_path = os.path.join(os.path.dirname(flux_cubes[0]), "lookup_table.csv")
        if os.path.exists(lookup_path):
            try:
                df = pd.read_csv(lookup_path, comment='#', nrows=1)
                is_combined = 'source_model' in df.columns or 'original_file' in df.columns
                if is_combined:
                    print(f"  Detected combined grid - will track individual wavelength coverage")
            except:
                pass
        
        for cube_path in flux_cubes:
            print(f"  Loading {os.path.basename(os.path.dirname(cube_path))}...")
            
            with open(cube_path, 'rb') as f:
                # Read header
                header = f.read(16)
                nt, nl, nm, nw = struct.unpack('4i', header)
                
                # Read grids
                teff = np.fromfile(f, dtype=np.float64, count=nt)
                logg = np.fromfile(f, dtype=np.float64, count=nl)
                meta = np.fromfile(f, dtype=np.float64, count=nm)
                wavelengths = np.fromfile(f, dtype=np.float64, count=nw)
                
                # Read flux cube
                flux_data = np.fromfile(f, dtype=np.float64, count=nt*nl*nm*nw)
                flux_cube = flux_data.reshape((nw, nm, nl, nt))
            
            # For combined grids, load individual SED files to check actual coverage
            coverage_map = None
            if is_combined:
                coverage_map = self._build_coverage_map(
                    os.path.dirname(cube_path), 
                    teff, logg, meta
                )
            
            # Sample SEDs for training
            n_samples = min(5000, nt * nl * nm)
            
            for _ in range(n_samples):
                i_t = np.random.randint(0, nt)
                i_l = np.random.randint(0, nl)
                i_m = np.random.randint(0, nm)
                
                # Get full SED from cube
                flux = flux_cube[:, i_m, i_l, i_t]
                
                # Get wavelength coverage for this specific SED
                if coverage_map is not None:
                    wl_min, wl_max = coverage_map.get(
                        (teff[i_t], logg[i_l], meta[i_m]),
                        (wavelengths[0], wavelengths[-1])
                    )
                else:
                    wl_min, wl_max = wavelengths[0], wavelengths[-1]
                
                # Interpolate to common grid
                flux_interp = np.interp(self.wavelength_grid, wavelengths, flux,
                                       left=0, right=0)
                
                # Create mask for valid data region
                valid_mask = (self.wavelength_grid >= wl_min) & (self.wavelength_grid <= wl_max)
                
                # Create training sample: known region + extension region
                # Use central 40% of VALID region as "known", rest as "to predict"
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) < 100:  # Skip if too little valid data
                    continue
                
                mid_idx = len(valid_indices) // 2
                width = len(valid_indices) // 5
                
                known_indices = valid_indices[max(0, mid_idx - width):min(len(valid_indices), mid_idx + width)]
                
                # Create full-size arrays with proper masking
                known_flux = np.zeros(len(self.wavelength_grid))
                known_mask = np.zeros(len(self.wavelength_grid), dtype=bool)
                known_flux[known_indices] = flux_interp[known_indices]
                known_mask[known_indices] = True
                
                # Extensions: everything valid but not in "known"
                extension_mask = valid_mask & ~known_mask
                extension = flux_interp.copy()
                extension[~extension_mask] = 0  # Zero out invalid/known regions
                
                all_seds.append(known_flux)
                all_extensions.append(extension)
                all_masks.append(extension_mask)  # Track which outputs are valid
                all_params.append([teff[i_t], logg[i_l], meta[i_m]])
        
        if not all_seds:
            return None, None, None
        
        X = np.array(all_seds)
        y = np.array(all_extensions)
        params = np.array(all_params)
        masks = np.array(all_masks)
        
        # Store masks for use in training
        self.training_masks = masks
        
        print(f"  Loaded {len(X)} training samples")
        if is_combined:
            print(f"  Using variable wavelength coverage per spectrum")
        
        return X, y, params
    
    def _build_coverage_map(self, model_dir: str, teff: np.ndarray, 
                           logg: np.ndarray, meta: np.ndarray) -> Dict:
        """
        Build map of actual wavelength coverage for each parameter combination.
        
        For combined grids, individual SEDs may not cover the full wavelength range.
        This reads the actual SED files to determine true coverage.
        """
        from glob import glob

        import pandas as pd
        
        coverage = {}
        
        # Try to load lookup table
        lookup_path = os.path.join(model_dir, "lookup_table.csv")
        if not os.path.exists(lookup_path):
            return coverage
        
        try:
            df = pd.read_csv(lookup_path, comment='#')
            
            # Find parameter columns (case-insensitive)
            col_map = {c.lower(): c for c in df.columns}
            file_col = col_map.get('file_name', col_map.get('filename', df.columns[0]))
            teff_col = col_map.get('teff', None)
            logg_col = col_map.get('logg', None)
            meta_col = col_map.get('meta', col_map.get('metallicity', None))
            
            if not (teff_col and logg_col and meta_col):
                return coverage
            
            # Sample subset of files to determine coverage (checking all is too slow)
            n_sample = min(500, len(df))
            sample_df = df.sample(n=n_sample, random_state=42)
            
            for _, row in sample_df.iterrows():
                filepath = os.path.join(model_dir, row[file_col])
                if not os.path.exists(filepath):
                    continue
                
                try:
                    # Read full file to get true range
                    sed_wl, _ = self._load_sed(filepath)
                    if len(sed_wl) > 0:
                        # Store min/max wavelength for this parameter combo
                        param_key = (
                            float(row[teff_col]), 
                            float(row[logg_col]), 
                            float(row[meta_col])
                        )
                        coverage[param_key] = (sed_wl[0], sed_wl[-1])
                
                except Exception:
                    continue
        
        except Exception as e:
            print(f"    Warning: Could not build coverage map: {e}")
        
        if coverage:
            print(f"    Built coverage map from {len(coverage)} spectra")
        
        return coverage
    
    def _load_sed(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Quick SED file loader for wavelength range detection."""
        wl, flux = [], []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            wl.append(float(parts[0]))
                            flux.append(float(parts[1]))
                        except ValueError:
                            continue
        except Exception:
            pass
        return np.array(wl), np.array(flux)
    
    def _plot_training_results(self, history, X_val, y_val, output_path: str):
        """
        Create comprehensive training visualization.
        
        Parameters
        ----------
        history : keras History
            Training history object
        X_val : np.ndarray
            Validation input data
        y_val : np.ndarray
            Validation target data
        output_path : str
            Base path for saving plots
        """
        import matplotlib.pyplot as plt

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Loss plot (linear scale)
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss plot (log scale)
        ax2 = plt.subplot(2, 3, 2)
        ax2.semilogy(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.semilogy(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (MSE, log scale)', fontsize=12)
        ax2.set_title('Loss (Logarithmic Scale)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        
        # 3. MAE plot
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(history.history['mae'], label='Training MAE', linewidth=2)
        ax3.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Mean Absolute Error', fontsize=12)
        ax3.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4-6. Example predictions
        # Select 3 random validation samples
        n_examples = min(3, len(X_val))
        example_indices = np.random.choice(len(X_val), n_examples, replace=False)
        
        for i, idx in enumerate(example_indices):
            ax = plt.subplot(2, 3, 4 + i)
            
            # Get prediction
            X_sample = X_val[idx:idx+1]
            y_true_norm = y_val[idx]
            y_pred_norm = self.model.predict(X_sample, verbose=0)[0]
            
            # Denormalize to log space
            y_true_log = (y_true_norm * self.scaler_params['ext_log_std']) + self.scaler_params['ext_log_mean']
            y_pred_log = (y_pred_norm * self.scaler_params['ext_log_std']) + self.scaler_params['ext_log_mean']
            
            # Convert back to linear flux
            y_true = 10**y_true_log
            y_pred = 10**y_pred_log
            
            # Plot
            wavelength_idx = np.arange(len(y_true))
            ax.plot(wavelength_idx, y_true, 'b-', linewidth=2, label='True', alpha=0.7)
            ax.plot(wavelength_idx, y_pred, 'r--', linewidth=2, label='Predicted', alpha=0.7)
            
            # Calculate error
            relative_error = np.median(np.abs(y_pred - y_true) / (y_true + 1e-50)) * 100
            
            ax.set_xlabel('Extension Point Index', fontsize=10)
            ax.set_ylabel('Flux (erg/s/cm²/Å)', fontsize=10)
            ax.set_title(f'Example {i+1} (Median Error: {relative_error:.1f}%)', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save figure
        plot_path = output_path + '_training_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved training plots to {plot_path}")
        plt.close()
        
        # Create a second figure with more detailed prediction examples
        self._plot_detailed_predictions(X_val, y_val, output_path)
    
    def _plot_detailed_predictions(self, X_val, y_val, output_path: str):
        """
        Create detailed prediction comparison plots.
        
        Parameters
        ----------
        X_val : np.ndarray
            Validation input data
        y_val : np.ndarray
            Validation target data  
        output_path : str
            Base path for saving plots
        """
        import matplotlib.pyplot as plt

        # Select 6 examples
        n_examples = min(6, len(X_val))
        example_indices = np.random.choice(len(X_val), n_examples, replace=False)
        
        fig = plt.figure(figsize=(20, 12))
        
        for i, idx in enumerate(example_indices):
            ax = plt.subplot(2, 3, i + 1)
            
            # Get prediction
            X_sample = X_val[idx:idx+1]
            y_true_norm = y_val[idx]
            y_pred_norm = self.model.predict(X_sample, verbose=0)[0]
            
            # Denormalize
            y_true_log = (y_true_norm * self.scaler_params['ext_log_std']) + self.scaler_params['ext_log_mean']
            y_pred_log = (y_pred_norm * self.scaler_params['ext_log_std']) + self.scaler_params['ext_log_mean']
            
            y_true = 10**y_true_log
            y_pred = 10**y_pred_log
            
            # Split into low and high extensions (approximate)
            mid = len(y_true) // 2
            
            # Wavelength axis (approximate - we don't have exact wavelengths here)
            wl_low = np.linspace(100, 3000, mid)
            wl_high = np.linspace(7000, 100000, len(y_true) - mid)
            
            # Plot low extension
            ax.plot(wl_low, y_true[:mid], 'b-', linewidth=2, label='True (UV)', alpha=0.7)
            ax.plot(wl_low, y_pred[:mid], 'r--', linewidth=2, label='Pred (UV)', alpha=0.7)
            
            # Plot high extension
            ax.plot(wl_high, y_true[mid:], 'g-', linewidth=2, label='True (IR)', alpha=0.7)
            ax.plot(wl_high, y_pred[mid:], 'm--', linewidth=2, label='Pred (IR)', alpha=0.7)
            
            # Calculate errors
            error_low = np.median(np.abs(y_pred[:mid] - y_true[:mid]) / (y_true[:mid] + 1e-50)) * 100
            error_high = np.median(np.abs(y_pred[mid:] - y_true[mid:]) / (y_true[mid:] + 1e-50)) * 100
            
            ax.set_xlabel('Wavelength (Å)', fontsize=10)
            ax.set_ylabel('Flux (erg/s/cm²/Å)', fontsize=10)
            ax.set_title(f'Sample {i+1}: UV err={error_low:.1f}%, IR err={error_high:.1f}%',
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save figure
        plot_path = output_path + '_prediction_examples.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction examples to {plot_path}")
        plt.close()
    
    def _resample_to_training_grid(self, wavelength: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Resample input SED to the model's training wavelength grid.
        
        This allows the model to work with SEDs that have different wavelength
        coverage than the training data. Uses logarithmic interpolation for
        better handling of stellar spectra.
        
        Parameters
        ----------
        wavelength : np.ndarray
            Input wavelength array (Angstroms)
        flux : np.ndarray
            Input flux array (erg/s/cm²/Å)
            
        Returns
        -------
        flux_resampled : np.ndarray
            Flux interpolated to self.wavelength_grid
        """
        # Use logarithmic interpolation for flux (better for stellar spectra)
        flux_log = np.log10(np.maximum(flux, 1e-50))
        
        # Interpolate to training grid
        # For regions outside input coverage, use 0 (will be filled by model/BB)
        flux_log_resampled = np.interp(
            self.wavelength_grid,
            wavelength,
            flux_log,
            left=0.0,  # Below input range
            right=0.0  # Above input range
        )
        
        # Convert back from log
        flux_resampled = 10**flux_log_resampled
        
        return flux_resampled
    
    def complete_sed(self, wavelength: np.ndarray, flux: np.ndarray,
                    teff: float, logg: float, meta: float,
                    extension_range: Tuple[float, float] = (100.0, 100000.0)) -> Tuple[np.ndarray, np.ndarray]:
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
            (min_wavelength, max_wavelength) for completed SED
        
        Returns
        -------
        wavelength_extended : np.ndarray
            Extended wavelength array
        flux_extended : np.ndarray
            Extended flux array
        """
        if self.model is None:
            raise ValueError("No trained model loaded. Train first or provide model_path.")
        
        # Create full wavelength grid
        full_grid = self._create_wavelength_grid(extension_range[0], extension_range[1])
        
        # Find regions that need extension
        wl_min = wavelength.min()
        wl_max = wavelength.max()
        
        need_low = full_grid < wl_min
        need_high = full_grid > wl_max
        
        if not (need_low.any() or need_high.any()):
            print("SED already covers requested range")
            return wavelength, flux
        
        # Calculate black body baseline
        print("Calculating black body baseline...")
        bb_flux = planck_function(full_grid, teff)
        
        # STEP 1: Resample input SED to the training wavelength grid
        # This allows the model to work with SEDs from different datasets
        print("Resampling input to training grid...")
        flux_on_training_grid = self._resample_to_training_grid(wavelength, flux)
        
        # STEP 2: Prepare input for neural network
        # CRITICAL: Must match training data preparation
        # During training, we used central 40% of wavelength_grid as "known"
        mid = len(self.wavelength_grid) // 2
        width = len(self.wavelength_grid) // 5
        known_start = mid - width
        known_end = mid + width
        
        # Extract the "known" region from the resampled data
        flux_known = flux_on_training_grid[known_start:known_end]
        
        # Log-scale and normalize flux (same as training)
        flux_log = np.log10(np.maximum(flux_known, 1e-50))
        flux_norm = (flux_log - self.scaler_params['flux_log_mean']) / self.scaler_params['flux_log_std']
        
        # Normalize stellar parameters
        params_norm = self._normalize_parameters(teff, logg, meta)
        
        # Prepare input: normalized flux + parameters
        X_input = np.concatenate([flux_norm, params_norm]).reshape(1, -1)
        
        # Predict extensions (in normalized log space)
        print("Predicting extensions with neural network...")
        y_pred_norm = self.model.predict(X_input, verbose=0)[0]
        
        # Denormalize predictions
        y_pred_log = (y_pred_norm * self.scaler_params['ext_log_std']) + self.scaler_params['ext_log_mean']
        y_pred = 10**y_pred_log
                
        # Split predictions into low/high based on TRAINING grid definition (not full_grid)
        n_low_pred = known_start
        n_high_pred = len(self.wavelength_grid) - known_end

        pred_low = y_pred[:n_low_pred]
        pred_high = y_pred[n_low_pred:n_low_pred + n_high_pred]

        wl_pred_low = self.wavelength_grid[:known_start]
        wl_pred_high = self.wavelength_grid[known_end:]

                
        # Combine: black body baseline + ML refinement
        # Use weighted combination: more ML weight near known region, more BB weight far away
        flux_extended = bb_flux.copy()
        


        # Low wavelength extension
        if need_low.any():
            idx_low = np.where(need_low)[0]
            wl_low = full_grid[idx_low]

            mask_ml = (wl_low >= wl_pred_low.min()) & (wl_low <= wl_pred_low.max())
            if np.any(mask_ml):
                pred_low_full = np.interp(wl_low[mask_ml], wl_pred_low, pred_low)
                weight_ml = np.exp(-(wl_min - wl_low[mask_ml]) / (wl_min / 3))
                ii = idx_low[mask_ml]
                flux_extended[ii] = (weight_ml * pred_low_full +
                                     (1 - weight_ml) * bb_flux[ii])


        # High wavelength extension
        if need_high.any():
            idx_high = np.where(need_high)[0]
            wl_high = full_grid[idx_high]

            mask_ml = (wl_high >= wl_pred_high.min()) & (wl_high <= wl_pred_high.max())
            if np.any(mask_ml):
                pred_high_full = np.interp(wl_high[mask_ml], wl_pred_high, pred_high)
                weight_ml = np.exp(-(wl_high[mask_ml] - wl_max) / (wl_max / 3))
                ii = idx_high[mask_ml]
                flux_extended[ii] = (weight_ml * pred_high_full +
                                     (1 - weight_ml) * bb_flux[ii])

        # Insert original data where we have it
        # Use the resampled flux for regions covered by the input
        overlap_mask = (full_grid >= wl_min) & (full_grid <= wl_max)
        if overlap_mask.any():
            # Interpolate original data to full_grid in the overlap region
            flux_interp_overlap = np.interp(full_grid[overlap_mask], wavelength, flux)
            flux_extended[overlap_mask] = flux_interp_overlap
        
        print(f"Extended SED from {wavelength.min():.0f}-{wavelength.max():.0f} Å")
        print(f"          to {full_grid.min():.0f}-{full_grid.max():.0f} Å")
        
        return full_grid, flux_extended
    
    def save_model(self, path: str):
        """Save trained model and parameters."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save Keras model
        self.model.save(path + '.keras')
        
        # Save scaler parameters
        with open(path + '.params', 'wb') as f:
            pickle.dump({
                'scaler_params': self.scaler_params,
                'wavelength_grid': self.wavelength_grid
            }, f)
        
        print(f"Saved model to {path}.keras and {path}.params")
    
    def load_model(self, path: str):
        """Load trained model and parameters."""
        try:
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow required. Install: pip install tensorflow")
        
        # Load Keras model
        self.model = keras.models.load_model(path + '.keras')
        
        # Load scaler parameters
        with open(path + '.params', 'rb') as f:
            data = pickle.load(f)
            self.scaler_params = data['scaler_params']
            self.wavelength_grid = data['wavelength_grid']
        
        print(f"Loaded model from {path}")


def plot_extended_seds(output_dir: str, n_examples: int = 6) -> Optional[str]:
    """
    Create diagnostic plots showing representative extended SEDs.
    
    Automatically selects SEDs spanning the parameter space (Teff, logg, [M/H])
    and visualizes the extended spectra.
    
    Parameters
    ----------
    output_dir : str
        Directory containing the extended SEDs and lookup_table.csv
    n_examples : int, default=6
        Number of example SEDs to plot
    
    Returns
    -------
    plot_path : str or None
        Path to saved plot, or None if plotting failed
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        print(f"[Plot] Skipping (requires pandas & matplotlib): {e}")
        return None
    
    # Load lookup table
    lookup_path = os.path.join(output_dir, "lookup_table.csv")
    if not os.path.exists(lookup_path):
        print(f"[Plot] Lookup table not found: {lookup_path}")
        return None
    
    # Read CSV - handle both '#file_name,...' and 'file_name,...' formats
    df = pd.read_csv(lookup_path)
    
    # Check if header was skipped (first column name looks like a filename)
    if df.columns[0].endswith('.txt'):
        # Header was treated as data - the file has '#file_name,...'
        # Re-read without comment parameter and strip '#' from column names
        with open(lookup_path, 'r') as f:
            first_line = f.readline().strip()
        
        if first_line.startswith('#'):
            # Extract proper column names
            col_names = [c.strip() for c in first_line.lstrip('#').split(',')]
            df = pd.read_csv(lookup_path, skiprows=1, names=col_names)
        else:
            # No header at all - use default names
            df.columns = ['file_name', 'teff', 'logg', 'meta', 'source_model', 'extended_by_ml']
    
    if len(df) == 0:
        print("[Plot] No SEDs in lookup table")
        return None
    
    # Identify parameter columns (flexible naming)
    teff_col = logg_col = meta_col = None
    for col in df.columns:
        col_lower = col.lower()
        if not teff_col and any(x in col_lower for x in ['teff', 't_eff']):
            teff_col = col
        elif not logg_col and any(x in col_lower for x in ['logg', 'log_g']):
            logg_col = col
        elif not meta_col and any(x in col_lower for x in ['meta', '[m/h]', 'feh', '[fe/h]']):
            meta_col = col
    
    if not all([teff_col, logg_col, meta_col]):
        print(f"[Plot] Cannot identify parameters. Columns: {df.columns.tolist()}")
        return None
    
    # Parameter ranges
    teff_min, teff_max = df[teff_col].min(), df[teff_col].max()
    logg_min, logg_max = df[logg_col].min(), df[logg_col].max()
    meta_min, meta_max = df[meta_col].min(), df[meta_col].max()
    
    print(f"\n[Plot] Parameter space:")
    print(f"  Teff:  {teff_min:.0f} - {teff_max:.0f} K")
    print(f"  logg:  {logg_min:.2f} - {logg_max:.2f}")
    print(f"  [M/H]: {meta_min:.2f} - {meta_max:.2f}")
    
    # Select representative SEDs across parameter space
    # Strategy: divide into bins and pick one from each bin
    selected = []
    n_per_dim = int(np.ceil(n_examples ** (1/3)))  # cube root for 3D grid
    
    teff_edges = np.linspace(teff_min, teff_max, n_per_dim + 1)
    logg_edges = np.linspace(logg_min, logg_max, n_per_dim + 1)
    meta_edges = np.linspace(meta_min, meta_max, n_per_dim + 1)
    
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            for k in range(n_per_dim):
                if len(selected) >= n_examples:
                    break
                
                mask = (
                    (df[teff_col] >= teff_edges[i]) & (df[teff_col] < teff_edges[i+1]) &
                    (df[logg_col] >= logg_edges[j]) & (df[logg_col] < logg_edges[j+1]) &
                    (df[meta_col] >= meta_edges[k]) & (df[meta_col] < meta_edges[k+1])
                )
                
                in_bin = df[mask]
                if len(in_bin) > 0:
                    # Pick the one closest to bin center
                    teff_c = (teff_edges[i] + teff_edges[i+1]) / 2
                    logg_c = (logg_edges[j] + logg_edges[j+1]) / 2
                    meta_c = (meta_edges[k] + meta_edges[k+1]) / 2
                    
                    dist = (
                        ((in_bin[teff_col] - teff_c) / (teff_max - teff_min + 1e-10)) ** 2 +
                        ((in_bin[logg_col] - logg_c) / (logg_max - logg_min + 1e-10)) ** 2 +
                        ((in_bin[meta_col] - meta_c) / (meta_max - meta_min + 1e-10)) ** 2
                    )
                    
                    selected.append(in_bin.iloc[dist.argmin()])
            
            if len(selected) >= n_examples:
                break
        if len(selected) >= n_examples:
            break
    
    # Fallback: evenly spaced if not enough found
    if len(selected) < n_examples:
        idx = np.linspace(0, len(df) - 1, n_examples, dtype=int)
        selected = [df.iloc[i] for i in idx]
    
    print(f"[Plot] Selected {len(selected)} representative SEDs")
    
    # Create figure
    n_cols = min(3, len(selected))
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, sed_info in enumerate(selected[:len(axes)]):
        ax = axes[idx]
        
        # Get filename
        fname = sed_info.get('file_name', sed_info.get('filename', sed_info.iloc[0]))
        sed_path = os.path.join(output_dir, fname)
        
        if not os.path.exists(sed_path):
            ax.text(0.5, 0.5, f"Not found:\n{fname}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            continue
        
        try:
            # Load SED
            data = np.loadtxt(sed_path, comments='#')
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            
            wl = data[:, 0]
            fl = data[:, 1]
            
            # Plot
            ax.plot(wl, fl, 'b-', linewidth=0.8, alpha=0.9)
            
            # Title with parameters
            T = sed_info[teff_col]
            g = sed_info[logg_col]
            m = sed_info[meta_col]
            ax.set_title(f"T={T:.0f}K, log g={g:.2f}, [M/H]={m:.2f}", fontsize=9)
            
            ax.set_xlabel('Wavelength (Å)', fontsize=8)
            ax.set_ylabel('Flux (erg/s/cm²/Å)', fontsize=8)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, which='both', linewidth=0.5)
            ax.tick_params(labelsize=7)
            
            # Set y-limits to avoid issues
            valid = fl[fl > 0]
            if len(valid) > 0:
                ymin = np.percentile(valid, 1)
                ymax = np.percentile(valid, 99)
                ax.set_ylim(ymin * 0.5, ymax * 2)
        
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=7)
    
    # Hide unused subplots
    for idx in range(len(selected), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("ML-Extended SEDs: Representative Sample", fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(output_dir, "ml_extended_seds_examples.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Plot] ✓ Saved: {plot_path}")
    
    return plot_path


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
    from tqdm import tqdm

    # Ensure directories exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ML SED COMPLETER")
    print("="*60)
    print("\nExtend incomplete SEDs using trained neural networks")
    
    # Main choice: train or complete
    print("\nWhat would you like to do?")
    print("  1) Train new model")
    print("  2) Complete/extend SEDs (inference)")
    print("  0) Back")
    
    choice = input("> ").strip()
    
    if choice == "0":
        return
    
    elif choice == "1":
        # TRAINING MODE
        print("\n" + "-"*60)
        print("TRAIN NEW MODEL")
        print("-"*60)
        
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
        print("-"*60)
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
        epochs = input("  Epochs [100]: ").strip() or "100"
        batch_size = input("  Batch size [32]: ").strip() or "32"
        
        try:
            epochs = int(epochs)
            batch_size = int(batch_size)
        except ValueError:
            print("Invalid parameters")
            return
        
        # Model name
        default_name = f"sed_model_{selected['name']}"
        model_name = input(f"  Model name [{default_name}]: ").strip() or default_name
        output_path = os.path.join(models_dir, model_name)
        
        print(f"\nModel will be saved to: {output_path}.keras")
        confirm = input("Start training? [Y/n]: ").strip().lower()
        
        if confirm and not confirm.startswith('y'):
            print("Cancelled")
            return
        
        # Train
        print("\n" + "="*60)
        print("TRAINING...")
        print("="*60)
        
        completer = SEDCompleter()
        history = completer.train_from_library(
            selected['path'],
            output_path,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Model saved: {output_path}.keras")
        print(f"Parameters: {output_path}.params")
        print(f"Plots: {output_path}_training_plots.png")
        print(f"       {output_path}_prediction_examples.png")
    
    elif choice == "2":
        # INFERENCE/COMPLETION MODE
        print("\n" + "-"*60)
        print("COMPLETE/EXTEND SEDs")
        print("-"*60)
        
        # Discover trained models
        trained_models = []
        if os.path.exists(models_dir):
            for fname in os.listdir(models_dir):
                if fname.endswith('.keras'):
                    model_name = fname[:-6]  # Remove .keras
                    params_file = os.path.join(models_dir, model_name + '.params')
                    if os.path.exists(params_file):
                        trained_models.append({
                            'name': model_name,
                            'path': os.path.join(models_dir, model_name)
                        })
        
        if not trained_models:
            print(f"\nNo trained models found in {models_dir}/")
            print("Train a model first (option 1)")
            return
        
        # Select trained model
        print("\nAvailable trained models:")
        print("-"*60)
        for i, model in enumerate(trained_models, 1):
            print(f"  {i:2d}) {model['name']}")
        
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
        completer = SEDCompleter(model_path=selected_model['path'])
        print("✓ Model loaded successfully")
        
        # Select source model set to complete
        print("\n" + "-"*60)
        print("SELECT MODEL SET TO EXTEND")
        print("-"*60)
        
        available_models = []
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
                    available_models.append({
                        'name': name,
                        'path': model_path,
                        'n_seds': n_seds
                    })
                except Exception:
                    pass
        
        if not available_models:
            print(f"\nNo model sets found in {base_dir}")
            return
        
        # Display models
        print("\nAvailable model sets:")
        print("-"*60)
        for i, model in enumerate(available_models, 1):
            print(f"  {i:2d}) {model['name']:30s} ({model['n_seds']} SEDs)")
        
        print("\nSelect model set to extend (enter number):")
        selection = input("> ").strip()
        
        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(available_models):
                print("Invalid selection")
                return
            source_model = available_models[idx]
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
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load source lookup table
        lookup_path = os.path.join(source_model['path'], "lookup_table.csv")
        df = pd.read_csv(lookup_path, comment='#')
        
        # Process each SED
        print("\n" + "="*60)
        print("EXTENDING SEDs...")
        print("="*60)
        
        extended_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing SEDs"):
            # Get SED filename and parameters
            if 'file_name' in row:
                sed_file = row['file_name']
            elif 'filename' in row:
                sed_file = row['filename']
            else:
                sed_file = row.iloc[0]
            
            # Get parameters
            teff = float(row.get('teff', row.get('Teff', row.get('T_eff', 5777))))
            logg = float(row.get('logg', row.get('Logg', row.get('log_g', 4.44))))
            meta = float(row.get('meta', row.get('metallicity', row.get('[M/H]', row.get('feh', 0.0)))))
            
            # Load SED
            sed_path = os.path.join(source_model['path'], sed_file)
            try:
                data = np.loadtxt(sed_path, comments='#')
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                wavelength = data[:, 0]
                flux = data[:, 1]
                
                # Complete SED
                wl_extended, flux_extended = completer.complete_sed(
                    wavelength, flux,
                    teff, logg, meta,
                    extension_range=(100.0, 100000.0)
                )
                
                # Save extended SED
                output_file = os.path.join(output_dir, sed_file)
                np.savetxt(
                    output_file,
                    np.column_stack([wl_extended, flux_extended]),
                    header=f'wavelength_A flux_erg/s/cm2/A (ML extended)\nteff={teff} logg={logg} meta={meta}',
                    fmt='%.6e'
                )
                
                # Track for lookup table
                extended_rows.append({
                    'file_name': sed_file,
                    'teff': teff,
                    'logg': logg,
                    'meta': meta,
                    'source_model': source_model['name'],
                    'extended_by_ml': True
                })
            except Exception as e:
                print(f"\nWarning: Failed to process {sed_file}: {e}")
                continue
        
        print(f"\n✓ Successfully extended {len(extended_rows)} SEDs")
        
        # Save lookup table
        if extended_rows:
            lookup_df = pd.DataFrame(extended_rows)
            output_lookup = os.path.join(output_dir, "lookup_table.csv")
            lookup_df.to_csv(output_lookup, index=False)
            print(f"✓ Saved lookup table: {output_lookup}")
        
        # Create diagnostic plots
        print("\n" + "-"*60)
        print("CREATING DIAGNOSTIC PLOTS")
        print("-"*60)
        
        plot_path = plot_extended_seds(output_dir, n_examples=6)
        if plot_path:
            print(f"✓ Diagnostic plots: {plot_path}")
        
        # Rebuild flux cube and HDF5
        print("\n" + "-"*60)
        print("REBUILDING DATA PRODUCTS")
        print("-"*60)
        
        rebuild = input("Rebuild flux cube and HDF5? [Y/n]: ").strip().lower()
        if not rebuild or rebuild.startswith('y'):
            try:
                # Import these here to avoid circular imports
                import importlib.util
                import sys

                # Helper functions defined inline to avoid circular imports
                def load_txt_spectrum(txt_path: str):
                    """Load wavelength and flux from text file."""
                    wl, fl = [], []
                    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            s = line.strip()
                            if not s or s.startswith("#"):
                                continue
                            parts = s.split()
                            if len(parts) >= 2:
                                try:
                                    wl.append(float(parts[0]))
                                    fl.append(float(parts[1]))
                                except ValueError:
                                    continue
                    return np.asarray(wl, dtype=float), np.asarray(fl, dtype=float)
                
                def numeric_from(meta: dict, key_candidates: list, default: float = np.nan) -> float:
                    """Extract first numeric token from metadata."""
                    import re
                    lower = {k.lower(): v for k, v in meta.items()}
                    for ck in key_candidates:
                        if ck.lower() in lower:
                            val = lower[ck.lower()]
                            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
                            if m:
                                try:
                                    return float(m.group(0))
                                except ValueError:
                                    pass
                    return default
                
                def parse_metadata(file_path: str) -> dict:
                    """Parse metadata from file headers."""
                    import re
                    metadata = {}
                    try:
                        with open(file_path, "r") as file:
                            for line in file:
                                line = line.strip()
                                if line.startswith("#") and "=" in line:
                                    try:
                                        key, value = line.split("=", 1)
                                        key = key.strip("#").strip()
                                        value = value.split("(")[0].strip()
                                        metadata[key] = value
                                    except ValueError:
                                        continue
                    except Exception:
                        pass
                    return metadata
                
                def build_h5_bundle_from_txt(model_dir: str, out_h5: str):
                    """Create an HDF5 file bundling all .txt spectra."""
                    import h5py
                    txt_files = sorted([f for f in os.listdir(model_dir) if f.lower().endswith(".txt")])
                    if not txt_files:
                        print(f"[H5 bundle] No .txt spectra found in {model_dir}; skipping.")
                        return
                    
                    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
                    with h5py.File(out_h5, "w") as h5:
                        spectra_grp = h5.create_group("spectra")
                        for fname in txt_files:
                            path = os.path.join(model_dir, fname)
                            try:
                                wl, fl = load_txt_spectrum(path)
                                if wl.size == 0 or fl.size == 0:
                                    continue
                                g = spectra_grp.create_group(fname)
                                g.create_dataset("lambda", data=wl, dtype="f8")
                                g.create_dataset("flux",   data=fl, dtype="f8")
                                
                                meta = parse_metadata(path)
                                teff = numeric_from(meta, ["Teff", "teff", "T_eff"])
                                logg = numeric_from(meta, ["logg", "Logg", "log_g"])
                                feh = numeric_from(meta, ["FeH", "feh", "metallicity", "[Fe/H]", "meta"])
                                if not np.isnan(teff): g.attrs["teff"] = teff
                                if not np.isnan(logg): g.attrs["logg"] = logg
                                if not np.isnan(feh):  g.attrs["feh"] = feh
                                for k, v in meta.items():
                                    g.attrs[f"raw:{k}"] = v
                            except Exception as e:
                                print(f"[H5 bundle] Error on {fname}: {e}")
                    
                    print(f"[H5 bundle] Wrote {out_h5}")
                
                # Try to import precompute_flux_cube
                try:
                    from .precompute_flux_cube import precompute_flux_cube
                except ImportError:
                    # Fallback if relative import fails
                    try:
                        module_name = "precompute_flux_cube"
                        # Try to find it in the same directory
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        module_path = os.path.join(script_dir, "precompute_flux_cube.py")
                        if os.path.exists(module_path):
                            spec = importlib.util.spec_from_file_location(module_name, module_path)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            precompute_flux_cube = module.precompute_flux_cube
                        else:
                            raise ImportError("Could not locate precompute_flux_cube")
                    except Exception:
                        precompute_flux_cube = None
                
                # HDF5 bundle
                print("Building HDF5 bundle...")
                h5_path = os.path.join(output_dir, f"{output_name}.h5")
                build_h5_bundle_from_txt(output_dir, h5_path)
                print(f"✓ HDF5: {h5_path}")
                
                # Flux cube
                if precompute_flux_cube:
                    print("Building flux cube...")
                    cube_path = os.path.join(output_dir, "flux_cube.bin")
                    precompute_flux_cube(output_dir, cube_path)
                    print(f"✓ Flux cube: {cube_path}")
                else:
                    print("Warning: Could not import precompute_flux_cube, skipping flux cube generation")
                    
            except ImportError as e:
                print(f"Warning: Could not rebuild data products (missing dependencies): {e}")
            except Exception as e:
                print(f"Warning: Error rebuilding data products: {e}")
        
        print("\n" + "="*60)
        print("EXTENSION COMPLETE!")
        print("="*60)
        print(f"Extended model: {output_dir}")
        print(f"You can now use this in MESA with:")
        print(f"  stellar_atm = '{output_dir}/'")
    
    else:
        print("Invalid choice")


def main():
    """Command-line interface for SED completer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ML-powered SED completer: extend incomplete SEDs"
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model on SED library')
    train_parser.add_argument('--library', required=True,
                             help='Path to SED library with flux_cube.bin files')
    train_parser.add_argument('--output', required=True,
                             help='Output path for trained model')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Training epochs (default: 100)')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size (default: 32)')
    
    # Complete command
    complete_parser = subparsers.add_parser('complete', help='Complete an incomplete SED')
    complete_parser.add_argument('--model', required=True,
                                help='Path to trained model')
    complete_parser.add_argument('--input', required=True,
                                help='Input SED file (wavelength flux)')
    complete_parser.add_argument('--output', required=True,
                                help='Output completed SED file')
    complete_parser.add_argument('--teff', type=float, required=True,
                                help='Effective temperature (K)')
    complete_parser.add_argument('--logg', type=float, required=True,
                                help='Surface gravity')
    complete_parser.add_argument('--meta', type=float, required=True,
                                help='Metallicity [M/H]')
    complete_parser.add_argument('--range', nargs=2, type=float,
                                default=[100.0, 100000.0],
                                help='Wavelength range (min max) in Angstroms')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        completer = SEDCompleter()
        completer.train_from_library(
            args.library,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    elif args.command == 'complete':
        # Load incomplete SED
        data = np.loadtxt(args.input)
        wavelength = data[:, 0]
        flux = data[:, 1]
        
        # Load model and complete
        completer = SEDCompleter(model_path=args.model)
        wl_extended, flux_extended = completer.complete_sed(
            wavelength, flux,
            args.teff, args.logg, args.meta,
            extension_range=tuple(args.range)
        )
        
        # Save result
        np.savetxt(args.output, np.column_stack([wl_extended, flux_extended]),
                  header='wavelength_A flux_erg/s/cm2/A', fmt='%.6e')
        print(f"Saved completed SED to {args.output}")


if __name__ == '__main__':
    main()