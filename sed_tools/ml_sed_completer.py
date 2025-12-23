#!/usr/bin/env python3
"""
ML-Powered SED Completer
=========================

Extends incomplete SEDs to broader wavelength ranges using:
1. Black body radiation as physical baseline
2. Neural network trained on complete SEDs for refinement

The tool only appends/extends - never modifies existing wavelength coverage.
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
        """Load complete SEDs from library for training."""
        import struct
        from glob import glob
        
        flux_cubes = glob(library_path + "/flux_cube.bin")
        
        if not flux_cubes:
            print(f"No flux_cube.bin files found in {library_path}")
            return None, None, None
        
        all_seds = []
        all_extensions = []
        all_params = []
        
        # Create target wavelength grid
        self.wavelength_grid = self._create_wavelength_grid()
        
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
            
            # Sample SEDs for training - increase sample size significantly
            n_samples = min(5000, nt * nl * nm)  # Get many more samples per cube
            
            for _ in range(n_samples):
                i_t = np.random.randint(0, nt)
                i_l = np.random.randint(0, nl)
                i_m = np.random.randint(0, nm)
                
                # Get full SED
                flux = flux_cube[:, i_m, i_l, i_t]
                
                # Interpolate to common grid
                flux_interp = np.interp(self.wavelength_grid, wavelengths, flux,
                                       left=0, right=0)
                
                # Create training sample: known region + extension region
                # Use central 40% as "known", rest as "to predict"
                mid = len(self.wavelength_grid) // 2
                width = len(self.wavelength_grid) // 5
                
                known_start = mid - width
                known_end = mid + width
                
                known_flux = flux_interp[known_start:known_end]
                
                # Extensions (what we want to predict)
                extension_low = flux_interp[:known_start]
                extension_high = flux_interp[known_end:]
                extension = np.concatenate([extension_low, extension_high])
                
                all_seds.append(known_flux)
                all_extensions.append(extension)
                all_params.append([teff[i_t], logg[i_l], meta[i_m]])
        
        if not all_seds:
            return None, None, None
        
        X = np.array(all_seds)
        y = np.array(all_extensions)
        params = np.array(all_params)
        
        return X, y, params
    
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