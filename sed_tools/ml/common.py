"""What the two ML engines share: the device, the training loop, and their common data preparation and menus.

The engines are completer.py (SEDCompleter) and generator.py (SEDGenerator); ml/__init__.py is the 0.4.1
completer API over them. torch is imported inside the functions, so importing any of them needs no torch.
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ui_utils import refused_without

DEFAULT_WL_MIN = 100.0      # Angstroms
DEFAULT_WL_MAX = 100000.0   # Angstroms
DEFAULT_WL_POINTS = 1000
_PARAMS = ('teff', 'logg', 'meta')

# Both engines' training-data and library warnings log under the completer's logger name.
_completer_log = logging.getLogger("sed_tools.ml_sed_completer")


def get_device():
    """CUDA if available (with cudnn.benchmark on), else Apple MPS, else the CPU."""
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_grid(
    min_wl: float = DEFAULT_WL_MIN,
    max_wl: float = DEFAULT_WL_MAX,
    n_points: int = DEFAULT_WL_POINTS,
) -> np.ndarray:
    """Logarithmic wavelength grid."""
    return np.logspace(np.log10(min_wl), np.log10(max_wl), n_points)


def mlp(n_in: int, hidden_layers: List[int], n_out: int, dropout: float):
    """Linear, BatchNorm1d and LeakyReLU per hidden layer, Dropout between them, then a Linear.

    The layer order is the model.pt format: its state_dict keys are the layer indices.
    """
    import torch.nn as nn

    layers = []
    for i, width in enumerate(hidden_layers):
        layers.append(nn.Linear(n_in, width))
        layers.append(nn.BatchNorm1d(width))
        layers.append(nn.LeakyReLU(0.1))
        if i < len(hidden_layers) - 1:
            layers.append(nn.Dropout(dropout))
        n_in = width
    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


# Training data

def resample(grid: np.ndarray, wavelengths: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """flux on grid (NaN outside its coverage), and where it is valid: covered, finite and positive."""
    flux_interp = np.interp(grid, wavelengths, flux, left=np.nan, right=np.nan)
    valid = (
        (grid >= wavelengths.min()) &
        (grid <= wavelengths.max()) &
        np.isfinite(flux_interp) &
        (flux_interp > 0)
    )
    return flux_interp, valid


def sample_spectra(teff, logg, meta, flux_cube, n_samples: int, cut, library_path: str):
    """(teff, logg, meta) rows and training samples cut from n_samples spectra drawn at random from the cube.

    ``cut(flux)`` returns ``(None, sample)`` or ``(reason, None)``. Warns how many spectra were
    dropped and why; raises ValueError if none was kept.
    """
    nl, nm = len(logg), len(meta)
    params = []
    samples = []
    dropped = Counter()
    indices = np.random.permutation(len(teff) * nl * nm)[:n_samples]
    for idx in indices:
        i_t = idx // (nl * nm)
        remainder = idx % (nl * nm)
        i_l = remainder // nm
        i_m = remainder % nm
        # The cube is stored as [nw, nm, nl, nt]
        reason, sample = cut(flux_cube[:, i_m, i_l, i_t])
        if reason:
            dropped[reason] += 1
            continue
        params.append([teff[i_t], logg[i_l], meta[i_m]])
        samples.append(sample)
    _warn_dropped(dropped, len(indices), library_path)
    if not params:
        raise ValueError(f"No valid training samples found in {library_path}")
    return params, samples


def _warn_dropped(dropped: Counter, n_sampled: int, library_path: str) -> None:
    """Say how many sampled spectra were left out of the training set, and why."""
    if not dropped:
        return
    reasons = ", ".join(f"{count} with {reason}" for reason, count in dropped.items())
    _completer_log.warning("Training data from %s: dropped %d of %d sampled spectra (%s)",
                           library_path, sum(dropped.values()), n_sampled, reasons)


def param_scaler(params: np.ndarray, with_range: bool = False) -> Dict[str, float]:
    """Mean and std (with_range: also min and max) of each (teff, logg, meta) column, keyed as config.json has them."""
    scaler = {}
    for i, name in enumerate(_PARAMS):
        scaler[f'{name}_mean'] = float(np.mean(params[:, i]))
        scaler[f'{name}_std'] = float(np.std(params[:, i]) + 1e-8)
        if with_range:
            scaler[f'{name}_min'] = float(np.min(params[:, i]))
            scaler[f'{name}_max'] = float(np.max(params[:, i]))
    return scaler


def normalised_params(scaler: Dict[str, float], teff, logg, meta) -> list:
    """teff, logg and meta scaled as in training; each may be one value or a column of them."""
    return [
        (teff - scaler['teff_mean']) / scaler['teff_std'],
        (logg - scaler['logg_mean']) / scaler['logg_std'],
        (meta - scaler['meta_mean']) / scaler['meta_std'],
    ]


# Training loop

def masked_mse(y_pred, y_true, mask):
    """Mean squared error over the points where mask is 1."""
    sq_error = (y_pred - y_true) ** 2
    masked_error = sq_error * mask
    n_valid = mask.sum() + 1e-8  # Avoid division by zero
    return masked_error.sum() / n_valid


def data_loaders(train_arrays, val_arrays, batch_size: int):
    """DataLoaders over (inputs, targets, masks) arrays; only training batches are shuffled."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(*(torch.from_numpy(a) for a in train_arrays))
    val_dataset = TensorDataset(*(torch.from_numpy(a) for a in val_arrays))
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size))


def fit(model, train_loader, val_loader, optimizer, scheduler, *, epochs: int, patience: int,
        verbose: bool, device, progress: str) -> Tuple[Dict[str, List[float]], float]:
    """
    Train with early stopping on the validation loss, then restore the best weights.

    Batches with a NaN loss are skipped; an epoch with no usable training or
    validation batch ends training. Every 10th epoch, if ``verbose``, prints
    ``progress`` formatted with ``epoch``, ``train``, ``val`` and ``lr``.

    Returns the loss history and the best validation loss. Raises
    RuntimeError, before any caller saves a model, if no epoch was usable.
    """
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    print(f"\nTraining for up to {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        train_losses = _train_epoch(model, train_loader, optimizer, device, epoch)
        val_losses = _validation_losses(model, val_loader, device)
        if not train_losses or not val_losses:
            print(f"  Epoch {epoch+1}: No valid batches, stopping")
            break

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(progress.format(epoch=epoch + 1, train=train_loss, val=val_loss, lr=lr))

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if not history['train_loss']:
        raise RuntimeError("Training produced no usable epoch (every training or validation batch "
                           "had a NaN loss); no model was saved")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val_loss


def _train_epoch(model, loader, optimizer, device, epoch: int) -> List[float]:
    """One pass over the training batches; returns their losses. NaN batches are skipped."""
    import torch

    model.train()
    losses = []
    for x_batch, y_batch, m_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        m_batch = m_batch.to(device)

        optimizer.zero_grad()
        loss = masked_mse(model(x_batch), y_batch, m_batch)
        if torch.isnan(loss):
            print(f"  Warning: NaN loss at epoch {epoch+1}, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return losses


def _validation_losses(model, loader, device) -> List[float]:
    """Losses of the validation batches whose loss is not NaN."""
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for x_batch, y_batch, m_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            m_batch = m_batch.to(device)
            loss = masked_mse(model(x_batch), y_batch, m_batch)
            if not torch.isnan(loss):
                losses.append(loss.item())
    return losses


# Model folders and menus

def model_configs(models_dir: str, config_file: str, logger: logging.Logger):
    """(folder, config) for each folder of models_dir with a config_file; logger warns of each one unreadable."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return
    for subdir in models_path.iterdir():
        config_path = subdir / config_file
        if not subdir.is_dir() or not config_path.exists():
            continue
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping %s: could not read its %s: %s", subdir, config_file, exc)
            continue
        yield subdir, config


def training_libraries(base_dir: str) -> List[Dict[str, Any]]:
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
                _completer_log.warning("Could not count the SEDs in %s: %s", lookup, exc)
        libraries.append({'name': name, 'path': path, 'n_seds': n_seds})
    return libraries


def print_libraries(libraries: List[Dict[str, Any]]) -> None:
    """One numbered line per library: its name and its number of SEDs."""
    for i, library in enumerate(libraries, 1):
        print(f"  {i:2d}) {library['name']:30s} ({library['n_seds']} SEDs)")


def action_menu(actions) -> Optional[int]:
    """Ask which of ``actions``, each (label, packages it needs, function), to run, and run it.

    Returns 1, having said what to install, when a package the chosen action needs is missing.
    """
    print("\nWhat would you like to do?")
    for number, (label, _, _) in enumerate(actions, 1):
        print(f"  {number}) {label}")
    print("  0) Back")
    
    choice = input("> ").strip()
    if choice == "0":
        return None
    for number, (_, packages, run) in enumerate(actions, 1):
        if choice == str(number):
            if packages and refused_without(*packages):
                return 1
            run()
            return None
    print("Invalid choice")
    return None
