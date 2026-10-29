import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple

def get_device():
    """Automatically detect and return the best available device."""
    if torch.cuda.is_available():
        # Set some GPU optimizations
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training loop shared by SEDCompleter and SEDGenerator
# ---------------------------------------------------------------------------

def masked_mse(y_pred, y_true, mask):
    """Mean squared error over the points where mask is 1."""
    sq_error = (y_pred - y_true) ** 2
    masked_error = sq_error * mask
    n_valid = mask.sum() + 1e-8  # Avoid division by zero
    return masked_error.sum() / n_valid


def data_loaders(train_arrays, val_arrays, batch_size: int):
    """DataLoaders over (inputs, targets, masks) arrays; only training batches are shuffled."""
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
