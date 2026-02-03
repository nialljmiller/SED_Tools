import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from .ml_utils import get_device, FluxCubeDataset, create_dataloaders
from tqdm import tqdm

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(x + self.block(x))

class SEDModel(nn.Module):
    """Enhanced Neural Network for SED tasks."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

class AutoTrainer:
    """Automated training pipeline with hyperparameter optimization."""
    def __init__(self, library_path: str, task: str = "generator"):
        self.library_path = library_path
        self.task = task
        self.device = get_device()
        print(f"Using device: {self.device}")

    def objective(self, trial, train_loader, val_loader, input_dim, output_dim):
        # Hyperparameters to optimize
        hidden_dim = trial.suggest_int("hidden_dim", 128, 1024, step=128)
        num_blocks = trial.suggest_int("num_blocks", 1, 5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        
        model = SEDModel(input_dim, output_dim, hidden_dim, num_blocks, dropout).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Short training for optimization
        for epoch in range(10):  # 10 epochs per trial
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        
        return val_loss / len(val_loader)

    def train_best(self, library_path, output_path, n_trials=20, epochs=100):
        dataset = FluxCubeDataset(library_path)
        # For simplicity in this example, we assume generator task (params -> flux)
        # A more robust version would handle completer (flux_part + params -> flux_full)
        
        input_dim = 3 # Teff, logg, [M/H]
        output_dim = dataset.nw
        
        train_loader, val_loader = create_dataloaders(dataset)
        
        print(f"Starting hyperparameter optimization ({n_trials} trials)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, train_loader, val_loader, input_dim, output_dim), n_trials=n_trials)
        
        print("Best hyperparameters:", study.best_params)
        
        # Train final model with best params
        best_params = study.best_params.copy()
        lr = best_params.pop('lr')
        model = SEDModel(input_dim, output_dim, **best_params).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
        criterion = nn.MSELoss()
        
        print(f"Training final model for {epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = model(x)
                    val_loss += criterion(pred, y).item()
            
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), Path(output_path) / "model_best.pt")
                
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
            
        return model, study.best_params
