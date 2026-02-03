import torch
import numpy as np
import os
import struct
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any

def get_device():
    """Automatically detect and return the best available device."""
    if torch.cuda.is_available():
        # Set some GPU optimizations
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class FluxCubeDataset(Dataset):
    """Memory-efficient dataset for reading from flux_cube.bin."""
    def __init__(self, library_path: str, max_samples: Optional[int] = None, transform=None):
        self.library_path = library_path
        self.transform = transform
        
        cube_path = os.path.join(library_path, "flux_cube.bin")
        if not os.path.exists(cube_path):
            raise FileNotFoundError(f"flux_cube.bin not found in {library_path}")
            
        with open(cube_path, 'rb') as f:
            header = f.read(16)
            self.nt, self.nl, self.nm, self.nw = struct.unpack('4i', header)
            
            self.teff_grid = np.fromfile(f, dtype=np.float64, count=self.nt)
            self.logg_grid = np.fromfile(f, dtype=np.float64, count=self.nl)
            self.meta_grid = np.fromfile(f, dtype=np.float64, count=self.nm)
            self.wavelengths = np.fromfile(f, dtype=np.float64, count=self.nw)
            
        self.total_size = self.nt * self.nl * self.nm
        self.indices = np.arange(self.total_size)
        if max_samples and max_samples < self.total_size:
            np.random.shuffle(self.indices)
            self.indices = self.indices[:max_samples]
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Calculate grid coordinates
        i_t = real_idx // (self.nl * self.nm)
        remainder = real_idx % (self.nl * self.nm)
        i_l = remainder // self.nm
        i_m = remainder % self.nm
        
        # Read specific flux vector from binary file
        # Offset: header (16) + grids (nt+nl+nm+nw)*8 + real_idx * nw * 8
        offset = 16 + (self.nt + self.nl + self.nm + self.nw) * 8 + real_idx * self.nw * 8
        
        with open(os.path.join(self.library_path, "flux_cube.bin"), 'rb') as f:
            f.seek(offset)
            flux = np.fromfile(f, dtype=np.float64, count=self.nw)
            
        params = np.array([self.teff_grid[i_t], self.logg_grid[i_l], self.meta_grid[i_m]], dtype=np.float32)
        
        if self.transform:
            params, flux = self.transform(params, flux)
            
        return torch.from_numpy(params), torch.from_numpy(flux.astype(np.float32))

def create_dataloaders(dataset, batch_size=64, val_split=0.2):
    """Split dataset and create DataLoaders."""
    from torch.utils.data import random_split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
