# stellar_colors/config.py
"""
Configuration management for the stellar_colors package.
"""

import os
from pathlib import Path
import logging

class Config:
    """Configuration class for stellar_colors package."""
    
    def __init__(self):
        # Default configuration values
        self.max_download_workers = 5
        self.download_timeout = 30.0
        self.default_interpolation_method = 'linear'
        self.magnitude_system = 'vega'
        self.data_dir = Path.home() / '.stellar_colors'
        self.models_dir = 'models'
        self.filters_dir = 'filters'
        
        # Ensure data directory exists
        self._create_data_dirs()

    def _create_data_dirs(self):
        """Create data directories if they don't exist."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.get_models_dir().mkdir(parents=True, exist_ok=True)
            self.get_filters_dir().mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create data directories: {e}")
            raise

    def get_data_dir(self) -> Path:
        """Return the main data directory."""
        return self.data_dir

    def get_models_dir(self) -> Path:
        """Return the models subdirectory."""
        return self.data_dir / self.models_dir

    def get_filters_dir(self) -> Path:
        """Return the filters subdirectory."""
        return self.data_dir / self.filters_dir

# Singleton configuration instance
conf = Config()

def get_data_dir() -> Path:
    """Convenience function to get data directory."""
    return conf.get_data_dir()

def get_models_dir() -> Path:
    """Convenience function to get models directory."""
    return conf.get_models_dir()

def get_filters_dir() -> Path:
    """Convenience function to get filters directory."""
    return conf.get_filters_dir()