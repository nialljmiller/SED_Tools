# stellar_colors/cube/__init__.py
"""
Data cube module for stellar-colors package.

This module provides tools for building and querying flux cubes from
stellar atmosphere model collections for efficient interpolation.
"""

from .builder import DataCubeBuilder, FluxCube, build_flux_cube

__all__ = [
    'DataCubeBuilder',
    'FluxCube', 
    'build_flux_cube',
]