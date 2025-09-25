#!/usr/bin/env python3
"""
Plotting and visualization module for stellar-colors package.

This module provides functions for visualizing stellar atmosphere model grids,
flux cubes, and parameter spaces using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'plot_parameter_space_2d',
    'plot_parameter_space_3d', 
    'plot_flux_cube_analysis',
    'plot_spectrum_comparison',
    'plot_model_coverage',
    'create_comprehensive_plots'
]


def plot_parameter_space_2d(
    teff: np.ndarray,
    logg: np.ndarray, 
    metallicity: np.ndarray,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create 2D projections of parameter space coverage.
    
    Parameters
    ----------
    teff : np.ndarray
        Effective temperature values
    logg : np.ndarray
        Surface gravity values
    metallicity : np.ndarray
        Metallicity values
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Stellar Parameter Space - 2D Projections', fontsize=16)
    
    # Teff vs log g (HR diagram style)
    ax1 = axes[0, 0]
    hist1, xedges1, yedges1 = np.histogram2d(teff, logg, bins=30)
    extent1 = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]]
    
    im1 = ax1.imshow(hist1.T, origin='lower', extent=extent1, 
                     cmap='Blues', aspect='auto')
    ax1.set_xlabel('Effective Temperature (K)')
    ax1.set_ylabel('Surface Gravity (log g)')
    ax1.set_title('HR Diagram Style (Teff vs log g)')
    ax1.invert_yaxis()  # Conventional HR diagram orientation  
    plt.colorbar(im1, ax=ax1, label='Number of Models')
    
    # Teff vs Metallicity
    ax2 = axes[0, 1]
    hist2, xedges2, yedges2 = np.histogram2d(teff, metallicity, bins=25)
    extent2 = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]]
    
    im2 = ax2.imshow(hist2.T, origin='lower', extent=extent2,
                     cmap='Greens', aspect='auto')
    ax2.set_xlabel('Effective Temperature (K)')
    ax2.set_ylabel('Metallicity [M/H]')
    ax2.set_title('Temperature vs Metallicity')
    plt.colorbar(im2, ax=ax2, label='Number of Models')
    
    # log g vs Metallicity
    ax3 = axes[1, 0]
    hist3, xedges3, yedges3 = np.histogram2d(logg, metallicity, bins=20)
    extent3 = [xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]]
    
    im3 = ax3.imshow(hist3.T, origin='lower', extent=extent3,
                     cmap='Reds', aspect='auto')
    ax3.set_xlabel('Surface Gravity (log g)')
    ax3.set_ylabel('Metallicity [M/H]')
    ax3.set_title('Surface Gravity vs Metallicity')
    plt.colorbar(im3, ax=ax3, label='Number of Models')
    
    # Parameter distributions
    ax4 = axes[1, 1]
    ax4.hist(teff/100, bins=30, alpha=0.7, label='Teff/100', density=True)
    ax4.hist(logg*1000, bins=30, alpha=0.7, label='log g × 1000', density=True)
    ax4.hist((metallicity+2)*1000, bins=30, alpha=0.7, 
             label='([M/H]+2) × 1000', density=True)
    
    ax4.set_xlabel('Scaled Parameter Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Parameter Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved 2D parameter space plot: {save_path}")
    
    return fig


def plot_parameter_space_3d(
    teff: np.ndarray,
    logg: np.ndarray,
    metallicity: np.ndarray,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Union[str, Path]] = None,
    color_by: str = 'teff'
) -> plt.Figure:
    """
    Create 3D scatter plot of parameter space coverage.
    
    Parameters
    ----------
    teff : np.ndarray
        Effective temperature values
    logg : np.ndarray
        Surface gravity values  
    metallicity : np.ndarray
        Metallicity values
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the plot
    color_by : str
        Parameter to color points by ('teff', 'logg', or 'metallicity')
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Choose color mapping
    color_maps = {
        'teff': ('viridis', teff, 'Temperature (K)'),
        'logg': ('plasma', logg, 'Surface Gravity (log g)'),
        'metallicity': ('RdYlBu_r', metallicity, 'Metallicity [M/H]')
    }
    
    cmap, color_values, color_label = color_maps[color_by]
    
    # Create scatter plot
    scatter = ax.scatter(teff, logg, metallicity,
                        c=color_values, cmap=cmap, alpha=0.6, s=30)
    
    ax.set_xlabel('Effective Temperature (K)', fontsize=12)
    ax.set_ylabel('Surface Gravity (log g)', fontsize=12)
    ax.set_zlabel('Metallicity [M/H]', fontsize=12)
    ax.set_title('3D Parameter Space Coverage', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label(color_label, fontsize=12)
    
    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved 3D parameter space plot: {save_path}")
    
    return fig


def plot_flux_cube_analysis(
    flux_cube,
    figsize: Tuple[int, int] = (20, 15),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create comprehensive analysis plots for a flux cube.
    
    Parameters
    ---------- 
    flux_cube : FluxCube
        The flux cube to analyze
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.subplots(2, 3, figsize=figsize)[0]
    fig.suptitle('Flux Cube Analysis', fontsize=16)
    
    # Get parameter grids from cube
    teff_grid = flux_cube.teff_grid
    logg_grid = flux_cube.logg_grid
    meta_grid = flux_cube.meta_grid
    wavelength_grid = flux_cube.wavelength_grid
    
    # Create coordinate meshes for plotting
    teff_coords, logg_coords, meta_coords = np.meshgrid(
        teff_grid, logg_grid, meta_grid, indexing='ij'
    )
    
    # Flatten for plotting
    teff_flat = teff_coords.flatten()
    logg_flat = logg_coords.flatten()
    meta_flat = meta_coords.flatten()
    
    # 1. 3D parameter space
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(teff_flat, logg_flat, meta_flat,
                         c=teff_flat, cmap='plasma', alpha=0.6, s=20)
    ax1.set_xlabel('Teff (K)')
    ax1.set_ylabel('log g')
    ax1.set_zlabel('[M/H]')
    ax1.set_title('3D Parameter Coverage')
    plt.colorbar(scatter, ax=ax1, shrink=0.5)
    
    # 2. HR diagram
    ax2 = fig.add_subplot(2, 3, 2)
    hist, xe, ye = np.histogram2d(teff_flat, logg_flat, bins=30)
    im2 = ax2.imshow(hist.T, origin='lower', 
                     extent=[xe[0], xe[-1], ye[0], ye[-1]],
                     cmap='Blues', aspect='auto')
    ax2.set_xlabel('Teff (K)')
    ax2.set_ylabel('log g')
    ax2.set_title('HR Diagram Coverage')
    ax2.invert_yaxis()
    plt.colorbar(im2, ax=ax2)
    
    # 3. Sample spectra at different temperatures
    ax3 = fig.add_subplot(2, 3, 3)
    mid_logg = len(logg_grid) // 2
    mid_meta = len(meta_grid) // 2
    
    # Plot spectra for different temperatures
    temp_indices = [0, len(teff_grid)//4, len(teff_grid)//2, 
                   3*len(teff_grid)//4, -1]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (temp_idx, color) in enumerate(zip(temp_indices, colors)):
        spectrum = flux_cube.flux_cube[temp_idx, mid_logg, mid_meta, :]
        ax3.plot(wavelength_grid, spectrum, color=color, 
                label=f'T={teff_grid[temp_idx]:.0f}K', alpha=0.8)
    
    ax3.set_xlabel('Wavelength (Å)')
    ax3.set_ylabel('Flux (arbitrary units)')
    ax3.set_title('Sample Spectra at Different Temperatures')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Wavelength coverage
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Show min/max flux at each wavelength
    flux_min = np.min(flux_cube.flux_cube, axis=(0, 1, 2))
    flux_max = np.max(flux_cube.flux_cube, axis=(0, 1, 2))
    flux_mean = np.mean(flux_cube.flux_cube, axis=(0, 1, 2))
    
    ax4.fill_between(wavelength_grid, flux_min, flux_max, alpha=0.3, 
                     label='Min-Max Range')
    ax4.plot(wavelength_grid, flux_mean, 'r-', label='Mean Flux')
    
    ax4.set_xlabel('Wavelength (Å)')
    ax4.set_ylabel('Flux Range')
    ax4.set_title('Wavelength Coverage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Parameter ranges
    ax5 = fig.add_subplot(2, 3, 5)
    
    ranges = {
        'Teff': (teff_grid.min(), teff_grid.max()),
        'log g': (logg_grid.min(), logg_grid.max()),
        '[M/H]': (meta_grid.min(), meta_grid.max()),
        'λ': (wavelength_grid.min(), wavelength_grid.max())
    }
    
    params = list(ranges.keys())
    mins = [ranges[p][0] for p in params]
    maxs = [ranges[p][1] for p in params]
    
    x_pos = np.arange(len(params))
    ax5.bar(x_pos, maxs, alpha=0.6, label='Maximum')
    ax5.bar(x_pos, mins, alpha=0.6, label='Minimum')
    
    ax5.set_xlabel('Parameters')
    ax5.set_ylabel('Values')
    ax5.set_title('Parameter Ranges')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(params)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Cube statistics
    ax6 = fig.add_subplot(2, 3, 6)
    
    stats_text = f"""
    Flux Cube Statistics:
    
    Shape: {flux_cube.flux_cube.shape}
    
    Temperature Points: {len(teff_grid)}
    Gravity Points: {len(logg_grid)}  
    Metallicity Points: {len(meta_grid)}
    Wavelength Points: {len(wavelength_grid)}
    
    Teff Range: {teff_grid.min():.0f} - {teff_grid.max():.0f} K
    log g Range: {logg_grid.min():.2f} - {logg_grid.max():.2f}
    [M/H] Range: {meta_grid.min():.2f} - {meta_grid.max():.2f}
    λ Range: {wavelength_grid.min():.0f} - {wavelength_grid.max():.0f} Å
    
    Total Grid Points: {flux_cube.flux_cube.shape[0] * flux_cube.flux_cube.shape[1] * flux_cube.flux_cube.shape[2]:,}
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Cube Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved flux cube analysis plot: {save_path}")
    
    return fig


def plot_spectrum_comparison(
    wavelengths_list: List[np.ndarray],
    fluxes_list: List[np.ndarray],
    labels: List[str],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Compare multiple stellar spectra.
    
    Parameters
    ----------
    wavelengths_list : list of np.ndarray
        List of wavelength arrays
    fluxes_list : list of np.ndarray  
        List of flux arrays
    labels : list of str
        Labels for each spectrum
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the plot
    normalize : bool
        Whether to normalize spectra for comparison
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(wavelengths_list)))
    
    for i, (waves, fluxes, label) in enumerate(zip(wavelengths_list, fluxes_list, labels)):
        if normalize:
            # Normalize to peak flux
            fluxes = fluxes / np.max(fluxes)
            # Add vertical offset for clarity
            fluxes = fluxes + i * 0.2
        
        ax.plot(waves, fluxes, color=colors[i], label=label, alpha=0.8)
    
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalized Flux' if normalize else 'Flux')
    ax.set_title('Stellar Spectrum Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved spectrum comparison plot: {save_path}")
    
    return fig


def plot_model_coverage(
    lookup_table: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot model coverage and parameter distributions from lookup table.
    
    Parameters
    ----------
    lookup_table : pd.DataFrame
        Model lookup table with stellar parameters
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure  
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Model Collection Coverage Analysis', fontsize=16)
    
    # Extract parameters
    teff = lookup_table['teff'].values
    logg = lookup_table['logg'].values
    metallicity = lookup_table['metallicity'].values
    
    # HR diagram
    ax1 = axes[0, 0]
    ax1.scatter(teff, logg, alpha=0.6, s=20)
    ax1.set_xlabel('Effective Temperature (K)')
    ax1.set_ylabel('Surface Gravity (log g)')
    ax1.set_title('HR Diagram')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # Parameter histograms
    ax2 = axes[0, 1]
    ax2.hist(teff, bins=30, alpha=0.7, color='blue', label='Teff')
    ax2.set_xlabel('Effective Temperature (K)')
    ax2.set_ylabel('Count')
    ax2.set_title('Temperature Distribution')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.hist(logg, bins=20, alpha=0.7, color='green', label='log g')
    ax3.set_xlabel('Surface Gravity (log g)')
    ax3.set_ylabel('Count')
    ax3.set_title('Surface Gravity Distribution')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.hist(metallicity, bins=20, alpha=0.7, color='red', label='[M/H]')
    ax4.set_xlabel('Metallicity [M/H]')
    ax4.set_ylabel('Count')
    ax4.set_title('Metallicity Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved model coverage plot: {save_path}")
    
    return fig


def create_comprehensive_plots(
    flux_cube,
    output_dir: Union[str, Path],
    prefix: str = "stellar_colors"
) -> Dict[str, Path]:
    """
    Create a comprehensive set of plots for flux cube analysis.
    
    Parameters
    ----------
    flux_cube : FluxCube
        The flux cube to analyze
    output_dir : str or Path
        Directory to save plots
    prefix : str
        Prefix for plot filenames
        
    Returns
    -------
    dict
        Dictionary mapping plot types to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_files = {}
    
    # Get parameter arrays
    teff_coords, logg_coords, meta_coords = np.meshgrid(
        flux_cube.teff_grid, flux_cube.logg_grid, flux_cube.meta_grid, indexing='ij'
    )
    teff_flat = teff_coords.flatten()
    logg_flat = logg_coords.flatten()
    meta_flat = meta_coords.flatten()
    
    # 2D parameter space plots
    fig_2d = plot_parameter_space_2d(teff_flat, logg_flat, meta_flat)
    path_2d = output_dir / f"{prefix}_parameter_space_2d.png"
    fig_2d.savefig(path_2d, dpi=150, bbox_inches='tight')
    plot_files['2d_parameter_space'] = path_2d
    plt.close(fig_2d)
    
    # 3D parameter space plot
    fig_3d = plot_parameter_space_3d(teff_flat, logg_flat, meta_flat)
    path_3d = output_dir / f"{prefix}_parameter_space_3d.png"
    fig_3d.savefig(path_3d, dpi=150, bbox_inches='tight')
    plot_files['3d_parameter_space'] = path_3d
    plt.close(fig_3d)
    
    # Comprehensive flux cube analysis
    fig_cube = plot_flux_cube_analysis(flux_cube)
    path_cube = output_dir / f"{prefix}_flux_cube_analysis.png"
    fig_cube.savefig(path_cube, dpi=150, bbox_inches='tight')
    plot_files['flux_cube_analysis'] = path_cube
    plt.close(fig_cube)
    
    logger.info(f"Created comprehensive plots in {output_dir}")
    for plot_type, path in plot_files.items():
        logger.info(f"  {plot_type}: {path}")
    
    return plot_files