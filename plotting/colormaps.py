"""
Colormaps personalizados para GHOSS
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt

from ..config import dirac_ver


if not dirac_ver:
    inferno = plt.get_cmap('inferno_r')
    cividis = plt.get_cmap('bone')
    inferno_ = plt.get_cmap('inferno')


def combined_colormap(x, min_val, max_val):
    """
    Colormap combinado para valores positivos y negativos
    """
    if min_val == 0:
        min_val = 1
    if x <= 0:
        return inferno((x - min_val) / (-min_val))
    else:
        return cividis((x) / max_val)


def make_colormap(min_val, max_val):
    """
    Genera colormap combinado
    
    Parameters:
    -----------
    min_val, max_val : float
        Valores mínimo y máximo
    
    Returns:
    --------
    LinearSegmentedColormap
    """
    n_bins = 100
    cmap_name = 'combined_colormap'
    return LinearSegmentedColormap.from_list(
        cmap_name,
        [combined_colormap(x, min_val, max_val) for x in np.linspace(min_val, max_val, n_bins)]
    )


def cmap_nem(x, min_val, max_val):
    """
    Colormap para nematicidad
    """
    return inferno_((x - min_val) / (max_val - min_val))


def make_cmap_nem(min_val, max_val):
    """
    Genera colormap para nematicidad
    
    Parameters:
    -----------
    min_val, max_val : float
        Valores mínimo y máximo
    
    Returns:
    --------
    LinearSegmentedColormap
    """
    n_bins = 100
    cmap_name = 'cmap_nem'
    return LinearSegmentedColormap.from_list(
        cmap_name,
        [cmap_nem(x, min_val, max_val) for x in np.linspace(min_val, max_val, n_bins)]
    )


def custom_cmap(x, p, cmapA='Purples', cmapB='plasma'):
    """
    Colormap personalizado con dos colormaps
    """
    if x <= p:
        return cmapA(x/p)
    else:
        return cmapB((x - p) / (1 - p))


def make_custom_cmap(p, cmapA='Purples', cmapB='plasma'):
    """
    Genera colormap personalizado
    
    Parameters:
    -----------
    p : float
        Punto de transición entre colormaps
    cmapA, cmapB : str
        Nombres de los colormaps
    
    Returns:
    --------
    LinearSegmentedColormap
    """
    if not dirac_ver:
        cmapA = plt.get_cmap(cmapA)
        cmapB = plt.get_cmap(cmapB)
    
    n_bins = 100
    cmap_name = 'custom_cmap'
    return LinearSegmentedColormap.from_list(
        cmap_name,
        [custom_cmap(x, p, cmapA, cmapB) for x in np.linspace(0, 1, n_bins)]
    )
