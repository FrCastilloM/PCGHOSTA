"""
Funciones para análisis de corrientes
"""

import numpy as np
from ..parameters import get_param


def calculate_current_density(totcurr, scurr, jext, normalize=True):
    """
    Calcula densidades de corriente normalizadas
    
    Parameters:
    -----------
    totcurr : ndarray
        Corriente total
    scurr : ndarray
        Supercorriente
    jext : ndarray
        Corriente externa
    normalize : bool
        Si True, normaliza por Jc
    
    Returns:
    --------
    dict : Diccionario con componentes de corriente
    """
    if normalize:
        xi = get_param('lambda')
        kappa = get_param('vparam2')
        cs = get_param('cspeed')
        
        london = xi * kappa
        hc2 = np.sqrt(2) * cs / xi
        jc = hc2 / london
    else:
        jc = 1.0
    
    J_total = totcurr / jc
    J_super = scurr / jc
    J_ext = jext / jc
    J_normal = J_total - J_super + J_ext
    
    return {
        'total': J_total,
        'super': J_super,
        'normal': J_normal,
        'external': J_ext
    }


def apply_nematic_correction(scurr, nem, axis='z'):
    """
    Aplica corrección nemática a la supercorriente
    
    Parameters:
    -----------
    scurr : ndarray
        Supercorriente
    nem : ndarray
        Parámetro de orden nemático
    axis : str
        Eje ('x' o 'z')
    
    Returns:
    --------
    ndarray : Supercorriente corregida
    """
    lambda1 = get_param('vparam8')
    
    if axis == 'x':
        Xi = 1 + lambda1 * nem
    elif axis == 'z':
        Xi = 1 - lambda1 * nem
    else:
        raise ValueError("axis debe ser 'x' o 'z'")
    
    return scurr * Xi


def calculate_london_field(current_gradient, normalize=True):
    """
    Calcula el campo magnético a partir del gradiente de corriente (ecuación de London)
    
    Parameters:
    -----------
    current_gradient : ndarray
        Gradiente de la corriente
    normalize : bool
        Si True, normaliza por Hc2
    
    Returns:
    --------
    ndarray : Campo magnético
    """
    xi = get_param('lambda')
    kappa = get_param('vparam2')
    
    london = xi * kappa
    
    if normalize:
        cs = get_param('cspeed')
        hc2 = np.sqrt(2) * cs / xi
        return current_gradient * london / hc2
    else:
        return current_gradient * london


def analyze_vortex_core(rho, wy, x_center, z_center, radius=5):
    """
    Analiza las propiedades del núcleo del vórtice
    
    Parameters:
    -----------
    rho : ndarray
        Densidad del parámetro de orden
    wy : ndarray
        Campo magnético
    x_center, z_center : int
        Coordenadas del centro del vórtice
    radius : int
        Radio de análisis en puntos de grilla
    
    Returns:
    --------
    dict : Propiedades del núcleo
    """
    NX, NZ = rho.shape
    
    # Extraer región del núcleo
    x_min = max(0, x_center - radius)
    x_max = min(NX, x_center + radius)
    z_min = max(0, z_center - radius)
    z_max = min(NZ, z_center + radius)
    
    rho_core = rho[x_min:x_max, z_min:z_max]
    wy_core = wy[x_min:x_max, z_min:z_max]
    
    return {
        'min_density': np.min(rho_core),
        'max_field': np.max(np.abs(wy_core)),
        'avg_density': np.mean(rho_core),
        'avg_field': np.mean(wy_core),
        'core_size': 2 * radius
    }
