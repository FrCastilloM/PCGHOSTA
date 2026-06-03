"""
Funciones matemáticas básicas para GHOSS
"""

import numpy as np
from scipy.interpolate import splprep, splev


def gtrap_(x, b, a):
    """
    Función trampa auxiliar
    """
    return 1/2 * (1 + np.tanh((np.cos(x) + np.cos(b/2)) / (a/2 * np.sin(b/2)))) - (1 - b/(2*np.pi))


def gtrap(x, z, b, a):
    """
    Función trampa en 2D
    """
    return gtrap_(x, b, a) + gtrap_(z, b, a)


def hext(h0, a, b, NX):
    """
    Campo externo del solenoide
    
    Parameters:
    -----------
    h0 : float
        Amplitud del campo
    a, b : float
        Parámetros de la trampa
    NX : int
        Número de puntos en x
    
    Returns:
    --------
    ndarray : Campo externo
    """
    i = np.arange(NX)
    varx = 2 * np.pi * i / NX
    k = -h0 * (2 - b/np.pi)
    return h0 * gtrap(varx, np.pi, b, a) + k


def spline(x, y, s=0):
    """
    Interpolación spline
    
    Parameters:
    -----------
    x, y : array_like
        Coordenadas de los puntos
    s : float
        Factor de suavizado
    
    Returns:
    --------
    tuple : Coordenadas interpoladas (x_new, y_new)
    """
    X = np.linspace(0, 1, 100 * len(x))
    tck, u = splprep([x, y], s=s)
    return splev(X, tck)


def f(x, p1, p2):
    """
    Función auxiliar para corriente externa
    """
    return 0.5 * (1 + np.tanh((np.cos(x) + np.cos(p2/2)) / (p1/2 * np.sin(p2/2))))


def Jext(x, j0, p1, p2):
    """
    Corriente externa
    
    Parameters:
    -----------
    x : array_like
        Coordenadas
    j0 : float
        Amplitud de corriente
    p1, p2 : float
        Parámetros de la corriente
    
    Returns:
    --------
    ndarray : Corriente externa
    """
    return j0 * (f(x, p1, p2) - (1 - p2/(2*np.pi)))


def perfil_vortice_tinkham(x, xi):
    """
    Perfil de vórtice según Tinkham
    
    Parameters:
    -----------
    x : array_like
        Coordenadas radiales
    xi : float
        Longitud de coherencia
    
    Returns:
    --------
    ndarray : Perfil del vórtice
    """
    return np.tanh(np.abs(x)/xi)**2
