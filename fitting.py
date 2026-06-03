"""
Funciones de ajuste y estadísticas
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def stats(ajuste, xdata, ydata, NumeroDeParametros):
    """
    Calcula estadasticas del ajuste
    
    Parameters:
    -----------
    ajuste : array_like
        Valores ajustados
    xdata, ydata : array_like
        Datos originales
    NumeroDeParametros : int
        Numero de parametros del modelo
    
    Returns:
    --------
    tuple : (r_cuadrado, r_CuadradoAjustado, chi_cuadrado, residuos, sigma)
    """
    residuos = ydata - ajuste
    promedio = np.mean(ydata)
    
    TSS = sum((ydata - promedio)**2)
    RSS = sum(residuos**2)
    
    # R^2 y R^2 ajustado
    r_cuadrado = 1 - RSS/TSS
    n = len(xdata)
    v = NumeroDeParametros
    r_CuadradoAjustado = 1 - (1 - r_cuadrado) * (n - 1) / (n - v)
    
    # Chi^2
    chi_cuadrado = sum((residuos)**2 / abs(ajuste))
    
    # Desviación estándar
    sigma = np.sqrt(1/(len(xdata) - 1) * TSS)
    
    return (r_cuadrado, r_CuadradoAjustado, chi_cuadrado, residuos, sigma)


def lineal(x, m, b):
    """
    Funcion lineal
    """
    return m * x + b


def linear_adjust(x, y, erry=None, asigma=False):
    """
    Ajuste lineal
    
    Parameters:
    -----------
    x, y : array_like
        Datos a ajustar
    erry : array_like, optional
        Errores en y
    asigma : bool
        Si True, trata erry como desviaciones absolutas
    
    Returns:
    --------
    tuple : ([m, b], [errm, errb])
    """
    parameters, covariance = curve_fit(lineal, x, y, sigma=erry, absolute_sigma=asigma)
    paramErrors = np.sqrt(np.diag(covariance))
    
    m, b = parameters
    errm, errb = paramErrors
    
    return [m, b], [errm, errb]


def gauss(x, A, mu, sdev, b):
    """
    Función gaussiana con offset
    """
    return A * np.exp(-(x - mu)**2 / (2 * sdev**2)) + b


def vortex_adjust(x, y, erry=None):
    """
    Ajuste gaussiano para vórtices
    
    Parameters:
    -----------
    x, y : array_like
        Datos del perfil del vórtice
    erry : array_like, optional
        Errores en y
    
    Returns:
    --------
    tuple : (parameters, paramErrors, [x_ajuste, y_ajuste])
    """
    from .parameters import get_param
    
    xi = get_param('lambda')
    peaks, _ = find_peaks(y, height=0.5, width=8, distance=10)
    
    parameters, covariance = curve_fit(
        gauss, x, y,
        sigma=erry,
        absolute_sigma=True,
        p0=[1, x[peaks[0]], xi**2, 0]
    )
    
    paramErrors = np.sqrt(np.diag(covariance))
    
    x_ = np.linspace(min(x), max(x), len(x) * 100)
    ajuste = gauss(x_, parameters[0], parameters[1], parameters[2], parameters[3])
    
    return parameters, paramErrors, [x_, ajuste]
