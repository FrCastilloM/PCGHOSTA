"""
Funciones base de plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp2d
from matplotlib.colors import ListedColormap

from ..config import outputs_path, dpi, dirac_ver
from ..parameters import get_param, get_Resolution
from ..io_utils import load_namefiles, readbin
from .colormaps import make_colormap, make_cmap_nem, make_custom_cmap


def densplot(fig, ax, title, NX, NZ, density, label, minimum=False, fs=14, maximo=0, pcmap=None, zoom_factor=1):
    """
    Gráfico de densidad genérico
    
    Parameters:
    -----------
    fig, ax : matplotlib objects
        Figura y ejes
    title : str
        Título del gráfico
    NX, NZ : int
        Resolución
    density : ndarray
        Datos a graficar
    label : str
        Etiqueta del colorbar
    minimum : bool
        Si True, usa el mínimo real en lugar de 0
    fs : int
        Tamaño de fuente
    maximo : float
        Valor máximo forzado
    pcmap : colormap, optional
        Colormap personalizado
    """
    ax.set_title(title, fontsize=fs, fontweight='bold')
    
    max_val = round(np.max(density), 3)
    min_val = round(np.min(density), 3)
    
    custom_cmap = make_colormap(min_val, max_val)
    bticks = [min_val, 0, max_val]
    
    if pcmap is not None:
        custom_cmap = pcmap
    
    if maximo != 0:
        max_val = maximo
    
    if min_val >= 0:
        if not minimum:
            min_val = 0
        if minimum:
            max_val = round(np.max(density), 6)
            min_val = round(np.min(density), 6)
            custom_cmap = make_cmap_nem(min_val, max_val)
            bticks = [min_val, 0.5 * (max_val - min_val), max_val]
        cax = ax.imshow(density, aspect='auto', cmap=custom_cmap, vmin=min_val, vmax=max_val,
                       origin='lower')
    
    if min_val < 0:
        def forward(x):
            return np.where(x < 0, 0.5 * (x - min_val) / -min_val, 0.5 + 0.5 * (x / max_val))
        
        def inverse(x):
            return np.where(x < 0.5, min_val + (x / 0.5) * -min_val, max_val * (x - 0.5) / 0.5)
        
        norm = mcolors.FuncNorm((forward, inverse), vmin=min_val, vmax=max_val)
        cax = ax.imshow(density, aspect='auto', cmap=custom_cmap,
                       origin='lower', norm=norm)
    
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', ticks=bticks)
    cbar.set_label(label, fontsize=fs)
    
    ax.set_ylabel(r'z/L', fontsize=fs)
    ax.set_xlabel(r'x/L', fontsize=fs)
    
    LX = int(NX/(2*zoom_factor))
    LZ = int(NZ/(2*zoom_factor))
    ax.set_xlim(NX//2-LX, NX//2+LX)
    ax.set_ylim(NZ//2-LZ, NZ//2+LZ)
    ax.set_xticks([NX//2-LX, NX//2, NX//2+LX])
    ax.set_xticklabels([fr'${(.5-LX/NX):.2f}\pi$', fr'$\pi$', fr'${(.5+LX/NX):.2f}\pi$'])
    ax.set_yticks([NZ//2-LZ, NZ//2, NZ//2+LZ])
    ax.set_yticklabels([fr'${(.5-LZ/NZ):.2f}\pi$', fr'$\pi$', fr'${(.5+LZ/NZ):.2f}\pi$'])

    
    ax.set_aspect(aspect='equal')


def vectorplot(fig, ax, title, NX, NZ, vecX, vecZ, label, K=30, maximo=0, fs=20):
    """
    Gráfico vectorial con streamplot
    
    Parameters:
    -----------
    fig, ax : matplotlib objects
        Figura y ejes
    title : str
        Título
    NX, NZ : int
        Resolución
    vecX, vecZ : ndarray
        Componentes del vector
    label : str
        Etiqueta
    K : int
        Densidad de líneas
    maximo : float
        Valor máximo
    fs : int
        Tamaño de fuente
    """
    ax.set_title(title, fontweight='bold', fontsize=fs)
    
    x = np.arange(0, NX)
    z = np.arange(0, NZ)
    X, Z = np.meshgrid(x, z)
    U = vecX
    V = vecZ
    
    # Mallado denso para streamplot
    N_interp = 1000
    x_new = np.linspace(0, NX, N_interp)
    z_new = np.linspace(0, NZ, N_interp)
    X_new, Z_new = np.meshgrid(x_new, z_new)
    interp_U = interp2d(x, z, U, kind='cubic')
    interp_V = interp2d(x, z, V, kind='cubic')
    U_new = interp_U(x_new, z_new)
    V_new = interp_V(x_new, z_new)
    magnitude = np.sqrt(U_new**2 + V_new**2)
    
    black_cmap = ListedColormap(['black'])
    blue_cmap = ListedColormap([(0, 0.75, 1)])
    custom_cmap = make_custom_cmap(0.001, cmapA=black_cmap, cmapB=blue_cmap)
    
    strm = ax.streamplot(X_new, Z_new, U_new, V_new, density=1.5, color=magnitude, 
                         cmap=custom_cmap, arrowstyle='simple')
    
    alpha = 0.4
    strm.lines.set_alpha(alpha)
    
    for obj in ax.get_children():
        if isinstance(obj, plt.matplotlib.patches.FancyArrowPatch):
            obj.set_alpha(alpha)
    
    ax.set_ylabel(r'z/L', fontsize=fs)
    ax.set_xlabel(r'x/L', fontsize=fs)
    ax.set_xticks([0, NX//2, NX])
    ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    ax.set_yticks([0, NZ//2, NZ])
    ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    ax.set_aspect(aspect='equal')


def Plots_densidad_sub(data, NX, NZ, name, title, label, fs=14, maximo=0, dots=[], pcmap=None, zoom_factor=1):
    """
    Crea un subplot de densidad
    
    Parameters:
    -----------
    data : ndarray
        Datos a graficar
    NX, NZ : int
        Resolución
    name : str
        Nombre del archivo de salida
    title : str
        Título
    label : str
        Etiqueta del colorbar
    fs : int
        Tamaño de fuente
    maximo : float
        Valor máximo
    dots : list
        Lista de puntos a marcar
    pcmap : colormap
        Colormap personalizado
    """
    fig, ax1 = plt.subplots()
    
    densplot(fig, ax1, title, NX, NZ, data, label, fs=fs, maximo=maximo, pcmap=pcmap, zoom_factor=zoom_factor)
    
    for i in range(len(dots)):
        ax1.axhline(y=dots[i][1] * NZ / (2 * np.pi), color='grey', lw=0.8)
        ax1.axvline(x=dots[i][0] * NX / (2 * np.pi), color='grey', lw=0.8)
    
    plt.tight_layout()
    plt.savefig(name, dpi=dpi)
    plt.close()


def Plots_vectorial_sub(dataX, dataZ, cantidad, NX, NZ, name, title, label, fs=20, maximo=0):
    """
    Crea un subplot vectorial con módulo
    
    Parameters:
    -----------
    dataX, dataZ : ndarray
        Componentes del vector
    cantidad : str
        Nombre de la cantidad
    NX, NZ : int
        Resolución
    name : str
        Nombre del archivo
    title : str
        Título
    label : str
        Etiqueta
    fs : int
        Tamaño de fuente
    maximo : float
        Valor máximo
    """
    fig, ax1 = plt.subplots()
    
    # Vectorplot
    vectorplot(fig, ax1, r'', NX, NZ, dataX, dataZ, label, K=10, maximo=maximo, fs=fs)
    
    # Densplot módulo
    modulo = np.sqrt(dataX**2 + dataZ**2)
    densplot(fig, ax1, title, NX, NZ, modulo, label, fs=fs, maximo=maximo, pcmap='inferno')
    
    plt.tight_layout()
    plt.savefig(name, dpi=dpi)
    plt.close()
