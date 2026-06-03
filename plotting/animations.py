"""
Funciones para crear animaciones y GIFs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ..config import outputs_path, dpi
from ..parameters import get_param, get_Resolution, get_timeparam
from ..io_utils import load_namefiles, readbin, create_gif
from .base import Plots_densidad_sub, Plots_vectorial_sub


def Plots_densidad(densidad, label, gif=True, step=1, initial=0, duration=500, fs=14, 
                   maximo=0, dots=[], factor=1, pcmap=None, zoom_factor=1):
    """
    Genera plots de densidad para todos los timesteps
    
    Parameters:
    -----------
    densidad : str
        Nombre de la cantidad a graficar
    label : str
        Etiqueta del colorbar
    gif : bool
        Si True, genera GIF
    step : int
        Paso entre frames
    initial : int
        Frame inicial
    duration : int
        Duración de cada frame en ms
    fs : int
        Tamaño de fuente
    maximo : float
        Valor máximo
    dots : list
        Puntos a marcar
    factor : float
        Factor de escala
    pcmap : colormap
        Colormap personalizado
    """
    path = outputs_path
    
    NX, NY, NZ = get_Resolution()
    dt, tstep = get_timeparam()
    shape = (NX, NY, NZ)
    
    xi = get_param('lambda')
    cs = get_param('cspeed')
    hc2 = np.sqrt(2) * cs / xi
    
    filenames, _ = load_namefiles(path, densidad)
    
    for i in range(initial, len(filenames), step):
        t = round(dt * tstep * i * hc2, 2)
        matriz = readbin(path + '/' + filenames[i], shape) * factor
        name = path + '/plots.' + densidad + '.{}.png'.format(i + 1)
        
        if np.any(np.isnan(matriz)):
            break
        
        suptitle = f'$t/t_R={t}$'
        Plots_densidad_sub(matriz, NX, NZ, name, suptitle, label, fs=fs, maximo=maximo,
                          dots=dots, pcmap=pcmap, zoom_factor=zoom_factor)
    
    if gif:
        create_gif(path, 'plots.' + densidad, duration=duration)


def Plots_vectorial(cantidad, label, gif=False, step=1, initial=0, duration=500, fs=20,
                    maximo=0, factor=1, operacion=None, figname=None):
    """
    Genera plots vectoriales para todos los timesteps
    
    Parameters:
    -----------
    cantidad : str or list
        Nombre(s) de la(s) cantidad(es)
    label : str
        Etiqueta
    gif : bool
        Si True, genera GIF
    step : int
        Paso entre frames
    initial : int
        Frame inicial
    duration : int
        Duración en ms
    fs : int
        Tamaño de fuente
    maximo : float
        Valor máximo
    factor : float
        Factor de escala
    operacion : list, optional
        Operaciones entre binarios
    figname : str, optional
        Nombre del archivo de salida
    """
    if figname is None:
        figname = cantidad
    
    path = outputs_path
    
    NX, NY, NZ = get_Resolution()
    dt, tstep = get_timeparam()
    shape = (NX, NY, NZ)
    
    xi = get_param('lambda')
    cs = get_param('cspeed')
    hc2 = np.sqrt(2) * cs / xi
    
    # Caso operación entre binarios
    filenamesX = []
    filenamesZ = []
    if operacion is not None:
        for j in range(len(cantidad)):
            filenamesX.append(load_namefiles(path, cantidad[j] + 'x')[0])
            filenamesZ.append(load_namefiles(path, cantidad[j] + 'z')[0])
        
        for i in range(initial, len(filenamesX[0]), step):
            t = round(dt * tstep * i * hc2, 2)
            
            matrizX = np.zeros((NX, NZ))
            matrizZ = np.zeros((NX, NZ))
            
            for j in range(len(cantidad)):
                matrizX += operacion[j] * readbin(path + '/' + filenamesX[j][i], shape) * factor
                matrizZ += operacion[j] * readbin(path + '/' + filenamesZ[j][i], shape) * factor
            
            if np.any(np.isnan(matrizX)):
                break
            
            name = path + '/plots.' + figname + '.{}.png'.format(i + 1)
            title = f'$t/t_R={t}$'
            
            Plots_vectorial_sub(matrizX, matrizZ, cantidad, NX, NZ, name, title, label, 
                               fs=fs, maximo=maximo)
    
    # Caso normal (1 binario)
    else:
        filenamesX, _ = load_namefiles(path, cantidad + 'x')
        filenamesZ, _ = load_namefiles(path, cantidad + 'z')
        
        for i in range(initial, len(filenamesX), step):
            t = round(dt * tstep * i * hc2, 2)
            
            matrizX = readbin(path + '/' + filenamesX[i], shape) * factor
            matrizZ = readbin(path + '/' + filenamesZ[i], shape) * factor
            
            if np.any(np.isnan(matrizX)):
                break
            
            name = path + '/plots.' + figname + '.{}.png'.format(i + 1)
            title = f'$t/t_R={t}$'
            
            Plots_vectorial_sub(matrizX, matrizZ, cantidad, NX, NZ, name, title, label,
                               fs=fs, maximo=maximo)
    
    if gif:
        create_gif(path, 'plots.' + cantidad, duration=duration)


def plot_energy():
    """
    Grafica la energía libre vs tiempo
    """
    from ..parameters import get_energy
    
    X = get_energy('en_comp.txt')
    
    plt.figure()
    plt.title('Energía libre', fontsize=14)
    plt.plot(X[0], X[1], '.', color='tab:red')
    plt.plot(X[0], X[1], color='grey', alpha=0.5)
    plt.xlabel(r'Tiempo', fontsize=14)
    plt.ylabel(r'Energía', fontsize=14)
    plt.tight_layout()
    plt.grid(True)
    
    plt.savefig('./out/energy.png', dpi=300)
