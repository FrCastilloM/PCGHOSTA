"""
Funciones para lectura y escritura de archivos
"""

import numpy as np
import re
import os
from PIL import Image


def load_namefiles(carpeta, nombre_base):
    """
    Carga y ordena los nombres de archivos de salida de GHOST
    
    Parameters:
    -----------
    carpeta : str
        Ruta de la carpeta con los archivos
    nombre_base : str
        Nombre base de los archivos (ej: 'rho', 'wy')
    
    Returns:
    --------
    list, int : Lista de nombres ordenados y número máximo de índice
    """
    archivos = os.listdir(carpeta)
    regex = re.compile(rf"{nombre_base}.(\d+)\.out")
    
    numeros = []
    filenames = []
    
    for archivo in archivos:
        match = regex.match(archivo)
        if match:
            filenames.append(archivo)
            numeros.append(int(match.group(1)))
    
    if not numeros:
        return None
    
    # Ordenar archivos
    pairs = zip(numeros, filenames)
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    
    return [f[1] for f in sorted_pairs], max(numeros)


def readbin(file, shape):
    """
    Lee archivos binarios de GHOST y extrae el plano central
    
    Parameters:
    -----------
    file : str
        Ruta del archivo binario
    shape : tuple
        Forma de la matriz (NX, NY, NZ)
    
    Returns:
    --------
    ndarray : Matriz 2D transpuesta del plano central
    """
    data = np.fromfile(file, dtype=np.float32).reshape(shape, order='F')
    NY = len(data[0, :, 0])
    data = data[:, NY//2, :].T
    return data


def density_extent(data):
    """
    Extiende la matriz por periodicidad
    
    Parameters:
    -----------
    data : ndarray
        Matriz de datos
    
    Returns:
    --------
    ndarray : Matriz extendida
    """
    data_extended = np.vstack([data, data[0, :]])
    data_extended = np.hstack([data_extended, data_extended[:, [0]]])
    return data_extended


def read_txt(path):
    """
    Lee un archivo de texto y devuelve las líneas
    
    Parameters:
    -----------
    path : str
        Ruta del archivo
    
    Returns:
    --------
    list : Lista de líneas del archivo
    """
    with open(path, 'r') as document:
        text = document.readlines()
    return text


def create_gif(path, name, duration=500):
    """
    Crea un GIF a partir de imágenes PNG numeradas
    
    Parameters:
    -----------
    path : str
        Ruta donde están las imágenes
    name : str
        Nombre base de las imágenes
    duration : int
        Duración de cada frame en ms
    """
    pattern = re.compile(rf"{re.escape(name)}\.\d+\.png")
    
    images = [img for img in os.listdir(path) if pattern.match(img)]
    images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    
    frames = [Image.open(os.path.join(path, img)) for img in images]
    
    if frames:
        frames[0].save(
            path + '/' + name + '.gif',
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
