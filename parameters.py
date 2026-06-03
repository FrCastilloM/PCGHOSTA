"""
Funciones para leer y modificar parámetros de simulación
"""

import re
from .config import GHOST_path
from .io_utils import read_txt


def grep(pattern, file_text):
    """
    Busca un patrón en el texto y devuelve el índice de línea
    
    Parameters:
    -----------
    pattern : str
        Patrón a buscar
    file_text : list
        Lista de líneas de texto
    
    Returns:
    --------
    int : Índice de la línea donde se encuentra el patrón
    """
    txt = file_text.copy()
    txt.reverse()
    for j in range(len(txt)):
        if re.search(pattern, txt[j]):
            return len(txt) - 1 - j


def findall(textline):
    """
    Extrae todos los números de una línea de texto
    
    Parameters:
    -----------
    textline : str
        Línea de texto
    
    Returns:
    --------
    list : Lista de números flotantes
    """
    x = re.findall(r'[-+]?\d*\.?\d+(?:e[+-]?\d+)?', textline)
    X = []
    for k in range(len(x)):
        X.append(float(x[k]))
    return X


def get_Resolution():
    """
    Obtiene la resolución espacial de la simulación
    
    Returns:
    --------
    tuple : (NX, NY, NZ)
    """
    makefile = read_txt(GHOST_path + '/src/Makefile.in')
    R = grep('Spatial resolution', makefile)
    NX = int(findall(makefile[R + 1])[0])
    NY = int(findall(makefile[R + 2])[0])
    NZ = int(findall(makefile[R + 3])[0])
    return NX, NY, NZ


def get_timeparam():
    """
    Obtiene los parámetros temporales de la simulación
    
    Returns:
    --------
    tuple : (dt, tstep)
    """
    paraminp = read_txt(GHOST_path + '/bin/parameter.inp')
    T = grep('Parameters for time int', paraminp)
    dt = findall(paraminp[T + 2])[0]
    tstep = findall(paraminp[T + 4])[0]
    return dt, tstep


def get_param(param):
    """
    Obtiene el valor de un parámetro específico
    
    Parameters:
    -----------
    param : str
        Nombre del parámetro
    
    Returns:
    --------
    float : Valor del parámetro
    """
    pattern = param + r'\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
    
    with open(GHOST_path + '/bin/parameter.inp', 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                value = float(match.group(1))
                break
    
    return value


def get_energy(data_file):
    """
    Lee datos de energía desde un archivo
    
    Parameters:
    -----------
    data_file : str
        Nombre del archivo de energía
    
    Returns:
    --------
    list : Lista de arrays con los datos de energía
    """
    data = open(r'./bin/' + data_file, 'r')
    data_text = data.readlines()
    data.close()
    
    n = len(re.findall(r'[+-]?\d+\.\d+E[+-]?\d+', data_text[0]))
    X = [[] for _ in range(n)]
    
    for j in range(len(data_text)):
        datos = re.findall(r'[+-]?\d+\.\d+E[+-]?\d+', data_text[j])
        if len(datos) != n:
            break
        for k in range(len(datos)):
            X[k].append(float(datos[k]))
    
    return X


def chg_param(param, new_value):
    """
    Cambia el valor de un parámetro en parameter.inp
    
    Parameters:
    -----------
    param : str
        Nombre del parámetro
    new_value : float
        Nuevo valor
    """
    pattern = r'(' + param + r'\s*=\s*)([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
    
    parameterinp = open('./bin/parameter.inp', 'r')
    txt = parameterinp.readlines()
    parameterinp.close()
    
    for i in range(len(txt)):
        match = re.search(pattern, txt[i])
        if match:
            txt[i] = re.sub(pattern, r'\g<1>{}'.format(new_value), txt[i])
            break
    
    parameterinp = open('./bin/parameter.inp', 'w')
    parameterinp.writelines(txt)
    parameterinp.close()


def chg_kappa(k):
    """Cambia el parámetro kappa"""
    chg_param('vparam2', k)


def chg_xi(xi):
    """Cambia el parámetro xi (correlation length)"""
    chg_param('lambda', xi)


def chg_h0(h0):
    """Cambia la amplitud del campo magnético externo"""
    chg_param('zparam3', h0)
