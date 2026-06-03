"""
Configuración global y paths del proyecto GHOSS
"""

import os

# Paths
outputs_path = "./out"
GHOST_path = '.'

# Versiones
dirac_ver = False
w10_ver = True

# DPI para gráficos
dpi = 300

# Configuración de matplotlib (solo si no es versión DIRAC)
if not dirac_ver:
    import matplotlib.pyplot as plt
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
