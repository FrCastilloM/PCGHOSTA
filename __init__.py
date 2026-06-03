"""
GHOSS Library - Modular version
Libreria de codigos para el uso y analisis de datos de GHOSS
Francisco Castillo Menegotto

Última modificación: 5/11/25

"""

from .config import *
from .io_utils import *
from .math_functions import *
from .fitting import *
from .parameters import *

# Import plotting modules
from .plotting import base as plot_base
from .plotting import animations
from .plotting import profiles
from .plotting import colormaps

# Import analysis modules
from .analysis import vortex
from .analysis import currents

__version__ = '1.0.0'
__all__ = [
    'config',
    'io_utils',
    'math_functions',
    'fitting',
    'parameters',
    'plotting',
    'analysis'
]
