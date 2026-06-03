"""
Funciones para graficar perfiles
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from ..config import outputs_path, dpi
from ..parameters import get_param, get_Resolution, get_timeparam
from ..io_utils import load_namefiles, readbin, create_gif
from ..math_functions import spline


def plot_perfiles_corriente2(gif=False, step=1, initial=0, duration=500, fs=20, factor=1, x_v=None):
    """
    Grafica perfiles de corriente con análisis detallado
    
    Parameters:
    -----------
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
    factor : float
        Factor de escala
    x_v : list
        Posiciones del vórtice
    """
    path = outputs_path
    
    NX, NY, NZ = get_Resolution()
    dt, tstep = get_timeparam()
    shape = (NX, NY, NZ)
    
    xi = get_param('lambda')
    kappa = get_param('vparam2')
    cs = get_param('cspeed')
    lambda1 = get_param('vparam8')
    
    london = xi * kappa
    hc2 = np.sqrt(2) * cs / xi
    jc = hc2 / london
    
    outs = ['totcurrx', 'totcurrz', 'scurrx', 'scurrz', 'rho', 'nem', 'wy']
    factor_ = [1/jc, 1/jc, 1/jc, 1/jc, 1, 1, -1/hc2]
    
    filenames = []
    for i in range(len(outs)):
        A, _ = load_namefiles(path, outs[i])
        filenames.append(A)
        
    if x_v == None:
        x_v = np.ones(len(filenames[0])) * np.pi
    
    # Cargar Jext
    Jext_x = -readbin(path + '/jextx.init.out', shape) * 1/jc
    Jext_z = -readbin(path + '/jextz.init.out', shape) * 1/jc
    
    for i in range(initial, len(filenames[0]), step):
        t = round(dt * tstep * i * hc2, 2)
        
        matrizes = []
        for j in range(len(filenames)):
            matrizes.append(readbin(path + '/' + filenames[j][i], shape) * factor_[j])
        
        if np.any(np.isnan(matrizes)):
            break
        
        name = path + '/perfiles_corriente.{}.png'.format(i + 1)
        title = f'$t/t_R={t}$'
        
        fig, ax = plt.subplots(figsize=(6.5, 5))
        
        # Corrección JS NEM
        Xi_x = 1 + lambda1 * matrizes[5]
        Xi_z = 1 - lambda1 * matrizes[5]
        
        matrizes[2] = matrizes[2] * Xi_x
        matrizes[3] = matrizes[3] * Xi_z
        
        # Perfiles
        ax.set_xlabel(r'$(x-\mu_x)/\lambda_L$', fontsize=fs)
        ax3 = ax.twinx()
        ax3.set_ylabel(r'$J_y(x,\pi)/J_c$', fontsize=fs)
        ax.set_ylabel(r'$B_z(x,\pi)/H_{c2}$', fontsize=fs)
        ax.tick_params(axis='both', direction='inout', top=True, right=False)
        
        # Colores
        c_super = (255/255, 151/255, 0)
        c_normal = (0, 0, 205/255)
        c_ext = (0, 120/255, 120/255)
        c_B = 'k'
        
        x = (np.arange(0, NX) - x_v[i] * NX / (2 * np.pi)) / (2 * xi * NX / (2 * np.pi))
        
        sm = 10
        ax3.scatter(x, matrizes[3][NZ//2, :], marker='o', s=sm, color=c_super, zorder=10, label=r'$J_s$')
        ax3.scatter(x, (matrizes[1] - matrizes[3] + Jext_z)[NZ//2, :],
                   marker='o', s=sm, color=c_normal, zorder=8, label=r'$J_n$')
        ax.scatter(x, matrizes[6][NZ//2, :],
                  marker='o', s=sm, color=c_B, zorder=8, label=r'$B$')
        
        # Splines
        x_spline, y_spline = spline(x, matrizes[3][NZ//2, :], s=0)  # SUPER
        ax3.plot(x_spline, y_spline, color=c_super, linestyle='-', alpha=1)
        x_spline, y_spline = spline(x, (matrizes[1] - matrizes[3] + Jext_z)[NZ//2, :], s=0)  # NORMAL
        ax3.plot(x_spline, y_spline, color=c_normal, linestyle='-', alpha=1)
        x_spline, y_spline = spline(x, matrizes[6][NZ//2, :], s=0)  # CAMPO B
        ax.plot(x_spline, y_spline, color=c_B, linestyle='-', alpha=1)
        
        ax.set_xlim(-6, 6)
        ax3.set_ylim(-0.15, 0.15)
        ax.set_ylim(-0.25, 0.25)
        ax.axhline(y=0, lw=1, color='grey', alpha=0.6, zorder=30)
        ax.axvspan(-1, 1, color='cyan', alpha=0.3)
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks([-1, 1])
        ax2.set_xticklabels([r'-2$\xi$', r'2$\xi$'])
        
        ax3.plot(x, Jext_z[NZ//2, :], linestyle='--', color=c_ext, lw=1.5, label=r'$J_{ext}$')
        ax.legend(fontsize=fs - 4, loc='upper left').set_zorder(50)
        ax3.legend(fontsize=fs - 4, loc='upper right').set_zorder(50)
        
        plt.tight_layout()
        plt.savefig(name, dpi=dpi)
        plt.close()
    
    if gif:
        create_gif(path, 'perfiles_corriente', duration=duration)


def plot_perfiles_dens(gif=False, step=1, initial=0, duration=500, fs=20, x_v=None, save_data=False):
    """
    Grafica perfiles de densidad
    
    Parameters:
    -----------
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
    x_v : list
        Posiciones del vórtice
    """
    path = outputs_path
    
    NX, NY, NZ = get_Resolution()
    dt, tstep = get_timeparam()
    shape = (NX, NY, NZ)
    
    xi = get_param('lambda')
    cs = get_param('cspeed')
    hc2 = np.sqrt(2) * cs / xi
    
    outs = ['rho', 'wy', 'nem']
    norm = [1, 1/hc2, 1]
    
    filenames = []
    for i in range(len(outs)):
        A, _ = load_namefiles(path, outs[i])
        filenames.append(A)
    
    if x_v == None:
        x_v = np.ones(len(filenames[0])) * np.pi
    
    for i in range(initial, len(filenames[0]), step):
        t = round(dt * tstep * i * hc2, 2)
        
        matrizes = []
        for j in range(len(filenames)):
            matrizes.append(readbin(path + '/' + filenames[j][i], shape) * norm[j])
        
        if np.any(np.isnan(matrizes)):
            break
        
        name = path + '/perfiles_dens.{}.png'.format(i + 1)
        title = f'$t/t_R={t}$'
        
        fig, (ax, inset) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})
        ax.set_title(title, fontweight='bold', fontsize=fs)
        
        # Perfiles
        ax.set_xlabel('x/L', fontsize=fs)
        ax.set_ylabel(r'Perfiles (z=$\pi$)', fontsize=fs)
        ax.set_xticks([0, NX//2, NX])
        ax.set_xticklabels([r'0', r'$\pi$', r'$2\pi$'])
        
        # Plots
        ax.plot(matrizes[0][NZ//2, :], color='k', label=r'$|\tilde\psi|^2$', zorder=20, lw=1.4)
        ax.plot(matrizes[1][NZ//2, :], color='b', label=r'B$_y$/H$_{c2}$', zorder=10, lw=1.6)
        ax.plot(matrizes[2][NZ//2, :], color='r', label=r'$\tilde\eta$', zorder=8, lw=1.5)
        
        ax.legend(fontsize=fs - 4, loc='upper left').set_zorder(50)
        
        x = (np.arange(0, NX) - x_v[i] * NX / (2 * np.pi)) / (2 * xi * NX / (2 * np.pi))
        inset.tick_params(axis='both', direction='inout', top=False, right=True)
        inset.set_xlabel(r'(x-$\mu_x$)/$\lambda_L$', fontsize=fs)
        inset.set_ylabel(r'Perfiles (z=$\pi$)', fontsize=fs)
        
        sm = 15
        inset.scatter(x, matrizes[0][NZ//2, :], marker='o', s=sm, color='k', zorder=20)
        inset.scatter(x, matrizes[1][NZ//2, :], marker='o', s=sm, color='b', zorder=10)
        inset.scatter(x, matrizes[2][NZ//2, :], marker='o', s=sm, color='r', zorder=8)
        
        x_spline, y_spline = spline(x, matrizes[0][NZ//2, :], s=0)
        inset.plot(x_spline, y_spline, color='k', linestyle='--', alpha=1)
        x_spline, y_spline = spline(x, matrizes[1][NZ//2, :], s=0)
        inset.plot(x_spline, y_spline, color='b', linestyle='--', alpha=1)
        x_spline, y_spline = spline(x, matrizes[2][NZ//2, :], s=0)
        inset.plot(x_spline, y_spline, color='r', linestyle='--', alpha=1)
        
        inset.set_xlim(-3, 3)
        inset.axhline(y=0, lw=1, color='grey', alpha=0.6, zorder=30)
        inset.axhline(y=1, lw=1, color='grey', alpha=0.6, zorder=30)
        inset.axvspan(-1, 1, color='cyan', alpha=0.3)
        
        ax2 = inset.secondary_xaxis('top')
        ax2.set_xticks([-1, 1])
        ax2.set_xticklabels([r'-2$\xi$', r'2$\xi$'])
        
        plt.tight_layout()
        plt.savefig(name, dpi=dpi)
        plt.close()
        
        if save_data == True:
            np.savetxt(f'profile_{i+1}.txt', 
                       np.column_stack([x, matrizes[0][NZ//2, :], matrizes[1][NZ//2, :], matrizes[2][NZ//2, :]])
                        ,delimiter=',', header = 'Delta x/lambda_L, |psi|^2, B/Hc2, eta')
    
    if gif:
        create_gif(path, 'perfiles_dens', duration=duration)
