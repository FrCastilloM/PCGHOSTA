"""
Funciones para tracking y análisis de vórtices
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from scipy.interpolate import interp2d

from ..config import outputs_path, dpi
from ..parameters import get_param, get_Resolution, get_timeparam
from ..io_utils import load_namefiles, readbin
from ..fitting import linear_adjust, lineal, vortex_adjust
from ..plotting.animations import Plots_densidad


def ellipse_res(params, xz, y, xi, hatlam1, A):
    """
    Función residual para ajuste de elipse (vórtice nemático)
    """
    mux, muz, nemx, nemz, B = params
    x, z = xz
    lx = 1 + hatlam1 * nemx
    lz = 1 - hatlam1 * nemx
    r2 = ((x - mux)**2 / lx**2 + (z - muz)**2 / lz**2)
    return A * B * np.tanh(np.sqrt(r2) / nemz)**2 - y


def ellipse_(params, x, z, xi, hatlam1, A):
    """
    Función de elipse para vórtice nemático
    """
    mux, muz, nemx, nemz, B = params
    lx = 1 + hatlam1 * nemx
    lz = 1 - hatlam1 * nemx
    r2 = ((x - mux)**2 / lx**2 + (z - muz)**2 / lz**2)
    return A * B * np.tanh(np.sqrt(r2) / nemz)**2


def track_elipse(fs = 18, cuttof = 0, plot3D = False, linear = False, gif = False, zoom_factor = 1, duration = 1000):
    
    """
    Tracking de vórtice nemático con ajuste elíptico
    
    Parameters:
    -----------
    fs : int
        Tamaño de fuente
    cuttof : int
        Número de frames a omitir al final
    plot3D : bool
        Si True, genera gráficos 3D
    linear : bool
        Si True, hace ajuste lineal de la trayectoria
    
    --------
    """
    
    path = outputs_path
    
    NX, NY, NZ = get_Resolution()
    dt, tstep = get_timeparam()
    shape = (NX, NY, NZ)
    
    filenames, _ = load_namefiles(path, 'rho')
    
    times = []
    x_v = []
    # errx = []
    z_v = []
    # errz = []
    
    b = get_param('pparam1')
    x0 = int(NX * (np.pi-.8*b/2)/(2*np.pi))
    x1 = NX - x0

    hatlam2 = get_param('vparam3')
    Gamma4 = get_param('vparam4')
    psivac = np.sqrt((1-hatlam2)/(1-hatlam2**2/Gamma4))
    
    xi = get_param('lambda')
    hatlam1 = get_param('vparam8')
    
    cs = get_param('cspeed')
    
    hc2 = np.sqrt(2)*cs/xi

    for i in range(0,len(filenames)-cuttof):
        
        t = dt*tstep*i
        times.append(t) 
        tnorm = round(t*hc2,1)
        
        data = readbin(path + '/' + filenames[i], shape)
        if np.any(np.isnan(data)) == True:
            break
        
        s = data[x0:x1, x0:x1]
        x = np.arange(len(s[0,:]))
        z = np.arange(len(s[:,0]))
        
        max_position = np.unravel_index(np.argmin(s), s.shape)
        
        initial_params = [max_position[1], max_position[0], 1, xi * NX/(2*np.pi), 1]
        
        X, Z = np.meshgrid(x, z)
        
        x_data = X.ravel()
        z_data = Z.ravel()
        s_data = s.ravel()
        
        params, covariance = leastsq(ellipse_res, initial_params, 
                                      args=((x_data, z_data), s_data , 
                                            xi * NX/(2*np.pi), hatlam1, psivac**2))
        
        # errors = np.sqrt(np.diag(covariance))
        
        elipse = ellipse_(params, X, Z, xi * NX/(2*np.pi), hatlam1, psivac**2)
        
        
        if plot3D == True:
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            N_new = 1000
            f_interp = interp2d(X[0, :], Z[:, 0], elipse, kind='cubic')
            X_new, Z_new = np.meshgrid(np.linspace(params[0] - 5 * xi * NX/(2*np.pi), 
                                                   params[0] + 5 * xi * NX/(2*np.pi), N_new),
                                       np.linspace(params[1] - 5 * xi * NX/(2*np.pi), 
                                                   params[1] + 5 * xi * NX/(2*np.pi), N_new))
            elipse_new = f_interp(X_new[0, :], Z_new[:, 0])
    
            ax.contour3D((X-params[0])/xi*2*np.pi/NX, (Z-params[0])/xi*2*np.pi/NX, s, 
                         levels = 10, colors = 'k', zorder = 50, linewidths = 1.5)
            ax.plot_surface((X_new-params[0])/xi*2*np.pi/NX, (Z_new-params[0])/xi*2*np.pi/NX, 
                            elipse_new, cmap='viridis', alpha = .6, shade = True, zorder = 10)
            
    
            ax.set_title(f'$t/t_R = {tnorm}$', fontsize = fs)
            ax.set_xlabel(r'$(x-\mu_x)/\xi$', fontsize = fs - 2)
            ax.set_ylabel(r'$(z-\mu_z)/\xi$', fontsize = fs - 2)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_xticks([-15, 0, 15])
            ax.set_yticks([-15, 0, 15])
       
            # Crear elementos de leyenda personalizados
            legend_elements = [Line2D([0], [0], color='k', lw=2, label='Simulación')]
            
            # Añadir la leyenda
            ax.legend(handles=legend_elements, fontsize=14)
            
            a = ax.set_zlabel(r'$|\tilde\psi|^2$', fontsize = fs)
            ax.set_zticks([0, 1])
            ax.set_zlim(1, 0)
            a.set_rotation(90)
            ax.view_init(elev = 12, azim = 40+90)
            
            plt.tight_layout()
            plt.savefig(path + f'/track_{i}.png', dpi = 300)
            plt.close()
        
        ux = (x0 + params[0]) * 2*np.pi/NX
        # uxerr = errors[0] * 2*np.pi/NX
        x_v.append(ux)
        # errx.append(uxerr)
        uz = (x0 + params[1]) * 2*np.pi/NZ
        # uzerr = errors[1] * 2*np.pi/NZ
        z_v.append(uz)
        # errz.append(uzerr)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # fig.suptitle('Perfiles vórtice', fontsize  = 18)
        ax1.set_title('Perfil en x', fontsize = fs, fontweight = 'bold')
        ax2.set_title('Perfil en z', fontsize = fs, fontweight = 'bold')
        
        ax1.axvline(x = 1, lw = 1.2, color = 'grey', alpha = .7, linestyle = '--')
        ax1.axvline(x = -1, lw = 1.2, color = 'grey', alpha = .7, linestyle = '--')
        ax2.axvline(x = 1, lw = 1.2, color = 'grey', alpha = .7, linestyle = '--')
        ax2.axvline(x = -1, lw = 1.2, color = 'grey', alpha = .7, linestyle = '--')
        
        # comparacion con formula cruda tinkham (perfil)
        # ejex = np.linspace(-5,5,1000)*xi
        # ax1.plot(ejex/xi, perfil_vortice_tinkham(ejex*1/np.sqrt(2), xi), color = 'orange', label = 'Aprox.\nAnalítica')
        
        
        n = 4
        cmap = plt.get_cmap('viridis', n)
        for j in range(3):
            sx = s[max_position[0]+n*j,:]
            sz = s[:,max_position[1]+n*j]
            a = 4
            
            ax1.scatter((x-params[0])/xi*2*np.pi/NX, sx, marker = 's', s = 10, label = 'Simulación', color = 'k', alpha = 1-j/a)
            ax2.scatter((-(z-params[1]))/xi*2*np.pi/NZ, sz, marker = 's', s = 10, label = 'Simulación', color = 'k', alpha = 1-j/a)
            
            dz = round((-(max_position[0]+n*j-params[1]))/xi*2*np.pi/NZ,2)
            ax1.text((np.argmin(sx)-params[0])/xi*2*np.pi/NX + 2, np.min(sx), 
                     fr'$(z-\mu_z)/\xi = {dz}$', fontsize = 12,
                      bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), zorder = 20)
            
            dx = round((max_position[1]+n*j-params[0])/xi*2*np.pi/NX,2)
            ax2.text((-(np.argmin(sz)-params[1]))/xi*2*np.pi/NZ + 2, np.min(sz), 
                     fr'$(x-\mu_x)/\xi = {dx}$', fontsize = 12,
                      bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), zorder = 20)
            
            x_continuo = np.linspace(np.min(x), np.max(x), 1000)
            z_continuo = np.linspace(np.min(z), np.max(z), 1000)
            
            elipsex_continua = ellipse_(params, x_continuo, max_position[0]+n*j, xi * NX/(2*np.pi), hatlam1, psivac**2)
            elipsez_continua = ellipse_(params, max_position[1]+n*j, z_continuo, xi * NX/(2*np.pi), hatlam1, psivac**2)
            
            ax1.plot((x_continuo-params[0])/xi*2*np.pi/NX, elipsex_continua, color = cmap(j), linestyle = '-',
                     alpha = 1-j/a, zorder = 10, label = 'Ajuste')
            ax2.plot((-(z_continuo-params[1]))/xi*2*np.pi/NZ, elipsez_continua, color = cmap(j), linestyle = '-',
                     alpha = 1-j/a, zorder = 10, label = 'Ajuste')

            if j == 0:
                ax1.legend(fontsize = 14)
                ax2.legend(fontsize = 14)
                
            ax1.annotate('', xy=((np.argmin(sx)-params[0])/xi*2*np.pi/NX + 2, np.min(sx)),
                         xytext=(0, np.min(sx)), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize = 14)
            ax2.annotate('', xy=((-(np.argmin(sz)-params[1]))/xi*2*np.pi/NZ + 2, np.min(sz)),
                         xytext=(0, np.min(sz)), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize = 14)
            
        ax1.set_xlim(-6,6)
        ax2.set_xlim(-6,6)
        ax1.set_xlabel(r'$(x-\mu_x)/\xi$', fontsize = fs)
        ax2.set_xlabel(r'$(z-\mu_z)/\xi$', fontsize = fs)
        ax1.set_ylabel(r'$|\tilde\psi|^2$', fontsize = fs)
        ax2.set_ylabel(r'$|\tilde\psi|^2$', fontsize = fs)
        plt.tight_layout()
        plt.savefig(path + f'/ajuste_{i}.png', dpi = 300)
        plt.close()
        
        
        if i < (len(filenames)-cuttof-1):
        
            Plots_densidad('rho', r'$|\tilde{\psi}|^2$', gif = False, step = len(filenames)+1, duration = duration, 
                              fs = fs, maximo = 0, dots = [[ux, uz], [np.pi, np.pi]], initial = i, zoom_factor = zoom_factor)
        
        if gif == True and i == (len(filenames)-cuttof-1):
            
            Plots_densidad('rho', r'$|\tilde{\psi}|^2$', gif = True, step = len(filenames)+1, duration = duration, 
                              fs = fs, maximo = 0, dots = [[ux, uz], [np.pi, np.pi]], initial = i, zoom_factor = zoom_factor)


    np.savetxt('track.txt', np.column_stack([times, x_v, z_v])
                ,delimiter=',', header = 'Tiempos, x, z')
    
    
    times = np.array(times)
    x_v = np.array(x_v)
    z_v = np.array(z_v)
    
    # errx = np.array(errx)
    # errz = np.array(errz)
    
    if linear == True:
        
        # para el ajuste elimino el dato inicial (para asegurar regimen lineal)
        [mx, bx], [errmx, errbx] = linear_adjust(times[1:], x_v[1:], erry = None)
        [mz, bz], [errmz, errbz] = linear_adjust(times[1:], z_v[1:], erry = None)
        
        fig, ax = plt.subplots()
        ax.set_title('Movimiento vortice',fontsize=fs, fontweight = 'bold')
        # ax.errorbar(times/dt, (x_v-np.pi)/xi, yerr = errx/xi, fmt = 's', capsize=5, capthick=2, ecolor='k', 
        #             color = 'k', label = 'Coordenada x')
        ax.scatter(times/dt, (x_v-np.pi)/xi, marker = 's', color = 'k', label = 'Coordenada x')
        ax.plot(times/dt, (lineal(times, mx, bx)-np.pi)/xi, '-', color = 'grey', alpha = .7, lw = 1.1,
                label = 'Ajuste lineal')
                # label = r'$m={m}$, $b={b}$'.format(m = round(mx,4), b = round(bx,4)))
        
        shift = (x_v[0] - z_v[0])/xi
        ax.scatter(times/dt, (z_v-np.pi)/xi + shift, marker = 'o', color = 'red', label = 'Coordenada z')
        ax.plot(times/dt, (lineal(times, mz, bz)-np.pi)/xi + shift, '-', color = 'grey', alpha = .7, lw = 1.1) 
                # label = r'$m={m}$, $b={b}$'.format(m = round(mz,4), b = round(bz,4)))
                
        
        ax.set_xlabel(r'$t/\Delta t$', fontsize=fs)
        ax.set_ylabel(r'$\Delta x/\xi$',fontsize=fs)
        ax.set_xlim(-300,np.max(times/dt)+300)
        ax.legend(fontsize = 14, loc = 'lower right')
        plt.tight_layout()
    
        plt.savefig(path + '/track.png', dpi = 300)
        plt.close()
    
        return hatlam1, [mx, mz], [errmx, errmz]
    
    if linear == False:
        return 