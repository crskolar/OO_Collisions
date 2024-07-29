# This script sees how a Maxwellian plasma is viewed differently for different magnetic field angles
# Compare approximate calculation to exact solution
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from ISRSpectraFunctions import *
from loadMCData import readDMP

import scipy.constants as const
u = const.physical_constants['unified atomic mass unit'][0]
kB = const.k
me = const.m_e
e = const.e
eps0 = const.epsilon_0
c = const.c

fileDir = 'C:/Users/Chirag/Documents/O+O/Monte-Carlo/'
# fileNameDMP = 'noE_v2              .DMP'
fileNameDMP = 'noE                 .DMP'
[fiSym, vperp, vpar, VVperp, VVpar, fi] = readDMP(fileDir + fileNameDMP)

# Set background and radar parameters
B = 1e-5
nu_ISR = 440e6
k_ISR = 2*math.pi*nu_ISR/c
Tn = 1000
nn = 1.8e14   # m^-3

# Assume a Maxwellian electron species. Set electron parameters
Te = 4000.
ne = 1.0e11
Oce = e*B/me
vthe = (2*kB*Te/me)**.5
rho_avge = vthe/Oce/2**.5
wpe = (ne*e**2/me/eps0)**.5
lambdaD = (eps0*kB*Te/(ne*e**2))**.5
alpha = 1/k_ISR/lambdaD

# Set ion parameters (used for exact calculation)
Ti = 1000
mi = 16*u
vthi = (2*kB*Ti/mi)**.5
ni = 1e11
Oci = e*B/mi
rho_avgi = vthi/Oci/2**.5
wpi = (ni*e**2/mi/eps0)**.5

# Calculate collisions frequencies
nuen = 8.9e-11*nn/1e6*(1+5.7e-4*Te)*Te**.5
nuei = 54.5*ni/1e6/Te**1.5
nuee = 54.5*ne/1e6/2**.5/Te**1.5 
nue = nuen + nuei + nuee
Tr = (Ti+Tn)/2
nuin = 3.67e-11*nn/1e6*Tr**.5*(1-0.064*np.log10(Tr))**2
nuii = 0.22*ni/1e6/Ti**1.5
nuie = me*nuei/mi
nui = nuin + nuii + nuie

# Make the array for omega based on 3 times ion acoustic speed. (assume gamma_e=gamma_i=5/3)
cs = (5/3*kB*(Ti+Te)/mi)**.5
omega_bounds = round(cs*k_ISR*3,-3)
omega_exact = np.linspace(-omega_bounds,omega_bounds,1000)

# Set the angles we will be looking at
angles = np.arange(0,90,10)

exactColor = 'k'
approxColor = 'C1'

exactLine = '-'
approxLine = '--'

# Set the angles we will be looking at
angles = np.arange(0,90,10)

maxwellColor = 'k'
toroidalColor = 'C1'

maxwellLine = '-'
toroidalLine = '-'


# # Iterate through the angles
for i in range(8,9):#0,len(angles)):
    
    # Tequiv = LOS_temps[i,1]
    print("Angle:", angles[i])
    # vthi = (2*kB*Tequiv/mi)**.5
    
    
    # Build the appropriate filename
    fileName = "E_0_theta_" + str(angles[i])
    
    # Check the convergence
    checkSumConvergence(fileDir + fileName, 1e-20)
    
    # Initialize figure
    font = {'size'   : 22}
    mpl.rc('font', **font)
    fig = plt.figure(1, figsize=(16,14))
    gs = gridspec.GridSpec(4,2,width_ratios=[1,1.2])
    gs.update(left=0.103 , right=.99, bottom=.06, top=.97, wspace=.2, hspace=.13)
    fig.patch.set_facecolor('white')

    ax2D = fig.add_subplot(gs[0:3,0])
    ax1D = fig.add_subplot(gs[3,0])

    ax = []
    for k in range(0,4):
        ax.append(fig.add_subplot(gs[k,1]))
        
    # Calculate kpar and kperp
    theta = np.deg2rad(angles[i])
    kpar = k_ISR*np.cos(theta)
    kperp = k_ISR*np.sin(theta)
    
#     # Get parametrization for 1D dist function for vpar and vperp
#     if i == 0:
#         vperpLine = np.zeros_like(vpar)
#         vparLine = vpar
#         v_1D = vpar
#     else:
#         vperpLine = np.linspace(-4*vthiperp,4*vthiperp,300)
#         vparLine = vperpLine*np.tan(math.pi/2-theta)
#         v_1D = (vperpLine**2+vparLine**2)**.5*np.sign(vperpLine)
#     f_1D = toroidal_norm(vperpLine, vparLine, vthiperp, vthipar, Dstar)
    
    im = ax2D.pcolormesh(VVperp, VVpar, fi, cmap='inferno', vmin=0, vmax=np.max(fi))
#     ax2D.plot(vperpLine, vparLine, 'k',linewidth=3)
#     ax2D.plot(vperpLine, vparLine, 'w',linewidth=2.8)
#     ax2D.set_xlim(-4*vthiperp, 4*vthiperp)
#     ax2D.set_ylim(-4*vthipar, 4*vthipar)
#     ax2D.set_aspect('equal')
#     ax2D.set_xlabel('$v_\perp$ (m/s)')
#     ax2D.set_ylabel('$v_\parallel$ (m/s)')    
#     ax2D.grid()
    cbar = fig.colorbar(im, orientation='horizontal', location='top',pad=0.04)
    cbar.set_label('$f_i$ (s$^3$m$^{-3}$)', labelpad=13)
    
    
    
    
#     f_1D_Max = toroidal_norm(vperpLine, vparLine, vthi, vthi, 0)
    
    
#     ax1D.plot(v_1D, f_1D_Max, color=maxwellColor, linestyle=maxwellLine, linewidth=3)
#     ax1D.plot(v_1D, f_1D, color=toroidalColor, linestyle=toroidalLine, linewidth=3)
    
#     ax1D.set_xlabel('$v_\phi$ (m/s)')
#     ax1D.set_ylabel('$f(v_\phi)$  (s$^3$m$^{-3}$)')
#     ax1D.grid()
#     ax1D.set_xlim(-6000, 6000)
    
    
#     # Calculate exact U, M, chi, and S for ions and electrons
#     U_e_exact = calcU_Maxwellian(omega_exact, kpar, kperp, vthe, 2000, rho_avge, Oce, nue)
#     M_e_exact = calcM_Maxwellian(omega_exact, kpar, kperp, vthe, 2000, rho_avge, Oce, nue, U_e_exact)
#     chi_e_exact = calcChi_Maxwellian(omega_exact, kpar, kperp, vthe, 2000, rho_avge, Oce, nue, alpha, U_e_exact, Te, Te)
    
#     U_i_exact = calcU_Maxwellian(omega_exact, kpar, kperp, vthi, 2000, rho_avgi, Oci, nui)
#     M_i_exact = calcM_Maxwellian(omega_exact, kpar, kperp, vthi, 2000, rho_avgi, Oci, nui, U_i_exact)
#     chi_i_exact = calcChi_Maxwellian(omega_exact, kpar, kperp, vthi, 2000, rho_avgi, Oci, nui, alpha, U_i_exact, Te, Ti)
    
#     S_exact = calcSpectra(M_i_exact, M_e_exact, chi_i_exact, chi_e_exact)
    
    # Load data
    [sum_U_i, sum_M_i, sum_chi_i, omega, kperp, kpar, nui, mi] = loadSumData(fileDir+fileName)
    
    # # Calculate U, M, chi, and the resulting spectrum
    U_e = calcU_Maxwellian(omega, kpar, kperp, vthe, 2000, rho_avge, Oce, nue)
    M_e = calcM_Maxwellian(omega, kpar, kperp, vthe, 2000, rho_avge, Oce, nue, U_e)
    chi_e = calcChi_Maxwellian(omega, kpar, kperp, vthe, 2000, rho_avge, Oce, nue, alpha, U_e, Te, Te)
    [U_i, M_i, chi_i] = calcFromSums(sum_U_i, sum_M_i, sum_chi_i, kpar, kperp, nui, wpi)
    S = calcSpectra(M_i, M_e, chi_i, chi_e)
    
    
#     ax[0].plot(omega_exact, np.real(M_i_exact), color=maxwellColor, linestyle=maxwellLine, linewidth=3,label='Maxwellian')
    ax[0].plot(omega, np.real(M_i), color=toroidalColor, linestyle=toroidalLine, linewidth=3,label='Toroidal')
    
#     ax[1].plot(omega_exact, np.real(chi_i_exact), color=maxwellColor, linestyle=maxwellLine, linewidth=3,label='Maxwellian')
    ax[1].plot(omega, np.real(chi_i), color=toroidalColor, linestyle=toroidalLine, linewidth=3,label='Toroidal')
    
#     ax[2].plot(omega_exact, np.imag(chi_i_exact), color=maxwellColor, linestyle=maxwellLine, linewidth=3,label='Maxwellian')
    ax[2].plot(omega, np.imag(chi_i), color=toroidalColor, linestyle=toroidalLine, linewidth=3,label='Toroidal')
    
#     ax[3].plot(omega_exact, S_exact, color=maxwellColor, linestyle=maxwellLine, linewidth=3,label='Maxwellian')
    ax[3].plot(omega, S, color=toroidalColor, linestyle=toroidalLine, linewidth=3,label='Toroidal')

#     ax[3].set_xlabel('$\omega$ (rad/s)')
#     ax[0].set_ylabel('$M_i$')
#     ax[1].set_ylabel('Re$(\chi_i)$')
#     ax[2].set_ylabel('Im$(\chi_i)$')
#     ax[3].set_ylabel('$S$')
#     ax[0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
#     ax[3].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    
#     ax[3].legend()
    
#     for k in range(0,4):
#         ax[k].grid()
#         ax[k].set_xticks([-40000,-20000,0,20000,40000])
#         ax[k].set_xlim(-5e4,5e4)
#         # ax[k].set_xlim(np.min(omega),np.max(omega))
#         if k != 3:
#             ax[k].set_xticklabels([])
            
#     ax1D.set_ylim(-.1e-10,1.8e-10)
#     ax[0].set_ylim(-.1e-4,2e-4)
#     ax[1].set_ylim(-200, 400)
#     ax[2].set_ylim(-200, 200)
#     ax[3].set_ylim(-.1e-4, 1.8e-4)
#     ax[3].ticklabel_format(style='sci',axis='x',scilimits=(0,0))
#     fig.savefig(fileDir + fileName + '_spectrum.png',format='png')
    
    
    
    
    
    
    