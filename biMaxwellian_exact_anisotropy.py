import numpy as np
import multiprocessing as mp
import math
import time
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import os
from functools import partial
from MC2ISR_functions import *

import scipy.constants as const
u = const.physical_constants['unified atomic mass unit'][0]
kB = const.k
me = const.m_e
e = const.e
eps0 = const.epsilon_0
c = const.c

# Set background parameters based on a reasonable F region plasma 
B = 1e-5
nu_ISR = 450e6

Tn = 1000
nn = 1e14   # m^-3

# Leave Tipar the same and allow Tiperp to change
Tipar = 1000
mi = 16*u
ni = 1e11

Te = 1000.
ne = 1e11
nuen = 8.9e-11*nn/1e6*(1+5.7e-4*Te)*Te**.5
nuei = 54.5*ni/1e6/Te**1.5
nuee = 54.5*ne/1e6/2**.5/Te**1.5 
nue = nuen + nuei + nuee

vthe = (2*kB*Te/me)**.5
Oci = e*B/mi
Oce = e*B/me
rho_avge = vthe/Oce/2**.5
wpi = (ni*e**2/mi/eps0)**.5
wpe = (ne*e**2/me/eps0)**.5
lambdaD = (eps0*kB*Te/(ne*e**2))**.5



# Set parameters for calculation of modified ion distribution and ion suseptability
nmax = 2000
mesh_n = 500


anisotropy_ratio = np.linspace(0.5,2.0,13)

font = {'size'   : 26}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(10,12))
gs = gridspec.GridSpec(5,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
gs.update(left=0.25, right=.99, bottom=.08, top=.94, wspace=0.015, hspace=.015)
fig.patch.set_facecolor('white')
ax = []
for i in range(0,5):    
    ax.append(plt.subplot(gs[i]))

colors = []
for i in range(0,len(anisotropy_ratio)):
    colors.append(pl.cm.inferno( i/(len(anisotropy_ratio))) )

# Set ISR parameters
k_ISR = 2*math.pi*nu_ISR/c
angles = np.array([0,20,40,60, 80])
theta = np.deg2rad(angles)
# Calculate alpha
alpha = 1/k_ISR/lambdaD

omega_max = 0.0
omega = np.linspace(-40000,40000,401)
for k in range(len(theta)):
    kpar = k_ISR*np.cos(theta[k])
    kperp = k_ISR*np.sin(theta[k])
    for i in range(len(anisotropy_ratio)):
        Tiperp = Tipar*anisotropy_ratio[i]
        print("theta:",angles[k], "      Tiperp:", Tiperp)
        Ti = (Tipar + 2*Tiperp)/3
        
        Tr = (Ti+Tn)/2
        nuin = 3.67e-11*nn/1e6*Tr**.5*(1-0.064*np.log10(Tr))**2
        nuii = 0.22*ni/1e6/Ti**1.5
        nuie = me*nuei/mi
        nui = nuin + nuii + nuie
        
        # Calculate thermal velocities, gyroradii, gyrofrequencies, plasma frequency, and electron Debye length
        vthipar = (2*kB*Tipar/mi)**.5
        vthiperp = (2*kB*Tiperp/mi)**.5
        rho_avgi = vthiperp/Oci/2**.5
        
        # # Make the array for omega based on 3 times ion acoustic speed. (assume gamma_e=gamma_i=5/3)
        # cs = (5/3*kB*(Ti+Te)/mi)**.5
        # omega_bounds = round(cs*k_ISR*3,-3)
        
        
        # Calculate exact values for Maxwellian electrons
        U_e_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue)
        M_e_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue, U_e_exact)
        chi_e_exact = calcChis_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue, alpha, U_e_exact, Te, Te)
        
        # Calculate bi-Maxwellian values for ions
        U_i_bimax = calcU_biMax(omega, kpar, kperp, vthipar, nmax, rho_avgi, Oci, nui)
        M_i_bimax = calcM_biMax(omega, kpar, kperp, vthipar, nmax, rho_avgi, Oci, nui, U_i_bimax)
        chi_i_bimax = calcchi_biMax(omega, kpar, kperp, vthipar, vthiperp, nmax, rho_avgi, Oci, nui, U_i_bimax, alpha, Te, Tipar)
        S_bimax = calcSpectra(M_i_bimax, M_e_exact, chi_i_bimax, chi_e_exact)
        
        ax[k].plot(omega, S_bimax, linewidth=2,color=colors[i])
        
        # omega_max = np.max([omega_max, np.max(omega)])
        # ax[1].plot(omega, np.real(M_i_bimax), linewidth=2,color=colors[i])
        # ax[2].plot(omega, np.imag(M_i_bimax), linewidth=2,color=colors[i])

for k in range(len(theta)):
    ax[k].grid()
    ax[k].set_xlim(np.min(omega), np.max(omega))
    ax[k].set_ylabel('$\\theta=%d$' % (angles[k]))
    
    
ax[4].set_xlabel('$\\omega$')

 
# ax[0].plot(omega,np.real(U_i_max),'k',linewidth=3)
# ax[0].plot(omega,np.real(U_i_bimax),'--',linewidth=3,color='C1')
# ax[0].set_ylabel('Re$(U_i)$')

# ax[1].plot(omega,np.imag(U_i_max),'k',linewidth=3)
# ax[1].plot(omega,np.imag(U_i_bimax),'--',linewidth=3,color='C1')
# ax[1].set_ylabel('Im$(U_i)$')

# ax[2].plot(omega,np.real(M_i_max),'k',linewidth=3)
# ax[2].plot(omega,np.real(M_i_bimax),'--',linewidth=3,color='C1')
# ax[2].set_ylabel('Re$(M_i)$')

# ax[3].plot(omega,np.real(chi_i_max),'k',linewidth=3)
# ax[3].plot(omega,np.real(chi_i_bimax),'--',linewidth=3,color='C1')
# ax[3].set_ylabel('Re$(\chi_i)$')

# ax[4].plot(omega,np.imag(chi_i_max),'k',linewidth=3)
# ax[4].plot(omega,np.imag(chi_i_bimax),'--',linewidth=3,color='C1')
# ax[4].set_ylabel('Im$(\chi_i)$')

# ax[5].plot(omega,S_max,'k',linewidth=3)
# ax[5].plot(omega,S_bimax,'--',linewidth=3,color='C1')
# ax[5].set_ylabel('$S$')

# ax[5].set_xlabel('$\omega$')
# for i in range(0,6):    
#     ax[i].grid()
#     ax[i].set_xticks([-20000,0,20000])
#     ax[i].set_xticklabels([])
#     ax[i].set_xlim(np.min(omega),np.max(omega))

# ax[5].set_xticklabels([-20000,0,20000])
