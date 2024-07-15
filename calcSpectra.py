import numpy as np
import multiprocessing as mp
import math
import time
import matplotlib.pyplot as plt
import os
from functools import partial
from MC2ISR_functions import *

fileDir = 'C:/Users/Chirag/Documents/O+O/data/maxwellian_all/'
fileName = 'par2_perp1_theta10'

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

Tipar = 1000.#1225.55
Tiperp = 1000.#1225.55
Ti = (Tipar+2*Tiperp)/3
mi = 16*u
ni = 1e11

Te = 1000.
ne = 1e11
nuen = 8.9e-11*nn/1e6*(1+5.7e-4*Te)*Te**.5
nuei = 54.5*ni/1e6/Te**1.5
nuee = 54.5*ne/1e6/2**.5/Te**1.5 
nue = nuen + nuei + nuee

Tr = (Ti+Tn)/2
nuin = 3.67e-11*nn/1e6*Tr**.5*(1-0.064*np.log10(Tr))**2
nuii = 0.22*ni/1e6/Ti**1.5
nuie = me*nuei/mi
nui = nuin + nuii + nuie
# nui = 0.00001


# Set ISR parameters
k_ISR = 2*math.pi*nu_ISR/c
theta = np.deg2rad(0)
kpar = k_ISR*np.cos(theta)
kperp = k_ISR*np.sin(theta)

# Calculate thermal velocities, gyroradii, gyrofrequencies, plasma frequency, and electron Debye length
vthiperp = (2*kB*Tiperp/mi)**.5
vthipar = (2*kB*Tipar/mi)**.5
vthe = (2*kB*Te/me)**.5
Oci = e*B/mi
Oce = e*B/me
rho_avgi = vthiperp/Oci/2**.5
rho_avge = vthe/Oce/2**.5
wpi = (ni*e**2/mi/eps0)**.5
wpe = (ne*e**2/me/eps0)**.5
lambdaD = (eps0*kB*Te/(ne*e**2))**.5

# Calculate alpha
alpha = 1/k_ISR/lambdaD

# Set parameters for calculation of modified ion distribution and ion suseptability
nmax = 2000
mesh_n = 500

# Make the array for omega based on 3 times ion acoustic speed. (assume gamma_e=gamma_i=5/3)
cs = (5/3*kB*(Ti+Te)/mi)**.5
omega_bounds = round(cs*k_ISR*3,-3)
omega = np.linspace(-omega_bounds,omega_bounds,201)

# load data
sum_U_i = np.loadtxt(fileDir + fileName + '_sum_U.txt', dtype=np.complex_)
sum_M_i = np.loadtxt(fileDir + fileName + '_sum_M.txt', dtype=np.complex_)
sum_chi_i = np.loadtxt(fileDir + fileName + '_sum_chi.txt', dtype=np.complex_)

# sum_U_i_1 = np.loadtxt(fileDir + fileName + '_sum_U.txt', dtype=np.complex_)
# sum_M_i_1 = np.loadtxt(fileDir + fileName + '_sum_M.txt', dtype=np.complex_)
# sum_chi_i_1 = np.loadtxt(fileDir + fileName + '_sum_chi.txt', dtype=np.complex_)

# sum_U_i_2 = np.loadtxt(fileDir + fileName + '_2_sum_U.txt', dtype=np.complex_)
# sum_M_i_2 = np.loadtxt(fileDir + fileName + '_2_sum_M.txt', dtype=np.complex_)
# sum_chi_i_2 = np.loadtxt(fileDir + fileName + '_2_sum_chi.txt', dtype=np.complex_)

# sum_U_i = np.concatenate((sum_U_i_1,sum_U_i_2),axis=0)
# sum_M_i = np.concatenate((sum_M_i_1,sum_M_i_2),axis=0)
# sum_chi_i = np.concatenate((sum_chi_i_1,sum_chi_i_2),axis=0)

 

# Calculate exact U, M, chi, and S
U_e_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue)
M_e_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue, U_e_exact)
chi_e_exact = calcChis_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue, alpha, U_e_exact, Te, Te)


# Calculate bi-Maxwellian values for ions (these should be the same as the Maxwellian in this case)
U_i_bimax = calcU_biMax(omega, kpar, kperp, vthipar, nmax, rho_avgi, Oci, nui)
M_i_bimax = calcM_biMax(omega, kpar, kperp, vthipar, nmax, rho_avgi, Oci, nui, U_i_bimax)
chi_i_bimax = calcchi_biMax(omega, kpar, kperp, vthipar, vthiperp, nmax, rho_avgi, Oci, nui, U_i_bimax, alpha, Te, Tipar)
S_exact = calcSpectra(M_i_bimax, M_e_exact, chi_i_bimax, chi_e_exact)


# S = calcSpectra(M_i, M_e_exact, chi_i, chi_e_exact)

font = {'size'   : 26}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(9,9))
gs = gridspec.GridSpec(4,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
gs.update(left=0.21, right=.99, bottom=.1, top=.99, wspace=0.015, hspace=.015)
fig.patch.set_facecolor('white')
ax = []
for i in range(0,4):
    ax.append(plt.subplot(gs[i]))

ax[0].plot(omega, M_i_bimax, linewidth=3, color='k')
# ax[0].plot(omega, M_i, '--', linewidth=3, color='C1')

l1 = ax[1].plot(omega, np.real(chi_i_bimax), linewidth=3, color='k',label='Exact')
# l2 = ax[1].plot(omega, np.real(chi_i), '--', linewidth=3, color='C1',label='Approx')

ax[2].plot(omega, np.imag(chi_i_bimax), linewidth=3, color='k')
# ax[2].plot(omega, np.imag(chi_i), '--', linewidth=3, color='C1')

ax[3].plot(omega, S_exact,linewidth=3,color='k')
# ax[3].plot(omega, S,'--',linewidth=3,color='C1')

ax[2].legend(handles=l1,loc='upper right')
# ax[1].legend(handles=l2,loc='upper right')

ax[3].set_xlabel('$\omega$ (rad/s)')

# Look up all of these units later
ax[0].set_ylabel('$M$')
ax[1].set_ylabel('Re$(\\chi_i)$')
ax[2].set_ylabel('Im$(\\chi_i)$')
ax[3].set_ylabel('Spectra') 

for i in range(0,4):    
    ax[i].grid()
    ax[i].set_xticks([-20000,0,20000])
    ax[i].set_xticklabels([])
    ax[i].set_xlim(np.min(omega),np.max(omega))

ax[3].set_xticklabels([-20000,0,20000])
ax[1].set_ylim(-80,240)

fig.savefig(fileDir + fileName + '.pdf',format='pdf')   