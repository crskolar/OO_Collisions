# Make a plot of how the solution scales with n for the Maxwellian exact solutions

import numpy as np
import math 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits import mplot3d
import scipy.special as sp
from scipy.interpolate import interp1d
from MC2ISR_functions import *
import time
import os

# Set physical constants
import scipy.constants as const
u = const.physical_constants['unified atomic mass unit'][0]
kB = const.k
me = const.m_e
e = const.e
eps0 = const.epsilon_0
c = const.c

def calcIterError(old, new):
    # return np.abs(np.max(np.abs(old)) - np.max(np.abs(new)))/np.max(np.abs(new))
    return np.max(np.abs(old - new))

# Set background parameters based on a reasonable F region plasma 
B = 1e-5
nu_ISR = 450e6

Tn = 1000
nn = 1e14   # m^-3

Ti = 1000
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


# Set ISR parameters
k = 2*math.pi*nu_ISR/c
angle = 0
theta = np.deg2rad(angle)
kpar = k*np.cos(theta)
kperp = k*np.sin(theta)

# Calculate thermal velocities, gyroradii, gyrofrequencies, plasma frequency, and electron Debye length
vthi = (2*kB*Ti/mi)**.5
vthe = (2*kB*Te/me)**.5
Oci = e*B/mi
Oce = e*B/me
rho_avgi = vthi/Oci/2**.5
rho_avge = vthe/Oce/2**.5
wpi = (ni*e**2/mi/eps0)**.5
wpe = (ne*e**2/me/eps0)**.5
lambdaD = (eps0*kB*Te/(ne*e**2))**.5

# Calculate alpha
alpha = 1/k/lambdaD

# Make the array for omega based on 3 times ion acoustic speed. (assume gamma_e=gamma_i=5/3)
cs = (5/3*kB*(Ti+Te)/mi)**.5
omega_bounds = round(cs*k*3,-3)
omega = np.linspace(-omega_bounds,omega_bounds,401)

nmax = 2000

# Make arrays for calculating relative iterative error
Re_Ue_error = np.zeros(nmax)
Im_Ue_error = np.zeros(nmax)
Re_Me_error = np.zeros(nmax)
Re_chie_error = np.zeros(nmax)
Im_chie_error = np.zeros(nmax)
Re_Ui_error = np.zeros(nmax)
Im_Ui_error = np.zeros(nmax)
Re_Mi_error = np.zeros(nmax)
Re_chii_error = np.zeros(nmax)
Im_chii_error = np.zeros(nmax)
S_error = np.zeros(nmax)

# Set colors based on length of phi_bias using inferno colormap\n",
colors = []
for i in range(0,nmax+1):
    colors.append(pl.cm.inferno( i/(nmax+1)) )

font = {'size'   : 26}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(18,12))
gs = gridspec.GridSpec(6,2)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
gs.update(left=0.11, right=.91, bottom=.08, top=.94, wspace=0.03, hspace=.03)
fig.patch.set_facecolor('white')
ax = []
for i in range(0,12):    
    ax.append(plt.subplot(gs[i])) 



# Iterate through n
for n in range(0,nmax+1):
    print("n:", n)
    # Calculate exact U, M, chi, and S
    U_e_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, n, rho_avge, Oce, nue)
    M_e_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, n, rho_avge, Oce, nue, U_e_exact)
    chi_e_exact = calcChis_Maxwellian(omega, kpar, kperp, vthe, n, rho_avge, Oce, nue, alpha, U_e_exact, Te, Te)
    
    U_i_exact = calcUs_Maxwellian(omega, kpar, kperp, vthi, n, rho_avgi, Oci, nui)
    M_i_exact = calcMs_Maxwellian(omega, kpar, kperp, vthi, n, rho_avgi, Oci, nui, U_i_exact)
    chi_i_exact = calcChis_Maxwellian(omega, kpar, kperp, vthi, n, rho_avgi, Oci, nui, alpha, U_i_exact, Te, Ti)
    
    S_exact = calcSpectra(M_i_exact, M_e_exact, chi_i_exact, chi_e_exact)
    
    if n != 0:
        # Calculate the error
        Re_Ue_error[n-1] = calcIterError(np.real(U_e_old), np.real(U_e_exact))
        Im_Ue_error[n-1] = calcIterError(np.imag(U_e_old), np.imag(U_e_exact))
        Re_Me_error[n-1] = calcIterError(np.real(M_e_old), np.real(M_e_exact))
        Re_chie_error[n-1] = calcIterError(np.real(chi_e_old), np.real(chi_e_exact))
        Im_chie_error[n-1] = calcIterError(np.imag(chi_e_old), np.imag(chi_e_exact))
        
        Re_Ui_error[n-1] = calcIterError(np.real(U_i_old), np.real(U_i_exact))
        Im_Ui_error[n-1] = calcIterError(np.imag(U_i_old), np.imag(U_i_exact))
        Re_Mi_error[n-1] = calcIterError(np.real(M_i_old), np.real(M_i_exact))
        Re_chii_error[n-1] = calcIterError(np.real(chi_i_old), np.real(chi_i_exact))
        Im_chii_error[n-1] = calcIterError(np.imag(chi_i_old), np.imag(chi_i_exact))
        
        S_error[n-1] = calcIterError(S_old, S_exact)
    
    
    U_e_old = U_e_exact*1.0
    M_e_old = M_e_exact*1.0
    chi_e_old = chi_e_exact*1.0
    U_i_old = U_i_exact*1.0
    M_i_old = M_i_exact*1.0
    chi_i_old = chi_i_exact*1.0
    S_old = S_exact*1.0
    
    
    ax[0].plot(omega, np.real(U_e_exact), linewidth=3, color=colors[n])
    ax[2].plot(omega, np.imag(U_e_exact), linewidth=3, color=colors[n])
    ax[4].plot(omega, np.real(M_e_exact), linewidth=3, color=colors[n])
    ax[6].plot(omega, np.real(chi_e_exact), linewidth=3, color=colors[n])
    ax[8].plot(omega, np.imag(chi_e_exact), linewidth=3, color=colors[n])
    ax[10].plot(omega, S_exact, linewidth=3, color=colors[n])
    
    ax[1].plot(omega, np.real(U_i_exact), linewidth=3, color=colors[n])
    ax[3].plot(omega, np.imag(U_i_exact), linewidth=3, color=colors[n])
    ax[5].plot(omega, np.real(M_i_exact), linewidth=3, color=colors[n])
    ax[7].plot(omega, np.real(chi_i_exact), linewidth=3, color=colors[n])
    ax[9].plot(omega, np.imag(chi_i_exact), linewidth=3, color=colors[n])
    ax[11].plot(omega, S_exact, linewidth=3, color=colors[n])
    
    
fig.suptitle('Angle=%d' % (angle))
ax[0].set_title('elc')
ax[1].set_title('ion')
ax[0].set_ylabel('Re$(U)$')
ax[2].set_ylabel('Im$(U)$')
ax[4].set_ylabel('Re$(M)$')
ax[6].set_ylabel('Re$(\chi)$')
ax[8].set_ylabel('Im$(\chi)$')
ax[10].set_ylabel('$S$')

ax[10].set_xlabel('$\omega$')
ax[11].set_xlabel('$\omega$')
for i in range(0,12):
    ax[i].grid()
    ax[i].set_xticks([-20000,0,20000])
    ax[i].set_xticklabels([])
    ax[i].set_xlim(np.min(omega),np.max(omega))
    
    if i % 2 == 1: # If odd
        ax[i].yaxis.tick_right()
    
ax[10].set_xticklabels([-20000,0,20000])
ax[11].set_xticklabels([-20000,0,20000])
ax[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[10].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[11].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

fig.savefig('nConvergence_Maxwellian_Exact/nConvergence_Maxwellian_Exact_angle_%d.png' % (angle), format='png')

font = {'size'   : 26}
mpl.rc('font', **font)
fig2 = plt.figure(2, figsize=(12,8))
gs2 = gridspec.GridSpec(1,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
gs2.update(left=0.14, right=.74, bottom=.13, top=.93, wspace=0.015, hspace=.015)
fig2.patch.set_facecolor('white')
ax2 = []
for i in range(0,1):    
    ax2.append(plt.subplot(gs2[i]))
    
nArray = np.arange(1,nmax+1)
ax2[0].plot(nArray, Re_Ue_error,linewidth=2,color='C0',label='Re($U_e$)')
ax2[0].plot(nArray, Im_Ue_error,linewidth=2,color='C1',label='Im($U_e$)')
ax2[0].plot(nArray, Re_Me_error,linewidth=2,color='C3',label='Re($M_e$)')
ax2[0].plot(nArray, Re_chie_error,linewidth=2,color='C8',label='Re($\chi_e$)')
ax2[0].plot(nArray, Im_chie_error,linewidth=2,color='C9',label='Im($\chi_e$)')

ax2[0].plot(nArray, Re_Ui_error,'--',linewidth=2,color='C0',label='Re($U_i$)')
ax2[0].plot(nArray, Im_Ui_error,'--',linewidth=2,color='C1',label='Im($U_i$)')
ax2[0].plot(nArray, Re_Mi_error,'--',linewidth=2,color='C3',label='Re($M_i$)')
ax2[0].plot(nArray, Re_chii_error,'--',linewidth=2,color='C8',label='Re($\chi_i$)')
ax2[0].plot(nArray, Im_chii_error,'--',linewidth=2,color='C9',label='Im($\chi_i$)')

ax2[0].plot(nArray, S_error,'k',linewidth=3,label='$S$')

ax2[0].set_yscale('log')
ax2[0].set_xscale('log')
ax2[0].grid()
ax2[0].set_xlabel('$n_{max}$')
ax2[0].set_ylabel('Iterative Error')
ax2[0].set_xlim(1,2e3)
ax2[0].set_title('Angle: %d' % (angle))
ax2[0].legend(ncols=1,loc='lower left',bbox_to_anchor=  (1.04,-.05,.4,.8))

fig2.savefig('nConvergence_Maxwelliean_Exact_error/nConvergence_Maxwellian_Exact_angle_%d_error.png' % (angle), format='png')



    
    