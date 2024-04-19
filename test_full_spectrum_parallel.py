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
# nui = 0.00001


# Set ISR parameters
k_ISR = 2*math.pi*nu_ISR/c
theta = np.deg2rad(10)
kpar = k_ISR*np.cos(theta)
kperp = k_ISR*np.sin(theta)

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
lambdaD_i = (eps0*kB*Ti/(ni*e**2))**.5

print(nui/kpar/vthi)

# Calculate alpha
alpha = 1/k_ISR/lambdaD
alpha_i = 1/k_ISR/lambdaD_i

# Set parameters for calculation of modified ion distribution and ion suseptability
nmax = 0
mesh_n = 500

# Build distribution function
def maxwellian_norm(vperp, vpar, vth):
    return np.exp(-(vperp**2+vpar**2)/vth**2)/vth**3/math.pi**1.5

# Make the array for omega based on 3 times ion acoustic speed. (assume gamma_e=gamma_i=5/3)
cs = (5/3*kB*(Ti+Te)/mi)**.5
omega_bounds = round(cs*k_ISR*3,-3)
omega = np.linspace(-omega_bounds,omega_bounds,31)

# Make pole for some arbitrary n
n = 0.0
z = (omega-n*Oci-1j*nui)/kpar

# Calculate exact U, M, chi, and S
U_e_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue)
M_e_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue, U_e_exact)
# chi_e_exact = calcChis_Maxwellian(omega, nue, U_e_exact, alpha, Te, Te)
chi_e_exact = calcChis_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue, alpha, U_e_exact, Te, Te)

U_i_exact = calcUs_Maxwellian(omega, kpar, kperp, vthi, nmax, rho_avgi, Oci, nui)
M_i_exact = calcMs_Maxwellian(omega, kpar, kperp, vthi, nmax, rho_avgi, Oci, nui, U_i_exact)
# chi_i_exact = calcChis_Maxwellian(omega, nui, U_i_exact, alpha, Te, Ti)
chi_i_exact = calcChis_Maxwellian(omega, kpar, kperp, vthi, nmax, rho_avgi, Oci, nui, alpha, U_i_exact, Te, Ti)
S_exact = calcSpectra(M_i_exact, M_e_exact, chi_i_exact, chi_e_exact)

U_i_approx = np.zeros_like(omega) + 1j*0.0
M_i_approx = np.zeros_like(omega) + 1j*0.0
chi_i_approx = np.zeros_like(omega) + 1j*0.0
S_approx = np.zeros_like(omega)

dvpar_order = -4.0
dvpar = 10**dvpar_order*vthi
vpar = np.arange(-4*vthi,4*vthi+dvpar,dvpar)
dvperp = 10**-2*vthi
vperp = np.arange(0,4*vthi+dvperp,dvperp)

# Make a set of arrays for approximate pole integrals
singlePoleOrder1_approx = np.zeros_like(vperp) + 1j*0.0
singlePoleOrder2_approx = np.zeros_like(vperp) + 1j*0.0
doublePole_approx = np.zeros_like(vperp) + 1j*0.0

# Iterate through omega
for k in range(len(omega)):
    print("k:", k, "out of", len(omega)-1)
    # Iterate through vperp
    for i in range(len(vperp)):
        # Calculate distribution function
        f0i = maxwellian_norm(vperp[i], vpar, vthi)
        # Do pole integrations
        singlePoleOrder1_approx[i] = poleIntegrate(np.array([z[k]]), np.array([1]), vpar, f0i, mesh_n, 0)
        singlePoleOrder2_approx[i] = poleIntegrate(np.array([z[k]]), np.array([2]), vpar, f0i, mesh_n, 0)
        doublePole_approx[i] = poleIntegrate(np.array([z[k],np.conjugate(z[k])]), np.array([1,1]), vpar, f0i, mesh_n, 0)
        
    # Calculate U, M, and chi
    U_i_approx[k] = np.trapz(sp.jv(n,kperp*vperp/Oci)**2*vperp*singlePoleOrder1_approx,vperp)    
    M_i_approx[k] = np.trapz(sp.jv(n,kperp*vperp/Oci)**2*vperp*doublePole_approx,vperp)
    chi_i_approx[k] = np.trapz(vperp*singlePoleOrder2_approx*sp.jv(n,kperp*vperp/Oci)**2*(-1),vperp) + n*kperp/kpar*np.trapz(singlePoleOrder1_approx*sp.jv(n,kperp*vperp/Oci)*(sp.jv(n-1,kperp*vperp/Oci)-sp.jv(n+1,kperp*vperp/Oci)),vperp)
    
U_i_approx = -2*math.pi*1j*nui/kpar*U_i_approx
M_i_approx = (M_i_approx*2*math.pi/kpar**2 -np.abs(U_i_approx)**2/nui**2)*nui/np.abs(1+U_i_approx)**2
chi_i_approx = chi_i_approx*2*math.pi*wpi**2/(kpar**2+kperp**2)/(1+U_i_approx)

S_approx = calcSpectra(M_i_approx, M_e_exact, chi_i_approx, chi_e_exact)
font = {'size'   : 26}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(10,12))
gs = gridspec.GridSpec(6,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
gs.update(left=0.25, right=.99, bottom=.08, top=.94, wspace=0.015, hspace=.015)
fig.patch.set_facecolor('white')
ax = []
for i in range(0,6):    
    ax.append(plt.subplot(gs[i])) 

ax[0].plot(omega,np.real(U_i_exact),'k',linewidth=3)
ax[0].plot(omega,np.real(U_i_approx),'--',linewidth=3,color='C1')
ax[0].set_ylabel('Re$(U_i)$')

ax[1].plot(omega,np.imag(U_i_exact),'k',linewidth=3)
ax[1].plot(omega,np.imag(U_i_approx),'--',linewidth=3,color='C1')
ax[1].set_ylabel('Im$(U_i)$')

ax[2].plot(omega,np.real(M_i_exact),'k',linewidth=3)
ax[2].plot(omega,np.real(M_i_approx),'--',linewidth=3,color='C1')
ax[2].set_ylabel('Re$(M_i)$')

ax[3].plot(omega,np.real(chi_i_exact),'k',linewidth=3)
ax[3].plot(omega,np.real(chi_i_approx),'--',linewidth=3,color='C1')
ax[3].set_ylabel('Re$(\chi_i)$')

ax[4].plot(omega,np.imag(chi_i_exact),'k',linewidth=3)
ax[4].plot(omega,np.imag(chi_i_approx),'--',linewidth=3,color='C1')
ax[4].set_ylabel('Im$(\chi_i)$')

ax[5].plot(omega,S_exact,'k',linewidth=3)
ax[5].plot(omega,S_approx,'--',linewidth=3,color='C1')
ax[5].set_ylabel('$S$')

ax[5].set_xlabel('$\omega$')
for i in range(0,6):
    ax[i].grid()
    ax[i].set_xticks([-20000,0,20000])
    ax[i].set_xticklabels([])
    ax[i].set_xlim(np.min(omega),np.max(omega))

ax[0].set_title('$\Delta v_\parallel/v_{th_i}=10^{%.1f}$' % (dvpar_order))
ax[5].set_xticklabels([-20000,0,20000])

# fig.savefig('Documentation/figures/test_full_spectrum_dvpar_%.1f.png' % (dvpar_order),format='png')




