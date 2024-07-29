import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from ISRSpectraFunctions import *

import scipy.constants as const
u = const.physical_constants['unified atomic mass unit'][0]
kB = const.k
me = const.m_e
e = const.e
eps0 = const.epsilon_0
c = const.c

# Set file info
fileDir = 'C:/Users/Chirag/Documents/2024_PoP_ISR/data/'
# fileNameIon = 'maxellian_ion_line_ion_theta_0'
# fileNameElc = 'maxellian_ion_line_elc_theta_0'
fileNameIon = 'maxellian_full_k230_ion_theta_60'
fileNameElc = 'maxellian_full_k230_elc_theta_60'
angle = 60

# Set background parameters based on a reasonable F region plasma 
nu_ISR = 230e6   # EISCAT 3D and regular VHF
B = 1.e-5
Tn = 1000
nn = 1.8e14   # m^-3

Ti = 1000
mi = 16*u
ni = 1e9

Te = 1000.
ne = 1e9
nuen = 8.9e-11*nn/1e6*(1+5.7e-4*Te)*Te**.5
nuei = 54.5*ni/1e6/Te**1.5
nuee = 54.5*ne/1e6/2**.5/Te**1.5 
nue = nuen + nuei + nuee

Tr = (Ti+Tn)/2
nuin = 3.67e-11*nn/1e6*Tr**.5*(1-0.064*np.log10(Tr))**2
nuii = 0.22*ni/1e6/Ti**1.5
nuie = me*nuei/mi
nui = nuin + nuii + nuie

# Calculate thermal velocities, gyroradii, gyrofrequencies, plasma frequency, and electron Debye length
vthi = (2*kB*Ti/mi)**.5
vthe = (2*kB*Te/me)**.5
Oci = e*B/mi
Oce = e*B/me

# Load data
[sum_U_i, sum_M_i, sum_chi_i, omega, kperp, kpar, nui, mi, wpi] = loadSumData(fileDir+fileNameIon)
[sum_U_e, sum_M_e, sum_chi_e, omega, kperp, kpar, nue, me, wpe] = loadSumData(fileDir+fileNameElc)

# Calculate U, M, and chi
[U_i, M_i, chi_i] = calcFromSums(sum_U_i, sum_M_i, sum_chi_i, kpar, kperp, nui, wpi)
[U_e, M_e, chi_e] = calcFromSums(sum_U_e, sum_M_e, sum_chi_e, kpar, kperp, nue, wpe)

S = calcSpectra(M_i, M_e, chi_i, chi_e)

# Calculate terms needed for exact solution calculations
rho_avgi = vthi/Oci/2**.5
rho_avge = vthe/Oce/2**.5 
k_ISR = 2*math.pi*nu_ISR/c
lambdaD = (eps0*kB*Te/(ne*e**2))**.5
alpha = 1/k_ISR/lambdaD
theta = np.deg2rad(angle)
kpar = k_ISR*np.cos(theta)
kperp = k_ISR*np.sin(theta)

# Make two arrays for omega: 
# One for ion line, the other to get gyro and plasma lines
cs = (5/3*kB*(Ti+Te)/mi)**.5
omega_bounds = round(cs*k_ISR*4,-3)
omega_exact_IA = np.linspace(-omega_bounds,omega_bounds,101)
omega_exact_full = np.linspace(-3e6,3e6,1001)
omega_exact = np.unique(np.concatenate((omega_exact_IA, omega_exact_full)))

# Calculate exact U, M, chi, and S for ions and electrons
nmax_exact = 2000
U_e_exact = calcU_Maxwellian(omega_exact, kpar, kperp, vthe, nmax_exact, rho_avge, Oce, nue)
M_e_exact = calcM_Maxwellian(omega_exact, kpar, kperp, vthe, nmax_exact, rho_avge, Oce, nue, U_e_exact)
chi_e_exact = calcChi_Maxwellian(omega_exact, kpar, kperp, vthe, nmax_exact, rho_avge, Oce, nue, alpha, U_e_exact, Te, Te)

U_i_exact = calcU_Maxwellian(omega_exact, kpar, kperp, vthi, nmax_exact, rho_avgi, Oci, nui)
M_i_exact = calcM_Maxwellian(omega_exact, kpar, kperp, vthi, nmax_exact, rho_avgi, Oci, nui, U_i_exact)
chi_i_exact = calcChi_Maxwellian(omega_exact, kpar, kperp, vthi, nmax_exact, rho_avgi, Oci, nui, alpha, U_i_exact, Te, Ti)

S_exact = calcSpectra(M_i_exact, M_e_exact, chi_i_exact, chi_e_exact)
    
# Initialize figure
font = {'size'   : 22}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(12,6))
gs = gridspec.GridSpec(1,1)
gs.update(left=0.083 , right=.99, bottom=.14, top=.945, wspace=0.15, hspace=.015)
fig.patch.set_facecolor('white')
ax = plt.subplot(gs[0])

# Make styles for exact and numerical
exactStyle = dict(color='k',linewidth=5,linestyle='-', label='Exact')
numStyle = dict(color='#8e82fe',linewidth=4,linestyle='--',dashes=(4,3),label='Numerical')
arrowStyle = dict(facecolor='black', shrink=0.05)

# Make full spectrum plot
ax.plot(omega_exact/1e6, S_exact, **exactStyle)
ax.plot(omega/1e6, S, **numStyle)

ax.set_ylim(-1e-8, 1.25e-6)
ax.set_xlim(-3,3)
ax.set_xlabel('$\omega$ (MHz)')
ax.set_ylabel('$S$ (Hz$^{-1}$)')
ax.legend(loc='upper right')
ax.grid()

# Annotate plot
ax.annotate('Ion Line', xy=(0,1.1e-6), xytext=(.65,1.1e-6), ha='center',va='center',arrowprops=arrowStyle)
ax.annotate('Gyro Line', xy=(.7,.25e-6), xytext=(.5,.45e-6), ha='center',va='center',arrowprops=arrowStyle)
ax.annotate('Plasma Line', xy=(2.5,.58e-6), xytext=(1.6,.7e-6), ha='center',va='center',arrowprops=arrowStyle)

# Make zoomed in ion line pplot
sub_ax = plt.axes([.26, .63, .25, .25]) 
sub_ax.plot(omega_exact/1000, S_exact, **exactStyle)
sub_ax.plot(omega/1e3, S, **numStyle)
sub_ax.set_xlim(-15,15)
sub_ax.set_ylim(-1e-5,2e-4)
sub_ax.set_yticks([0,1e-4,2e-4])
sub_ax.set_xlabel('$\omega$ (kHz)')
sub_ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
sub_ax.grid()

fig.savefig('C:/Users/Chirag/Documents/2024_PoP_ISR/figures/compareMaxwellian.pdf',format='pdf')









