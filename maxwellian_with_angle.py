# This script sees how a Maxwellian plasma is viewed differently for different magnetic field angles
# Compare approximate calculation to exact solution
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

fileDir = 'C:/Users/Chirag/Documents/O+O/data/maxwellian_all/'

# Set background and radar parameters
B = 1e-5
nu_ISR = 450e6
k_ISR = 2*math.pi*nu_ISR/c
Tn = 1000
nn = 1e14   # m^-3

# Assume a Maxwellian electron species. Set electron parameters
Te = 1000.
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

# Initialize figure
font = {'size'   : 22}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(8,6))
gs = gridspec.GridSpec(1,1)
gs.update(left=0.13 , right=.99, bottom=.14, top=.945, wspace=0.15, hspace=.015)
fig.patch.set_facecolor('white')

# Iterate through the angles
for i in range(0,8):#len(angles)):
    print(angles[i])
    
    # Build the appropriate filename
    fileName = "par2.3_perp2_theta" + str(angles[i])
    
    # Check the convergence
    # checkSumConvergence(fileDir + fileName, 1e-20)
    
    
    
    # Calculate kpar and kperp
    theta = np.deg2rad(angles[i])
    kpar = k_ISR*np.cos(theta)
    kperp = k_ISR*np.sin(theta)
    
    # Calculate exact U, M, chi, and S for ions and electrons
    U_e_exact = calcU_Maxwellian(omega_exact, kpar, kperp, vthe, 2000, rho_avge, Oce, nue)
    M_e_exact = calcM_Maxwellian(omega_exact, kpar, kperp, vthe, 2000, rho_avge, Oce, nue, U_e_exact)
    chi_e_exact = calcChi_Maxwellian(omega_exact, kpar, kperp, vthe, 2000, rho_avge, Oce, nue, alpha, U_e_exact, Te, Te)
    
    U_i_exact = calcU_Maxwellian(omega_exact, kpar, kperp, vthi, 2000, rho_avgi, Oci, nui)
    M_i_exact = calcM_Maxwellian(omega_exact, kpar, kperp, vthi, 2000, rho_avgi, Oci, nui, U_i_exact)
    chi_i_exact = calcChi_Maxwellian(omega_exact, kpar, kperp, vthi, 2000, rho_avgi, Oci, nui, alpha, U_i_exact, Te, Ti)
    
    S_exact = calcSpectra(M_i_exact, M_e_exact, chi_i_exact, chi_e_exact)
    
    # # Load data
    # [sum_U_i, sum_M_i, sum_chi_i, omega, kperp, kpar, nui, mi] = loadSumData(fileDir+fileName)
    
    # # # Calculate U, M, chi, and the resulting spectrum
    # U_e = calcU_Maxwellian(omega, kpar, kperp, vthe, 2000, rho_avge, Oce, nue)
    # M_e = calcM_Maxwellian(omega, kpar, kperp, vthe, 2000, rho_avge, Oce, nue, U_e)
    # chi_e = calcChi_Maxwellian(omega, kpar, kperp, vthe, 2000, rho_avge, Oce, nue, alpha, U_e, Te, Te)
    # [U_i, M_i, chi_i] = calcFromSums(sum_U_i, sum_M_i, sum_chi_i, kpar, kperp, nui, wpi)
    # S = calcSpectra(M_i, M_e, chi_i, chi_e)
    
    ax = plt.subplot(gs[0])
    ax.plot(omega_exact, S_exact, linestyle=exactLine, linewidth=3,label='Exact')
    # ax.plot(omega, S, color=approxColor, linestyle=approxLine, linewidth=3,label='Approx')
    
    ax.grid()
    ax.set_xlabel('$\omega$ (rad/s)')
    ax.set_ylabel('$S$')
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
    ax.set_xlim(np.min(omega),np.max(omega))
    ax.legend()
    # fig.savefig(fileDir + fileName+'_spectrum.png',format='png')
    
    
    
    
    