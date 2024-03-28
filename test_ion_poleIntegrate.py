import numpy as np
import math 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
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


# Set ISR parameters
k = 2*math.pi*nu_ISR/c
theta = np.deg2rad(0)
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

# Set parameters for calculation of modified ion distribution and ion suseptability
nmax = 2000
mesh_n = 1000

# Build a velocity mesh
dv = 10**-4.5*vthi
vpar = np.arange(-4*vthi,4*vthi+dv,dv)
vperp = np.arange(0,4*vthi+dv,dv)
# [VVperp, VVpar] = np.meshgrid(vperp, vpar)

# Build distribution function
def maxwellian_norm(vperp, vpar, vth):
    return np.exp(-(vperp**2+vpar**2)/vth**2)/vth**3/math.pi**1.5

# Set an omega
omega = 100.0

# Make pole for some arbitrary n
n = 0.0
z = (omega-n*Oci-1j*nui)/kpar

# Make a set of arrays for approximate pole integrals
singlePoleOrder1_approx = np.zeros_like(vperp) + 1j*0.0
singlePoleOrder2_approx = np.zeros_like(vperp) + 1j*0.0
# Iterate through vperp 
for i in range(len(vperp)):
    print("i:",i,"of",len(vperp)-1)
    # Calculate distribution function
    f0i = maxwellian_norm(vperp[i], vpar, vthi)
    
    # Do pole integrations
    singlePoleOrder1_approx[i] = poleIntegrate(np.array([z]), np.array([1]), vpar, f0i, mesh_n, 0)
    singlePoleOrder2_approx[i] = poleIntegrate(np.array([z]), np.array([2]), vpar, f0i, mesh_n, 0)



# # Make the array for omega based on 3 times ion acoustic speed. (assume gamma_e=gamma_i=5/3)
# # cs = (5/3*kB*(Ti+Te)/mi)**.5
# # omega_bounds = round(cs*k*3,-3)
# # omega = np.linspace(-omega_bounds,omega_bounds,401)
# omega = 100.0

# # Calculate the exact solutions for the ions, electrons, adn resulting spectra
# U_e_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue)
# M_e_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue, U_e_exact)
# chi_e_exact = calcChis_Maxwellian(omega, nue, U_e_exact, alpha, Te, Te)

# U_i_exact = calcUs_Maxwellian(omega, kpar, kperp, vthi, nmax, rho_avgi, Oci, nui)
# M_i_exact = calcMs_Maxwellian(omega, kpar, kperp, vthi, nmax, rho_avgi, Oci, nui, U_i_exact)
# chi_i_exact = calcChis_Maxwellian(omega, nui, U_i_exact, alpha, Te, Ti)
# S_exact = calcSpectra(M_i_exact, M_e_exact, chi_i_exact, chi_e_exact)




# # Make the correct solutions for the pole integrations
singlePoleOrder1_exact = vthi**-3*math.pi**-1.5*np.exp(-(vperp**2+z*2)/vthi**2)*(-math.pi*sp.erfi(z/vthi)+np.log(-1/z)+np.log(z))
singlePoleOrder2_exact = -2*math.pi**-1.5*vthi**-5*np.exp(-(vperp**2+z**2)/vthi**2)*(vthi*math.pi**.5*np.exp(-z**2/vthi**2)+z*(-math.pi*sp.erfi(z/vthi)+np.log(-1/z)+np.log(z)))

# # Do pole integration calculation
# singlePoleOrder1_approx = poleIntegrate(np.array([z]), np.array([1]), vpar, f0i, mesh_n, 0)
# singlePoleOrder2_approx = poleIntegrate(np.array([z]), np.array([2]), vpar, f0i, mesh_n, 0)


plt.plot(vperp, np.real(singlePoleOrder2_exact), 'k-', linewidth=2)
plt.plot(vperp, np.real(singlePoleOrder2_approx), '--', linewidth=2)






