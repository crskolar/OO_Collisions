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

# Build a velocity mesh
dvpar = 1e-6*vthi
dvperp = 1e-1*vthi
vpar = np.arange(-6*vthi,6*vthi+dvpar,dvpar)
vperp = np.arange(0,6*vthi+dvperp,dvperp)
[VVperp, VVpar] = np.meshgrid(vperp, vpar)

# Build distribution function
f0i = np.exp( -(VVperp**2 + VVpar**2)/vthi**2)/vthi**3/math.pi**1.5

# Calculate alpha
alpha = 1/k/lambdaD
# Set parameters for calculation of modified ion distribution and ion suseptability
nmax = 2000
mesh_n = 500


# Make the array for omega based on 3 times ion acoustic speed. (assume gamma_e=gamma_i=5/3)
cs = (5/3*kB*(Ti+Te)/mi)**.5
omega_bounds = round(cs*k*3,-3)
omega = np.linspace(-omega_bounds,omega_bounds,31)
np.savetxt('omega.txt',omega,fmt='%.18e')

# Calculate the exact solutions for the ions, electrons, adn resulting spectra
U_e_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue)
M_e_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue, U_e_exact)
chi_e_exact = calcChis_Maxwellian(omega, nue, U_e_exact, alpha, Te, Te)

U_i_exact = calcUs_Maxwellian(omega, kpar, kperp, vthi, nmax, rho_avgi, Oci, nui)
M_i_exact = calcMs_Maxwellian(omega, kpar, kperp, vthi, nmax, rho_avgi, Oci, nui, U_i_exact)
chi_i_exact = calcChis_Maxwellian(omega, nui, U_i_exact, alpha, Te, Ti)
S_exact = calcSpectra(M_i_exact, M_e_exact, chi_i_exact, chi_e_exact)

# Save exact solutions
np.savetxt('U_e_exact.txt',U_e_exact,fmt='%.18e')
np.savetxt('M_e_exact.txt',M_e_exact,fmt='%.18e')
np.savetxt('chi_e_exact.txt',chi_e_exact,fmt='%.18e')
np.savetxt('U_i_exact.txt',U_i_exact,fmt='%.18e')
np.savetxt('M_i_exact.txt',M_i_exact,fmt='%.18e')
np.savetxt('chi_i_exact.txt',chi_i_exact,fmt='%.18e')
np.savetxt('S_exact.txt',S_exact,fmt='%.18e')


# Check if a file exists for each of U, M, chi, and their internal sums
if not os.path.exists('U.txt'):
    np.savetxt('U.txt',np.zeros((len(omega), nmax+1)) + 1j*0.0,fmt='%.18e')
    
if not os.path.exists('M.txt'):
    np.savetxt('M.txt',np.zeros((len(omega), nmax+1)) + 1j*0.0,fmt='%.18e')
    
if not os.path.exists('chi.txt'):
    np.savetxt('chi.txt',np.zeros((len(omega), nmax+1)) + 1j*0.0,fmt='%.18e')

if not os.path.exists('sum_U.txt'):
    np.savetxt('sum_U.txt',np.zeros((len(omega), nmax+1)) + 1j*0.0,fmt='%.18e')
    
if not os.path.exists('sum_M.txt'):
    np.savetxt('sum_M.txt',np.zeros((len(omega), nmax+1)) + 1j*0.0,fmt='%.18e')
    
if not os.path.exists('sum_chi.txt'):
    np.savetxt('sum_chi.txt',np.zeros((len(omega), nmax+1)) + 1j*0.0,fmt='%.18e')

# Load U, M, chi, and their internal sums
U = np.loadtxt('U.txt', dtype=np.complex_)
M = np.loadtxt('M.txt', dtype=np.complex_)
chi = np.loadtxt('chi.txt', dtype=np.complex_)
sum_U = np.loadtxt('sum_U.txt', dtype=np.complex_)
sum_M = np.loadtxt('sum_M.txt', dtype=np.complex_)
sum_chi = np.loadtxt('sum_chi.txt', dtype=np.complex_)


saveFreq = 20
# Iterate through n and omega
for n in range(0,nmax+1):
    start_time = time.time()
    for k in range(0,len(omega)):
        print("n:",n, "omega:",omega[k])
        # If U is not 0, then we have not done this calculation yet. 
        if U[k,n] == 0:
            # Check if n is 0. For this case, we need our initial sums to be 0s
            if n == 0:
                [M[k,n], chi[k,n], U[k,n], sum_M[k,n], sum_chi[k,n], sum_U[k,n]] = calcU_M_chi(vpar, vperp, f0i, omega[k], kpar, kperp, n, n, Oci, nui, mesh_n, 0, wpi, 0.0+1j*0.0, 0.0+1j*0.0, 0.0+1j*0.0)
            # Otherwise, we use the previous value to get the current sum
            else:
                [M[k,n], chi[k,n], U[k,n], sum_M[k,n], sum_chi[k,n], sum_U[k,n]] = calcU_M_chi(vpar, vperp, f0i, omega[k], kpar, kperp, n, n, Oci, nui, mesh_n, 0, wpi, sum_U[k,n-1], sum_M[k,n-1], sum_chi[k,n-1])
    
    if np.round(n/saveFreq) == n/saveFreq:
        # Save the data every saveFreq frames
        np.savetxt('U.txt', U, fmt='%.18e')
        np.savetxt('M.txt', M, fmt='%.18e')
        np.savetxt('chi.txt', chi, fmt='%.18e')
        np.savetxt('sum_U.txt', sum_U, fmt='%.18e')
        np.savetxt('sum_M.txt', sum_M, fmt='%.18e')
        np.savetxt('sum_chi.txt', sum_chi, fmt='%.18e')
            
    print("time:",time.time()-start_time)

print("Done!")


