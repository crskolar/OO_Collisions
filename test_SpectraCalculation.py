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

Ti = 1000
mi = 16*u
ni = 1e11
nui = 1.0

Te = 2000.
ne = 1e11
nue = 1.0

# Set ISR parameters
k = 2*math.pi*nu_ISR/c
theta = np.deg2rad(30)
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
nmax = 200
mesh_n = 1000

# Set up omega array 
omega = np.concatenate((np.linspace(-2.5e4,-2.1e4,3), np.linspace(-2.1e4,-1.7,51),np.linspace(-1.7e4,1.8e4,3), np.linspace(1.8e4,2.2e4,51), np.linspace(2.2e4,2.5e4,3)))
omega = np.unique(omega)  # This version is still flawed because it skips over the plasma and gyro lines. And I think I'm only getting the plasma lines here
# omega = np.linspace(-2.5e4, 2.5e4, 51)
# omega = np.linspace(-6e3, 6e3, 51)


# Build velocity mesh
dv = 1e-2*vthi   # Based on plots examining error of pole integration scheme
vpar = np.arange(-4*vthi,4*vthi+dv,dv)
vperp = np.arange(0,4*vthi+dv,dv)
[VVperp, VVpar] = np.meshgrid(vperp, vpar)

# Calculate distribution function
f0i = np.exp( -(VVperp**2 + VVpar**2)/vthi**2)/vthi**3/math.pi**1.5
 
# Calculate U, M, and chi for electrons using exact Maxwellian solutions
U_e = calcUs_Maxwellian(omega, kpar, kperp, vthe, 4, rho_avge, Oce, nue)
M_e = calcMs_Maxwellian(omega, kpar, kperp, vthe, 4, rho_avge, Oce, nue, U_e)
chi_e = calcChis_Maxwellian(omega, nue, U_e, alpha, Te, Te)

# Calculate M and chi for ions using numerical integration (with pole refined velocity mesh)
[M_i, chi_i] = calcM_Chi(vpar, vperp, f0i, omega, kpar, kperp, nmax, Oci, nui, mesh_n, 0, wpi)

# Calculate expected ISR spectra
S = calcSpectra(M_i, M_e, chi_i, chi_e)

plt.plot(omega, np.real(S))
plt.xlabel('$\omega$ (rad/s)')
plt.ylabel('S')
plt.grid()