import numpy as np
import math
import time
import matplotlib.pyplot as plt
import scipy.special as sp
import logging
import sys
sys.path.insert(0, '/mnt/c/Users/Chirag/Documents/repos/OO_Collisions/')
from ISRSpectraFunctions import *



# Set file stuff and aspect angle
fileDir = '/mnt/c/Users/Chirag/Documents/2024_PoP_ISR/data/'
fileName= 'superG_p_3_full_theta_60'
p = 3.0
dvpar_order = -2.3
extent = 4
angle = 60
num_proc = 10

# Import constants
import scipy.constants as const
u = const.physical_constants['unified atomic mass unit'][0]
kB = const.k
me = const.m_e
e = const.e
eps0 = const.epsilon_0
c = const.c

# Set background parameters based on a reasonable F region plasma 
nu_ISR = 230.e6   # EISCAT 3D and regular VHF radar frequency
B = 2.e-5

Te = 1200.
ne = 1.e10 
nue = 1000.

# Set ISR parameters
k_ISR = 2*2*math.pi*nu_ISR/c
theta = np.deg2rad(angle)
kpar = k_ISR*np.cos(theta)
kperp = k_ISR*np.sin(theta)

# Calculate thermal velocities, gyroradii, gyrofrequencies, plasma frequency, and electron Debye length
vthe = (2*kB*Te/me)**.5
Oce = e*B/me
wpe = (ne*e**2/me/eps0)**.5

# Set parameters for calculation of modified ion distribution and ion susceptibility
nStart = 0
nmax = 17

# Make the array for omega based on finer resolution at the ion, gyro, and plasma lines
omega_full = np.linspace(-8e6,8e6,151)
omega_ion = np.linspace(-6e4,6e4,151)
omega_gyro = np.linspace(1856000-5e5,1856000+5e5,51)
omega_plasma= np.linspace(6143200-1e6,6143200+1e6,101)
omega = np.unique(np.concatenate((omega_ion, omega_full,omega_gyro,-omega_gyro,omega_plasma,-omega_plasma)))  # Remove duplicate omegas and sort

# Build ion and elc velocity meshes
dvpar_order = -2.3
dvperp_order = -2.0
dvpar = 10**dvpar_order*vthe
vpar = np.arange(-extent*vthe,extent*vthe+dvpar,dvpar)
dvperp = 10**dvperp_order*vthe
vperp = np.arange(0,extent*vthe+dvperp,dvperp)

[VVperp, VVpar] = np.meshgrid(vperp, vpar)

f0 = superG_norm(VVperp, VVpar, vthe, p)

# Get the linear interpolation coefficients
[a, b] = getLinearInterpCoeff(VVpar, f0)

# Do calculation for elc
initialize_logger(fileName, fileDir, num_proc)
if __name__ == '__main__':
    start_time = time.time()
    [sum_U, sum_M, sum_chi] = calcSumTerms_par(num_proc, nStart, nmax, vpar, vperp, a, b, omega, kpar, kperp, Oce, nue, me, wpe, fileDir, fileName)
    end_time = time.time() - start_time
    logging.info("Finished in time=%.2e s" % (end_time))
logging.shutdown()  