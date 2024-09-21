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
fileName = 'toroidal_theta_10'
angle = 10
extent = 4.0
dvpar_order = -2.3
dvperp_order = -2.0
num_proc = 10
Dstar = 1.8

# Import constants
import scipy.constants as const
u = const.physical_constants['unified atomic mass unit'][0]
kB = const.k
e = const.e
eps0 = const.epsilon_0
c = const.c

# Set background parameters based on a reasonable F region plasma 
nu_ISR = 440.e6
B = 5.e-5

Tiperp = 2000.
Tipar = 1000.
ni = 1.e11
nui = 0.5
mi = 16*u

# Set ISR parameters
k_ISR = 2*2*math.pi*nu_ISR/c
theta = np.deg2rad(angle)
kpar = k_ISR*np.cos(theta)
kperp = k_ISR*np.sin(theta)

# Calculate thermal velocities, gyroradii, gyrofrequencies, plasma frequency, and electron Debye length
vthiperp = (2*kB*Tiperp/mi)**.5
vthipar = (2*kB*Tipar/mi)**.5
Oci = e*B/mi
wpi = (ni*e**2/mi/eps0)**.5

# Set parameters for calculation of modified ion distribution and ion susceptibility
nStart = 0
nmax = 2000 

# Make the array for omega based on ion line
# Base omega for theta > 40  # Double check that number here
# omega = np.linspace(-12e4,12e4,201)

# theta = 40
# omega_full = np.linspace(-12e4, 12e4, 101)
# omega_IA = np.linspace(-9e4, 9e4, 251)
# omega = np.unique(np.concatenate((omega_full, omega_IA)))

# theta = 30
# omega_full = np.linspace(-12e4,12e4, 81)
# omega_IA = np.linspace(-6e4, 6e4, 101)
# omega = np.unique(np.concatenate((omega_full, omega_IA)))

# Theta for theta=20
# omega_full = np.linspace(-12e4,12e4, 31)
# omega_IA = np.linspace(-6e4,6e4, 101)
# omega = np.unique(np.concatenate((omega_full, omega_IA)))

# theta for theta=10
omega_full = np.linspace(-12e4, 12e4, 31)
omega_IA = np.linspace(-6e4, 6e4, 41)
omega_fine = np.linspace(2.5e4, 5e4, 41)
omega = np.unique(np.concatenate((omega_full, omega_IA, omega_fine, -omega_fine)))

dvpar = 10**dvpar_order*vthipar
vpar = np.arange(-extent*vthipar,extent*vthipar+dvpar,dvpar)
dvperp = 10**dvperp_order*vthiperp
vperp = np.arange(0,extent*vthiperp+dvperp,dvperp)

[VVperp, VVpar] = np.meshgrid(vperp, vpar)

f0 = toroidal_norm(VVperp, VVpar, vthiperp, vthipar, Dstar)

# Get the linear interpolation coefficients
[a, b] = getLinearInterpCoeff(VVpar, f0)

# Do calculation for elc
initialize_logger(fileName, fileDir, num_proc)
if __name__ == '__main__':
    start_time = time.time()
    [sum_U, sum_M, sum_chi] = calcSumTerms_par(num_proc, nStart, nmax, vpar, vperp, a, b, omega, kpar, kperp, Oci, nui, mi, wpi, fileDir, fileName)
    end_time = time.time() - start_time
    logging.info("Finished in time=%.2e s" % (end_time))
logging.shutdown()  