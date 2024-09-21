import numpy as np
import math
import time
import matplotlib.pyplot as plt
import scipy.special as sp
import logging
import sys
sys.path.insert(0, '/mnt/c/Users/Chirag/Documents/repos/OO_Collisions/')   # This is the path to where ISRSpectraFunctions is
from ISRSpectraFunctions import *

# Set file stuff
fileDir = '/mnt/c/Users/Chirag/Documents/2024_PoP_ISR/data/'   # This is path to where data will be saved
fileNameIon = 'maxellian_full_ion_theta_60_highColls'
fileNameElc = 'maxellian_full_elc_theta_60_highColls' 

angle = 60              # Aspect angle
num_proc = 10           # Number of processors to use 

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

Ti = 1000.
mi = 16*u
ni = 1.e10
nui = 1000.

Te = 1200.
ne = 1.e10 
nue = 1000.

# Set ISR parameters
k_ISR = 2*2*math.pi*nu_ISR/c
theta = np.deg2rad(angle)
kpar = k_ISR*np.cos(theta)
kperp = k_ISR*np.sin(theta)

# Calculate thermal velocities, gyrofrequencies, and plasma frequencies
vthi = (2*kB*Ti/mi)**.5
vthe = (2*kB*Te/me)**.5
Oci = e*B/mi
Oce = e*B/me
wpi = (ni*e**2/mi/eps0)**.5
wpe = (ne*e**2/me/eps0)**.5

# Set parameters for calculation of modified ion distribution and ion susceptibility
nStart = 0   # Almost always start at 0, but added functionality in case you needed to run extra summation terms
nmax_ion = 1000   # This is sufficient to reach machine precision for this problem. Will need more or fewer depending on paramters and distribution shape
nmax_elc = 13     # This is sufficient to reach machine precision for this problem. Will need more or fewer depending on paramters and distribution shape

# Make the array for omega based on finer resolution at the ion, gyro, and plasma lines
# Choose whatever omega you want for your spectrum. These are chosen to focus on the ion, gyro, and plasma lines for this particular set of parameters.
# Ideally, you want your omega to by symmetric about 0
omega_full = np.linspace(-8e6,8e6,151)
omega_ion = np.linspace(-6e4,6e4,151)
omega_gyro = np.linspace(1856000-5e5,1856000+5e5,51)
omega_plasma= np.linspace(6143200-1e6,6143200+1e6,101)
omega = np.unique(np.concatenate((omega_ion, omega_full,omega_gyro,-omega_gyro,omega_plasma,-omega_plasma)))  # Remove duplicate omegas and sort

# Build ion and elc velocity meshes

# dvpar_order and dvperp_order determine how refined we want the velocity mesh.
# The delta v is defined as 10**dv_order*vthi.
# This is useful because then you can use the error plots in conjunction with the pole location to get an estimate for your dvpar
# dvperp has less constraints on it to get good results
# But, when trying a new problem, it is useful to try differing dv's so that you can make sure you've actually converged to the solution
dvipar_order = -2.3
dviperp_order = -2.0
dvipar = 10**dvipar_order*vthi
vipar = np.arange(-4*vthi,4*vthi+dvipar,dvipar)
dviperp = 10**dviperp_order*vthi
viperp = np.arange(0,4*vthi+dviperp,dviperp)

dvepar_order = -2.3
dveperp_order = -2.0
dvepar = 10**dvepar_order*vthe
vepar = np.arange(-4*vthe,4*vthe+dvepar,dvepar)
dveperp = 10**dveperp_order*vthe
veperp = np.arange(0,4*vthe+dveperp,dveperp)

# Use np meshgrid (which probably functions similarly to matlab's) to build velocity grid
# Note, it is very important that you build the velocity grid exactly this way for the Python code to work.
# You need to have [VVperp, VVpar] be the ouput and (vperp, vpar) be the inputs in that exact order.
# The linear interpolation function (linearInterpCoeff) only works in that case (I have not generalized it further)
# For your Matlab, you can set it up however you want, but the key is that you need to make sure you are doing the linear interpolation integration in the parellel direction and trapz in the perpendicular direction
[VViperp, VVipar] = np.meshgrid(viperp, vipar)
[VVeperp, VVepar] = np.meshgrid(veperp, vepar)

# Build the distribution function
f0i = maxwellian_norm(VViperp, VVipar, vthi)
f0e = maxwellian_norm(VVeperp, VVepar, vthe)

# Get the linear interpolation coefficients
[ai, bi] = getLinearInterpCoeff(VVipar, f0i)
[ae, be] = getLinearInterpCoeff(VVepar, f0e)

# Do calculation for ions
initialize_logger(fileNameIon, fileDir, num_proc)
if __name__ == '__main__':
    start_time = time.time()
    [sum_U, sum_M, sum_chi] = calcSumTerms_par(num_proc, nStart, nmax_ion, vipar, viperp, ai, bi, omega, kpar, kperp, Oci, nui, mi, wpi, fileDir, fileNameIon)
    end_time = time.time() - start_time
    logging.info("Finished in time=%.2e s" % (end_time))
logging.shutdown()  

# Do calculation for elc
initialize_logger(fileNameElc, fileDir, num_proc)
if __name__ == '__main__':
    start_time = time.time()
    [sum_U, sum_M, sum_chi] = calcSumTerms_par(num_proc, nStart, nmax_elc, vepar, veperp, ae, be, omega, kpar, kperp, Oce, nue, me, wpe, fileDir, fileNameElc)
    end_time = time.time() - start_time
    logging.info("Finished in time=%.2e s" % (end_time))
logging.shutdown()  