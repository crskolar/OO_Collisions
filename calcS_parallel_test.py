import numpy as np
import multiprocessing as mp
import math
import time
import matplotlib.pyplot as plt
import os
from functools import partial
from MC2ISR_functions import *
import logging

fileDir = '/mnt/c/Users/Chirag/Documents/O+O/data/toroidal_ions/'
fileName = 'par1_perp1'
initialize_logger(fileName, fileDir)

total_processors = mp.cpu_count()

numProcessors = np.arange(1,total_processors+1)

# end_time = np.zeros(total_processors)


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

Tipar = 1000
Tiperp = 2000
Ti = (Tipar+2*Tiperp)/3
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
vthipar = (2*kB*Tipar/mi)**.5
vthiperp = (2*kB*Tiperp/mi)**.5
vthe = (2*kB*Te/me)**.5
Oci = e*B/mi
Oce = e*B/me
rho_avge = vthe/Oce/2**.5
wpi = (ni*e**2/mi/eps0)**.5
wpe = (ne*e**2/me/eps0)**.5
lambdaD = (eps0*kB*Te/(ne*e**2))**.5

# Calculate alpha
alpha = 1/k_ISR/lambdaD

# Set parameters for calculation of modified ion distribution and ion suseptability
nmax = 1000
mesh_n = 500

# Make the array for omega based on 3 times ion acoustic speed. (assume gamma_e=gamma_i=5/3)
cs = (5/3*kB*(Ti+Te)/mi)**.5
omega_bounds = round(cs*k_ISR*3,-3)
omega = np.linspace(-omega_bounds,omega_bounds,31)

dvpar_order = -1.0
dvpar = 10**dvpar_order*vthipar
vpar = np.arange(-4*vthipar,4*vthipar+dvpar,dvpar)
dvperp = 10**-1*vthiperp
vperp = np.arange(0,4*vthiperp+dvperp,dvperp)

[VVperp, VVpar] = np.meshgrid(vperp, vpar)

Dstar = 1.8

# Build distribution function
def toroidal_norm(vperp, vpar, vthperp, vthpar, Dstar):
    Cperp = np.abs(vperp)/vthperp;
    Cpar = vpar/vthpar;
    return sp.ive(0,2*Dstar*Cperp)*np.exp(-Cpar**2-(Cperp-Dstar)**2)/vthperp**2/vthpar/math.pi**1.5

f0i = toroidal_norm(VVperp,VVpar, vthiperp, vthipar, Dstar)

if __name__ == '__main__':
    start_time = time.time()
    [sum_U, sum_M, sum_chi] = calcSumTerms_par(18, nmax, vpar, vperp, f0i, omega, kpar, kperp, Oci, nui, mesh_n, 0, fileDir, fileName)

    end_time = time.time() - start_time
    logging.info("Finished in time=%.2e s" % (end_time))
logging.shutdown()