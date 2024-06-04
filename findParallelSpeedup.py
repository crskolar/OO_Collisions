import numpy as np
import multiprocessing as mp
import math
import time
import matplotlib.pyplot as plt
import os
from functools import partial
from MC2ISR_functions import *
import scipy.special as sp
import logging

# Goal is to iterate over using each processor. Do this for various velocity space resolutions.
# Do this using the same Maxwellian I've been using
# Do to nmax = 2*numProcessors

dvpar_order = -5.0
dvperp_order = -1.0
angle = 50.0

fileDir = '/mnt/f/Cherin/Documents/interim_data/O+O/speed/'

# Set the background plasma values
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
Tiperp = 1000
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

# Set ISR parameters
k_ISR = 2*math.pi*nu_ISR/c
theta = np.deg2rad(angle)
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

# Make the array for omega based on 3 times ion acoustic speed. (assume gamma_e=gamma_i=5/3)
cs = (5/3*kB*(Ti+Te)/mi)**.5
omega_bounds = round(cs*k_ISR*3,-3)
omega = np.linspace(-omega_bounds,omega_bounds,31)

dvpar = 10**dvpar_order*vthipar
vpar = np.arange(-4*vthipar,4*vthipar+dvpar,dvpar)
dvperp = 10**dvperp_order*vthiperp
vperp = np.arange(0,4*vthiperp+dvperp,dvperp)

[VVperp, VVpar] = np.meshgrid(vperp, vpar)

Dstar =  0.0

# Build distribution function
def toroidal_norm(vperp, vpar, vthperp, vthpar, Dstar):
    Cperp = np.abs(vperp)/vthperp
    Cpar = vpar/vthpar
    return sp.ive(0,2*Dstar*Cperp)*np.exp(-Cpar**2-(Cperp-Dstar)**2)/vthperp**2/vthpar/math.pi**1.5


f0i = toroidal_norm(VVperp,VVpar, vthiperp, vthipar, Dstar)

mesh_n = 500


# Iterate through the number of processors
maxNumProc = 8
for i in range(0,maxNumProc):
    if __name__ == '__main__':
        print("Running on", i+1, "processors.")

        # Set nmax so that we have at 2 operations per processor
        nmax = 2*i+1
        fileName = 'par%d_perp%d_theta%d_np%d' % (int(np.abs(dvpar_order)),int(np.abs(dvperp_order)), int(angle), i+1)
        initialize_logger(fileName, fileDir, i+1)
        [sum_U, sum_M, sum_chi] = calcSumTerms_par(i+1, 0, nmax, vpar, vperp, f0i, omega, kpar, kperp, Oci, nui, mesh_n, 0, fileDir, fileName)

        logging.info("Finished!")
        logging.shutdown()
        