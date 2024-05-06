# This script tests the standard trapezoidal integration of the Bessel function
import numpy as np
import math 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import scipy.special as sp
from MC2ISR_functions import *

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
mi = 16*u
Ti = 1000
vthi = (2*kB*Ti/mi)**.5
Oci = e*B/mi

def integrand(vperp,n):
    return vperp*sp.jv(n,kperp*vperp/Oci)**2*np.exp(-vperp**2/vthi**2)

# Set ISR parameters
nu_ISR = 450e6
k = 2*math.pi*nu_ISR/c
angle = 10
theta = np.deg2rad(angle)
kpar = k*np.cos(theta)
kperp = k*np.sin(theta)


orders = np.linspace(-1,-6,10)
nOrders = len(orders)

intVal_approx = np.zeros(nOrders)

n = 0
intVal_exact =vthi**2/2*sp.ive(n, kperp**2*vthi**2*.5/Oci**2)

for i in range(nOrders):
    dvperp = 10**orders[i]*vthi
    vperp = np.arange(0,4*vthi+dvperp, dvperp)
    intVal_approx[i] = np.trapz(integrand(vperp,n),vperp)


DE = np.abs(intVal_approx - intVal_exact)/intVal_exact
