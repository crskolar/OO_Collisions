#%% 
# This script will calculate the susceptibility of an arbitrary distribution function
import numpy as np
import matplotlib.pyplot as plt
from calcChiFunctions import *

# Import constants
import scipy.constants as const
me = const.m_e
e = const.e

# Set plasma and radar parameters
B = 1e-5
ne = 1e10
Te = 1000
nue = 100
nu_radar = 230.e6
theta = 45.

nmax = 5

# Calculate thermal velocity (needed to get 1D and 2D velocity meshes)
vthe = getVth(Te, me)

# Set parameters for making velocity mesh. Then build mesh. 
dvperp_order = -2.0   # Determines dvperp as 10**dvperp_order*vth
dvpar_order = -2.3    # Determines dvperp as 10**dvpar_order*vth
extentPerp = 4        # Determines extent of velocity space in perp direction
extentPar = 4         # Determines extent of velocity space in par direction
[VVperp, VVpar, vperp, vpar] = makeVelocityMesh(vthe, dvperp_order, dvpar_order, extentPerp, extentPar)

# Make distribution function
f0e = maxwellian_norm(VVperp, VVpar, vthe)

# Choose the omega you want to calculate the suceptibilities for
omega = np.linspace(-8e6,8e6,151)
# omega = np.linspace(6143200-1e6,6143200+1e6,101)  # This commented out line is the frequencies around the plasma line for these parameters

chi = calcChi(vperp, vpar, f0e, omega, nu_radar, theta, e, B, me, nue, ne, nmax)

# Calculate the exact solution for chi for a Maxwellian
Oce = getGyroFrequency(e, B, me)
[kperp, kpar] = getK(nu_radar, theta)
rho_avge = getLarmorRadius_avg(vthe, Oce)
alpha = getAlpha(kperp, kpar, ne, Te)
U_exact = calcU_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue)
chi_exact = calcChi_Maxwellian(omega, kpar, kperp, vthe, 2000, rho_avge, Oce, nue, alpha, U_exact, Te, Te)

plt.subplot(2,1,1)
plt.plot(omega, np.real(chi_exact), linewidth=4, color='k',label='Exact')
plt.plot(omega, np.real(chi), '--',linewidth = 3,color='#75bbfd',label='Numerical')
plt.ylabel('Re$(\\chi)$')
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(omega, np.imag(chi_exact), linewidth=4, color='k',label='Exact')
plt.plot(omega, np.imag(chi), '--',linewidth = 3,color='#75bbfd',label='Numerical')
plt.xlabel('$\\omega$ (rad/s)')
plt.ylabel('Im$(\\chi)$')
plt.grid()
# %%
