# This script makes a plot showing how our pole integrator is better than a naive trapezoidal integration
# Will show results for single pole and a double pole at the complex conjugate
import numpy as np
import math 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import scipy.special as sp
from MC2ISR_functions import *

# Make functions for exact solutions
def p1(z):  # Single first order pole at z
    return np.exp(-z**2)*(-math.pi*sp.erfi(z)+np.log(-1/z)+np.log(z))

def p2(z):
    return -2*math.pi**.5-2*z*p1(z)

def pstar(z): # Two first order poles at z and z*
    return 1j*(p1(np.conjugate(z))-p1(z))/2/np.imag(z)

def calcIntegrand(f, v, z, order):
    integrand = f + 1j*0.0
    for i in range(len(z)):
        integrand /= (v-z[i])**order[i]
    return integrand

# Initialize figure
font = {'size'   : 22}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(16,6))
gs = gridspec.GridSpec(2,3)
gs.update(left=0.1, right=.99, bottom=.14, top=.92, wspace=0.15, hspace=.015)
fig.patch.set_facecolor('white')
ax = []
for i in range(0,6):
    ax.append(plt.subplot(gs[i]))

# Make an array of gammas ranging from 1e-6 to 1e0
gamma = np.logspace(-6,0,51)
real_z = 0.25  # This is the real part of z. Choice is arbitrary.
z = real_z - 1j*gamma

# Initialize pole integration values
p1_approx = np.zeros_like(gamma)+1j*0.0
p1_naive = np.zeros_like(gamma)+1j*0.0

pstar_approx = np.zeros_like(gamma)+1j*0.0
pstar_naive = np.zeros_like(gamma)+1j*0.0

p2_approx = np.zeros_like(gamma)+1j*0.0
p2_naive = np.zeros_like(gamma)+1j*0.0

# Set the refined number of points we want to consider
mesh_n = 500

# Make the delta v and velocity mesh
dv_order = -5.0
dv = 10**dv_order
v = np.arange(-4,4+dv,dv)

# Calculate distribution function. Assume a simple normalized Maxwellian with normalized velocity coords
f0 = np.exp(-v**2)

# Iterate through gamma
for k in range(len(gamma)): 
    # Calculate poles and orders for each type
    p1_approx[k] = poleIntegrate([z[k]], [1], v, f0, mesh_n, 0)
    p1_naive[k] = np.trapz(calcIntegrand(f0, v, [z[k]], [1]), v)
    pstar_approx[k] = poleIntegrate([z[k],np.conjugate(z[k])], [1,1], v, f0, mesh_n, 0)
    pstar_naive[k] = np.trapz(calcIntegrand(f0, v,[z[k],np.conjugate(z[k])], [1,1]), v)
    p2_approx[k] = poleIntegrate([z[k]], [2], v, f0, mesh_n, 0)
    p2_naive[k] = np.trapz(calcIntegrand(f0, v, [z[k]], [2]), v)

# Plot exact solutions
ax[0].plot(gamma, np.real(p1(z)),'k',linewidth=3)
ax[3].plot(gamma, np.imag(p1(z)),'k',linewidth=3)
ax[1].plot(gamma, np.real(pstar(z)),'k',linewidth=3)
ax[4].plot(gamma, np.imag(pstar(z)),'k',linewidth=3)
ax[2].plot(gamma, np.real(p2(z)),'k',linewidth=3)
ax[5].plot(gamma, np.imag(p2(z)),'k',linewidth=3)

# Plot naive solutions
ax[0].plot(gamma,np.real(p1_naive),'.',color='C1')
ax[3].plot(gamma,np.imag(p1_naive),'.',color='C1')
ax[1].plot(gamma,np.real(pstar_naive),'.',color='C1')
ax[4].plot(gamma,np.imag(pstar_naive),'.',color='C1')
ax[2].plot(gamma,np.real(p2_naive),'.',color='C1')
ax[5].plot(gamma,np.imag(p2_naive),'.',color='C1')

# Plot pole refined solutions
ax[0].plot(gamma,np.real(p1_approx),'o',fillstyle='none',markersize=8,color='C0')
ax[3].plot(gamma,np.imag(p1_approx),'o',fillstyle='none',markersize=8,color='C0')
ax[1].plot(gamma,np.real(pstar_approx),'o',fillstyle='none',markersize=8,color='C0')
ax[4].plot(gamma,np.imag(pstar_approx),'o',fillstyle='none',markersize=8,color='C0')
ax[2].plot(gamma,np.real(p2_approx),'o',fillstyle='none',markersize=8,color='C0')
ax[5].plot(gamma,np.imag(p2_approx),'o',fillstyle='none',markersize=8,color='C0')

# Make plot look nice
ax[0].set_title('$(v-z)$')
ax[1].set_title('$(v-z)(v-z^*)$')
ax[2].set_title('$(v-z)^2$')

ax[0].set_ylabel('Real')
ax[3].set_ylabel('Imag')

ax[0].set_ylim(-.9,-.18)
ax[1].set_ylim(-.1e6,3.6e6)
ax[2].set_ylim(-8,8)
ax[3].set_ylim(-7,0.2)
ax[4].set_ylim(-2,2)
ax[5].set_ylim(-8,8)


ax[4].set_yticks([-1,0,1])
ax[0].legend(['Exact','Trapz','Pole Refined'],framealpha=.9)

for k in range(0,6):
    ax[k].axvline(x=8.085163789706391e-05, color='C3', linestyle='--')
    ax[k].set_xscale('log')
    ax[k].grid()
    ax[k].set_xlim(10**-6.3,10**.3)
    ax[k].set_xticks(np.logspace(-6,0,4))
    
    if k < 3:
        ax[k].set_xticklabels([])
    else:
        ax[k].set_xlabel('$(\\nu/k_{\parallel})/v_{th}$')

fig.savefig('poleIntegrate_error_dv_%d.png' % (dv_order),format='png')