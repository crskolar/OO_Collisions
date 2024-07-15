import numpy as np
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.special as sp
from MC2ISR_functions import *

# Should change vertical line to a shaded region corresponding to general realistic values of gamma

# Make functions for exact solutions of a Maxwellian with a particular type of pole
def p1(z):  # Single first order pole at z
    return np.exp(-z**2)*(-math.pi*sp.erfi(z)+np.log(-1/z)+np.log(z))

def p2(z): # Single second order pole at z
    return -2*math.pi**.5-2*z*p1(z)

def pstar(z): # Two first order poles at z and z*
    return 1j*(p1(np.conjugate(z))-p1(z))/2/np.imag(z)

def calcIntegrand(f, v, z, order):
    integrand = f + 1j*0.0
    for i in range(len(z)):
        integrand /= (v-z[i])**order[i]
    return integrand

# Make an array of gammas ranging from 1e-6 to 1e0
gamma = np.logspace(-6,0,51)

# This is the real part of z. Choice is arbitrary.
# For this test, we will be using a z that IS on one of the og mesh points
# This is probably the ideal case of dealing with the poles
real_z = 1.0
z = real_z - 1j*gamma

# This is the extra number of refined points about the pole using the Longley refinement method
mesh_n = 500

# Initialize pole integration values
# Line corresponds to our new linear interpolation integration method
# Trapz corresponds to a simple trapz approach with no modification
# Longley corresponds to the method from Longley 2024: https://doi.org/10.1029/2023GL107212
p1_line = np.zeros_like(gamma)+1j*0.0
p1_trapz = np.zeros_like(gamma)+1j*0.0
p1_longley = np.zeros_like(gamma)+1j*0.0

pstar_line = np.zeros_like(gamma)+1j*0.0
pstar_trapz = np.zeros_like(gamma)+1j*0.0
pstar_longley = np.zeros_like(gamma)+1j*0.0

p2_line = np.zeros_like(gamma)+1j*0.0
p2_trapz = np.zeros_like(gamma)+1j*0.0
p2_longley = np.zeros_like(gamma)+1j*0.0

# Make the delta v and velocity mesh
# Make plots for various dv_order from 0 to -5
dv_order_list = np.arange(0.0,-3.0,-1.0)

# Iterate through all the dv_orders
for dv_order in dv_order_list:
    
    print("Order:", dv_order)
    
    # Make the delta v
    dv = 10**dv_order
    
    # Make the velocity mesh
    v = np.arange(-4,4+dv,dv)

    # Calculate distribution function
    f0 = np.exp(-v*v)
    
    # Get linear interpolation coefficients
    [a, b] = getLinearInterpCoeff(v, f0)

    # Iterate through gamma
    for k in range(0,len(gamma)):
        # Calculate using trapz
        p1_trapz[k] = np.trapz(calcIntegrand(f0, v, [z[k]], [1]), v)
        pstar_trapz[k] = np.trapz(calcIntegrand(f0, v,[z[k],np.conjugate(z[k])], [1,1]), v)
        p2_trapz[k] = np.trapz(calcIntegrand(f0, v, [z[k]], [2]), v)
        
        # Calculate using linear interpolation
        p1_line[k] = interpolatedIntegral(a, b, z[k], v, p1IndefiniteIntegral)
        pstar_line[k] = interpolatedIntegral(a, b, z[k], v, pstarIndefiniteIntegral)
        p2_line[k] = interpolatedIntegral(a, b, z[k], v, p2IndefiniteIntegral)
        
        # Calculate using the pole refinement technique from Longley 2024
        p1_longley[k] = poleIntegrate([z[k]], [1], v, f0, mesh_n, 0)
        pstar_longley[k] = poleIntegrate([z[k],np.conjugate(z[k])], [1,1], v, f0, mesh_n, 0)
        p2_longley[k] = poleIntegrate([z[k]], [2], v, f0, mesh_n, 0)

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
    
    # Plot exact solutions
    ax[0].plot(gamma, np.real(p1(z)),'k',linewidth=3)
    ax[3].plot(gamma, np.imag(p1(z)),'k',linewidth=3)
    ax[1].plot(gamma, np.real(pstar(z)),'k',linewidth=3,label='_nolegend_')
    ax[4].plot(gamma, np.imag(pstar(z)),'k',linewidth=3)
    ax[2].plot(gamma, np.real(p2(z)),'k',linewidth=3)
    ax[5].plot(gamma, np.imag(p2(z)),'k',linewidth=3)


    # Plot Longley 2024 pole refined solutions
    ax[0].plot(gamma,np.real(p1_longley),'o',fillstyle='none',markersize=8,color='C0',label='_nolegend_')
    ax[3].plot(gamma,np.imag(p1_longley),'o',fillstyle='none',markersize=8,color='C0')
    ax[1].plot(gamma,np.real(pstar_longley),'o',fillstyle='none',markersize=8,color='C0',label='Longley\n2024')
    ax[4].plot(gamma,np.imag(pstar_longley),'o',fillstyle='none',markersize=8,color='C0')
    ax[2].plot(gamma,np.real(p2_longley),'o',fillstyle='none',markersize=8,color='C0')
    ax[5].plot(gamma,np.imag(p2_longley),'o',fillstyle='none',markersize=8,color='C0')

    # Plot linear interpolated solutions
    ax[0].plot(gamma,np.real(p1_line),'x',markersize=8,color='C3',label='_nolegend_')
    ax[3].plot(gamma,np.imag(p1_line),'x',markersize=8,color='C3')
    ax[1].plot(gamma,np.real(pstar_line),'x',markersize=8,color='C3',label='_nolegend_')
    ax[4].plot(gamma,np.imag(pstar_line),'x',markersize=8,color='C3',label='Linear\nInterpolation')
    ax[2].plot(gamma,np.real(p2_line),'x',markersize=8  ,color='C3')
    ax[5].plot(gamma,np.imag(p2_line),'x',markersize=8,color='C3')
    
    # Plot trapz solutions
    ax[0].plot(gamma,np.real(p1_trapz),'.',color='C1')
    ax[3].plot(gamma,np.imag(p1_trapz),'.',color='C1')
    ax[1].plot(gamma,np.real(pstar_trapz),'.',color='C1',label='_nolegend_')
    ax[4].plot(gamma,np.imag(pstar_trapz),'.',color='C1')
    ax[2].plot(gamma,np.real(p2_trapz),'.',color='C1')
    ax[5].plot(gamma,np.imag(p2_trapz),'.',color='C1')


    # Make plot look nice
    ax[0].set_title('$(v-z)$')
    ax[1].set_title('$(v-z)(v-z^*)$')
    ax[2].set_title('$(v-z)^2$')
    
    ax[0].set_ylabel('Real')
    ax[3].set_ylabel('Imag')
    
    ax[0].set_ylim(-.6e-12,1e-12)
    ax[1].set_ylim(-.1e6,3.6e6)
    ax[2].set_ylim(-8,8)
    ax[3].set_ylim(-7,0.2)
    ax[4].set_ylim(-2,2)
    ax[5].set_ylim(-8,8)
    
    ax[4].set_yticks([-1,0,1])
    for k in range(0,6):
        # ax[k].axvline(x=8.085163789706391e-05, color='C3', linestyle='--')
        ax[k].set_xscale('log')
        ax[k].grid()
        ax[k].set_xlim(10**-6.3,10**.3)
        ax[k].set_xticks(np.logspace(-6,0,4))
        
        if k < 3:
            ax[k].set_xticklabels([])
        else:
            ax[k].set_xlabel('$(\\nu/k_{\parallel})/v_{th}$')
            
    ax[0].legend(['Exact','Trapz'])
    ax[1].legend()
    ax[4].legend()
    
    fig.savefig('C:/Users/Chirag/Documents/repos/OO_Collisions/Documentation/figures/poleIntegrate_error_pole1_%d.pdf' % (dv_order),format='pdf')
    
    for k in range(0,6):
        ax[k].remove()
    
print('Done!')