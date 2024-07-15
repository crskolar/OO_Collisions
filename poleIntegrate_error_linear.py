import numpy as np
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.special as sp
from MC2ISR_functions import *

# Make functions for exact solutions
def p1(z):  # Single first order pole at z
    return np.exp(-z**2)*(-math.pi*sp.erfi(z)+np.log(-1/z)+np.log(z))

def p2(z):
    return -2*math.pi**.5-2*z*p1(z)

def pstar(z): # Two first order poles at z and z*
    return 1j*(p1(np.conjugate(z))-p1(z))/2/np.imag(z)

# Make a function for getting the interpolation coefficients
# def getLinearInterpCoeff2(x, y):
#     numElements = len(y)-1
#     a = np.zeros(numElements)
#     b = np.zeros(numElements)

#     # Iterate through numElements
#     for i in range(numElements):
#         a[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
#         b[i] = y[i]-a[i]*x[i]
        
    return np.vstack((a,b))

# # Make function for the approximate integral calculations
# def p1IndefiniteIntegral(coeff, z, x):
#     a = coeff[0]
#     b = coeff[1]
#     return a*(x-z)+(b+a*z)*np.log(x-z)

# def p2IndefiniteIntegral(coeff, z, x):
#     a = coeff[0]
#     b = coeff[1]
#     return (-b-a*z)/(x-z)+a*np.log(x-z)

# def pstarIndefiniteIntegral(coeff, z, x):
#     a = coeff[0]
#     b = coeff[1]
#     return ((b+a*np.conjugate(z))*np.log(x-np.conjugate(z)) - (b+a*z)*np.log(x-z))*1j/np.imag(z)/2.

# def p1CubeIndefiniteIntegral(coeff, z, x):
#     a = coeff[0]
#     b = coeff[1]
#     c = coeff[2]
#     d = coeff[3]
#     return (x-z)*(6*c+3*b*x+2*a*x**2+9*b*z+5*a*x*z+11*a*z**2)+(d+z*(c+z*(b+a*z)))*np.log(x-z)/6


# def interpolatedIntegral(coeff, z, x, integralFunction):
#     output = 0.0  # Initialize a zero value that we will add to
#     for i in range(len(coeff[0])):
#         output += integralFunction(coeff[:,i], z, x[i+1]) - integralFunction(coeff[:,i], z, x[i])
#     return output

def calcIntegrand(f, v, z, order):
    integrand = f + 1j*0.0
    for i in range(len(z)):
        integrand /= (v-z[i])**order[i]
    return integrand

# Make the delta v and velocity mesh
dv_order = -3.
dv = 10**dv_order
v = np.arange(-4,4+dv,dv)

f0 = np.exp(-v*v)

[a, b] = getLinearInterpCoeff(v, f0)
# linearCoeff = getLinearInterpCoeff2(v, f0)

# cubicFunction = CubicSpline(v, f0)
# cubicCoeff = cubicFunction.c

# Make an array of gammas ranging from 1e-6 to 1e0
gamma = np.logspace(-6,0,51)
real_z = 0.0#0.25  # This   is the real part of z. Choice is arbitrary.
z = real_z - 1j*gamma

# Initialize pole integration values
p1_line = np.zeros_like(gamma)+1j*0.0
p1_cube = np.zeros_like(gamma)+1j*0.0
p1_naive = np.zeros_like(gamma)+1j*0.0
p1_approx = np.zeros_like(gamma)+1j*0.0

pstar_line = np.zeros_like(gamma)+1j*0.0
p1_star = np.zeros_like(gamma)+1j*0.0
pstar_naive = np.zeros_like(gamma)+1j*0.0
pstar_approx = np.zeros_like(gamma)+1j*0.0

p2_line = np.zeros_like(gamma)+1j*0.0
p2_cube = np.zeros_like(gamma)+1j*0.0
p2_naive = np.zeros_like(gamma)+1j*0.0
p2_approx = np.zeros_like(gamma)+1j*0.0

mesh_n = 500

# Iterate through gamma
for k in range(0,len(gamma)):
    p1_naive[k] = np.trapz(calcIntegrand(f0, v, [z[k]], [1]), v)
    pstar_naive[k] = np.trapz(calcIntegrand(f0, v,[z[k],np.conjugate(z[k])], [1,1]), v)
    p2_naive[k] = np.trapz(calcIntegrand(f0, v, [z[k]], [2]), v)
    
    p1_line[k] = interpolatedIntegral(a, b, z[k], v, p1IndefiniteIntegral)
    pstar_line[k] = interpolatedIntegral(a, b, z[k], v, pstarIndefiniteIntegral)
    p2_line[k] = interpolatedIntegral(a, b, z[k], v, p2IndefiniteIntegral)
    
    # p1_cube[k] = interpolatedIntegral(cubicCoeff, z[k], v, p1CubeIndefiniteIntegral)
    
    p1_approx[k] = poleIntegrate([z[k]], [1], v, f0, mesh_n, 0)
    pstar_approx[k] = poleIntegrate([z[k],np.conjugate(z[k])], [1,1], v, f0, mesh_n, 0)
    p2_approx[k] = poleIntegrate([z[k]], [2], v, f0, mesh_n, 0)

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

# Plot naive solutions
ax[0].plot(gamma,np.real(p1_naive),'.',color='C1')
ax[3].plot(gamma,np.imag(p1_naive),'.',color='C1')
ax[1].plot(gamma,np.real(pstar_naive),'.',color='C1',label='_nolegend_')
ax[4].plot(gamma,np.imag(pstar_naive),'.',color='C1')
ax[2].plot(gamma,np.real(p2_naive),'.',color='C1')
ax[5].plot(gamma,np.imag(p2_naive),'.',color='C1')

# Plot pole refined solutions
ax[0].plot(gamma,np.real(p1_approx),'o',fillstyle='none',markersize=8,color='C0')
ax[3].plot(gamma,np.imag(p1_approx),'o',fillstyle='none',markersize=8,color='C0')
ax[1].plot(gamma,np.real(pstar_approx),'o',fillstyle='none',markersize=8,color='C0',label='Pole Refined')
ax[4].plot(gamma,np.imag(pstar_approx),'o',fillstyle='none',markersize=8,color='C0')
ax[2].plot(gamma,np.real(p2_approx),'o',fillstyle='none',markersize=8,color='C0')
ax[5].plot(gamma,np.imag(p2_approx),'o',fillstyle='none',markersize=8,color='C0')

# Plot linear interpolated solutions
ax[0].plot(gamma,np.real(p1_line),'x',markersize=8,color='C3')
ax[3].plot(gamma,np.imag(p1_line),'x',markersize=8,color='C3')
ax[1].plot(gamma,np.real(pstar_line),'x',markersize=8,color='C3',label='_nolegend_')
ax[4].plot(gamma,np.imag(pstar_line),'x',markersize=8,color='C3',label='Linear\nInterpolation')
ax[2].plot(gamma,np.real(p2_line),'x',markersize=8,color='C3')
ax[5].plot(gamma,np.imag(p2_line),'x',markersize=8,color='C3')

# Initial results suggest that cubic spline interpolation does not do a good job for us. Can look into why later
# For now, just use linear
# Plot cubic interpolated solutions
# ax[0].plot(gamma,np.real(p1_cube),'x',markersize=8,color='C3')
# ax[3].plot(gamma,np.imag(p1_cube),'x',markersize=8,color='C3')

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
        
ax[0].legend(['Exact','Trapz'])
ax[1].legend()
ax[4].legend()

fig.savefig('C:/Users/Chirag/Documents/repos/OO_Collisions/Documentation/figures/poleIntegrate_pole0_error_%d.pdf' % (dv_order),format='pdf')