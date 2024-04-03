import numpy as np
import math 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits import mplot3d
import scipy.special as sp
from scipy.interpolate import interp1d
from MC2ISR_functions import *
import time
import os

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
nu_ISR = 450e6

Tn = 1000
nn = 1e14   # m^-3

Ti = 1000
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
k = 2*math.pi*nu_ISR/c
theta = np.deg2rad(0)
kpar = k*np.cos(theta)
kperp = k*np.sin(theta)

# Calculate thermal velocities, gyroradii, gyrofrequencies, plasma frequency, and electron Debye length
vthi = (2*kB*Ti/mi)**.5
vthe = (2*kB*Te/me)**.5
Oci = e*B/mi
Oce = e*B/me
rho_avgi = vthi/Oci/2**.5
rho_avge = vthe/Oce/2**.5
wpi = (ni*e**2/mi/eps0)**.5
wpe = (ne*e**2/me/eps0)**.5
lambdaD = (eps0*kB*Te/(ne*e**2))**.5

print(nui/kpar/vthi)

# Calculate alpha
alpha = 1/k/lambdaD

# Set parameters for calculation of modified ion distribution and ion suseptability
nmax = 2000
mesh_n = 1000

# Build distribution function
def maxwellian_norm(vperp, vpar, vth):
    return np.exp(-(vperp**2+vpar**2)/vth**2)/vth**3/math.pi**1.5

# Set an omega
omega = 100.0

# Make pole for some arbitrary n
n = 1.0
z = (omega-n*Oci-1j*nui)/kpar

dvperp = 10**-1*vthi
vperp = np.arange(0,4*vthi+dvperp,dvperp)

singlePoleOrder1_exact = vthi**-3*math.pi**-1.5*np.exp(-(vperp**2+z*2)/vthi**2)*(-math.pi*sp.erfi(z/vthi)+np.log(-1/z)+np.log(z))
singlePoleOrder2_exact = -2*math.pi**-1.5*vthi**-5*np.exp(-(vperp**2+z**2)/vthi**2)*(vthi*math.pi**.5*np.exp(-z**2/vthi**2)+z*(-math.pi*sp.erfi(z/vthi)+np.log(-1/z)+np.log(z)))

# Make a set of arrays for approximate pole integrals
singlePoleOrder1_approx = np.zeros_like(vperp) + 1j*0.0
singlePoleOrder2_approx = np.zeros_like(vperp) + 1j*0.0

font = {'size'   : 30}
mpl.rc('font', **font)
fig = plt.figure(1, figsize=(14,8))
gs = gridspec.GridSpec(1,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
gs.update(left=0.1, right=.96, bottom=.13, top=.92, wspace=0.015, hspace=.015)
fig.patch.set_facecolor('white')
ax = plt.subplot(gs[0])

# ax.plot(vperp/vthi, np.real(singlePoleOrder2_exact), 'k-', linewidth=4)

meshes = np.array([-3,-3.5,-4,-4.5,-5,-5.5,-6,-6.5])
# meshes = np.array([-3.,-4])
mesh_string = ['Exact']

chi_approx = np.zeros_like(meshes) + 1j*0.0

# meshes = np.array([-6.])
# Set colors based on length of phi_bias using inferno colormap
colors = []
for i in range(0,len(meshes)):
    colors.append(pl.cm.inferno( i/(len(meshes))) )

for k in range(len(meshes)):
    dvpar = 10**meshes[k]*vthi
    vpar = np.arange(-4*vthi,4*vthi+dvpar,dvpar)
    mesh_string.append('%.1f' % (meshes[k]))
    # Iterate through vperp 
    for i in range(len(vperp)):
        print("Mesh size:", meshes[k],"i:",i,"of",len(vperp)-1)
        # Calculate distribution function
        f0i = maxwellian_norm(vperp[i], vpar, vthi)
        
        # Do pole integrations
        singlePoleOrder1_approx[i] = poleIntegrate(np.array([z]), np.array([1]), vpar, f0i, mesh_n, 0)
        singlePoleOrder2_approx[i] = poleIntegrate(np.array([z]), np.array([2]), vpar, f0i, mesh_n, 0)
        
    # ax.plot(vperp/vthi, np.real(singlePoleOrder2_approx), '--', linewidth=3,color=colors[k])
    
    chi_approx[k] = np.trapz(vperp*singlePoleOrder2_approx*sp.jv(n,kperp*vperp/Oci)**2*(-1),vperp) + n*kperp/kpar*np.trapz(singlePoleOrder1_approx*sp.jv(n,kperp*vperp/Oci)*(sp.jv(n-1,kperp*vperp/Oci)-sp.jv(n+1,kperp*vperp/Oci)),vperp)
    
# ax.set_ylim(-6.2e-13,4e-13)
# ax.set_xlabel('$v_\perp/v_{th_i}$')
# ax.set_ylabel('Re$[p_2(z)]$')
# ax.set_xlim(np.min(vperp/vthi),np.max(vperp/vthi))
# ax.legend(mesh_string,ncols=2)
# ax.set_title('$\omega=%d,  n=%d,  mesh=%d$' % (omega,n,mesh_n))
# ax.grid()

ax.plot(meshes,np.real(chi_approx),'.')
ax.plot(meshes,np.imag(chi_approx),'.')
ax.set_title('$\Delta v_\perp=10^{-1}$')
# fig.savefig('Documentation/figures/test_p2_omega_%d_' % (omega) + 'n_%d_' % (n) + 'mesh_%d' % (mesh_n) + '.png',format='png')









