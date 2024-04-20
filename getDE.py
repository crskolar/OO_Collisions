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

epsilon = 1e-12  # Additional value to add in denominator for discretization errors

# Make a function for checking if a file exists. 
# If it does, load it
# If not, make a 2D array of zeroes with specified number of rows and cols
def loadTxtFile(fileName, nr, nc):
    if os.path.exists(fileName):
        data = np.loadtxt(fileName)
    else:
        data = np.zeros((nr, nc))
    return data

# Make a function to calculate the discretization error
def calcDE(exact, approx, epsilon):
    return np.abs( (exact - approx) / np.max(np.abs(exact)) )

def calcNorm(data, p):
    return np.sum(data**p)**(1.0/p)/len(data)**(1.0/p)

def calcDENorm(exact, approx, epsilon, p):
    DE = calcDE(exact, approx, epsilon)
    return calcNorm(DE, p)

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
angle = 10
theta = np.deg2rad(angle)
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
nmax = 0
mesh_n = 500

# Build distribution function
def maxwellian_norm(vperp, vpar, vth):
    return np.exp(-(vperp**2+vpar**2)/vth**2)/vth**3/math.pi**1.5

# Make the array for omega based on 3 times ion acoustic speed. (assume gamma_e=gamma_i=5/3)
cs = (5/3*kB*(Ti+Te)/mi)**.5
omega_bounds = round(cs*k*3,-3)
omega = np.linspace(-omega_bounds,omega_bounds,31)

# What orders do we want to consider for vperp and vpar
meshes_par = np.linspace(-4,-6,3)
meshes_perp = np.linspace(-1,-3,3)

# Save the meshes
np.savetxt('meshes_par.txt',meshes_par)
np.savetxt('meshes_perp.txt',meshes_perp)

npar = len(meshes_par)
nperp = len(meshes_perp)

# Make arrays for the discretization errors 
DE_U_i_real_norm = loadTxtFile('DE_U_i_real_angle_%d_n_%d.txt' % (angle,nmax), npar, nperp)
DE_U_i_imag_norm = loadTxtFile('DE_U_i_angle_%d_n_%d.txt' % (angle,nmax), npar, nperp)
DE_M_i_real_norm = loadTxtFile('DE_M_i_real_angle_%d_n_%d.txt' % (angle,nmax), npar, nperp)
DE_chi_i_real_norm = loadTxtFile('DE_chi_i_real_angle_%d_n_%d.txt' % (angle,nmax), npar, nperp)
DE_chi_i_imag_norm = loadTxtFile('DE_chi_i_imag_angle_%d_n_%d.txt' % (angle,nmax), npar, nperp)
DE_S_norm = loadTxtFile('DE_S_angle_%d_n_%d.txt' % (angle,nmax), npar, nperp)

# Calculate exact U, M, chi, and S
U_e_exact = calcUs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue)
M_e_exact = calcMs_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue, U_e_exact)
chi_e_exact = calcChis_Maxwellian(omega, kpar, kperp, vthe, nmax, rho_avge, Oce, nue, alpha, U_e_exact, Te, Te)

U_i_exact = calcUs_Maxwellian(omega, kpar, kperp, vthi, nmax, rho_avgi, Oci, nui)
M_i_exact = calcMs_Maxwellian(omega, kpar, kperp, vthi, nmax, rho_avgi, Oci, nui, U_i_exact)
chi_i_exact = calcChis_Maxwellian(omega, kpar, kperp, vthi, nmax, rho_avgi, Oci, nui, alpha, U_i_exact, Te, Ti)
S_exact = calcSpectra(M_i_exact, M_e_exact, chi_i_exact, chi_e_exact)

# Initialize U, M, and chi approximations
U_i_approx = np.zeros_like(omega) + 1j*0.0
M_i_approx = np.zeros_like(omega) + 1j*0.0
chi_i_approx = np.zeros_like(omega) + 1j*0.0

# Iterate through vpar meshes
for i in range(npar):
    # Make the parallel velocity mesh
    dvpar = 10**meshes_par[i]*vthi
    vpar = np.arange(-4*vthi,4*vthi+dvpar,dvpar)
    
    # Iterate through vperp meshes
    for j in range(nperp):
        
        if DE_S_norm[i,j] == 0:
            # Make perpendicular velocity mesh
            dvperp = 10**meshes_perp[j]*vthi
            vperp = np.arange(0,4*vthi+dvperp,dvperp)
            print(len(vperp))
            singlePoleOrder1_approx = np.zeros_like(vperp) + 1j*0.0
            singlePoleOrder2_approx = np.zeros_like(vperp) + 1j*0.0
            doublePole_approx = np.zeros_like(vperp) + 1j*0.0
            
            # Iterate through omega
            for k in range(len(omega)):
                print("dvpar:", meshes_par[i], "   dvperp:", meshes_perp[j],"   k:", k, "out of", len(omega)-1)
                # Iterate through vperp array
                    
                # Initialize the sums as 0s
                sum_U = 0.0 + 1j*0.0
                sum_M = 0.0 + 1j*0.0
                sum_chi = 0.0 + 1j*0.0
                
                # Iterate through n
                for nBase in range(nmax+1):
                    
                    for sgn in [-1.0,1.0]:
                        n = sgn*nBase
                        # Make the pole
                        z = (omega-n*Oci-1j*nui)/kpar
                        
                        for l in range(len(vperp)):
                            # Make distribution function
                            f0i = maxwellian_norm(vperp[l], vpar, vthi)
                            
                            # Calculate pole integrals
                            singlePoleOrder1_approx[l] = poleIntegrate(np.array([z[k]]), np.array([1]), vpar, f0i, mesh_n, 0)
                            singlePoleOrder2_approx[l] = poleIntegrate(np.array([z[k]]), np.array([2]), vpar, f0i, mesh_n, 0)
                            doublePole_approx[l] = poleIntegrate(np.array([z[k],np.conjugate(z[k])]), np.array([1,1]), vpar, f0i, mesh_n, 0)
                        
                        sum_U += np.trapz(sp.jv(n,kperp*vperp/Oci)**2*vperp*singlePoleOrder1_approx,vperp)
                        sum_M += np.trapz(sp.jv(n,kperp*vperp/Oci)**2*vperp*doublePole_approx,vperp)
                        sum_chi += np.trapz(vperp*singlePoleOrder2_approx*sp.jv(n,kperp*vperp/Oci)**2*(-1),vperp) + n*kperp/kpar*np.trapz(singlePoleOrder1_approx*sp.jv(n,kperp*vperp/Oci)*(sp.jv(n-1,kperp*vperp/Oci)-sp.jv(n+1,kperp*vperp/Oci)),vperp)
                        
                        # Make sure we are only doing this once for n=0 (otherwise this loop will double count zeroth order term)
                        if n == 0:
                            break
                        
                        
                # Perform remaining operations to calculate U, M, and Chi
                U_i_approx[k] = -2*math.pi*1j*nui/kpar*sum_U
                M_i_approx[k] = (sum_M*2*math.pi/kpar**2 -np.abs(U_i_approx[k])**2/nui**2)*nui/np.abs(1+U_i_approx[k])**2
                chi_i_approx[k] = sum_chi*2*math.pi*wpi**2/(kpar**2+kperp**2)/(1+U_i_approx[k])
            
            # Calculate IS spectra
            S_approx = np.real(calcSpectra(M_i_approx, M_e_exact, chi_i_approx, chi_e_exact))
            
            # Get the norms of the dicretization error
            DE_U_i_real_norm[i,j] = calcDENorm(np.real(U_i_exact), np.real(U_i_approx), epsilon, 2)
            DE_U_i_imag_norm[i,j] = calcDENorm(np.imag(U_i_exact), np.imag(U_i_approx), epsilon, 2)
            DE_M_i_real_norm[i,j] = calcDENorm(np.real(M_i_exact), np.real(M_i_approx), epsilon, 2)
            DE_chi_i_real_norm[i,j] = calcDENorm(np.real(chi_i_exact), np.real(chi_i_approx), epsilon, 2)
            DE_chi_i_imag_norm[i,j] = calcDENorm(np.imag(chi_i_exact), np.imag(chi_i_approx), epsilon, 2)
            DE_S_norm[i,j] = calcDENorm(S_exact, S_approx, epsilon, 2)
            
            # Save the norm data
            print("Saving data...")
            np.savetxt('DE_U_i_real_angle_%d_n_%d.txt' % (angle,nmax),DE_U_i_real_norm)
            np.savetxt('DE_U_i_imag_angle_%d_n_%d.txt' % (angle,nmax),DE_U_i_imag_norm)
            np.savetxt('DE_M_i_real_angle_%d_n_%d.txt' % (angle,nmax),DE_M_i_real_norm)
            np.savetxt('DE_chi_i_real_angle_%d_n_%d.txt' % (angle,nmax),DE_chi_i_real_norm)
            np.savetxt('DE_chi_i_imag_angle_%d_n_%d.txt' % (angle,nmax),DE_chi_i_imag_norm)
            np.savetxt('DE_S_angle_%d_n_%d.txt' % (angle,nmax),DE_S_norm)
        
        
        # Calculate the difference between 
        # Iterate through n
        # for n in range(0,nmax+1):
            # print(n)
        
            # print(DE_U_i[i,j])


# # Make a set of arrays for approximate pole integrals
# singlePoleOrder1_approx = np.zeros_like(vperp) + 1j*0.0
# singlePoleOrder2_approx = np.zeros_like(vperp) + 1j*0.0
# doublePole_approx = np.zeros_like(vperp) + 1j*0.0

# # Iterate through omega
# for k in range(len(omega)):
#     print("k:", k, "out of", len(omega)-1)
#     # Iterate through vperp
#     for i in range(len(vperp)):
#         # Calculate distribution function
#         f0i = maxwellian_norm(vperp[i], vpar, vthi)
#         # Do pole integrations
#         singlePoleOrder1_approx[i] = poleIntegrate(np.array([z[k]]), np.array([1]), vpar, f0i, mesh_n, 0)
#         singlePoleOrder2_approx[i] = poleIntegrate(np.array([z[k]]), np.array([2]), vpar, f0i, mesh_n, 0)
#         doublePole_approx[i] = poleIntegrate(np.array([z[k],np.conjugate(z[k])]), np.array([1,1]), vpar, f0i, mesh_n, 0)
        
#     # Calculate U, M, and chi
#     U_i_approx[k] = np.trapz(sp.jv(n,kperp*vperp/Oci)**2*vperp*singlePoleOrder1_approx,vperp)    
#     M_i_approx[k] = np.trapz(sp.jv(n,kperp*vperp/Oci)**2*vperp*doublePole_approx,vperp)
#     chi_i_approx[k] = np.trapz(vperp*singlePoleOrder2_approx*sp.jv(n,kperp*vperp/Oci)**2*(-1),vperp) + n*kperp/kpar*np.trapz(singlePoleOrder1_approx*sp.jv(n,kperp*vperp/Oci)*(sp.jv(n-1,kperp*vperp/Oci)-sp.jv(n+1,kperp*vperp/Oci)),vperp)
    
# U_i_approx = -2*math.pi*1j*nui/kpar*U_i_approx
# M_i_approx = (M_i_approx*2*math.pi/kpar**2 -np.abs(U_i_approx)**2/nui**2)*nui/np.abs(1+U_i_approx)**2
# chi_i_approx = chi_i_approx*2*math.pi*wpi**2/(kpar**2+kperp**2)/(1+U_i_approx)

# S_approx = calcSpectra(M_i_approx, M_e_exact, chi_i_approx, chi_e_exact)
# font = {'size'   : 26}
# mpl.rc('font', **font)
# fig = plt.figure(1, figsize=(10,12))
# gs = gridspec.GridSpec(6,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
# gs.update(left=0.25, right=.99, bottom=.08, top=.94, wspace=0.015, hspace=.015)
# fig.patch.set_facecolor('white')
# ax = []
# for i in range(0,6):    
#     ax.append(plt.subplot(gs[i])) 

# ax[0].plot(omega,np.real(U_i_exact),'k',linewidth=3)
# ax[0].plot(omega,np.real(U_i_approx),'--',linewidth=3,color='C1')
# ax[0].set_ylabel('Re$(U_i)$')

# ax[1].plot(omega,np.imag(U_i_exact),'k',linewidth=3)
# ax[1].plot(omega,np.imag(U_i_approx),'--',linewidth=3,color='C1')
# ax[1].set_ylabel('Im$(U_i)$')

# ax[2].plot(omega,np.real(M_i_exact),'k',linewidth=3)
# ax[2].plot(omega,np.real(M_i_approx),'--',linewidth=3,color='C1')
# ax[2].set_ylabel('Re$(M_i)$')

# ax[3].plot(omega,np.real(chi_i_exact),'k',linewidth=3)
# ax[3].plot(omega,np.real(chi_i_approx),'--',linewidth=3,color='C1')
# ax[3].set_ylabel('Re$(\chi_i)$')

# ax[4].plot(omega,np.imag(chi_i_exact),'k',linewidth=3)
# ax[4].plot(omega,np.imag(chi_i_approx),'--',linewidth=3,color='C1')
# ax[4].set_ylabel('Im$(\chi_i)$')

# ax[5].plot(omega,S_exact,'k',linewidth=3)
# ax[5].plot(omega,S_approx,'--',linewidth=3,color='C1')
# ax[5].set_ylabel('$S$')

# ax[5].set_xlabel('$\omega$')
# for i in range(0,6):
#     ax[i].grid()
#     ax[i].set_xticks([-20000,0,20000])
#     ax[i].set_xticklabels([])
#     ax[i].set_xlim(np.min(omega),np.max(omega))

# ax[0].set_title('$\Delta v_\parallel/v_{th_i}=10^{%.1f}$' % (dvpar_order))
# ax[5].set_xticklabels([-20000,0,20000])

# fig.savefig('Documentation/figures/test_full_spectrum_dvpar_%.1f.png' % (dvpar_order),format='png')




