import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
import scipy.special as sp
from scipy.interpolate import interp1d

# Calculate the interpolated velocity space based on location of poles
# Still need to do a basic test of this
def getVInterp(poles, v_input, mesh_n):
    # First make sure that the mesh_n is odd. If not, change it such that it is
    if mesh_n % 2 == 0:
        mesh_n += 1
        
    # Create a refined mesh about each pole
    v_fine = np.array([])
    for i in range(len(poles)):
        re_v = np.real(poles[i])
        im_v = np.imag(poles[i])
        
        v_0 = re_v - 10*im_v;
        v_1 = re_v - 2*im_v;
        v_2 = re_v + 2*im_v;
        v_3 = re_v + 10*im_v
        
        # v_mesh = np.concatenate(np.linspace(v_0,v_1,mesh_n))
        v_mesh = np.concatenate((np.linspace(v_0,v_1,mesh_n),np.linspace(v_1,v_2,mesh_n),np.linspace(v_2,v_3,mesh_n)))
        v_fine = np.concatenate((v_fine, v_mesh))
        
    # Combine refined meshes
    v_int = np.concatenate((v_fine, v_input))

    
    # Sort and just get unique values
    v_int = np.unique(v_int)
    
    # Just get the velocity values in the original velocity domain
    v_int = v_int[ v_int >= np.min(v_input) ]
    v_int = v_int[ v_int <= np.max(v_input) ]
    
    return v_int

def getIntegrand(poles, orders, v, Fv):
    integrand = np.transpose(Fv)
    
    for i in range(0, len(poles)):
        integrand = integrand / (v - poles[i])**orders[i]
    return np.transpose(integrand)

# This function does the pole integration
def poleIntegrate(poles, orders, v_input, Fv, mesh_n, int_dir):
    
    # Get the interpolated velocity
    v_int = getVInterp(poles, v_input, mesh_n)
    
    # Make interpolation function and interpolate
    interp_fun = interp1d(v_input, Fv, axis=int_dir)
    
    Fv_int = interp_fun(v_int)
    
    # Check to see if Fv is a 2D array. 
    # If so, we need to copy v_int a bunch of times in the appropriate direction to properly calculate the integrand
        
    
    # Calculate integrand
    integrand = getIntegrand(poles, orders, v_int, Fv_int)
    
    return np.trapz(integrand, v_int,axis=int_dir)

# This function does the analytic Plemelj calculation (not including the principal value term)
def plemelj(poles, orders, v_input, Fv):
    
    # Make interpolation function and interpolate
    interp_fun = interp1d(v_input, Fv)
    
    # Iterate through the poles and carry out Plemelj sum
    plemelj_sum = 0.0+0.0*1j
    for i in range(len(poles)):
        poles_product = 1.0+0.*1j
        Fv_resid = interp_fun(np.real(poles[i]))
        
        # Get the product of all the poles except for the one we're on
        for j in range(len(poles)):
            if i != j:
                poles_product /= (poles[i] - poles[j])**orders[i]
        plemelj_sum += 1j*math.pi*np.sign(np.imag(poles[i]))*poles_product*Fv_resid
    return plemelj_sum

# This function calculates the exact Us for a Maxwellian distribution
def calcUs_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu):
    # Calculate yn
    Us = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from 0 to nmax
    for n in range(nmax+1):
        # Calculate yn
        yn = (omega - n*Oc-1j*nu)/kpar/vth
        
        # Do summation term in Us
        Us += sp.ive(n, kperp**2*rho_avg**2)*(2*sp.dawsn(yn) + 1j*math.pi**.5*np.exp(-yn*yn))
        
    # Multiply by everything else and return
    return Us*1j*nu/kpar/vth


# This function calculates the exact modified distribution for a Maxwellian distribution
def calcMs_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu, Us):
    Ms = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from 0 to nmax
    for n in range(nmax+1):
        yn = (omega - n*Oc-1j*nu)/kpar/vth
        
        Ms += np.exp(-yn**2)*sp.ive(n,kperp**2*rho_avg**2)
    
    return np.real(Ms*math.pi**.5/kpar/vth/np.abs(1+Us)**2-np.abs(Us)**2/nu/np.abs(1+Us)**2)

def calcChis_Maxwellian(omega, nu, Us, alpha, Te, Ts):
    return (1+Us*(1+1j*omega/nu))*Te*alpha**2/Ts/(1+Us)


# Calculate Us numerically
def calcUs(vpar, vperp, f0, omega, kpar, kperp, nmax, Oc, nu, mesh_n, par_dir):
    
    # Initialize Us, which is a function of omega
    Us = np.zeros_like(omega) + 1j*0.0
    
    # Iterate through omega
    for k in range(len(omega)):
        # print(k)
        # Iterate through nmax
        for n in range(nmax+1):
            # Calculate the pole in the parallel integral
            pole = (omega[k] - n*Oc-1j*nu)/kpar
            
            # Do pole refined integration
            int_vpar = poleIntegrate(np.array([pole]), np.array([1]), vpar, f0, mesh_n, par_dir)
            
            Us[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*int_vpar,vperp)
    return -Us*2*math.pi*1j*nu/kpar

# Calculate Ms numerically
def calcMs(vpar, vperp, f0, omega, kpar, kperp, nmax, Oc, nu, mesh_n, par_dir, Us):
    # Initialize Ms
    Ms = np.zeros_like(omega) + 1j*0.0
    
    # Iterate through omega
    for k in range(len(omega)):
        # Iterate through nmax
        for n in range(nmax+1):
            # Calculate poles in parallel integral
            poles = np.array([omega[k]-n*Oc+1j*nu,omega[k]-n*Oc-1j*nu])/kpar
            
            # Do pole refined integration
            int_vpar = poleIntegrate(poles, np.array([1,1]), vpar, f0, mesh_n, par_dir)
            
            Ms[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*int_vpar,vperp)

    return (Ms*2*math.pi/kpar**2 -np.abs(Us)**2/nu**2)*nu/np.abs(1+Us)**2  #(Ms*2*math.pi/kpar**2-np.abs(Us)**2/nu**2)*nu/np.abs(1+Us)**2

# Calculate Chis numerically
# wp is plasma frequency
def calcChis(vpar, vperp, f0, omega, kpar, kperp, nmax, Oc, nu, mesh_n, par_dir, Us, wp):
    # font = {'size'   : 22}
    # mpl.rc('font', **font)
    # fig = plt.figure(1, figsize=(7,6))
    # gs = gridspec.GridSpec(1,1)#, height_ratios=[0.05,1,0.2], width_ratios=[1,0.2,0.2,1])
    # gs.update(left=0.3, right=.96, bottom=.13, top=.99, wspace=0.015, hspace=.015)
    # fig.patch.set_facecolor('white')
    # ax = []
    # for i in range(0,1):    
    #     ax.append(plt.subplot(gs[i])) 
        
    # Initialize Chis
    Chis = np.zeros_like(omega) + 1j*0.0
    # Iterate through omega
    for k in range(len(omega)):
        # Iterate through nmax
        for n in range(0,nmax+1):
            # Calculate pole in parallel integral
            pole = np.array([(omega[k] - n*Oc-1j*nu)/kpar])
            
            # Do both pole refined integrations
            int_vpar_first_order = poleIntegrate(pole, np.array([1]), vpar, f0, mesh_n, par_dir)
            int_vpar_second_order = poleIntegrate(pole, np.array([2]), vpar, f0, mesh_n, par_dir)
            
            Chis[k] += np.trapz(vperp*int_vpar_second_order*sp.jv(n,kperp*vperp/Oc)**2*(-1),vperp) + n*kperp/kpar*np.trapz(int_vpar_first_order*sp.jv(n,kperp*vperp/Oc)*(sp.jv(n-1,kperp*vperp/Oc)-sp.jv(n+1,kperp*vperp/Oc)),vperp)
        
        
    return Chis*2*math.pi*wp**2/(kpar**2+kperp**2)/(1+Us)
    
# We can do all of these calculations faster (which will be important due to how many times we need to run these) if we calculate them together in the same loop
def calcM_Chi(vpar, vperp, f0, omega, kpar, kperp, nmax, Oc, nu, mesh_n, par_dir, wp):
    # Initialize U, M, and Chi
    U = np.zeros_like(omega) + 1j*0.0
    M = np.zeros_like(omega) + 1j*0.0
    Chi = np.zeros_like(omega) + 1j*0.0
    
    # Iterate through omega
    for k in range(len(omega)):
        # Iterate through nmax
        for n in range(nmax+1):
            print("omega:", omega[k])
            print("n:", n)
            # Calculate the base pole of this problem
            z = (omega[k] - n*Oc-1j*nu)/kpar
            
            # Do the integrals for the three types of poles that show up in the calculations
            # 1: A single pole at z with order 1
            # 2: A single pole at z with order 2
            # 3: A double pole at z and z* (each first order)
            singlePole_Order1 = poleIntegrate(np.array([z]), np.array([1]), vpar, f0, mesh_n, par_dir)
            singlePole_Order2 = poleIntegrate(np.array([z]), np.array([2]), vpar, f0, mesh_n, par_dir)
            doublePole = poleIntegrate(np.array([z,np.conjugate(z)]), np.array([1,1]), vpar, f0, mesh_n, par_dir)
            
            # Perform summation for U, M, and Chi
            U[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*singlePole_Order1,vperp)
            M[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*doublePole,vperp)
            Chi[k] += np.trapz(vperp*singlePole_Order2*sp.jv(n,kperp*vperp/Oc)**2*(-1),vperp) + n*kperp/kpar*np.trapz(singlePole_Order1*sp.jv(n,kperp*vperp/Oc)*(sp.jv(n-1,kperp*vperp/Oc)-sp.jv(n+1,kperp*vperp/Oc)),vperp)
    
    # Perform remaining operations to calculate U, M, and Chi
    U = -2*math.pi*1j*nu/kpar*U
    M = (M*2*math.pi/kpar**2 -np.abs(U)**2/nu**2)*nu/np.abs(1+U)**2
    Chi = Chi*2*math.pi*wp**2/(kpar**2+kperp**2)/(1+U)
    
    return [M, Chi]
    
# Calculate the spectra based on modified distributions and susceptabilities of ions and electrons
# Assume only one ion species
def calcSpectra(M_i, M_e, chi_i, chi_e):
    # Calculate the dielectric function
    eps = 1 + chi_i + chi_e
    return 2*np.abs(1-chi_e/eps)**2*M_e+2*np.abs(chi_e/eps)**2*M_i
    
# We can do all of these calculations faster (which will be important due to how many times we need to run these) if we calculate them together in the same loop
# Also include the capability to restart based on an input 
# Have omega be a scalar. Thus, have sum_U, sum_M, sum_chi, U, M, and chi all be scalars
# If this is the first time you are using this, set sum_U, sum_M, and sum_chi to be 0+1j*0.0 (need to do this so that we retain the imagninary components)
def calcU_M_chi(vpar, vperp, f0, omega, kpar, kperp, nStart, nEnd, Oc, nu, mesh_n, par_dir, wp, sum_U, sum_M, sum_chi):
    # Iterate from nStart to nEnd+1
    for n in range(nStart, nEnd+1):
        # Calculate the base pole of this problem
        z = (omega - n*Oc-1j*nu)/kpar
        
        # Do the integrals for the three types of poles that show up in the calculations
        # 1: A single pole at z with order 1
        # 2: A single pole at z with order 2
        # 3: A double pole at z and z* (each first order)
        singlePole_Order1 = poleIntegrate(np.array([z]), np.array([1]), vpar, f0, mesh_n, par_dir)
        singlePole_Order2 = poleIntegrate(np.array([z]), np.array([2]), vpar, f0, mesh_n, par_dir)
        doublePole = poleIntegrate(np.array([z,np.conjugate(z)]), np.array([1,1]), vpar, f0, mesh_n, par_dir)
        
        # Perform summation for U, M, and Chi
        sum_U += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*singlePole_Order1,vperp)
        sum_M += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*doublePole,vperp)
        sum_chi += np.trapz(vperp*singlePole_Order2*sp.jv(n,kperp*vperp/Oc)**2*(-1),vperp) + n*kperp/kpar*np.trapz(singlePole_Order1*sp.jv(n,kperp*vperp/Oc)*(sp.jv(n-1,kperp*vperp/Oc)-sp.jv(n+1,kperp*vperp/Oc)),vperp)
    
    # Perform remaining operations to calculate U, M, and Chi
    U = -2*math.pi*1j*nu/kpar*sum_U
    M = (sum_M*2*math.pi/kpar**2 -np.abs(U)**2/nu**2)*nu/np.abs(1+U)**2
    chi = sum_chi*2*math.pi*wp**2/(kpar**2+kperp**2)/(1+U)
    
    return [M, chi, U, sum_M, sum_chi, sum_U]
    
    
    
    