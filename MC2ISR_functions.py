import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
import scipy
import scipy.special as sp
from scipy.interpolate import interp1d, CubicSpline
import multiprocessing as mp
from functools import partial
import logging
import tracemalloc
import time

# Calculate the interpolated velocity space based on location of poles
# Inputs are:
# poles: a 1D array of complex numbers whose elements correspond to the location of each pole in your problem
# v_input: a 1D numpy array of the initial velocity mesh corresponding to your data
# mesh_n: an integer for the number of additional points to consider in each of the additional mesh refinements
# Outputs the pole refined velocity mesh
def getVInterp(poles, v_input, mesh_n):
    # First make sure that the mesh_n is odd. If not, change it such that it is
    # We need a symmetric adaptive mesh about the real location of each pole
    if mesh_n % 2 == 0:
        mesh_n += 1
        
    # Create a refined mesh about each pole
    v_fine = np.array([])
    for i in range(len(poles)):
        # Get real and imaginary components of pole
        re_v = np.real(poles[i])
        im_v = np.imag(poles[i])
        
        # Make bounds for adaptive mesh based on size of imaginary component
        v_0 = re_v - 10*im_v;
        v_1 = re_v - 2*im_v;
        v_2 = re_v + 2*im_v;
        v_3 = re_v + 10*im_v
        
        # Make new mesh based on all thes bounds
        v_mesh = np.concatenate((np.linspace(v_0,v_1,mesh_n),np.linspace(v_1,v_2,mesh_n),np.linspace(v_2,v_3,mesh_n)))
        
        # Concatenate each pole's refined mesh together
        v_fine = np.concatenate((v_fine, v_mesh))
        
    # Combine refined mesh with original velocity mesh
    v_int = np.concatenate((v_fine, v_input))
    
    # Sort and just get unique values
    v_int = np.unique(v_int)
    
    # Just get the velocity values in the original velocity domain
    v_int = v_int[ v_int >= np.min(v_input) ]
    v_int = v_int[ v_int <= np.max(v_input) ]
    
    return v_int

# Get the integrand of some distribution with an arbitrary number of poles
# Inputs are:
# poles: a 1D array of complex numbers whose elements correspond to the location of each pole in your problem
# orders: a 1D array whose elements correspond to the exponent of the corresponding pole
# v: a velocity mesh numpy array corresponding to direction of pole integration
# Fv: A distribution function numpy array
# Note:v and Fv must have same shape
def getIntegrand(poles, orders, v, Fv):
    # Note that this is transposed to work more easily with 2D python arrays
    integrand = np.transpose(Fv)
    
    for i in range(0, len(poles)):
        integrand = integrand / (v - poles[i])**orders[i]
    return np.transpose(integrand)  # Need to transpose result back

# This function does the pole integration using the refined mesh
def poleIntegrate(poles, orders, v_input, Fv, mesh_n, int_dir):
    
    # Get the interpolated velocity
    v_int = getVInterp(poles, v_input, mesh_n)
    
    # Make interpolation function and interpolate
    # interp_fun = interp1d(v_input, Fv, axis=int_dir)
    interp_fun = CubicSpline(v_input, Fv, axis=int_dir)
    
    Fv_int = interp_fun(v_int)
    
    # Calculate integrand
    integrand = getIntegrand(poles, orders, v_int, Fv_int)
    
    return np.trapz(integrand, v_int,axis=int_dir)  # Do the trapezoidal integration

def getLinearInterpCoeff(x, y):
    # Set everything up based on if we have 1d or 2d data
    if len(np.shape(y)) == 2:
        [numPar, numPerp] = np.shape(y)
        numElements = numPar - 1
        a = np.zeros((numElements, numPerp))
        b = np.zeros((numElements, numPerp))
    elif len(np.shape(y)) == 1:
        numElements = len(y) - 1
        a = np.zeros(numElements)
        b = np.zeros(numElements)

    # Iterate through numElements
    for i in range(numElements):
        a[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
        b[i] = y[i]-a[i]*x[i]
    return a,b

# Make functions for the exact solution to linear representation
def p1IndefiniteIntegral(a, b, z, x):
    return a*(x-z)+(b+a*z)*np.log(x-z)

def p2IndefiniteIntegral(a, b, z, x):
    return (-b-a*z)/(x-z)+a*np.log(x-z)

def pstarIndefiniteIntegral(a, b, z, x):
    return ((b+a*np.conjugate(z))*np.log(x-np.conjugate(z)) - (b+a*z)*np.log(x-z))*1j/np.imag(z)/2.

# Conduct parallel pole integrations
def interpolatedIntegral(a, b, z, x, integralFunction):
    integratedValue = 0.0 + 1j*0.0# might need to rethink this
    # Iterate through each element
    for i in range(len(a)):
        integratedValue += integralFunction(a[i], b[i], z, x[i+1]) - integralFunction(a[i], b[i], z, x[i])
    return integratedValue
    

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

# This function calculates the exact collisional term, U, for a Maxwellian distribution
def calcUs_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu):
    # Calculate yn
    Us = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from 0 to nmax
    for n in range(-nmax,nmax+1):
        # Calculate yn
        yn = (omega - n*Oc-1j*nu)/kpar/vth
        
        # Do summation term in Us
        Us += sp.ive(n, kperp**2*rho_avg**2)*(2*sp.dawsn(yn) + 1j*math.pi**.5*np.exp(-yn*yn))
        
    # Multiply by everything else and return
    return Us*1j*nu/kpar/vth

# This function calculates the exact modified distribution, M, for a Maxwellian distribution
def calcMs_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu, Us):
    Ms = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from 0 to nmax
    for n in range(-nmax,nmax+1):
        yn = (omega - n*Oc-1j*nu)/kpar/vth
        
        Ms += np.exp(-yn**2)*sp.ive(n,kperp**2*rho_avg**2)
    
    return np.real(Ms*math.pi**.5/kpar/vth/np.abs(1+Us)**2-np.abs(Us)**2/nu/np.abs(1+Us)**2)

# This function calculates the exact susceptibility, chi, for a Maxwellian distribution
def calcChis_Maxwellian(omega, kpar, kperp, vth, nmax, rho_avg, Oc, nu, alpha, U, Te, Ts):
    chi = np.zeros_like(omega) + 1j*0.0
    for n in range(-nmax,nmax+1):
        yn = (omega - n*Oc-1j*nu)/kpar/vth
        
        chi += sp.ive(n, kperp**2*rho_avg**2)*(1 - (omega-1j*nu)*(2*sp.dawsn(yn)+1j*math.pi**.5*np.exp(-yn**2))/(kpar*vth))
        
    return alpha**2*(Te/Ts)*chi/(1+U)

# This function calculates the exact collisional term, U, for a bi-Maxwellian distribution
def calcU_biMax(omega, kpar, kperp, vthpar, nmax, rho_avg, Oc, nu):
    # Initialize U
    U = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from -nmax to nmax
    for n in range(-nmax, nmax+1):
        # Calculate yn
        yn = (omega - n*Oc-1j*nu)/kpar/vthpar
        
        # Do summation term in U
        U += sp.ive(n, kperp**2*rho_avg**2)*(2*sp.dawsn(yn) + 1j*math.pi**.5*np.exp(-yn**2))
    return U*1j*nu/kpar/vthpar

# This function calculates the exact modified distribution function, M, for a bi-Maxwellian distribution
def calcM_biMax(omega, kpar, kperp, vthpar, nmax, rho_avg, Oc, nu, U):
    # Initialize M
    M = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from -nmax to nmax
    for n in range(-nmax, nmax+1):
        # Calculate yn and ynstar
        yn = (omega - n*Oc-1j*nu)/kpar/vthpar
        ynstar = (omega - n*Oc+1j*nu)/kpar/vthpar
        
        # Do summation term for M
        M += sp.ive(n, kperp**2*rho_avg**2)*np.exp(-yn**2)
        
        #sp.ive(n,kperp**2*rho_avg**2)*np.exp(-ynstar**2)*(np.exp(4*1j*nu*(omega-n*Oc)/kpar**2/vthpar**2)*sp.erfc(1j*yn)+sp.erf(1j*ynstar)+1)
    return np.real(M*math.pi**.5/kpar/vthpar/np.abs(1+U)**2-np.abs(U)**2/nu/np.abs(1+U)**2)    #  np.real(M*math.pi**0.5/2/kpar/vthpar/np.abs(1+U)**2-np.abs(U)**2/nu/np.abs(1+U)**2)

def calcchi_biMax(omega, kpar, kperp, vthpar, vthperp, nmax, rho_avg, Oc, nu, U, alpha, Te, Tpar):
    # Initialize chi
    chi = np.zeros_like(omega) + 1j*0.0
    
    # Iterate from -nmax to nmax
    for n in range(-nmax, nmax+1):
        # Calculate yn
        yn = (omega - n*Oc-1j*nu)/kpar/vthpar
        
        # Do summation term
        chi += sp.ive(n, kperp**2*rho_avg**2)*(1-(n*Oc*vthpar/kpar/vthperp**2+yn)*(2*sp.dawsn(yn)+1j*math.pi**.5*np.exp(-yn**2)))
        
    return chi*Te*alpha**2/Tpar/(1+U)

# This function calculates the collisional term, U, using the pole refined integration method
# The inputs are:
# vpar: a 1D numpy array corresponding to your parallel velocity mesh
# vperp: a 1D numpy array corresponding to your perpendicular velocity mesh
def calcUs(vpar, vperp, f0, omega, kpar, kperp, nmax, Oc, nu, mesh_n, par_dir):
    # Initialize Us, which is a function of omega
    Us = np.zeros_like(omega) + 1j*0.0
    
    # Iterate through omega
    for k in range(len(omega)):
        # print(k)
        # Iterate through nmax
        for n in range(-nmax,nmax+1):
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
        for n in range(-nmax,nmax+1):
            # Calculate poles in parallel integral
            poles = np.array([omega[k]-n*Oc+1j*nu,omega[k]-n*Oc-1j*nu])/kpar
            
            # Do pole refined integration
            int_vpar = poleIntegrate(poles, np.array([1,1]), vpar, f0, mesh_n, par_dir)
            
            Ms[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*int_vpar,vperp)

    return (Ms*2*math.pi/kpar**2 -np.abs(Us)**2/nu**2)*nu/np.abs(1+Us)**2  #(Ms*2*math.pi/kpar**2-np.abs(Us)**2/nu**2)*nu/np.abs(1+Us)**2

# Calculate Chis numerically
# wp is plasma frequency
def calcChis(vpar, vperp, f0, omega, kpar, kperp, nmax, Oc, nu, mesh_n, par_dir, Us, wp):
    # Initialize Chis
    Chis = np.zeros_like(omega) + 1j*0.0
    # Iterate through omega
    for k in range(len(omega)):
        # Iterate through nmax
        for n in range(-nmax,nmax+1):
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
        for n in range(-nmax,nmax+1):
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
# Currently assumes only one ion species
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
    for nBase in range(nStart, nEnd+1):
        
        # Need to do this symmetrically. So must do positive and negative side
        for sgn in [-1.0,1.0]:
            n = sgn*nBase
            print(n)
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
 
# Calculate the individual summation terms. 
# Takes an input of one n value
# Returns one set of values for the summation terms corresponding to n and -n for U, chi, and M   
def calcSumTerms(nBase, vpar, vperp, f0, omega, kpar, kperp, Oc, nu, mesh_n, par_dir):
    proc = mp.Process()
    # logging.info("Starting n = %d on %s" % (nBase, proc.name))

    sum_U = np.zeros_like(omega) + 1j*0.0
    sum_M = np.zeros_like(omega) + 1j*0.0
    sum_chi = np.zeros_like(omega) + 1j*0.0
    # Iterate through the two signs. Summation term needs to be symmetric
    for sgn in [-1.0, 1.0]:
        n = sgn*nBase

        # Calculate the base pole
        z = (omega - n*Oc - 1j*nu)/kpar

        # Iterate through omega to get summation terms for each omega
        for k in range(0, len(omega)):
            start = time.time()
            # Do the integrals for the three types of poles that show up in the calculations
            # 1: A single pole at z with order 1
            # 2: A single pole at z with order 2
            # 3: A double pole at z and z* (each first order)
            singlePole_Order1 = poleIntegrate(np.array([z[k]]), np.array([1]), vpar, f0, mesh_n, par_dir)
            singlePole_Order2 = poleIntegrate(np.array([z[k]]), np.array([2]), vpar, f0, mesh_n, par_dir)
            doublePole = poleIntegrate(np.array([z[k],np.conjugate(z[k])]), np.array([1,1]), vpar, f0, mesh_n, par_dir)

            # Perform summation for U, M, and Chi
            sum_U[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*singlePole_Order1,vperp)
            sum_M[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*doublePole,vperp)
            sum_chi[k] += np.trapz(vperp*singlePole_Order2*sp.jv(n,kperp*vperp/Oc)**2*(-1),vperp) + n*kperp/kpar*np.trapz(singlePole_Order1*sp.jv(n,kperp*vperp/Oc)*(sp.jv(n-1,kperp*vperp/Oc)-sp.jv(n+1,kperp*vperp/Oc)),vperp)

            # print("k=%d of %d finished in %.2es" % (k, len(omega)-1, time.time()-start))
        # If n is 0, break the loop so we aren't double counting the zeroth order term
        if n == 0:
            break
    logging.info("n = %d is done on %s" % (nBase, proc.name))
    # logging.info(tracemalloc.get_traced_memory())
    return sum_U, sum_M, sum_chi

# A parallelized function that will call calcSumTerms and iterate from -nmax to nmax
# Will output all of the individual summation terms
def calcSumTerms_par(num_processors, nStart, nmax, vpar, vperp, f0, omega, kpar, kperp, Oc, nu, mesh_n, par_dir, fileDir, fileName):
    # Make an array for all n values we want to consider
    nArray = np.arange(nStart, nmax+1)
    
    # Check to make sure that the number of processors is <= to available number on computer
    if num_processors > mp.cpu_count():
        num_processors = mp.cpu_count()  # If too many called, default to max available
    
    # Make an iterable function that only changes n in calcSumTerms and leaves everything else a constant
    caclSumTerms_iterable = partial(calcSumTerms,vpar=vpar, vperp=vperp, f0=f0, omega=omega, kpar=kpar, kperp=kperp, Oc=Oc, nu=nu, mesh_n=mesh_n, par_dir=par_dir)
    
    # Set pool based on number of processors
    with mp.Pool(processes=num_processors) as pool:
        # Perform the parallel calculation
        out = np.array(pool.map(caclSumTerms_iterable, nArray))
    
    # Output to previous dumps everything into a single 3D array that is of the following shape:
    # nmax+1 x 3 x len(omega)
    # Return the values in appropriate 2D summation arrays that will be of shape:
    # nmax+1 x len(omega)
    sum_U =out[:,0,:]
    sum_M = out[:,1,:]
    sum_chi = out[:,2,:]
    
    # Save the data so that we have it for the future
    np.savetxt(fileDir + fileName + '_sum_U.txt',sum_U,fmt='%.18e')
    np.savetxt(fileDir + fileName + '_sum_M.txt',sum_M,fmt='%.18e')
    np.savetxt(fileDir + fileName + '_sum_chi.txt',sum_chi,fmt='%.18e')
    
    return sum_U, sum_M, sum_chi
    
# Calculate the individual summation terms. 
# Takes an input of one n value
# Returns one set of values for the summation terms corresponding to n and -n for U, chi, and M   
# This is using the linear interpolation interpretation
def calcSumTerms_interp(nBase, vpar, vperp, a, b, omega, kpar, kperp, Oc, nu):
    start_time = time.time()
    proc = mp.Process()
    # logging.info("Starting n = %d on %s" % (nBase, proc.name))

    sum_U = np.zeros_like(omega) + 1j*0.0
    sum_M = np.zeros_like(omega) + 1j*0.0
    sum_chi = np.zeros_like(omega) + 1j*0.0
    # Iterate through the two signs. Summation term needs to be symmetric
    for sgn in [-1.0, 1.0]:
        n = sgn*nBase

        # Calculate the base pole
        z = (omega - n*Oc - 1j*nu)/kpar

        # Iterate through omega to get summation terms for each omega
        for k in range(0, len(omega)):
            # Do the integrals for the three types of poles that show up in the calculations
            # 1: A single pole at z with order 1
            # 2: A single pole at z with order 2
            # 3: A double pole at z and z* (each first order)
            p1 = interpolatedIntegral(a, b, z[k], vpar, p1IndefiniteIntegral)
            pstar = interpolatedIntegral(a, b, z[k], vpar, pstarIndefiniteIntegral)
            p2 = interpolatedIntegral(a, b, z[k], vpar, p2IndefiniteIntegral)

            # Perform summation for U, M, and Chi
            sum_U[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*p1,vperp)
            sum_M[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*pstar,vperp)
            sum_chi[k] += np.trapz(vperp*p2*sp.jv(n,kperp*vperp/Oc)**2*(-1),vperp) + n*kperp/kpar*np.trapz(p1*sp.jv(n,kperp*vperp/Oc)*(sp.jv(n-1,kperp*vperp/Oc)-sp.jv(n+1,kperp*vperp/Oc)),vperp)

            # print("k=%d of %d finished in %.2es" % (k, len(omega)-1, time.time()-start))
        # If n is 0, break the loop so we aren't double counting the zeroth order term
        if n == 0:
            break
    end_time = time.time() - start_time
    logging.info("n = %d is done on %s in %.2es" % (nBase, proc.name, end_time))
    # logging.info(tracemalloc.get_traced_memory())
    return sum_U, sum_M, sum_chi


# A parallelized function that will call calcSumTerms and iterate from -nmax to nmax
# Will output all of the individual summation terms
def calcSumTerms_interp_par(num_processors, nStart, nmax, vpar, vperp, a, b, omega, kpar, kperp, Oc, nu, fileDir, fileName):
    # Make an array for all n values we want to consider
    nArray = np.arange(nStart, nmax+1)
    
    # Check to make sure that the number of processors is <= to available number on computer
    if num_processors > mp.cpu_count():
        num_processors = mp.cpu_count()  # If too many called, default to max available
    
    # Make an iterable function that only changes n in calcSumTerms and leaves everything else a constant
    caclSumTerms_iterable = partial(calcSumTerms_interp,vpar=vpar, vperp=vperp, a=a, b=b, omega=omega, kpar=kpar, kperp=kperp, Oc=Oc, nu=nu)
    
    # Set pool based on number of processors
    with mp.Pool(processes=num_processors) as pool:
        # Perform the parallel calculation
        out = np.array(pool.map(caclSumTerms_iterable, nArray))
    
    # Output to previous dumps everything into a single 3D array that is of the following shape:
    # nmax+1 x 3 x len(omega)
    # Return the values in appropriate 2D summation arrays that will be of shape:
    # nmax+1 x len(omega)
    sum_U =out[:,0,:]
    sum_M = out[:,1,:]
    sum_chi = out[:,2,:]
    
    # Save the data so that we have it for the future
    np.savetxt(fileDir + fileName + '_sum_U.txt',sum_U,fmt='%.18e')
    np.savetxt(fileDir + fileName + '_sum_M.txt',sum_M,fmt='%.18e')
    np.savetxt(fileDir + fileName + '_sum_chi.txt',sum_chi,fmt='%.18e')
    
    return sum_U, sum_M, sum_chi

# Calculate U, M, and chi from their summation terms
# Input summations are assumed to be of the shape nmax+1 x len(omega) as per the output of the previous function
# Output 
def calcFromSums(sum_U, sum_M, sum_chi, kpar, kperp, nu, wp):
    
    if len(np.shape(sum_U)) == 2:
        # Perform summations and calculations
        U = -2*math.pi*1j*nu/kpar*np.sum(sum_U,axis=0)
        M = (np.sum(sum_M,axis=0)*2*math.pi/kpar**2 -np.abs(U)**2/nu**2)*nu/np.abs(1+U)**2
        chi = np.sum(sum_chi,axis=0)*2*math.pi*wp**2/(kpar**2+kperp**2)/(1+U)
    
    elif len(np.shape(sum_U)) == 1:
        # Perform summations and calculations
        U = -2*math.pi*1j*nu/kpar*sum_U
        M = (sum_M*2*math.pi/kpar**2 -np.abs(U)**2/nu**2)*nu/np.abs(1+U)**2
        chi = sum_chi*2*math.pi*wp**2/(kpar**2+kperp**2)/(1+U)
    return U, M, chi

# Initialize the logger
def initialize_logger(fileName, fileDir, num_processors):
    logging.root.handlers = []
    logging.basicConfig(level=logging.DEBUG, 
                        format='[%(asctime)s] %(message)s', 
                        datefmt='%m/%d/%Y %H:%M:%S',
                        handlers=[logging.FileHandler(fileDir+fileName+'.log',mode='w'),
                                  logging.StreamHandler()])
    logging.info('Initializing log for %s using %d processors' % (fileName, num_processors))


# Calculate the individual summation terms using simple trapz
# Takes an input of one n value
# Returns one set of values for the summation terms corresponding to n and -n for U, chi, and M   
def calcSumTerms_trapz(nBase, vpar, vperp, f0, omega, kpar, kperp, Oc, nu, mesh_n, par_dir):
    proc = mp.Process()

    sum_U = np.zeros_like(omega) + 1j*0.0
    sum_M = np.zeros_like(omega) + 1j*0.0
    sum_chi = np.zeros_like(omega) + 1j*0.0
    # Iterate through the two signs. Summation term needs to be symmetric
    for sgn in [-1.0, 1.0]:
        n = sgn*nBase

        # Calculate the base pole
        z = (omega - n*Oc - 1j*nu)/kpar

        # Iterate through omega to get summation terms for each omega
        for k in range(0, len(omega)):
            # start = time.time()
            # Do the integrals for the three types of poles that show up in the calculations
            # 1: A single pole at z with order 1
            # 2: A single pole at z with order 2
            # 3: A double pole at z and z* (each first order)
            singlePole_Order1 = np.trapz(np.transpose(np.transpose(f0)/(vpar-z[k])), vpar, axis=par_dir)
            singlePole_Order2 = np.trapz(np.transpose(np.transpose(f0)/(vpar-z[k])**2), vpar, axis=par_dir)
            doublePole = np.trapz(np.transpose(np.transpose(f0)/(vpar-z[k])/(vpar-np.conjugate(z[k]))), vpar, axis=par_dir)

            # Perform summation for U, M, and Chi
            sum_U[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*singlePole_Order1,vperp)
            sum_M[k] += np.trapz(sp.jv(n,kperp*vperp/Oc)**2*vperp*doublePole,vperp)
            sum_chi[k] += np.trapz(vperp*singlePole_Order2*sp.jv(n,kperp*vperp/Oc)**2*(-1),vperp) + n*kperp/kpar*np.trapz(singlePole_Order1*sp.jv(n,kperp*vperp/Oc)*(sp.jv(n-1,kperp*vperp/Oc)-sp.jv(n+1,kperp*vperp/Oc)),vperp)

            # print("k=%d of %d finished in %.2es" % (k, len(omega)-1, time.time()-start))
        # If n is 0, break the loop so we aren't double counting the zeroth order term
        if n == 0:
            break
    logging.info("n = %d is done on %s" % (nBase, proc.name))
    # logging.info(tracemalloc.get_traced_memory())
    return sum_U, sum_M, sum_chi